import asyncio
import io
import json
import logging
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import gspread
import httpx
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance, ImageOps

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Presion Bot Backend", version="0.1.0")

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_SECRET",
    "GOOGLE_SHEETS_ID",
    "GOOGLE_SERVICE_ACCOUNT_JSON",
]

PROCESSING_TIMEOUT_SECONDS = float(
    os.getenv("PROCESSING_TIMEOUT_SECONDS", "600")
)
SHEETS_TIMEOUT_SECONDS = float(os.getenv("SHEETS_TIMEOUT_SECONDS", "30"))
WEBHOOK_DEDUP_TTL_SECONDS = int(os.getenv("WEBHOOK_DEDUP_TTL_SECONDS", "300"))

_recent_update_ids: dict[int, float] = {}
_recent_file_keys: dict[str, float] = {}


def _cleanup_dedup_cache(now: float) -> None:
    for cache in (_recent_update_ids, _recent_file_keys):
        expired = [
            k
            for k, ts in cache.items()
            if now - ts > WEBHOOK_DEDUP_TTL_SECONDS
        ]
        for key in expired:
            cache.pop(key, None)


def _mark_if_duplicate(update_id: int | None, file_key: str | None) -> bool:
    now = perf_counter()
    _cleanup_dedup_cache(now)

    duplicate = False
    if update_id is not None:
        duplicate = update_id in _recent_update_ids
        _recent_update_ids[update_id] = now

    if file_key:
        duplicate = duplicate or (file_key in _recent_file_keys)
        _recent_file_keys[file_key] = now

    return duplicate


@dataclass
class Settings:
    telegram_bot_token: str
    telegram_webhook_secret: str
    google_sheets_id: str
    google_service_account_json: str
    timezone: str


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    # Railway env values may include accidental wrapping quotes or spaces.
    cleaned = value.strip()
    if (
        (cleaned.startswith('"') and cleaned.endswith('"'))
        or (cleaned.startswith("'") and cleaned.endswith("'"))
    ) and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _get_missing_env_vars() -> list[str]:
    return [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]


def _get_settings() -> Settings:
    return Settings(
        telegram_bot_token=_get_required_env("TELEGRAM_BOT_TOKEN"),
        telegram_webhook_secret=_get_required_env("TELEGRAM_WEBHOOK_SECRET"),
        google_sheets_id=_get_required_env("GOOGLE_SHEETS_ID"),
        google_service_account_json=_get_required_env(
            "GOOGLE_SERVICE_ACCOUNT_JSON"
        ),
        timezone=os.getenv("TZ", "America/Argentina/Buenos_Aires"),
    )


def _require_settings() -> Settings:
    missing = _get_missing_env_vars()
    if missing:
        raise RuntimeError(
            "Missing required env vars: " + ", ".join(sorted(missing))
        )
    return _get_settings()


def _get_sheet(settings: Settings):
    creds_info = json.loads(settings.google_service_account_json)
    client = gspread.service_account_from_dict(creds_info)
    sheet = client.open_by_key(settings.google_sheets_id).sheet1
    return sheet


def _extract_first_number(raw: str, patterns: list[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _extract_first_float(raw: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _fallback_extract_measurements(raw: str) -> dict[str, Any] | None:
    systolic = _extract_first_number(
        raw,
        [
            r"(?:systolic|sistolica)\D{0,12}(\d{2,3})",
        ],
    )
    diastolic = _extract_first_number(
        raw,
        [
            r"(?:diastolic|diastolica)\D{0,12}(\d{2,3})",
        ],
    )
    pulse = _extract_first_number(
        raw,
        [
            r"(?:pulse|pulso|heart\s*rate|hr)\D{0,12}(\d{2,3})",
        ],
    )

    if systolic is None or diastolic is None:
        return None

    confidence = _extract_first_float(
        raw,
        [
            r"confidence\D{0,12}([01](?:\.\d+)?)",
            r"confianza\D{0,12}([01](?:\.\d+)?)",
        ],
    )

    return {
        "systolic": systolic,
        "diastolic": diastolic,
        "pulse": pulse,
        "confidence": confidence,
        "notes": "Valores recuperados con parseo de respaldo.",
    }


def _extract_triplet_from_numbers(
    numbers: list[int],
) -> tuple[int | None, int | None, int | None, float]:
    best: tuple[int | None, int | None, int | None, float] = (
        None,
        None,
        None,
        0.0,
    )
    for idx in range(max(0, len(numbers) - 2)):
        s, d, p = numbers[idx], numbers[idx + 1], numbers[idx + 2]
        if _is_valid_range(s, d, p):
            score = 0.85
        elif 70 <= s <= 250 and 40 <= d <= 150:
            score = 0.55
        else:
            score = 0.2
        if score > best[3]:
            best = (s, d, p, score)
    return best


def _extract_pair_from_numbers(
    numbers: list[int],
) -> tuple[int | None, int | None, float]:
    best: tuple[int | None, int | None, float] = (None, None, 0.0)
    for idx in range(max(0, len(numbers) - 1)):
        a, b = numbers[idx], numbers[idx + 1]
        # Most monitors show SYS first and DIA second.
        if 70 <= a <= 250 and 40 <= b <= 150 and a > b:
            score = 0.72
        elif 70 <= a <= 250 and 40 <= b <= 150:
            score = 0.55
        else:
            score = 0.0

        if score > best[2]:
            best = (a, b, score)
    return best


def _ocr_image_variants(image_bytes: bytes) -> list[Image.Image]:
    base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    large = base_img.resize(
        (base_img.width * 2, base_img.height * 2),
        Image.Resampling.LANCZOS,
    )
    gray = ImageOps.grayscale(large)
    contrasted = ImageEnhance.Contrast(gray).enhance(2.2)
    binary = contrasted.point(lambda p: 255 if p > 145 else 0)
    inverted = ImageOps.invert(binary.convert("L"))
    return [large, gray, contrasted, binary, inverted]


def _best_number_in_range(
    raw: str,
    min_value: int,
    max_value: int,
) -> int | None:
    numbers = [int(n) for n in re.findall(r"\b\d{2,3}\b", raw)]
    valid = [n for n in numbers if min_value <= n <= max_value]
    if not valid:
        return None
    # Keep the highest valid token; SYS/DIA/PUL tend to be displayed once.
    return max(valid)


def _ocr_from_monitor_regions(variant: Image.Image) -> dict[str, int | None]:
    width, height = variant.size
    x0 = int(width * 0.18)
    x1 = int(width * 0.90)
    sys_box = (x0, int(height * 0.08), x1, int(height * 0.42))
    dia_box = (x0, int(height * 0.34), x1, int(height * 0.68))
    pul_box = (x0, int(height * 0.62), x1, int(height * 0.96))

    digit_cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    sys_text = pytesseract.image_to_string(
        variant.crop(sys_box),
        config=digit_cfg,
    )
    dia_text = pytesseract.image_to_string(
        variant.crop(dia_box),
        config=digit_cfg,
    )
    pul_text = pytesseract.image_to_string(
        variant.crop(pul_box),
        config=digit_cfg,
    )

    return {
        "sys": _best_number_in_range(sys_text, 70, 250),
        "dia": _best_number_in_range(dia_text, 40, 150),
        "pul": _best_number_in_range(pul_text, 30, 220),
    }


def _normalize_ocr_text_for_digits(raw: str) -> str:
    return (
        raw.replace("O", "0")
        .replace("o", "0")
        .replace("I", "1")
        .replace("l", "1")
        .replace("S", "5")
        .replace("B", "8")
    )


def _extract_valid_numbers(
    raw: str,
    min_value: int,
    max_value: int,
) -> list[int]:
    cleaned = _normalize_ocr_text_for_digits(raw)
    numbers = [int(n) for n in re.findall(r"\b\d{2,3}\b", cleaned)]
    return [n for n in numbers if min_value <= n <= max_value]


def _pick_stable_value(candidates: list[int]) -> int | None:
    if not candidates:
        return None
    counts = Counter(candidates)
    most_common = counts.most_common()
    top_freq = most_common[0][1]
    top_values = [value for value, freq in most_common if freq == top_freq]
    return max(top_values)


def _pick_weighted_value(
    candidates: list[tuple[int, float]],
) -> int | None:
    if not candidates:
        return None

    scores: dict[int, float] = {}
    for value, weight in candidates:
        scores[value] = scores.get(value, 0.0) + max(weight, 1.0)

    best_value = None
    best_score = -1.0
    for value, score in scores.items():
        same_score_higher_value = (
            score == best_score and value > (best_value or -1)
        )
        if score > best_score or same_score_higher_value:
            best_value = value
            best_score = score
    return best_value


def _read_field_candidates(
    variant: Image.Image,
    box: tuple[int, int, int, int],
    min_value: int,
    max_value: int,
) -> list[int]:
    x0, y0, x1, y1 = box
    width, height = variant.size
    margin_x = int(width * 0.015)
    margin_y = int(height * 0.01)
    crops = [
        (x0, y0, x1, y1),
        (
            max(0, x0 - margin_x),
            max(0, y0 - margin_y),
            min(width, x1 + margin_x),
            min(height, y1 + margin_y),
        ),
    ]

    configs = [
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
        "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
    ]

    numbers: list[int] = []
    for crop_box in crops:
        crop = variant.crop(crop_box)
        # Upscale and auto-contrast improve readability for segmented digits.
        enlarged = crop.resize(
            (crop.width * 4, crop.height * 4),
            Image.Resampling.LANCZOS,
        )
        prepared = ImageOps.autocontrast(ImageOps.grayscale(enlarged))
        thresholded = prepared.point(lambda p: 255 if p > 150 else 0)
        prepared_variants = [prepared, thresholded]

        for prepared_variant in prepared_variants:
            for config in configs:
                text = pytesseract.image_to_string(
                    prepared_variant,
                    config=config,
                )
                numbers.extend(
                    _extract_valid_numbers(text, min_value, max_value)
                )

    return numbers


def _collect_display_row_candidates(
    display_img: Image.Image,
) -> dict[str, list[tuple[int, float]]]:
    rows: dict[str, list[tuple[int, float]]] = {
        "sys": [],
        "dia": [],
        "pul": [],
    }

    configs = [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789",
        "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789",
    ]

    variants = [
        ImageOps.autocontrast(ImageOps.grayscale(display_img)),
    ]
    variants.append(variants[0].point(lambda p: 255 if p > 145 else 0))

    for prepared in variants:
        for config in configs:
            data = pytesseract.image_to_data(
                prepared,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            height = max(1, prepared.height)
            for idx, raw_text in enumerate(data.get("text", [])):
                if not raw_text or not raw_text.strip():
                    continue

                try:
                    conf = float(data.get("conf", ["0"])[idx])
                except (TypeError, ValueError, IndexError):
                    conf = 0.0

                numbers_sys = _extract_valid_numbers(raw_text, 70, 250)
                numbers_dia = _extract_valid_numbers(raw_text, 40, 150)
                numbers_pul = _extract_valid_numbers(raw_text, 30, 220)
                if not (numbers_sys or numbers_dia or numbers_pul):
                    continue

                try:
                    y = float(data.get("top", [0])[idx])
                    h = float(data.get("height", [0])[idx])
                except (TypeError, ValueError, IndexError):
                    y = 0.0
                    h = 0.0

                y_center_ratio = (y + (h / 2.0)) / height
                if y_center_ratio < 0.34:
                    row_key = "sys"
                    row_numbers = numbers_sys
                elif y_center_ratio < 0.68:
                    row_key = "dia"
                    row_numbers = numbers_dia
                else:
                    row_key = "pul"
                    row_numbers = numbers_pul

                for number in row_numbers:
                    rows[row_key].append((number, conf))

    return rows


def _extract_measurements_by_regions(
    image_bytes: bytes,
) -> dict[str, Any] | None:
    base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = base.size

    # Main display window for Omron framing.
    display_box = (
        int(width * 0.24),
        int(height * 0.17),
        int(width * 0.82),
        int(height * 0.80),
    )
    display_img = base.crop(display_box).resize(
        (
            max(1, int((display_box[2] - display_box[0]) * 2.4)),
            max(1, int((display_box[3] - display_box[1]) * 2.4)),
        ),
        Image.Resampling.LANCZOS,
    )

    rows = _collect_display_row_candidates(display_img)
    systolic = _pick_weighted_value(rows["sys"])
    diastolic = _pick_weighted_value(rows["dia"])
    pulse = _pick_weighted_value(rows["pul"])

    if (
        systolic is not None
        and diastolic is not None
        and pulse is not None
        and systolic > diastolic
    ):
        logger.info(
            "Display-row OCR success SYS=%s DIA=%s PUL=%s",
            systolic,
            diastolic,
            pulse,
        )
        return {
            "systolic": systolic,
            "diastolic": diastolic,
            "pulse": pulse,
            "confidence": 0.95,
            "notes": "Lectura OCR local por filas del display.",
        }

    region_candidates = {
        "sys": [],
        "dia": [],
        "pul": [],
    }

    prepared_variant = ImageOps.autocontrast(ImageOps.grayscale(base))
    width, height = prepared_variant.size
    # Tuned for Omron framing: right display, 3 stacked numeric rows.
    x0 = int(width * 0.30)
    x1 = int(width * 0.74)
    sys_box = (x0, int(height * 0.23), x1, int(height * 0.43))
    dia_box = (x0, int(height * 0.40), x1, int(height * 0.61))
    pul_box = (x0, int(height * 0.57), x1, int(height * 0.78))

    region_candidates["sys"].extend(
        _read_field_candidates(prepared_variant, sys_box, 70, 250)
    )
    region_candidates["dia"].extend(
        _read_field_candidates(prepared_variant, dia_box, 40, 150)
    )
    region_candidates["pul"].extend(
        _read_field_candidates(prepared_variant, pul_box, 30, 220)
    )

    logger.info(
        "Regional OCR candidates SYS=%s DIA=%s PUL=%s",
        region_candidates["sys"][:8],
        region_candidates["dia"][:8],
        region_candidates["pul"][:8],
    )

    systolic = _pick_stable_value(region_candidates["sys"])
    diastolic = _pick_stable_value(region_candidates["dia"])
    pulse = _pick_stable_value(region_candidates["pul"])

    if systolic is None or diastolic is None or pulse is None:
        return None
    if systolic <= diastolic:
        return None

    return {
        "systolic": systolic,
        "diastolic": diastolic,
        "pulse": pulse,
        "confidence": 0.94,
        "notes": "Lectura OCR local por sectores fijos (arriba/medio/abajo).",
    }


def _run_ocr_text_candidates(image_bytes: bytes) -> list[str]:
    variants = _ocr_image_variants(image_bytes)
    results: list[str] = []
    ocr_configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.:/SYSDIAPULHR",
    ]
    for variant in variants:
        for config in ocr_configs:
            text = pytesseract.image_to_string(variant, config=config)
            if text and text.strip():
                results.append(text)

        region_read = _ocr_from_monitor_regions(variant)
        if region_read["sys"] is not None and region_read["dia"] is not None:
            results.append(
                " ".join(
                    [
                        f"SYS {region_read['sys']}",
                        f"DIA {region_read['dia']}",
                        (
                            f"PUL {region_read['pul']}"
                            if region_read["pul"] is not None
                            else ""
                        ),
                    ]
                ).strip()
            )
    return results


def _extract_measurements_from_ocr_text(raw: str) -> dict[str, Any] | None:
    normalized = re.sub(r"\s+", " ", raw).strip()
    normalized = (
        normalized.replace("O", "0")
        .replace("o", "0")
        .replace("I", "1")
        .replace("l", "1")
    )
    if not normalized:
        return None

    labeled_systolic = _extract_first_number(
        normalized,
        [r"(?:sys|systolic|sistolica)\D{0,10}(\d{2,3})"],
    )
    labeled_diastolic = _extract_first_number(
        normalized,
        [r"(?:dia|diastolic|diastolica)\D{0,10}(\d{2,3})"],
    )
    labeled_pulse = _extract_first_number(
        normalized,
        [r"(?:pul|pulse|pulso|hr|heart\s*rate)\D{0,10}(\d{2,3})"],
    )

    if labeled_systolic is not None and labeled_diastolic is not None:
        confidence = 0.92 if labeled_pulse is not None else 0.82
        return {
            "systolic": labeled_systolic,
            "diastolic": labeled_diastolic,
            "pulse": labeled_pulse,
            "confidence": confidence,
            "notes": "Lectura OCR local por etiquetas.",
        }

    numbers = [
        int(match)
        for match in re.findall(r"\b\d{2,3}\b", normalized)
    ]
    if len(numbers) < 2:
        return None

    pulse: int | None = None
    systolic, diastolic, triplet_pulse, score = (
        _extract_triplet_from_numbers(numbers)
    )
    if systolic is not None and diastolic is not None:
        pulse = triplet_pulse
        return {
            "systolic": systolic,
            "diastolic": diastolic,
            "pulse": pulse,
            "confidence": score,
            "notes": "Lectura OCR local por patron numerico.",
        }

    systolic, diastolic, pair_score = _extract_pair_from_numbers(numbers)
    if systolic is None or diastolic is None:
        return None

    return {
        "systolic": systolic,
        "diastolic": diastolic,
        "pulse": None,
        "confidence": pair_score,
        "notes": "Lectura OCR local parcial (sin pulso).",
    }


def _is_valid_range(systolic: int, diastolic: int, pulse: int) -> bool:
    return (
        70 <= systolic <= 250
        and 40 <= diastolic <= 150
        and 30 <= pulse <= 220
    )


def _guess_mime_type(file_path: str) -> str:
    lower_path = file_path.lower()
    if lower_path.endswith(".png"):
        return "image/png"
    if lower_path.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


async def _telegram_get_file_path(settings: Settings, file_id: str) -> str:
    telegram_api_base = (
        f"https://api.telegram.org/bot{settings.telegram_bot_token}"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{telegram_api_base}/getFile",
            params={"file_id": file_id},
        )
        resp.raise_for_status()
        data = resp.json()
    if not data.get("ok"):
        raise ValueError("Telegram getFile failed")
    result = data.get("result", {})
    file_path = result.get("file_path")
    if not file_path:
        raise ValueError("Telegram file_path missing")
    return file_path


async def _telegram_download_file(settings: Settings, file_path: str) -> bytes:
    telegram_file_base = (
        f"https://api.telegram.org/file/bot{settings.telegram_bot_token}"
    )
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(f"{telegram_file_base}/{file_path}")
        resp.raise_for_status()
        return resp.content


async def _telegram_send_message(
    settings: Settings,
    chat_id: int,
    text: str,
) -> None:
    telegram_api_base = (
        f"https://api.telegram.org/bot{settings.telegram_bot_token}"
    )
    payload = {"chat_id": chat_id, "text": text}
    try:
        logger.info(
            "Sending Telegram message to chat_id=%s: %s...",
            chat_id,
            text[:50],
        )
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{telegram_api_base}/sendMessage",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
        logger.info(f"Message sent successfully to chat_id={chat_id}")
    except Exception as e:
        logger.error(
            "Failed to send Telegram message to chat_id=%s: %s",
            chat_id,
            e,
        )
        raise


async def _safe_telegram_send_message(
    settings: Settings,
    chat_id: int,
    text: str,
) -> None:
    try:
        await _telegram_send_message(settings, chat_id, text)
    except Exception:
        logger.exception(
            "Could not deliver Telegram message chat_id=%s",
            chat_id,
        )


def _optimize_image(image_bytes: bytes) -> bytes:
    """Compress and optimize image for faster OCR processing."""
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Resize if image is too large (max 1920px on longest side)
        max_size = 1920
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Compress and save
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()
    except Exception:
        # If optimization fails, return original
        return image_bytes


def _extract_measurement_from_image(
    image_bytes: bytes,
    trace_id: str,
) -> dict[str, Any]:
    step_start = perf_counter()
    logger.info(
        "[%s] Starting local OCR (size=%s bytes)",
        trace_id,
        len(image_bytes),
    )

    try:
        regional_data = _extract_measurements_by_regions(image_bytes)
    except pytesseract.TesseractNotFoundError as exc:
        logger.exception("[%s] Local OCR engine not found", trace_id)
        raise RuntimeError("ocr_engine_not_found") from exc
    except Exception as exc:
        logger.exception("[%s] Regional OCR failed", trace_id)
        raise RuntimeError("ocr_local_error") from exc

    if regional_data is not None:
        logger.info(
            "[%s] Regional OCR success: SYS=%s DIA=%s PUL=%s",
            trace_id,
            regional_data["systolic"],
            regional_data["diastolic"],
            regional_data["pulse"],
        )
        return {
            "sistolica": regional_data["systolic"],
            "diastolica": regional_data["diastolic"],
            "pulso": regional_data["pulse"],
            "confianza_ia": regional_data["confidence"],
            "observacion": regional_data["notes"],
            "estado": "auto",
        }

    try:
        text_candidates = _run_ocr_text_candidates(image_bytes)
    except pytesseract.TesseractNotFoundError as exc:
        logger.exception("[%s] Local OCR engine not found", trace_id)
        raise RuntimeError("ocr_engine_not_found") from exc
    except Exception as exc:
        logger.exception("[%s] Local OCR failed", trace_id)
        raise RuntimeError("ocr_local_error") from exc

    ocr_time = perf_counter() - step_start
    logger.info(
        "[%s] OCR completed in %.2fs (candidates=%s)",
        trace_id,
        ocr_time,
        len(text_candidates),
    )
    if text_candidates:
        preview = " | ".join(
            re.sub(r"\s+", " ", c).strip()[:110] for c in text_candidates[:3]
        )
        logger.info("[%s] OCR preview: %s", trace_id, preview)

    parsed = None
    for raw_text in text_candidates:
        parsed = _extract_measurements_from_ocr_text(raw_text)
        if parsed:
            break

    if parsed is None and text_candidates:
        parsed = _fallback_extract_measurements("\n".join(text_candidates))
    if parsed is None:
        raise ValueError("ocr_values_not_found")

    systolic = parsed.get("systolic")
    diastolic = parsed.get("diastolic")
    pulse = parsed.get("pulse")
    confidence = parsed.get("confidence")
    notes = parsed.get("notes", "")

    if systolic is None or diastolic is None or pulse is None:
        status = "pendiente_revision"
    else:
        systolic = int(systolic)
        diastolic = int(diastolic)
        pulse = int(pulse)
        status = (
            "auto"
            if _is_valid_range(systolic, diastolic, pulse)
            else "pendiente_revision"
        )

    return {
        "sistolica": systolic,
        "diastolica": diastolic,
        "pulso": pulse,
        "confianza_ia": confidence,
        "observacion": notes,
        "estado": status,
    }


def _append_row(
    settings: Settings,
    data: dict[str, Any],
    telegram_file_id: str,
) -> None:
    sheet = _get_sheet(settings)
    now = datetime.now(ZoneInfo(settings.timezone)).isoformat(
        timespec="seconds"
    )
    row = [
        now,
        data.get("sistolica"),
        data.get("diastolica"),
        data.get("pulso"),
        "telegram",
        data.get("confianza_ia"),
        data.get("estado"),
        data.get("observacion"),
        telegram_file_id,
    ]
    sheet.append_row(row, value_input_option="USER_ENTERED")


async def _process_telegram_photo(
    settings: Settings,
    chat_id: int,
    file_id: str,
) -> None:
    started_at = perf_counter()
    trace_id = uuid.uuid4().hex[:8]
    logger.info(
        "[%s] Start processing Telegram photo chat_id=%s file_id=%s",
        trace_id,
        chat_id,
        file_id,
    )
    try:
        file_path = await _telegram_get_file_path(settings, file_id)

        image_bytes = await _telegram_download_file(settings, file_path)
        logger.info(
            "[%s] Downloaded Telegram photo chat_id=%s "
            "file_id=%s size_bytes=%s",
            trace_id,
            chat_id,
            file_id,
            len(image_bytes),
        )

        # Optimize image for faster processing
        optimized_bytes = await asyncio.to_thread(_optimize_image, image_bytes)
        logger.info(
            "[%s] Optimized image chat_id=%s "
            "original_size=%s optimized_size=%s",
            trace_id,
            chat_id,
            len(image_bytes),
            len(optimized_bytes),
        )

        data = await asyncio.wait_for(
            asyncio.to_thread(
                _extract_measurement_from_image,
                optimized_bytes,
                trace_id,
            ),
            timeout=PROCESSING_TIMEOUT_SECONDS,
        )
        logger.info(
            "[%s] OCR extraction completed for chat_id=%s",
            trace_id,
            chat_id,
        )

        logger.info(
            "[%s] Appending row to Sheets for chat_id=%s",
            trace_id,
            chat_id,
        )
        await asyncio.wait_for(
            asyncio.to_thread(_append_row, settings, data, file_id),
            timeout=SHEETS_TIMEOUT_SECONDS,
        )
        logger.info(
            "[%s] Row appended to Sheets for chat_id=%s",
            trace_id,
            chat_id,
        )

        if data["estado"] == "auto":
            text = (
                f"Guardado: {data['sistolica']}/{data['diastolica']} mmHg, "
                f"pulso {data['pulso']}"
            )
        else:
            text = (
                "Guardado con revision pendiente. "
                "Valores detectados: "
                f"{data.get('sistolica')}/{data.get('diastolica')} mmHg, "
                f"pulso {data.get('pulso')}."
            )

        logger.info(
            "[%s] Sending final message to chat_id=%s",
            trace_id,
            chat_id,
        )
        await _telegram_send_message(settings, chat_id, text)
        elapsed = perf_counter() - started_at
        logger.info(
            "[%s] Finished processing Telegram photo "
            "chat_id=%s file_id=%s elapsed=%.2fs",
            trace_id,
            chat_id,
            file_id,
            elapsed,
        )
    except TimeoutError:
        elapsed = perf_counter() - started_at
        logger.exception(
            "[%s] Timeout processing Telegram photo "
            "chat_id=%s file_id=%s elapsed=%.2fs",
            trace_id,
            chat_id,
            file_id,
            elapsed,
        )
        await _safe_telegram_send_message(
            settings,
            chat_id,
            "La imagen tardo demasiado en procesarse. "
            "Intenta de nuevo con una foto bien iluminada y enfocada.",
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = perf_counter() - started_at
        logger.exception(
            "[%s] Error processing Telegram photo "
            "chat_id=%s file_id=%s elapsed=%.2fs",
            trace_id,
            chat_id,
            file_id,
            elapsed,
        )
        # If local OCR fails, communicate that clearly for diagnosis.
        error_text = (
            "Hubo un error al procesar la imagen. "
            "Intenta de nuevo con una foto mas clara."
        )
        if str(exc) == "ocr_engine_not_found":
            error_text = (
                "El motor OCR local no esta disponible en el servidor. "
                "Contactame para habilitarlo."
            )
        elif str(exc) == "ocr_values_not_found":
            error_text = (
                "No pude leer valores claros en la foto. "
                "Intenta con mejor luz, sin reflejos y bien enfocada."
            )
        elif isinstance(exc, RuntimeError) and str(exc).startswith("ocr_"):
            error_text = (
                "Hubo un problema con el OCR local en este momento. "
                "Intenta de nuevo en unos minutos."
            )
        await _safe_telegram_send_message(
            settings,
            chat_id,
            error_text,
        )


def _log_background_task_result(task: asyncio.Task) -> None:
    try:
        task.result()
    except Exception:  # noqa: BLE001
        logger.exception("Background Telegram task failed")


@app.get("/health")
async def health() -> dict[str, Any]:
    missing = _get_missing_env_vars()
    if missing:
        return {
            "status": "degraded",
            "configured": False,
            "missing_env_vars": missing,
        }
    return {"status": "ok", "configured": True}


@app.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    settings = _require_settings()

    if x_telegram_bot_api_secret_token != settings.telegram_webhook_secret:
        raise HTTPException(status_code=401, detail="invalid webhook secret")

    update = await request.json()
    update_id_raw = update.get("update_id")
    update_id = update_id_raw if isinstance(update_id_raw, int) else None
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")

    if not chat_id:
        return JSONResponse({"ok": True, "ignored": "missing chat id"})

    photos = message.get("photo") or []
    if not photos:
        # Send error message as background task to not block webhook response
        task = asyncio.create_task(
            _telegram_send_message(
                settings,
                chat_id,
                "No detecte una foto. Enviame una imagen del tensiometro.",
            )
        )
        task.add_done_callback(_log_background_task_result)
        return JSONResponse({"ok": True, "ignored": "no photo"})

    # Telegram sends multiple sizes. The last one is typically the largest.
    best_photo = photos[-1]
    file_id = best_photo.get("file_id")
    if not file_id:
        # Send error message as background task to not block webhook response
        task = asyncio.create_task(
            _telegram_send_message(
                settings,
                chat_id,
                "No pude obtener el archivo de la foto.",
            )
        )
        task.add_done_callback(_log_background_task_result)
        return JSONResponse({"ok": True, "ignored": "missing file id"})

    file_key = f"{chat_id}:{file_id}"
    if _mark_if_duplicate(update_id, file_key):
        logger.info(
            "Ignoring duplicated Telegram update_id=%s file_key=%s",
            update_id,
            file_key,
        )
        return JSONResponse({"ok": True, "ignored": "duplicate update"})

    # Send immediate confirmation message
    confirmation_task = asyncio.create_task(
        _telegram_send_message(
            settings,
            chat_id,
            "📷 Foto recibida. Procesando...",
        )
    )
    confirmation_task.add_done_callback(_log_background_task_result)

    task = asyncio.create_task(
        _process_telegram_photo(settings, chat_id, file_id)
    )
    task.add_done_callback(_log_background_task_result)
    return JSONResponse({"ok": True, "queued": True})
