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
from functools import lru_cache
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import gspread
import httpx
import numpy as np
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


def _pick_best_in_range(candidates: list[int]) -> int | None:
    if not candidates:
        return None
    filtered = [value for value in candidates if 0 < value < 300]
    if not filtered:
        return None
    return _pick_stable_value(filtered)


def _seven_segment_digit_patterns() -> dict[str, tuple[int, ...]]:
    patterns: dict[str, tuple[int, ...]] = {}
    for pattern, digit in SEVEN_SEGMENT_DIGITS.items():
        patterns[digit] = pattern
    return patterns


@lru_cache(maxsize=1)
def _build_seven_segment_templates() -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    patterns = _seven_segment_digit_patterns()
    width, height = 80, 140

    segments_boxes = [
        (18, 5, 62, 22),
        (6, 18, 22, 66),
        (58, 18, 74, 66),
        (18, 58, 62, 82),
        (6, 74, 22, 122),
        (58, 74, 74, 122),
        (18, 118, 62, 136),
    ]

    for digit, pattern in patterns.items():
        canvas = np.zeros((height, width), dtype=np.uint8)
        for idx, is_on in enumerate(pattern):
            if not is_on:
                continue
            x0, y0, x1, y1 = segments_boxes[idx]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), 255, -1)
        templates[digit] = canvas
    return templates


def _opencv_normalize_digit(slot: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(slot > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = slot[y0:y1, x0:x1]
    if cropped.size == 0:
        return None

    normalized = cv2.resize(
        cropped,
        (80, 140),
        interpolation=cv2.INTER_NEAREST,
    )
    _, normalized = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)
    return normalized


def _opencv_classify_digit(slot: np.ndarray) -> str | None:
    normalized = _opencv_normalize_digit(slot)
    if normalized is None:
        return None

    templates = _build_seven_segment_templates()
    best_digit = None
    best_score = -1.0
    for digit, template in templates.items():
        score = cv2.matchTemplate(
            normalized,
            template,
            cv2.TM_CCOEFF_NORMED,
        )[0][0]
        if score > best_score:
            best_score = score
            best_digit = digit

    if best_score < 0.20:
        return None
    return best_digit


def _opencv_decode_row_value(
    binary_img: np.ndarray,
    row_box: tuple[float, float, float, float],
    digit_count: int,
) -> int | None:
    height, width = binary_img.shape[:2]
    x0 = int(width * row_box[0])
    y0 = int(height * row_box[1])
    x1 = int(width * row_box[2])
    y1 = int(height * row_box[3])
    row = binary_img[y0:y1, x0:x1]
    if row.size == 0:
        return None

    col_sum = np.sum(row > 0, axis=0)
    active_cols = np.where(col_sum > max(2, int(row.shape[0] * 0.06)))[0]
    if len(active_cols) == 0:
        return None

    start = int(active_cols.min())
    end = int(active_cols.max()) + 1
    strip = row[:, start:end]
    if strip.size == 0:
        return None

    slot_width = strip.shape[1] / float(digit_count)
    if slot_width <= 2:
        return None

    digits: list[str] = []
    for idx in range(digit_count):
        sx0 = int(round(idx * slot_width))
        sx1 = int(round((idx + 1) * slot_width))
        pad = max(1, int((sx1 - sx0) * 0.08))
        sx0 = max(0, sx0 - pad)
        sx1 = min(strip.shape[1], sx1 + pad)
        if sx1 <= sx0:
            return None
        slot = strip[:, sx0:sx1]
        digit = _opencv_classify_digit(slot)
        if digit is None:
            return None
        digits.append(digit)

    try:
        return int("".join(digits))
    except ValueError:
        return None


def _extract_measurements_with_opencv(
    image_bytes: bytes,
) -> dict[str, Any] | None:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    display = gray[
        int(height * 0.16):int(height * 0.69),
        int(width * 0.20):int(width * 0.73),
    ]
    if display.size == 0:
        return None

    display = cv2.resize(
        display,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC,
    )
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(display)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, otsu_binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    adaptive_binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        9,
    )
    morph_kernel = np.ones((3, 3), np.uint8)
    threshold_variants = [
        otsu_binary,
        adaptive_binary,
        cv2.morphologyEx(otsu_binary, cv2.MORPH_CLOSE, morph_kernel),
        cv2.morphologyEx(adaptive_binary, cv2.MORPH_CLOSE, morph_kernel),
    ]

    row_boxes = {
        "sys": [
            ((0.18, 0.14, 0.76, 0.40), 3, 70, 250),
            ((0.20, 0.15, 0.78, 0.41), 3, 70, 250),
        ],
        "dia": [
            ((0.24, 0.40, 0.74, 0.68), 2, 40, 150),
            ((0.26, 0.40, 0.72, 0.68), 2, 40, 150),
        ],
        "pul": [
            ((0.37, 0.73, 0.71, 0.94), 2, 30, 220),
            ((0.40, 0.72, 0.72, 0.94), 2, 30, 220),
        ],
    }
    candidates: dict[str, list[int]] = {"sys": [], "dia": [], "pul": []}

    for threshold_variant in threshold_variants:
        for key, row_options in row_boxes.items():
            for row_box, digits, min_value, max_value in row_options:
                value = _opencv_decode_row_value(
                    threshold_variant,
                    row_box,
                    digits,
                )
                if value is not None and min_value <= value <= max_value:
                    candidates[key].append(value)

    logger.info(
        "OpenCV display OCR SYS=%s DIA=%s PUL=%s",
        candidates["sys"][:8],
        candidates["dia"][:8],
        candidates["pul"][:8],
    )

    systolic = _pick_best_in_range(candidates["sys"])
    diastolic = _pick_best_in_range(candidates["dia"])
    pulse = _pick_best_in_range(candidates["pul"])
    if (
        systolic is None
        or diastolic is None
        or pulse is None
        or systolic <= diastolic
    ):
        return None

    return {
        "systolic": systolic,
        "diastolic": diastolic,
        "pulse": pulse,
        "confidence": 0.995,
        "notes": "Lectura OpenCV del display Omron.",
    }


def _extract_measurements_by_omron_layout(
    image_bytes: bytes,
) -> dict[str, Any] | None:
    base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = base.size
    prepared = ImageOps.autocontrast(ImageOps.grayscale(base))

    sys_box = (
        int(width * 0.28),
        int(height * 0.22),
        int(width * 0.67),
        int(height * 0.41),
    )
    dia_box = (
        int(width * 0.34),
        int(height * 0.38),
        int(width * 0.64),
        int(height * 0.59),
    )
    pul_box = (
        int(width * 0.41),
        int(height * 0.54),
        int(width * 0.61),
        int(height * 0.69),
    )

    sys_candidates = _read_field_candidates(prepared, sys_box, 70, 250)
    dia_candidates = _read_field_candidates(prepared, dia_box, 40, 150)
    pul_candidates = _read_field_candidates(prepared, pul_box, 30, 220)

    logger.info(
        "Omron layout OCR SYS=%s DIA=%s PUL=%s",
        sys_candidates[:8],
        dia_candidates[:8],
        pul_candidates[:8],
    )

    systolic = _pick_best_in_range(sys_candidates)
    diastolic = _pick_best_in_range(dia_candidates)
    pulse = _pick_best_in_range(pul_candidates)
    if (
        systolic is None
        or diastolic is None
        or pulse is None
        or systolic <= diastolic
    ):
        return None

    return {
        "systolic": systolic,
        "diastolic": diastolic,
        "pulse": pulse,
        "confidence": 0.97,
        "notes": "Lectura OCR local por layout fijo Omron.",
    }


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


def _read_fixed_row_candidates(
    display_img: Image.Image,
    row_range: tuple[float, float],
    min_value: int,
    max_value: int,
    x_range: tuple[float, float] = (0.16, 0.92),
) -> list[tuple[int, float]]:
    width, height = display_img.size
    x0 = int(width * x_range[0])
    x1 = int(width * x_range[1])
    y0 = int(height * row_range[0])
    y1 = int(height * row_range[1])
    if x1 <= x0 or y1 <= y0:
        return []

    crop = display_img.crop((x0, y0, x1, y1)).resize(
        (max(1, (x1 - x0) * 4), max(1, (y1 - y0) * 4)),
        Image.Resampling.LANCZOS,
    )

    base_gray = ImageOps.autocontrast(ImageOps.grayscale(crop))
    variants = [
        base_gray,
        base_gray.point(lambda p: 255 if p > 130 else 0),
        base_gray.point(lambda p: 255 if p > 145 else 0),
        ImageOps.invert(base_gray).point(lambda p: 255 if p > 145 else 0),
    ]
    configs = [
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
        "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
    ]

    candidates: list[tuple[int, float]] = []
    for prepared in variants:
        for config in configs:
            text = pytesseract.image_to_string(prepared, config=config)
            numbers = _extract_valid_numbers(text, min_value, max_value)
            for number in numbers:
                candidates.append((number, 80.0))

            data = pytesseract.image_to_data(
                prepared,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            for idx, raw_text in enumerate(data.get("text", [])):
                if not raw_text or not raw_text.strip():
                    continue
                numbers = _extract_valid_numbers(
                    raw_text,
                    min_value,
                    max_value,
                )
                if not numbers:
                    continue
                try:
                    conf = float(data.get("conf", ["0"])[idx])
                except (TypeError, ValueError, IndexError):
                    conf = 0.0
                for number in numbers:
                    candidates.append((number, conf))

    return candidates


SEVEN_SEGMENT_DIGITS: dict[tuple[int, ...], str] = {
    (1, 1, 1, 0, 1, 1, 1): "0",
    (0, 0, 1, 0, 0, 1, 0): "1",
    (1, 0, 1, 1, 1, 0, 1): "2",
    (1, 0, 1, 1, 0, 1, 1): "3",
    (0, 1, 1, 1, 0, 1, 0): "4",
    (1, 1, 0, 1, 0, 1, 1): "5",
    (1, 1, 0, 1, 1, 1, 1): "6",
    (1, 0, 1, 0, 0, 1, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9",
}


def _threshold_dark_foreground(
    image: Image.Image,
    threshold: int = 150,
) -> Image.Image:
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    return gray.point(lambda pixel: 255 if pixel < threshold else 0)


def _column_dark_counts(image: Image.Image) -> list[int]:
    width, height = image.size
    minimum_pixels = max(1, int(height * 0.08))
    counts: list[int] = []
    for x in range(width):
        histogram = image.crop((x, 0, x + 1, height)).histogram()
        count = histogram[255] if len(histogram) > 255 else 0
        counts.append(count if count >= minimum_pixels else 0)
    return counts


def _find_digit_spans(
    image: Image.Image,
    expected_digits: int,
) -> list[tuple[int, int]]:
    counts = _column_dark_counts(image)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for x, count in enumerate(counts):
        if count > 0 and start is None:
            start = x
        elif count == 0 and start is not None:
            if x - start >= 4:
                spans.append((start, x))
            start = None
    if start is not None and len(counts) - start >= 4:
        spans.append((start, len(counts)))

    if not spans:
        return []

    merged: list[tuple[int, int]] = [spans[0]]
    max_gap = max(2, image.size[0] // 40)
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))

    while len(merged) > expected_digits:
        gaps = [
            merged[idx + 1][0] - merged[idx][1]
            for idx in range(len(merged) - 1)
        ]
        if not gaps:
            break
        merge_idx = gaps.index(min(gaps))
        merged[merge_idx] = (merged[merge_idx][0], merged[merge_idx + 1][1])
        merged.pop(merge_idx + 1)

    return merged


def _segment_is_active(
    image: Image.Image,
    bounds: tuple[float, float, float, float],
    threshold: float,
) -> int:
    width, height = image.size
    x0 = max(0, min(width, int(width * bounds[0])))
    y0 = max(0, min(height, int(height * bounds[1])))
    x1 = max(x0 + 1, min(width, int(width * bounds[2])))
    y1 = max(y0 + 1, min(height, int(height * bounds[3])))
    region = image.crop((x0, y0, x1, y1))
    histogram = region.histogram()
    dark_pixels = histogram[255] if len(histogram) > 255 else 0
    area = max(1, region.size[0] * region.size[1])
    return 1 if (dark_pixels / area) >= threshold else 0


def _recognize_seven_segment_digit(image: Image.Image) -> str | None:
    bbox = image.getbbox()
    if bbox is None:
        return None

    padded = image.crop(bbox).resize((90, 150), Image.Resampling.NEAREST)
    segments = (
        _segment_is_active(padded, (0.22, 0.02, 0.78, 0.16), 0.30),
        _segment_is_active(padded, (0.05, 0.12, 0.24, 0.48), 0.24),
        _segment_is_active(padded, (0.76, 0.12, 0.95, 0.48), 0.24),
        _segment_is_active(padded, (0.22, 0.42, 0.78, 0.58), 0.24),
        _segment_is_active(padded, (0.05, 0.52, 0.24, 0.88), 0.24),
        _segment_is_active(padded, (0.76, 0.52, 0.95, 0.88), 0.24),
        _segment_is_active(padded, (0.22, 0.84, 0.78, 0.98), 0.30),
    )

    if segments in SEVEN_SEGMENT_DIGITS:
        return SEVEN_SEGMENT_DIGITS[segments]

    # Thin digits like 1 can miss one side segment after thresholding.
    if segments in ((0, 0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0)):
        return "1"
    return None


def _decode_row_with_seven_segments(
    display_img: Image.Image,
    row_box: tuple[float, float, float, float],
    expected_digits: int,
) -> int | None:
    width, height = display_img.size
    crop = display_img.crop(
        (
            int(width * row_box[0]),
            int(height * row_box[1]),
            int(width * row_box[2]),
            int(height * row_box[3]),
        )
    )
    if crop.size[0] <= 0 or crop.size[1] <= 0:
        return None

    threshold_candidates = [110, 130, 150, 170]
    decoded_values: list[int] = []
    for threshold in threshold_candidates:
        binary = _threshold_dark_foreground(crop, threshold)
        spans = _find_digit_spans(binary, expected_digits)
        if len(spans) != expected_digits:
            continue

        digits: list[str] = []
        for start, end in spans:
            digit = _recognize_seven_segment_digit(
                binary.crop((start, 0, end, binary.size[1]))
            )
            if digit is None:
                digits = []
                break
            digits.append(digit)

        if len(digits) == expected_digits:
            decoded_values.append(int("".join(digits)))

    return _pick_stable_value(decoded_values)


def _extract_measurements_by_regions(
    image_bytes: bytes,
) -> dict[str, Any] | None:
    opencv_data = _extract_measurements_with_opencv(image_bytes)
    if opencv_data is not None:
        return opencv_data

    direct_layout_data = _extract_measurements_by_omron_layout(image_bytes)
    if direct_layout_data is not None:
        return direct_layout_data

    base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = base.size

    # Main LCD window tuned to the Omron photo framing used by the bot.
    display_box = (
        int(width * 0.20),
        int(height * 0.16),
        int(width * 0.73),
        int(height * 0.69),
    )
    display_img = base.crop(display_box).resize(
        (
            max(1, int((display_box[2] - display_box[0]) * 2.4)),
            max(1, int((display_box[3] - display_box[1]) * 2.4)),
        ),
        Image.Resampling.LANCZOS,
    )

    seven_segment_sys = _decode_row_with_seven_segments(
        display_img,
        (0.18, 0.14, 0.76, 0.40),
        3,
    )
    seven_segment_dia = _decode_row_with_seven_segments(
        display_img,
        (0.24, 0.40, 0.74, 0.68),
        2,
    )
    seven_segment_pul = _decode_row_with_seven_segments(
        display_img,
        (0.37, 0.73, 0.71, 0.94),
        2,
    )
    logger.info(
        "Seven-segment OCR SYS=%s DIA=%s PUL=%s",
        seven_segment_sys,
        seven_segment_dia,
        seven_segment_pul,
    )
    if (
        seven_segment_sys is not None
        and seven_segment_dia is not None
        and seven_segment_pul is not None
        and seven_segment_sys > seven_segment_dia
    ):
        return {
            "systolic": seven_segment_sys,
            "diastolic": seven_segment_dia,
            "pulse": seven_segment_pul,
            "confidence": 0.99,
            "notes": "Lectura por reconocimiento de display de 7 segmentos.",
        }

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

    fixed_row_sys = _read_fixed_row_candidates(
        display_img,
        (0.12, 0.42),
        70,
        250,
        (0.18, 0.78),
    )
    fixed_row_dia = _read_fixed_row_candidates(
        display_img,
        (0.34, 0.68),
        40,
        150,
        (0.24, 0.70),
    )
    fixed_row_pul = _read_fixed_row_candidates(
        display_img,
        (0.66, 0.92),
        30,
        220,
        (0.42, 0.74),
    )
    logger.info(
        "Fixed-row OCR candidates SYS=%s DIA=%s PUL=%s",
        [v for v, _ in fixed_row_sys[:10]],
        [v for v, _ in fixed_row_dia[:10]],
        [v for v, _ in fixed_row_pul[:10]],
    )

    systolic = _pick_weighted_value(fixed_row_sys)
    diastolic = _pick_weighted_value(fixed_row_dia)
    pulse = _pick_weighted_value(fixed_row_pul)
    if (
        systolic is not None
        and diastolic is not None
        and pulse is not None
        and systolic > diastolic
    ):
        logger.info(
            "Fixed-row OCR success SYS=%s DIA=%s PUL=%s",
            systolic,
            diastolic,
            pulse,
        )
        return {
            "systolic": systolic,
            "diastolic": diastolic,
            "pulse": pulse,
            "confidence": 0.96,
            "notes": "Lectura OCR local por filas fijas del display.",
        }

    region_candidates = {
        "sys": [],
        "dia": [],
        "pul": [],
    }

    prepared_variant = ImageOps.autocontrast(ImageOps.grayscale(base))
    width, height = prepared_variant.size
    # Tuned for Omron framing: right display, 3 stacked numeric rows.
    sys_box = (
        int(width * 0.24),
        int(height * 0.23),
        int(width * 0.67),
        int(height * 0.40),
    )
    dia_box = (
        int(width * 0.26),
        int(height * 0.39),
        int(width * 0.66),
        int(height * 0.57),
    )
    pul_box = (
        int(width * 0.38),
        int(height * 0.61),
        int(width * 0.58),
        int(height * 0.71),
    )

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
        img.save(
            output,
            format="JPEG",
            quality=95,
            optimize=False,
            subsampling=0,
        )
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
