import asyncio
import base64
import io
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import gspread
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Presion Bot Backend", version="0.1.0")

REQUIRED_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_SECRET",
    "GEMINI_API_KEY",
    "GOOGLE_SHEETS_ID",
    "GOOGLE_SERVICE_ACCOUNT_JSON",
]

PROCESSING_TIMEOUT_SECONDS = float(
    os.getenv("PROCESSING_TIMEOUT_SECONDS", "600")
)
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "300"))
SHEETS_TIMEOUT_SECONDS = float(os.getenv("SHEETS_TIMEOUT_SECONDS", "30"))


@dataclass
class Settings:
    telegram_bot_token: str
    telegram_webhook_secret: str
    gemini_api_key: str
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
        gemini_api_key=_get_required_env("GEMINI_API_KEY"),
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


def _clean_json(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini response does not contain valid JSON object")
    return json.loads(content[start:end + 1])


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
    """Compress and optimize image for faster Gemini processing."""
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
    settings: Settings,
    image_bytes: bytes,
    mime_type: str,
    trace_id: str,
) -> dict[str, Any]:
    step_start = perf_counter()

    prompt = (
        "Extract blood pressure values from this image of a pressure monitor. "
        "Return only JSON with keys: systolic, diastolic, pulse, "
        "confidence, notes. "
        "Rules: all values must be integers, confidence from 0 to 1. "
        "If any value is not readable, set it to null and explain in notes."
    )
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    model_name = model_name.replace("models/", "")
    fallback_raw = os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash,gemini-2.0-flash,gemini-1.5-flash-latest",
    )
    fallback_models = [
        m.strip().replace("models/", "")
        for m in fallback_raw.split(",")
        if m.strip()
    ]
    model_candidates = [model_name, *fallback_models]
    # Preserve order while deduplicating.
    model_candidates = list(dict.fromkeys(model_candidates))
    config_time = perf_counter() - step_start
    logger.info(f"Gemini request setup in {config_time:.2f}s")

    step_start = perf_counter()
    logger.info(
        "[%s] Sending image to Gemini REST API (models=%s size=%s bytes)",
        trace_id,
        model_candidates,
        len(image_bytes),
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(image_bytes).decode(
                                "ascii"
                            ),
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    timeout = httpx.Timeout(
        connect=min(15.0, GEMINI_TIMEOUT_SECONDS),
        read=GEMINI_TIMEOUT_SECONDS,
        write=min(30.0, GEMINI_TIMEOUT_SECONDS),
        pool=min(15.0, GEMINI_TIMEOUT_SECONDS),
    )
    response = None
    try:
        with httpx.Client(timeout=timeout) as client:
            for current_model in model_candidates:
                gemini_url = (
                    "https://generativelanguage.googleapis.com/"
                    f"v1beta/models/{current_model}:generateContent"
                )
                try:
                    response = client.post(
                        gemini_url,
                        headers={"x-goog-api-key": settings.gemini_api_key},
                        json=payload,
                    )
                    response.raise_for_status()
                    logger.info(
                        "[%s] Gemini model selected=%s",
                        trace_id,
                        current_model,
                    )
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        logger.warning(
                            "[%s] Gemini model not found: %s. Trying next.",
                            trace_id,
                            current_model,
                        )
                        continue
                    raise
            else:
                raise RuntimeError("gemini_model_not_found")
    except httpx.TimeoutException as exc:
        logger.exception("[%s] Gemini request timed out", trace_id)
        raise RuntimeError("gemini_timeout") from exc
    except httpx.HTTPStatusError as exc:
        body_preview = exc.response.text[:600]
        logger.error(
            "[%s] Gemini HTTP error status=%s body=%s",
            trace_id,
            exc.response.status_code,
            body_preview,
        )
        raise RuntimeError("gemini_http_error") from exc
    except httpx.HTTPError as exc:
        logger.exception("[%s] Gemini transport error", trace_id)
        raise RuntimeError("gemini_transport_error") from exc

    if response is None:
        raise RuntimeError("gemini_model_not_found")

    response_json = response.json()
    candidates = response_json.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response missing candidates")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    if not parts:
        raise ValueError("Gemini response missing content parts")
    raw = parts[0].get("text") or ""
    if not raw:
        raise ValueError("Gemini response missing text payload")

    gemini_time = perf_counter() - step_start
    logger.info(
        "[%s] Gemini response received in %.2fs",
        trace_id,
        gemini_time,
    )

    parsed = _clean_json(raw)

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
                settings,
                optimized_bytes,
                "image/jpeg",
                trace_id,
            ),
            timeout=PROCESSING_TIMEOUT_SECONDS,
        )
        logger.info(
            "[%s] Gemini extraction completed for chat_id=%s",
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
        # If Gemini fails, communicate that explicitly to speed up diagnosis.
        error_text = (
            "Hubo un error al procesar la imagen. "
            "Intenta de nuevo con una foto mas clara."
        )
        if isinstance(exc, RuntimeError) and str(exc).startswith("gemini_"):
            error_text = (
                "No pude conectar con Gemini en este momento. "
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
