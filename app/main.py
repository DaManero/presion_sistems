import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import gspread
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import google.generativeai as genai

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
    return value


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
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{telegram_api_base}/sendMessage",
            json=payload,
        )
        resp.raise_for_status()


def _extract_measurement_from_image(
    settings: Settings,
    image_bytes: bytes,
) -> dict[str, Any]:
    prompt = (
        "Extract blood pressure values from this image of a pressure monitor. "
        "Return only JSON with keys: systolic, diastolic, pulse, "
        "confidence, notes. "
        "Rules: all values must be integers, confidence from 0 to 1. "
        "If any value is not readable, set it to null and explain in notes."
    )
    genai.configure(api_key=settings.gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content(
        [
            {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            },
            prompt,
        ]
    )
    raw = response.text or ""
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
        await _telegram_send_message(
            settings,
            chat_id,
            "No detecte una foto. Enviame una imagen del tensiometro.",
        )
        return JSONResponse({"ok": True, "ignored": "no photo"})

    # Telegram sends multiple sizes. The last one is typically the largest.
    best_photo = photos[-1]
    file_id = best_photo.get("file_id")
    if not file_id:
        await _telegram_send_message(
            settings,
            chat_id,
            "No pude obtener el archivo de la foto.",
        )
        return JSONResponse({"ok": True, "ignored": "missing file id"})

    try:
        file_path = await _telegram_get_file_path(settings, file_id)
        image_bytes = await _telegram_download_file(settings, file_path)
        data = _extract_measurement_from_image(settings, image_bytes)
        _append_row(settings, data, file_id)

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

        await _telegram_send_message(settings, chat_id, text)
        return JSONResponse({"ok": True})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error processing Telegram webhook")
        await _telegram_send_message(
            settings,
            chat_id,
            "Hubo un error al procesar la imagen. "
            "Intenta de nuevo con una foto mas clara.",
        )
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
