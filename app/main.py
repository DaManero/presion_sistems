import json
import logging
import os
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


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


TELEGRAM_BOT_TOKEN = _get_required_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = _get_required_env("TELEGRAM_WEBHOOK_SECRET")
GEMINI_API_KEY = _get_required_env("GEMINI_API_KEY")
GOOGLE_SHEETS_ID = _get_required_env("GOOGLE_SHEETS_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = _get_required_env("GOOGLE_SERVICE_ACCOUNT_JSON")
TIMEZONE = os.getenv("TZ", "America/Argentina/Buenos_Aires")

TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TELEGRAM_FILE_BASE = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"


genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")


def _get_sheet():
    creds_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    client = gspread.service_account_from_dict(creds_info)
    sheet = client.open_by_key(GOOGLE_SHEETS_ID).sheet1
    return sheet


SHEET = _get_sheet()


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


async def _telegram_get_file_path(file_id: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{TELEGRAM_API_BASE}/getFile",
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


async def _telegram_download_file(file_path: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(f"{TELEGRAM_FILE_BASE}/{file_path}")
        resp.raise_for_status()
        return resp.content


async def _telegram_send_message(chat_id: int, text: str) -> None:
    payload = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{TELEGRAM_API_BASE}/sendMessage",
            json=payload,
        )
        resp.raise_for_status()


def _extract_measurement_from_image(image_bytes: bytes) -> dict[str, Any]:
    prompt = (
        "Extract blood pressure values from this image of a pressure monitor. "
        "Return only JSON with keys: systolic, diastolic, pulse, "
        "confidence, notes. "
        "Rules: all values must be integers, confidence from 0 to 1. "
        "If any value is not readable, set it to null and explain in notes."
    )
    response = GEMINI_MODEL.generate_content(
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


def _append_row(data: dict[str, Any], telegram_file_id: str) -> None:
    now = datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")
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
    SHEET.append_row(row, value_input_option="USER_ENTERED")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
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
            chat_id,
            "No detecte una foto. Enviame una imagen del tensiometro.",
        )
        return JSONResponse({"ok": True, "ignored": "no photo"})

    # Telegram sends multiple sizes. The last one is typically the largest.
    best_photo = photos[-1]
    file_id = best_photo.get("file_id")
    if not file_id:
        await _telegram_send_message(
            chat_id,
            "No pude obtener el archivo de la foto.",
        )
        return JSONResponse({"ok": True, "ignored": "missing file id"})

    try:
        file_path = await _telegram_get_file_path(file_id)
        image_bytes = await _telegram_download_file(file_path)
        data = _extract_measurement_from_image(image_bytes)
        _append_row(data, file_id)

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

        await _telegram_send_message(chat_id, text)
        return JSONResponse({"ok": True})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error processing Telegram webhook")
        await _telegram_send_message(
            chat_id,
            "Hubo un error al procesar la imagen. "
            "Intenta de nuevo con una foto mas clara.",
        )
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
