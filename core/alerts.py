from __future__ import annotations

import os
from dataclasses import dataclass

import requests


@dataclass(slots=True)
class TelegramAlertResult:
    sent: bool
    detail: str


class TelegramAlerter:
    """Small Telegram Bot API wrapper driven by environment variables."""

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(self, message: str) -> TelegramAlertResult:
        if not self.is_configured:
            return TelegramAlertResult(
                sent=False,
                detail="Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
            )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        response = requests.post(
            url,
            json={
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        if response.ok:
            return TelegramAlertResult(sent=True, detail="Telegram alert sent.")
        return TelegramAlertResult(sent=False, detail=f"Telegram API error: {response.text}")


def build_traffic_alert_message(
    source: str,
    vehicle_count: float,
    chaos_index: float,
    predicted_volume_30min: float | None = None,
    prediction_level: str | None = None,
    tft_confidence: float | None = None,
    vehicle_density: str | None = None,
    location: str | None = None,
    weather: str | None = None,
    time_period: str | None = None,
    reason: str | None = None,
) -> str:
    risk = (prediction_level or "high").upper()
    density = (vehicle_density or "unknown").upper()
    confidence = f"{tft_confidence:.1f}%" if tft_confidence is not None else "N/A"
    volume = f"{predicted_volume_30min:.0f}" if predicted_volume_30min is not None else "N/A"
    return (
        "<b>ALERT CONGESTION</b>\n"
        f"LOCATION: {location or 'unknown'}\n"
        f"RISK: <b>{risk}</b> | DENSITY: <b>{density}</b>\n"
        f"30-MIN: <b>{(prediction_level or 'unknown').upper()}</b> ({volume})\n"
        f"TFT CONFIDENCE: <b>{confidence}</b>\n"
        f"CONTEXT: {time_period or 'current time'}, {weather or 'weather unknown'}\n"
        f"REASON: {reason or 'Congestion threshold crossed.'}\n"
        "ACTION: Monitor route and consider diversion."
    )
