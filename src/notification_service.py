"""
Notification Service — Sprint 29.1 (résumé enrichi + lien Drive).

Envoie automatiquement le résumé de production quotidien via un bot Telegram
(voir .env.example : TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID), avec un lien
direct vers le package de production sur Google Drive quand il a été
uploadé avec succès (voir src/google_drive_uploader.py, UploadResult.remote_url).

LoggingNotificationService reste l'implémentation de repli tant que Telegram
n'est pas configuré (dev local, CI sans secrets, etc.) — aucune régression.

Pour brancher un canal supplémentaire (Discord, Email...) : implémenter une
nouvelle classe héritant de NotificationService et l'injecter via
build_notification_service() — aucun appelant n'a besoin d'être modifié.

TelegramNotificationService n'interrompt jamais le pipeline : toute erreur de
formatage, réseau, API ou timeout est capturée, journalisée, et le résumé est
tout de même journalisé localement en repli (voir send_daily_summary()).
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_TELEGRAM_API_BASE = "https://api.telegram.org"


@dataclass
class ChannelSummary:
    niche_name: str
    channel_name: str
    subject: str
    duration_seconds: int
    scene_count: int
    drive_link: Optional[str] = None  # lien du package Drive de cette production, si uploadé avec succès


@dataclass
class DailyProductionSummary:
    date: str
    channels: List[ChannelSummary] = field(default_factory=list)
    pipeline_duration_seconds: Optional[float] = None


def _format_duration(total_seconds: float) -> str:
    """Formate une durée en secondes façon '11m 24s' (lisible sur mobile)."""
    total = max(int(round(total_seconds)), 0)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def format_summary_text(summary: DailyProductionSummary) -> str:
    """
    Formate un résumé compact, lisible sur téléphone (Telegram), avec accès
    direct au package de production (lien Drive) quand il est disponible.
    """
    lines = [
        "🎬 Daily Production Ready",
        "",
        "✅ Pipeline completed successfully",
        "",
        "📈 Active niches",
    ]
    for channel in summary.channels:
        lines.append(f"• {channel.niche_name}")

    lines += [
        "",
        "🎥 Videos generated",
        str(len(summary.channels)),
        "",
        "🖼 Image prompts",
        "Ready",
        "",
        "🎞 Animation prompts",
        "Ready",
    ]

    links = [(c.niche_name, c.drive_link) for c in summary.channels if c.drive_link]
    if links:
        lines += ["", "📁 Production package"]
        if len(links) == 1:
            lines.append(links[0][1])
        else:
            for niche_name, link in links:
                lines.append(f"• {niche_name}: {link}")

    if summary.pipeline_duration_seconds is not None:
        lines += ["", "⏱ Pipeline duration", _format_duration(summary.pipeline_duration_seconds)]

    return "\n".join(lines)


class NotificationService(ABC):
    """Interface abstraite d'envoi du résumé de production quotidien."""

    @abstractmethod
    def send_daily_summary(self, summary: DailyProductionSummary) -> None:
        """Envoie (ou journalise) le résumé de la production du jour."""
        ...


class LoggingNotificationService(NotificationService):
    """Implémentation de repli tant qu'aucun canal réel n'est configuré."""

    def send_daily_summary(self, summary: DailyProductionSummary) -> None:
        try:
            text = format_summary_text(summary)
        except Exception as exc:
            logger.warning("Formatage du résumé échoué (%s) — notification annulée.", exc)
            return
        logger.info("\n%s", text)


class TelegramNotificationService(NotificationService):
    """
    Envoi réel via l'API Bot Telegram (sendMessage).

    Service annexe : aucune erreur (formatage, réseau, API, timeout) ne doit
    jamais remonter à l'appelant — le pipeline doit toujours pouvoir se
    terminer avec succès, packages de production déjà sauvegardés localement.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    def send_daily_summary(self, summary: DailyProductionSummary) -> None:
        try:
            text = format_summary_text(summary)
        except Exception as exc:
            logger.warning("Formatage du résumé Telegram échoué (%s) — notification annulée.", exc)
            return

        import requests

        url = f"{_TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage"
        try:
            response = requests.post(url, data={"chat_id": self._chat_id, "text": text}, timeout=10)
            response.raise_for_status()
            logger.info("Notification Telegram envoyée (chat_id=%s).", self._chat_id)
        except Exception as exc:
            logger.warning(
                "Envoi Telegram échoué (%s) — résumé journalisé localement en repli.", exc,
            )
            logger.info("\n%s", text)


def build_notification_service() -> NotificationService:
    """
    Construit le service selon la configuration disponible dans l'environnement.

    - TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID présents → TelegramNotificationService.
    - Variables absentes → LoggingNotificationService.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if bot_token and chat_id:
        logger.info("Notification active : Telegram (chat_id=%s).", chat_id)
        return TelegramNotificationService(bot_token, chat_id)

    logger.info("Telegram non configuré — notification en mode log uniquement.")
    return LoggingNotificationService()
