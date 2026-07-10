"""
Notification Service — Sprint 29.1 (résumé enrichi + lien de stockage + diagnostic),
mis à jour Sprint 30 (Google Drive → Supabase Storage).

Envoie automatiquement le résumé de production quotidien via un bot Telegram
(voir .env.example : TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID), avec un lien
direct vers le package de production sur Supabase Storage quand il a été
uploadé avec succès (voir src/supabase_storage_uploader.py, UploadResult.remote_url).

LoggingNotificationService reste l'implémentation de repli tant que Telegram
n'est pas configuré (dev local, CI sans secrets, etc.) — aucune régression.

Pour brancher un canal supplémentaire (Discord, Email...) : implémenter une
nouvelle classe héritant de NotificationService et l'injecter via
build_notification_service() — aucun appelant n'a besoin d'être modifié.

TelegramNotificationService n'interrompt jamais le pipeline : toute erreur de
formatage, réseau, API ou timeout est capturée, journalisée, et le résumé est
tout de même journalisé localement en repli (voir send_daily_summary()).

Fiabilité (Sprint 29.1) : send_daily_summary() ne retourne plus None mais un
NotificationResult qui distingue précisément le sort de la notification —
envoyée, secret manquant, bot token invalide, chat id invalide, erreur
HTTP/API, erreur réseau, timeout, erreur de formatage — pour qu'un échec ne
soit jamais silencieux (voir NotificationResult.status).
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_TELEGRAM_API_BASE = "https://api.telegram.org"


@dataclass
class NotificationResult:
    """
    Résultat précis de l'envoi d'une notification (Sprint 29.1).

    `status` (toujours renseigné, jamais None) documente exactement ce qui
    s'est passé :
      "sent"               : notification envoyée avec succès.
      "not_configured"     : aucun canal réel configuré (secrets absents) —
                             le résumé a été journalisé en local.
      "formatting_error"   : format_summary_text() a levé une exception.
      "network_error"      : requête réseau impossible (connexion, DNS...).
      "timeout"             : la requête a expiré.
      "invalid_bot_token"   : Telegram a renvoyé 401 (bot token invalide).
      "invalid_chat_id"     : Telegram a renvoyé 400 avec un message indiquant
                             que le chat_id est invalide/introuvable.
      "http_error"          : toute autre réponse HTTP non-200.
    """
    success: bool
    status: str
    detail: str = ""


@dataclass
class ChannelSummary:
    niche_name: str
    channel_name: str
    subject: str
    duration_seconds: int
    scene_count: int
    storage_link: Optional[str] = None  # lien Supabase Storage de cette production, si uploadé avec succès


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

    lines += ["", "📦 Storage:", "Supabase Storage"]
    links = [(c.niche_name, c.storage_link) for c in summary.channels if c.storage_link]
    if links:
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
    def send_daily_summary(self, summary: DailyProductionSummary) -> NotificationResult:
        """Envoie (ou journalise) le résumé de la production du jour."""
        ...


class LoggingNotificationService(NotificationService):
    """
    Implémentation de repli tant qu'aucun canal réel n'est configuré.

    `reason` (Sprint 29.1) documente précisément pourquoi ce canal de repli
    est utilisé (ex: "not_configured" si TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID
    sont absents) — remonté tel quel dans NotificationResult.status.
    """

    def __init__(self, reason: str = "not_configured") -> None:
        self._reason = reason

    def send_daily_summary(self, summary: DailyProductionSummary) -> NotificationResult:
        try:
            text = format_summary_text(summary)
        except Exception as exc:
            logger.warning("Formatage du résumé échoué (%s) — notification annulée.", exc)
            return NotificationResult(success=False, status="formatting_error", detail=str(exc))
        logger.info("\n%s", text)
        return NotificationResult(success=False, status=self._reason, detail="Telegram non configuré.")


class TelegramNotificationService(NotificationService):
    """
    Envoi réel via l'API Bot Telegram (sendMessage).

    Service annexe : aucune erreur (formatage, réseau, API, timeout) ne doit
    jamais remonter à l'appelant — le pipeline doit toujours pouvoir se
    terminer avec succès, packages de production déjà sauvegardés localement.

    Fiabilité (Sprint 29.1) : la cause précise d'un échec (bot token invalide,
    chat id invalide, erreur HTTP, réseau, timeout, formatage) est toujours
    identifiée et journalisée — jamais un échec silencieux.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    def send_daily_summary(self, summary: DailyProductionSummary) -> NotificationResult:
        try:
            text = format_summary_text(summary)
        except Exception as exc:
            logger.warning("Formatage du résumé Telegram échoué (%s) — notification annulée.", exc)
            return NotificationResult(success=False, status="formatting_error", detail=str(exc))

        import requests

        url = f"{_TELEGRAM_API_BASE}/bot{self._bot_token}/sendMessage"
        try:
            response = requests.post(url, data={"chat_id": self._chat_id, "text": text}, timeout=10)
        except requests.exceptions.Timeout as exc:
            logger.warning("Timeout lors de l'envoi Telegram (%s) — résumé journalisé localement en repli.", exc)
            logger.info("\n%s", text)
            return NotificationResult(success=False, status="timeout", detail=str(exc))
        except requests.exceptions.RequestException as exc:
            logger.warning("Erreur réseau lors de l'envoi Telegram (%s) — résumé journalisé localement en repli.", exc)
            logger.info("\n%s", text)
            return NotificationResult(success=False, status="network_error", detail=str(exc))

        if response.status_code == 200:
            logger.info("Notification Telegram envoyée (chat_id=%s).", self._chat_id)
            return NotificationResult(success=True, status="sent")

        detail = self._response_detail(response)
        if response.status_code == 401:
            status = "invalid_bot_token"
        elif response.status_code == 400 and "chat" in detail.lower():
            status = "invalid_chat_id"
        else:
            status = "http_error"

        logger.warning(
            "Envoi Telegram échoué (HTTP %s, raison=%s : %s) — résumé journalisé localement en repli.",
            response.status_code, status, detail,
        )
        logger.info("\n%s", text)
        return NotificationResult(success=False, status=status, detail=f"HTTP {response.status_code} — {detail}")

    @staticmethod
    def _response_detail(response) -> str:
        """Extrait le message d'erreur Telegram (`description`) sans jamais lever."""
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("description"):
                return str(payload["description"])
        except Exception:
            pass
        try:
            return (response.text or "")[:300]
        except Exception:
            return ""


def build_notification_service() -> NotificationService:
    """
    Construit le service selon la configuration disponible dans l'environnement.

    - TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID présents → TelegramNotificationService.
    - Un seul des deux secrets présent → LoggingNotificationService(reason="missing_secret")
      (configuration partielle, jamais silencieuse — voir NotificationResult.status).
    - Aucun des deux → LoggingNotificationService(reason="not_configured").
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if bot_token and chat_id:
        logger.info("Notification active : Telegram (chat_id=%s).", chat_id)
        return TelegramNotificationService(bot_token, chat_id)

    if bot_token or chat_id:
        missing = "TELEGRAM_CHAT_ID" if bot_token else "TELEGRAM_BOT_TOKEN"
        logger.warning("Configuration Telegram incomplète (%s manquant) — notification en mode log uniquement.", missing)
        return LoggingNotificationService(reason="missing_secret")

    logger.info("Telegram non configuré — notification en mode log uniquement.")
    return LoggingNotificationService(reason="not_configured")
