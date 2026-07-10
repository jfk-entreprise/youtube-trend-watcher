"""
Utilitaires partagés entre les moteurs du YouTube Trend Watcher.

Centralise les fonctions de parsing, formatage et conversion
qui étaient dupliquées dans plusieurs modules.

But :
  - Éliminer la dette technique (code dupliqué)
  - Standardiser les formats d'affichage
  - Faciliter la maintenance et les tests
"""

import csv
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Parsing des durées ISO 8601 ──────────────────────────────────────────────

_ISO_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def parse_iso_duration(duration: str) -> int:
    """
    Convertit une durée ISO 8601 (ex: PT14M48S, PT1H30M15S) en secondes.

    Args:
        duration: Chaîne de durée ISO 8601.

    Returns:
        Nombre total de secondes. Retourne 0 si la chaîne est invalide ou vide.
    """
    match = _ISO_DURATION_RE.match(duration or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


# ── Conversion sécurisée d'entiers ────────────────────────────────────────────

def safe_int(value: Optional[str | int]) -> Optional[int]:
    """
    Convertit une valeur en entier ; retourne None si absent ou invalide.

    Args:
        value: Chaîne, entier ou None.

    Returns:
        Entier ou None si la conversion est impossible.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ── Parsing de dates ISO 8601 ────────────────────────────────────────────────

def parse_dt(s: str) -> datetime:
    """
    Parse une chaîne ISO 8601 en datetime UTC.

    Supporte les formats avec Z, +00:00, ou sans fuseau horaire.

    Args:
        s: Chaîne au format ISO 8601.

    Returns:
        Datetime en timezone UTC.
    """
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def age_days(dt_str: str) -> float:
    """
    Calcule l'âge d'une date ISO 8601 en jours par rapport à maintenant.

    Args:
        dt_str: Chaîne ISO 8601 (ex: "2026-06-21T10:10:13Z").

    Returns:
        Nombre de jours (float). Minimum 0.0.
        Retourne 30.0 si le parsing échoue.
    """
    try:
        published = parse_dt(dt_str)
        return max((datetime.now(timezone.utc) - published).total_seconds() / 86400, 0.0)
    except Exception:
        return 30.0


# ── Formatage des durées ─────────────────────────────────────────────────────

def fmt_duration(seconds: int) -> str:
    """
    Formate une durée en secondes en format lisible.

    Exemples :
        45    → "45s"
        125   → "2m05s"
        3661  → "1h01m01s"

    Args:
        seconds: Durée en secondes.

    Returns:
        Chaîne formatée.
    """
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def fmt_views(n: Optional[int]) -> str:
    """
    Formate un nombre de vues en format lisible court.

    Exemples :
        42       → "42"
        1500     → "1.5k"
        2500000  → "2.5M"
        None     → "N/A"

    Args:
        n: Nombre de vues ou None.

    Returns:
        Chaîne formatée.
    """
    if n is None:
        return "N/A"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


# ── Chargement CSV partagé (ViralityEngine + NicheAnalyzer) ──────────────────

def csv_snapshots_to_timelines(csv_path: Path) -> dict[str, list]:
    """
    Charge un fichier CSV de snapshots et les regroupe par video_id.

    Fonction partagée entre ViralityEngine._load_timelines()
    et NicheAnalyzer._load_timelines() pour éliminer la duplication.

    Args:
        csv_path: Chemin vers le fichier CSV (format défini par CSV_COLUMNS).

    Returns:
        Dictionnaire {video_id: [VideoSnapshot, ...]} trié par collected_at.
        Retourne un dict vide si le fichier est absent.

    Note :
        Les lignes mal formées sont ignorées et comptabilisées dans les logs.
    """
    from src.models import VideoSnapshot

    if not csv_path.exists():
        logger.error("Fichier CSV introuvable : %s", csv_path)
        return {}

    buckets: dict[str, list[VideoSnapshot]] = {}
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                snap = VideoSnapshot(
                    video_id=row["video_id"],
                    title=row["title"],
                    channel_id=row["channel_id"],
                    channel_title=row["channel_title"],
                    published_at=row["published_at"],
                    description=row.get("description", ""),
                    duration_iso=row["duration_iso"],
                    duration_seconds=int(row["duration_seconds"] or 0),
                    view_count=safe_int(row.get("view_count")),
                    like_count=safe_int(row.get("like_count")),
                    comment_count=safe_int(row.get("comment_count")),
                    keyword=row["keyword"],
                    source=row.get("source", "keyword"),
                    collected_at=row["collected_at"],
                )
                buckets.setdefault(snap.video_id, []).append(snap)
            except Exception as exc:
                skipped += 1
                logger.debug("Ligne ignorée (%s)", exc)

    if skipped:
        logger.warning("%d ligne(s) ignorée(s) lors du chargement.", skipped)

    # Tri chronologique à l'intérieur de chaque groupe
    for vid_id in buckets:
        buckets[vid_id].sort(key=lambda s: s.collected_at)

    logger.info(
        "%d snapshots → %d vidéos uniques (depuis %s)",
        sum(len(snaps) for snaps in buckets.values()),
        len(buckets),
        csv_path.name,
    )
    return buckets
