"""
Modèles de données du projet.

VideoSnapshot représente une capture d'une vidéo à un instant précis.
Chaque collecte génère de nouveaux snapshots — les anciennes lignes ne sont jamais modifiées.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class VideoSnapshot:
    """Capture instantanée d'une vidéo YouTube à un instant t."""

    # --- Identité ---
    video_id: str
    title: str
    channel_id: str
    channel_title: str
    published_at: str       # ISO 8601, ex: "2026-06-21T10:10:13Z"
    description: str        # Tronquée à 500 caractères

    # --- Durée ---
    duration_iso: str       # Format brut de l'API, ex: "PT9M33S"
    duration_seconds: int   # Converti en secondes, ex: 573

    # --- Statistiques (Optional : le créateur peut les masquer) ---
    view_count: Optional[int]
    like_count: Optional[int]
    comment_count: Optional[int]

    # --- Contexte de collecte ---
    keyword: str            # Mot-clé ayant ramené la vidéo ; vide ("") pour les agents sans recherche
    source: str = "keyword" # Agent source : 'keyword' | 'trending'
    collected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
