"""
Collecteur YouTube — encapsule tous les appels à l'API YouTube Data v3.

Coûts en quota :
  - search.list  : 100 unités / appel
  - videos.list  :   1 unité  / appel (jusqu'à 50 IDs)
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from googleapiclient.discovery import build

from .models import VideoSnapshot

logger = logging.getLogger(__name__)

_ISO_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _parse_iso_duration(duration: str) -> int:
    """Convertit une durée ISO 8601 (ex: PT14M48S) en secondes."""
    match = _ISO_DURATION_RE.match(duration or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _safe_int(value: Optional[str]) -> Optional[int]:
    """Convertit une chaîne en entier ; retourne None si absent ou invalide."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class YouTubeCollector:
    """
    Façade pour l'API YouTube Data v3.

    Utilisation :
        collector = YouTubeCollector(api_key="...", region_code="FR", language="fr")
        snapshots = collector.collect_for_keywords(["IA", "Business"], days_back=7)
    """

    def __init__(self, api_key: str, region_code: str = "FR", language: str = "fr"):
        self._youtube = build("youtube", "v3", developerKey=api_key)
        self._region_code = region_code
        self._language = language

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def collect_for_keywords(
        self,
        keywords: list[str],
        max_results_per_keyword: int = 20,
        days_back: int = 7,
    ) -> list[VideoSnapshot]:
        """
        Collecte principale : cherche des vidéos pour chaque mot-clé
        puis enrichit les résultats avec les métadonnées complètes.

        Args:
            keywords: Liste de mots-clés à rechercher.
            max_results_per_keyword: Nombre de vidéos par mot-clé (max 50).
            days_back: Fenêtre temporelle de recherche (jours en arrière).

        Returns:
            Liste de VideoSnapshot prêts à être persistés.
        """
        published_after = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Étape 1 : search.list — collecte des IDs (100 unités × nb de mots-clés)
        # On garde le premier mot-clé qui trouve une vidéo donnée.
        id_to_keyword: dict[str, str] = {}
        for keyword in keywords:
            ids = self._search_video_ids(keyword, max_results_per_keyword, published_after)
            for vid_id in ids:
                id_to_keyword.setdefault(vid_id, keyword)
            logger.info("Mot-clé '%s' → %d IDs", keyword, len(ids))

        if not id_to_keyword:
            logger.warning("Aucun résultat pour les mots-clés fournis.")
            return []

        # Étape 2 : videos.list — enrichissement par batch de 50 (1 unité / batch)
        all_ids = list(id_to_keyword.keys())
        snapshots: list[VideoSnapshot] = []
        for i in range(0, len(all_ids), 50):
            batch = all_ids[i : i + 50]
            snapshots.extend(self._fetch_details(batch, id_to_keyword))

        logger.info("Total : %d snapshots collectés", len(snapshots))
        return snapshots

    # ------------------------------------------------------------------
    # Méthodes privées (appels API)
    # ------------------------------------------------------------------

    def _search_video_ids(
        self,
        keyword: str,
        max_results: int,
        published_after: str,
    ) -> list[str]:
        """
        Appelle search.list et retourne uniquement les IDs de vidéos.
        Coût : 100 unités.
        """
        response = (
            self._youtube.search()
            .list(
                q=keyword,
                type="video",
                part="id",
                maxResults=min(max_results, 50),
                publishedAfter=published_after,
                regionCode=self._region_code,
                relevanceLanguage=self._language,
                order="date",
            )
            .execute()
        )
        return [item["id"]["videoId"] for item in response.get("items", [])]

    def _fetch_details(
        self,
        video_ids: list[str],
        id_to_keyword: dict[str, str],
    ) -> list[VideoSnapshot]:
        """
        Appelle videos.list pour enrichir un batch d'IDs.
        Coût : 1 unité pour jusqu'à 50 IDs.
        """
        response = (
            self._youtube.videos()
            .list(
                id=",".join(video_ids),
                part="snippet,statistics,contentDetails",
            )
            .execute()
        )

        snapshots = []
        for item in response.get("items", []):
            vid_id = item["id"]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            details = item.get("contentDetails", {})
            duration_iso = details.get("duration", "PT0S")

            snapshots.append(
                VideoSnapshot(
                    video_id=vid_id,
                    title=snippet.get("title", ""),
                    channel_id=snippet.get("channelId", ""),
                    channel_title=snippet.get("channelTitle", ""),
                    published_at=snippet.get("publishedAt", ""),
                    description=snippet.get("description", "")[:500],
                    duration_iso=duration_iso,
                    duration_seconds=_parse_iso_duration(duration_iso),
                    view_count=_safe_int(stats.get("viewCount")),
                    like_count=_safe_int(stats.get("likeCount")),
                    comment_count=_safe_int(stats.get("commentCount")),
                    keyword=id_to_keyword[vid_id],
                )
            )
        return snapshots
