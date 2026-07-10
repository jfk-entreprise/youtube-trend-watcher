"""
TrendingAgent — récupère les vidéos les plus populaires via videos.list (chart=mostPopular).

Contrairement au KeywordAgent, cet agent n'effectue pas de search.list.
Un seul appel videos.list par pays suffit à obtenir jusqu'à 50 vidéos complètes.

Coût quota : 1 unité par pays (très économique).
"""

import logging
import re
from typing import Optional

from googleapiclient.discovery import build

from src.agents.base import BaseAgent
from src.models import VideoSnapshot

logger = logging.getLogger(__name__)

_ISO_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _parse_duration(iso: str) -> int:
    m = _ISO_RE.match(iso or "")
    if not m:
        return 0
    h, mn, s = (int(g or 0) for g in m.groups())
    return h * 3600 + mn * 60 + s


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class TrendingAgent(BaseAgent):
    """
    Collecte les vidéos tendance (chart=mostPopular) pour une liste de pays.

    Exemple :
        agent = TrendingAgent(api_key, region_codes=["CI", "FR", "US"], max_results=50)
        snapshots = agent.collect()
    """

    def __init__(
        self,
        api_key: str,
        region_codes: Optional[list[str]] = None,
        max_results: int = 50,
        category_id: Optional[str] = None,
    ) -> None:
        self._youtube = build("youtube", "v3", developerKey=api_key)
        self._region_codes = region_codes or ["FR"]
        self._max_results = min(max_results, 50)  # limite API : 50 par appel
        self._category_id = category_id

    @property
    def name(self) -> str:
        return "TrendingAgent"

    def collect(self) -> list[VideoSnapshot]:
        logger.info("[%s] Collecte tendances pour : %s", self.name, ", ".join(self._region_codes))
        all_snapshots: list[VideoSnapshot] = []
        for region_code in self._region_codes:
            snapshots = self._fetch_trending(region_code)
            logger.info("[%s] %s → %d vidéos", self.name, region_code, len(snapshots))
            all_snapshots.extend(snapshots)
        logger.info("[%s] %d snapshots collectés au total", self.name, len(all_snapshots))
        return all_snapshots

    def _fetch_trending(self, region_code: str) -> list[VideoSnapshot]:
        """
        Appelle videos.list avec chart=mostPopular.
        Coût : 1 unité (retourne snippet + statistics + contentDetails en un seul appel).
        """
        params: dict = dict(
            chart="mostPopular",
            part="snippet,statistics,contentDetails",
            maxResults=self._max_results,
            regionCode=region_code,
        )
        if self._category_id:
            params["videoCategoryId"] = self._category_id

        try:
            response = self._youtube.videos().list(**params).execute()
        except Exception as exc:
            logger.error("[%s] videos.list error (%s) : %s", self.name, region_code, exc)
            return []

        snapshots: list[VideoSnapshot] = []
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
                    duration_seconds=_parse_duration(duration_iso),
                    view_count=_safe_int(stats.get("viewCount")),
                    like_count=_safe_int(stats.get("likeCount")),
                    comment_count=_safe_int(stats.get("commentCount")),
                    keyword=region_code,   # pays source, pas de mot-clé pour cet agent
                    source="trending",
                )
            )
        return snapshots
