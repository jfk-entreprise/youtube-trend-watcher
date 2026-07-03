"""
KeywordAgent — recherche de vidéos par mots-clés via l'API YouTube Data v3.

Délègue la collecte à YouTubeCollector existant.
Les snapshots produits ont source='keyword' (valeur par défaut du modèle).

Coût quota : 100 unités × nb de mots-clés  +  1 unité par batch de 50 IDs.
"""

import logging

from src.agents.base import BaseAgent
from src.collector import YouTubeCollector
from src.models import VideoSnapshot

logger = logging.getLogger(__name__)


class KeywordAgent(BaseAgent):
    """
    Collecte des vidéos correspondant à une liste de mots-clés.

    Exemple :
        agent = KeywordAgent(api_key, keywords=["IA", "Business"], days_back=7)
        snapshots = agent.collect()
    """

    def __init__(
        self,
        api_key: str,
        keywords: list[str],
        max_results_per_keyword: int = 20,
        days_back: int = 7,
        region_code: str = "FR",
        language: str = "fr",
    ) -> None:
        self._collector = YouTubeCollector(api_key, region_code=region_code, language=language)
        self._keywords = keywords
        self._max_results = max_results_per_keyword
        self._days_back = days_back

    @property
    def name(self) -> str:
        return "KeywordAgent"

    def collect(self) -> list[VideoSnapshot]:
        logger.info("[%s] Collecte sur %d mots-clés (fenêtre : %dj)", self.name, len(self._keywords), self._days_back)
        snapshots = self._collector.collect_for_keywords(
            keywords=self._keywords,
            max_results_per_keyword=self._max_results,
            days_back=self._days_back,
        )
        logger.info("[%s] %d snapshots collectés", self.name, len(snapshots))
        return snapshots
