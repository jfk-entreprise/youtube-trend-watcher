"""
Sprint 2 — Collecteur de données YouTube
Usage : python scripts/sprint2_collect.py
"""

import logging
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.collector import YouTubeCollector
from src.storage import build_storage

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration — modifier ici pour ajuster la collecte
# ------------------------------------------------------------------

KEYWORDS = ["IA", "Business", "Argent", "Histoire", "Technologie"]
MAX_RESULTS_PER_KEYWORD = 20   # max 50 ; quota : 100 unités × nb de mots-clés
DAYS_BACK = 7                  # fenêtre de recherche en jours

DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "videos.csv"


def main() -> None:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("YOUTUBE_API_KEY absent du fichier .env — arrêt.")
        sys.exit(1)

    region_code = os.getenv("YOUTUBE_REGION_CODE", "FR")
    language = os.getenv("YOUTUBE_LANGUAGE", "fr")

    logger.info(
        "Démarrage — %d mots-clés | fenêtre : %d jours | région : %s",
        len(KEYWORDS), DAYS_BACK, region_code,
    )

    collector = YouTubeCollector(api_key, region_code=region_code, language=language)
    storage = build_storage(CSV_PATH)

    snapshots = collector.collect_for_keywords(
        keywords=KEYWORDS,
        max_results_per_keyword=MAX_RESULTS_PER_KEYWORD,
        days_back=DAYS_BACK,
    )

    if not snapshots:
        logger.warning("Aucune vidéo collectée — vérifier les mots-clés ou la clé API.")
        return

    saved = storage.save(snapshots)
    logger.info(
        "Collecte terminée — %d snapshots dans %s",
        saved, CSV_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
