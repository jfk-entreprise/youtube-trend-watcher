"""
Test multi-agents — Sprint 3
Usage : python scripts/test_agents.py

Lance KeywordAgent et TrendingAgent, fusionne les résultats,
déduplique sur video_id et persiste dans le stockage existant.
"""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

from src.agents import KeywordAgent, TrendingAgent
from src.models import VideoSnapshot
from src.storage import build_storage

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

KEYWORDS = ["IA", "Business", "Argent", "Histoire", "Technologie"]
TRENDING_REGIONS = ["CI", "FR", "US"]

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "videos.csv"


# ── Utilitaire de déduplication ────────────────────────────────────────────────

def deduplicate(
    snapshots: list[VideoSnapshot],
    seen: set[str],
) -> tuple[list[VideoSnapshot], int]:
    """
    Filtre les doublons basés sur video_id.
    Retourne (uniques, nombre_de_doublons_trouvés).
    Modifie `seen` en place pour permettre la déduplication inter-agents.
    """
    unique: list[VideoSnapshot] = []
    dupes = 0
    for snap in snapshots:
        if snap.video_id in seen:
            dupes += 1
        else:
            seen.add(snap.video_id)
            unique.append(snap)
    return unique, dupes


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("YOUTUBE_API_KEY absent du fichier .env — arrêt.")
        sys.exit(1)

    region_code = os.getenv("YOUTUBE_REGION_CODE", "FR")
    language = os.getenv("YOUTUBE_LANGUAGE", "fr")

    # ── Collecte ──────────────────────────────────────────────────────────────

    keyword_agent = KeywordAgent(
        api_key,
        keywords=KEYWORDS,
        max_results_per_keyword=20,
        days_back=7,
        region_code=region_code,
        language=language,
    )
    trending_agent = TrendingAgent(
        api_key,
        region_codes=TRENDING_REGIONS,
        max_results=50,
    )

    logger.info("=== Sprint 3 — Test multi-agents ===")
    logger.info("")

    kw_raw = keyword_agent.collect()
    tr_raw = trending_agent.collect()

    # ── Déduplication ─────────────────────────────────────────────────────────
    # On déduplique d'abord les résultats du KeywordAgent, puis on élimine
    # les doublons du TrendingAgent en tenant compte des IDs déjà vus.

    seen: set[str] = set()
    kw_unique, kw_dupes = deduplicate(kw_raw, seen)
    tr_unique, tr_dupes = deduplicate(tr_raw, seen)

    all_unique = kw_unique + tr_unique

    # ── Persistance ───────────────────────────────────────────────────────────

    storage = build_storage(CSV_PATH)
    storage.save(all_unique)

    # ── Résumé ────────────────────────────────────────────────────────────────

    print(f"""
KeywordAgent :
- {len(kw_raw)} vidéos récupérées
- {kw_dupes} doublons supprimés

TrendingAgent :
- {len(tr_raw)} vidéos récupérées
- {tr_dupes} doublons supprimés

Total final :
- {len(all_unique)} vidéos uniques
""")


if __name__ == "__main__":
    main()
