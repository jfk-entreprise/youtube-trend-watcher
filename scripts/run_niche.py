"""
Niche Intelligence Engine — Script de lancement.
Usage : python scripts/run_niche.py

Lit data/videos.csv, analyse les niches par mot-clé, affiche et sauvegarde le rapport.
Si Supabase est configuré, les données sont d'abord synchronisées depuis
Supabase vers le CSV local avant l'analyse.
"""

import csv as csv_module
import dataclasses
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.storage import CSV_COLUMNS, CsvStorage, build_storage
from src.niche_intelligence import NicheAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_PATH = ROOT / "data" / "videos.csv"
REPORTS_DIR = ROOT / "reports"


def _sync_to_csv_if_needed(csv_path: Path) -> None:
    """Synchronise Supabase → CSV local si le backend actif n'est pas CsvStorage."""
    backend = build_storage(csv_path)
    if isinstance(backend, CsvStorage):
        return

    snapshots = backend.load()
    if not snapshots:
        logger.warning("Backend Supabase vide — aucune donnée à synchroniser vers le CSV.")
        return

    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for snap in snapshots:
            writer.writerow(dataclasses.asdict(snap))

    logger.info(
        "%d snapshots synchronisés depuis Supabase → %s",
        len(snapshots), csv_path.name,
    )


def main() -> None:
    _sync_to_csv_if_needed(CSV_PATH)

    analyzer = NicheAnalyzer(csv_path=CSV_PATH, top_n=10)
    report = analyzer.run()

    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"niche_{timestamp}.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nRapport sauvegardé → {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
