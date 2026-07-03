"""
Virality Engine — Script de lancement.
Usage : python scripts/run_virality.py

Lit data/videos.csv, calcule les scores, affiche et sauvegarde le rapport.
Aucune donnée n'est modifiée ou collectée — ce script est en lecture seule
vis-à-vis du système de stockage.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

from src.virality_engine import ViralityEngine

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

CSV_PATH = ROOT / "data" / "videos.csv"
REPORTS_DIR = ROOT / "reports"


def main() -> None:
    engine = ViralityEngine(csv_path=CSV_PATH, top_n=20)
    report = engine.run()

    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"virality_{timestamp}.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nRapport sauvegardé → {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
