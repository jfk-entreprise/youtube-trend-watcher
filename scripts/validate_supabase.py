"""
validate_supabase.py — Valide la connexion Supabase (insertion + lecture).
Usage : python scripts/validate_supabase.py
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        logger.error("SUPABASE_URL ou SUPABASE_SERVICE_ROLE_KEY absent du .env — arrêt.")
        sys.exit(1)

    logger.info("Connexion à Supabase : %s", url)

    try:
        from supabase import create_client
        client = create_client(url, key)
    except ImportError:
        logger.error("Package 'supabase' introuvable. Exécuter : pip install supabase")
        sys.exit(1)

    from src.models import VideoSnapshot
    from src.storage import SupabaseStorage

    test_snap = VideoSnapshot(
        video_id="__validate_test__",
        title="[VALIDATION] Snapshot de test Sprint 6",
        channel_id="__test_channel__",
        channel_title="Validation Channel",
        published_at="2026-01-01T00:00:00Z",
        description="Snapshot créé par validate_supabase.py — peut être supprimé.",
        duration_iso="PT1M",
        duration_seconds=60,
        view_count=42,
        like_count=7,
        comment_count=1,
        keyword="validation",
        source="keyword",
        collected_at=datetime.now(timezone.utc).isoformat(),
    )

    storage = SupabaseStorage(url, key)

    # ── Test 1 : Insertion ────────────────────────────────────────────────────
    logger.info("Test 1/3 — Insertion d'un snapshot de validation...")
    inserted = storage.save([test_snap])
    assert inserted == 1, f"Attendu 1 ligne insérée, obtenu {inserted}"
    logger.info("  OK — %d ligne insérée.", inserted)

    # ── Test 2 : Lecture ──────────────────────────────────────────────────────
    logger.info("Test 2/3 — Lecture du snapshot inséré...")
    result = (
        client.table("video_snapshots")
        .select("video_id, title, source, collected_at")
        .eq("video_id", "__validate_test__")
        .order("collected_at", desc=True)
        .limit(1)
        .execute()
    )
    assert result.data, "Aucune ligne retournée — la lecture a échoué."
    row = result.data[0]
    logger.info("  OK — Ligne lue : %s", row)

    # ── Test 3 : Nettoyage ────────────────────────────────────────────────────
    logger.info("Test 3/3 — Suppression de la ligne de test...")
    client.table("video_snapshots").delete().eq("video_id", "__validate_test__").execute()
    logger.info("  OK — Ligne de test supprimée.")

    print("""
╔══════════════════════════════════════════╗
║  Validation Supabase : SUCCES            ║
║  Insertion + lecture + suppression OK.   ║
║  Le backend est pret pour la production. ║
╚══════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
