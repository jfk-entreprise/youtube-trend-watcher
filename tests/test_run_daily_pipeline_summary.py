"""
Tests du résumé de fin de run (Issue 5, Sprint 29.1 ; Storage renommé Sprint 30)
— build_production_summary_text() dans scripts/run_daily_pipeline.py.

scripts/ n'est pas un package Python — le module est chargé directement
depuis son chemin de fichier (importlib), comme le ferait `python scripts/run_daily_pipeline.py`.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# run_daily_pipeline.py appelle load_dotenv() au chargement du module — sans
# précaution, ça injecterait les vrais secrets de .env (DEEPSEEK_API_KEY...)
# dans os.environ pour le reste de la session pytest, faussant les tests
# d'auto-détection de provider LLM (test_llm.py, test_llm_groq.py) qui
# tournent dans le même processus. On restaure l'environnement juste après.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_daily_pipeline.py"
_env_backup = dict(os.environ)
_spec = importlib.util.spec_from_file_location("run_daily_pipeline", _SCRIPT_PATH)
run_daily_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["run_daily_pipeline"] = run_daily_pipeline
_spec.loader.exec_module(run_daily_pipeline)
os.environ.clear()
os.environ.update(_env_backup)

from src.supabase_storage_uploader import UploadResult
from src.notification_service import NotificationResult


def _prod(images_providers, animations_providers):
    return {
        "images": [
            {"scene_order": i + 1, "image_prompt": SimpleNamespace(metadata={"provider": p})}
            for i, p in enumerate(images_providers)
        ],
        "animations": [
            {"scene_order": i + 1, "animation_prompt": SimpleNamespace(metadata={"provider": p})}
            for i, p in enumerate(animations_providers)
        ],
    }


class TestCountLlmVsFallback:
    def test_counts_llm_and_fallback_separately(self):
        prods = [
            _prod(["deepseek", "heuristic_image_v1"], ["deepseek", "deepseek"]),
            _prod(["heuristic_image_v1"], ["fallback_heuristic"]),
        ]
        images_llm, images_fallback = run_daily_pipeline._count_llm_vs_fallback(
            prods, "images", "image_prompt", run_daily_pipeline._IMAGE_FALLBACK_PROVIDER,
        )
        assert (images_llm, images_fallback) == (1, 2)

        animations_llm, animations_fallback = run_daily_pipeline._count_llm_vs_fallback(
            prods, "animations", "animation_prompt", run_daily_pipeline._ANIMATION_FALLBACK_PROVIDER,
        )
        assert (animations_llm, animations_fallback) == (2, 1)


class TestBuildProductionSummaryText:
    def test_full_success(self):
        prods = [_prod(["deepseek"] * 9, ["deepseek"] * 9), _prod(["deepseek"] * 9, ["deepseek"] * 9)]
        storage_results = [
            UploadResult(success=True, uploaded_count=21, total_count=21,
                         remote_url="https://proj.supabase.co/storage/v1/object/public/production/a"),
            UploadResult(success=True, uploaded_count=21, total_count=21,
                         remote_url="https://proj.supabase.co/storage/v1/object/public/production/b"),
        ]
        telegram_result = NotificationResult(success=True, status="sent")

        text = run_daily_pipeline.build_production_summary_text(prods, storage_results, telegram_result, 684.0)

        assert "Videos produced: 2" in text
        assert "LLM: 18" in text
        assert "Fallback: 0" in text
        assert "Uploaded files: 42/42" in text
        assert "Upload status: SUCCESS" in text
        assert "Status: SENT" in text
        assert "11m 24s" in text
        assert "Overall status:\nSUCCESS" in text
        assert "Overall status:\nSUCCESS (with warnings)" not in text

    def test_partial_storage_upload_flags_warning(self):
        prods = [_prod(["deepseek", "heuristic_image_v1"], ["deepseek"])]
        storage_results = [
            UploadResult(success=False, uploaded_count=2, total_count=3,
                         remote_url="https://proj.supabase.co/storage/v1/object/public/production/a",
                         error="1 fichier(s) en échec"),
        ]
        telegram_result = NotificationResult(success=True, status="sent")

        text = run_daily_pipeline.build_production_summary_text(prods, storage_results, telegram_result, 60.0)

        assert "Fallback: 1" in text
        assert "Uploaded files: 2/3" in text
        assert "Upload status: PARTIAL" in text
        assert "Overall status:\nSUCCESS (with warnings)" in text

    def test_telegram_failure_flags_warning_without_breaking(self):
        prods = [_prod(["deepseek"], ["deepseek"])]
        storage_results = [
            UploadResult(success=True, uploaded_count=3, total_count=3,
                         remote_url="https://proj.supabase.co/storage/v1/object/public/production/a"),
        ]
        telegram_result = NotificationResult(success=False, status="invalid_bot_token", detail="401 Unauthorized")

        text = run_daily_pipeline.build_production_summary_text(prods, storage_results, telegram_result, 45.0)

        assert "Status: INVALID_BOT_TOKEN" in text
        assert "Overall status:\nSUCCESS (with warnings)" in text

    def test_no_storage_results_reports_skipped(self):
        prods = [_prod(["deepseek"], ["deepseek"])]
        telegram_result = NotificationResult(success=True, status="sent")

        text = run_daily_pipeline.build_production_summary_text(prods, [], telegram_result, 10.0)

        assert "Uploaded files: 0/0" in text
        assert "Upload status: SKIPPED" in text
