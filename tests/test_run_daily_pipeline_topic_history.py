"""
Tests du câblage anti-doublon (Sprint 33 — TopicHistoryFilter/topic_history)
dans scripts/run_daily_pipeline.py : step_filter_recent_topics() et
l'enregistrement des sujets produits dans step_build_packages().

scripts/ n'est pas un package Python — le module est chargé directement
depuis son chemin de fichier (importlib), comme le fait déjà
tests/test_run_daily_pipeline_summary.py.
"""

import importlib.util
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_daily_pipeline.py"
_env_backup = dict(os.environ)
_spec = importlib.util.spec_from_file_location("run_daily_pipeline_topic_history_test", _SCRIPT_PATH)
run_daily_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["run_daily_pipeline_topic_history_test"] = run_daily_pipeline
_spec.loader.exec_module(run_daily_pipeline)
os.environ.clear()
os.environ.update(_env_backup)

from src.niche_intelligence import Niche
from src.opportunity_engine import Opportunity
from src.topic_history import JsonTopicHistoryStore, TopicHistoryFilter, TopicRecord


def _opportunity(title: str, source_video_id: str) -> Opportunity:
    return Opportunity(
        title=title, niche="IA", source_video_id=source_video_id,
        overall_score=0.8, virality_score=0.7, growth_score=0.6, evergreen_score=0.7,
        trend_score=0.7, competition_score=0.4, production_difficulty=0.5, urgency=0.6,
        recommendation="Produire.", rationale=[], metadata={},
    )


def _niche(name: str) -> Niche:
    return Niche(name=name, volume=10, avg_views=10_000.0, avg_engagement=1.5,
                 avg_growth_speed=100.0, niche_score=5.0, timelines=[])


class TestStepFilterRecentTopics:
    def test_filters_each_niche_independently(self, tmp_path):
        store = JsonTopicHistoryStore(tmp_path / "topic_history.json")
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        store.save_topic(TopicRecord(
            title="5 métiers développeur transformés par l'IA en 2027",
            niche="IA", brand_id="ia_fr", produced_date=yesterday, source_video_id="old",
        ))
        topic_filter = TopicHistoryFilter(store=store)

        opportunities_by_niche = {
            "IA": [
                _opportunity("5 métiers développeur transformés par l'IA en 2027", "dup"),
                _opportunity("Recette de cuisine facile pour débutants", "new_ia"),
            ],
            "Histoire": [
                _opportunity("La chute de l'Empire romain expliquée", "hist_1"),
            ],
        }

        result = run_daily_pipeline.step_filter_recent_topics(opportunities_by_niche, topic_filter)

        assert [o.source_video_id for o in result["IA"]] == ["new_ia"]
        assert [o.source_video_id for o in result["Histoire"]] == ["hist_1"]

    def test_empty_niche_stays_empty(self, tmp_path):
        store = JsonTopicHistoryStore(tmp_path / "topic_history.json")
        topic_filter = TopicHistoryFilter(store=store)

        result = run_daily_pipeline.step_filter_recent_topics({"IA": []}, topic_filter)

        assert result == {"IA": []}


def _production(final_script_title: str = "Titre du script") -> dict:
    """Sprint 35 — un dict de production couvre désormais 1 niche/2 langues."""
    final_script_en = SimpleNamespace(title=final_script_title, scenes=[], estimated_duration=100)
    final_script_fr = SimpleNamespace(title=final_script_title, scenes=[], estimated_duration=100)
    return {
        "niche": _niche("IA"),
        "brand_en": SimpleNamespace(id="global_us", name="Global US"),
        "brand_fr": SimpleNamespace(id="ia_fr", name="IA FR"),
        "final_script_en": final_script_en,
        "final_script_fr": final_script_fr,
        "images": [],
        "animations_en": [],
        "animations_fr": [],
        "rewrite_result": None,
        "best_entry": {"opportunity": _opportunity("Sujet du jour", "vid_today")},
    }


class TestStepBuildPackagesSavesTopicHistory:
    def test_saves_one_topic_record_per_niche_production(self, tmp_path):
        store = JsonTopicHistoryStore(tmp_path / "topic_history.json")
        prod = _production()

        class _StubBuilder:
            def build(self, output_dir, idx, package_result):
                return output_dir / f"niche_{idx:02d}"

        run_daily_pipeline.step_build_packages(tmp_path, [prod], _StubBuilder(), topic_store=store)

        records = store.load_recent(days=30)
        assert len(records) == 1
        assert records[0].title == "Titre du script"
        assert records[0].niche == "IA"
        assert records[0].brand_id == "ia_fr"
        assert records[0].source_video_id == "vid_today"
        assert records[0].market == "FR"

    def test_save_failure_does_not_break_package_building(self, tmp_path):
        class _BrokenStore:
            def save_topic(self, record):
                raise RuntimeError("relation 'topic_history' does not exist")

        prod = _production()

        class _StubBuilder:
            def build(self, output_dir, idx, package_result):
                return output_dir / f"niche_{idx:02d}"

        package_dirs = run_daily_pipeline.step_build_packages(
            tmp_path, [prod], _StubBuilder(), topic_store=_BrokenStore()
        )

        assert package_dirs == [tmp_path / "niche_01"]
