from datetime import date

import pytest

from src.opportunity_engine import Opportunity
from src.topic_history import (
    FallbackTopicHistoryStore,
    JsonTopicHistoryStore,
    TopicHistoryFilter,
    TopicHistoryStore,
    TopicRecord,
    topic_similarity,
)


def _opportunity(title: str, niche: str = "IA", source_video_id: str = "vid_1") -> Opportunity:
    return Opportunity(
        title=title, niche=niche, source_video_id=source_video_id,
        overall_score=0.8, virality_score=0.7, growth_score=0.6, evergreen_score=0.7,
        trend_score=0.7, competition_score=0.4, production_difficulty=0.5, urgency=0.6,
        recommendation="Produire.", rationale=[], metadata={},
    )


def _record(title: str, niche: str = "IA", produced_date: str = "2026-01-01", market: str = "FR") -> TopicRecord:
    return TopicRecord(
        title=title, niche=niche, brand_id="ia_fr",
        produced_date=produced_date, source_video_id="vid_old", market=market,
    )


@pytest.fixture
def store(tmp_path):
    return JsonTopicHistoryStore(tmp_path / "topic_history.json")


# ── topic_similarity ─────────────────────────────────────────────────────────

class TestTopicSimilarity:
    def test_identical_titles_are_maximally_similar(self):
        assert topic_similarity("L'IA va-t-elle remplacer les développeurs ?",
                                 "L'IA va-t-elle remplacer les développeurs ?") == pytest.approx(1.0)

    def test_reworded_same_topic_is_highly_similar(self):
        score = topic_similarity(
            "5 métiers développeur transformés par l'IA",
            "Comment l'IA transforme 5 métiers de développeur",
        )
        assert score > 0.4

    def test_unrelated_titles_are_dissimilar(self):
        score = topic_similarity(
            "5 métiers développeur transformés par l'IA",
            "Recette de cuisine facile pour débutants",
        )
        assert score < 0.3

    def test_empty_title_is_not_similar(self):
        assert topic_similarity("", "Un sujet quelconque") == 0.0


# ── Store ────────────────────────────────────────────────────────────────────

class TestJsonTopicHistoryStore:
    def test_empty_store_returns_no_records(self, store):
        assert store.load_recent(days=5) == []

    def test_saved_topic_is_returned(self, store):
        store.save_topic(_record("Un sujet", produced_date="2026-01-10"))
        records = store.load_recent(days=5, today=date(2026, 1, 12))
        assert [r.title for r in records] == ["Un sujet"]

    def test_old_topic_outside_window_is_excluded(self, store):
        store.save_topic(_record("Vieux sujet", produced_date="2026-01-01"))
        records = store.load_recent(days=5, today=date(2026, 1, 12))
        assert records == []

    def test_multiple_topics_accumulate(self, store):
        store.save_topic(_record("Sujet A", produced_date="2026-01-10"))
        store.save_topic(_record("Sujet B", produced_date="2026-01-11"))
        records = store.load_recent(days=5, today=date(2026, 1, 12))
        assert {r.title for r in records} == {"Sujet A", "Sujet B"}


class _BrokenStore(TopicHistoryStore):
    """Simule une table Supabase absente (sql/create_topic_history.sql non exécuté)."""

    def load_recent(self, days=5, today=None):
        raise RuntimeError("relation 'topic_history' does not exist")

    def save_topic(self, record):
        raise RuntimeError("relation 'topic_history' does not exist")


class TestFallbackTopicHistoryStore:
    def test_falls_back_to_secondary_on_load_failure(self, store):
        store.save_topic(_record("Sujet A", produced_date="2026-01-10"))
        fallback_store = FallbackTopicHistoryStore(primary=_BrokenStore(), fallback=store)

        records = fallback_store.load_recent(days=5, today=date(2026, 1, 12))

        assert [r.title for r in records] == ["Sujet A"]

    def test_falls_back_to_secondary_on_save_failure(self, store):
        fallback_store = FallbackTopicHistoryStore(primary=_BrokenStore(), fallback=store)

        fallback_store.save_topic(_record("Sujet A", produced_date="2026-01-10"))

        assert [r.title for r in store.load_recent(days=5, today=date(2026, 1, 12))] == ["Sujet A"]


# ── TopicHistoryFilter ───────────────────────────────────────────────────────

class TestClassify:
    def test_new_topic_with_empty_history(self, store):
        filt = TopicHistoryFilter(store=store)
        result = filt.classify("Un sujet totalement neuf", "IA", today=date(2026, 1, 12))
        assert result.status == "new"

    def test_near_duplicate_is_classified_as_duplicate(self, store):
        store.save_topic(_record("5 métiers développeur transformés par l'IA en 2027", produced_date="2026-01-10"))
        filt = TopicHistoryFilter(store=store)
        result = filt.classify("5 métiers développeur transformés par l'IA en 2027", "IA", today=date(2026, 1, 12))
        assert result.status == "duplicate"

    def test_moderately_related_topic_is_classified_as_sequel(self, store):
        store.save_topic(_record("Pourquoi l'IA générative explose en 2026", produced_date="2026-01-10"))
        filt = TopicHistoryFilter(
            store=store, duplicate_threshold=0.9, sequel_threshold=0.1,
        )
        result = filt.classify("L'IA générative va-t-elle remplacer les artistes ?", "IA", today=date(2026, 1, 12))
        assert result.status == "sequel"

    def test_different_niche_is_never_matched(self, store):
        store.save_topic(_record("5 métiers développeur transformés par l'IA", niche="IA", produced_date="2026-01-10"))
        filt = TopicHistoryFilter(store=store)
        result = filt.classify("5 métiers développeur transformés par l'IA", "Histoire", today=date(2026, 1, 12))
        assert result.status == "new"

    def test_topic_outside_lookback_window_is_ignored(self, store):
        store.save_topic(_record("5 métiers développeur transformés par l'IA", produced_date="2026-01-01"))
        filt = TopicHistoryFilter(store=store, lookback_days=5)
        result = filt.classify("5 métiers développeur transformés par l'IA", "IA", today=date(2026, 1, 12))
        assert result.status == "new"

    def test_same_niche_different_market_is_never_matched(self, store):
        """Sprint 34 — la même niche produite côté US ne doit jamais être vue
        comme un doublon de la version FR (et inversement)."""
        store.save_topic(_record(
            "5 métiers développeur transformés par l'IA en 2027",
            niche="IA", produced_date="2026-01-10",
        ))  # market="FR" par défaut
        filt = TopicHistoryFilter(store=store)
        result = filt.classify(
            "5 métiers développeur transformés par l'IA en 2027", "IA", market="US",
            today=date(2026, 1, 12),
        )
        assert result.status == "new"


class TestFilterOpportunities:
    def test_no_history_keeps_all_opportunities_unmodified(self, store):
        filt = TopicHistoryFilter(store=store)
        opportunities = [_opportunity("Sujet A"), _opportunity("Sujet B")]

        result = filt.filter_opportunities(opportunities, "IA", today=date(2026, 1, 12))

        assert result == opportunities

    def test_duplicate_opportunity_is_dropped_when_alternative_exists(self, store):
        store.save_topic(_record("5 métiers développeur transformés par l'IA en 2027", produced_date="2026-01-10"))
        filt = TopicHistoryFilter(store=store)
        opportunities = [
            _opportunity("5 métiers développeur transformés par l'IA en 2027", source_video_id="dup"),
            _opportunity("Recette de cuisine facile pour débutants", source_video_id="new"),
        ]

        result = filt.filter_opportunities(opportunities, "IA", today=date(2026, 1, 12))

        assert [o.source_video_id for o in result] == ["new"]

    def test_duplicate_in_other_market_does_not_affect_this_market(self, store):
        """Sprint 34 — un sujet produit côté FR ne doit jamais faire écarter
        une opportunité identique côté US (marchés isolés)."""
        store.save_topic(_record(
            "5 métiers développeur transformés par l'IA en 2027",
            produced_date="2026-01-10", market="FR",
        ))
        filt = TopicHistoryFilter(store=store)
        opportunities = [_opportunity("5 métiers développeur transformés par l'IA en 2027", source_video_id="us_vid")]

        result = filt.filter_opportunities(opportunities, "IA", market="US", today=date(2026, 1, 12))

        assert [o.source_video_id for o in result] == ["us_vid"]
        assert "sequel_of" not in result[0].metadata

    def test_sequel_opportunity_is_kept_and_annotated(self, store):
        store.save_topic(_record("Pourquoi l'IA générative explose en 2026", produced_date="2026-01-10"))
        filt = TopicHistoryFilter(store=store, duplicate_threshold=0.9, sequel_threshold=0.1)
        opportunities = [_opportunity("L'IA générative va-t-elle remplacer les artistes ?", source_video_id="seq")]

        result = filt.filter_opportunities(opportunities, "IA", today=date(2026, 1, 12))

        assert len(result) == 1
        assert result[0].metadata["sequel_of"]["title"] == "Pourquoi l'IA générative explose en 2026"

    def test_never_leaves_niche_with_zero_opportunities(self, store):
        """Si TOUTES les candidates ressemblent à un sujet récent, on garde la
        moins similaire (traitée en suite) plutôt que de vider la niche."""
        store.save_topic(_record("5 métiers développeur transformés par l'IA en 2027", produced_date="2026-01-10"))
        filt = TopicHistoryFilter(store=store)
        opportunities = [_opportunity("5 métiers développeur transformés par l'IA en 2027", source_video_id="only")]

        result = filt.filter_opportunities(opportunities, "IA", today=date(2026, 1, 12))

        assert len(result) == 1
        assert result[0].source_video_id == "only"
        assert "sequel_of" in result[0].metadata

    def test_empty_opportunities_returns_empty(self, store):
        filt = TopicHistoryFilter(store=store)
        assert filt.filter_opportunities([], "IA", today=date(2026, 1, 12)) == []
