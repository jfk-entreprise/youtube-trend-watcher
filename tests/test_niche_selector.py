import pytest

from src.niche_intelligence import Niche
from src.niche_selector import (
    ActiveNicheRecord,
    FallbackNicheSelectionStore,
    JsonNicheSelectionStore,
    NicheSelectionStore,
    NicheSelector,
)


def _niche(name: str, score: float) -> Niche:
    return Niche(
        name=name, volume=10, avg_views=10_000.0, avg_engagement=1.5,
        avg_growth_speed=100.0, niche_score=score, timelines=[],
    )


@pytest.fixture
def store(tmp_path):
    return JsonNicheSelectionStore(tmp_path / "active_niches.json")


class TestFirstRun:
    def test_no_prior_state_picks_top_candidates(self, store):
        selector = NicheSelector(store=store, max_niches=2)
        candidates = [_niche("ia", 5.0), _niche("histoire", 4.0), _niche("cuisine", 2.0)]

        selected = selector.select_daily_niches(candidates, today=__import__("datetime").date(2026, 1, 1))

        assert [n.name for n in selected] == ["ia", "histoire"]

    def test_persists_state_after_first_run(self, store):
        selector = NicheSelector(store=store, max_niches=2)
        candidates = [_niche("ia", 5.0), _niche("histoire", 4.0)]
        selector.select_daily_niches(candidates)

        records = store.load_active()
        assert {r.niche_name for r in records} == {"ia", "histoire"}


class TestConservation:
    def test_keeps_active_niche_when_no_significantly_better_candidate(self, store):
        store.save_active([
            ActiveNicheRecord("ia", niche_score=5.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
            ActiveNicheRecord("histoire", niche_score=4.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
        ])
        selector = NicheSelector(store=store, max_niches=2, replacement_threshold=0.15)
        # 'cuisine' is a new contender but not significantly better than either active niche.
        candidates = [_niche("ia", 4.8), _niche("histoire", 3.9), _niche("cuisine", 4.2)]

        selected = selector.select_daily_niches(candidates)

        assert {n.name for n in selected} == {"ia", "histoire"}

    def test_first_selected_date_preserved_across_runs(self, store):
        store.save_active([
            ActiveNicheRecord("ia", niche_score=5.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
        ])
        selector = NicheSelector(store=store, max_niches=1, replacement_threshold=0.15)
        candidates = [_niche("ia", 5.2)]

        selector.select_daily_niches(candidates, today=__import__("datetime").date(2026, 1, 5))

        records = {r.niche_name: r for r in store.load_active()}
        assert records["ia"].first_selected_date == "2026-01-01"
        assert records["ia"].last_confirmed_date == "2026-01-05"


class TestReplacement:
    def test_replaces_active_niche_when_new_candidate_is_significantly_better(self, store):
        store.save_active([
            ActiveNicheRecord("ia", niche_score=5.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
            ActiveNicheRecord("histoire", niche_score=4.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
        ])
        selector = NicheSelector(store=store, max_niches=2, replacement_threshold=0.15)
        # 'crypto' beats 'histoire' (4.0) by more than 15%.
        candidates = [_niche("ia", 5.0), _niche("histoire", 4.0), _niche("crypto", 5.5)]

        selected = selector.select_daily_niches(candidates)

        assert {n.name for n in selected} == {"ia", "crypto"}

    def test_dropped_niche_is_removed_from_persisted_state(self, store):
        store.save_active([
            ActiveNicheRecord("ia", niche_score=5.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
            ActiveNicheRecord("histoire", niche_score=4.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
        ])
        selector = NicheSelector(store=store, max_niches=2, replacement_threshold=0.15)
        candidates = [_niche("ia", 5.0), _niche("histoire", 4.0), _niche("crypto", 5.5)]

        selector.select_daily_niches(candidates)

        names = {r.niche_name for r in store.load_active()}
        assert "histoire" not in names
        assert "crypto" in names

    def test_active_niche_absent_from_candidates_is_dropped(self, store):
        store.save_active([
            ActiveNicheRecord("ia", niche_score=5.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
        ])
        selector = NicheSelector(store=store, max_niches=2, replacement_threshold=0.15)
        candidates = [_niche("histoire", 4.0), _niche("cuisine", 3.0)]

        selected = selector.select_daily_niches(candidates)

        assert {n.name for n in selected} == {"histoire", "cuisine"}


class _BrokenStore(NicheSelectionStore):
    """Simule une table Supabase absente (ex. sql/create_active_niches.sql non exécuté)."""

    def load_active(self):
        raise RuntimeError("relation 'active_niches' does not exist")

    def save_active(self, records):
        raise RuntimeError("relation 'active_niches' does not exist")


class TestFallbackNicheSelectionStore:
    def test_falls_back_to_secondary_on_load_failure(self, store):
        store.save_active([
            ActiveNicheRecord("ia", niche_score=5.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
        ])
        fallback_store = FallbackNicheSelectionStore(primary=_BrokenStore(), fallback=store)

        records = fallback_store.load_active()

        assert [r.niche_name for r in records] == ["ia"]

    def test_falls_back_to_secondary_on_save_failure(self, store):
        fallback_store = FallbackNicheSelectionStore(primary=_BrokenStore(), fallback=store)
        records = [ActiveNicheRecord("ia", niche_score=5.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01")]

        fallback_store.save_active(records)

        assert [r.niche_name for r in store.load_active()] == ["ia"]

    def test_selector_works_end_to_end_with_broken_primary(self, store):
        selector = NicheSelector(store=FallbackNicheSelectionStore(primary=_BrokenStore(), fallback=store), max_niches=2)
        candidates = [_niche("ia", 5.0), _niche("histoire", 4.0)]

        selected = selector.select_daily_niches(candidates)

        assert {n.name for n in selected} == {"ia", "histoire"}


class TestEdgeCases:
    def test_raises_on_empty_candidates(self, store):
        selector = NicheSelector(store=store)
        with pytest.raises(RuntimeError):
            selector.select_daily_niches([])

    def test_fills_remaining_slot_when_fewer_active_than_max(self, store):
        store.save_active([
            ActiveNicheRecord("ia", niche_score=5.0, first_selected_date="2026-01-01", last_confirmed_date="2026-01-01"),
        ])
        selector = NicheSelector(store=store, max_niches=2, replacement_threshold=0.15)
        candidates = [_niche("ia", 5.0), _niche("histoire", 4.0), _niche("cuisine", 3.0)]

        selected = selector.select_daily_niches(candidates)

        assert {n.name for n in selected} == {"ia", "histoire"}
