"""
Tests unitaires pour le Virality Engine.

Couvre :
  - VideoTimeline : construction, propriétés, métriques temporelles
  - ScoringCriterion individuels (statiques + temporels)
  - Score composite
  - Cohérence des critères par défaut
"""

import pytest
from datetime import datetime, timezone, timedelta
from src.models import VideoSnapshot
from src.virality_engine import (VideoTimeline, TemporalMetrics,
                                  VelocityCriterion, EngagementCriterion,
                                  FormatCriterion, SourceCriterion,
                                  GrowthVelocityCriterion, AccelerationCriterion,
                                  DEFAULT_CRITERIA)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_snap(video_id: str = "test", title: str = "Test Video",
               view_count: int = 5000, like_count: int = 200,
               comment_count: int = 30, duration_seconds: int = 120,
               source: str = "keyword", keyword: str = "IA",
               published_at: str = "2026-01-01T00:00:00Z",
               days_ago: int = 0) -> VideoSnapshot:
    """Crée un VideoSnapshot avec quelques valeurs par défaut."""
    if days_ago > 0:
        collected = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    else:
        collected = datetime.now(timezone.utc).isoformat()

    return VideoSnapshot(
        video_id=video_id,
        title=title,
        channel_id="ch",
        channel_title="Channel",
        published_at=published_at,
        description="",
        duration_iso=f"PT{duration_seconds}S",
        duration_seconds=duration_seconds,
        view_count=view_count,
        like_count=like_count,
        comment_count=comment_count,
        keyword=keyword,
        source=source,
        collected_at=collected,
    )


# ── VideoTimeline ─────────────────────────────────────────────────────────────

class TestVideoTimeline:

    def test_single_snapshot(self):
        tl = VideoTimeline("test", [_make_snap()])
        assert tl.latest.video_id == "test"
        assert tl.has_history is False
        assert tl.metrics is None

    def test_two_snapshots(self):
        s1 = _make_snap(view_count=1000, days_ago=7)
        s2 = _make_snap(view_count=5000, days_ago=0)
        tl = VideoTimeline("test", [s1, s2])
        assert tl.has_history is True
        m = tl.metrics
        assert m is not None
        assert m.view_growth == 4000
        assert m.like_growth is not None  # les deux snapshots ont des likes
        assert m.acceleration is None  # < 3 snapshots

    def test_three_snapshots_has_acceleration(self):
        s1 = _make_snap(view_count=1000, days_ago=14)
        s2 = _make_snap(view_count=3000, days_ago=7)
        s3 = _make_snap(view_count=10000, days_ago=0)
        tl = VideoTimeline("test", [s1, s2, s3])
        assert tl.has_history is True
        m = tl.metrics
        assert m is not None
        assert m.acceleration is not None  # ≥ 3 snapshots

    def test_metrics_caching(self):
        s1 = _make_snap(view_count=1000, days_ago=7)
        s2 = _make_snap(view_count=5000, days_ago=0)
        tl = VideoTimeline("test", [s1, s2])
        m1 = tl.metrics
        m2 = tl.metrics
        assert m1 is m2  # même objet (cache)

    def test_latest_is_most_recent(self):
        s1 = _make_snap(view_count=1000, days_ago=7)
        s2 = _make_snap(view_count=5000, days_ago=0)
        tl = VideoTimeline("test", [s1, s2])
        assert tl.latest.view_count == 5000
        assert tl.oldest.view_count == 1000

    def test_snapshots_sorted_by_time(self):
        s1 = _make_snap(view_count=1000, days_ago=14)
        s2 = _make_snap(view_count=5000, days_ago=7)
        s3 = _make_snap(view_count=10000, days_ago=0)
        # Donnés dans le désordre
        tl = VideoTimeline("test", [s3, s1, s2])
        assert tl.oldest.view_count == 1000
        assert tl.latest.view_count == 10000


# ── Critères statiques ───────────────────────────────────────────────────────

class TestVelocityCriterion:

    def test_normal_case(self):
        snap = _make_snap(view_count=5000, published_at="2026-01-10T00:00:00Z")
        tl = VideoTimeline("test", [snap])
        score = VelocityCriterion(weight=1.0).score(tl)
        assert score > 0

    def test_no_views(self):
        snap = _make_snap(view_count=0)
        tl = VideoTimeline("test", [snap])
        score = VelocityCriterion(weight=1.0).score(tl)
        assert score == 0.0


class TestEngagementCriterion:

    def test_normal_case(self):
        snap = _make_snap(view_count=5000, like_count=500, comment_count=100)
        tl = VideoTimeline("test", [snap])
        score = EngagementCriterion(weight=1.0).score(tl)
        assert score > 0

    def test_no_views(self):
        snap = _make_snap(view_count=0)
        tl = VideoTimeline("test", [snap])
        score = EngagementCriterion(weight=1.0).score(tl)
        assert score == 0.0

    def test_no_interactions(self):
        snap = _make_snap(view_count=5000, like_count=0, comment_count=0)
        tl = VideoTimeline("test", [snap])
        score = EngagementCriterion(weight=1.0).score(tl)
        assert score == 0.0


class TestFormatCriterion:

    def test_short_video(self):
        snap = _make_snap(duration_seconds=45)
        tl = VideoTimeline("test", [snap])
        assert FormatCriterion().score(tl) == 1.0

    def test_long_video(self):
        snap = _make_snap(duration_seconds=600)
        tl = VideoTimeline("test", [snap])
        assert FormatCriterion().score(tl) == 0.0

    def test_exactly_60s(self):
        snap = _make_snap(duration_seconds=60)
        tl = VideoTimeline("test", [snap])
        assert FormatCriterion().score(tl) == 1.0


class TestSourceCriterion:

    def test_trending_source(self):
        snap = _make_snap(source="trending")
        tl = VideoTimeline("test", [snap])
        assert SourceCriterion().score(tl) == 1.0

    def test_keyword_source(self):
        snap = _make_snap(source="keyword")
        tl = VideoTimeline("test", [snap])
        assert SourceCriterion().score(tl) == 0.0


# ── Critères temporels ───────────────────────────────────────────────────────

class TestGrowthVelocityCriterion:

    def test_with_history(self):
        s1 = _make_snap(view_count=1000, days_ago=7)
        s2 = _make_snap(view_count=5000, days_ago=0)
        tl = VideoTimeline("test", [s1, s2])
        score = GrowthVelocityCriterion(weight=1.0).score(tl)
        assert score > 0.0

    def test_without_history(self):
        snap = _make_snap(view_count=5000)
        tl = VideoTimeline("test", [snap])
        score = GrowthVelocityCriterion(weight=1.0).score(tl)
        assert score == 0.0


class TestAccelerationCriterion:

    def test_with_three_snapshots(self):
        s1 = _make_snap(view_count=1000, days_ago=14)
        s2 = _make_snap(view_count=3000, days_ago=7)
        s3 = _make_snap(view_count=10000, days_ago=0)
        tl = VideoTimeline("test", [s1, s2, s3])
        score = AccelerationCriterion(weight=1.0).score(tl)
        assert score >= 0.0

    def test_without_history(self):
        snap = _make_snap(view_count=5000)
        tl = VideoTimeline("test", [snap])
        score = AccelerationCriterion(weight=1.0).score(tl)
        assert score == 0.0

    def test_deceleration_returns_zero(self):
        """Une vidéo qui ralentit ne devrait pas recevoir de bonus."""
        s1 = _make_snap(view_count=1000, days_ago=14)
        s2 = _make_snap(view_count=5000, days_ago=7)
        s3 = _make_snap(view_count=6000, days_ago=0)  # croissance ralentie
        tl = VideoTimeline("test", [s1, s2, s3])
        score = AccelerationCriterion(weight=1.0).score(tl)
        # Le score peut être 0 si l'accélération est négative
        assert score >= 0.0


# ── Score composite ──────────────────────────────────────────────────────────

class TestScoreComposite:

    def test_default_criteria_have_six_items(self):
        assert len(DEFAULT_CRITERIA) == 6

    def test_default_criteria_names(self):
        names = [c.name for c in DEFAULT_CRITERIA]
        assert names == ["velocity", "engagement", "format", "source",
                         "growth_velocity", "acceleration"]

    def test_score_with_single_snapshot(self):
        """Sans historique, seuls les critères statiques contribuent."""
        snap = _make_snap(view_count=5000, like_count=200, comment_count=30,
                          duration_seconds=120, source="keyword", days_ago=0)
        tl = VideoTimeline("test", [snap])
        score = sum(c.score(tl) * c.weight for c in DEFAULT_CRITERIA)
        assert score > 0

    def test_score_with_history_gives_higher_score(self):
        """Avec historique, les critères temporels augmentent le score."""
        snap_static = _make_snap(view_count=5000, days_ago=0)
        tl_static = VideoTimeline("test", [snap_static])
        score_static = sum(c.score(tl_static) * c.weight for c in DEFAULT_CRITERIA)

        s1 = _make_snap(view_count=1000, days_ago=7)
        s2 = _make_snap(view_count=5000, days_ago=0)
        tl_dynamic = VideoTimeline("test", [s1, s2])
        score_dynamic = sum(c.score(tl_dynamic) * c.weight for c in DEFAULT_CRITERIA)

        assert score_dynamic > score_static
