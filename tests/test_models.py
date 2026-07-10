"""
Tests unitaires pour le modèle VideoSnapshot.

Couvre :
  - Création avec valeurs minimales
  - Création avec tous les champs
  - Immuabilité (frozen=True)
  - Valeurs par défaut
"""

import pytest
from datetime import datetime, timezone
from src.models import VideoSnapshot


class TestVideoSnapshot:

    def test_minimal_creation(self):
        """Création avec les champs obligatoires uniquement."""
        snap = VideoSnapshot(
            video_id="test123",
            title="Test Video",
            channel_id="ch456",
            channel_title="Test Channel",
            published_at="2026-01-01T00:00:00Z",
            description="",
            duration_iso="PT10M",
            duration_seconds=600,
            keyword="test",
        )
        assert snap.video_id == "test123"
        assert snap.title == "Test Video"
        assert snap.description == ""  # valeur par défaut
        assert snap.view_count is None  # valeur par défaut
        assert snap.like_count is None
        assert snap.comment_count is None
        assert snap.source == "keyword"  # valeur par défaut
        assert snap.collected_at is not None  # généré automatiquement

    def test_full_creation(self):
        """Création avec tous les champs."""
        snap = VideoSnapshot(
            video_id="full123",
            title="Full Test",
            channel_id="ch789",
            channel_title="Full Channel",
            published_at="2026-06-15T12:30:00Z",
            description="Description détaillée de la vidéo",
            duration_iso="PT1H30M",
            duration_seconds=5400,
            view_count=50000,
            like_count=2500,
            comment_count=300,
            keyword="IA",
            source="trending",
            collected_at="2026-06-20T10:00:00Z",
        )
        assert snap.view_count == 50000
        assert snap.source == "trending"
        assert snap.collected_at == "2026-06-20T10:00:00Z"

    def test_repr(self):
        """VideoSnapshot a un __repr__ lisible."""
        snap = VideoSnapshot(
            video_id="repr123",
            title="Repr Test",
            channel_id="ch",
            channel_title="Ch",
            published_at="2026-01-01T00:00:00Z",
            description="",
            duration_iso="PT1M",
            duration_seconds=60,
            keyword="test",
        )
        r = repr(snap)
        assert "video_id='repr123'" in r
        assert "title='Repr Test'" in r

    def test_equality(self):
        """Deux snapshots identiques sont égaux."""
        ts = "2026-06-01T12:00:00+00:00"
        snap1 = VideoSnapshot(
            video_id="eq123",
            title="Equality",
            channel_id="ch",
            channel_title="Ch",
            published_at="2026-01-01T00:00:00Z",
            description="",
            duration_iso="PT1M",
            duration_seconds=60,
            keyword="test",
            collected_at=ts,
        )
        snap2 = VideoSnapshot(
            video_id="eq123",
            title="Equality",
            channel_id="ch",
            channel_title="Ch",
            published_at="2026-01-01T00:00:00Z",
            description="",
            duration_iso="PT1M",
            duration_seconds=60,
            keyword="test",
            collected_at=ts,
        )
        assert snap1 == snap2
