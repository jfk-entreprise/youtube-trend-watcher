"""
Tests unitaires pour le Knowledge Engine.

Couvre :
  - KnowledgeFact : construction et immuabilité
  - KnowledgeBase : construction, accesseurs top() et top_combinations()
  - FrequencyDiscoverer : découverte de paires et triples
  - KnowledgeEngine.build() : pipeline complet
  - JsonKnowledgeStore : sauvegarde et chargement
"""

import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from src.content_understanding import ContentProfile
from src.knowledge_engine import (KnowledgeFact, KnowledgeBase,
                                   FrequencyDiscoverer, KnowledgeEngine,
                                   JsonKnowledgeStore)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_profile(primary_topic: str = "IA",
                  content_type: str = "Tutorial",
                  target_audience: str = "Développeurs",
                  emotion: str = "Curiosité",
                  language: str = "fr",
                  evergreen_score: float = 0.5,
                  trend_score: float = 0.5,
                  duration_s: int = 600) -> ContentProfile:
    return ContentProfile(
        video_id="test",
        primary_topic=primary_topic,
        secondary_topics=[],
        language=language,
        target_audience=target_audience,
        content_type=content_type,
        emotion=emotion,
        evergreen_score=evergreen_score,
        trend_score=trend_score,
        confidence=0.8,
        metadata={"content_type": {"duration_s": duration_s}},
    )


# ── KnowledgeFact ─────────────────────────────────────────────────────────────

class TestKnowledgeFact:

    def test_creation(self):
        now = datetime.now(timezone.utc)
        fact = KnowledgeFact(
            name="topic.IA",
            description="Intelligence Artificielle",
            value={"frequency": 10, "pct": 50.0},
            confidence=0.8,
            observations=10,
            updated_at=now,
            metadata={},
        )
        assert fact.name == "topic.IA"
        assert fact.observations == 10

    def test_immutability(self):
        now = datetime.now(timezone.utc)
        fact = KnowledgeFact(
            name="test", description="Test",
            value=42, confidence=0.5,
            observations=5, updated_at=now, metadata={},
        )
        with pytest.raises(Exception):
            fact.name = "modified"  # type: ignore


# ── KnowledgeBase ─────────────────────────────────────────────────────────────

class TestKnowledgeBase:

    def test_empty_base(self):
        now = datetime.now(timezone.utc)
        kb = KnowledgeBase(
            generated_at=now,
            total_profiles=0,
            topics={},
            emotions={},
            audiences={},
            content_types={},
            languages={},
            evergreen=_make_fact("evergreen", 0),
            trend=_make_fact("trend", 0),
            durations=_make_fact("duration", 0),
            combinations=[],
        )
        assert kb.total_profiles == 0
        assert kb.top("topics", 5) == []

    def test_top_returns_sorted(self):
        now = datetime.now(timezone.utc)
        kb = KnowledgeBase(
            generated_at=now,
            total_profiles=10,
            topics={
                "IA": _make_fact("topic.IA", 5),
                "Finance": _make_fact("topic.Finance", 3),
                "Histoire": _make_fact("topic.Histoire", 8),
            },
            emotions={},
            audiences={},
            content_types={},
            languages={},
            evergreen=_make_fact("evergreen", 0),
            trend=_make_fact("trend", 0),
            durations=_make_fact("duration", 0),
            combinations=[],
        )
        top = kb.top("topics", 2)
        assert len(top) == 2
        assert top[0].observations == 8  # Histoire d'abord
        assert top[1].observations == 5  # IA ensuite

    def test_to_dict_serializable(self):
        now = datetime.now(timezone.utc)
        kb = KnowledgeBase(
            generated_at=now,
            total_profiles=10,
            topics={"IA": _make_fact("topic.IA", 5)},
            emotions={},
            audiences={},
            content_types={},
            languages={},
            evergreen=_make_fact("evergreen", 0),
            trend=_make_fact("trend", 0),
            durations=_make_fact("duration", 0),
            combinations=[],
        )
        d = kb.to_dict()
        assert d["total_profiles"] == 10
        assert "topics" in d
        assert json.dumps(d, ensure_ascii=False)  # doit être sérialisable


def _make_fact(name: str, observations: int) -> KnowledgeFact:
    return KnowledgeFact(
        name=name,
        description=name,
        value={"frequency": observations},
        confidence=0.7,
        observations=observations,
        updated_at=datetime.now(timezone.utc),
        metadata={},
    )


# ── FrequencyDiscoverer ──────────────────────────────────────────────────────

class TestFrequencyDiscoverer:

    def test_discover_single_profile(self):
        discoverer = FrequencyDiscoverer(min_freq=1)
        profiles = [_make_profile()]
        facts = discoverer.discover(profiles, datetime.now(timezone.utc))
        assert len(facts) > 0

    def test_discover_pairs_and_triples(self):
        discoverer = FrequencyDiscoverer(min_freq=1)
        profiles = [
            _make_profile("IA", "Tutorial", "Développeurs", "Curiosité"),
            _make_profile("IA", "Analyse", "Développeurs", "Curiosité"),
            _make_profile("Finance", "Tutorial", "Investisseurs", "Confiance"),
        ]
        facts = discoverer.discover(profiles, datetime.now(timezone.utc))
        assert len(facts) > 0
        # Vérifie qu'on a des paires ET des triples
        scopes = {f.metadata.get("scope") for f in facts}
        assert "pair" in scopes
        assert "triple" in scopes

    def test_min_freq_filters(self):
        discoverer = FrequencyDiscoverer(min_freq=3)
        profiles = [
            _make_profile("IA"),
            _make_profile("Finance"),
        ]
        facts = discoverer.discover(profiles, datetime.now(timezone.utc))
        # Rien ne dépasse min_freq=3
        assert len(facts) == 0


# ── KnowledgeEngine ──────────────────────────────────────────────────────────

class TestKnowledgeEngine:

    def test_build_with_profiles(self):
        engine = KnowledgeEngine()
        profiles = [
            _make_profile("IA", "Tutorial", "Développeurs", "Curiosité"),
            _make_profile("IA", "Tutorial", "Développeurs", "Curiosité"),
            _make_profile("Finance", "Analyse", "Investisseurs", "Confiance"),
        ]
        kb = engine.build(profiles)
        assert kb.total_profiles == 3
        assert len(kb.topics) == 2  # IA, Finance
        assert len(kb.combinations) > 0

    def test_build_with_empty_raises(self):
        engine = KnowledgeEngine()
        with pytest.raises(ValueError):
            engine.build([])


# ── JsonKnowledgeStore ───────────────────────────────────────────────────────

class TestJsonKnowledgeStore:

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonKnowledgeStore(Path(tmpdir))
            profiles = [_make_profile("IA")]
            kb = KnowledgeEngine().build(profiles)
            
            # Sauvegarde
            store.save(kb)
            files = list(Path(tmpdir).glob("knowledge_*.json"))
            assert len(files) == 1
            
            # Chargement
            loaded = store.load()
            assert loaded is not None
            assert loaded.total_profiles == 1
            assert "IA" in loaded.topics

    def test_load_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonKnowledgeStore(Path(tmpdir))
            loaded = store.load()
            assert loaded is None

    def test_load_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonKnowledgeStore(Path(tmpdir))
            profiles = [_make_profile("IA")]
            kb = KnowledgeEngine().build(profiles)
            store.save(kb)
            
            history = store.load_history(days=30)
            assert len(history) >= 1
