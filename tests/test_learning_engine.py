"""
Tests unitaires pour le Learning Engine v1 (Sprint 15).

Couvre :
  - PerformanceMetrics : création, propriétés calculées, score composite
  - LearningSignal     : création, frozen
  - LearningProfile    : construction, cache, API (best_hook, best_angle, etc.)
  - LearningEngine     : record(), build()
  - JsonLearningStore  : save, load, list, delete
  - Découplage         : le moteur n'importe pas les moteurs internes
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.learning_engine import (PerformanceMetrics, LearningSignal,
                                  LearningProfile, LearningEngine,
                                  LearningStore, JsonLearningStore)
from src.opportunity_engine import Opportunity
from src.creative_engine import CreativeBrief
from src.brand_engine import BrandProfile
from src.script_engine import Script, ScriptScene


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_opportunity() -> Opportunity:
    return Opportunity(
        title="Test Video",
        niche="Intelligence Artificielle",
        source_video_id="test123",
        overall_score=0.75,
        virality_score=0.65,
        growth_score=0.55,
        evergreen_score=0.70,
        trend_score=0.80,
        competition_score=0.30,
        production_difficulty=0.40,
        urgency=0.60,
        recommendation="Produire rapidement.",
        rationale=["Potentiel viral élevé"],
        metadata={
            "content_type": "Analyse",
            "language": "fr",
            "target_audience": "Professionnels",
            "emotion": "Curiosité",
        },
    )


@pytest.fixture
def sample_brief(sample_opportunity) -> CreativeBrief:
    return CreativeBrief(
        opportunity_id=sample_opportunity.source_video_id,
        title="IA : le guide complet",
        angle="Liste",
        hook="L'IA change tout.",
        promise="Un tour d'horizon complet.",
        audience="Professionnels",
        emotion="Curiosité",
        format="Analyse",
        duration_seconds=720,
        structure=["Hook", "Intro", "Point #1", "Point #2", "Point #3", "Conclusion", "CTA"],
        visual_style="Data-centric",
        cta="Abonne-toi pour plus d'IA.",
        originality_score=0.85,
        production_notes=["Préparer un script"],
        rationale=["Format liste optimal"],
        metadata={"niche": "IA", "language": "fr"},
    )


@pytest.fixture
def sample_brand() -> BrandProfile:
    return BrandProfile(
        id="test_brand",
        name="Test Brand",
        description="Chaîne de test",
        niche="Technologie",
        target_audience="Professionnels",
        primary_language="fr",
        tone="Professionnel",
        personality="Expert",
        writing_style="Clair",
        emotion_level=0.4,
        humor_level=0.1,
        authority_level=0.8,
        curiosity_level=0.6,
        storytelling_level=0.5,
        voice_speed="Modéré",
        preferred_video_duration=600,
        preferred_formats=["Analyse", "Liste"],
        preferred_hooks=["L'IA change tout"],
        preferred_cta=["Abonne-toi"],
        forbidden_words=[],
        visual_style="Minimaliste",
        color_palette=["#1a237e"],
        typography_style="Sans-serif",
        logo_description="Logo abstrait",
        thumbnail_style="Texte en haut",
        metadata={"fréquence": "hebdomadaire"},
    )


@pytest.fixture
def sample_script(sample_brief, sample_brand) -> Script:
    return Script(
        title=sample_brief.title,
        hook=sample_brief.hook,
        introduction="Introduction complète.",
        scenes=[
            ScriptScene(order=1, title="Hook", narration="L'IA change tout.",
                       visual_description="V", image_prompt="I",
                       animation_notes="A", sound_effects="S",
                       duration_seconds=8),
            ScriptScene(order=2, title="Intro", narration="Intro.",
                       visual_description="V", image_prompt="I",
                       animation_notes="A", sound_effects="S",
                       duration_seconds=12),
            ScriptScene(order=3, title="CTA", narration="Abonne-toi.",
                       visual_description="V", image_prompt="I",
                       animation_notes="A", sound_effects="S",
                       duration_seconds=10),
        ],
        conclusion="Conclusion.",
        call_to_action=sample_brief.cta,
        estimated_duration=30,
        language=sample_brand.primary_language,
        target_audience=sample_brand.target_audience,
        style=sample_brand.tone,
        metadata={"generator": "test"},
    )


# ── PerformanceMetrics ────────────────────────────────────────────────────────

class TestPerformanceMetrics:

    def test_creation_minimal(self):
        metrics = PerformanceMetrics(video_id="v1")
        assert metrics.video_id == "v1"
        assert metrics.views == 0
        assert metrics.likes == 0

    def test_creation_full(self):
        metrics = PerformanceMetrics(
            video_id="v1", views=10000, likes=500, comments=120,
            retention=0.55, watch_time=850.0, impressions_ctr=0.12,
            shares=80, subscribers_gained=150,
        )
        assert metrics.views == 10000
        assert metrics.retention == 0.55
        assert metrics.shares == 80

    def test_engagement_rate(self):
        m1 = PerformanceMetrics(video_id="v1", views=1000, likes=50, comments=10)
        assert m1.engagement_rate == 0.06  # (50+10)/1000

        m2 = PerformanceMetrics(video_id="v2", views=0, likes=0, comments=0)
        assert m2.engagement_rate == 0.0   # évite division par zéro

    def test_views_per_share(self):
        m = PerformanceMetrics(video_id="v1", views=1000, shares=50)
        assert m.views_per_share == 20.0

        m_zero = PerformanceMetrics(video_id="v2", views=1000, shares=0)
        assert m_zero.views_per_share == 1000.0  # division par max(shares, 1)

    def test_performance_score_range(self):
        """Le score composite doit être dans [0.0, 1.0]."""
        m = PerformanceMetrics(video_id="v1", views=100000, likes=5000,
                               comments=500, retention=0.60, watch_time=5000.0,
                               impressions_ctr=0.10, shares=200, subscribers_gained=300)
        assert 0.0 <= m.performance_score <= 1.0

    def test_performance_score_zero(self):
        m = PerformanceMetrics(video_id="v1")
        assert m.performance_score == 0.0

    def test_performance_score_high(self):
        m = PerformanceMetrics(video_id="v1", views=1_000_000, likes=100000,
                               comments=20000, retention=0.70, watch_time=500000.0,
                               impressions_ctr=0.20, shares=5000, subscribers_gained=10000)
        assert 0.5 <= m.performance_score <= 1.0

    def test_frozen(self):
        m = PerformanceMetrics(video_id="v1")
        with pytest.raises(Exception):
            m.views = 999  # type: ignore

    def test_equality(self):
        kwargs = dict(video_id="v1", views=100, likes=10)
        assert PerformanceMetrics(**kwargs) == PerformanceMetrics(**kwargs)


# ── LearningSignal ───────────────────────────────────────────────────────────

class TestLearningSignal:

    def test_creation_minimal(self):
        sig = LearningSignal(
            dimension="hook",
            value="L'IA change tout.",
            performance_score=0.85,
        )
        assert sig.dimension == "hook"
        assert sig.performance_score == 0.85
        assert sig.brand_id == ""
        assert sig.niche == ""

    def test_creation_full(self):
        sig = LearningSignal(
            dimension="angle",
            value="Liste",
            performance_score=0.72,
            views=10000, likes=500, comments=50,
            retention=0.55,
            brand_id="test_brand",
            niche="IA",
            opportunity_id="opp123",
            metadata={"extra": "data"},
        )
        assert sig.views == 10000
        assert sig.brand_id == "test_brand"
        assert sig.metadata["extra"] == "data"

    def test_frozen(self):
        sig = LearningSignal(dimension="d", value="v", performance_score=0.5)
        with pytest.raises(Exception):
            sig.performance_score = 0.9  # type: ignore


# ── LearningProfile ───────────────────────────────────────────────────────────

class TestLearningProfile:

    def test_empty_profile(self):
        profile = LearningProfile(brand_id="test", signals=[])
        assert profile.total_signals == 0
        assert profile.dimensions == []

        # Les requêtes doivent retourner des valeurs par défaut
        assert profile.best_hook() == ("", 0.0)
        assert profile.best_angle() == ("", 0.0)
        assert profile.top_hooks(3) == []

    def test_single_signal(self):
        sig = LearningSignal(
            dimension="hook", value="Mon hook", performance_score=0.75,
            brand_id="test", niche="IA",
        )
        profile = LearningProfile(brand_id="test", signals=[sig])
        assert profile.total_signals == 1
        assert profile.dimensions == ["hook"]

        hook, score = profile.best_hook()
        assert hook == "Mon hook"
        assert score == 0.75

    def test_multiple_signals(self):
        signals = [
            LearningSignal(dimension="hook", value="Hook A", performance_score=0.8, brand_id="b1"),
            LearningSignal(dimension="hook", value="Hook B", performance_score=0.6, brand_id="b1"),
            LearningSignal(dimension="hook", value="Hook A", performance_score=0.9, brand_id="b1"),
            LearningSignal(dimension="angle", value="Liste", performance_score=0.7, brand_id="b1"),
            LearningSignal(dimension="angle", value="Histoire", performance_score=0.5, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)

        # Meilleur hook : Hook A avec moyenne (0.8+0.9)/2 = 0.85
        hook, score = profile.best_hook()
        assert hook == "Hook A"
        assert score == 0.85

        # Meilleur angle : Liste (0.7)
        angle, ascore = profile.best_angle()
        assert angle == "Liste"
        assert ascore == 0.7

    def test_top_hooks(self):
        signals = [
            LearningSignal(dimension="hook", value="H1", performance_score=0.9, brand_id="b1"),
            LearningSignal(dimension="hook", value="H2", performance_score=0.8, brand_id="b1"),
            LearningSignal(dimension="hook", value="H3", performance_score=0.7, brand_id="b1"),
            LearningSignal(dimension="hook", value="H4", performance_score=0.6, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)
        top = profile.top_hooks(2)
        assert len(top) == 2
        assert top[0][0] == "H1"
        assert top[1][0] == "H2"

    def test_best_duration(self):
        signals = [
            LearningSignal(dimension="duration_seconds", value="120",
                          performance_score=0.8, brand_id="b1"),
            LearningSignal(dimension="duration_seconds", value="300",
                          performance_score=0.6, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)
        dur, score = profile.best_duration()
        assert dur == "120"
        assert score == 0.8

    def test_best_cta(self):
        signals = [
            LearningSignal(dimension="cta", value="Abonne-toi",
                          performance_score=0.9, brand_id="b1"),
            LearningSignal(dimension="cta", value="Like",
                          performance_score=0.7, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)
        cta, score = profile.best_cta()
        assert cta == "Abonne-toi"
        assert score == 0.9

    def test_best_style(self):
        signals = [
            LearningSignal(dimension="style", value="Professionnel",
                          performance_score=0.85, brand_id="b1"),
            LearningSignal(dimension="style", value="Décontracté",
                          performance_score=0.65, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)
        style, score = profile.best_style()
        assert style == "Professionnel"
        assert score == 0.85

    def test_best_emotion(self):
        signals = [
            LearningSignal(dimension="emotion", value="Curiosité",
                          performance_score=0.78, brand_id="b1"),
            LearningSignal(dimension="emotion", value="Urgence",
                          performance_score=0.72, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)
        emotion, score = profile.best_emotion()
        assert emotion == "Curiosité"
        assert score == 0.78

    def test_best_format(self):
        signals = [
            LearningSignal(dimension="format", value="Short",
                          performance_score=0.88, brand_id="b1"),
            LearningSignal(dimension="format", value="Long",
                          performance_score=0.62, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)
        fmt, score = profile.best_format()
        assert fmt == "Short"
        assert score == 0.88

    def test_filtre_par_niche(self):
        signals = [
            LearningSignal(dimension="hook", value="H IA",
                          performance_score=0.9, brand_id="b1", niche="IA"),
            LearningSignal(dimension="hook", value="H Gaming",
                          performance_score=0.8, brand_id="b1", niche="Gaming"),
            LearningSignal(dimension="hook", value="H IA 2",
                          performance_score=0.7, brand_id="b1", niche="IA"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)

        # Sans filtre
        hook, _ = profile.best_hook()
        assert hook == "H IA"

        # Filtré par niche Gaming
        hook, _ = profile.best_hook(niche="Gaming")
        assert hook == "H Gaming"

    def test_filtre_par_brand(self):
        signals = [
            LearningSignal(dimension="angle", value="Liste",
                          performance_score=0.9, brand_id="b1"),
            LearningSignal(dimension="angle", value="Histoire",
                          performance_score=0.8, brand_id="b2"),
            LearningSignal(dimension="angle", value="Liste",
                          performance_score=0.7, brand_id="b2"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)

        # Filtré par brand b1
        angle, score = profile.best_angle(brand_id="b1")
        assert angle == "Liste"
        assert score == 0.9

        # Filtré par brand b2
        angle, score = profile.best_angle(brand_id="b2")
        assert angle == "Histoire"
        assert score == 0.8

    def test_summary(self):
        signals = [
            LearningSignal(dimension="hook", value="H1", performance_score=0.8, brand_id="b1"),
            LearningSignal(dimension="angle", value="Liste", performance_score=0.7, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)
        summary = profile.summary()
        assert summary["brand_id"] == "b1"
        assert summary["total_signals"] == 2
        assert "best_hook" in summary
        assert "best_angle" in summary

    def test_dimensions_property(self):
        signals = [
            LearningSignal(dimension="hook", value="H", performance_score=0.5, brand_id="b1"),
            LearningSignal(dimension="angle", value="A", performance_score=0.5, brand_id="b1"),
            LearningSignal(dimension="cta", value="C", performance_score=0.5, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)
        assert sorted(profile.dimensions) == ["angle", "cta", "hook"]


# ── LearningEngine ────────────────────────────────────────────────────────────

class TestLearningEngine:

    def test_record(self, sample_opportunity, sample_brief, sample_brand, sample_script):
        """record() doit produire 7 signaux (hook, angle, duration, cta, style, emotion, format)."""
        engine = LearningEngine()
        metrics = PerformanceMetrics(
            video_id="test123", views=15000, likes=750,
            comments=120, retention=0.55, watch_time=850.0,
            impressions_ctr=0.12, shares=80, subscribers_gained=200,
        )
        signals = engine.record(sample_opportunity, sample_brief, sample_brand, sample_script, metrics)
        assert len(signals) == 7

        dimensions = {s.dimension for s in signals}
        assert "hook" in dimensions
        assert "angle" in dimensions
        assert "duration_seconds" in dimensions
        assert "cta" in dimensions
        assert "style" in dimensions
        assert "emotion" in dimensions
        assert "format" in dimensions

        # Tous les signaux doivent avoir le même performance_score
        scores = {s.performance_score for s in signals}
        assert len(scores) == 1
        assert 0.0 < list(scores)[0] <= 1.0

    def test_record_scores_consistent(self, sample_opportunity, sample_brief, sample_brand, sample_script):
        """Deux enregistrements identiques → signaux identiques."""
        engine = LearningEngine()
        metrics = PerformanceMetrics(video_id="v1", views=10000, likes=500,
                                      comments=50, retention=0.50)
        signals1 = engine.record(sample_opportunity, sample_brief, sample_brand, sample_script, metrics)
        signals2 = engine.record(sample_opportunity, sample_brief, sample_brand, sample_script, metrics)
        assert len(signals1) == len(signals2)
        for s1, s2 in zip(signals1, signals2):
            assert s1.dimension == s2.dimension
            assert s1.value == s2.value
            assert s1.performance_score == s2.performance_score

    def test_build(self):
        engine = LearningEngine()
        signals = [
            LearningSignal(dimension="hook", value="H1", performance_score=0.8, brand_id="b1"),
            LearningSignal(dimension="angle", value="Liste", performance_score=0.7, brand_id="b1"),
        ]
        profile = engine.build(signals, brand_id="b1")
        assert isinstance(profile, LearningProfile)
        assert profile.total_signals == 2
        assert profile.brand_id == "b1"

    def test_build_empty(self):
        engine = LearningEngine()
        profile = engine.build([], brand_id="empty")
        assert profile.total_signals == 0
        assert profile.dimensions == []

    def test_record_frozen_profile(self, sample_opportunity, sample_brief, sample_brand, sample_script):
        """Le profil construit depuis un record doit être queryable."""
        engine = LearningEngine()
        metrics = PerformanceMetrics(video_id="v1", views=20000, likes=1000,
                                      comments=200, retention=0.60)
        signals = engine.record(sample_opportunity, sample_brief, sample_brand, sample_script, metrics)
        profile = engine.build(signals, brand_id="test_brand")
        hook, score = profile.best_hook()
        assert hook == "L'IA change tout."
        assert score > 0.0


# ── JsonLearningStore ────────────────────────────────────────────────────────

class TestJsonLearningStore:

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_save_and_load(self, tmp_dir):
        signals = [
            LearningSignal(dimension="hook", value="H1", performance_score=0.8, brand_id="b1"),
            LearningSignal(dimension="angle", value="Liste", performance_score=0.7, brand_id="b1"),
        ]
        profile = LearningProfile(brand_id="b1", signals=signals)
        store = JsonLearningStore(tmp_dir)
        store.save(profile)

        loaded = store.load("b1")
        assert loaded is not None
        assert loaded.brand_id == "b1"
        assert loaded.total_signals == 2
        assert loaded.best_hook() == ("H1", 0.8)

    def test_load_missing(self, tmp_dir):
        store = JsonLearningStore(tmp_dir)
        loaded = store.load("unknown")
        assert loaded is None

    def test_list_brands(self, tmp_dir):
        store = JsonLearningStore(tmp_dir)
        # Créer deux profils
        for bid in ("b1", "b2"):
            p = LearningProfile(brand_id=bid, signals=[
                LearningSignal(dimension="hook", value="H", performance_score=0.5, brand_id=bid),
            ])
            store.save(p)
        brands = store.list_brands()
        assert "b1" in brands
        assert "b2" in brands
        assert len(brands) == 2

    def test_delete(self, tmp_dir):
        store = JsonLearningStore(tmp_dir)
        p = LearningProfile(brand_id="b1", signals=[])
        store.save(p)
        assert store.load("b1") is not None
        assert store.delete("b1") is True
        assert store.load("b1") is None
        assert store.delete("b1") is False  # déjà supprimé

    def test_roundtrip_preserves_data(self, tmp_dir):
        signals = [
            LearningSignal(dimension="hook", value="Mon super hook",
                          performance_score=0.92, views=50000, likes=2500,
                          comments=300, retention=0.65, brand_id="b1",
                          niche="IA", opportunity_id="opp123",
                          metadata={"extra": "value"}),
            LearningSignal(dimension="angle", value="Histoire",
                          performance_score=0.78, brand_id="b1", niche="IA"),
        ]
        original = LearningProfile(brand_id="b1", signals=signals)
        store = JsonLearningStore(tmp_dir)
        store.save(original)

        loaded = store.load("b1")
        assert loaded is not None
        assert loaded.total_signals == 2
        hook, score = loaded.best_hook()
        assert hook == "Mon super hook"
        assert score == 0.92
        assert loaded.best_angle() == ("Histoire", 0.78)


# ── Découplage ────────────────────────────────────────────────────────────────

class TestDecoupling:

    def test_no_video_snapshot_import(self):
        with pytest.raises(ImportError):
            from src.learning_engine import VideoSnapshot  # type: ignore

    def test_no_knowledge_engine_import(self):
        with pytest.raises(ImportError):
            from src.learning_engine import KnowledgeEngine  # type: ignore

    def test_no_virality_engine_import(self):
        with pytest.raises(ImportError):
            from src.learning_engine import ViralityEngine  # type: ignore

    def test_no_collector_import(self):
        with pytest.raises(ImportError):
            from src.learning_engine import YouTubeCollector  # type: ignore

    def test_no_storage_import(self):
        with pytest.raises(ImportError):
            from src.learning_engine import CsvStorage  # type: ignore

    def test_no_script_engine_import(self):
        with pytest.raises(ImportError):
            from src.learning_engine import ScriptEngine  # type: ignore

    def test_no_creative_engine_import(self):
        with pytest.raises(ImportError):
            from src.learning_engine import CreativeEngine  # type: ignore

    def test_no_brand_engine_import(self):
        with pytest.raises(ImportError):
            from src.learning_engine import BrandEngine  # type: ignore

    def test_only_contracts_imported(self):
        """Le module n'importe que les dataclasses (contrats), pas les moteurs."""
        import src.learning_engine as le
        mod_src = Path(le.__file__).read_text(encoding="utf-8")
        # Vérifie qu'il importe les contrats mais pas les moteurs
        assert "from src.opportunity_engine import Opportunity" in mod_src
        assert "from src.creative_engine import CreativeBrief" in mod_src
        assert "from src.brand_engine import BrandProfile" in mod_src
        assert "from src.script_engine import Script" in mod_src
