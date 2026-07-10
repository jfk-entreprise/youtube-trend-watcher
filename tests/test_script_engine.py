"""
Tests unitaires pour le Script Engine v1.

Couvre :
  - ScriptScene     : création, frozen, égalité
  - Script          : création, champs, métadonnées
  - HeuristicScriptGenerator : génération complète, tous les angles
  - ScriptEngine    : orchestration, generate_single, generate_all
  - Découplage      : le moteur n'importe pas VideoSnapshot / KnowledgeEngine

Contraintes de conception vérifiées :
  - ScriptGenerator (ABC) interchangeable
  - ScriptEngine ne dépend pas des moteurs internes
  - Chaque scène prépare les futurs Visual/Animation/Video Engine
"""

import pytest
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ── Imports du Script Engine uniquement ──────────────────────────────────────
from src.script_engine import (ScriptScene, Script, ScriptGenerator,
                                HeuristicScriptGenerator, ScriptEngine)

# ── Imports des contrats (uniquement ce que le moteur est censé connaître) ───
from src.opportunity_engine import Opportunity
from src.creative_engine import CreativeBrief
from src.brand_engine import BrandProfile


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_opportunity() -> Opportunity:
    return Opportunity(
        title="Test Video - Découvrez l'IA générative",
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
        recommendation="Produire rapidement — tendance active.",
        rationale=[
            "Potentiel viral élevé",
            "Sujet pérenne",
            "Tendance active",
        ],
        metadata={
            "content_type": "Analyse",
            "language": "fr",
            "target_audience": "Professionnels du numérique",
            "emotion": "Curiosité",
        },
    )


@pytest.fixture
def sample_brief(sample_opportunity) -> CreativeBrief:
    return CreativeBrief(
        opportunity_id=sample_opportunity.source_video_id,
        title="IA générative : le guide complet",
        angle="Liste",
        hook="L'IA générative change tout — voici pourquoi.",
        promise="Un tour d'horizon complet des applications IA en 2026.",
        audience="Professionnels du numérique",
        emotion="Curiosité",
        format="Analyse",
        duration_seconds=720,
        structure=[
            "Hook accrocheur",
            "Introduction",
            "Point #1",
            "Point #2",
            "Point #3",
            "Point bonus",
            "Conclusion",
            "CTA",
        ],
        visual_style="Graphiques et données visuelles, slides épurés, voix off posée",
        cta="Abonne-toi pour plus d'analyses sur l'IA.",
        originality_score=0.85,
        production_notes=["Préparer un script écrit", "Citer les sources"],
        rationale=["Format liste optimal pour ce sujet"],
        metadata={"niche": "IA", "language": "fr"},
    )


@pytest.fixture
def sample_brand() -> BrandProfile:
    return BrandProfile(
        id="test_brand",
        name="Test Brand",
        description="Chaîne de test",
        niche="Technologie",
        target_audience="Professionnels du numérique",
        primary_language="fr",
        tone="Professionnel",
        personality="Expert pédagogique",
        writing_style="Clair et structuré",
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
        forbidden_words=["cliqué", "viral"],
        visual_style="Minimaliste, tons bleus, data-centric",
        color_palette=["#1a237e", "#0d47a1", "#42a5f5"],
        typography_style="Sans-serif, titres en gras",
        logo_description="Logo abstrait bleu",
        thumbnail_style="Texte en haut, image en fond",
        metadata={"fréquence": "hebdomadaire"},
    )


# ── ScriptScene ──────────────────────────────────────────────────────────────

class TestScriptScene:

    def test_creation(self):
        scene = ScriptScene(
            order=1,
            title="Hook",
            narration="Accroche puissante.",
            visual_description="Plan large, lumière naturelle",
            image_prompt="Cinematic shot, dramatic lighting",
            animation_notes="Fade-in from black",
            sound_effects="Whoosh impact",
            duration_seconds=8,
        )
        assert scene.order == 1
        assert scene.title == "Hook"
        assert scene.narration == "Accroche puissante."
        assert scene.duration_seconds == 8
        assert scene.visual_description == "Plan large, lumière naturelle"

    def test_frozen(self):
        scene = ScriptScene(
            order=1, title="T", narration="N",
            visual_description="V", image_prompt="I",
            animation_notes="A", sound_effects="S",
            duration_seconds=10,
        )
        with pytest.raises(Exception):
            scene.title = "Modifié"  # type: ignore

    def test_equality(self):
        kwargs = dict(
            order=1, title="T", narration="N",
            visual_description="V", image_prompt="I",
            animation_notes="A", sound_effects="S",
            duration_seconds=10,
        )
        assert ScriptScene(**kwargs) == ScriptScene(**kwargs)

    def test_all_fields_future_proof(self):
        """Chaque scène contient les champs pour les futurs moteurs."""
        scene = ScriptScene(
            order=1, title="Hook", narration="N",
            visual_description="V", image_prompt="I",
            animation_notes="A", sound_effects="S",
            duration_seconds=8,
        )
        assert hasattr(scene, "image_prompt")
        assert hasattr(scene, "visual_description")
        assert hasattr(scene, "animation_notes")
        assert hasattr(scene, "sound_effects")

    def test_repr(self):
        scene = ScriptScene(
            order=1, title="Hook", narration="N",
            visual_description="V", image_prompt="I",
            animation_notes="A", sound_effects="S",
            duration_seconds=8,
        )
        r = repr(scene)
        assert "ScriptScene" in r
        assert "title='Hook'" in r


# ── Script ───────────────────────────────────────────────────────────────────

class TestScript:

    def test_creation(self, sample_brief, sample_brand):
        scenes = [
            ScriptScene(order=1, title="Hook", narration="Accroche.",
                       visual_description="V", image_prompt="I",
                       animation_notes="A", sound_effects="S",
                       duration_seconds=8),
            ScriptScene(order=2, title="Introduction", narration="Intro.",
                       visual_description="V", image_prompt="I",
                       animation_notes="A", sound_effects="S",
                       duration_seconds=12),
        ]
        script = Script(
            title=sample_brief.title,
            hook=sample_brief.hook,
            introduction="Introduction complète.",
            scenes=scenes,
            conclusion="Conclusion finale.",
            call_to_action=sample_brief.cta,
            estimated_duration=20,
            language=sample_brand.primary_language,
            target_audience=sample_brand.target_audience,
            style=sample_brand.tone,
            metadata={"generator": "test", "scene_count": 2},
        )
        assert script.title == "IA générative : le guide complet"
        assert script.estimated_duration == 20
        assert script.language == "fr"
        assert len(script.scenes) == 2

    def test_frozen(self, sample_brief, sample_brand):
        script = Script(
            title="Test", hook="H", introduction="I",
            scenes=[], conclusion="C", call_to_action="CTA",
            estimated_duration=0, language="fr",
            target_audience="Tous", style="Neutre",
            metadata={},
        )
        with pytest.raises(Exception):
            script.title = "Modifié"  # type: ignore

    def test_estimated_duration_sum(self, sample_brief, sample_brand):
        """estimated_duration doit correspondre à la somme des scènes."""
        scenes = [
            ScriptScene(order=1, title="Hook", narration="N",
                       visual_description="V", image_prompt="I",
                       animation_notes="A", sound_effects="S",
                       duration_seconds=8),
            ScriptScene(order=2, title="C", narration="N",
                       visual_description="V", image_prompt="I",
                       animation_notes="A", sound_effects="S",
                       duration_seconds=12),
        ]
        script = Script(
            title="Test", hook="H", introduction="I",
            scenes=scenes, conclusion="C", call_to_action="CTA",
            estimated_duration=20, language="fr",
            target_audience="Tous", style="Neutre",
            metadata={},
        )
        total = sum(s.duration_seconds for s in script.scenes)
        assert script.estimated_duration == total


# ── HeuristicScriptGenerator ─────────────────────────────────────────────────

class TestHeuristicScriptGenerator:

    def test_generate_minimal(self, sample_opportunity, sample_brief, sample_brand):
        """Génération de base : un script avec 8 scènes."""
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert isinstance(script, Script)
        assert len(script.scenes) >= 4
        assert script.scenes[0].title == "Hook"
        assert script.scenes[-1].title == "CTA"
        assert script.estimated_duration > 0
        assert script.language == "fr"
        assert script.style == "Professionnel"

    def test_generate_all_angles(self, sample_opportunity, sample_brand):
        """Teste les 5 angles disponibles."""
        angles = ["Liste", "Histoire", "Erreurs fréquentes", "Comparaison", "Challenge"]
        for angle in angles:
            brief = CreativeBrief(
                opportunity_id=sample_opportunity.source_video_id,
                title=f"Test {angle}",
                angle=angle,
                hook=f"Hook {angle}",
                promise="Promise test.",
                audience="Test",
                emotion="Curiosité",
                format="Analyse",
                duration_seconds=600,
                structure=[],
                visual_style="Standard",
                cta="Abonne-toi",
                originality_score=0.5,
                production_notes=[],
                rationale=[],
                metadata={},
            )
            gen = HeuristicScriptGenerator()
            script = gen.generate(sample_opportunity, brief, sample_brand)
            assert len(script.scenes) == 8
            assert "Hook" in [s.title for s in script.scenes]
            assert "CTA" in [s.title for s in script.scenes]

    def test_hook_injected(self, sample_opportunity, sample_brief, sample_brand):
        """Le hook du brief est injecté dans la première scène."""
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert script.hook == sample_brief.hook
        assert script.scenes[0].narration == sample_brief.hook

    def test_cta_injected(self, sample_opportunity, sample_brief, sample_brand):
        """Le CTA du brief est injecté dans la dernière scène."""
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert script.call_to_action == sample_brief.cta
        assert script.scenes[-1].narration == sample_brief.cta

    def test_scene_has_future_fields(self, sample_opportunity, sample_brief, sample_brand):
        """Chaque scène contient image_prompt, visual_description, etc."""
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        for scene in script.scenes:
            assert scene.image_prompt, f"Scène {scene.order} : image_prompt vide"
            assert scene.visual_description, f"Scène {scene.order} : visual_description vide"
            assert scene.animation_notes, f"Scène {scene.order} : animation_notes vide"
            assert scene.sound_effects, f"Scène {scene.order} : sound_effects vide"
            assert scene.duration_seconds >= 4

    def test_duration_adjusted_by_brand(self, sample_opportunity, sample_brief):
        """La durée des scènes est ajustée par le profil de marque."""
        gen = HeuristicScriptGenerator()

        # Marque rapide (vidéos courtes)
        fast_brand = BrandProfile(
            id="fast", name="Fast", description="", niche="Tech",
            target_audience="Tous", primary_language="fr",
            tone="Neutre", personality="", writing_style="",
            emotion_level=0.5, humor_level=0.2, authority_level=0.5,
            curiosity_level=0.5, storytelling_level=0.5,
            voice_speed="Rapide", preferred_video_duration=60,
            preferred_formats=["Short"], preferred_hooks=["H"],
            preferred_cta=["C"], forbidden_words=[],
            visual_style="", color_palette=[], typography_style="",
            logo_description="", thumbnail_style="", metadata={},
        )
        # Marque longue (vidéos longues)
        long_brand = BrandProfile(
            id="long", name="Long", description="", niche="Tech",
            target_audience="Tous", primary_language="fr",
            tone="Neutre", personality="", writing_style="",
            emotion_level=0.5, humor_level=0.2, authority_level=0.5,
            curiosity_level=0.5, storytelling_level=0.5,
            voice_speed="Lent", preferred_video_duration=1800,
            preferred_formats=["Long"], preferred_hooks=["H"],
            preferred_cta=["C"], forbidden_words=[],
            visual_style="", color_palette=[], typography_style="",
            logo_description="", thumbnail_style="", metadata={},
        )

        fast_script = gen.generate(sample_opportunity, sample_brief, fast_brand)
        long_script = gen.generate(sample_opportunity, sample_brief, long_brand)

        assert fast_script.estimated_duration < long_script.estimated_duration

    def test_metadata_includes_generator(self, sample_opportunity, sample_brief, sample_brand):
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert script.metadata.get("generator") == "heuristic_v1"
        assert script.metadata.get("angle") == "Liste"
        assert script.metadata.get("scene_count") == 8

    def test_script_is_frozen(self, sample_opportunity, sample_brief, sample_brand):
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        with pytest.raises(Exception):
            script.title = "Modifié"  # type: ignore

    def test_generator_name(self):
        gen = HeuristicScriptGenerator()
        assert gen.name == "heuristic_v1"


# ── ScriptEngine ─────────────────────────────────────────────────────────────

class TestScriptEngine:

    def test_default_generator(self):
        engine = ScriptEngine()
        assert engine.generator_name == "heuristic_v1"

    def test_generate_single(self, sample_opportunity, sample_brief, sample_brand):
        engine = ScriptEngine()
        script = engine.generate_single(sample_opportunity, sample_brief, sample_brand)
        assert isinstance(script, Script)
        assert len(script.scenes) == 8

    def test_generate_all(self, sample_opportunity, sample_brief, sample_brand):
        engine = ScriptEngine()
        briefs_map = {
            sample_opportunity.source_video_id: [sample_brief],
        }
        result = engine.generate_all(
            [sample_opportunity], briefs_map, sample_brand,
        )
        assert len(result) == 1
        assert sample_opportunity.source_video_id in result
        assert len(result[sample_opportunity.source_video_id]) == 1

    def test_generate_all_multiple_briefs(self, sample_opportunity, sample_brand):
        """Plusieurs briefs par opportunité → plusieurs scripts."""
        brief1 = CreativeBrief(
            opportunity_id=sample_opportunity.source_video_id,
            title="Test Liste", angle="Liste",
            hook="Hook 1", promise="P1", audience="A",
            emotion="Curiosité", format="Analyse",
            duration_seconds=600, structure=[], visual_style="S",
            cta="CTA 1", originality_score=0.5,
            production_notes=[], rationale=[], metadata={},
        )
        brief2 = CreativeBrief(
            opportunity_id=sample_opportunity.source_video_id,
            title="Test Histoire", angle="Histoire",
            hook="Hook 2", promise="P2", audience="A",
            emotion="Curiosité", format="Analyse",
            duration_seconds=600, structure=[], visual_style="S",
            cta="CTA 2", originality_score=0.6,
            production_notes=[], rationale=[], metadata={},
        )
        engine = ScriptEngine()
        result = engine.generate_all(
            [sample_opportunity],
            {sample_opportunity.source_video_id: [brief1, brief2]},
            sample_brand,
        )
        assert len(result[sample_opportunity.source_video_id]) == 2

    def test_missing_opportunity(self, sample_brand):
        """Si une vidéo dans briefs_map n'a pas d'Opportunity, elle est ignorée."""
        engine = ScriptEngine()
        result = engine.generate_all(
            [],
            {"unknown_id": []},
            sample_brand,
        )
        # L'opportunité inconnue n'est PAS ajoutée au résultat
        assert "unknown_id" not in result
        assert result == {}


# ── Découplage ──────────────────────────────────────────────────────────────

class TestDecoupling:

    def test_no_video_snapshot_import(self):
        """Le Script Engine ne doit pas importer VideoSnapshot."""
        import src.script_engine as se
        mod = se.__name__
        # Vérifie que l'import de VideoSnapshot n'est pas dans le module
        with pytest.raises(ImportError):
            from src.script_engine import VideoSnapshot  # type: ignore

    def test_no_knowledge_engine_import(self):
        """Le Script Engine ne doit pas importer KnowledgeEngine."""
        with pytest.raises(ImportError):
            from src.script_engine import KnowledgeEngine  # type: ignore

    def test_no_virality_engine_import(self):
        """Le Script Engine ne doit pas importer ViralityEngine."""
        with pytest.raises(ImportError):
            from src.script_engine import ViralityEngine  # type: ignore

    def test_no_collector_import(self):
        with pytest.raises(ImportError):
            from src.script_engine import YouTubeCollector  # type: ignore

    def test_no_storage_import(self):
        with pytest.raises(ImportError):
            from src.script_engine import CsvStorage  # type: ignore
