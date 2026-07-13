"""
Tests unitaires pour le Script Engine v1 (storyboard cinématographique, Sprint 32.1).

Couvre :
  - Dialogue        : création, frozen
  - SceneDescription / Scene : création, frozen
  - ScriptScene     : création, frozen, égalité, narration_text, order (alias)
  - estimate_scene_duration : calcul de durée centralisé
  - Script          : création, champs, métadonnées, propriétés dérivées
    (hook/introduction/conclusion/call_to_action)
  - HeuristicScriptGenerator : génération complète, tous les angles
  - ScriptEngine    : orchestration, generate_single, generate_all
  - Découplage      : le moteur n'importe pas VideoSnapshot / KnowledgeEngine
"""

import pytest

# ── Imports du Script Engine uniquement ──────────────────────────────────────
from src.script_engine import (
    Dialogue, Scene, SceneDescription, ScriptScene, Script, ScriptGenerator,
    HeuristicScriptGenerator, ScriptEngine, estimate_scene_duration,
    NARRATION_WORDS_PER_MINUTE,
)

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


def _description(**overrides) -> SceneDescription:
    base = dict(
        setting="A dimly lit futuristic office at night.",
        composition="Rule of thirds, subject centered-left.",
        characters="A calm off-screen narrator.",
        lighting="Cool blue rim lighting, hard shadows.",
        camera="Slow 6-second dolly-in, slight low angle.",
        mood="Tense curiosity.",
        symbolism="The darkness represents the unknown risk.",
        director_notes="This scene must hook attention in under 3 seconds.",
        viewer_emotion="Curiosity slowly turning into unease.",
    )
    base.update(overrides)
    return SceneDescription(**base)


def _scene(number, replique, personnage="NARRATEUR", scene_type="hook", transition="Cut.", duration_seconds=10):
    return ScriptScene(
        scene=Scene(number=number, type=scene_type, description=_description()),
        dialogues=[Dialogue(personnage=personnage, replique=replique)],
        transition=transition, duration_seconds=duration_seconds,
    )


# ── Dialogue ──────────────────────────────────────────────────────────────────

class TestDialogue:
    def test_creation(self):
        d = Dialogue(personnage="NARRATEUR", replique="Accroche puissante.")
        assert d.personnage == "NARRATEUR"
        assert d.replique == "Accroche puissante."

    def test_frozen(self):
        d = Dialogue(personnage="NARRATEUR", replique="N")
        with pytest.raises(Exception):
            d.replique = "Modifié"  # type: ignore


# ── SceneDescription / Scene ─────────────────────────────────────────────────

class TestSceneDescription:
    def test_creation_all_nine_fields(self):
        desc = _description()
        assert desc.setting
        assert desc.composition
        assert desc.characters
        assert desc.lighting
        assert desc.camera
        assert desc.mood
        assert desc.symbolism
        assert desc.director_notes
        assert desc.viewer_emotion

    def test_frozen(self):
        desc = _description()
        with pytest.raises(Exception):
            desc.setting = "Modifié"  # type: ignore


class TestScene:
    def test_creation(self):
        desc = _description()
        scene = Scene(number=1, type="hook", description=desc)
        assert scene.number == 1
        assert scene.type == "hook"
        assert scene.description is desc

    def test_frozen(self):
        scene = Scene(number=1, type="hook", description=_description())
        with pytest.raises(Exception):
            scene.number = 2  # type: ignore


# ── ScriptScene ──────────────────────────────────────────────────────────────

class TestScriptScene:

    def test_creation(self):
        scene = ScriptScene(
            scene=Scene(number=1, type="hook", description=_description()),
            dialogues=[Dialogue(personnage="NARRATEUR", replique="Accroche puissante.")],
            transition="Fondu entrant.",
            duration_seconds=8,
        )
        assert scene.order == 1
        assert scene.scene.number == 1
        assert scene.scene.type == "hook"
        assert scene.dialogues[0].replique == "Accroche puissante."
        assert scene.duration_seconds == 8
        assert scene.transition == "Fondu entrant."

    def test_order_is_alias_for_scene_number(self):
        scene = _scene(7, "N")
        assert scene.order == 7
        assert scene.order == scene.scene.number

    def test_frozen(self):
        scene = _scene(1, "N")
        with pytest.raises(Exception):
            scene.scene = Scene(number=2, type="hook", description=_description())  # type: ignore

    def test_equality(self):
        desc = _description()
        kwargs = dict(
            scene=Scene(number=1, type="hook", description=desc),
            dialogues=[Dialogue(personnage="NARRATEUR", replique="N")],
            transition="T", duration_seconds=10,
        )
        assert ScriptScene(**kwargs) == ScriptScene(**kwargs)

    def test_narration_text_joins_dialogues(self):
        scene = ScriptScene(
            scene=Scene(number=1, type="hook", description=_description()),
            dialogues=[
                Dialogue(personnage="Roi", replique="Bonjour."),
                Dialogue(personnage="Conseiller", replique="Sire."),
            ],
            transition="T", duration_seconds=10,
        )
        assert scene.narration_text == "Bonjour. Sire."

    def test_narration_text_single_narrator(self):
        scene = _scene(1, "Une narration classique.")
        assert scene.narration_text == "Une narration classique."

    def test_repr(self):
        scene = _scene(1, "N")
        r = repr(scene)
        assert "ScriptScene" in r


# ── estimate_scene_duration ──────────────────────────────────────────────────

class TestEstimateSceneDuration:
    def test_computes_from_word_count(self):
        # 150 words/minute (default) = 2.5 words/second
        dialogues = [Dialogue(personnage="NARRATEUR", replique=" ".join(["mot"] * 25))]
        duration = estimate_scene_duration(dialogues)
        assert duration == round(25 / NARRATION_WORDS_PER_MINUTE * 60)

    def test_empty_dialogues_returns_minimum(self):
        assert estimate_scene_duration([Dialogue(personnage="NARRATEUR", replique="")]) == 2

    def test_never_below_minimum(self):
        dialogues = [Dialogue(personnage="NARRATEUR", replique="Un mot.")]
        assert estimate_scene_duration(dialogues) >= 2

    def test_custom_words_per_minute(self):
        dialogues = [Dialogue(personnage="NARRATEUR", replique=" ".join(["mot"] * 30))]
        fast = estimate_scene_duration(dialogues, words_per_minute=300.0)
        slow = estimate_scene_duration(dialogues, words_per_minute=100.0)
        assert fast < slow

    def test_multi_dialogue_scene_sums_all_repliques(self):
        dialogues = [
            Dialogue(personnage="A", replique=" ".join(["mot"] * 10)),
            Dialogue(personnage="B", replique=" ".join(["mot"] * 10)),
        ]
        combined = estimate_scene_duration(dialogues)
        single = estimate_scene_duration([Dialogue(personnage="A", replique=" ".join(["mot"] * 10))])
        assert combined > single


# ── Script ───────────────────────────────────────────────────────────────────

class TestScript:

    def test_creation(self, sample_brief, sample_brand):
        scenes = [
            _scene(1, "Accroche.", duration_seconds=8),
            _scene(2, "Intro.", duration_seconds=12),
        ]
        script = Script(
            title=sample_brief.title,
            scenes=scenes,
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

    def test_frozen(self):
        script = Script(
            title="Test", scenes=[], estimated_duration=0, language="fr",
            target_audience="Tous", style="Neutre", metadata={},
        )
        with pytest.raises(Exception):
            script.title = "Modifié"  # type: ignore

    def test_estimated_duration_sum(self):
        scenes = [
            _scene(1, "N", duration_seconds=8),
            _scene(2, "N", duration_seconds=12),
        ]
        script = Script(
            title="Test", scenes=scenes, estimated_duration=20, language="fr",
            target_audience="Tous", style="Neutre", metadata={},
        )
        total = sum(s.duration_seconds for s in script.scenes)
        assert script.estimated_duration == total

    def test_hook_is_first_scene_narration(self):
        scenes = [_scene(1, "Accroche."), _scene(2, "Suite."), _scene(3, "CTA final.")]
        script = Script(
            title="T", scenes=scenes, estimated_duration=30, language="fr",
            target_audience="Tous", style="Neutre", metadata={},
        )
        assert script.hook == "Accroche."

    def test_call_to_action_is_last_scene_narration(self):
        scenes = [_scene(1, "Accroche."), _scene(2, "Suite."), _scene(3, "CTA final.")]
        script = Script(
            title="T", scenes=scenes, estimated_duration=30, language="fr",
            target_audience="Tous", style="Neutre", metadata={},
        )
        assert script.call_to_action == "CTA final."

    def test_introduction_and_conclusion_derived(self):
        scenes = [_scene(1, "A"), _scene(2, "B"), _scene(3, "C"), _scene(4, "D")]
        script = Script(
            title="T", scenes=scenes, estimated_duration=40, language="fr",
            target_audience="Tous", style="Neutre", metadata={},
        )
        assert script.introduction == "B"
        assert script.conclusion == "C"

    def test_hook_and_cta_empty_for_no_scenes(self):
        script = Script(
            title="T", scenes=[], estimated_duration=0, language="fr",
            target_audience="Tous", style="Neutre", metadata={},
        )
        assert script.hook == ""
        assert script.call_to_action == ""

    def test_derived_properties_not_in_asdict(self):
        """hook/introduction/conclusion/call_to_action sont des propriétés
        calculées — jamais sérialisées par dataclasses.asdict() (Sprint 31.1/32.1)."""
        import dataclasses
        scenes = [_scene(1, "A"), _scene(2, "B")]
        script = Script(
            title="T", scenes=scenes, estimated_duration=20, language="fr",
            target_audience="Tous", style="Neutre", metadata={},
        )
        keys = set(dataclasses.asdict(script).keys())
        assert "hook" not in keys
        assert "introduction" not in keys
        assert "conclusion" not in keys
        assert "call_to_action" not in keys
        assert keys == {"title", "scenes", "estimated_duration", "language", "target_audience", "style", "metadata"}

    def test_scene_asdict_has_nested_storyboard_shape(self):
        """Sprint 32.1 : dataclasses.asdict() sur une scène doit refléter
        exactement le contrat storyboard imbriqué."""
        import dataclasses
        scene = _scene(1, "A")
        data = dataclasses.asdict(scene)
        assert set(data.keys()) == {"scene", "dialogues", "transition", "duration_seconds"}
        assert set(data["scene"].keys()) == {"number", "type", "description"}
        assert set(data["scene"]["description"].keys()) == {
            "setting", "composition", "characters", "lighting", "camera",
            "mood", "symbolism", "director_notes", "viewer_emotion",
        }


# ── HeuristicScriptGenerator ─────────────────────────────────────────────────

class TestHeuristicScriptGenerator:

    def test_generate_minimal(self, sample_opportunity, sample_brief, sample_brand):
        """Génération de base : un script avec 8 scènes."""
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert isinstance(script, Script)
        assert len(script.scenes) >= 4
        assert script.hook == sample_brief.hook
        assert script.call_to_action == sample_brief.cta
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
            assert script.hook == f"Hook {angle}"
            assert script.call_to_action == "Abonne-toi"

    def test_hook_injected(self, sample_opportunity, sample_brief, sample_brand):
        """Le hook du brief est injecté dans la première scène."""
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert script.hook == sample_brief.hook
        assert script.scenes[0].narration_text == sample_brief.hook

    def test_cta_injected(self, sample_opportunity, sample_brief, sample_brand):
        """Le CTA du brief est injecté dans la dernière scène."""
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert script.call_to_action == sample_brief.cta
        assert script.scenes[-1].narration_text == sample_brief.cta

    def test_scene_has_storyboard_fields(self, sample_opportunity, sample_brief, sample_brand):
        """Chaque scène contient un Scene/SceneDescription complet + dialogues/transition."""
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        for scene in script.scenes:
            assert scene.scene.type, f"Scène {scene.order} : type vide"
            desc = scene.scene.description
            for field_name in (
                "setting", "composition", "characters", "lighting", "camera",
                "mood", "symbolism", "director_notes", "viewer_emotion",
            ):
                assert getattr(desc, field_name), f"Scène {scene.order} : {field_name} vide"
            assert scene.dialogues, f"Scène {scene.order} : dialogues vides"
            assert scene.narration_text, f"Scène {scene.order} : narration_text vide"
            assert scene.transition, f"Scène {scene.order} : transition vide"
            assert scene.duration_seconds >= 2

    def test_first_scene_type_is_hook_last_is_cta(self, sample_opportunity, sample_brief, sample_brand):
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert script.scenes[0].scene.type == "hook"
        assert script.scenes[-1].scene.type == "cta"

    def test_duration_derived_from_estimate_scene_duration(self, sample_opportunity, sample_brief, sample_brand):
        """Sprint 32.1 : la durée de chaque scène vient de estimate_scene_duration(),
        pondérée par le facteur de marque — jamais une valeur fixe arbitraire."""
        gen = HeuristicScriptGenerator()
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        for scene in script.scenes:
            baseline = estimate_scene_duration(scene.dialogues)
            # Le facteur de marque (~0.6-1.5) explique l'écart avec la baseline brute.
            assert scene.duration_seconds >= max(2, round(baseline * 0.6) - 1)

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
