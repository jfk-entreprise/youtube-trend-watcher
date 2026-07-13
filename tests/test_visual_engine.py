"""
Tests unitaires pour le Visual Engine (Sprint 18, migré Sprint 31.1).

Teste :
  1. VisualScene — création, immutabilité, types attendus
  2. VisualPlan — création, structure, aspect_ratio
  3. VisualGenerator — interface ABC
  4. HeuristicVisualGenerator — règles heuristiques, mappings PAR POSITION
     (Sprint 31.1 : les scènes n'ont plus de titre nommé — première scène =
     hook, dernière = CTA, scènes intermédiaires en rotation)
  5. VisualEngine — orchestration, generate, generate_all
  6. Découplage — n'importe aucun moteur interne
  7. Cas limites — script vide, etc.
"""

import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.visual_engine import (
    VisualScene,
    VisualPlan,
    VisualGenerator,
    HeuristicVisualGenerator,
    VisualEngine,
    _SHOT_TYPES,
    _CAMERA_MOTIONS,
    _TRANSITIONS,
    _TRANSITION_MAP,
    _SECTION_SHOT_MAP,
    _SECTION_MOTION_MAP,
    _COLOR_PALETTES,
    _LIGHTING_MAP,
)
from src.script_engine import Dialogue, Scene, SceneDescription, Script, ScriptScene


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _description(setting="Description visuelle.") -> SceneDescription:
    return SceneDescription(
        setting=setting,
        composition="Rule of thirds.",
        characters="Narrator only.",
        lighting="Neutral lighting.",
        camera="Static shot.",
        mood="Neutral.",
        symbolism="None.",
        director_notes="Standard scene.",
        viewer_emotion="Attentive curiosity.",
    )


def _scene(order, replique, scene_desc="Description visuelle.", duration_seconds=10, scene_type="scene"):
    return ScriptScene(
        scene=Scene(number=order, type=scene_type, description=_description(setting=scene_desc)),
        dialogues=[Dialogue(personnage="NARRATEUR", replique=replique)],
        transition="Coupe.", duration_seconds=duration_seconds,
    )


@pytest.fixture
def script_scene_hook() -> ScriptScene:
    return _scene(
        1, "Voici pourquoi 80% des developpeurs sous-estiment l'IA.",
        scene_desc="Plan d'accroche dynamique — visuel choc ou question à l'écran.",
        duration_seconds=8,
    )


@pytest.fixture
def script_scene_intro() -> ScriptScene:
    return _scene(
        2, "Aujourd'hui, on va parler de Intelligence Artificielle.",
        scene_desc="Tête parlante face caméra OU écran titre avec musique douce en fond.",
        duration_seconds=12,
    )


@pytest.fixture
def script_scene_point1() -> ScriptScene:
    return _scene(
        3, "Premier point clé : l'IA transforme le métier de développeur.",
        scene_desc="Infographie ou liste animée. Numéro à l'écran.",
        duration_seconds=16,
    )


@pytest.fixture
def script_scene_cta() -> ScriptScene:
    return _scene(
        4, "Abonne-toi pour plus d'analyses tech.",
        scene_desc="Fond de chaîne ou miniature finale. Boutons abonnement animés.",
        duration_seconds=10,
    )


@pytest.fixture
def sample_script(script_scene_hook, script_scene_intro, script_scene_point1, script_scene_cta) -> Script:
    return Script(
        title="5 métiers de développeur transformés par l'IA",
        scenes=[script_scene_hook, script_scene_intro, script_scene_point1, script_scene_cta],
        estimated_duration=46,
        language="fr",
        target_audience="Développeurs curieux de l'IA",
        style="Innovant",
        metadata={"generator": "heuristic_v1", "angle": "Liste", "niche": "Intelligence Artificielle"},
    )


@pytest.fixture
def sample_script_default_style(script_scene_hook, script_scene_intro) -> Script:
    return Script(
        title="Vidéo sans style",
        scenes=[script_scene_hook, script_scene_intro],
        estimated_duration=20,
        language="fr",
        target_audience="Grand public",
        style="Inconnu",  # Style non standard
        metadata={},
    )


@pytest.fixture
def full_script() -> Script:
    """Script complet avec 8 scènes."""
    scenes = [_scene(i, f"Replique {i}", duration_seconds=10 + i) for i in range(1, 9)]
    return Script(
        title="Test complet",
        scenes=scenes,
        estimated_duration=sum(s.duration_seconds for s in scenes),
        language="fr",
        target_audience="Test",
        style="Innovant",
        metadata={"niche": "Tech", "angle": "Liste"},
    )


# ── Tests : VisualScene ─────────────────────────────────────────────────────

class TestVisualScene:
    def test_creation_minimal(self):
        """Création avec tous les champs requis."""
        scene = VisualScene(
            scene_order=1,
            shot_type="close_up",
            camera_motion="static",
            visual_prompt="Test prompt",
            composition="Rule of thirds",
            lighting="Soft",
            color_palette=["#FFF", "#000"],
            transition="cut",
            overlay_text="Test",
            animation_notes="None",
            duration_seconds=10,
        )
        assert scene.scene_order == 1
        assert scene.shot_type == "close_up"
        assert scene.duration_seconds == 10

    def test_creation_with_metadata(self):
        """Création avec metadata."""
        scene = VisualScene(
            scene_order=1,
            shot_type="wide",
            camera_motion="dolly_in",
            visual_prompt="Test",
            composition="Centered",
            lighting="Natural",
            color_palette=["#FFF"],
            transition="fade",
            overlay_text="Hello",
            animation_notes="Animate",
            duration_seconds=5,
            metadata={"source": "test"},
        )
        assert scene.metadata["source"] == "test"

    def test_frozen(self):
        """VisualScene est immuable (frozen dataclass)."""
        scene = VisualScene(
            scene_order=1,
            shot_type="medium",
            camera_motion="static",
            visual_prompt="P",
            composition="C",
            lighting="L",
            color_palette=["#000"],
            transition="cut",
            overlay_text="O",
            animation_notes="A",
            duration_seconds=10,
        )
        with pytest.raises(FrozenInstanceError):
            scene.shot_type = "wide"  # type: ignore
        with pytest.raises(FrozenInstanceError):
            scene.duration_seconds = 20  # type: ignore

    def test_all_shot_types_are_valid(self):
        """Les shot_type utilisés dans les mappings sont tous valides."""
        for shot in _SECTION_SHOT_MAP.values():
            assert shot in _SHOT_TYPES, f"Shot '{shot}' n'est pas dans _SHOT_TYPES"

    def test_all_camera_motions_are_valid(self):
        """Les camera_motion utilisés dans les mappings sont tous valides."""
        for motion in _SECTION_MOTION_MAP.values():
            assert motion in _CAMERA_MOTIONS, f"Motion '{motion}' n'est pas dans _CAMERA_MOTIONS"

    def test_all_lighting_keys_are_valid(self):
        """Les clés de _LIGHTING_MAP sont des shot types valides."""
        for key in _LIGHTING_MAP:
            assert key in _SHOT_TYPES, f"Clé d'éclairage '{key}' n'est pas un shot type valide"

    def test_repr(self):
        """Le repr est lisible."""
        scene = VisualScene(
            scene_order=1,
            shot_type="close_up",
            camera_motion="static",
            visual_prompt="P",
            composition="C",
            lighting="L",
            color_palette=["#000"],
            transition="cut",
            overlay_text="O",
            animation_notes="A",
            duration_seconds=10,
        )
        r = repr(scene)
        assert "VisualScene" in r
        assert "close_up" in r

    def test_equality(self):
        """Deux scenes identiques sont egales (non hashables: liste dans dataclass)."""
        kwargs = dict(
            scene_order=1, shot_type="medium", camera_motion="static",
            visual_prompt="P", composition="C", lighting="L",
            color_palette=["#000"], transition="cut", overlay_text="O",
            animation_notes="A", duration_seconds=10,
        )
        s1 = VisualScene(**kwargs)
        s2 = VisualScene(**kwargs)
        assert s1 == s2
        # Les dataclasses frozen avec List ne sont pas hashables
        with pytest.raises(TypeError):
            hash(s1)


# ── Tests : VisualPlan ──────────────────────────────────────────────────────

class TestVisualPlan:
    def test_creation_minimal(self):
        """Création avec le minimum."""
        plan = VisualPlan(title="Test", style="default")
        assert plan.title == "Test"
        assert plan.style == "default"
        assert plan.aspect_ratio == "9:16"
        assert plan.scenes == []
        assert plan.color_palette == []

    def test_creation_with_scenes(self):
        """Création avec des scènes."""
        scenes = [
            VisualScene(scene_order=1, shot_type="medium", camera_motion="static",
                        visual_prompt="P", composition="C", lighting="L",
                        color_palette=["#000"], transition="cut", overlay_text="O",
                        animation_notes="A", duration_seconds=10),
        ]
        plan = VisualPlan(
            title="Test",
            style="innovant",
            aspect_ratio="16:9",
            scenes=scenes,
            color_palette=["#0D0D0D", "#00D4FF"],
            metadata={"key": "value"},
        )
        assert len(plan.scenes) == 1
        assert plan.aspect_ratio == "16:9"
        assert plan.color_palette == ["#0D0D0D", "#00D4FF"]
        assert plan.metadata["key"] == "value"

    def test_default_aspect_ratio(self):
        """Aspect ratio par défaut = 9:16."""
        plan = VisualPlan(title="T", style="default")
        assert plan.aspect_ratio == "9:16"

    def test_scenes_list_immutable(self):
        """La liste de scenes ne peut pas être réaffectée."""
        plan = VisualPlan(title="T", style="default")
        with pytest.raises(FrozenInstanceError):
            plan.scenes = []  # type: ignore


# ── Tests : VisualGenerator (ABC) ───────────────────────────────────────────

class TestVisualGenerator:
    def test_cannot_instantiate(self):
        """VisualGenerator est une interface, pas instanciable."""
        with pytest.raises(TypeError):
            VisualGenerator()  # type: ignore

    def test_subclass_must_implement_generate(self):
        """Une sous-classe doit implémenter generate()."""
        class BadGenerator(VisualGenerator):
            @property
            def name(self):
                return "bad"

        with pytest.raises(TypeError):
            BadGenerator()

    def test_subclass_must_implement_name(self):
        """Une sous-classe doit implémenter name."""
        class BadGenerator(VisualGenerator):
            def generate(self, script):
                return None

        with pytest.raises(TypeError):
            BadGenerator()

    def test_valid_subclass(self):
        """Une sous-classe complète est instanciable."""
        class GoodGenerator(VisualGenerator):
            @property
            def name(self):
                return "good"

            def generate(self, script):
                return VisualPlan(title=script.title, style="default")

        gen = GoodGenerator()
        assert gen.name == "good"
        script = Script(
            title="T", scenes=[], estimated_duration=0, language="fr",
            target_audience="A", style="S", metadata={},
        )
        plan = gen.generate(script)
        assert isinstance(plan, VisualPlan)
        assert plan.title == "T"


# ── Tests : HeuristicVisualGenerator ────────────────────────────────────────

class TestHeuristicVisualGenerator:
    def test_name(self):
        """Le générateur a un nom identifiable."""
        gen = HeuristicVisualGenerator()
        assert gen.name == "heuristic_visual_v1"

    def test_generate_returns_visualplan(self, sample_script):
        """generate retourne un VisualPlan."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        assert isinstance(plan, VisualPlan)
        assert plan.title == sample_script.title

    def test_scene_count_matches_script(self, sample_script):
        """Autant de VisualScene que de ScriptScene."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        assert len(plan.scenes) == len(sample_script.scenes)

    def test_scene_order_preserved(self, sample_script):
        """Les ordres des scenes sont preserves."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        expected_orders = [s.order for s in sample_script.scenes]
        visual_orders = [vs.scene_order for vs in plan.scenes]
        assert visual_orders == expected_orders

    def test_shot_type_mapped_by_position(self, sample_script):
        """Le type de plan est mappé depuis la POSITION de la scène (Sprint 31.1)."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        # Première scène → Hook → medium_close_up
        assert plan.scenes[0].shot_type == "medium_close_up"
        # Dernière scène → CTA → close_up
        assert plan.scenes[-1].shot_type == "close_up"

    def test_camera_motion_mapped_by_position(self, sample_script):
        """Le mouvement caméra est mappé par position."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        assert plan.scenes[0].camera_motion == "dolly_in"  # Hook
        assert plan.scenes[-1].camera_motion == "zoom_out"  # CTA

    def test_lighting_from_shot_type(self, sample_script):
        """L'éclairage dépend du type de plan."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        # medium_close_up → Rembrandt
        assert "Rembrandt" in plan.scenes[0].lighting
        # close_up (dernière scène) → trois points
        assert len(plan.scenes[-1].lighting) > 0

    def test_transition_mapped_by_position(self, sample_script):
        """La transition est mappée par position."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        assert plan.scenes[0].transition == "fade_from_black"  # Hook
        assert plan.scenes[-1].transition == "fade_to_black"  # CTA

    def test_transition_list_valid(self):
        """Les transitions des mappings sont dans la liste valide."""
        for trans in _TRANSITION_MAP.values():
            assert trans in _TRANSITIONS, f"Transition '{trans}' n'est pas dans _TRANSITIONS"

    def test_overlay_text_rendered(self, sample_script):
        """L'overlay text est rendu avec les données du script."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        # Hook → contient le hook
        assert "développeurs" in plan.scenes[0].overlay_text or "sous-estiment" in plan.scenes[0].overlay_text
        # CTA → texte fixe "Abonne-toi !"
        assert plan.scenes[-1].overlay_text == "Abonne-toi !"

    def test_color_palette_from_style(self, sample_script):
        """La palette de couleurs vient du style."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        # Style = Innovant → palette innovant
        assert plan.color_palette == _COLOR_PALETTES["innovant"]

    def test_color_palette_fallback_unknown_style(self, sample_script_default_style):
        """Style inconnu → fallback 'default'."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script_default_style)
        assert plan.color_palette == _COLOR_PALETTES["default"]

    def test_visual_prompt_enhanced(self, sample_script):
        """Le prompt visuel est enrichi avec les métadonnées."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        prompt = plan.scenes[0].visual_prompt
        assert "Shot type:" in prompt
        assert "camera motion:" in prompt
        assert "style:" in prompt
        assert "color palette:" in prompt
        assert "duration:" in prompt

    def test_base_prompt_preserved(self, sample_script):
        """La description riche de la scène (ScriptScene.scene) est conservée dans l'enrichi."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        prompt = plan.scenes[0].visual_prompt
        assert "accroche dynamique" in prompt

    def test_animation_notes_sourced_from_transition(self, sample_script):
        """Sprint 31.1 : animation_notes de la VisualScene vient de ScriptScene.transition
        (il n'y a plus de champ animation_notes dédié sur ScriptScene)."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        assert plan.scenes[0].animation_notes == sample_script.scenes[0].transition

    def test_duration_preserved(self, sample_script):
        """La durée est copiée depuis ScriptScene."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        for i, vs in enumerate(plan.scenes):
            assert vs.duration_seconds == sample_script.scenes[i].duration_seconds

    def test_composition_mapped(self, sample_script):
        """La composition est mappée."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        assert len(plan.scenes[0].composition) > 0
        assert "typographie" in plan.scenes[0].composition.lower() or "centrage" in plan.scenes[0].composition.lower()

    def test_metadata_in_each_scene(self, sample_script):
        """Chaque VisualScene a des métadonnées de résolution."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        for vs in plan.scenes:
            assert "script_scene_order" in vs.metadata
            assert "position_key_resolved" in vs.metadata
            assert vs.metadata["script_scene_order"] == vs.scene_order

    def test_global_metadata(self, sample_script):
        """Le VisualPlan a des métadonnées globales."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        assert plan.metadata["generator"] == "heuristic_visual_v1"
        assert plan.metadata["scene_count"] == len(sample_script.scenes)
        assert plan.metadata["total_duration_seconds"] > 0

    def test_aspect_ratio_default_9_16(self, sample_script):
        """L'aspect ratio par défaut est 9:16."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        assert plan.aspect_ratio == "9:16"

    def test_full_script_all_positions_mapped(self, full_script):
        """Script complet avec 8 scènes → toutes les positions sont mappées sans erreur."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(full_script)
        assert len(plan.scenes) == 8
        assert plan.scenes[0].shot_type == "medium_close_up"  # Hook
        assert plan.scenes[-1].shot_type == "close_up"  # CTA
        for vs in plan.scenes:
            assert vs.shot_type in _SHOT_TYPES

    def test_scene_color_palette_inherits_global(self, sample_script):
        """Chaque scène a sa propre palette (copie de la globale)."""
        gen = HeuristicVisualGenerator()
        plan = gen.generate(sample_script)
        for vs in plan.scenes:
            assert vs.color_palette == plan.color_palette

    def test_all_color_palettes_have_5_colors(self):
        """Toutes les palettes ont exactement 5 couleurs."""
        for name, palette in _COLOR_PALETTES.items():
            assert len(palette) == 5, f"Palette '{name}' a {len(palette)} couleurs (attendu: 5)"


# ── Tests : HeuristicVisualGenerator._resolve_position_key (Sprint 31.1) ────

class TestResolvePositionKey:
    def test_first_scene_is_hook(self):
        assert HeuristicVisualGenerator._resolve_position_key(1, 5) == "Hook"

    def test_last_scene_is_cta(self):
        assert HeuristicVisualGenerator._resolve_position_key(5, 5) == "CTA"

    def test_single_scene_is_hook(self):
        """Une seule scène : la position 1 == total → priorité au Hook."""
        assert HeuristicVisualGenerator._resolve_position_key(1, 1) == "Hook"

    def test_middle_scenes_rotate(self):
        """Les scènes intermédiaires tournent sur un jeu de clés génériques."""
        keys = [HeuristicVisualGenerator._resolve_position_key(o, 8) for o in range(2, 8)]
        assert keys[0] != "Hook" and keys[0] != "CTA"
        # Deux appels avec le même order/total donnent toujours la même clé (déterministe)
        assert HeuristicVisualGenerator._resolve_position_key(3, 8) == HeuristicVisualGenerator._resolve_position_key(3, 8)


# ── Tests : HeuristicVisualGenerator._render_overlay ────────────────────────

class TestRenderOverlay:
    def test_simple_template(self, sample_script):
        """Template simple."""
        result = HeuristicVisualGenerator._render_overlay("Titre: {title}", sample_script.scenes[0], sample_script)
        assert "métiers" in result

    def test_hook_template(self, sample_script):
        """Template avec hook."""
        result = HeuristicVisualGenerator._render_overlay("{hook_text}", sample_script.scenes[0], sample_script)
        assert len(result) > 0
        assert "développeurs" in result or "sous-estiment" in result or "IA" in result

    def test_topic_from_metadata(self, sample_script):
        """Topic vient des métadonnées si disponible."""
        result = HeuristicVisualGenerator._render_overlay("{topic}", sample_script.scenes[0], sample_script)
        assert "Intelligence Artificielle" in result

    def test_scene_title_uses_scene_description(self, sample_script):
        """{scene_title} est désormais dérivé du début de la description riche de la scène."""
        result = HeuristicVisualGenerator._render_overlay("{scene_title}", sample_script.scenes[0], sample_script)
        assert "accroche" in result.lower()


# ── Tests : HeuristicVisualGenerator._build_visual_prompt ───────────────────

class TestBuildVisualPrompt:
    def test_prompt_contains_all_elements(self, sample_script):
        """Le prompt enrichi contient tous les éléments."""
        scene = sample_script.scenes[0]
        prompt = HeuristicVisualGenerator._build_visual_prompt(
            scene, "close_up", "dolly_in", "Innovant",
            ["#0D0D0D", "#00D4FF", "#7B2FBE", "#FFFFFF", "#1A1A2E"],
        )
        assert "Shot type:" in prompt
        assert "camera motion:" in prompt
        assert "style:" in prompt
        assert "color palette:" in prompt
        assert "duration:" in prompt

    def test_prompt_preserves_base(self, sample_script):
        """La description de scène d'origine est conservée."""
        scene = sample_script.scenes[0]
        prompt = HeuristicVisualGenerator._build_visual_prompt(
            scene, "medium", "static", "default", ["#FFF", "#000"],
        )
        assert "accroche dynamique" in prompt


# ── Tests : VisualEngine ────────────────────────────────────────────────────

class TestVisualEngine:
    def test_default_generator(self):
        """VisualEngine utilise HeuristicVisualGenerator par défaut."""
        engine = VisualEngine()
        assert engine.generator_name == "heuristic_visual_v1"

    def test_custom_generator(self):
        """VisualEngine accepte un générateur personnalisé."""
        class CustomGen(VisualGenerator):
            @property
            def name(self):
                return "custom"

            def generate(self, script):
                return VisualPlan(title=script.title, style="custom")

        engine = VisualEngine(generator=CustomGen())
        assert engine.generator_name == "custom"

    def test_generate_single(self, sample_script):
        """generate() retourne un VisualPlan."""
        engine = VisualEngine()
        plan = engine.generate(sample_script)
        assert isinstance(plan, VisualPlan)
        assert len(plan.scenes) == 4

    def test_generate_empty_script_raises(self):
        """Script sans scènes → ValueError."""
        script = Script(
            title="T", scenes=[], estimated_duration=0,
            language="fr", target_audience="A", style="S", metadata={},
        )
        engine = VisualEngine()
        with pytest.raises(ValueError, match="aucune scène"):
            engine.generate(script)

    def test_generate_all(self, sample_script):
        """generate_all() retourne une liste de VisualPlans."""
        engine = VisualEngine()
        plans = engine.generate_all([sample_script, sample_script])
        assert len(plans) == 2
        assert all(isinstance(p, VisualPlan) for p in plans)

    def test_generate_all_empty(self):
        """generate_all([]) retourne une liste vide."""
        engine = VisualEngine()
        plans = engine.generate_all([])
        assert plans == []

    def test_generate_all_with_error(self, sample_script):
        """generate_all ignore les scripts qui échouent."""
        empty_script = Script(
            title="Vide", scenes=[], estimated_duration=0,
            language="fr", target_audience="A", style="S", metadata={},
        )
        engine = VisualEngine()
        plans = engine.generate_all([sample_script, empty_script, sample_script])
        assert len(plans) == 2  # le script vide est ignoré

    def test_generate_all_preserves_order(self):
        """L'ordre des plans correspond à l'ordre des scripts."""
        script_a = Script(
            title="A", scenes=[_scene(1, "N", duration_seconds=5)],
            estimated_duration=5, language="fr", target_audience="A", style="S", metadata={},
        )
        script_b = Script(
            title="B", scenes=[_scene(1, "N", duration_seconds=5)],
            estimated_duration=5, language="fr", target_audience="A", style="S", metadata={},
        )
        engine = VisualEngine()
        plans = engine.generate_all([script_b, script_a])
        assert plans[0].title == "B"
        assert plans[1].title == "A"

    def test_engine_does_not_modify_script(self, sample_script):
        """Le moteur ne modifie pas le Script d'entrée."""
        engine = VisualEngine()
        original_duration = sample_script.estimated_duration
        engine.generate(sample_script)
        assert sample_script.estimated_duration == original_duration


# ── Tests : Découplage ──────────────────────────────────────────────────────

class TestDecoupling:
    def test_import_does_not_import_internal_engines(self):
        """Le module source n'importe pas les autres moteurs (verification du source)."""
        mod_src = Path(__file__).resolve().parent.parent / "src" / "visual_engine.py"
        content = mod_src.read_text(encoding="utf-8")
        # Le module ne doit PAS importer ces modules directement
        for forbidden_token in [
            "opportunity_engine", "brand_engine", "creative_engine",
            "knowledge_engine", "virality_engine", "content_understanding",
            "niche_intelligence", "collector", "storage", "agents",
            "learning_engine", "llm", "animation_engine", "video_engine",
        ]:
            assert f"import {forbidden_token}" not in content, \
                f"Import interdit: {forbidden_token}"
            assert f"from src.{forbidden_token}" not in content, \
                f"Import interdit: src.{forbidden_token}"

    def test_only_script_imported(self):
        """Le module importe seulement Script et ScriptScene."""
        mod_src = Path(__file__).resolve().parent.parent / "src" / "visual_engine.py"
        content = mod_src.read_text(encoding="utf-8")
        # Vérifie les imports de contrats
        assert "from src.script_engine import Script, ScriptScene" in content
        # Vérifie qu'il n'importe PAS les autres modules
        assert "opportunity_engine" not in content
        assert "brand_engine" not in content
        assert "creative_engine" not in content
        assert "knowledge_engine" not in content
        assert "virality_engine" not in content
        assert "collector" not in content
        assert "storage" not in content

    def test_no_video_snapshot_import(self):
        """Aucune dépendance vers VideoSnapshot."""
        mod_src = Path(__file__).resolve().parent.parent / "src" / "visual_engine.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "VideoSnapshot" not in content

    def test_generate_signature(self):
        """generate() ne prend QUE Script."""
        import inspect
        from src.visual_engine import VisualGenerator
        sig = inspect.signature(VisualGenerator.generate)
        params = list(sig.parameters.keys())
        assert "script" in params
        # Vérifie qu'il n'y a pas d'autres paramètres métier
        for extra in ["opportunity", "brief", "brand", "profile"]:
            assert extra not in params, f"Paramètre interdit: {extra}"
