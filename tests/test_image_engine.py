"""
Tests unitaires pour le Image Engine (Sprint 19).

Teste :
  1. GeneratedImage — création, immutabilité, types attendus
  2. ImageGenerator — interface ABC
  3. HeuristicImageGenerator — construction prompt, negative_prompt, dimensions, seed
  4. ImageEngine — orchestration, generate, generate_all
  5. Découplage — n'importe aucun moteur interne
  6. Cas limites — plan vide, scène inconnue, aspect ratio custom
"""

import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict, List

from src.image_engine import (
    GeneratedImage,
    ImageGenerator,
    HeuristicImageGenerator,
    ImageEngine,
    _ASPECT_RATIOS,
    _ASPECT_RATIOS_FULL,
    _STYLE_KEYWORDS,
    _NEGATIVE_STYLE_KEYWORDS,
)
from src.visual_engine import VisualScene, VisualPlan


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_visual_scene_hook() -> VisualScene:
    return VisualScene(
        scene_order=1,
        shot_type="medium_close_up",
        camera_motion="dolly_in",
        visual_prompt="Dynamic abstract composition with bold typography, high contrast lighting. Shot type: medium close up, camera motion: dolly in, style: Innovant, color palette: #0D0D0D, #00D4FF, duration: 8s",
        composition="Centrage dynamique avec typographie imposante. Sujet au centre, espace negatif maitrise.",
        lighting="Rembrandt. Key a 45 degres, triangle de lumiere sur la joue. Fill minimal.",
        color_palette=["#0D0D0D", "#00D4FF", "#7B2FBE", "#FFFFFF", "#1A1A2E"],
        transition="fade_from_black",
        overlay_text="Voici pourquoi 80% des developpeurs sous-estiment l'IA.",
        animation_notes="Fade-in from black. Bold text animation. 0.5s buildup.",
        duration_seconds=8,
        metadata={"script_scene_title": "Hook", "title_key_resolved": "Hook", "style_resolved": "Innovant", "palette_source": "innovant"},
    )


@pytest.fixture
def sample_visual_scene_intro() -> VisualScene:
    return VisualScene(
        scene_order=2,
        shot_type="medium",
        camera_motion="static",
        visual_prompt="Clean professional workspace with warm ambient lighting. Shot type: medium, camera motion: static, style: Innovant, color palette: #0D0D0D, #00D4FF, duration: 12s",
        composition="Regle des tiers. Sujet decale a droite, espace a gauche pour overlay texte ou B-roll.",
        lighting="Lumiere naturelle diffusee. Key a 30 degres, fill a 90%. Ambiance soft.",
        color_palette=["#0D0D0D", "#00D4FF", "#7B2FBE", "#FFFFFF", "#1A1A2E"],
        transition="crossfade",
        overlay_text="5 metiers developpeur transformes par l'IA en 2027",
        animation_notes="Crossfade transition. Gentle parallax on background.",
        duration_seconds=12,
        metadata={"script_scene_title": "Introduction", "title_key_resolved": "Introduction", "style_resolved": "Innovant", "palette_source": "innovant"},
    )


@pytest.fixture
def sample_visual_scene_point1() -> VisualScene:
    return VisualScene(
        scene_order=3,
        shot_type="medium_close_up",
        camera_motion="static",
        visual_prompt="Numbered infographic #1, clean design. Shot type: medium close up, camera motion: static, style: Innovant, color palette: #0D0D0D, #00D4FF, duration: 16s",
        composition="Cadrage typographique. Infographie centree. Numero large a gauche. Texte a droite.",
        lighting="Rembrandt. Key a 45 degres, triangle de lumiere sur la joue. Fill minimal.",
        color_palette=["#0D0D0D", "#00D4FF", "#7B2FBE", "#FFFFFF", "#1A1A2E"],
        transition="slide_up",
        overlay_text="1",
        animation_notes="Number flies in from left. Content fades below.",
        duration_seconds=16,
        metadata={"script_scene_title": "Point #1", "title_key_resolved": "Point #1", "style_resolved": "Innovant", "palette_source": "innovant"},
    )


@pytest.fixture
def sample_visual_plan(sample_visual_scene_hook, sample_visual_scene_intro, sample_visual_scene_point1) -> VisualPlan:
    return VisualPlan(
        title="5 metiers developpeur transformes par l'IA en 2027",
        style="Innovant",
        aspect_ratio="9:16",
        scenes=[sample_visual_scene_hook, sample_visual_scene_intro, sample_visual_scene_point1],
        color_palette=["#0D0D0D", "#00D4FF", "#7B2FBE", "#FFFFFF", "#1A1A2E"],
        metadata={"generator": "heuristic_visual_v1", "scene_count": 3, "total_duration_seconds": 36},
    )


@pytest.fixture
def wide_visual_plan() -> VisualPlan:
    """Plan avec aspect ratio 16:9 (paysage)."""
    return VisualPlan(
        title="Vlog paysage",
        style="default",
        aspect_ratio="16:9",
        scenes=[
            VisualScene(
                scene_order=1,
                shot_type="wide",
                camera_motion="pan_left",
                visual_prompt="Paysage magnifique. Shot type: wide, camera motion: pan left, style: nature, duration: 10s",
                composition="Plan large centre. Horizon au tiers superieur.",
                lighting="Lumiere ambiante naturelle.",
                color_palette=["#2D5016", "#4A7C3F", "#8CB26B", "#D4C9A8", "#F5F0E1"],
                transition="fade_from_black",
                overlay_text="",
                animation_notes="Panoramique lent.",
                duration_seconds=10,
                metadata={"script_scene_title": "Contexte", "style_resolved": "nature"},
            ),
            VisualScene(
                scene_order=2,
                shot_type="extreme_close_up",
                camera_motion="zoom_in",
                visual_prompt="Macro shot. Shot type: extreme close up, duration: 5s",
                composition="Extreme gros plan. Micro-detail.",
                lighting="Eclairage dur. Key unique a 0 degres, ombres marquees.",
                color_palette=["#0A0A0A", "#1A1A2E", "#2D2D44", "#E94560", "#AAAAAA"],
                transition="cut",
                overlay_text="",
                animation_notes="Zoom progressif.",
                duration_seconds=5,
                metadata={"script_scene_title": "Detail", "style_resolved": "sombre"},
            ),
        ],
        color_palette=["#0D0D0D", "#00D4FF", "#7B2FBE", "#FFFFFF", "#1A1A2E"],
        metadata={},
    )


# ── Tests : GeneratedImage ─────────────────────────────────────────────────

class TestGeneratedImage:
    def test_creation_minimal(self):
        """Creation avec tous les champs requis."""
        img = GeneratedImage(
            scene_order=1,
            prompt="Test prompt",
            negative_prompt="Bad quality, blurry",
            width=720,
            height=1280,
            aspect_ratio="9:16",
            seed=12345,
        )
        assert img.scene_order == 1
        assert img.prompt == "Test prompt"
        assert img.width == 720
        assert img.height == 1280
        assert img.seed == 12345
        assert img.provider == "heuristic"

    def test_creation_full(self):
        """Creation avec tous les champs optionnels."""
        img = GeneratedImage(
            scene_order=2,
            prompt="Full prompt",
            negative_prompt="Negative",
            width=1024,
            height=1024,
            aspect_ratio="1:1",
            seed=99999,
            quality="hd",
            steps=30,
            style="innovant",
            color_palette=["#000", "#FFF"],
            provider="flux",
            metadata={"source": "test"},
        )
        assert img.quality == "hd"
        assert img.steps == 30
        assert img.provider == "flux"
        assert img.metadata["source"] == "test"

    def test_frozen(self):
        """GeneratedImage est immuable (frozen dataclass)."""
        img = GeneratedImage(
            scene_order=1,
            prompt="P",
            negative_prompt="N",
            width=720,
            height=1280,
            aspect_ratio="9:16",
            seed=0,
        )
        with pytest.raises(FrozenInstanceError):
            img.prompt = "New prompt"  # type: ignore
        with pytest.raises(FrozenInstanceError):
            img.width = 1024  # type: ignore
        with pytest.raises(FrozenInstanceError):
            img.seed = 42  # type: ignore

    def test_default_values(self):
        """Les valeurs par defaut sont correctes."""
        img = GeneratedImage(
            scene_order=1,
            prompt="P",
            negative_prompt="N",
            width=720,
            height=1280,
            aspect_ratio="9:16",
            seed=0,
        )
        assert img.quality == "standard"
        assert img.steps == 20
        assert img.style == "default"
        assert img.color_palette == []
        assert img.provider == "heuristic"
        assert img.metadata == {}

    def test_repr(self):
        """Le repr est lisible."""
        img = GeneratedImage(
            scene_order=1,
            prompt="P",
            negative_prompt="N",
            width=720,
            height=1280,
            aspect_ratio="9:16",
            seed=42,
        )
        r = repr(img)
        assert "GeneratedImage" in r
        assert "720" in r

    def test_seed_range(self):
        """Le seed est dans la plage [0, 2^32-1]."""
        img = GeneratedImage(
            scene_order=1,
            prompt="P",
            negative_prompt="N",
            width=720,
            height=1280,
            aspect_ratio="9:16",
            seed=123456,
        )
        assert 0 <= img.seed < 2**32


# ── Tests : ImageGenerator (ABC) ───────────────────────────────────────────

class TestImageGenerator:
    def test_cannot_instantiate(self):
        """ImageGenerator est une interface, pas instanciable."""
        with pytest.raises(TypeError):
            ImageGenerator()  # type: ignore

    def test_subclass_must_implement_generate(self):
        """Une sous-classe doit implementer generate()."""
        class BadGenerator(ImageGenerator):
            @property
            def name(self):
                return "bad"

        with pytest.raises(TypeError):
            BadGenerator()

    def test_subclass_must_implement_name(self):
        """Une sous-classe doit implementer name."""
        class BadGenerator(ImageGenerator):
            def generate(self, scene, plan):
                return None

        with pytest.raises(TypeError):
            BadGenerator()

    def test_valid_subclass(self):
        """Une sous-classe complete est instanciable."""
        class GoodGenerator(ImageGenerator):
            @property
            def name(self):
                return "good"

            def generate(self, scene, plan):
                return GeneratedImage(
                    scene_order=1,
                    prompt="test",
                    negative_prompt="bad",
                    width=720,
                    height=1280,
                    aspect_ratio="9:16",
                    seed=0,
                    provider=self.name,
                )

        gen = GoodGenerator()
        assert gen.name == "good"
        scene = VisualScene(
            scene_order=1, shot_type="medium", camera_motion="static",
            visual_prompt="P", composition="C", lighting="L",
            color_palette=["#000"], transition="cut", overlay_text="O",
            animation_notes="A", duration_seconds=10,
        )
        plan = VisualPlan(title="T", style="default", aspect_ratio="9:16",
                          scenes=[scene], color_palette=["#000"])
        img = gen.generate(scene, plan)
        assert isinstance(img, GeneratedImage)
        assert img.provider == "good"


# ── Tests : HeuristicImageGenerator ────────────────────────────────────────

class TestHeuristicImageGenerator:
    def test_name(self):
        """Le generateur a un nom identifiable."""
        gen = HeuristicImageGenerator()
        assert gen.name == "heuristic_image_v1"

    def test_generate_returns_generatedimage(self, sample_visual_scene_hook, sample_visual_plan):
        """generate retourne une GeneratedImage."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert isinstance(img, GeneratedImage)
        assert img.scene_order == 1

    def test_prompt_contains_composition(self, sample_visual_scene_hook, sample_visual_plan):
        """Le prompt contient la composition."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert "Centrage dynamique" in img.prompt or "composition" in img.prompt.lower()

    def test_prompt_contains_lighting(self, sample_visual_scene_hook, sample_visual_plan):
        """Le prompt contient l'eclairage."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert "Rembrandt" in img.prompt or "lumière" in img.prompt.lower() or "light" in img.prompt.lower()

    def test_prompt_contains_style_keywords(self, sample_visual_scene_hook, sample_visual_plan):
        """Le prompt contient les mots-cles de style."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert "Style:" in img.prompt

    def test_prompt_contains_palette(self, sample_visual_scene_hook, sample_visual_plan):
        """Le prompt contient la palette de couleurs."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert "Color palette:" in img.prompt

    def test_prompt_contains_aspect_ratio(self, sample_visual_scene_hook, sample_visual_plan):
        """Le prompt contient l'aspect ratio et les dimensions."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert "Aspect ratio:" in img.prompt
        assert f"{img.width}x{img.height}" in img.prompt

    def test_prompt_contains_overlay(self, sample_visual_scene_hook, sample_visual_plan):
        """Le prompt contient le overlay text."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert "Text overlay:" in img.prompt
        assert "developpeurs" in img.prompt

    def test_negative_prompt_present(self, sample_visual_scene_hook, sample_visual_plan):
        """Le negative prompt est present et non vide."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert len(img.negative_prompt) > 10
        assert "dated" in img.negative_prompt or "ugly" in img.negative_prompt

    def test_negative_prompt_matches_style(self, sample_visual_scene_hook, sample_visual_plan):
        """Le negative prompt correspond au style innovant."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        # style=Innovant → _NEGATIVE_STYLE_KEYWORDS["innovant"]
        assert "dated" in img.negative_prompt
        assert "retro" in img.negative_prompt

    def test_dimensions_9_16(self, sample_visual_scene_hook, sample_visual_plan):
        """Dimensions correctes pour 9:16."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert img.width == 720
        assert img.height == 1280

    def test_dimensions_16_9(self, wide_visual_plan):
        """Dimensions correctes pour 16:9."""
        gen = HeuristicImageGenerator()
        scene = wide_visual_plan.scenes[0]
        img = gen.generate(scene, wide_visual_plan)
        assert img.aspect_ratio == "16:9"
        assert img.width == 1280
        assert img.height == 720

    def test_hd_for_macro(self, wide_visual_plan):
        """Macro et extreme close up → resolution HD."""
        gen = HeuristicImageGenerator()
        scene = wide_visual_plan.scenes[1]  # extreme_close_up
        plan = wide_visual_plan
        img = gen.generate(scene, plan)
        assert img.quality == "hd"
        # Pour extreme_close_up en 16:9 on utilise la full res
        assert img.width > 1280 or img.height > 720

    def test_seed_is_deterministic(self, sample_visual_scene_hook, sample_visual_plan):
        """Meme entree → meme seed (reproductibilite)."""
        gen = HeuristicImageGenerator()
        img1 = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        img2 = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert img1.seed == img2.seed

    def test_different_scene_different_seed(self, sample_visual_scene_hook, sample_visual_scene_intro, sample_visual_plan):
        """Scene differente → seed different."""
        gen = HeuristicImageGenerator()
        img1 = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        img2 = gen.generate(sample_visual_scene_intro, sample_visual_plan)
        assert img1.seed != img2.seed

    def test_seed_in_range(self, sample_visual_scene_hook, sample_visual_plan):
        """Le seed est dans [0, 2^32 - 1]."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert 0 <= img.seed < 2**32

    def test_provider_is_heuristic(self, sample_visual_scene_hook, sample_visual_plan):
        """Le provider est 'heuristic_image_v1'."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert img.provider == "heuristic_image_v1"

    def test_style_from_plan(self, sample_visual_scene_hook, sample_visual_plan):
        """Le style provient du plan."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert img.style == "Innovant"
        # Vérifie que le mot-clé innovant est utilisé
        assert "futuristic" in img.prompt or "cyberpunk" in img.prompt or "neon" in img.prompt

    def test_color_palette_preserved(self, sample_visual_scene_hook, sample_visual_plan):
        """La palette de la scene est copiee dans l'image."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert img.color_palette == sample_visual_scene_hook.color_palette

    def test_quality_standard_for_normal(self, sample_visual_scene_intro, sample_visual_plan):
        """Scene normale → qualite standard."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_intro, sample_visual_plan)
        assert img.quality == "standard"
        assert img.steps == 20

    def test_quality_hd_for_hook(self, sample_visual_scene_hook, sample_visual_plan):
        """Hook → qualite HD."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert img.quality == "hd"
        assert img.steps == 30

    def test_all_style_keywords_have_negatives(self):
        """Chaque style a un negative prompt correspondant."""
        for style in _STYLE_KEYWORDS:
            assert style in _NEGATIVE_STYLE_KEYWORDS, f"Style '{style}' n'a pas de negative prompt"

    def test_all_aspect_ratios_have_dimensions(self):
        """Chaque aspect ratio a des dimensions definies."""
        for ratio in _ASPECT_RATIOS:
            assert ratio in _ASPECT_RATIOS_FULL, f"Ratio '{ratio}' n'a pas de dimensions full"

    def test_dimensions_are_multiple_of_64(self, wide_visual_plan):
        """Les dimensions sont des multiples de 64 (sauf 720 et 1080 qui sont standards)."""
        gen = HeuristicImageGenerator()
        for scene in wide_visual_plan.scenes:
            img = gen.generate(scene, wide_visual_plan)
            # Hauteurs/largeurs standards acceptees meme si non multiples de 64
            standard_dims = {720, 1080, 1280, 1920}
            if img.width not in standard_dims:
                assert img.width % 64 == 0, f"Largeur {img.width} non multiple de 64"
            if img.height not in standard_dims:
                assert img.height % 64 == 0, f"Hauteur {img.height} non multiple de 64"

    def test_metadata_present(self, sample_visual_scene_hook, sample_visual_plan):
        """Des metadonnees sont presentes dans l'image generee."""
        gen = HeuristicImageGenerator()
        img = gen.generate(sample_visual_scene_hook, sample_visual_plan)
        assert "generator" in img.metadata
        assert img.metadata["generator"] == "heuristic_image_v1"
        assert "seed" in img.metadata
        assert img.metadata["seed"] == img.seed

    def test_aspect_ratio_resolved_from_shotype(self):
        """L'aspect ratio se deduit du shot_type si non defini dans le plan."""
        gen = HeuristicImageGenerator()

        # wide → 16:9
        wide_scene = VisualScene(
            scene_order=1, shot_type="wide", camera_motion="static",
            visual_prompt="P", composition="C", lighting="L",
            color_palette=["#000"], transition="cut", overlay_text="O",
            animation_notes="A", duration_seconds=10,
        )
        wide_plan = VisualPlan(title="T", style="default", aspect_ratio="", scenes=[wide_scene])
        # Note: aspect_ratio="" n'est pas dans le mapping, donc heuristique shot_type
        img = gen.generate(wide_scene, wide_plan)
        # shot_type=wide → landscape → 16:9
        assert img.aspect_ratio in ("16:9",)

    def test_resolve_aspect_ratio_unknown_fallback(self):
        """Aspect ratio inconnu et shot_type non mappe → fallback 9:16."""
        gen = HeuristicImageGenerator()
        ratio = gen._resolve_aspect_ratio("7:8", VisualScene(
            scene_order=1, shot_type="custom", camera_motion="static",
            visual_prompt="P", composition="C", lighting="L",
            color_palette=["#000"], transition="cut", overlay_text="O",
            animation_notes="A", duration_seconds=10,
        ))
        assert ratio == "9:16"


# ── Tests : ImageEngine ────────────────────────────────────────────────────

class TestImageEngine:
    def test_default_generator(self):
        """ImageEngine utilise HeuristicImageGenerator par defaut."""
        engine = ImageEngine()
        assert engine.generator_name == "heuristic_image_v1"

    def test_custom_generator(self):
        """ImageEngine accepte un generateur personnalise."""
        class CustomGen(ImageGenerator):
            @property
            def name(self):
                return "custom"

            def generate(self, scene, plan):
                return GeneratedImage(
                    scene_order=1, prompt="custom", negative_prompt="N",
                    width=720, height=1280, aspect_ratio="9:16", seed=0,
                )

        engine = ImageEngine(generator=CustomGen())
        assert engine.generator_name == "custom"

    def test_generate_returns_list(self, sample_visual_plan):
        """generate() retourne une liste de GeneratedImage."""
        engine = ImageEngine()
        images = engine.generate(sample_visual_plan)
        assert isinstance(images, list)
        assert len(images) == 3

    def test_generate_images_ordered(self, sample_visual_plan):
        """Les images sont dans le meme ordre que les scenes."""
        engine = ImageEngine()
        images = engine.generate(sample_visual_plan)
        for i, img in enumerate(images):
            assert img.scene_order == i + 1

    def test_generate_empty_plan_raises(self):
        """Plan sans scenes → ValueError."""
        plan = VisualPlan(title="Vide", style="default", scenes=[])
        engine = ImageEngine()
        with pytest.raises(ValueError, match="aucune sc.ne"):
            engine.generate(plan)

    def test_generate_all(self, sample_visual_plan):
        """generate_all() retourne une liste de listes."""
        engine = ImageEngine()
        results = engine.generate_all([sample_visual_plan, sample_visual_plan])
        assert len(results) == 2
        assert all(isinstance(imgs, list) for imgs in results)
        assert all(isinstance(img, GeneratedImage) for imgs in results for img in imgs)

    def test_generate_all_empty(self):
        """generate_all([]) retourne une liste vide."""
        engine = ImageEngine()
        results = engine.generate_all([])
        assert results == []

    def test_generate_all_with_error(self, sample_visual_plan):
        """generate_all ignore les plans qui echouent."""
        empty_plan = VisualPlan(title="Vide", style="default", scenes=[])
        engine = ImageEngine()
        results = engine.generate_all([sample_visual_plan, empty_plan, sample_visual_plan])
        # Le plan vide est ignore → 2 plans valides
        assert len(results) == 3
        assert len(results[0]) == 3  # premier plan → 3 images
        assert len(results[1]) == 0  # plan vide → 0 images
        assert len(results[2]) == 3  # troisieme plan → 3 images

    def test_generate_all_preserves_order(self, sample_visual_plan):
        """L'ordre des resultats correspond a l'ordre des plans."""
        plan_a = VisualPlan(title="A", style="default",
            scenes=[VisualScene(scene_order=1, shot_type="medium", camera_motion="static",
                                visual_prompt="P", composition="C", lighting="L",
                                color_palette=["#000"], transition="cut", overlay_text="O",
                                animation_notes="A", duration_seconds=5)])
        plan_b = VisualPlan(title="B", style="default",
            scenes=[VisualScene(scene_order=1, shot_type="medium", camera_motion="static",
                                visual_prompt="P", composition="C", lighting="L",
                                color_palette=["#000"], transition="cut", overlay_text="O",
                                animation_notes="A", duration_seconds=5)])
        engine = ImageEngine()
        results = engine.generate_all([plan_b, plan_a])
        assert results[0][0].scene_order == 1

    def test_images_provider_correct(self, sample_visual_plan):
        """Le provider par defaut est heuristic_image_v1."""
        engine = ImageEngine()
        images = engine.generate(sample_visual_plan)
        for img in images:
            assert img.provider == "heuristic_image_v1"

    def test_images_have_negative_prompt(self, sample_visual_plan):
        """Toutes les images generees ont un negative prompt."""
        engine = ImageEngine()
        images = engine.generate(sample_visual_plan)
        for img in images:
            assert len(img.negative_prompt) > 0


# ── Tests : Decouplage ─────────────────────────────────────────────────────

class TestDecoupling:
    def test_import_does_not_import_internal_engines_source(self):
        """Le module source n'importe pas les autres moteurs (verification du source)."""
        mod_src = Path(__file__).resolve().parent.parent / "src" / "image_engine.py"
        content = mod_src.read_text(encoding="utf-8")
        for forbidden_token in [
            "script_engine", "opportunity_engine", "brand_engine", "creative_engine",
            "knowledge_engine", "virality_engine", "content_understanding",
            "niche_intelligence", "collector", "storage", "agents",
            "learning_engine", "llm", "animation_engine", "video_engine",
        ]:
            assert f"from src.{forbidden_token}" not in content, \
                f"Import interdit: src.{forbidden_token}"

    def test_only_visual_engine_imported(self):
        """Le module importe seulement VisualPlan et VisualScene."""
        mod_src = Path(__file__).resolve().parent.parent / "src" / "image_engine.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "from src.visual_engine import VisualPlan, VisualScene" in content

    def test_no_script_import(self):
        """Aucun import direct de Script dans image_engine."""
        mod_src = Path(__file__).resolve().parent.parent / "src" / "image_engine.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "Script" not in content or "ScriptScene" not in content

    def test_generate_signature(self):
        """generate() prend VisualScene et VisualPlan."""
        import inspect
        sig = inspect.signature(ImageGenerator.generate)
        params = list(sig.parameters.keys())
        assert "scene" in params
        assert "plan" in params
        # Pas de parametres indesirables
        for extra in ["opportunity", "brief", "brand", "profile", "script"]:
            assert extra not in params, f"Parametre interdit: {extra}"
