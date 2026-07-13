import json

from src.brand_engine import BrandProfile
from src.llm_animation_generator import AnimationPrompt
from src.llm_image_generator import ImagePrompt
from src.niche_intelligence import Niche
from src.production_package_builder import (
    NicheProductionResult,
    ProductionPackageBuilder,
)
from src.script_engine import Dialogue, Scene, SceneDescription, Script, ScriptScene


def _brand() -> BrandProfile:
    return BrandProfile(
        id="ia_fr", name="IA FR", description="", niche="IA", target_audience="",
        primary_language="fr", tone="Innovant", personality="", writing_style="",
        emotion_level=0.5, humor_level=0.5, authority_level=0.5, curiosity_level=0.5,
        storytelling_level=0.5, voice_speed="Modéré", preferred_video_duration=600,
        preferred_formats=["Analyse"], preferred_hooks=["Hook ?"], preferred_cta=["Abonne-toi."],
        forbidden_words=[], visual_style="", color_palette=["Warm Orange", "Cyan"], typography_style="",
        logo_description="", thumbnail_style="", metadata={},
    )


def _niche() -> Niche:
    return Niche(name="ia", volume=10, avg_views=10_000.0, avg_engagement=1.5,
                 avg_growth_speed=100.0, niche_score=5.0, timelines=[])


def _description() -> SceneDescription:
    return SceneDescription(
        setting="Un decor de laboratoire futuriste, lumiere bleutee.",
        composition="Sujet centre, profondeur de champ nette.",
        characters="Narrateur uniquement.",
        lighting="Lumiere bleutee, contrastes doux.",
        camera="Plan fixe, leger dolly-in.",
        mood="Curiosite.",
        symbolism="Le laboratoire evoque la decouverte.",
        director_notes="Garder le rythme, guider le regard vers le sujet.",
        viewer_emotion="Curiosite grandissante.",
    )


def _script() -> Script:
    scene = ScriptScene(
        scene=Scene(number=1, type="hook", description=_description()),
        dialogues=[Dialogue(personnage="NARRATEUR", replique="Bonjour")],
        transition="Fondu.", duration_seconds=10,
    )
    return Script(title="Titre", scenes=[scene], estimated_duration=100,
                  language="fr", target_audience="Curieux", style="Innovant",
                  metadata={"generator": "llm_v1"})


def _image_prompt(fallback=False) -> ImagePrompt:
    metadata = {
        "goal": "g", "emotion": "e", "characters": [],
        "appearance": "young woman, short dark hair", "clothing": "white lab coat",
        "accessories": "safety goggles", "pose": "leaning over a workbench",
        "facial_expression": "focused", "weather": "N/A", "time_of_day": "night",
        "background": "rows of glowing server racks",
    }
    if fallback:
        metadata.update({"provider": "heuristic_image_v1", "model": "", "time_ms": 0,
                          "cost_usd": 0.0, "fallback_reason": "validation_failed"})
    else:
        metadata.update({"provider": "deepseek", "model": "deepseek-chat",
                          "time_ms": 1234, "cost_usd": 0.0021})
    return ImagePrompt(subject="robot", scene_description="lab", style="cinematic",
                        prompt="show the robot", negative_prompt="blurry", metadata=metadata)


def _animation_prompt() -> AnimationPrompt:
    return AnimationPrompt(
        camera_motion="pan", subject_motion="walk", environment_motion="none",
        lighting_changes="none", effects="none", sound_design="ambient",
        dialogues=[Dialogue(personnage="NARRATEUR", replique="Bonjour")],
        transition="cut", duration=10, prompt="animate the robot",
        metadata={"goal": "g", "emotion": "e", "provider": "deepseek",
                  "model": "deepseek-chat", "time_ms": 987, "cost_usd": 0.0015,
                  "animation_style": "smooth 24fps motion", "voice": "female, calm",
                  "sound_effects": "faint beep", "background_music": "low tense synth"},
    )


def _result(fallback_image=False) -> NicheProductionResult:
    return NicheProductionResult(
        niche=_niche(), brand=_brand(), final_script=_script(),
        images=[{"scene_order": 1, "image_prompt": _image_prompt(fallback=fallback_image), "shot_plan": None}],
        animations=[{"scene_order": 1, "animation_prompt": _animation_prompt()}],
        rewrite_result=None,
    )


class TestPackageStructure:
    def test_creates_expected_files(self, tmp_path):
        builder = ProductionPackageBuilder()

        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        assert package_dir == tmp_path / "niche_01"
        assert (package_dir / "final_script.json").exists()
        assert (package_dir / "image_prompts" / "scene_01.json").exists()
        assert (package_dir / "animation_prompts" / "scene_01.json").exists()
        assert (package_dir / "report.md").exists()
        assert not (package_dir / "script_final.txt").exists()

    def test_final_script_content_matches_source(self, tmp_path):
        """Sprint 32.1 : final_script.json adopte le contrat storyboard
        cinematographique — UNIQUEMENT {title, scenes[{scene: {number, type,
        description{9 champs}}, dialogues, transition, duration_seconds}]}."""
        builder = ProductionPackageBuilder()
        result = _result()

        package_dir = builder.build(tmp_path, niche_index=2, result=result)

        data = json.loads((package_dir / "final_script.json").read_text(encoding="utf-8"))
        assert set(data.keys()) == {"title", "scenes"}
        assert data["title"] == result.final_script.title
        assert len(data["scenes"]) == 1
        scene = data["scenes"][0]
        assert set(scene.keys()) == {"scene", "dialogues", "transition", "duration_seconds"}
        source_scene = result.final_script.scenes[0]
        assert scene["scene"]["number"] == source_scene.scene.number
        assert scene["scene"]["type"] == source_scene.scene.type
        assert set(scene["scene"]["description"].keys()) == {
            "setting", "composition", "characters", "lighting", "camera",
            "mood", "symbolism", "director_notes", "viewer_emotion",
        }
        assert scene["scene"]["description"]["setting"] == source_scene.scene.description.setting
        assert scene["dialogues"] == [{"personnage": "NARRATEUR", "replique": "Bonjour"}]
        assert scene["transition"] == "Fondu."
        assert scene["duration_seconds"] == 10
        # Aucune information interne (metadata, language, style, estimated_duration)
        assert "metadata" not in data
        assert "language" not in data

    def test_no_technical_directories_in_package(self, tmp_path):
        builder = ProductionPackageBuilder()

        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        assert not (package_dir / "shot_plans").exists()
        assert not (package_dir / "benchmark.json").exists()
        assert not (package_dir / ".cache").exists()

    def test_report_contains_key_facts(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()

        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        report = (package_dir / "report.md").read_text(encoding="utf-8")
        assert result.niche.name in report
        assert result.brand.name in report
        assert result.final_script.title in report


class TestImagePromptMegaPrompt:
    """Sprint 34.6 — image_prompts/scene_XX.json adopte un format compact à
    3 clés {prompt, negative_prompt, instruction_format} : "prompt" concatène
    des libellés riches (Subject/Appearance/.../Language), construits à
    partir du contenu déjà généré (ImagePrompt, ShotPlan, SceneDescription,
    BrandProfile) — aucune nouvelle génération LLM."""

    def test_file_has_exactly_three_keys(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        data = json.loads((package_dir / "image_prompts" / "scene_01.json").read_text(encoding="utf-8"))
        assert set(data.keys()) == {"prompt", "negative_prompt", "instruction_format"}

    def test_prompt_contains_expected_labels_and_content(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        data = json.loads((package_dir / "image_prompts" / "scene_01.json").read_text(encoding="utf-8"))
        prompt = data["prompt"]
        for label in (
            "Subject:", "Appearance:", "Clothing:", "Accessories:", "Pose:", "Action:",
            "Facial Expression:", "Emotion:", "Environment:", "Background:", "Weather:",
            "Time of Day:", "Lighting:", "Camera Angle:", "Lens:", "Composition:", "Style:",
            "Color Palette:", "Details:", "Text (optional):", "Language:",
        ):
            assert label in prompt, f"Libellé manquant : {label}"

        image_prompt = result.images[0]["image_prompt"]
        assert image_prompt.subject in prompt
        assert image_prompt.metadata["appearance"] in prompt
        assert image_prompt.metadata["clothing"] in prompt
        assert image_prompt.style in prompt
        # Color Palette retombe sur la marque (aucun ShotPlan fourni)
        assert "Warm Orange" in prompt

    def test_negative_prompt_and_instruction_format(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        data = json.loads((package_dir / "image_prompts" / "scene_01.json").read_text(encoding="utf-8"))
        assert data["negative_prompt"] == result.images[0]["image_prompt"].negative_prompt
        assert data["instruction_format"] == (
            "Respond STRICTLY in valid JSON. Do not include any explanation or markdown."
        )

    def test_no_technical_metadata_leaks_into_prompt(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        raw = (package_dir / "image_prompts" / "scene_01.json").read_text(encoding="utf-8")
        assert "deepseek" not in raw
        assert "cost_usd" not in raw
        assert "1234" not in raw


class TestAnimationPromptMegaPrompt:
    """Sprint 34.6 — animation_prompts/scene_XX.json : même principe, en
    réutilisant en plus l'ImagePrompt de la même scène (apparence déjà
    établie) et les dialogues verbatim de la scène."""

    def test_file_has_exactly_three_keys(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        data = json.loads((package_dir / "animation_prompts" / "scene_01.json").read_text(encoding="utf-8"))
        assert set(data.keys()) == {"prompt", "negative_prompt", "instruction_format"}

    def test_prompt_contains_expected_labels_and_content(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        data = json.loads((package_dir / "animation_prompts" / "scene_01.json").read_text(encoding="utf-8"))
        prompt = data["prompt"]
        for label in (
            "Subject:", "Appearance:", "Clothing:", "Accessories:", "Initial Pose:",
            "Character Action:", "Secondary Actions:", "Facial Expression:", "Emotion:",
            "Environment:", "Background:", "Weather:", "Time of Day:", "Lighting:",
            "Camera Shot:", "Camera Angle:", "Camera Movement:", "Lens:", "Composition:",
            "Visual Style:", "Animation Style:", "Scene Duration:", "Frame Rate:",
            "Dialogue:", "Speaker:", "Narration:", "Language:", "Voice:", "Lip Sync:",
            "Sound Effects:", "Ambient Sounds:", "Background Music:", "Atmosphere:",
            "Ending Scene:",
        ):
            assert label in prompt, f"Libellé manquant : {label}"

        animation_prompt = result.animations[0]["animation_prompt"]
        image_prompt = result.images[0]["image_prompt"]
        assert animation_prompt.camera_motion in prompt
        assert animation_prompt.transition in prompt
        assert image_prompt.metadata["appearance"] in prompt
        assert "Bonjour" in prompt  # dialogue copié verbatim
        assert "NARRATOR" in prompt  # speaker (NARRATEUR normalisé)
        assert result.final_script.language in prompt
        assert "24 fps" in prompt

    def test_negative_prompt_reuses_image_negative_prompt(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        data = json.loads((package_dir / "animation_prompts" / "scene_01.json").read_text(encoding="utf-8"))
        assert data["negative_prompt"] == result.images[0]["image_prompt"].negative_prompt

    def test_no_technical_metadata_leaks_into_prompt(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        raw = (package_dir / "animation_prompts" / "scene_01.json").read_text(encoding="utf-8")
        assert "deepseek" not in raw
        assert "cost_usd" not in raw
        assert "987" not in raw


class TestReportTechnicalMetrics:
    """Sprint 31.1 — report.md reste la source des métriques techniques
    (provider, modèle, temps, coût, statut, fallback), indépendamment du
    format des fichiers image_prompts/*.json et animation_prompts/*.json."""

    def test_report_contains_llm_metrics_for_scene(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result(fallback_image=False)
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        report = (package_dir / "report.md").read_text(encoding="utf-8")
        assert "deepseek" in report
        assert "1234" in report or "1234 ms" in report
        assert "LLM" in report

    def test_report_flags_fallback_scene(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result(fallback_image=True)
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        report = (package_dir / "report.md").read_text(encoding="utf-8")
        assert "heuristic_image_v1" in report
        assert "fallback" in report.lower()
        assert "validation_failed" in report
