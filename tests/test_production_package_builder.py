import json

from src.brand_engine import BrandProfile
from src.llm_animation_generator import AnimationPrompt
from src.llm_image_generator import ImagePrompt
from src.niche_intelligence import Niche
from src.production_package_builder import (
    NicheProductionResult,
    ProductionPackageBuilder,
)
from src.script_engine import Script, ScriptScene


def _brand() -> BrandProfile:
    return BrandProfile(
        id="ia_fr", name="IA FR", description="", niche="IA", target_audience="",
        primary_language="fr", tone="Innovant", personality="", writing_style="",
        emotion_level=0.5, humor_level=0.5, authority_level=0.5, curiosity_level=0.5,
        storytelling_level=0.5, voice_speed="Modéré", preferred_video_duration=600,
        preferred_formats=["Analyse"], preferred_hooks=["Hook ?"], preferred_cta=["Abonne-toi."],
        forbidden_words=[], visual_style="", color_palette=[], typography_style="",
        logo_description="", thumbnail_style="", metadata={},
    )


def _niche() -> Niche:
    return Niche(name="ia", volume=10, avg_views=10_000.0, avg_engagement=1.5,
                 avg_growth_speed=100.0, niche_score=5.0, timelines=[])


def _script() -> Script:
    scene = ScriptScene(order=1, title="Intro", narration="Bonjour", visual_description="",
                        image_prompt="", animation_notes="", sound_effects="", duration_seconds=10)
    return Script(title="Titre", hook="Hook ?", introduction="Intro", scenes=[scene],
                  conclusion="Fin", call_to_action="Abonne-toi", estimated_duration=100,
                  language="fr", target_audience="Curieux", style="Innovant",
                  metadata={"generator": "llm_v1"})


def _image_prompt() -> ImagePrompt:
    return ImagePrompt(subject="robot", scene_description="lab", style="cinematic",
                        prompt="show the robot", negative_prompt="blurry", metadata={})


def _animation_prompt() -> AnimationPrompt:
    return AnimationPrompt(camera_motion="pan", subject_motion="walk", environment_motion="none",
                            lighting_changes="none", effects="none", sound_design="ambient",
                            transition="cut", duration=10, prompt="animate the robot", metadata={})


def _result() -> NicheProductionResult:
    return NicheProductionResult(
        niche=_niche(), brand=_brand(), final_script=_script(),
        images=[{"scene_order": 1, "image_prompt": _image_prompt()}],
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

    def test_final_script_content_matches_source(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()

        package_dir = builder.build(tmp_path, niche_index=2, result=result)

        data = json.loads((package_dir / "final_script.json").read_text(encoding="utf-8"))
        assert data["title"] == result.final_script.title
        assert data["hook"] == result.final_script.hook

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
