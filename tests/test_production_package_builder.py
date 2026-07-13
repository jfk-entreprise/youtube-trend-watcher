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
        forbidden_words=[], visual_style="", color_palette=[], typography_style="",
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
    metadata = {"goal": "g", "emotion": "e", "characters": []}
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
                  "model": "deepseek-chat", "time_ms": 987, "cost_usd": 0.0015},
    )


def _result(fallback_image=False) -> NicheProductionResult:
    return NicheProductionResult(
        niche=_niche(), brand=_brand(), final_script=_script(),
        images=[{"scene_order": 1, "image_prompt": _image_prompt(fallback=fallback_image)}],
        animations=[{"scene_order": 1, "animation_prompt": _animation_prompt()}],
        rewrite_result=None,
    )


class TestPackageStructure:
    def test_creates_expected_files(self, tmp_path):
        builder = ProductionPackageBuilder()

        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        assert package_dir == tmp_path / "niche_01"
        assert (package_dir / "final_script.json").exists()
        assert (package_dir / "script_final.txt").exists()
        assert (package_dir / "image_prompts" / "scene_01.json").exists()
        assert (package_dir / "animation_prompts" / "scene_01.json").exists()
        assert (package_dir / "report.md").exists()

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


class TestScriptFinalTxt:
    """Sprint 34.1, allégé aux Sprints 34.4/34.5 — export texte brut pour
    Google Flow Storyboard Studio : chaque scène tient sur 3 lignes
    consécutives (SCENE / NARRATOR ou CHARACTER / TRANSITION), sans saut de
    ligne interne ni numérotation, séparées des autres scènes par une seule
    ligne vide (pas de "---"). Labels toujours en anglais, seules les
    répliques suivent la langue du script. Le paragraphe SCENE ne garde que
    l'essentiel (setting + characters) — le script complet s'est révélé trop
    long pour un import direct dans Storyboard Studio."""

    def test_contains_expected_labels_on_three_consecutive_lines(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()

        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        txt = (package_dir / "script_final.txt").read_text(encoding="utf-8")
        assert txt.startswith(f"TITLE : {result.final_script.title}")
        source_desc = result.final_script.scenes[0].scene.description
        expected_block = (
            f"SCENE: {source_desc.setting} {source_desc.characters}\n"
            f"NARRATOR: Bonjour\n"
            f"TRANSITION: Fondu."
        )
        assert expected_block in txt
        # Pas de numérotation ("SCENE 1"), pas de séparateur "---"
        assert "SCENE 1" not in txt
        assert "---" not in txt

    def test_other_description_fields_are_dropped_from_txt(self, tmp_path):
        """Sprint 34.4 — seuls setting + characters restent dans
        script_final.txt ; composition/lighting/camera/mood/symbolism/
        director_notes/viewer_emotion ne sont plus dupliqués (script jugé
        trop long) mais restent dans final_script.json (contrat complet)."""
        builder = ProductionPackageBuilder()
        result = _result()

        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        txt = (package_dir / "script_final.txt").read_text(encoding="utf-8")
        source_desc = result.final_script.scenes[0].scene.description
        assert source_desc.composition not in txt
        assert source_desc.lighting not in txt
        assert source_desc.camera not in txt
        assert source_desc.mood not in txt
        assert source_desc.symbolism not in txt
        assert source_desc.director_notes not in txt
        assert source_desc.viewer_emotion not in txt

    def test_character_dialogue_uses_character_label(self, tmp_path):
        builder = ProductionPackageBuilder()
        scene = ScriptScene(
            scene=Scene(number=1, type="hook", description=_description()),
            dialogues=[Dialogue(personnage="Robot", replique="Bip bip.")],
            transition="Fondu.", duration_seconds=10,
        )
        script = Script(title="Titre", scenes=[scene], estimated_duration=100,
                         language="fr", target_audience="Curieux", style="Innovant",
                         metadata={"generator": "llm_v1"})
        result = NicheProductionResult(
            niche=_niche(), brand=_brand(), final_script=script,
            images=[], animations=[], rewrite_result=None,
        )

        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        txt = (package_dir / "script_final.txt").read_text(encoding="utf-8")
        assert "CHARACTER (Robot): Bip bip." in txt

    def test_scenes_separated_by_single_blank_line(self, tmp_path):
        desc = _description()
        scene_1 = ScriptScene(
            scene=Scene(number=1, type="hook", description=desc),
            dialogues=[Dialogue(personnage="NARRATEUR", replique="Un.")],
            transition="Cut.", duration_seconds=5,
        )
        scene_2 = ScriptScene(
            scene=Scene(number=2, type="context", description=desc),
            dialogues=[Dialogue(personnage="NARRATEUR", replique="Deux.")],
            transition="Fondu.", duration_seconds=5,
        )
        script = Script(title="Titre", scenes=[scene_1, scene_2], estimated_duration=10,
                         language="fr", target_audience="Curieux", style="Innovant",
                         metadata={"generator": "llm_v1"})
        result = NicheProductionResult(
            niche=_niche(), brand=_brand(), final_script=script,
            images=[], animations=[], rewrite_result=None,
        )
        builder = ProductionPackageBuilder()

        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        txt = (package_dir / "script_final.txt").read_text(encoding="utf-8")
        assert "TRANSITION: Cut.\n\nSCENE:" in txt
        assert "---" not in txt

    def test_no_json_braces_in_txt_output(self, tmp_path):
        """Le fichier doit être du texte brut lisible, pas du JSON imbriqué."""
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        txt = (package_dir / "script_final.txt").read_text(encoding="utf-8")
        assert "{" not in txt
        assert "}" not in txt


class TestTechnicalMetadataStripped:
    """Sprint 31.1 — provider/model/time_ms/cost_usd ne doivent plus jamais
    apparaître dans image_prompts/*.json ni animation_prompts/*.json — ces
    informations vivent désormais uniquement dans report.md."""

    def test_image_prompt_file_has_no_technical_metadata(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        data = json.loads((package_dir / "image_prompts" / "scene_01.json").read_text(encoding="utf-8"))
        for key in ("provider", "model", "time_ms", "cost_usd"):
            assert key not in data["metadata"], f"'{key}' ne doit plus apparaître dans image_prompt.json"
        # Les champs non techniques restent présents
        assert data["metadata"]["goal"] == "g"
        assert data["metadata"]["emotion"] == "e"

    def test_animation_prompt_file_has_no_technical_metadata(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        data = json.loads((package_dir / "animation_prompts" / "scene_01.json").read_text(encoding="utf-8"))
        for key in ("provider", "model", "time_ms", "cost_usd"):
            assert key not in data["metadata"], f"'{key}' ne doit plus apparaître dans animation_prompt.json"
        assert data["metadata"]["goal"] == "g"

    def test_animation_prompt_file_contains_dialogues_verbatim(self, tmp_path):
        """Sprint 31.1 — l'animation_prompt.json doit être autonome : mêmes
        dialogues que la scène correspondante, copiés sans reformulation."""
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        data = json.loads((package_dir / "animation_prompts" / "scene_01.json").read_text(encoding="utf-8"))
        expected = [
            {"personnage": d.personnage, "replique": d.replique}
            for d in result.final_script.scenes[0].dialogues
        ]
        assert data["dialogues"] == expected


class TestReportTechnicalMetrics:
    """Sprint 31.1 — report.md devient la source unique des métriques
    techniques (provider, modèle, temps, coût, statut, fallback)."""

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
