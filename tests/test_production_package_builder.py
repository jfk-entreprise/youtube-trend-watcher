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


def _brand_fr() -> BrandProfile:
    return BrandProfile(
        id="ia_fr", name="IA FR", description="", niche="IA", target_audience="",
        primary_language="fr", tone="Innovant", personality="", writing_style="",
        emotion_level=0.5, humor_level=0.5, authority_level=0.5, curiosity_level=0.5,
        storytelling_level=0.5, voice_speed="Modéré", preferred_video_duration=600,
        preferred_formats=["Analyse"], preferred_hooks=["Hook ?"], preferred_cta=["Abonne-toi."],
        forbidden_words=[], visual_style="", color_palette=["Warm Orange", "Cyan"], typography_style="",
        logo_description="", thumbnail_style="", metadata={}, market="FR",
    )


def _brand_en() -> BrandProfile:
    return BrandProfile(
        id="global_us", name="Global US", description="", niche="General", target_audience="",
        primary_language="en", tone="Bold", personality="", writing_style="",
        emotion_level=0.5, humor_level=0.3, authority_level=0.7, curiosity_level=0.9,
        storytelling_level=0.5, voice_speed="Rapide", preferred_video_duration=600,
        preferred_formats=["Short"], preferred_hooks=["Hook?"], preferred_cta=["Follow."],
        forbidden_words=[], visual_style="", color_palette=["#0A0A0A"], typography_style="",
        logo_description="", thumbnail_style="", metadata={}, market="US",
    )


def _niche() -> Niche:
    return Niche(name="ia", volume=10, avg_views=10_000.0, avg_engagement=1.5,
                 avg_growth_speed=100.0, niche_score=5.0, timelines=[])


def _description() -> SceneDescription:
    return SceneDescription(
        setting="A futuristic lab, blue lighting.",
        composition="Subject centered, sharp depth of field.",
        characters="Narrator only.",
        lighting="Blue glow, soft contrast.",
        camera="Static shot, slight dolly-in.",
        mood="Curiosity.",
        symbolism="The lab evokes discovery.",
        director_notes="Keep the pace, guide the eye to the subject.",
        viewer_emotion="Growing curiosity.",
    )


def _script_en() -> Script:
    scene = ScriptScene(
        scene=Scene(number=1, type="hook", description=_description()),
        dialogues=[Dialogue(personnage="NARRATOR", replique="Hello")],
        transition="Fade.", duration_seconds=10,
    )
    return Script(title="Title", scenes=[scene], estimated_duration=100,
                  language="en", target_audience="Curious", style="Bold",
                  metadata={"generator": "llm_v1"})


def _script_fr() -> Script:
    scene = ScriptScene(
        scene=Scene(number=1, type="hook", description=_description()),
        dialogues=[Dialogue(personnage="NARRATOR", replique="Bonjour")],
        transition="Fade.", duration_seconds=10,
    )
    return Script(title="Title", scenes=[scene], estimated_duration=100,
                  language="fr", target_audience="Curious", style="Bold",
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


def _animation_prompt(replique="Hello") -> AnimationPrompt:
    return AnimationPrompt(
        camera_motion="pan", subject_motion="walk", environment_motion="none",
        lighting_changes="none", effects="none", sound_design="ambient",
        dialogues=[Dialogue(personnage="NARRATOR", replique=replique)],
        transition="cut", duration=10, prompt="animate the robot",
        metadata={"goal": "g", "emotion": "e", "provider": "deepseek",
                  "model": "deepseek-chat", "time_ms": 987, "cost_usd": 0.0015,
                  "animation_style": "smooth 24fps motion", "voice": "female, calm",
                  "sound_effects": "faint beep", "background_music": "low tense synth"},
    )


def _result(fallback_image=False) -> NicheProductionResult:
    images = [{"scene_order": 1, "image_prompt": _image_prompt(fallback=fallback_image), "shot_plan": None}]
    animations_en = [{"scene_order": 1, "animation_prompt": _animation_prompt("Hello")}]
    animations_fr = [{"scene_order": 1, "animation_prompt": _animation_prompt("Bonjour")}]
    return NicheProductionResult(
        niche=_niche(), brand_en=_brand_en(), brand_fr=_brand_fr(),
        final_script_en=_script_en(), final_script_fr=_script_fr(),
        images=images, animations_en=animations_en, animations_fr=animations_fr,
        rewrite_result=None,
    )


class TestPackageStructure:
    def test_creates_expected_files(self, tmp_path):
        builder = ProductionPackageBuilder()

        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        assert package_dir == tmp_path / "niche_01"
        assert (package_dir / "final_script_en.json").exists()
        assert (package_dir / "final_script_fr.json").exists()
        assert (package_dir / "image_prompts" / "scene_01.txt").exists()
        assert (package_dir / "animation_prompts_en" / "scene_01.txt").exists()
        assert (package_dir / "animation_prompts_fr" / "scene_01.txt").exists()
        assert (package_dir / "report.md").exists()
        assert (package_dir / "story.txt").exists()
        assert not (package_dir / "final_script.json").exists()
        assert not (package_dir / "animation_prompts").exists()
        assert not (package_dir / "image_prompts" / "scene_01.json").exists()

    def test_final_script_content_matches_source(self, tmp_path):
        """Sprint 32.1 : final_script_*.json adopte le contrat storyboard
        cinematographique — UNIQUEMENT {title, scenes[{scene: {number, type,
        description{9 champs}}, dialogues, transition, duration_seconds}]}."""
        builder = ProductionPackageBuilder()
        result = _result()

        package_dir = builder.build(tmp_path, niche_index=2, result=result)

        data_en = json.loads((package_dir / "final_script_en.json").read_text(encoding="utf-8"))
        data_fr = json.loads((package_dir / "final_script_fr.json").read_text(encoding="utf-8"))
        assert set(data_en.keys()) == {"title", "scenes"}
        assert data_en["title"] == result.final_script_en.title
        assert data_en["scenes"][0]["dialogues"] == [{"personnage": "NARRATOR", "replique": "Hello"}]
        assert data_fr["scenes"][0]["dialogues"] == [{"personnage": "NARRATOR", "replique": "Bonjour"}]
        # Aucune information interne (metadata, language, style, estimated_duration)
        assert "metadata" not in data_en
        assert "language" not in data_en

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
        assert result.brand_en.name in report
        assert result.brand_fr.name in report
        assert result.final_script_en.title in report


class TestImagePromptText:
    """Sprint 38 — image_prompts/scene_XX.txt (UNIQUE, partagé entre les 2
    langues) est un paragraphe de texte brut unique : image_prompt.prompt,
    déjà écrit et garanti conforme (format Google Veo3/Imagen) par
    LLMImageGenerator — complété seulement par la référence de personnage
    récurrent le cas échéant. Aucun JSON, aucun label, aucun negative_prompt."""

    def test_file_is_plain_text_not_json(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        raw = (package_dir / "image_prompts" / "scene_01.txt").read_text(encoding="utf-8")
        assert not raw.strip().startswith("{")

    def test_content_is_the_image_prompt_text(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        raw = (package_dir / "image_prompts" / "scene_01.txt").read_text(encoding="utf-8")
        assert raw == result.images[0]["image_prompt"].prompt

    def test_no_negative_prompt_or_labels_in_file(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        raw = (package_dir / "image_prompts" / "scene_01.txt").read_text(encoding="utf-8")
        assert result.images[0]["image_prompt"].negative_prompt not in raw
        for label in ("Subject:", "Appearance:", "Camera Angle:", "Character Reference:"):
            assert label not in raw

    def test_no_technical_metadata_leaks_into_prompt(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        raw = (package_dir / "image_prompts" / "scene_01.txt").read_text(encoding="utf-8")
        assert "deepseek" not in raw
        assert "cost_usd" not in raw
        assert "1234" not in raw


class TestAnimationPromptTextBothLanguages:
    """Sprint 38/35 — animation_prompts_en/ et animation_prompts_fr/ sont du
    texte brut : le paragraphe motion/son partagé (animation_prompt.prompt,
    identique pour les 2 langues — aucun second appel LLM pour le FR) suivi
    de la ligne Audio/Narration déterministe (dialogues verbatim, langue
    correcte). Aucun JSON, aucun label."""

    def test_both_folders_are_plain_text(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        for lang_dir in ("animation_prompts_en", "animation_prompts_fr"):
            raw = (package_dir / lang_dir / "scene_01.txt").read_text(encoding="utf-8")
            assert not raw.strip().startswith("{")

    def test_motion_prompt_shared_between_en_and_fr(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        data_en = (package_dir / "animation_prompts_en" / "scene_01.txt").read_text(encoding="utf-8")
        data_fr = (package_dir / "animation_prompts_fr" / "scene_01.txt").read_text(encoding="utf-8")
        motion = result.animations_en[0]["animation_prompt"].prompt
        assert motion in data_en
        assert motion in data_fr

    def test_dialogue_and_language_differ_between_en_and_fr(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        data_en = (package_dir / "animation_prompts_en" / "scene_01.txt").read_text(encoding="utf-8")
        data_fr = (package_dir / "animation_prompts_fr" / "scene_01.txt").read_text(encoding="utf-8")

        assert "Narrator voiceover in English: Hello" in data_en
        assert "Bonjour" not in data_en

        assert "Narrator voiceover in French: Bonjour" in data_fr
        assert "Hello" not in data_fr

    def test_no_technical_metadata_leaks_into_prompt(self, tmp_path):
        builder = ProductionPackageBuilder()
        package_dir = builder.build(tmp_path, niche_index=1, result=_result())

        for lang_dir in ("animation_prompts_en", "animation_prompts_fr"):
            raw = (package_dir / lang_dir / "scene_01.txt").read_text(encoding="utf-8")
            assert "deepseek" not in raw
            assert "cost_usd" not in raw
            assert "987" not in raw


def _two_scene_script(language, repliques) -> Script:
    scenes = [
        ScriptScene(
            scene=Scene(number=i + 1, type="hook" if i == 0 else "development", description=_description()),
            dialogues=[Dialogue(personnage="NARRATOR", replique=repliques[i])],
            transition="Fade.", duration_seconds=10,
        )
        for i in range(len(repliques))
    ]
    return Script(title="Title", scenes=scenes, estimated_duration=sum(s.duration_seconds for s in scenes),
                  language=language, target_audience="Curious", style="Bold", metadata={"generator": "llm_v1"})


def _image_prompt_with_characters(characters, subject="scene subject") -> ImagePrompt:
    metadata = {
        "goal": "g", "emotion": "e", "characters": characters,
        "appearance": "n/a", "clothing": "n/a", "accessories": "n/a", "pose": "n/a",
        "facial_expression": "n/a", "weather": "N/A", "time_of_day": "night", "background": "n/a",
        "provider": "deepseek", "model": "deepseek-chat", "time_ms": 1000, "cost_usd": 0.001,
    }
    return ImagePrompt(subject=subject, scene_description="scene", style="cinematic",
                        prompt="show the scene", negative_prompt="blurry", metadata=metadata)


def _result_with_characters(characters_by_scene) -> NicheProductionResult:
    """Construit un NicheProductionResult à N scènes, chacune avec sa propre
    liste `characters` (metadata d'ImagePrompt) — pour tester le suivi des
    personnages récurrents entre scènes."""
    n = len(characters_by_scene)
    repliques_en = [f"Line {i + 1}" for i in range(n)]
    repliques_fr = [f"Ligne {i + 1}" for i in range(n)]
    images = [
        {"scene_order": i + 1, "image_prompt": _image_prompt_with_characters(characters_by_scene[i]), "shot_plan": None}
        for i in range(n)
    ]
    animations_en = [
        {"scene_order": i + 1, "animation_prompt": _animation_prompt(repliques_en[i])} for i in range(n)
    ]
    animations_fr = [
        {"scene_order": i + 1, "animation_prompt": _animation_prompt(repliques_fr[i])} for i in range(n)
    ]
    return NicheProductionResult(
        niche=_niche(), brand_en=_brand_en(), brand_fr=_brand_fr(),
        final_script_en=_two_scene_script("en", repliques_en),
        final_script_fr=_two_scene_script("fr", repliques_fr),
        images=images, animations_en=animations_en, animations_fr=animations_fr,
        rewrite_result=None,
    )


class TestCharacterNameTokens:
    """Sprint 37.3 — extraction du nom probable d'un personnage depuis une
    entrée de la liste `characters` (format libre, nom généralement en tête)."""

    def test_extracts_leading_proper_name(self):
        from src.production_package_builder import _character_name_tokens
        tokens = _character_name_tokens("Maya Hart, late 40s, short gray hair, director's cap")
        assert tokens == {"maya", "hart"}

    def test_single_word_name(self):
        from src.production_package_builder import _character_name_tokens
        tokens = _character_name_tokens("Ravi, middle-aged jeweler, warm brown skin")
        assert tokens == {"ravi"}

    def test_generic_descriptive_lead_returns_empty(self):
        """Sprint 37.3 — 'Young woman...' n'est pas un nom propre, juste une
        description générique : aucun token ne doit en être extrait."""
        from src.production_package_builder import _character_name_tokens
        tokens = _character_name_tokens("Young woman, early 20s, focused expression")
        assert tokens == set()

    def test_no_characters_returns_empty(self):
        from src.production_package_builder import _character_name_tokens
        assert _character_name_tokens("") == set()


class TestCharacterReferenceTracking:
    """Sprint 37.3 — un personnage nommé qui réapparaît dans une scène
    ultérieure doit renvoyer vers la première scène où il est apparu, pour
    que l'utilisateur fournisse l'image déjà générée comme référence
    visuelle (même visage/coiffure/tenue) à son outil de génération."""

    def test_first_appearance_has_no_reference(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result_with_characters([
            ["Maya Hart, late 40s, short gray hair, director's cap"],
        ])
        package_dir = builder.build(tmp_path, niche_index=1, result=result)
        raw = (package_dir / "image_prompts" / "scene_01.txt").read_text(encoding="utf-8")
        assert raw == result.images[0]["image_prompt"].prompt
        assert "already appeared" not in raw

    def test_recurring_character_references_first_scene(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result_with_characters([
            ["Maya Hart, late 40s, short gray hair, director's cap"],
            ["Maya Hart, director, tense expression, casual jacket"],
        ])
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        scene1 = (package_dir / "image_prompts" / "scene_01.txt").read_text(encoding="utf-8")
        scene2 = (package_dir / "image_prompts" / "scene_02.txt").read_text(encoding="utf-8")

        assert "already appeared" not in scene1
        assert "Maya Hart" in scene2
        assert "scene_01" in scene2
        assert "image_prompts/scene_01.txt" in scene2

    def test_reference_also_present_in_animation_prompts(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result_with_characters([
            ["Maya Hart, late 40s, short gray hair"],
            ["Maya Hart, tense expression"],
        ])
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        for lang_dir in ("animation_prompts_en", "animation_prompts_fr"):
            scene2 = (package_dir / lang_dir / "scene_02.txt").read_text(encoding="utf-8")
            assert "scene_01" in scene2

    def test_unrelated_characters_get_no_reference(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result_with_characters([
            ["Maya Hart, late 40s, short gray hair"],
            ["Ravi, middle-aged jeweler, warm brown skin"],
        ])
        package_dir = builder.build(tmp_path, niche_index=1, result=result)
        scene2 = (package_dir / "image_prompts" / "scene_02.txt").read_text(encoding="utf-8")
        assert "already appeared" not in scene2

    def test_three_scenes_all_reference_earliest_appearance(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result_with_characters([
            ["Maya Hart, late 40s, short gray hair"],
            ["Some unrelated background extra"],
            ["Maya Hart, tense, gripping the table"],
        ])
        package_dir = builder.build(tmp_path, niche_index=1, result=result)
        scene3 = (package_dir / "image_prompts" / "scene_03.txt").read_text(encoding="utf-8")
        assert "scene_01" in scene3


class TestStoryText:
    """Sprint 37.4 — story.txt : uniquement l'histoire, en français, via la
    narration/les dialogues du script FR déjà traduit — aucun élément
    technique du storyboard (caméra, transition, durée, type de scène,
    metadata) ne doit y apparaître."""

    def test_story_file_is_plain_text_in_french(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        story = (package_dir / "story.txt").read_text(encoding="utf-8")
        assert story.strip() == "Bonjour"
        assert "Hello" not in story  # jamais la version anglaise

    def test_multi_scene_story_joins_narrations_in_order(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result_with_characters([
            ["Maya Hart, late 40s, short gray hair"],
            ["Some unrelated background extra"],
            ["Maya Hart, tense, gripping the table"],
        ])
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        story = (package_dir / "story.txt").read_text(encoding="utf-8")
        assert story == "Ligne 1\n\nLigne 2\n\nLigne 3"

    def test_no_technical_storyboard_fields_in_story(self, tmp_path):
        """Ni caméra/transition/durée/type de scène, ni les libellés
        techniques du storyboard (setting/mood/symbolism en anglais) ne
        doivent fuiter dans story.txt — uniquement l'histoire elle-même."""
        builder = ProductionPackageBuilder()
        result = _result()
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        story = (package_dir / "story.txt").read_text(encoding="utf-8")
        for leaked in (
            "Transition", "Fade.", "duration_seconds", "hook",
            "A futuristic lab", "Curiosity.", "camera", "Scene 1",
        ):
            assert leaked not in story

    def test_story_text_takes_precedence_over_fallback_join(self, tmp_path):
        """Sprint 38.1 — si result.story_text est fourni (StoryNarrator),
        il est utilisé tel quel plutôt que le simple enchaînement des
        narrations de scène."""
        import dataclasses
        builder = ProductionPackageBuilder()
        result = dataclasses.replace(_result(), story_text="Un récit fluide déjà écrit par StoryNarrator.")
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        story = (package_dir / "story.txt").read_text(encoding="utf-8")
        assert story == "Un récit fluide déjà écrit par StoryNarrator."

    def test_falls_back_to_simple_join_when_story_text_is_none(self, tmp_path):
        builder = ProductionPackageBuilder()
        result = _result()
        assert result.story_text is None
        package_dir = builder.build(tmp_path, niche_index=1, result=result)

        story = (package_dir / "story.txt").read_text(encoding="utf-8")
        assert story.strip() == "Bonjour"


class TestReportTechnicalMetrics:
    """Sprint 31.1 — report.md reste la source des métriques techniques
    (provider, modèle, temps, coût, statut, fallback), indépendamment du
    format des fichiers image_prompts/*.json et animation_prompts_*/*.json."""

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
