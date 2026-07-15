"""
Tests unitaires pour le LLM Image Generator.
(Sprint 23 → 25 : Nano Banana, raisonnement explicite, contrat ImagePrompt
universel Sprint 24.1, prompt système Art Director Sprint 24.2, cohérence
visuelle / bible de personnages Sprint 24.3, fiabilisation JSON Sprint 24.5.)

Teste :
  1. LLMImageGenerator — création, nom, stats, bible de personnages
  2. build_user_prompt — contient les 3 sources + bloc de continuité narrative
  3. extract_json — extraction robuste (<think>, Markdown, texte parasite,
     commentaires, virgules traînantes, caractères de contrôle — Sprint 24.5)
  4. validate_json_structure / parse_and_validate — validation stricte des
     8 champs + classification des causes d'échec (Sprint 24.1 / 24.5)
  5. finalize_style_for_render / finalize_negative_prompt — garanties
     déterministes (9:16, HDR/8K, photoréaliste, cinématographique)
  6. build_image_prompt / from_generated_image / to_generated_image —
     conversions du contrat ImagePrompt (Sprint 24.1)
  7. retry intelligent — correction JSON via un second appel LLM avant tout
     fallback (Sprint 24.5)
  8. generate_from_scenes — fallback heuristique uniquement en dernier
     recours, avec raison enregistrée (Sprint 24.5)
  9. generate(scene, plan) — interface ImageGenerator, retourne un GeneratedImage
  10. Conserve ImageGenerator comme interface (isinstance)
  11. _resolve_model / mode Reasoning DeepSeek — deepseek-chat par défaut
      pour les images (Sprint 24.5)
  12. Bible de personnages — verrouillage, non-écrasement, continuité (Sprint 24.3)
"""

import json
from pathlib import Path

import pytest

from src.llm import LLMResponse
from src.llm_image_generator import (
    ImagePrompt,
    LLMImageGenerator,
    _ImageJsonError,
    _REQUIRED_FIELDS,
    _REQUIRED_STRING_FIELDS,
    _REQUIRED_LIST_FIELDS,
    _finalize_style_for_render,
    _finalize_negative_prompt,
    _build_repair_instruction,
    _JSON_REPAIR_INSTRUCTION,
)
from src.image_engine import GeneratedImage, HeuristicImageGenerator, ImageGenerator
from src.visual_engine import VisualPlan, VisualScene
from src.script_engine import Dialogue, Scene, SceneDescription, Script, ScriptScene
from src.brand_engine import BrandProfile, JsonBrandStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def brand() -> BrandProfile:
    store = JsonBrandStore(Path(__file__).resolve().parent.parent / "brands")
    profile = store.load("ia_fr")
    assert profile is not None, "Brand ia_fr doit exister"
    return profile


@pytest.fixture
def script_scene() -> ScriptScene:
    return ScriptScene(
        scene=Scene(
            number=1,
            type="hook",
            description=SceneDescription(
                setting="Plan choc sur une interface d'edition video futuriste",
                composition="Plan serre, sujet centre",
                characters="Aucun personnage visible, juste l'interface",
                lighting="Lumiere froide bleutee",
                camera="Zoom rapide avant",
                mood="Tension, futuriste",
                symbolism="La machine qui prend le controle",
                director_notes="Insister sur le contraste sombre/lumineux",
                viewer_emotion="Surprise, curiosite",
            ),
        ),
        dialogues=[
            Dialogue(
                personnage="NARRATEUR",
                replique="Et si votre prochain outil IA remplacait votre monteur video ?",
            )
        ],
        transition="Fade-in rapide",
        duration_seconds=8,
    )


@pytest.fixture
def second_scene() -> ScriptScene:
    return ScriptScene(
        scene=Scene(
            number=2,
            type="context",
            description=SceneDescription(
                setting="Sarah Chen face a l'ecran, stupefaite",
                composition="Plan moyen, sujet centre-gauche",
                characters="Sarah Chen, journaliste",
                lighting="Lumiere douce de l'ecran",
                camera="Zoom lent",
                mood="Etonnement",
                symbolism="La prise de conscience du changement",
                director_notes="Capturer l'expression du visage",
                viewer_emotion="Empathie, curiosite",
            ),
        ),
        dialogues=[
            Dialogue(
                personnage="NARRATEUR",
                replique="La journaliste Sarah Chen decouvre l'ampleur du changement.",
            )
        ],
        transition="Zoom lent",
        duration_seconds=9,
    )


@pytest.fixture
def visual_scene() -> VisualScene:
    return VisualScene(
        scene_order=1, shot_type="close_up", camera_motion="static",
        visual_prompt="close up on a glowing screen",
        composition="Rule of thirds, subject centered-left",
        lighting="Dramatic rim lighting",
        color_palette=["#0A0A23", "#00F5FF"],
        transition="cut", overlay_text="", animation_notes="Fade-in",
        duration_seconds=8,
    )


@pytest.fixture
def sample_script(script_scene, second_scene) -> Script:
    return Script(
        title="L'IA qui remplace les monteurs video",
        scenes=[script_scene, second_scene],
        estimated_duration=17,
        language="fr",
        target_audience="Createurs de contenu",
        style="Innovant",
        metadata={"generator": "llm_v1"},
    )


@pytest.fixture
def valid_llm_json():
    return {
        "goal": "Faire comprendre que l'IA prend le controle du montage video",
        "emotion": "tension, futuriste, innovant",
        "characters": ["Sarah Chen : journaliste, la trentaine, blazer rouge, cheveux courts noirs, lunettes"],
        "subject": "Sarah Chen, journaliste, blazer rouge, cheveux courts noirs, lunettes",
        "scene_description": (
            "A dark futuristic post-production studio, glowing holographic interface reflecting "
            "on the walls, cool blue rim lighting, shallow depth of field, subtle film grain, "
            "the room feels tense and pivotal in the story."
        ),
        "style": (
            "Arcane character design, painterly stylized illustration, cinematic AI animation, "
            "vertical 9:16 aspect ratio, ultra-detailed, HDR, 8K resolution"
        ),
        "prompt": "Generate an image where Sarah Chen stares at the glowing screen, realizing the scale of the change.",
        "negative_prompt": "blurry, low quality, distorted, extra limbs, watermark",
        "appearance": "trentaine, cheveux courts noirs, lunettes",
        "clothing": "blazer rouge",
        "accessories": "lunettes",
        "pose": "debout, face a l'ecran",
        "facial_expression": "stupefaction",
        "weather": "N/A",
        "time_of_day": "nuit",
        "background": "ecrans holographiques en arriere-plan",
    }


# ── Tests : création ───────────────────────────────────────────────────────────

class TestCreation:
    def test_default_creation(self):
        gen = LLMImageGenerator()
        assert gen is not None
        assert "llm_image" in gen.name
        assert gen.stats["llm_calls"] == 0

    def test_custom_provider(self):
        gen = LLMImageGenerator(provider_name="claude")
        assert "claude" in gen.name

    def test_is_image_generator(self):
        assert isinstance(LLMImageGenerator(), ImageGenerator)

    def test_default_fallback_is_heuristic(self):
        gen = LLMImageGenerator()
        assert isinstance(gen._fallback, HeuristicImageGenerator)

    def test_stats_immutable_copy(self):
        gen = LLMImageGenerator()
        stats = gen.stats
        stats["llm_calls"] = 999
        assert gen.stats["llm_calls"] == 0

    def test_characters_bible_starts_empty(self):
        gen = LLMImageGenerator()
        assert gen.characters_bible == {}

    def test_characters_bible_immutable_copy(self):
        gen = LLMImageGenerator()
        bible = gen.characters_bible
        bible["Fake"] = "should not persist"
        assert gen.characters_bible == {}


# ── Tests : build_user_prompt ──────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_prompt_contains_script_scene(self, script_scene, visual_scene, brand):
        gen = LLMImageGenerator()
        prompt = gen._build_user_prompt(None, script_scene, visual_scene, brand)
        assert script_scene.narration_text in prompt
        assert script_scene.scene.description.setting in prompt

    def test_prompt_contains_visual_scene(self, script_scene, visual_scene, brand):
        gen = LLMImageGenerator()
        prompt = gen._build_user_prompt(None, script_scene, visual_scene, brand)
        assert visual_scene.composition in prompt
        assert visual_scene.lighting in prompt
        assert "#0A0A23" in prompt

    def test_prompt_contains_brand(self, script_scene, visual_scene, brand):
        gen = LLMImageGenerator()
        prompt = gen._build_user_prompt(None, script_scene, visual_scene, brand)
        assert brand.name in prompt
        assert brand.tone in prompt

    def test_no_continuity_block_without_script(self, script_scene, visual_scene, brand):
        gen = LLMImageGenerator()
        prompt = gen._build_user_prompt(None, script_scene, visual_scene, brand)
        assert "NARRATIVE CONTINUITY" not in prompt

    def test_continuity_block_lists_all_scenes_in_order(self, sample_script, script_scene, visual_scene, brand):
        gen = LLMImageGenerator()
        prompt = gen._build_user_prompt(sample_script, script_scene, visual_scene, brand)
        assert "NARRATIVE CONTINUITY" in prompt
        assert sample_script.scenes[0].narration_text in prompt
        assert sample_script.scenes[1].narration_text in prompt
        assert "SCENE ACTUELLE" in prompt

    def test_continuity_block_includes_established_characters(self, sample_script, script_scene, visual_scene, brand):
        gen = LLMImageGenerator()
        gen._characters_bible["Sarah Chen"] = "journaliste, blazer rouge, cheveux courts noirs"
        prompt = gen._build_user_prompt(sample_script, script_scene, visual_scene, brand)
        assert "CHARACTERS ALREADY ESTABLISHED" in prompt
        assert "Sarah Chen" in prompt
        assert "blazer rouge" in prompt


# ── Tests : extract_json ────────────────────────────────────────────────────────

class TestExtractJson:
    def test_raw_json(self):
        text = '{"prompt": "x"}'
        assert LLMImageGenerator._extract_json(text) == text

    def test_markdown_block(self):
        text = '```json\n{"prompt": "x"}\n```'
        assert LLMImageGenerator._extract_json(text) == '{"prompt": "x"}'

    def test_text_around_json(self):
        text = 'Voici:\n{"prompt": "x"}\nfin.'
        assert LLMImageGenerator._extract_json(text) == '{"prompt": "x"}'


# ── Tests : robustesse de l'extraction JSON (Sprint 24.5) ─────────────────────

class TestExtractJsonRobustness:
    def test_strips_think_tags(self):
        text = "<think>je reflechis a la meilleure scene...</think>{\"a\": 1}"
        assert LLMImageGenerator._extract_json(text) == '{"a": 1}'

    def test_strips_think_tags_case_insensitive_multiline(self):
        text = "<THINK>\nligne 1\nligne 2\n</THINK>\n{\"a\": 1}"
        assert LLMImageGenerator._extract_json(text) == '{"a": 1}'

    def test_text_before_and_after_with_code_fence(self):
        text = 'Voici le resultat:\n```json\n{"a": 1}\n```\nMerci !'
        assert LLMImageGenerator._extract_json(text) == '{"a": 1}'

    def test_isolates_first_balanced_object_amid_garbage(self):
        text = 'blah blah {"a": {"nested": 1}} blah {"b": 2}'
        assert LLMImageGenerator._extract_json(text) == '{"a": {"nested": 1}}'

    def test_braces_inside_string_values_do_not_break_balance(self):
        text = 'noise {"a": "contains a { brace } inside a string"} noise'
        result = LLMImageGenerator._extract_json(text)
        json.loads(result)  # ne doit pas lever

    def test_removes_trailing_comma(self):
        text = '{"a": 1, "b": [1,2,],}'
        result = LLMImageGenerator._extract_json(text)
        assert json.loads(result) == {"a": 1, "b": [1, 2]}

    def test_removes_control_characters(self):
        text = '{"a": "x\x07\x01y"}'
        result = LLMImageGenerator._extract_json(text)
        assert json.loads(result) == {"a": "xy"}

    def test_removes_line_comments(self):
        text = '{\n// ceci est un commentaire\n"a": 1\n}'
        result = LLMImageGenerator._extract_json(text)
        assert json.loads(result) == {"a": 1}

    def test_removes_block_comments(self):
        text = '{"a": 1, /* commentaire */ "b": 2}'
        result = LLMImageGenerator._extract_json(text)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_unbalanced_json_still_returned_for_downstream_classification(self):
        """Un JSON tronqué (max_tokens atteint) doit rester détectable comme tel, pas planter ici."""
        text = '{"a": "unterminated'
        result = LLMImageGenerator._extract_json(text)
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    def test_no_json_at_all_returns_empty_or_unparsable(self):
        text = "Desole, je ne peux pas repondre a cette demande."
        result = LLMImageGenerator._extract_json(text)
        assert result == "" or result == text.strip()


# ── Tests : validate_json_structure ───────────────────────────────────────────

class TestValidateJsonStructure:
    def test_valid_json(self, valid_llm_json):
        LLMImageGenerator._validate_json_structure(valid_llm_json)

    def test_all_eight_fields_required(self):
        """Sprint 24.1, étendu Sprint 34.6 : contrat ImagePrompt — champs texte
        + 'characters' (liste) + 8 champs granulaires additionnels."""
        assert set(_REQUIRED_FIELDS) == {
            "goal", "emotion", "characters",
            "subject", "scene_description", "style", "prompt", "negative_prompt",
            "appearance", "clothing", "accessories", "pose", "facial_expression",
            "weather", "time_of_day", "background",
        }
        assert set(_REQUIRED_STRING_FIELDS) == {
            "goal", "emotion", "subject", "scene_description", "style", "prompt", "negative_prompt",
            "appearance", "clothing", "accessories", "pose", "facial_expression",
            "weather", "time_of_day", "background",
        }
        assert _REQUIRED_LIST_FIELDS == ("characters",)

    def test_reasoning_fields_come_before_contract_fields(self):
        assert _REQUIRED_STRING_FIELDS.index("goal") < _REQUIRED_STRING_FIELDS.index("subject")
        assert _REQUIRED_STRING_FIELDS.index("emotion") < _REQUIRED_STRING_FIELDS.index("subject")
        assert _REQUIRED_STRING_FIELDS.index("subject") < _REQUIRED_STRING_FIELDS.index("prompt")
        assert _REQUIRED_STRING_FIELDS.index("prompt") < _REQUIRED_STRING_FIELDS.index("negative_prompt")

    def test_missing_string_field(self, valid_llm_json):
        data = dict(valid_llm_json)
        del data["scene_description"]
        with pytest.raises(ValueError, match="scene_description"):
            LLMImageGenerator._validate_json_structure(data)

    def test_empty_string_field(self, valid_llm_json):
        data = dict(valid_llm_json)
        data["style"] = "   "
        with pytest.raises(ValueError, match="style"):
            LLMImageGenerator._validate_json_structure(data)

    def test_non_string_field(self, valid_llm_json):
        data = dict(valid_llm_json)
        data["prompt"] = 42
        with pytest.raises(ValueError, match="prompt"):
            LLMImageGenerator._validate_json_structure(data)

    def test_missing_characters_field(self, valid_llm_json):
        data = dict(valid_llm_json)
        del data["characters"]
        with pytest.raises(ValueError, match="characters"):
            LLMImageGenerator._validate_json_structure(data)

    def test_characters_must_be_a_list(self, valid_llm_json):
        data = dict(valid_llm_json)
        data["characters"] = "Sarah Chen"
        with pytest.raises(ValueError, match="characters"):
            LLMImageGenerator._validate_json_structure(data)

    def test_empty_characters_list_is_valid(self, valid_llm_json):
        data = dict(valid_llm_json)
        data["characters"] = []
        LLMImageGenerator._validate_json_structure(data)

    def test_characters_list_of_dicts_is_rejected(self, valid_llm_json):
        """
        Régression : DeepSeek répond parfois avec des objets structurés
        (ex: {"name": "...", "age": "..."}) au lieu de descriptions textuelles
        — cela passait la validation (liste OK) puis faisait planter les
        consommateurs en aval (join() sur des dicts). Doit maintenant être
        rejeté ici pour déclencher la correction JSON intelligente.
        """
        data = dict(valid_llm_json)
        data["characters"] = [{"name": "Sarah Chen", "age": "30"}]
        with pytest.raises(ValueError, match="characters"):
            LLMImageGenerator._validate_json_structure(data)


# ── Tests : parse_and_validate — classification des causes d'échec (Sprint 24.5) ──

class _FakeLlmResponse:
    def __init__(self, content, finish_reason="stop"):
        self.content = content
        self.finish_reason = finish_reason


class TestParseAndValidate:
    def test_success_returns_data(self, valid_llm_json):
        data = LLMImageGenerator._parse_and_validate(_FakeLlmResponse(json.dumps(valid_llm_json)))
        assert data["subject"] == valid_llm_json["subject"]

    def test_empty_response_reason(self):
        with pytest.raises(_ImageJsonError) as exc_info:
            LLMImageGenerator._parse_and_validate(_FakeLlmResponse(""))
        assert exc_info.value.reason == "empty_response"

    def test_whitespace_only_response_is_empty(self):
        with pytest.raises(_ImageJsonError) as exc_info:
            LLMImageGenerator._parse_and_validate(_FakeLlmResponse("   \n  "))
        assert exc_info.value.reason == "empty_response"

    def test_invalid_json_reason(self):
        with pytest.raises(_ImageJsonError) as exc_info:
            LLMImageGenerator._parse_and_validate(_FakeLlmResponse("Desole, je ne peux pas aider."))
        assert exc_info.value.reason == "json_invalid"

    def test_truncated_due_to_length_is_json_incomplete(self):
        with pytest.raises(_ImageJsonError) as exc_info:
            LLMImageGenerator._parse_and_validate(
                _FakeLlmResponse('{"subject": "a truncated str', finish_reason="length")
            )
        assert exc_info.value.reason == "json_incomplete"

    def test_valid_json_but_missing_field_is_validation_failed(self, valid_llm_json):
        data = dict(valid_llm_json)
        del data["style"]
        with pytest.raises(_ImageJsonError) as exc_info:
            LLMImageGenerator._parse_and_validate(_FakeLlmResponse(json.dumps(data)))
        assert exc_info.value.reason == "validation_failed"

    def test_extracts_json_despite_think_tags_and_prose(self, valid_llm_json):
        content = f"<think>je reflechis...</think>Voici le resultat :\n```json\n{json.dumps(valid_llm_json)}\n```\nVoila !"
        data = LLMImageGenerator._parse_and_validate(_FakeLlmResponse(content))
        assert data["subject"] == valid_llm_json["subject"]


# ── Tests : garanties déterministes de rendu ──────────────────────────────────

class TestFinalizeStyleForRender:
    def test_adds_missing_format_and_quality(self):
        result = _finalize_style_for_render("moody scene")
        low = result.lower()
        assert "9:16" in low
        assert "hdr" in low or "8k" in low
        assert "arcane" in low or "painterly" in low
        assert "cinematic" in low

    def test_does_not_duplicate_present_markers(self):
        style = (
            "Arcane character design, painterly stylized illustration, cinematic, "
            "vertical 9:16 aspect ratio, ultra-detailed, HDR, 8K resolution."
        )
        result = _finalize_style_for_render(style)
        assert result == style

    def test_case_insensitive_detection(self):
        style = "ARCANE, PAINTERLY, CINEMATIC, VERTICAL 9:16, ULTRA-DETAILED HDR 8K"
        result = _finalize_style_for_render(style)
        assert result == style

    def test_partial_markers_only_missing_appended(self):
        style = "vertical 9:16 aspect ratio, portrait orientation"
        result = _finalize_style_for_render(style)
        low = result.lower()
        assert low.count("9:16") == 1
        assert "hdr" in low
        assert "arcane" in low or "painterly" in low
        assert "cinematic" in low

    def test_never_forces_photorealistic(self):
        """Sprint 37.2 — la garantie de rendu ne doit plus jamais imposer
        un registre photoréaliste, en contradiction avec le style de marque
        (Arcane / Lord of Mysteries, painterly stylisé)."""
        result = _finalize_style_for_render("simple scene")
        assert "photorealistic" not in result.lower()


class TestFinalizeNegativePrompt:
    def test_adds_missing_baseline_terms(self):
        result = _finalize_negative_prompt("bad hands")
        low = result.lower()
        assert "bad hands" in low
        assert "watermark" in low
        assert "text" in low
        assert "photorealistic" in low

    def test_does_not_duplicate_present_terms(self):
        from src.llm_image_generator import _NEGATIVE_PROMPT_BASELINE
        full = ", ".join(_NEGATIVE_PROMPT_BASELINE)
        result = _finalize_negative_prompt(full)
        assert result == full

    def test_never_bans_illustration_or_painting(self):
        """Sprint 37.2 — le style de marque EST une illustration peinte
        stylisée : bannir "cartoon/illustration/painting" contredisait le
        style voulu et poussait vers le photoréalisme."""
        result = _finalize_negative_prompt("bad hands").lower()
        assert "illustration" not in result
        assert "painting" not in result
        assert "cartoon" not in result


# ── Tests : contrat ImagePrompt (Sprint 24.1) ─────────────────────────────────

class _FakeResponse:
    provider_name = "deepseek"
    model = "deepseek-chat"
    total_tokens = 300
    cost_usd = 0.001


class TestBuildImagePrompt:
    def test_reconstruction(self, valid_llm_json):
        image_prompt = LLMImageGenerator._build_image_prompt(valid_llm_json, _FakeResponse(), elapsed_ms=1200)
        assert isinstance(image_prompt, ImagePrompt)
        assert image_prompt.subject == valid_llm_json["subject"]
        assert image_prompt.scene_description == valid_llm_json["scene_description"]
        # Le style de la fixture couvre déjà tous les marqueurs de rendu → inchangé
        assert image_prompt.style == valid_llm_json["style"]
        assert image_prompt.prompt == valid_llm_json["prompt"]
        assert valid_llm_json["negative_prompt"] in image_prompt.negative_prompt

    def test_metadata_shape_is_exact(self, valid_llm_json):
        """Sprint 34.6 : le contrat metadata inclut désormais aussi les 8
        champs granulaires additionnels (appearance, clothing, ...)."""
        image_prompt = LLMImageGenerator._build_image_prompt(valid_llm_json, _FakeResponse(), elapsed_ms=1200)
        assert set(image_prompt.metadata.keys()) == {
            "goal", "emotion", "characters", "provider", "model", "time_ms", "cost_usd",
            "appearance", "clothing", "accessories", "pose", "facial_expression",
            "weather", "time_of_day", "background",
        }
        assert image_prompt.metadata["goal"] == valid_llm_json["goal"]
        assert image_prompt.metadata["emotion"] == valid_llm_json["emotion"]
        assert image_prompt.metadata["characters"] == valid_llm_json["characters"]
        assert image_prompt.metadata["appearance"] == valid_llm_json["appearance"]
        assert image_prompt.metadata["background"] == valid_llm_json["background"]
        assert image_prompt.metadata["provider"] == "deepseek"
        assert image_prompt.metadata["model"] == "deepseek-chat"
        assert image_prompt.metadata["time_ms"] == 1200
        assert image_prompt.metadata["cost_usd"] == 0.001

    def test_top_level_contract_shape_is_exact(self, valid_llm_json):
        """Sprint 24.4 : chaque scène doit contenir exactement ces 6 clés au niveau racine."""
        import dataclasses
        image_prompt = LLMImageGenerator._build_image_prompt(valid_llm_json, _FakeResponse(), elapsed_ms=0)
        assert set(dataclasses.asdict(image_prompt).keys()) == {
            "subject", "scene_description", "style", "prompt", "negative_prompt", "metadata",
        }


class TestFromGeneratedImage:
    def test_wraps_heuristic_result(self, visual_scene, brand):
        plan = VisualPlan(title="T", style=brand.visual_style, scenes=[visual_scene])
        generated = HeuristicImageGenerator().generate(visual_scene, plan)
        image_prompt = LLMImageGenerator._from_generated_image(generated)
        assert isinstance(image_prompt, ImagePrompt)
        assert image_prompt.scene_description == generated.prompt
        assert image_prompt.negative_prompt == generated.negative_prompt
        assert image_prompt.metadata["provider"] == "heuristic_image_v1"
        assert image_prompt.metadata["characters"] == []

    def test_prompt_field_is_never_empty(self, visual_scene, brand):
        """Sprint 29.1 — un ImagePrompt de secours doit rester exploitable en
        production : le champ `prompt` ne doit jamais être vide."""
        plan = VisualPlan(title="T", style=brand.visual_style, scenes=[visual_scene])
        generated = HeuristicImageGenerator().generate(visual_scene, plan)
        image_prompt = LLMImageGenerator._from_generated_image(generated, reason="validation_failed")
        assert image_prompt.prompt
        assert image_prompt.prompt == generated.prompt

    def test_carries_fallback_detail(self, visual_scene, brand):
        plan = VisualPlan(title="T", style=brand.visual_style, scenes=[visual_scene])
        generated = HeuristicImageGenerator().generate(visual_scene, plan)
        image_prompt = LLMImageGenerator._from_generated_image(
            generated, reason="validation_failed", detail="Champ obligatoire manquant : 'prompt'",
        )
        assert image_prompt.metadata["fallback_reason"] == "validation_failed"
        assert image_prompt.metadata["fallback_detail"] == "Champ obligatoire manquant : 'prompt'"


class TestToGeneratedImage:
    def test_reconstruction(self, valid_llm_json, visual_scene, brand):
        image_prompt = LLMImageGenerator._build_image_prompt(valid_llm_json, _FakeResponse(), elapsed_ms=0)
        image = LLMImageGenerator._to_generated_image(image_prompt, visual_scene, brand)
        assert isinstance(image, GeneratedImage)
        assert image.scene_order == visual_scene.scene_order
        assert image_prompt.subject in image.prompt
        assert image_prompt.scene_description in image.prompt
        assert image_prompt.prompt in image.prompt
        assert image.negative_prompt == image_prompt.negative_prompt
        assert image.style == image_prompt.style

    def test_dimensions_reuse_heuristic_helpers(self, valid_llm_json, visual_scene, brand):
        image_prompt = LLMImageGenerator._build_image_prompt(valid_llm_json, _FakeResponse(), elapsed_ms=0)
        image = LLMImageGenerator._to_generated_image(image_prompt, visual_scene, brand)
        plan = VisualPlan(title="", style=brand.visual_style, scenes=[visual_scene])
        expected_ratio = HeuristicImageGenerator._resolve_aspect_ratio(plan.aspect_ratio, visual_scene)
        expected_w, expected_h = HeuristicImageGenerator._resolve_dimensions(expected_ratio, visual_scene)
        assert image.aspect_ratio == expected_ratio
        assert (image.width, image.height) == (expected_w, expected_h)

    def test_deterministic_seed(self, valid_llm_json, visual_scene, brand):
        image_prompt = LLMImageGenerator._build_image_prompt(valid_llm_json, _FakeResponse(), elapsed_ms=0)
        img1 = LLMImageGenerator._to_generated_image(image_prompt, visual_scene, brand)
        img2 = LLMImageGenerator._to_generated_image(image_prompt, visual_scene, brand)
        assert img1.seed == img2.seed


# ── Tests : generate_from_scenes (fallback sans clé API) ──────────────────────

class TestGenerateFromScenesFallback:
    def test_falls_back_to_heuristic(self, script_scene, visual_scene, brand):
        """
        Sans clé API dans l'environnement de test, toutes les tentatives LLM
        échouent — generate_from_scenes() ne doit jamais lever d'exception,
        et doit toujours retourner un ImagePrompt (contrat uniforme).
        """
        gen = LLMImageGenerator(max_retries=1)
        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)
        assert isinstance(image_prompt, ImagePrompt)
        assert image_prompt.metadata["provider"] == "heuristic_image_v1"
        assert image_prompt.metadata["fallback_reason"] in ("api_error", "timeout")
        assert gen.stats["fallbacks"] == 1
        assert gen.stats["llm_failures"] >= 1
        assert gen.stats["fallback_reasons"]

    def test_fallback_image_prompt_is_still_valid(self, script_scene, visual_scene, brand):
        gen = LLMImageGenerator(max_retries=1)
        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)
        assert image_prompt.scene_description
        assert isinstance(image_prompt.metadata["characters"], list)

    def test_script_param_defaults_to_constructor_script(self, sample_script, script_scene, visual_scene, brand):
        """Si `script=` n'est pas passé à l'appel, self._script (constructeur) est utilisé."""
        gen = LLMImageGenerator(script=sample_script, max_retries=1)
        # Ne doit pas lever — le script constructeur alimente la continuité même
        # en mode dégradé (fallback heuristique, pas de clé API en test).
        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)
        assert isinstance(image_prompt, ImagePrompt)


# ── Tests : retry intelligent — correction JSON avant fallback (Sprint 24.5) ──

def _make_llm_response(content, finish_reason="stop", model="deepseek-chat"):
    return LLMResponse(
        content=content, model=model, provider_name="deepseek",
        finish_reason=finish_reason, prompt_tokens=10, completion_tokens=10,
        total_tokens=20, time_ms=5, cost_usd=0.0001,
    )


class _ScriptedProvider:
    """Provider fake retournant des réponses scriptées, dans l'ordre — pour
    tester le flux de retry intelligent sans appel réseau réel."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.name = "deepseek"
        self.model = "deepseek-chat"
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


class TestIntelligentRetry:
    def test_recovers_via_repair_retry(self, script_scene, visual_scene, brand, valid_llm_json):
        """Un premier JSON invalide est corrigé par un second appel — pas de fallback."""
        gen = LLMImageGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response('Sure! Here you go: {"subject": "oops, truncated'),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])

        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)

        assert isinstance(image_prompt, ImagePrompt)
        assert image_prompt.subject == valid_llm_json["subject"]
        assert gen._provider.calls == 2
        assert gen.stats["json_repair_attempts"] == 1
        assert gen.stats["json_repairs_success"] == 1
        assert gen.stats["fallbacks"] == 0
        assert gen.stats["llm_success"] == 1

    def test_repair_prompt_asks_to_fix_only_the_json(self, script_scene, visual_scene, brand, valid_llm_json):
        gen = LLMImageGenerator(max_retries=1)
        provider = _ScriptedProvider([
            _make_llm_response("not json at all"),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])
        gen._provider = provider

        captured_messages = []
        original_generate = provider.generate

        def spy_generate(messages, **kwargs):
            captured_messages.append(messages)
            return original_generate(messages, **kwargs)

        provider.generate = spy_generate
        gen.generate_from_scenes(script_scene, visual_scene, brand)

        repair_call_messages = captured_messages[1]
        assert repair_call_messages[-1].role == "user"
        assert "Corrige" in repair_call_messages[-1].content
        assert "JSON" in repair_call_messages[-1].content

    def test_falls_back_with_reason_when_repair_also_fails(self, script_scene, visual_scene, brand):
        gen = LLMImageGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("not json at all"),
            _make_llm_response("still not json either"),
        ])

        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)

        assert image_prompt.metadata["provider"] == "heuristic_image_v1"
        assert image_prompt.metadata["fallback_reason"] in ("json_invalid", "json_incomplete")
        assert gen.stats["fallbacks"] == 1
        assert gen.stats["json_repair_attempts"] == 1
        assert gen.stats["json_repairs_success"] == 0
        assert sum(gen.stats["fallback_reasons"].values()) == 1

    def test_api_error_skips_repair_entirely(self, script_scene, visual_scene, brand):
        """Une erreur API/timeout n'est pas un problème de format JSON — pas de tentative de correction."""
        gen = LLMImageGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("[DeepSeek API Error: connection refused]", finish_reason="error"),
        ])

        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)

        assert image_prompt.metadata["fallback_reason"] == "api_error"
        assert gen.stats["json_repair_attempts"] == 0
        assert gen._provider.calls == 1

    def test_timeout_response_classified_as_timeout(self, script_scene, visual_scene, brand):
        gen = LLMImageGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("[DeepSeek API Error: Read timeout]", finish_reason="error"),
        ])

        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)

        assert image_prompt.metadata["fallback_reason"] == "timeout"

    def test_validation_failure_carries_precise_detail(self, script_scene, visual_scene, brand):
        """Sprint 29.1 — 'validation_failed' seul ne suffit pas : le champ
        manquant/invalide précis doit être exposé dans fallback_detail."""
        incomplete_json = json.dumps({"subject": "a hero"})  # champs requis manquants
        gen = LLMImageGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response(incomplete_json),  # 1ère tentative
            _make_llm_response(incomplete_json),  # tentative de correction JSON — échoue aussi
        ])

        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)

        assert image_prompt.metadata["fallback_reason"] == "validation_failed"
        assert "Champ obligatoire manquant" in image_prompt.metadata["fallback_detail"]

    def test_repair_updates_characters_bible(self, script_scene, visual_scene, brand, valid_llm_json):
        gen = LLMImageGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("garbage"),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])

        gen.generate_from_scenes(script_scene, visual_scene, brand)

        assert gen.characters_bible  # le personnage de valid_llm_json a bien été verrouillé


# ── Tests : réparation sémantique du champ "characters" (Sprint 30.2) ────────
# Régression du bug confirmé par l'audit du run GitHub Actions réel (#5,
# 29120900792) : 10/10 fallbacks observés venaient de "characters" renvoyé
# comme une liste d'objets structurés au lieu d'une liste de chaînes, et le
# message de réparation générique ("Corrige UNIQUEMENT le JSON") ne corrigeait
# jamais ce cas car le JSON était déjà syntaxiquement valide (0/10 réparations
# réussies). Ces tests verrouillent : (1) le validateur continue de rejeter
# la structure fautive, (2) l'instruction de réparation cible maintenant ce
# problème sémantique précis, (3) le flux complet se rétablit sans fallback
# quand le second appel corrige correctement le format.

_CHARACTERS_AS_OBJECTS = json.dumps({
    "hair": "Long black braids",
    "clothes": "Blue ceremonial robe",
})


class TestBuildRepairInstruction:
    def test_validation_failed_gets_semantic_instruction(self):
        error = _ImageJsonError(
            "validation_failed",
            "Le champ 'characters' doit être une liste de chaînes "
            "(le LLM a renvoyé des objets structurés au lieu de descriptions textuelles)",
        )
        instruction = _build_repair_instruction(error)
        assert "array of plain strings" in instruction
        assert "characters" in instruction
        assert "descriptions textuelles" in instruction  # message de validation réel transmis
        assert instruction != _JSON_REPAIR_INSTRUCTION

    def test_non_validation_reason_keeps_generic_syntax_instruction(self):
        error = _ImageJsonError("json_invalid", "Expecting value: line 1 column 1 (char 0)")
        instruction = _build_repair_instruction(error)
        assert instruction == _JSON_REPAIR_INSTRUCTION


class TestCharactersObjectRegression:
    """Reproduit exactement le bug de production (Sprint 30.1/30.2)."""

    def test_validator_still_rejects_character_objects(self, valid_llm_json):
        """Le validateur n'a pas été assoupli (contrainte du sprint) — il
        continue de rejeter des objets structurés dans 'characters'."""
        data = dict(valid_llm_json)
        data["characters"] = [json.loads(_CHARACTERS_AS_OBJECTS)]
        with pytest.raises(ValueError, match="characters"):
            LLMImageGenerator._validate_json_structure(data)

    def test_repair_prompt_targets_characters_when_that_is_the_cause(
        self, script_scene, visual_scene, brand, valid_llm_json,
    ):
        """Quand le premier échec est bien 'characters' en objets, le message
        de réparation envoyé au LLM doit nommer précisément ce problème —
        pas l'instruction générique de syntaxe JSON."""
        first_attempt = dict(valid_llm_json)
        first_attempt["characters"] = [json.loads(_CHARACTERS_AS_OBJECTS)]

        provider = _ScriptedProvider([
            _make_llm_response(json.dumps(first_attempt)),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])
        gen = LLMImageGenerator(max_retries=1)
        gen._provider = provider

        captured_messages = []
        original_generate = provider.generate

        def spy_generate(messages, **kwargs):
            captured_messages.append(messages)
            return original_generate(messages, **kwargs)

        provider.generate = spy_generate
        gen.generate_from_scenes(script_scene, visual_scene, brand)

        repair_call_messages = captured_messages[1]
        repair_instruction = repair_call_messages[-1].content
        assert "array of plain strings" in repair_instruction
        assert "characters" in repair_instruction
        assert "Corrige UNIQUEMENT le JSON" not in repair_instruction

    def test_end_to_end_recovers_without_fallback_when_repair_flattens_characters(
        self, script_scene, visual_scene, brand, valid_llm_json,
    ):
        """Bout en bout : première réponse avec des objets dans 'characters',
        seconde réponse (corrigée) avec des chaînes — le résultat final doit
        respecter le contrat List[str], sans jamais tomber en fallback."""
        first_attempt = dict(valid_llm_json)
        first_attempt["characters"] = [json.loads(_CHARACTERS_AS_OBJECTS)]

        gen = LLMImageGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response(json.dumps(first_attempt)),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])

        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)

        assert image_prompt.metadata["provider"] != "heuristic_image_v1"
        assert gen.stats["fallbacks"] == 0
        assert gen.stats["json_repair_attempts"] == 1
        assert gen.stats["json_repairs_success"] == 1
        characters = image_prompt.metadata["characters"]
        assert isinstance(characters, list)
        assert all(isinstance(item, str) for item in characters)

    def test_falls_back_if_second_attempt_still_returns_objects(
        self, script_scene, visual_scene, brand, valid_llm_json,
    ):
        """Si la réparation échoue elle aussi, le fallback heuristique reste
        le filet de sécurité — aucune régression sur ce comportement."""
        broken = dict(valid_llm_json)
        broken["characters"] = [json.loads(_CHARACTERS_AS_OBJECTS)]

        gen = LLMImageGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response(json.dumps(broken)),
            _make_llm_response(json.dumps(broken)),
        ])

        image_prompt = gen.generate_from_scenes(script_scene, visual_scene, brand)

        assert image_prompt.metadata["provider"] == "heuristic_image_v1"
        assert image_prompt.metadata["fallback_reason"] == "validation_failed"
        assert "characters" in image_prompt.metadata["fallback_detail"]


# ── Tests : generate(scene, plan) — interface ImageGenerator ──────────────────

class TestGenerateInterface:
    def test_generate_without_context_uses_fallback(self, visual_scene, brand):
        """Sans script/brand injectés au constructeur, generate() bascule direct sur le fallback."""
        gen = LLMImageGenerator()
        plan = VisualPlan(title="T", style="default", scenes=[visual_scene])
        image = gen.generate(visual_scene, plan)
        assert isinstance(image, GeneratedImage)
        assert image.provider == "heuristic_image_v1"

    def test_generate_returns_generated_image(self, sample_script, visual_scene, brand, script_scene):
        """generate() reste ABC-compatible : retourne toujours un GeneratedImage, jamais un ImagePrompt."""
        gen = LLMImageGenerator(script=sample_script, brand_profile=brand, max_retries=1)
        plan = VisualPlan(title="T", style="default", scenes=[visual_scene])
        image = gen.generate(visual_scene, plan)
        assert isinstance(image, GeneratedImage)

    def test_find_script_scene_matches_by_order(self, sample_script, script_scene):
        gen = LLMImageGenerator(script=sample_script)
        found = gen._find_script_scene(script_scene.order)
        assert found is script_scene

    def test_find_script_scene_returns_none_if_no_script(self):
        gen = LLMImageGenerator()
        assert gen._find_script_scene(1) is None

    def test_find_script_scene_returns_none_if_order_missing(self, sample_script):
        gen = LLMImageGenerator(script=sample_script)
        assert gen._find_script_scene(999) is None


# ── Tests : bible de personnages / continuité (Sprint 24.3) ──────────────────

class TestCharactersBible:
    def test_new_character_is_added(self):
        gen = LLMImageGenerator()
        gen._update_characters_bible(["Sarah Chen : journaliste, blazer rouge"])
        assert gen.characters_bible == {"Sarah Chen": "journaliste, blazer rouge"}

    def test_existing_character_is_not_overwritten(self):
        gen = LLMImageGenerator()
        gen._update_characters_bible(["Sarah Chen : journaliste, blazer rouge"])
        gen._update_characters_bible(["Sarah Chen : DESCRIPTION DIFFERENTE"])
        assert gen.characters_bible["Sarah Chen"] == "journaliste, blazer rouge"

    def test_multiple_characters_accumulate(self):
        gen = LLMImageGenerator()
        gen._update_characters_bible(["Sarah Chen : journaliste"])
        gen._update_characters_bible(["Marc Dubois : scientifique, blouse blanche"])
        assert set(gen.characters_bible.keys()) == {"Sarah Chen", "Marc Dubois"}

    def test_ignores_malformed_entries(self):
        gen = LLMImageGenerator()
        gen._update_characters_bible(["", "   ", 42, None])
        assert gen.characters_bible == {}

    def test_entry_without_colon_uses_whole_string_as_key_and_desc(self):
        gen = LLMImageGenerator()
        gen._update_characters_bible(["Un homme mysterieux en costume noir"])
        assert gen.characters_bible == {"Un homme mysterieux en costume noir": "Un homme mysterieux en costume noir"}

    def test_generic_name_label_falls_back_to_whole_string_as_key(self):
        """
        Le LLM répond parfois par un label générique avant le premier ':'
        (ex: "Name: Young man. Description: ..."). Utiliser "Name" comme clé
        polluerait la bible — on retombe alors sur la chaîne entière.
        """
        gen = LLMImageGenerator()
        entry = "Name: Young man. Description: Mid-20s, messy dark hair, black hoodie."
        gen._update_characters_bible([entry])
        assert "Name" not in gen.characters_bible
        assert gen.characters_bible == {entry: entry}

    def test_generic_label_case_insensitive_french_and_english(self):
        gen = LLMImageGenerator()
        gen._update_characters_bible(["Nom : Sarah, journaliste"])
        assert "Nom" not in gen.characters_bible
        assert list(gen.characters_bible.keys()) == ["Nom : Sarah, journaliste"]

    def test_reset_continuity_clears_bible(self):
        gen = LLMImageGenerator()
        gen._update_characters_bible(["Sarah Chen : journaliste"])
        gen.reset_continuity()
        assert gen.characters_bible == {}


# ── Tests : Découplage ────────────────────────────────────────────────────────

class TestInterfacePreserved:
    def test_conserves_image_generator_interface(self):
        mod_src = Path(__file__).resolve().parent.parent / "src" / "llm_image_generator.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "from src.image_engine import" in content
        assert "ImageGenerator" in content
        assert "class LLMImageGenerator(ImageGenerator)" in content

    def test_generate_signature_matches_abc(self, visual_scene, brand):
        """generate(scene, plan) reste la signature de l'ABC ImageGenerator."""
        gen = LLMImageGenerator()
        plan = VisualPlan(title="T", style="default", scenes=[visual_scene])
        gen.generate(visual_scene, plan)


# ── Tests : résolution du modèle DeepSeek / mode Reasoning ────────────────────

class TestResolveModel:
    def test_explicit_model_always_wins(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
        gen = LLMImageGenerator(provider_name="deepseek", model="deepseek-chat")
        assert gen._resolve_model() == "deepseek-chat"

    def test_deepseek_provider_uses_configured_constant(self, monkeypatch):
        import src.llm_image_generator as mod
        monkeypatch.setattr(mod, "_DEEPSEEK_IMAGE_MODEL", "deepseek-reasoner")
        gen = LLMImageGenerator(provider_name="deepseek")
        assert gen._resolve_model() == "deepseek-reasoner"

    def test_env_var_overrides_default_model(self, monkeypatch):
        import src.llm_image_generator as mod
        monkeypatch.setattr(mod, "_DEEPSEEK_IMAGE_MODEL", "deepseek-chat")
        gen = LLMImageGenerator(provider_name="deepseek")
        assert gen._resolve_model() == "deepseek-chat"

    def test_source_default_is_deepseek_chat_not_reasoner(self):
        """
        Sprint 24.5 : deepseek-reasoner est réservé aux tâches de raisonnement
        (scripts, évaluation, réécriture) — les images utilisent deepseek-chat,
        plus fiable en json_mode strict. Vérifié sur le littéral du code source
        (indépendant de l'environnement réel, cf. DEEPSEEK_IMAGE_MODEL en .env).
        """
        src_path = Path(__file__).resolve().parent.parent / "src" / "llm_image_generator.py"
        content = src_path.read_text(encoding="utf-8")
        assert 'os.environ.get("DEEPSEEK_IMAGE_MODEL", "deepseek-chat")' in content

    def test_non_deepseek_provider_has_no_forced_model(self):
        gen = LLMImageGenerator(provider_name="claude")
        assert gen._resolve_model() is None

    def test_auto_detected_deepseek_via_env_key(self, monkeypatch):
        import src.llm_image_generator as mod
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
        gen = LLMImageGenerator()
        assert gen._resolve_model() == mod._DEEPSEEK_IMAGE_MODEL


class TestReasoningSystemPrompt:
    def test_reasoning_enabled_and_disabled_return_same_base_prompt(self):
        """
        Le prompt système (externalisé dans prompts/image_system_prompt.txt)
        couvre déjà le raisonnement étape par étape pour tous les modèles —
        _build_system_prompt() ne varie plus selon reasoning_enabled.
        """
        from src.llm_image_generator import _build_system_prompt
        assert _build_system_prompt(reasoning_enabled=True) == _build_system_prompt(reasoning_enabled=False)

    def test_schema_order_matches_reasoning_sequence(self):
        from src.llm_image_generator import _build_system_prompt
        prompt = _build_system_prompt(reasoning_enabled=True)
        assert prompt.index('"goal"') < prompt.index('"emotion"')
        assert prompt.index('"characters"') < prompt.index('"subject"')
        assert prompt.index('"subject"') < prompt.index('"prompt"')
        assert prompt.index('"prompt"') < prompt.index('"negative_prompt"')

    def test_mentions_art_director_persona(self):
        from src.llm_image_generator import _build_system_prompt
        prompt = _build_system_prompt(reasoning_enabled=True).lower()
        assert "art director" in prompt
        assert "prompt engineer" in prompt

    def test_mentions_field_separation_rules(self):
        from src.llm_image_generator import _build_system_prompt
        prompt = _build_system_prompt(reasoning_enabled=True)
        assert "subject" in prompt and "scene_description" in prompt and "style" in prompt

    def test_mentions_continuity_requirements(self):
        from src.llm_image_generator import _build_system_prompt
        prompt = _build_system_prompt(reasoning_enabled=True).upper()
        assert "CHARACTER CONSISTENCY" in prompt or "CONTINUITY" in prompt


class TestSupportsReasoningIntegration:
    def test_supports_reasoning_true_for_reasoner(self):
        from src.llm import supports_reasoning
        assert supports_reasoning("deepseek-reasoner") is True

    def test_supports_reasoning_false_for_chat(self):
        from src.llm import supports_reasoning
        assert supports_reasoning("deepseek-chat") is False
        assert supports_reasoning("gpt-4o") is False
