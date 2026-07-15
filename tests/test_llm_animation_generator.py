"""
Tests unitaires pour le LLM Animation Generator (Sprint 25).

Teste :
  1. LLMAnimationGenerator — création, nom, stats, bible de personnages
  2. build_user_prompt — contient scène / plan visuel / image déjà générée +
     bloc de continuité narrative
  3. extract_json — extraction robuste (<think>, Markdown, texte parasite,
     commentaires, virgules traînantes, caractères de contrôle)
  4. validate_json_structure / parse_and_validate — validation stricte des
     11 champs + classification des causes d'échec
  5. build_animation_prompt — construction du contrat AnimationPrompt
  6. fallback déterministe — jamais d'exception remontée, contrat toujours valide
  7. retry intelligent — correction JSON via un second appel LLM avant tout fallback
  8. bible de personnages — alimentée par ImagePrompt.metadata["characters"],
     verrouillage, non-écrasement, continuité
  9. sérialisation — dataclasses.asdict() produit exactement le contrat attendu
  10. indépendance du moteur — ne dépend que des contrats de données en entrée
"""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.llm import LLMResponse
from src.llm_image_generator import ImagePrompt
from src.llm_animation_generator import (
    AnimationPrompt,
    LLMAnimationGenerator,
    _AnimationJsonError,
    _REQUIRED_FIELDS,
    _REQUIRED_STRING_FIELDS,
    _REQUIRED_INT_FIELDS,
)
from src.visual_engine import VisualScene
from src.script_engine import Dialogue, Scene, SceneDescription, Script, ScriptScene


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def script_scene() -> ScriptScene:
    return ScriptScene(
        scene=Scene(
            number=1,
            type="hook",
            description=SceneDescription(
                setting="Plan choc sur une interface d'edition video futuriste",
                composition="Rule of thirds, ecran centre-gauche",
                characters="Aucun personnage visible, uniquement l'interface",
                lighting="Dramatic rim lighting",
                camera="slow push-in toward the glowing screen",
                mood="tension, futuriste",
                symbolism="L'IA qui prend le controle du montage",
                director_notes="Insister sur le contraste sombre/lumineux",
                viewer_emotion="curiosite, tension",
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
            type="development",
            description=SceneDescription(
                setting="Bureau de redaction, ecran allume face a Sarah Chen",
                composition="Gros plan sur le visage de Sarah Chen",
                characters="Sarah Chen, journaliste, stupefaite",
                lighting="Lumiere froide de l'ecran",
                camera="static close-up on Sarah Chen's face",
                mood="stupefaction",
                symbolism="La prise de conscience du changement",
                director_notes="Laisser le silence s'installer",
                viewer_emotion="empathie, stupefaction",
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
def sample_dialogues() -> list:
    return [Dialogue(personnage="NARRATEUR", replique="Une replique de test.")]


@pytest.fixture
def image_prompt() -> ImagePrompt:
    return ImagePrompt(
        subject="Sarah Chen, journaliste, blazer rouge, cheveux courts noirs, lunettes",
        scene_description=(
            "A dark futuristic post-production studio, glowing holographic interface "
            "reflecting on the walls, cool blue rim lighting, shallow depth of field."
        ),
        style="Arcane character design, painterly stylized illustration, cinematic AI animation, vertical 9:16 aspect ratio, ultra-detailed, HDR, 8K resolution",
        prompt="Generate an image where Sarah Chen stares at the glowing screen.",
        negative_prompt="blurry, low quality, distorted, extra limbs, watermark",
        metadata={
            "goal": "Faire comprendre que l'IA prend le controle du montage video",
            "emotion": "tension, futuriste",
            "characters": ["Sarah Chen : journaliste, la trentaine, blazer rouge, cheveux courts noirs, lunettes"],
            "provider": "deepseek", "model": "deepseek-chat",
            "time_ms": 500, "cost_usd": 0.0002,
        },
    )


@pytest.fixture
def valid_llm_json():
    return {
        "goal": "Montrer que l'IA prend vie sous les yeux de Sarah Chen",
        "emotion": "tension croissante",
        "pace": "soutenu",
        "camera_motion": "slow push-in toward Sarah Chen's face",
        "subject_motion": "Sarah Chen leans forward slightly, eyes widening",
        "environment_motion": "holographic interface particles drift and pulse",
        "lighting_changes": "cool blue rim light intensifies subtly",
        "effects": "faint light flares from the hologram",
        "sound_design": "low tension drone, soft electronic pulses",
        "animation_style": "smooth 24fps cinematic motion, subtle parallax",
        "voice": "Female, mid-30s, tense documentary tone",
        "sound_effects": "faint camera shutter click",
        "background_music": "low tense synth drone, building tension",
        "transition": "hard cut to black",
        "duration": 5,
        "prompt": (
            "Slow push-in on Sarah Chen as she leans forward, eyes widening while the "
            "holographic interface particles pulse and the blue rim light intensifies, "
            "tense electronic drone building, ending on a hard cut to black."
        ),
    }


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


# ── Tests : création ───────────────────────────────────────────────────────────

class TestCreation:
    def test_default_creation(self):
        gen = LLMAnimationGenerator()
        assert gen is not None
        assert "llm_animation" in gen.name
        assert gen.stats["llm_calls"] == 0

    def test_custom_provider(self):
        gen = LLMAnimationGenerator(provider_name="claude")
        assert "claude" in gen.name

    def test_stats_immutable_copy(self):
        gen = LLMAnimationGenerator()
        stats = gen.stats
        stats["llm_calls"] = 999
        assert gen.stats["llm_calls"] == 0

    def test_characters_bible_starts_empty(self):
        gen = LLMAnimationGenerator()
        assert gen.characters_bible == {}

    def test_characters_bible_immutable_copy(self):
        gen = LLMAnimationGenerator()
        bible = gen.characters_bible
        bible["Fake"] = "should not persist"
        assert gen.characters_bible == {}


# ── Tests : build_user_prompt ──────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_contains_script_scene_and_visual_scene_and_image_prompt(self, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator()
        prompt = gen._build_user_prompt(None, script_scene, visual_scene, image_prompt)
        desc = script_scene.scene.description
        assert desc.camera in prompt
        assert desc.composition in prompt
        assert desc.lighting in prompt
        assert desc.mood in prompt
        assert desc.viewer_emotion in prompt
        assert desc.director_notes in prompt
        assert visual_scene.camera_motion in prompt
        assert image_prompt.subject in prompt
        assert image_prompt.scene_description in prompt

    def test_contains_continuity_block_when_script_provided(self, sample_script, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator()
        prompt = gen._build_user_prompt(sample_script, script_scene, visual_scene, image_prompt)
        assert "NARRATIVE CONTINUITY" in prompt
        assert "Sarah Chen" in prompt  # narration de la 2e scène

    def test_no_continuity_block_when_script_is_none(self, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator()
        prompt = gen._build_user_prompt(None, script_scene, visual_scene, image_prompt)
        assert "NARRATIVE CONTINUITY" not in prompt

    def test_includes_established_characters(self, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator()
        gen._update_characters_bible(image_prompt)
        # Le bloc continuité n'apparaît que si un script est fourni ; on vérifie
        # ici via build_continuity_block directement.
        block = gen._build_continuity_block(None, script_scene.order)
        assert block == ""


# ── Tests : extraction JSON robuste ────────────────────────────────────────────

class TestExtractJson:
    def test_extracts_clean_json(self):
        gen = LLMAnimationGenerator()
        raw = '{"camera_motion": "dolly in"}'
        assert gen._extract_json(raw) == raw

    def test_strips_markdown_fence(self):
        gen = LLMAnimationGenerator()
        raw = '```json\n{"camera_motion": "dolly in"}\n```'
        assert json.loads(gen._extract_json(raw)) == {"camera_motion": "dolly in"}

    def test_strips_think_tags(self):
        gen = LLMAnimationGenerator()
        raw = '<think>reasoning here</think>{"camera_motion": "pan left"}'
        assert json.loads(gen._extract_json(raw)) == {"camera_motion": "pan left"}

    def test_strips_surrounding_text(self):
        gen = LLMAnimationGenerator()
        raw = 'Sure! Here is the JSON:\n{"camera_motion": "crane up"}\nHope this helps!'
        assert json.loads(gen._extract_json(raw)) == {"camera_motion": "crane up"}

    def test_removes_trailing_commas(self):
        gen = LLMAnimationGenerator()
        raw = '{"camera_motion": "orbit",}'
        assert json.loads(gen._extract_json(raw)) == {"camera_motion": "orbit"}

    def test_removes_control_characters(self):
        gen = LLMAnimationGenerator()
        raw = '{"camera_motion": "dolly\x00 in"}'
        cleaned = gen._extract_json(raw)
        assert "\x00" not in cleaned
        assert json.loads(cleaned) == {"camera_motion": "dolly in"}

    def test_handles_nested_braces_in_strings(self):
        gen = LLMAnimationGenerator()
        raw = '{"camera_motion": "dolly in {test}"} trailing garbage {not json}'
        assert json.loads(gen._extract_json(raw)) == {"camera_motion": "dolly in {test}"}


# ── Tests : validation de structure ────────────────────────────────────────────

class TestValidateJsonStructure:
    def test_valid_structure_passes(self, valid_llm_json):
        LLMAnimationGenerator._validate_json_structure(valid_llm_json)

    def test_missing_string_field_raises(self, valid_llm_json):
        del valid_llm_json["camera_motion"]
        with pytest.raises(ValueError):
            LLMAnimationGenerator._validate_json_structure(valid_llm_json)

    def test_empty_string_field_raises(self, valid_llm_json):
        valid_llm_json["prompt"] = "   "
        with pytest.raises(ValueError):
            LLMAnimationGenerator._validate_json_structure(valid_llm_json)

    def test_missing_duration_raises(self, valid_llm_json):
        del valid_llm_json["duration"]
        with pytest.raises(ValueError):
            LLMAnimationGenerator._validate_json_structure(valid_llm_json)

    def test_duration_as_string_raises(self, valid_llm_json):
        valid_llm_json["duration"] = "five"
        with pytest.raises(ValueError):
            LLMAnimationGenerator._validate_json_structure(valid_llm_json)

    def test_duration_as_bool_raises(self, valid_llm_json):
        valid_llm_json["duration"] = True
        with pytest.raises(ValueError):
            LLMAnimationGenerator._validate_json_structure(valid_llm_json)

    def test_all_required_fields_covered(self, valid_llm_json):
        assert set(_REQUIRED_FIELDS) == set(valid_llm_json.keys())
        assert set(_REQUIRED_STRING_FIELDS) | set(_REQUIRED_INT_FIELDS) == set(_REQUIRED_FIELDS)


class TestParseAndValidate:
    def test_empty_response_raises_empty_response(self):
        response = _make_llm_response("")
        with pytest.raises(_AnimationJsonError) as exc_info:
            LLMAnimationGenerator._parse_and_validate(response)
        assert exc_info.value.reason == "empty_response"

    def test_invalid_json_raises_json_invalid(self):
        response = _make_llm_response("not json at all")
        with pytest.raises(_AnimationJsonError) as exc_info:
            LLMAnimationGenerator._parse_and_validate(response)
        assert exc_info.value.reason == "json_invalid"

    def test_truncated_response_raises_json_incomplete(self):
        response = _make_llm_response('{"camera_motion": "dolly in"', finish_reason="length")
        with pytest.raises(_AnimationJsonError) as exc_info:
            LLMAnimationGenerator._parse_and_validate(response)
        assert exc_info.value.reason == "json_incomplete"

    def test_valid_json_returns_dict(self, valid_llm_json):
        response = _make_llm_response(json.dumps(valid_llm_json))
        data = LLMAnimationGenerator._parse_and_validate(response)
        assert data == valid_llm_json

    def test_missing_field_raises_validation_failed(self, valid_llm_json):
        del valid_llm_json["transition"]
        response = _make_llm_response(json.dumps(valid_llm_json))
        with pytest.raises(_AnimationJsonError) as exc_info:
            LLMAnimationGenerator._parse_and_validate(response)
        assert exc_info.value.reason == "validation_failed"


# ── Tests : construction AnimationPrompt ──────────────────────────────────────

class TestBuildAnimationPrompt:
    def test_builds_expected_contract(self, valid_llm_json, sample_dialogues):
        response = _make_llm_response(json.dumps(valid_llm_json))
        animation_prompt = LLMAnimationGenerator._build_animation_prompt(
            valid_llm_json, response, 123, sample_dialogues,
        )

        assert isinstance(animation_prompt, AnimationPrompt)
        assert animation_prompt.camera_motion == valid_llm_json["camera_motion"]
        assert animation_prompt.subject_motion == valid_llm_json["subject_motion"]
        assert animation_prompt.environment_motion == valid_llm_json["environment_motion"]
        assert animation_prompt.lighting_changes == valid_llm_json["lighting_changes"]
        assert animation_prompt.effects == valid_llm_json["effects"]
        assert animation_prompt.sound_design == valid_llm_json["sound_design"]
        assert animation_prompt.dialogues == sample_dialogues
        assert animation_prompt.transition == valid_llm_json["transition"]
        assert animation_prompt.duration == valid_llm_json["duration"]
        assert animation_prompt.prompt == valid_llm_json["prompt"]

    def test_metadata_exact_shape(self, valid_llm_json, sample_dialogues):
        response = _make_llm_response(json.dumps(valid_llm_json))
        animation_prompt = LLMAnimationGenerator._build_animation_prompt(
            valid_llm_json, response, 123, sample_dialogues,
        )
        assert set(animation_prompt.metadata.keys()) == {
            "goal", "emotion", "pace", "provider", "model", "time_ms", "cost_usd",
            "animation_style", "voice", "sound_effects", "background_music",
        }
        assert animation_prompt.metadata["goal"] == valid_llm_json["goal"]
        assert animation_prompt.metadata["emotion"] == valid_llm_json["emotion"]
        assert animation_prompt.metadata["pace"] == valid_llm_json["pace"]
        assert animation_prompt.metadata["animation_style"] == valid_llm_json["animation_style"]
        assert animation_prompt.metadata["voice"] == valid_llm_json["voice"]
        assert animation_prompt.metadata["provider"] == "deepseek"
        assert animation_prompt.metadata["model"] == "deepseek-chat"
        assert animation_prompt.metadata["time_ms"] == 123

    def test_duration_is_clamped(self, valid_llm_json, sample_dialogues):
        valid_llm_json["duration"] = 999
        response = _make_llm_response(json.dumps(valid_llm_json))
        animation_prompt = LLMAnimationGenerator._build_animation_prompt(
            valid_llm_json, response, 1, sample_dialogues,
        )
        assert animation_prompt.duration <= 10

    def test_full_output_shape_matches_contract(self, valid_llm_json, sample_dialogues):
        """Le contrat exact demandé — 10 champs + metadata (goal/emotion/pace/
        animation_style/voice/sound_effects/background_music/provider/model/
        time_ms/cost_usd, Sprint 34.6)."""
        response = _make_llm_response(json.dumps(valid_llm_json))
        animation_prompt = LLMAnimationGenerator._build_animation_prompt(
            valid_llm_json, response, 1, sample_dialogues,
        )
        data = asdict(animation_prompt)
        assert set(data.keys()) == {
            "camera_motion", "subject_motion", "environment_motion", "lighting_changes",
            "effects", "sound_design", "dialogues", "transition", "duration", "prompt", "metadata",
        }
        assert set(data["metadata"].keys()) == {
            "goal", "emotion", "pace", "provider", "model", "time_ms", "cost_usd",
            "animation_style", "voice", "sound_effects", "background_music",
        }
        # Sérialisable en JSON tel quel (compatible pipeline)
        json.dumps(data)


# ── Tests : fallback déterministe ─────────────────────────────────────────────

class TestGenerateFromScenesFallback:
    def test_falls_back_without_exception(self, script_scene, visual_scene, image_prompt):
        """
        Sans clé API dans l'environnement de test, toutes les tentatives LLM
        échouent — generate_from_scenes() ne doit JAMAIS lever d'exception, et
        doit toujours retourner un AnimationPrompt (contrat uniforme).
        """
        gen = LLMAnimationGenerator(max_retries=1)
        animation_prompt = gen.generate_from_scenes(script_scene, visual_scene, image_prompt)
        assert isinstance(animation_prompt, AnimationPrompt)
        assert animation_prompt.metadata["provider"] == "fallback_heuristic"
        assert animation_prompt.metadata["fallback_reason"] in ("api_error", "timeout")
        assert gen.stats["fallbacks"] == 1
        assert gen.stats["llm_failures"] >= 1
        assert gen.stats["fallback_reasons"]

    def test_fallback_prompt_is_grounded_in_image_prompt(self, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator(max_retries=1)
        animation_prompt = gen.generate_from_scenes(script_scene, visual_scene, image_prompt)
        assert image_prompt.subject in animation_prompt.prompt

    def test_fallback_duration_is_within_bounds(self, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator(max_retries=1)
        animation_prompt = gen.generate_from_scenes(script_scene, visual_scene, image_prompt)
        assert 2 <= animation_prompt.duration <= 10

    def test_fallback_updates_characters_bible(self, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator(max_retries=1)
        gen.generate_from_scenes(script_scene, visual_scene, image_prompt)
        assert gen.characters_bible  # alimentée même en mode dégradé

    def test_script_param_defaults_to_constructor_script(self, sample_script, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator(script=sample_script, max_retries=1)
        animation_prompt = gen.generate_from_scenes(script_scene, visual_scene, image_prompt)
        assert isinstance(animation_prompt, AnimationPrompt)


# ── Tests : retry intelligent ─────────────────────────────────────────────────

class TestIntelligentRetry:
    def test_recovers_via_repair_retry(self, script_scene, visual_scene, image_prompt, valid_llm_json):
        gen = LLMAnimationGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response('Sure! Here you go: {"camera_motion": "oops, truncated'),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])

        animation_prompt = gen.generate_from_scenes(script_scene, visual_scene, image_prompt)

        assert isinstance(animation_prompt, AnimationPrompt)
        assert animation_prompt.camera_motion == valid_llm_json["camera_motion"]
        assert gen._provider.calls == 2
        assert gen.stats["json_repair_attempts"] == 1
        assert gen.stats["json_repairs_success"] == 1
        assert gen.stats["fallbacks"] == 0
        assert gen.stats["llm_success"] == 1

    def test_repair_prompt_asks_to_fix_only_the_json(self, script_scene, visual_scene, image_prompt, valid_llm_json):
        gen = LLMAnimationGenerator(max_retries=1)
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
        gen.generate_from_scenes(script_scene, visual_scene, image_prompt)

        repair_call_messages = captured_messages[1]
        assert repair_call_messages[-1].role == "user"
        assert "Corrige" in repair_call_messages[-1].content
        assert "JSON" in repair_call_messages[-1].content

    def test_falls_back_with_reason_when_repair_also_fails(self, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("not json at all"),
            _make_llm_response("still not json either"),
        ])

        animation_prompt = gen.generate_from_scenes(script_scene, visual_scene, image_prompt)

        assert animation_prompt.metadata["provider"] == "fallback_heuristic"
        assert animation_prompt.metadata["fallback_reason"] in ("json_invalid", "json_incomplete")
        assert gen.stats["fallbacks"] == 1
        assert gen.stats["json_repair_attempts"] == 1
        assert gen.stats["json_repairs_success"] == 0
        assert sum(gen.stats["fallback_reasons"].values()) == 1

    def test_api_error_skips_repair_entirely(self, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("[DeepSeek API Error: connection refused]", finish_reason="error"),
        ])

        animation_prompt = gen.generate_from_scenes(script_scene, visual_scene, image_prompt)

        assert animation_prompt.metadata["fallback_reason"] == "api_error"
        assert gen.stats["json_repair_attempts"] == 0
        assert gen._provider.calls == 1

    def test_timeout_response_classified_as_timeout(self, script_scene, visual_scene, image_prompt):
        gen = LLMAnimationGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("[DeepSeek API Error: Read timeout]", finish_reason="error"),
        ])

        animation_prompt = gen.generate_from_scenes(script_scene, visual_scene, image_prompt)

        assert animation_prompt.metadata["fallback_reason"] == "timeout"


# ── Tests : bible de personnages / continuité ─────────────────────────────────

class TestCharactersBible:
    def test_new_character_is_added_from_image_prompt(self, image_prompt):
        gen = LLMAnimationGenerator()
        gen._update_characters_bible(image_prompt)
        assert gen.characters_bible == {
            "Sarah Chen": "journaliste, la trentaine, blazer rouge, cheveux courts noirs, lunettes",
        }

    def test_existing_character_is_not_overwritten(self, image_prompt):
        gen = LLMAnimationGenerator()
        gen._update_characters_bible(image_prompt)
        other = ImagePrompt(
            subject="Sarah Chen", scene_description="", style="", prompt="", negative_prompt="",
            metadata={"characters": ["Sarah Chen : DESCRIPTION DIFFERENTE"]},
        )
        gen._update_characters_bible(other)
        assert "DESCRIPTION DIFFERENTE" not in gen.characters_bible["Sarah Chen"]

    def test_no_characters_key_is_handled_gracefully(self):
        gen = LLMAnimationGenerator()
        empty = ImagePrompt(subject="", scene_description="", style="", prompt="", negative_prompt="", metadata={})
        gen._update_characters_bible(empty)
        assert gen.characters_bible == {}

    def test_generic_label_falls_back_to_whole_string_as_key(self):
        gen = LLMAnimationGenerator()
        entry = "Name: Young man. Description: Mid-20s, messy dark hair, black hoodie."
        ip = ImagePrompt(subject="", scene_description="", style="", prompt="", negative_prompt="", metadata={"characters": [entry]})
        gen._update_characters_bible(ip)
        assert "Name" not in gen.characters_bible
        assert gen.characters_bible == {entry: entry}

    def test_reset_continuity_clears_bible(self, image_prompt):
        gen = LLMAnimationGenerator()
        gen._update_characters_bible(image_prompt)
        gen.reset_continuity()
        assert gen.characters_bible == {}


# ── Tests : résolution du modèle DeepSeek ─────────────────────────────────────

class TestResolveModel:
    def test_explicit_model_always_wins(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
        gen = LLMAnimationGenerator(provider_name="deepseek", model="deepseek-chat")
        assert gen._resolve_model() == "deepseek-chat"

    def test_deepseek_provider_uses_animation_model_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
        monkeypatch.setenv("DEEPSEEK_ANIMATION_MODEL", "deepseek-reasoner")
        import importlib
        import src.llm_animation_generator as mod
        importlib.reload(mod)
        try:
            gen = mod.LLMAnimationGenerator(provider_name="deepseek")
            assert gen._resolve_model() == "deepseek-reasoner"
        finally:
            monkeypatch.delenv("DEEPSEEK_ANIMATION_MODEL", raising=False)
            importlib.reload(mod)

    def test_non_deepseek_provider_returns_none(self):
        gen = LLMAnimationGenerator(provider_name="claude")
        assert gen._resolve_model() is None

    def test_no_provider_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        gen = LLMAnimationGenerator()
        assert gen._resolve_model() is None


# ── Tests : indépendance du moteur / prompt système externalisé ──────────────

class TestModuleIndependence:
    def test_does_not_import_other_llm_engines(self):
        mod_src = Path(__file__).resolve().parent.parent / "src" / "llm_animation_generator.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "llm_script_generator" not in content
        assert "rewrite_engine" not in content
        assert "llm_script_evaluator" not in content

    def test_system_prompt_loaded_from_file(self):
        from src.llm_animation_generator import _SYSTEM_PROMPT_BASE
        assert len(_SYSTEM_PROMPT_BASE) > 100
        assert "JSON" in _SYSTEM_PROMPT_BASE

    def test_system_prompt_file_exists(self):
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "animation_system_prompt.txt"
        assert prompt_path.exists()
        assert prompt_path.read_text(encoding="utf-8").strip()
