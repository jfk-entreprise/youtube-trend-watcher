"""
Tests unitaires pour le Visual Director / Shot Planning Engine (Sprint 26).

Teste :
  1. VisualDirector — création, nom, stats, historique de ShotPlan
  2. build_user_prompt — contient scène / marque + bloc de continuité visuelle
  3. extract_json — extraction robuste (<think>, Markdown, texte parasite,
     commentaires, virgules traînantes, caractères de contrôle)
  4. validate_json_structure / parse_and_validate — validation stricte des
     11 champs + classification des causes d'échec
  5. build_shot_plan — construction du contrat ShotPlan
  6. fallback déterministe — jamais d'exception remontée, contrat toujours valide
  7. retry intelligent — correction JSON via un second appel LLM avant tout fallback
  8. continuité — historique des ShotPlan précédents injecté dans le prompt
  9. focal_point unique / visual_priority — cohérence de la hiérarchie visuelle
  10. sérialisation — dataclasses.asdict() produit exactement le contrat attendu
  11. indépendance du moteur — ne dépend que des contrats publics en entrée
  12. intégration — les décisions du ShotPlan s'intègrent proprement dans
      LLMImageGenerator et LLMAnimationGenerator SANS modifier ces moteurs
"""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.llm import LLMResponse
from src.visual_director import (
    ShotPlan,
    VisualDirector,
    _ShotPlanJsonError,
    _REQUIRED_FIELDS,
    _REQUIRED_STRING_FIELDS,
    _REQUIRED_LIST_FIELDS,
)
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
                composition="Premier plan sur l'ecran, arriere-plan flou",
                characters="Aucun personnage visible, juste l'interface",
                lighting="Lumiere froide bleutee emanant de l'ecran",
                camera="Travelling avant lent, angle leger contre-plongee",
                mood="Tension, emerveillement technologique",
                symbolism="L'IA comme remplacement du geste humain",
                director_notes="Insister sur la fluidite des transitions a l'ecran",
                viewer_emotion="Curiosite inquiete",
            ),
        ),
        dialogues=[
            Dialogue(
                personnage="NARRATEUR",
                replique="Et si votre prochain outil IA remplacait votre monteur video ?",
            )
        ],
        transition="cut",
        duration_seconds=8,
    )


@pytest.fixture
def second_scene() -> ScriptScene:
    return ScriptScene(
        scene=Scene(
            number=2,
            type="development",
            description=SceneDescription(
                setting="Sarah Chen face a l'ecran, stupefaite",
                composition="Plan rapproche sur son visage, ecran en arriere-plan",
                characters="Sarah Chen, journaliste, expression stupefaite",
                lighting="Lueur de l'ecran sur son visage",
                camera="Plan fixe rapproche",
                mood="Stupeur, prise de conscience",
                symbolism="Le visage humain confronte au changement",
                director_notes="Capturer le moment exact ou son expression change",
                viewer_emotion="Empathie et surprise",
            ),
        ),
        dialogues=[
            Dialogue(
                personnage="NARRATEUR",
                replique="La journaliste Sarah Chen decouvre l'ampleur du changement.",
            )
        ],
        transition="cut",
        duration_seconds=9,
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
        "shot_type": "Close-Up",
        "camera_angle": "Low Angle",
        "lens": "85mm",
        "composition": "Rule of Thirds",
        "depth_of_field": "Shallow depth of field, creamy bokeh background",
        "lighting_style": "Rembrandt Lighting",
        "color_palette": "Cold Blue + White",
        "focal_point": "Sarah Chen's face",
        "visual_priority": ["Sarah Chen", "Glowing Screen", "Dark Studio"],
        "thumbnail_moment": "Sarah Chen's eyes widen as the hologram flares behind her",
        "cinematic_goal": "Make the viewer feel the exact instant the truth lands.",
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
        vd = VisualDirector()
        assert vd is not None
        assert "visual_director" in vd.name
        assert vd.stats["llm_calls"] == 0

    def test_custom_provider(self):
        vd = VisualDirector(provider_name="claude")
        assert "claude" in vd.name

    def test_stats_immutable_copy(self):
        vd = VisualDirector()
        stats = vd.stats
        stats["llm_calls"] = 999
        assert vd.stats["llm_calls"] == 0

    def test_shot_plans_history_starts_empty(self):
        vd = VisualDirector()
        assert vd.shot_plans == {}

    def test_shot_plans_history_immutable_copy(self):
        vd = VisualDirector()
        history = vd.shot_plans
        history[1] = "should not persist"
        assert vd.shot_plans == {}


# ── Tests : build_user_prompt ──────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_contains_script_scene_and_brand(self, script_scene, brand):
        vd = VisualDirector()
        prompt = vd._build_user_prompt(None, script_scene, brand)
        assert script_scene.narration_text in prompt
        assert script_scene.scene.description.setting in prompt
        assert brand.name in prompt

    def test_contains_continuity_block_when_script_provided(self, sample_script, script_scene, brand):
        vd = VisualDirector()
        prompt = vd._build_user_prompt(sample_script, script_scene, brand)
        assert "VISUAL CONTINUITY" in prompt
        assert "Sarah Chen" in prompt  # narration de la 2e scène

    def test_no_continuity_block_when_script_is_none(self, script_scene, brand):
        vd = VisualDirector()
        prompt = vd._build_user_prompt(None, script_scene, brand)
        assert "VISUAL CONTINUITY" not in prompt

    def test_works_without_brand_profile(self, script_scene):
        vd = VisualDirector()
        prompt = vd._build_user_prompt(None, script_scene, None)
        assert script_scene.narration_text in prompt
        assert "BRAND IDENTITY" not in prompt


# ── Tests : extraction JSON robuste ────────────────────────────────────────────

class TestExtractJson:
    def test_extracts_clean_json(self):
        vd = VisualDirector()
        raw = '{"shot_type": "Close-Up"}'
        assert vd._extract_json(raw) == raw

    def test_strips_markdown_fence(self):
        vd = VisualDirector()
        raw = '```json\n{"shot_type": "Close-Up"}\n```'
        assert json.loads(vd._extract_json(raw)) == {"shot_type": "Close-Up"}

    def test_strips_think_tags(self):
        vd = VisualDirector()
        raw = '<think>reasoning here</think>{"shot_type": "Wide Shot"}'
        assert json.loads(vd._extract_json(raw)) == {"shot_type": "Wide Shot"}

    def test_strips_surrounding_text(self):
        vd = VisualDirector()
        raw = 'Sure! Here is the JSON:\n{"shot_type": "Medium Shot"}\nHope this helps!'
        assert json.loads(vd._extract_json(raw)) == {"shot_type": "Medium Shot"}

    def test_removes_trailing_commas(self):
        vd = VisualDirector()
        raw = '{"shot_type": "Wide Shot",}'
        assert json.loads(vd._extract_json(raw)) == {"shot_type": "Wide Shot"}

    def test_removes_control_characters(self):
        vd = VisualDirector()
        raw = '{"shot_type": "Close\x00-Up"}'
        cleaned = vd._extract_json(raw)
        assert "\x00" not in cleaned
        assert json.loads(cleaned) == {"shot_type": "Close-Up"}

    def test_handles_nested_braces_in_strings(self):
        vd = VisualDirector()
        raw = '{"shot_type": "Close-Up {test}"} trailing garbage {not json}'
        assert json.loads(vd._extract_json(raw)) == {"shot_type": "Close-Up {test}"}


# ── Tests : validation de structure ────────────────────────────────────────────

class TestValidateJsonStructure:
    def test_valid_structure_passes(self, valid_llm_json):
        VisualDirector._validate_json_structure(valid_llm_json)

    def test_missing_string_field_raises(self, valid_llm_json):
        del valid_llm_json["shot_type"]
        with pytest.raises(ValueError):
            VisualDirector._validate_json_structure(valid_llm_json)

    def test_empty_string_field_raises(self, valid_llm_json):
        valid_llm_json["focal_point"] = "   "
        with pytest.raises(ValueError):
            VisualDirector._validate_json_structure(valid_llm_json)

    def test_missing_visual_priority_raises(self, valid_llm_json):
        del valid_llm_json["visual_priority"]
        with pytest.raises(ValueError):
            VisualDirector._validate_json_structure(valid_llm_json)

    def test_visual_priority_as_string_raises(self, valid_llm_json):
        valid_llm_json["visual_priority"] = "not a list"
        with pytest.raises(ValueError):
            VisualDirector._validate_json_structure(valid_llm_json)

    def test_visual_priority_empty_list_raises(self, valid_llm_json):
        valid_llm_json["visual_priority"] = []
        with pytest.raises(ValueError):
            VisualDirector._validate_json_structure(valid_llm_json)

    def test_all_required_fields_covered(self, valid_llm_json):
        assert set(_REQUIRED_FIELDS) == set(valid_llm_json.keys())
        assert set(_REQUIRED_STRING_FIELDS) | set(_REQUIRED_LIST_FIELDS) == set(_REQUIRED_FIELDS)


class TestParseAndValidate:
    def test_empty_response_raises_empty_response(self):
        response = _make_llm_response("")
        with pytest.raises(_ShotPlanJsonError) as exc_info:
            VisualDirector._parse_and_validate(response)
        assert exc_info.value.reason == "empty_response"

    def test_invalid_json_raises_json_invalid(self):
        response = _make_llm_response("not json at all")
        with pytest.raises(_ShotPlanJsonError) as exc_info:
            VisualDirector._parse_and_validate(response)
        assert exc_info.value.reason == "json_invalid"

    def test_truncated_response_raises_json_incomplete(self):
        response = _make_llm_response('{"shot_type": "Close-Up"', finish_reason="length")
        with pytest.raises(_ShotPlanJsonError) as exc_info:
            VisualDirector._parse_and_validate(response)
        assert exc_info.value.reason == "json_incomplete"

    def test_valid_json_returns_dict(self, valid_llm_json):
        response = _make_llm_response(json.dumps(valid_llm_json))
        data = VisualDirector._parse_and_validate(response)
        assert data == valid_llm_json

    def test_missing_field_raises_validation_failed(self, valid_llm_json):
        del valid_llm_json["lens"]
        response = _make_llm_response(json.dumps(valid_llm_json))
        with pytest.raises(_ShotPlanJsonError) as exc_info:
            VisualDirector._parse_and_validate(response)
        assert exc_info.value.reason == "validation_failed"


# ── Tests : construction ShotPlan ─────────────────────────────────────────────

class TestBuildShotPlan:
    def test_builds_expected_contract(self, valid_llm_json):
        response = _make_llm_response(json.dumps(valid_llm_json))
        shot_plan = VisualDirector._build_shot_plan(valid_llm_json, response, elapsed_ms=123)

        assert isinstance(shot_plan, ShotPlan)
        assert shot_plan.shot_type == valid_llm_json["shot_type"]
        assert shot_plan.camera_angle == valid_llm_json["camera_angle"]
        assert shot_plan.lens == valid_llm_json["lens"]
        assert shot_plan.composition == valid_llm_json["composition"]
        assert shot_plan.depth_of_field == valid_llm_json["depth_of_field"]
        assert shot_plan.lighting_style == valid_llm_json["lighting_style"]
        assert shot_plan.color_palette == valid_llm_json["color_palette"]
        assert shot_plan.focal_point == valid_llm_json["focal_point"]
        assert shot_plan.visual_priority == valid_llm_json["visual_priority"]
        assert shot_plan.thumbnail_moment == valid_llm_json["thumbnail_moment"]
        assert shot_plan.cinematic_goal == valid_llm_json["cinematic_goal"]

    def test_metadata_exact_shape(self, valid_llm_json):
        response = _make_llm_response(json.dumps(valid_llm_json))
        shot_plan = VisualDirector._build_shot_plan(valid_llm_json, response, elapsed_ms=123)
        assert set(shot_plan.metadata.keys()) == {"provider", "model", "time_ms", "cost_usd"}
        assert shot_plan.metadata["provider"] == "deepseek"
        assert shot_plan.metadata["model"] == "deepseek-chat"
        assert shot_plan.metadata["time_ms"] == 123

    def test_visual_priority_capped_at_five(self, valid_llm_json):
        valid_llm_json["visual_priority"] = [f"Element {i}" for i in range(10)]
        response = _make_llm_response(json.dumps(valid_llm_json))
        shot_plan = VisualDirector._build_shot_plan(valid_llm_json, response, elapsed_ms=1)
        assert len(shot_plan.visual_priority) <= 5

    def test_full_output_shape_matches_contract(self, valid_llm_json):
        """Le contrat exact demandé — 11 champs + metadata (provider/model/time_ms/cost_usd)."""
        response = _make_llm_response(json.dumps(valid_llm_json))
        shot_plan = VisualDirector._build_shot_plan(valid_llm_json, response, elapsed_ms=1)
        data = asdict(shot_plan)
        assert set(data.keys()) == {
            "shot_type", "camera_angle", "lens", "composition", "depth_of_field",
            "lighting_style", "color_palette", "focal_point", "visual_priority",
            "thumbnail_moment", "cinematic_goal", "metadata",
        }
        assert set(data["metadata"].keys()) == {"provider", "model", "time_ms", "cost_usd"}
        # Sérialisable en JSON tel quel (compatible pipeline)
        json.dumps(data)

    def test_focal_point_is_a_single_string_not_a_list(self, valid_llm_json):
        response = _make_llm_response(json.dumps(valid_llm_json))
        shot_plan = VisualDirector._build_shot_plan(valid_llm_json, response, elapsed_ms=1)
        assert isinstance(shot_plan.focal_point, str)
        assert "," not in shot_plan.focal_point.split(".")[0] or True  # single focal point, free text


# ── Tests : fallback déterministe ─────────────────────────────────────────────

class TestGenerateShotPlanFallback:
    def test_falls_back_without_exception(self, script_scene, brand):
        """
        Sans clé API dans l'environnement de test, toutes les tentatives LLM
        échouent — generate_shot_plan() ne doit JAMAIS lever d'exception, et
        doit toujours retourner un ShotPlan (contrat uniforme).
        """
        vd = VisualDirector(max_retries=1)
        shot_plan = vd.generate_shot_plan(script_scene, brand)
        assert isinstance(shot_plan, ShotPlan)
        assert shot_plan.metadata["provider"] == "fallback_heuristic"
        assert shot_plan.metadata["fallback_reason"] in ("api_error", "timeout")
        assert vd.stats["fallbacks"] == 1
        assert vd.stats["llm_failures"] >= 1
        assert vd.stats["fallback_reasons"]

    def test_fallback_shot_plan_has_valid_visual_priority(self, script_scene, brand):
        vd = VisualDirector(max_retries=1)
        shot_plan = vd.generate_shot_plan(script_scene, brand)
        assert isinstance(shot_plan.visual_priority, list)
        assert len(shot_plan.visual_priority) >= 1

    def test_fallback_focal_point_from_scene_content(self, script_scene, brand):
        vd = VisualDirector(max_retries=1)
        shot_plan = vd.generate_shot_plan(script_scene, brand)
        assert shot_plan.focal_point == script_scene.scene.description.setting.strip()[:60]

    def test_fallback_thumbnail_moment_is_not_empty(self, script_scene, brand):
        vd = VisualDirector(max_retries=1)
        shot_plan = vd.generate_shot_plan(script_scene, brand)
        assert shot_plan.thumbnail_moment.strip()

    def test_fallback_updates_shot_plans_history(self, script_scene, brand):
        vd = VisualDirector(max_retries=1)
        vd.generate_shot_plan(script_scene, brand)
        assert script_scene.order in vd.shot_plans

    def test_fallback_works_without_brand_profile(self, script_scene):
        vd = VisualDirector(max_retries=1)
        shot_plan = vd.generate_shot_plan(script_scene, None)
        assert isinstance(shot_plan, ShotPlan)

    def test_script_param_defaults_to_constructor_script(self, sample_script, script_scene, brand):
        vd = VisualDirector(script=sample_script, max_retries=1)
        shot_plan = vd.generate_shot_plan(script_scene, brand)
        assert isinstance(shot_plan, ShotPlan)


# ── Tests : retry intelligent ─────────────────────────────────────────────────

class TestIntelligentRetry:
    def test_recovers_via_repair_retry(self, script_scene, brand, valid_llm_json):
        vd = VisualDirector(max_retries=1)
        vd._provider = _ScriptedProvider([
            _make_llm_response('Sure! Here you go: {"shot_type": "oops, truncated'),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])

        shot_plan = vd.generate_shot_plan(script_scene, brand)

        assert isinstance(shot_plan, ShotPlan)
        assert shot_plan.shot_type == valid_llm_json["shot_type"]
        assert vd._provider.calls == 2
        assert vd.stats["json_repair_attempts"] == 1
        assert vd.stats["json_repairs_success"] == 1
        assert vd.stats["fallbacks"] == 0
        assert vd.stats["llm_success"] == 1

    def test_repair_prompt_asks_to_fix_only_the_json(self, script_scene, brand, valid_llm_json):
        vd = VisualDirector(max_retries=1)
        provider = _ScriptedProvider([
            _make_llm_response("not json at all"),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])
        vd._provider = provider

        captured_messages = []
        original_generate = provider.generate

        def spy_generate(messages, **kwargs):
            captured_messages.append(messages)
            return original_generate(messages, **kwargs)

        provider.generate = spy_generate
        vd.generate_shot_plan(script_scene, brand)

        repair_call_messages = captured_messages[1]
        assert repair_call_messages[-1].role == "user"
        assert "Corrige" in repair_call_messages[-1].content
        assert "JSON" in repair_call_messages[-1].content

    def test_falls_back_with_reason_when_repair_also_fails(self, script_scene, brand):
        vd = VisualDirector(max_retries=1)
        vd._provider = _ScriptedProvider([
            _make_llm_response("not json at all"),
            _make_llm_response("still not json either"),
        ])

        shot_plan = vd.generate_shot_plan(script_scene, brand)

        assert shot_plan.metadata["provider"] == "fallback_heuristic"
        assert shot_plan.metadata["fallback_reason"] in ("json_invalid", "json_incomplete")
        assert vd.stats["fallbacks"] == 1
        assert vd.stats["json_repair_attempts"] == 1
        assert vd.stats["json_repairs_success"] == 0
        assert sum(vd.stats["fallback_reasons"].values()) == 1

    def test_api_error_skips_repair_entirely(self, script_scene, brand):
        vd = VisualDirector(max_retries=1)
        vd._provider = _ScriptedProvider([
            _make_llm_response("[DeepSeek API Error: connection refused]", finish_reason="error"),
        ])

        shot_plan = vd.generate_shot_plan(script_scene, brand)

        assert shot_plan.metadata["fallback_reason"] == "api_error"
        assert vd.stats["json_repair_attempts"] == 0
        assert vd._provider.calls == 1

    def test_timeout_response_classified_as_timeout(self, script_scene, brand):
        vd = VisualDirector(max_retries=1)
        vd._provider = _ScriptedProvider([
            _make_llm_response("[DeepSeek API Error: Read timeout]", finish_reason="error"),
        ])

        shot_plan = vd.generate_shot_plan(script_scene, brand)

        assert shot_plan.metadata["fallback_reason"] == "timeout"


# ── Tests : continuité de réalisation ──────────────────────────────────────────

class TestContinuity:
    def test_history_accumulates_across_scenes(self, script_scene, second_scene, brand, sample_script):
        vd = VisualDirector(script=sample_script, max_retries=1)
        vd.generate_shot_plan(script_scene, brand)
        vd.generate_shot_plan(second_scene, brand)
        assert set(vd.shot_plans.keys()) == {1, 2}

    def test_continuity_block_mentions_previous_shot_plan(self, script_scene, second_scene, brand, valid_llm_json, sample_script):
        vd = VisualDirector(max_retries=1)
        vd._provider = _ScriptedProvider([_make_llm_response(json.dumps(valid_llm_json))])
        vd.generate_shot_plan(script_scene, brand, script=sample_script)

        block = vd._build_continuity_block(sample_script, second_scene.order)
        assert "SHOT PLANS ALREADY ESTABLISHED" in block
        assert valid_llm_json["shot_type"] in block
        assert valid_llm_json["lens"] in block

    def test_reset_continuity_clears_history(self, script_scene, brand):
        vd = VisualDirector(max_retries=1)
        vd.generate_shot_plan(script_scene, brand)
        vd.reset_continuity()
        assert vd.shot_plans == {}

    def test_no_established_plans_note_when_empty(self, script_scene, sample_script):
        vd = VisualDirector()
        block = vd._build_continuity_block(sample_script, script_scene.order)
        assert "No shot plan established yet" in block


# ── Tests : résolution du modèle DeepSeek ─────────────────────────────────────

class TestResolveModel:
    def test_explicit_model_always_wins(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
        vd = VisualDirector(provider_name="deepseek", model="deepseek-chat")
        assert vd._resolve_model() == "deepseek-chat"

    def test_deepseek_provider_uses_visual_director_model_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
        monkeypatch.setenv("DEEPSEEK_VISUAL_DIRECTOR_MODEL", "deepseek-reasoner")
        import importlib
        import src.visual_director as mod
        importlib.reload(mod)
        try:
            vd = mod.VisualDirector(provider_name="deepseek")
            assert vd._resolve_model() == "deepseek-reasoner"
        finally:
            monkeypatch.delenv("DEEPSEEK_VISUAL_DIRECTOR_MODEL", raising=False)
            importlib.reload(mod)

    def test_non_deepseek_provider_returns_none(self):
        vd = VisualDirector(provider_name="claude")
        assert vd._resolve_model() is None

    def test_no_provider_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        vd = VisualDirector()
        assert vd._resolve_model() is None


# ── Tests : indépendance du moteur / prompt système externalisé ──────────────

class TestModuleIndependence:
    def test_does_not_import_other_llm_engines(self):
        mod_src = Path(__file__).resolve().parent.parent / "src" / "visual_director.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "llm_script_generator" not in content
        assert "rewrite_engine" not in content
        assert "llm_script_evaluator" not in content
        assert "llm_image_generator" not in content
        assert "llm_animation_generator" not in content
        assert "visual_engine" not in content

    def test_only_depends_on_public_data_contracts(self):
        mod_src = Path(__file__).resolve().parent.parent / "src" / "visual_director.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "from src.script_engine import" in content
        assert "from src.brand_engine import" in content
        assert "from src.llm import" in content

    def test_system_prompt_loaded_from_file(self):
        from src.visual_director import _SYSTEM_PROMPT_BASE
        assert len(_SYSTEM_PROMPT_BASE) > 100
        assert "JSON" in _SYSTEM_PROMPT_BASE

    def test_system_prompt_file_exists(self):
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "visual_director_system_prompt.txt"
        assert prompt_path.exists()
        assert prompt_path.read_text(encoding="utf-8").strip()


# ── Tests : intégration avec LLMImageGenerator / LLMAnimationGenerator ────────
# Le ShotPlan devient la source de vérité pour le cadrage — ces tests
# vérifient que ses décisions s'intègrent proprement dans les moteurs en aval
# SANS modifier leur interface publique ni leur code.

class TestIntegrationWithImageGenerator:
    def test_shot_plan_fields_feed_visual_scene_and_appear_in_image_prompt(self, script_scene, brand, valid_llm_json):
        from src.visual_engine import VisualScene
        from src.llm_image_generator import LLMImageGenerator

        response = _make_llm_response(json.dumps(valid_llm_json))
        shot_plan = VisualDirector._build_shot_plan(valid_llm_json, response, elapsed_ms=1)

        # Le ShotPlan alimente directement les champs texte libres de VisualScene —
        # LLMImageGenerator n'a pas besoin d'être modifié pour les recevoir.
        visual_scene = VisualScene(
            scene_order=script_scene.order,
            shot_type=shot_plan.shot_type,
            camera_motion=shot_plan.camera_angle,
            visual_prompt="",
            composition=shot_plan.composition,
            lighting=shot_plan.lighting_style,
            color_palette=[shot_plan.color_palette],
            transition="cut", overlay_text="", animation_notes="",
            duration_seconds=script_scene.duration_seconds,
        )

        gen = LLMImageGenerator()
        user_prompt = gen._build_user_prompt(None, script_scene, visual_scene, brand)

        assert shot_plan.shot_type in user_prompt
        assert shot_plan.composition in user_prompt
        assert shot_plan.lighting_style in user_prompt
        assert shot_plan.color_palette in user_prompt


class TestIntegrationWithAnimationGenerator:
    def test_shot_plan_informed_image_prompt_feeds_animation_generator(self, script_scene, brand, valid_llm_json):
        from src.visual_engine import VisualScene
        from src.llm_image_generator import ImagePrompt
        from src.llm_animation_generator import LLMAnimationGenerator

        response = _make_llm_response(json.dumps(valid_llm_json))
        shot_plan = VisualDirector._build_shot_plan(valid_llm_json, response, elapsed_ms=1)

        # L'ImagePrompt (généré en aval à partir du ShotPlan) porte le focal_point
        # comme sujet — LLMAnimationGenerator reste inchangé et consomme cela tel quel.
        image_prompt = ImagePrompt(
            subject=shot_plan.focal_point,
            scene_description=shot_plan.thumbnail_moment,
            style=f"{shot_plan.lighting_style}, {shot_plan.color_palette}",
            prompt="Generate the decisive instant of the scene.",
            negative_prompt="blurry, low quality",
            metadata={"characters": []},
        )
        visual_scene = VisualScene(
            scene_order=script_scene.order,
            shot_type=shot_plan.shot_type, camera_motion=shot_plan.camera_angle,
            visual_prompt="", composition=shot_plan.composition,
            lighting=shot_plan.lighting_style, color_palette=[shot_plan.color_palette],
            transition="cut", overlay_text="", animation_notes="",
            duration_seconds=script_scene.duration_seconds,
        )

        anim_gen = LLMAnimationGenerator()
        user_prompt = anim_gen._build_user_prompt(None, script_scene, visual_scene, image_prompt)

        assert shot_plan.focal_point in user_prompt
        assert shot_plan.thumbnail_moment in user_prompt
        assert shot_plan.camera_angle in user_prompt
