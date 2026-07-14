"""
Tests unitaires pour DialogueTranslator (Sprint 35).

Teste :
  1. Construction du prompt utilisateur (répliques uniquement, jamais
     scene.description/transition).
  2. Validation stricte du JSON (nombre de scènes/répliques identique).
  3. Construction du Script traduit (dialogues substitués, duration_seconds
     recalculé, scene/transition inchangés).
  4. Retry intelligent avant fallback.
  5. Fallback déterministe (répliques non traduites, jamais d'exception).
"""

import json

import pytest

from src.dialogue_translator import DialogueTranslator, _TranslationJsonError
from src.llm import LLMResponse
from src.script_engine import Dialogue, Scene, SceneDescription, Script, ScriptScene, estimate_scene_duration


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


def _script(scene_dialogues=None) -> Script:
    scene_dialogues = scene_dialogues or [
        [Dialogue(personnage="NARRATOR", replique="This is the hook.")],
        [Dialogue(personnage="NARRATOR", replique="Here is the context.")],
    ]
    scenes = [
        ScriptScene(
            scene=Scene(number=i + 1, type="hook" if i == 0 else "context", description=_description()),
            dialogues=dialogues,
            transition="Cut.",
            duration_seconds=estimate_scene_duration(dialogues),
        )
        for i, dialogues in enumerate(scene_dialogues)
    ]
    return Script(
        title="Test Title", scenes=scenes,
        estimated_duration=sum(s.duration_seconds for s in scenes),
        language="en", target_audience="Curious", style="Bold",
        metadata={"generator": "llm_v1"},
    )


def _make_llm_response(content, finish_reason="stop", model="deepseek-chat"):
    return LLMResponse(
        content=content, model=model, provider_name="deepseek",
        finish_reason=finish_reason, prompt_tokens=10, completion_tokens=10,
        total_tokens=20, time_ms=5, cost_usd=0.0001,
    )


class _ScriptedProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.name = "deepseek"
        self.model = "deepseek-chat"
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return self._responses.pop(0)


def _valid_translation_json(script: Script):
    return {
        "scenes": [
            {
                "number": scene.scene.number,
                "dialogues": [{"replique": f"[FR] {d.replique}"} for d in scene.dialogues],
            }
            for scene in script.scenes
        ]
    }


class TestBuildUserPrompt:
    def test_contains_only_dialogues_not_description(self):
        script = _script()
        prompt = DialogueTranslator._build_user_prompt(script, "fr")
        assert "This is the hook." in prompt
        assert "Here is the context." in prompt
        assert script.title in prompt
        # scene.description/transition ne doivent jamais apparaître ici —
        # seules les répliques sont envoyées à la traduction.
        assert script.scenes[0].scene.description.setting not in prompt
        assert script.scenes[0].transition not in prompt

    def test_mentions_target_language(self):
        prompt = DialogueTranslator._build_user_prompt(_script(), "fr")
        assert "FRENCH" in prompt


class TestValidateJsonStructure:
    def test_valid_structure_passes(self):
        script = _script()
        data = _valid_translation_json(script)
        DialogueTranslator._validate_json_structure(data, script)

    def test_wrong_scene_count_raises(self):
        script = _script()
        data = _valid_translation_json(script)
        data["scenes"].pop()
        with pytest.raises(_TranslationJsonError):
            DialogueTranslator._validate_json_structure(data, script)

    def test_wrong_dialogue_count_in_scene_raises(self):
        script = _script()
        data = _valid_translation_json(script)
        data["scenes"][0]["dialogues"].append({"replique": "extra"})
        with pytest.raises(_TranslationJsonError):
            DialogueTranslator._validate_json_structure(data, script)

    def test_empty_replique_raises(self):
        script = _script()
        data = _valid_translation_json(script)
        data["scenes"][0]["dialogues"][0]["replique"] = "   "
        with pytest.raises(_TranslationJsonError):
            DialogueTranslator._validate_json_structure(data, script)

    def test_missing_scenes_field_raises(self):
        script = _script()
        with pytest.raises(_TranslationJsonError):
            DialogueTranslator._validate_json_structure({}, script)


class TestBuildTranslatedScript:
    def test_dialogues_replaced_personnage_preserved(self):
        script = _script()
        data = _valid_translation_json(script)
        translated = DialogueTranslator._build_translated_script(script, data, "fr")

        assert translated.language == "fr"
        assert translated.scenes[0].dialogues[0].replique == "[FR] This is the hook."
        assert translated.scenes[0].dialogues[0].personnage == "NARRATOR"
        assert translated.scenes[1].dialogues[0].replique == "[FR] Here is the context."

    def test_scene_description_and_transition_unchanged(self):
        script = _script()
        data = _valid_translation_json(script)
        translated = DialogueTranslator._build_translated_script(script, data, "fr")

        assert translated.scenes[0].scene is script.scenes[0].scene
        assert translated.scenes[0].transition == script.scenes[0].transition

    def test_duration_recomputed_from_translated_dialogues(self):
        script = _script()
        data = {
            "scenes": [
                {"number": 1, "dialogues": [{"replique": "A much much much longer French sentence with many more words than the original one."}]},
                {"number": 2, "dialogues": [{"replique": "Court."}]},
            ]
        }
        translated = DialogueTranslator._build_translated_script(script, data, "fr")
        expected_0 = estimate_scene_duration(translated.scenes[0].dialogues)
        expected_1 = estimate_scene_duration(translated.scenes[1].dialogues)
        assert translated.scenes[0].duration_seconds == expected_0
        assert translated.scenes[1].duration_seconds == expected_1
        assert translated.estimated_duration == expected_0 + expected_1


class TestTranslateIntegration:
    def test_recovers_via_repair_retry(self):
        script = _script()
        valid_json = _valid_translation_json(script)
        translator = DialogueTranslator(max_retries=1)
        translator._provider = _ScriptedProvider([
            _make_llm_response('{"scenes": [oops, truncated'),
            _make_llm_response(json.dumps(valid_json)),
        ])

        translated = translator.translate(script, target_language="fr")

        assert translated.scenes[0].dialogues[0].replique == "[FR] This is the hook."
        assert translator._provider.calls == 2
        assert translator.stats["fallbacks"] == 0

    def test_falls_back_to_original_dialogues_when_llm_fails(self):
        script = _script()
        translator = DialogueTranslator(max_retries=1)
        translator._provider = _ScriptedProvider([
            _make_llm_response("not json at all"),
            _make_llm_response("still not json"),
        ])

        translated = translator.translate(script, target_language="fr")

        assert translated.scenes[0].dialogues[0].replique == script.scenes[0].dialogues[0].replique
        assert translated.metadata["translation_fallback_reason"]
        assert translator.stats["fallbacks"] == 1

    def test_fallback_never_raises_and_returns_script(self):
        script = _script()
        translator = DialogueTranslator(max_retries=1)
        translator._provider = _ScriptedProvider([
            _make_llm_response("garbage"),
            _make_llm_response("garbage again"),
        ])

        translated = translator.translate(script, target_language="fr")

        assert isinstance(translated, Script)
        assert len(translated.scenes) == len(script.scenes)
