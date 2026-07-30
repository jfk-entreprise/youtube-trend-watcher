"""
Tests unitaires pour StoryNarrator (Sprint 38.1).

Teste :
  1. Construction du prompt utilisateur (répliques dans l'ordre, titre).
  2. Validation stricte du JSON ({"story": "..."}).
  3. Extraction du récit final depuis le JSON validé.
  4. Retry intelligent avant fallback.
  5. Fallback déterministe (enchaînement simple des répliques, jamais d'exception).
"""

import json

import pytest

from src.story_narrator import StoryNarrator, _StoryJsonError
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
        [Dialogue(personnage="NARRATEUR", replique="Voici l'accroche.")],
        [Dialogue(personnage="NARRATEUR", replique="Voici le contexte.")],
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
        title="Titre de test", scenes=scenes,
        estimated_duration=sum(s.duration_seconds for s in scenes),
        language="fr", target_audience="Curieux", style="Audacieux",
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


class TestBuildUserPrompt:
    def test_contains_dialogues_in_order(self):
        script = _script()
        prompt = StoryNarrator._build_user_prompt(script)
        assert "Voici l'accroche." in prompt
        assert "Voici le contexte." in prompt
        assert prompt.index("Voici l'accroche.") < prompt.index("Voici le contexte.")

    def test_mentions_title(self):
        prompt = StoryNarrator._build_user_prompt(_script())
        assert _script().title in prompt


class TestParseAndValidate:
    def test_valid_story_passes(self):
        data = StoryNarrator._parse_and_validate(_make_llm_response(json.dumps({"story": "Il était une fois..."})))
        assert data["story"] == "Il était une fois..."

    def test_missing_story_field_raises(self):
        with pytest.raises(_StoryJsonError):
            StoryNarrator._parse_and_validate(_make_llm_response(json.dumps({})))

    def test_empty_story_field_raises(self):
        with pytest.raises(_StoryJsonError):
            StoryNarrator._parse_and_validate(_make_llm_response(json.dumps({"story": "   "})))

    def test_empty_response_raises(self):
        with pytest.raises(_StoryJsonError):
            StoryNarrator._parse_and_validate(_make_llm_response(""))

    def test_invalid_json_raises(self):
        with pytest.raises(_StoryJsonError):
            StoryNarrator._parse_and_validate(_make_llm_response("not json at all"))

    def test_extracts_json_despite_think_tags_and_prose(self):
        content = '<think>je reflechis...</think>Voici :\n```json\n{"story": "Un recit."}\n```\nVoila !'
        data = StoryNarrator._parse_and_validate(_make_llm_response(content))
        assert data["story"] == "Un recit."


class TestNarrateIntegration:
    def test_recovers_via_repair_retry(self):
        script = _script()
        narrator = StoryNarrator(max_retries=1)
        narrator._provider = _ScriptedProvider([
            _make_llm_response('{"story": oops, truncated'),
            _make_llm_response(json.dumps({"story": "Un recit fluide reconstitue."})),
        ])

        story = narrator.narrate(script)

        assert story == "Un recit fluide reconstitue."
        assert narrator._provider.calls == 2
        assert narrator.stats["fallbacks"] == 0

    def test_falls_back_to_simple_concatenation_when_llm_fails(self):
        script = _script()
        narrator = StoryNarrator(max_retries=1)
        narrator._provider = _ScriptedProvider([
            _make_llm_response("not json at all"),
            _make_llm_response("still not json"),
        ])

        story = narrator.narrate(script)

        assert "Voici l'accroche." in story
        assert "Voici le contexte." in story
        assert narrator.stats["fallbacks"] == 1

    def test_fallback_never_raises_and_returns_nonempty_string(self):
        script = _script()
        narrator = StoryNarrator(max_retries=1)
        narrator._provider = _ScriptedProvider([
            _make_llm_response("garbage"),
            _make_llm_response("garbage again"),
        ])

        story = narrator.narrate(script)

        assert isinstance(story, str)
        assert story.strip()

    def test_llm_success_returns_woven_story_not_raw_lines(self):
        script = _script()
        narrator = StoryNarrator(max_retries=1)
        woven = "Tout a commence dans un laboratoire baigne de lumiere bleue..."
        narrator._provider = _ScriptedProvider([_make_llm_response(json.dumps({"story": woven}))])

        story = narrator.narrate(script)

        assert story == woven
        assert narrator.stats["llm_success"] == 1


class TestFallbackStory:
    def test_joins_narration_text_with_blank_line(self):
        script = _script()
        fallback = StoryNarrator._fallback_story(script)
        assert fallback == "Voici l'accroche.\n\nVoici le contexte."

    def test_skips_scenes_with_empty_narration(self):
        script = _script([
            [Dialogue(personnage="NARRATEUR", replique="Seule ligne.")],
            [Dialogue(personnage="NARRATEUR", replique="")],
        ])
        fallback = StoryNarrator._fallback_story(script)
        assert fallback == "Seule ligne."
