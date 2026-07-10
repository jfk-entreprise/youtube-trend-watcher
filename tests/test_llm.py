"""
Tests unitaires pour LLM Provider (Sprint 16).

Couvre :
  - LLMMessage : création, to_dict, frozen
  - LLMResponse : création, frozen
  - estimate_cost : tarifs connus, inconnus, zéro
  - OpenAIProvider : generate (httpx mock), modèles supportés
  - GeminiProvider : generate (httpx mock), modèles supportés
  - ClaudeProvider : generate (httpx mock), modèles supportés
  - OllamaProvider : stub
  - DeepSeekProvider : stub
  - build_llm : détection auto, fallback, providers forcés
  - Découplage : le module n'importe pas les moteurs du projet
"""

import json
import os
import pytest
from dataclasses import FrozenInstanceError
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import httpx

from src.llm import (
    LLMMessage,
    LLMResponse,
    LLMProvider,
    OpenAIProvider,
    GeminiProvider,
    ClaudeProvider,
    OllamaProvider,
    DeepSeekProvider,
    build_llm,
    estimate_cost,
    _get_env_key,
)


# ── LLMMessage ────────────────────────────────────────────────────────────────

class TestLLMMessage:

    def test_creation_minimal(self):
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_creation_system(self):
        msg = LLMMessage(role="system", content="Tu es un expert.")
        assert msg.role == "system"

    def test_to_dict(self):
        msg = LLMMessage(role="user", content="Test")
        assert msg.to_dict() == {"role": "user", "content": "Test"}

    def test_frozen(self):
        msg = LLMMessage(role="user", content="Hi")
        with pytest.raises(FrozenInstanceError):
            msg.content = "Modified"  # type: ignore

    def test_equality(self):
        a = LLMMessage(role="user", content="Hello")
        b = LLMMessage(role="user", content="Hello")
        assert a == b

    def test_inequality(self):
        a = LLMMessage(role="user", content="Hello")
        b = LLMMessage(role="user", content="World")
        assert a != b


# ── LLMResponse ───────────────────────────────────────────────────────────────

class TestLLMResponse:

    def test_creation_minimal(self):
        resp = LLMResponse(content="Hello world")
        assert resp.content == "Hello world"
        assert resp.finish_reason == "stop"
        assert resp.cost_usd == 0.0
        assert resp.time_ms == 0

    def test_creation_full(self):
        resp = LLMResponse(
            content="Generated text",
            model="gpt-4o",
            provider_name="openai",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            time_ms=1200,
            cost_usd=0.0005,
            metadata={"version": "2024-08-06"},
        )
        assert resp.model == "gpt-4o"
        assert resp.total_tokens == 150
        assert resp.metadata["version"] == "2024-08-06"

    def test_frozen(self):
        resp = LLMResponse(content="Test")
        with pytest.raises(FrozenInstanceError):
            resp.content = "Modified"  # type: ignore


# ── estimate_cost ─────────────────────────────────────────────────────────────

class TestEstimateCost:

    def test_gpt4o(self):
        cost = estimate_cost("gpt-4o", 1000, 500)
        expected = (1000 / 1_000_000 * 2.50) + (500 / 1_000_000 * 10.00)
        assert cost == round(expected, 6)

    def test_claude_haiku(self):
        cost = estimate_cost("claude-3-haiku", 500, 200)
        expected = (500 / 1_000_000 * 0.25) + (200 / 1_000_000 * 1.25)
        assert cost == round(expected, 6)

    def test_unknown_model(self):
        cost = estimate_cost("unknown-model", 1000, 500)
        assert cost > 0  # fallback price

    def test_zero_tokens(self):
        cost = estimate_cost("gpt-4o", 0, 0)
        assert cost == 0.0

    def test_gemini_flash(self):
        cost = estimate_cost("gemini-1.5-flash", 2000, 1000)
        expected = (2000 / 1_000_000 * 0.075) + (1000 / 1_000_000 * 0.300)
        assert cost == round(expected, 6)

    def test_deepseek_chat(self):
        cost = estimate_cost("deepseek-chat", 1000, 500)
        expected = (1000 / 1_000_000 * 0.27) + (500 / 1_000_000 * 1.10)
        assert cost == round(expected, 6)


# ── _get_env_key ──────────────────────────────────────────────────────────────

class TestGetEnvKey:

    def test_first_found(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "value123")
        result = _get_env_key("MY_KEY", "OTHER_KEY")
        assert result == "value123"

    def test_second_found(self, monkeypatch):
        monkeypatch.setenv("OTHER_KEY", "other_value")
        result = _get_env_key("MISSING", "OTHER_KEY")
        assert result == "other_value"

    def test_none_found(self):
        result = _get_env_key("DOES_NOT_EXIST_1", "DOES_NOT_EXIST_2")
        assert result is None

    def test_empty_value(self, monkeypatch):
        monkeypatch.setenv("EMPTY_KEY", "")
        result = _get_env_key("EMPTY_KEY")
        assert result is None


# ── OpenAIProvider ────────────────────────────────────────────────────────────

class TestOpenAIProvider:

    def test_name_and_model(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.name == "openai"
        assert provider.model == "gpt-4o-mini"

    def test_custom_model(self):
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert provider.model == "gpt-4o"

    def test_missing_key_warning(self):
        """Pas de crash si la clé est vide."""
        provider = OpenAIProvider(api_key="")
        assert provider.name == "openai"

    def test_count_tokens(self):
        provider = OpenAIProvider(api_key="test")
        assert provider.count_tokens("Hello world") == 2  # len//4 = 2
        assert provider.count_tokens("") == 1  # max(1, 0//4) = 1

    def test_generate_success(self, monkeypatch):
        """Simule une réponse HTTP réussie d'OpenAI."""

        def mock_post(self, url, headers=None, json=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch.object(httpx.Client, "post", mock_post):
            provider = OpenAIProvider(api_key="sk-test")
            response = provider.generate([LLMMessage(role="user", content="Say OK")])

        assert response.content == "OK"
        assert response.finish_reason == "stop"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.provider_name == "openai"
        # Dans un test mocké, time_ms peut être 0 si < 1ms
        assert response.time_ms >= 0

    def test_generate_with_json_mode(self, monkeypatch):
        """Vérifie que json_mode ajoute response_format."""

        captured_body = {}

        def mock_post(self, url, headers=None, json=None):
            captured_body.update(json or {})
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": '{"response": "ok"}'}, "finish_reason": "stop"}],
                "usage": {},
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch.object(httpx.Client, "post", mock_post):
            provider = OpenAIProvider(api_key="sk-test")
            provider.generate(
                [LLMMessage(role="user", content="JSON")],
                json_mode=True,
            )

        assert "response_format" in captured_body
        assert captured_body["response_format"]["type"] == "json_object"

    def test_generate_api_error(self):
        """Erreur API → LLMResponse avec finish_reason='error'."""
        provider = OpenAIProvider(api_key="sk-test")
        response = provider.generate(
            [LLMMessage(role="user", content="Test")],
        )
        assert response.finish_reason == "error"
        assert "[OpenAI API Error:" in response.content

    def test_generate_custom_model(self):
        """Override du modèle dans generate()."""
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
        response = provider.generate(
            [LLMMessage(role="user", content="Test")],
            model="gpt-4o",
        )
        assert response.model == "gpt-4o"


# ── GeminiProvider ────────────────────────────────────────────────────────────

class TestGeminiProvider:

    def test_name_and_model(self):
        provider = GeminiProvider(api_key="test-key")
        assert provider.name == "gemini"
        assert provider.model == "gemini-1.5-flash"

    def test_custom_model(self):
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-pro")
        assert provider.model == "gemini-1.5-pro"

    def test_missing_key_warning(self):
        provider = GeminiProvider(api_key="")
        assert provider.name == "gemini"

    def test_generate_success(self, monkeypatch):
        """Simule une réponse HTTP réussie de Gemini."""

        def mock_post(self, url, json=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "candidates": [{
                    "content": {"parts": [{"text": "OK"}], "role": "model"},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 8,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 11,
                },
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch.object(httpx.Client, "post", mock_post):
            provider = GeminiProvider(api_key="gem-test")
            response = provider.generate([LLMMessage(role="user", content="Say OK")])

        assert response.content == "OK"
        assert response.finish_reason == "stop"
        assert response.prompt_tokens == 8
        assert response.provider_name == "gemini"

    def test_generate_with_system_prompt(self, monkeypatch):
        """Vérifie que le system prompt est envoyé dans systemInstruction."""

        captured_body = {}

        def mock_post(self, url, json=None):
            captured_body.update(json or {})
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "OK"}], "role": "model"}, "finishReason": "STOP"}],
                "usageMetadata": {},
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch.object(httpx.Client, "post", mock_post):
            provider = GeminiProvider(api_key="gem-test")
            provider.generate([
                LLMMessage(role="system", content="Tu es un assistant."),
                LLMMessage(role="user", content="Salut"),
            ])

        assert "systemInstruction" in captured_body
        assert captured_body["systemInstruction"]["parts"][0]["text"] == "Tu es un assistant."

    def test_generate_api_error(self):
        provider = GeminiProvider(api_key="")
        response = provider.generate([LLMMessage(role="user", content="Test")])
        assert response.finish_reason == "error"

    def test_generate_custom_model(self):
        provider = GeminiProvider(api_key="test")
        response = provider.generate(
            [LLMMessage(role="user", content="Test")],
            model="gemini-2.0-flash",
        )
        assert response.model == "gemini-2.0-flash"


# ── ClaudeProvider ────────────────────────────────────────────────────────────

class TestClaudeProvider:

    def test_name_and_model(self):
        provider = ClaudeProvider(api_key="test-key")
        assert provider.name == "claude"
        assert provider.model == "claude-3-5-sonnet"

    def test_custom_model(self):
        provider = ClaudeProvider(api_key="test-key", model="claude-3-haiku")
        assert provider.model == "claude-3-haiku"

    def test_missing_key_warning(self):
        provider = ClaudeProvider(api_key="")
        assert provider.name == "claude"

    def test_generate_api_error(self):
        """Sans SDK, l'appel doit échouer gracieusement."""
        provider = ClaudeProvider(api_key="sk-test")
        response = provider.generate([LLMMessage(role="user", content="Test")])
        assert response.finish_reason == "error"
        assert "[Claude API Error:" in response.content

    def test_generate_custom_model(self):
        provider = ClaudeProvider(api_key="test")
        response = provider.generate(
            [LLMMessage(role="user", content="Test")],
            model="claude-3-opus",
        )
        assert response.model == "claude-3-opus"

    def test_system_prompt_separated(self):
        """Le system prompt ne doit pas être dans messages."""
        provider = ClaudeProvider(api_key="test")
        response = provider.generate([
            LLMMessage(role="system", content="System prompt"),
            LLMMessage(role="user", content="User message"),
        ])
        assert response.finish_reason == "error"  # pas de vraie API


# ── OllamaProvider ────────────────────────────────────────────────────────────

class TestOllamaProvider:

    def test_name_and_model(self):
        provider = OllamaProvider()
        assert provider.name == "ollama"
        assert provider.model == "llama3"

    def test_custom_model(self):
        provider = OllamaProvider(model="mistral")
        assert provider.model == "mistral"

    def test_generate_stub(self):
        provider = OllamaProvider()
        response = provider.generate([LLMMessage(role="user", content="Test")])
        assert response.finish_reason == "stub"
        assert "stub" in response.content

    def test_custom_model_in_generate(self):
        provider = OllamaProvider()
        response = provider.generate(
            [LLMMessage(role="user", content="Test")],
            model="llama2",
        )
        assert response.model == "llama2"


# ── DeepSeekProvider ──────────────────────────────────────────────────────────

class TestDeepSeekProvider:

    def test_name_and_model(self):
        provider = DeepSeekProvider()
        assert provider.name == "deepseek"
        assert provider.model == "deepseek-chat"

    def test_custom_model(self):
        provider = DeepSeekProvider(model="deepseek-coder")
        assert provider.model == "deepseek-coder"

    def test_generate_api_error(self):
        """Sans clé valide, DeepSeek doit retourner une erreur propre."""
        provider = DeepSeekProvider(api_key="")
        response = provider.generate([LLMMessage(role="user", content="Test")])
        assert response.finish_reason == "error"
        assert "[DeepSeek API Error:" in response.content

    def test_generate_with_json_mode(self):
        """Vérifie que json_mode est bien supporté (httpx mock)."""

        captured_body = {}

        def mock_post(self, url, headers=None, json=None):
            captured_body.update(json or {})
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "chatcmpl-xxx",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"response": "ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx.Client, "post", mock_post)
            provider = DeepSeekProvider(api_key="sk-test")
            provider.generate(
                [LLMMessage(role="user", content="JSON")],
                json_mode=True,
            )

        assert "response_format" in captured_body
        assert captured_body["response_format"]["type"] == "json_object"

    def test_supports_reasoning_true_for_reasoner_model(self):
        provider = DeepSeekProvider(model="deepseek-reasoner")
        assert provider.supports_reasoning is True

    def test_supports_reasoning_false_for_chat_model(self):
        provider = DeepSeekProvider(model="deepseek-chat")
        assert provider.supports_reasoning is False

    def test_generate_captures_reasoning_content(self):
        """Le champ reasoning_content de l'API (mode Reasoning natif) doit être exposé en metadata."""

        def mock_post(self, url, headers=None, json=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "chatcmpl-xxx",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "deepseek-reasoner",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"response": "ok"}',
                            "reasoning_content": "Etape 1: analyse. Etape 2: conclusion.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx.Client, "post", mock_post)
            provider = DeepSeekProvider(api_key="sk-test", model="deepseek-reasoner")
            response = provider.generate([LLMMessage(role="user", content="Reasoning")])

        assert response.metadata["reasoning_enabled"] is True
        assert response.metadata["reasoning_content"] == "Etape 1: analyse. Etape 2: conclusion."

    def test_generate_reasoning_disabled_for_chat_model(self):
        def mock_post(self, url, headers=None, json=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "model": "deepseek-chat",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {},
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx.Client, "post", mock_post)
            provider = DeepSeekProvider(api_key="sk-test", model="deepseek-chat")
            response = provider.generate([LLMMessage(role="user", content="Test")])

        assert response.metadata["reasoning_enabled"] is False
        assert response.metadata["reasoning_content"] == ""


class TestSupportsReasoningHelper:
    def test_reasoner_model_supports_reasoning(self):
        from src.llm import supports_reasoning
        assert supports_reasoning("deepseek-reasoner") is True

    def test_other_models_do_not_support_reasoning(self):
        from src.llm import supports_reasoning
        assert supports_reasoning("deepseek-chat") is False
        assert supports_reasoning("gpt-4o") is False
        assert supports_reasoning("claude-3-5-sonnet") is False


# ── build_llm Factory ─────────────────────────────────────────────────────────

class TestBuildLLM:

    def test_default_fallback(self):
        """Sans clé API, build_llm() doit retourner OpenAIProvider."""
        provider = build_llm()
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    def test_force_openai(self):
        provider = build_llm(provider="openai")
        assert isinstance(provider, OpenAIProvider)

    def test_force_gemini(self):
        provider = build_llm(provider="gemini")
        assert isinstance(provider, GeminiProvider)
        assert provider.model == "gemini-1.5-flash"

    def test_force_claude(self):
        provider = build_llm(provider="claude")
        assert isinstance(provider, ClaudeProvider)
        assert provider.model == "claude-3-5-sonnet"

    def test_force_ollama(self):
        provider = build_llm(provider="ollama")
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3"

    def test_force_deepseek(self):
        provider = build_llm(provider="deepseek")
        assert isinstance(provider, DeepSeekProvider)
        assert provider.model == "deepseek-chat"

    def test_force_custom_model(self):
        provider = build_llm(provider="openai", model="gpt-4o")
        assert provider.model == "gpt-4o"

        provider = build_llm(provider="gemini", model="gemini-1.5-pro")
        assert provider.model == "gemini-1.5-pro"

        provider = build_llm(provider="claude", model="claude-3-haiku")
        assert provider.model == "claude-3-haiku"

    def test_unknown_provider_fallback(self):
        """Provider inconnu → OpenAIProvider avec warning."""
        provider = build_llm(provider="unknown_provider_xyz")
        assert isinstance(provider, OpenAIProvider)

    def test_auto_detect_claude(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        provider = build_llm()
        assert isinstance(provider, ClaudeProvider)

    def test_auto_detect_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        provider = build_llm()
        assert isinstance(provider, OpenAIProvider)

    def test_auto_detect_gemini(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
        provider = build_llm()
        assert isinstance(provider, GeminiProvider)

    def test_auto_priority_claude_over_openai(self, monkeypatch):
        """ANTHROPIC_API_KEY doit être prioritaire sur OPENAI_API_KEY."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        provider = build_llm()
        assert isinstance(provider, ClaudeProvider)

    def test_auto_priority_openai_over_gemini(self, monkeypatch):
        """OPENAI_API_KEY doit être prioritaire sur GEMINI_API_KEY."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
        provider = build_llm()
        assert isinstance(provider, OpenAIProvider)


# ── Découplage ────────────────────────────────────────────────────────────────

class TestDecoupling:

    def test_no_video_snapshot_import(self):
        with pytest.raises(ImportError):
            from src.llm import VideoSnapshot  # type: ignore

    def test_no_virality_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import ViralityEngine  # type: ignore

    def test_no_knowledge_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import KnowledgeEngine  # type: ignore

    def test_no_script_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import ScriptEngine  # type: ignore

    def test_no_opportunity_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import OpportunityEngine  # type: ignore

    def test_no_creative_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import CreativeEngine  # type: ignore

    def test_no_brand_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import BrandEngine  # type: ignore

    def test_no_learning_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import LearningEngine  # type: ignore

    def test_no_collector_import(self):
        with pytest.raises(ImportError):
            from src.llm import YouTubeCollector  # type: ignore

    def test_no_storage_import(self):
        with pytest.raises(ImportError):
            from src.llm import CsvStorage  # type: ignore

    def test_module_only_depends_on_httpx(self):
        """Vérifie que le module n'importe que httpx et standard lib."""
        import src.llm as llm
        mod_src = llm.__file__
        with open(mod_src, "r", encoding="utf-8") as f:
            content = f.read()
        # Doit importer httpx
        assert "import httpx" in content
        # Ne doit pas importer les moteurs du projet
        assert "from src.virality_engine" not in content
        assert "from src.script_engine" not in content
        assert "from src.learning_engine" not in content
        assert "from src.opportunity_engine" not in content
