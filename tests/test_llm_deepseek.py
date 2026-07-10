"""
Tests unitaires pour DeepSeekProvider (Sprint 21).

Couvre :
  - DeepSeekProvider : création, propriétés, count_tokens
  - DeepSeekProvider : generate (httpx mock), json_mode
  - DeepSeekProvider : generate error
  - build_llm : détection DEEPSEEK_API_KEY
  - build_llm : force deepseek provider avec modèle custom
  - LLMMessage, LLMResponse : compatibilité inchangée
"""

import json
import os
from typing import Any, Dict
from unittest.mock import MagicMock
from dataclasses import FrozenInstanceError

import httpx
import pytest

from src.llm import (
    LLMMessage,
    LLMResponse,
    DeepSeekProvider,
    build_llm,
    _MODEL_PRICING,
    estimate_cost,
)


# ── Tarifs DeepSeek ───────────────────────────────────────────────────────────

class TestDeepSeekPricing:

    def test_deepseek_models_in_pricing(self):
        assert "deepseek-chat" in _MODEL_PRICING
        assert "deepseek-reasoner" in _MODEL_PRICING

    def test_deepseek_cost_estimation(self):
        cost = estimate_cost("deepseek-chat", 1000, 500)
        expected = (1000 / 1_000_000 * 0.27) + (500 / 1_000_000 * 1.10)
        assert cost == round(expected, 6)

    def test_deepseek_reasoner_cost(self):
        cost = estimate_cost("deepseek-reasoner", 2000, 800)
        expected = (2000 / 1_000_000 * 0.55) + (800 / 1_000_000 * 2.19)
        assert cost == round(expected, 6)


# ── DeepSeekProvider ──────────────────────────────────────────────────────────

class TestDeepSeekProvider:

    def test_name_and_model(self):
        provider = DeepSeekProvider(api_key="sk-test")
        assert provider.name == "deepseek"
        assert provider.model == "deepseek-chat"

    def test_custom_model(self):
        provider = DeepSeekProvider(api_key="sk-test", model="deepseek-reasoner")
        assert provider.model == "deepseek-reasoner"

    def test_missing_key_warning(self):
        provider = DeepSeekProvider(api_key="")
        assert provider.name == "deepseek"
        assert provider.model == "deepseek-chat"

    def test_count_tokens(self):
        provider = DeepSeekProvider(api_key="sk-test")
        assert provider.count_tokens("Hello world") == 2  # 10//4 = 2
        assert provider.count_tokens("") == 1  # max(1, 0//4) = 1
        assert provider.count_tokens("Un message plus long pour tester") == 8  # 32//4 = 8

    def test_default_timeout(self):
        provider = DeepSeekProvider(api_key="sk-test")
        assert provider._timeout == 120  # Les appels DeepSeek peuvent être longs

    def test_base_url(self):
        provider = DeepSeekProvider(api_key="sk-test")
        assert provider._base_url == "https://api.deepseek.com"

    def test_generate_success(self):
        """Simule une réponse HTTP réussie de DeepSeek."""

        def mock_post(self, url, headers=None, json=None):
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
                        "message": {
                            "role": "assistant",
                            "content": "Bonjour, je suis DeepSeek !"
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 45,
                    "completion_tokens": 10,
                    "total_tokens": 55,
                },
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx.Client, "post", mock_post)
            provider = DeepSeekProvider(api_key="sk-test")
            response = provider.generate([
                LLMMessage(role="user", content="Dis bonjour"),
            ])

        assert response.content == "Bonjour, je suis DeepSeek !"
        assert response.finish_reason == "stop"
        assert response.prompt_tokens == 45
        assert response.completion_tokens == 10
        assert response.total_tokens == 55
        assert response.provider_name == "deepseek"
        assert response.model == "deepseek-chat"
        assert response.time_ms >= 0

    def test_generate_with_system_prompt(self):
        """Vérifie que le system prompt est bien envoyé."""

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
                        "message": {
                            "role": "assistant",
                            "content": "OK"
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx.Client, "post", mock_post)
            provider = DeepSeekProvider(api_key="sk-test")
            provider.generate([
                LLMMessage(role="system", content="Tu es un assistant."),
                LLMMessage(role="user", content="Salut"),
            ])

        messages = captured_body.get("messages", [])
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_generate_with_json_mode(self):
        """Vérifie que json_mode active response_format."""

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
                        "message": {
                            "role": "assistant",
                            "content": '{"response": "ok"}'
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
            provider = DeepSeekProvider(api_key="sk-test")
            provider.generate(
                [LLMMessage(role="user", content="JSON")],
                json_mode=True,
            )

        assert "response_format" in captured_body
        assert captured_body["response_format"]["type"] == "json_object"

    def test_generate_custom_model(self):
        """Override du modèle dans generate()."""

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
                            "content": "OK"
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
            provider = DeepSeekProvider(api_key="sk-test")
            response = provider.generate(
                [LLMMessage(role="user", content="Test")],
                model="deepseek-reasoner",
            )

        assert response.model == "deepseek-reasoner"

    def test_generate_custom_max_tokens(self):
        """Le max_tokens doit être plus élevé par défaut (4096)."""

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
                        "message": {"role": "assistant", "content": "OK"},
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
                [LLMMessage(role="user", content="Test")],
                max_tokens=2048,
            )

        assert captured_body.get("max_tokens") == 2048

    def test_generate_api_error(self):
        """Erreur API → LLMResponse avec finish_reason='error'."""
        provider = DeepSeekProvider(api_key="sk-invalid")
        response = provider.generate([
            LLMMessage(role="user", content="Test"),
        ])
        assert response.finish_reason == "error"
        assert "[DeepSeek API Error:" in response.content

    def test_metadata_in_response(self):
        """Les métadonnées doivent contenir les infos DeepSeek."""

        def mock_post(self, url, headers=None, json=None):
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
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx.Client, "post", mock_post)
            provider = DeepSeekProvider(api_key="sk-test")
            response = provider.generate([
                LLMMessage(role="user", content="Test"),
            ])

        assert "model" in response.metadata
        assert response.metadata["actual_model"] == "deepseek-chat"
        assert response.metadata["json_mode"] is False

    def test_estimate_cost_in_response(self):
        """Le coût doit être estimé dans la réponse."""

        def mock_post(self, url, headers=None, json=None):
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
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(httpx.Client, "post", mock_post)
            provider = DeepSeekProvider(api_key="sk-test")
            response = provider.generate([
                LLMMessage(role="user", content="Test"),
            ])

        assert response.cost_usd > 0
        assert response.total_tokens == 1500

    def test_actual_model_from_api(self):
        """Le modèle réel retourné par l'API doit être dans response.model."""

        def mock_post(self, url, headers=None, json=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": "chatcmpl-xxx",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "deepseek-chat",  # Nom réel DeepSeek
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
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
            response = provider.generate([
                LLMMessage(role="user", content="Test"),
            ])

        # Le modèle réel de l'API est renvoyé
        assert response.model == "deepseek-chat"


# ── build_llm — Détection DEEPSEEK_API_KEY ──────────────────────────────────

class TestBuildLLMDeepSeek:

    def test_auto_detect_deepseek(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
        provider = build_llm()
        assert isinstance(provider, DeepSeekProvider)
        assert provider.name == "deepseek"

    def test_auto_priority_deepseek_over_groq(self, monkeypatch):
        """DEEPSEEK_API_KEY doit être prioritaire sur GROQ_API_KEY."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
        provider = build_llm()
        assert isinstance(provider, DeepSeekProvider)

    def test_auto_priority_deepseek_over_claude(self, monkeypatch):
        """DEEPSEEK_API_KEY doit être prioritaire sur ANTHROPIC_API_KEY."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        provider = build_llm()
        assert isinstance(provider, DeepSeekProvider)

    def test_auto_priority_deepseek_over_openai(self, monkeypatch):
        """DEEPSEEK_API_KEY doit être prioritaire sur OPENAI_API_KEY."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        provider = build_llm()
        assert isinstance(provider, DeepSeekProvider)

    def test_force_deepseek(self):
        provider = build_llm(provider="deepseek")
        assert isinstance(provider, DeepSeekProvider)
        assert provider.model == "deepseek-chat"

    def test_force_deepseek_custom_model(self):
        provider = build_llm(provider="deepseek", model="deepseek-reasoner")
        assert isinstance(provider, DeepSeekProvider)
        assert provider.model == "deepseek-reasoner"

    def test_deepseek_with_kwargs(self):
        """Les kwargs sont passés au constructeur."""
        provider = build_llm(
            provider="deepseek",
            model="deepseek-chat",
            api_key="sk-custom",
            timeout=180,
        )
        assert isinstance(provider, DeepSeekProvider)
        assert provider._api_key == "sk-custom"
        assert provider._timeout == 180

    def test_other_providers_still_work(self):
        """Les autres providers ne sont pas cassés."""
        provider = build_llm(provider="openai")
        assert provider.name == "openai"

        provider = build_llm(provider="gemini")
        assert provider.name == "gemini"

        provider = build_llm(provider="claude")
        assert provider.name == "claude"

        provider = build_llm(provider="groq")
        assert provider.name == "groq"


# ── LLMMessage / LLMResponse (compatibilité) ─────────────────────────────────

class TestMessageCompatibility:

    def test_llm_message_unchanged(self):
        msg = LLMMessage(role="user", content="Test DeepSeek")
        assert msg.to_dict() == {"role": "user", "content": "Test DeepSeek"}

    def test_llm_response_unchanged(self):
        resp = LLMResponse(content="DeepSeek response", provider_name="deepseek")
        assert resp.content == "DeepSeek response"
        assert resp.provider_name == "deepseek"
        assert resp.finish_reason == "stop"

    def test_deepseek_provider_extends_abc(self):
        from src.llm import LLMProvider
        assert issubclass(DeepSeekProvider, LLMProvider)


# ── Découplage ───────────────────────────────────────────────────────────────

class TestDecoupling:

    def test_no_video_snapshot_import(self):
        with pytest.raises(ImportError):
            from src.llm import VideoSnapshot  # type: ignore

    def test_no_virality_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import ViralityEngine  # type: ignore

    def test_no_script_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import ScriptEngine  # type: ignore

    def test_no_opportunity_engine_import(self):
        with pytest.raises(ImportError):
            from src.llm import OpportunityEngine  # type: ignore
