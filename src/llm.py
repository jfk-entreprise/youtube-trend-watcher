"""
LLM Provider — Couche d'abstraction unique pour tous les modèles IA.

Philosophie :
  - Aucun moteur du projet ne doit appeler directement une API LLM.
  - Tous les appels passent par `provider.generate(messages)`.
  - Interchangeable : swap de provider sans toucher au code métier.

Utilisation standard :
    from src.llm import build_llm

    provider = build_llm()  # lit le .env automatiquement
    response = provider.generate([
        {"role": "user", "content": "Explique l'IA simplement."},
    ])
    print(response.content)

Architecture :
  - LLMMessage   : message standardisé (role + content).
  - LLMResponse  : réponse standardisée (content, model, usage, timing, cost).
  - LLMProvider  : ABC pour tous les providers.
  - build_llm()  : factory qui lit les clés dans l'ordre de priorité.

Providers implémentés :
  - OpenAIProvider    : gpt-4o, gpt-4o-mini, gpt-4-turbo, etc. (REST httpx)
  - GeminiProvider    : gemini-1.5-pro, gemini-1.5-flash, etc.  (REST httpx)
  - ClaudeProvider    : claude-3-opus, claude-3-sonnet, etc.     (SDK Anthropic)
  - GroqProvider      : openai/gpt-oss-120b, llama-3.3-70b-versatile, etc. (REST httpx)

Providers prévus (interface extensible) :
  - OllamaProvider    : modèles locaux (llama3, mistral, etc.)
  - DeepSeekProvider  : deepseek-chat, deepseek-coder

Coût :
  - Basé sur les tarifs officiels au moment de l'écriture.
  - Mis à jour via `_MODEL_PRICING` — extensible par fichier JSON.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# ── Types standardisés ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMMessage:
    """
    Message standardisé pour tous les providers.

    Respecte le format universel ChatML :
      - system     : instruction système (optionnel)
      - user       : message de l'utilisateur
      - assistant  : réponse de l'assistant (pour historique)

    Exemple :
        messages = [
            LLMMessage(role="system", content="Tu es un expert YouTube."),
            LLMMessage(role="user", content="Génère un hook pour une vidéo sur l'IA."),
        ]
    """
    role: str       # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMResponse:
    """
    Réponse standardisée d'un appel LLM.

    Champs :
      - content        : texte généré.
      - model          : modèle utilisé (ex: "gpt-4o", "claude-3-sonnet").
      - provider_name  : nom du provider (ex: "openai", "claude").
      - finish_reason  : reason de terminaison ("stop", "length", etc.).
      - prompt_tokens  : tokens en entrée.
      - completion_tokens : tokens générés.
      - total_tokens   : sum(prompt_tokens, completion_tokens).
      - time_ms        : temps de génération en ms.
      - cost_usd       : coût estimé en USD.
      - cost_currency  : devise du coût ("USD").
      - metadata       : données extensibles (provider, version, etc.).
    """
    content: str
    model: str = ""
    provider_name: str = ""
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_ms: int = 0
    cost_usd: float = 0.0
    cost_currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Tarifs officiels (USD / 1M tokens) ────────────────────────────────────────

# Format : (input_price, output_price) en USD par million de tokens
_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI — GPT-4o
    "gpt-4o":              (2.50,   10.00),
    "gpt-4o-2024-08-06":   (2.50,   10.00),
    "gpt-4o-mini":         (0.150,  0.600),
    "gpt-4-turbo":         (10.00,  30.00),
    "gpt-4":               (30.00,  60.00),
    "gpt-3.5-turbo":       (0.50,   1.50),
    # Anthropic — Claude 3
    "claude-3-opus":        (15.00,  75.00),
    "claude-3-sonnet":      (3.00,   15.00),
    "claude-3-haiku":       (0.25,   1.25),
    "claude-3-5-sonnet":    (3.00,   15.00),
    "claude-3-5-haiku":     (0.80,   4.00),
    # Google — Gemini
    "gemini-1.5-pro":       (3.50,   10.50),
    "gemini-1.5-flash":     (0.075,  0.300),
    "gemini-2.0-flash":     (0.10,   0.40),
    "gemini-1.0-pro":       (0.50,   1.50),
    # DeepSeek
    "deepseek-chat":        (0.27,   1.10),
    "deepseek-reasoner":    (0.55,   2.19),
    "deepseek-coder":       (0.14,   0.28),
    # Groq (hébergement ultra-rapide via API compatible OpenAI)
    "openai/gpt-oss-120b":      (0.59,   0.79),
    "llama-3.3-70b-versatile":  (0.59,   0.79),
    "groq-llama3-8b":          (0.05,   0.10),
    "groq-llama3-70b":         (0.59,   0.79),
    "groq-mixtral-8x7b":       (0.24,   0.24),
    "groq-gemma2-9b":          (0.05,   0.10),
    # Ollama (local — pas de coût)
    "ollama":               (0.0,    0.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estime le coût d'un appel LLM en USD.

    Args:
        model: nom du modèle (ex: "gpt-4o", "claude-3-sonnet").
        prompt_tokens: tokens en entrée.
        completion_tokens: tokens générés.

    Returns:
        Coût estimé en USD.
    """
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        # Fallback : modèle inconnu → prix moyen approximatif
        pricing = (2.0, 8.0)

    input_price, output_price = pricing
    cost = (prompt_tokens / 1_000_000 * input_price +
            completion_tokens / 1_000_000 * output_price)
    return round(cost, 6)


# ── LLMProvider (ABC) ─────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Interface abstraite pour tous les providers LLM.

    Le reste du projet ne voit QUE cette interface.
    Tous les providers sont interchangeables via build_llm().

    Méthodes :
      - generate(messages, **kwargs) : appel standard.
      - count_tokens(text)           : estimation du nombre de tokens.
      - name                         : nom du provider.
      - model                        : modèle actif.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du provider (ex: "openai", "claude", "gemini")."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Modèle actif (ex: "gpt-4o", "claude-3-sonnet")."""
        ...

    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Appel standardisé à un LLM.

        Args:
            messages    : liste de messages (system + user + historique).
            temperature : créativité [0.0 – 2.0] (défaut: 0.7).
            max_tokens  : nombre max de tokens en sortie (défaut: 1024).
            model       : override du modèle (ex: "gpt-4o-mini").
            json_mode   : si True, force le format JSON en sortie.

        Returns:
            LLMResponse standardisé.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """
        Estimation du nombre de tokens.

        Approximation : 1 token ≈ 4 caractères pour les langues latines.
        Certains providers (Anthropic) ont leur propre compteur.
        """
        return max(1, len(text) // 4)


# ── API Keys ─────────────────────────────────────────────────────────────────

def _get_env_key(*names: str) -> Optional[str]:
    """Cherche une clé API parmi plusieurs noms de variables d'environnement."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


# ── OpenAI Provider (REST via httpx) ──────────────────────────────────────────

OPENAI_BASE_URL = "https://api.openai.com/v1"

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

_OPENAI_MODELS = {
    "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini",
    "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
}


class OpenAIProvider(LLMProvider):
    """
    Provider OpenAI — gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.

    Communication via REST (httpx) — pas de SDK lourd.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_OPENAI_MODEL,
        base_url: str = OPENAI_BASE_URL,
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key or _get_env_key("OPENAI_API_KEY") or ""
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        if not self._api_key:
            logger.warning("OpenAIProvider: aucune clé API fournie.")

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        model_name = model or self._model

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            body["response_format"] = {"type": "json_object"}

        start = time.time()
        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            elapsed = int((time.time() - start) * 1000)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            logger.error("OpenAI API error after %d ms: %s", elapsed, exc)
            return LLMResponse(
                content=f"[OpenAI API Error: {exc}]",
                model=model_name,
                provider_name="openai",
                finish_reason="error",
                time_ms=elapsed,
            )

        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason", "stop")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        cost = estimate_cost(model_name, prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=model_name,
            provider_name="openai",
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_ms=elapsed,
            cost_usd=cost,
            metadata={
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            },
        )

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()


# ── Gemini Provider (REST via httpx) ─────────────────────────────────────────

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

_GEMINI_MODELS = {
    "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.0-pro",
}


class GeminiProvider(LLMProvider):
    """
    Provider Google Gemini — gemini-1.5-pro, gemini-1.5-flash, etc.

    Communication via REST (httpx).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_GEMINI_MODEL,
        base_url: str = GEMINI_BASE_URL,
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key or _get_env_key("GEMINI_API_KEY", "GOOGLE_API_KEY") or ""
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        if not self._api_key:
            logger.warning("GeminiProvider: aucune clé API fournie.")

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        model_name = model or self._model

        # Gemini utilise "contents" et un system_instruction séparé
        system_content = ""
        contents: List[Dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                system_content = m.content
            elif m.role in ("user", "assistant"):
                role_map = {"user": "user", "assistant": "model"}
                contents.append({
                    "role": role_map.get(m.role, "user"),
                    "parts": [{"text": m.content}],
                })

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_content:
            body["systemInstruction"] = {
                "parts": [{"text": system_content}]
            }

        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        url = f"{self._base_url}/models/{model_name}:generateContent?key={self._api_key}"

        start = time.time()
        try:
            resp = self._client.post(url, json=body)
            elapsed = int((time.time() - start) * 1000)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            logger.error("Gemini API error after %d ms: %s", elapsed, exc)
            return LLMResponse(
                content=f"[Gemini API Error: {exc}]",
                model=model_name,
                provider_name="gemini",
                finish_reason="error",
                time_ms=elapsed,
            )

        # Extraction du texte
        try:
            candidate = data["candidates"][0]
            content = candidate["content"]["parts"][0]["text"]
            finish_reason = candidate.get("finishReason", "STOP").lower()
        except (KeyError, IndexError):
            content = json.dumps(data, ensure_ascii=False)
            finish_reason = "unknown"

        # Gemini ne retourne pas toujours les tokens — on estime
        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
        total_tokens = usage_meta.get("totalTokenCount", 0)

        cost = estimate_cost(model_name, prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=model_name,
            provider_name="gemini",
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_ms=elapsed,
            cost_usd=cost,
            metadata={
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            },
        )

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()


# ── Claude Provider (SDK Anthropic) ──────────────────────────────────────────

_DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet"

_CLAUDE_MODELS = {
    "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
    "claude-3-5-sonnet", "claude-3-5-haiku",
}


class ClaudeProvider(LLMProvider):
    """
    Provider Anthropic Claude — claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.

    Utilise le SDK officiel Anthropic.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_CLAUDE_MODEL,
        timeout: int = 60,
    ) -> None:
        self._api_key = api_key or _get_env_key("ANTHROPIC_API_KEY") or ""
        self._model = model
        self._timeout = timeout
        self._client: Optional[Any] = None

        if not self._api_key:
            logger.warning("ClaudeProvider: aucune clé API fournie.")

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    @property
    def name(self) -> str:
        return "claude"

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        model_name = model or self._model

        # Claude sépare system des messages utilisateur
        system_prompt = ""
        claude_messages: List[Dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            elif m.role in ("user", "assistant"):
                claude_messages.append({
                    "role": m.role,
                    "content": m.content,
                })

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": claude_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if json_mode:
            # Claude supporte le JSON via l'extension "extended_output"
            kwargs["extra_headers"] = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}

        start = time.time()
        try:
            client = self._get_client()
            resp = client.messages.create(**kwargs)
            elapsed = int((time.time() - start) * 1000)
        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            logger.error("Claude API error after %d ms: %s", elapsed, exc)
            return LLMResponse(
                content=f"[Claude API Error: {exc}]",
                model=model_name,
                provider_name="claude",
                finish_reason="error",
                time_ms=elapsed,
            )

        content = resp.content[0].text if resp.content else ""
        finish_reason = resp.stop_reason or "stop"

        usage = resp.usage
        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0
        total_tokens = prompt_tokens + completion_tokens

        cost = estimate_cost(model_name, prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=model_name,
            provider_name="claude",
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_ms=elapsed,
            cost_usd=cost,
            metadata={
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "anthropic_version": getattr(resp, "model", ""),
            },
        )


# ── Ollama Provider (prévu — stub) ───────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """
    Provider Ollama — modèles locaux (llama3, mistral, qwen, etc.).

    Communication via REST sur http://localhost:11434.

    Prérequis :
      - Ollama installé : https://ollama.com
      - Modèle téléchargé : ollama pull llama3

    Implémentation prévue Sprint 17.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        logger.warning("OllamaProvider: non implémenté — stub.")
        return LLMResponse(
            content="[Ollama: stub — provider non implémenté]",
            model=model or self._model,
            provider_name="ollama",
            finish_reason="stub",
        )


# ── DeepSeek Provider (REST via httpx — API compatible OpenAI) ────────────────

_DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

_DEEPSEEK_MODELS = {
    "deepseek-chat",
    "deepseek-reasoner",
}

# Modèles DeepSeek dotés d'un mode Reasoning natif (chain-of-thought interne,
# exposé via le champ `reasoning_content` de la réponse API).
_REASONING_MODELS = {
    "deepseek-reasoner",
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def supports_reasoning(model: str) -> bool:
    """
    Indique si un modèle (par nom) dispose d'un mode Reasoning natif.

    Utilisable par tout moteur consommant build_llm()/LLMProvider sans avoir
    à connaître la liste des modèles de raisonnement — ex: pour décider
    d'activer un prompt de raisonnement explicite en fallback quand le
    modèle actif n'a pas de raisonnement natif.
    """
    return model in _REASONING_MODELS


class DeepSeekProvider(LLMProvider):
    """
    Provider DeepSeek — deepseek-chat, deepseek-reasoner.

    API compatible OpenAI → REST httpx.
    Documentation : https://platform.deepseek.com/api-docs

    Modèles :
      - deepseek-chat     : modèle conversationnel généraliste (V3)
      - deepseek-reasoner : modèle avec capacité de raisonnement (R1)

    Utilisation :
        provider = DeepSeekProvider(api_key="sk_...")
        # ou via build_llm(provider="deepseek")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_DEEPSEEK_MODEL,
        base_url: str = DEEPSEEK_BASE_URL,
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key or _get_env_key("DEEPSEEK_API_KEY") or ""
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        if not self._api_key:
            logger.warning("DeepSeekProvider: aucune clé API fournie.")

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    @property
    def supports_reasoning(self) -> bool:
        """True si le modèle actif de ce provider a un mode Reasoning natif."""
        return supports_reasoning(self._model)

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Appel à l'API DeepSeek (compatible OpenAI).

        Note : DeepSeek supporte le json_mode via response_format.
        Pour deepseek-reasoner, temperature est ignorée par l'API et le
        modèle raisonne nativement avant de répondre — ce raisonnement est
        renvoyé par l'API dans `message.reasoning_content` et exposé ici via
        `LLMResponse.metadata["reasoning_content"]`.
        """
        model_name = model or self._model

        # Vérification explicite : pas d'appel sans clé
        if not self._api_key:
            elapsed = 0
            logger.error("DeepSeek API: clé API manquante — configurez DEEPSEEK_API_KEY")
            return LLMResponse(
                content="[DeepSeek API Error: clé API manquante — définissez DEEPSEEK_API_KEY]",
                model=model_name,
                provider_name="deepseek",
                finish_reason="error",
                time_ms=elapsed,
            )

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            body["response_format"] = {"type": "json_object"}

        # Retry avec backoff progressif (pour rate limiting)
        max_retries = 3
        base_delay = 2.0  # secondes

        for attempt in range(1, max_retries + 1):
            start = time.time()
            try:
                resp = self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                elapsed = int((time.time() - start) * 1000)

                # Gestion du rate limiting (429)
                if resp.status_code == 429:
                    retry_after_header = resp.headers.get("retry-after", "")
                    wait = base_delay * attempt
                    if retry_after_header and retry_after_header.isdigit():
                        wait = int(retry_after_header)
                    logger.warning(
                        "DeepSeek rate limit (429) — tentative %d/%d, attente %ds",
                        attempt, max_retries, wait,
                    )
                    if attempt < max_retries:
                        time.sleep(wait)
                        continue
                    else:
                        logger.error("DeepSeek rate limit — toutes les tentatives épuisées")
                        return LLMResponse(
                            content=f"[DeepSeek API Error: Rate limit dépassé après {max_retries} tentatives]",
                            model=model_name,
                            provider_name="deepseek",
                            finish_reason="error",
                            time_ms=elapsed,
                        )

                resp.raise_for_status()
                data = resp.json()
                break  # Succès

            except httpx.HTTPStatusError as exc:
                elapsed = int((time.time() - start) * 1000)
                if exc.response.status_code == 429 and attempt < max_retries:
                    wait = base_delay * attempt
                    logger.warning(
                        "DeepSeek rate limit (429, httpx) — tentative %d/%d, attente %ds",
                        attempt, max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error("DeepSeek API error after %d ms: %s", elapsed, exc)
                return LLMResponse(
                    content=f"[DeepSeek API Error: {exc}]",
                    model=model_name,
                    provider_name="deepseek",
                    finish_reason="error",
                    time_ms=elapsed,
                )
            except Exception as exc:
                elapsed = int((time.time() - start) * 1000)
                logger.error("DeepSeek API error after %d ms: %s", elapsed, exc)
                return LLMResponse(
                    content=f"[DeepSeek API Error: {exc}]",
                    model=model_name,
                    provider_name="deepseek",
                    finish_reason="error",
                    time_ms=elapsed,
                )

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # DeepSeek peut retourner le modèle via la réponse
        actual_model = data.get("model", model_name)

        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason", "stop")
        reasoning_content = choice["message"].get("reasoning_content", "") or ""

        cost = estimate_cost(model_name, prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=actual_model,
            provider_name="deepseek",
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_ms=elapsed,
            cost_usd=cost,
            metadata={
                "model": model_name,
                "actual_model": actual_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "reasoning_enabled": supports_reasoning(model_name),
                "reasoning_content": reasoning_content,
            },
        )

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()

    def count_tokens(self, text: str) -> int:
        """
        Estimation du nombre de tokens.

        Approximation : 1 token ≈ 4 caractères pour les langues latines.
        """
        return max(1, len(text) // 4)


# ── Groq Provider (REST via httpx — API compatible OpenAI) ────────────────────

_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

_GROQ_MODELS = {
    "openai/gpt-oss-120b", "llama-3.3-70b-versatile",
    "groq-llama3-8b", "groq-llama3-70b", "groq-mixtral-8x7b", "groq-gemma2-9b",
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(LLMProvider):
    """
    Provider Groq — inférence ultra-rapide via LPU.

    API compatible OpenAI → REST httpx.
    Documentation : https://console.groq.com/docs/api-reference

    Un seul provider, deux modèles principaux :
      - openai/gpt-oss-120b      : génération de scripts créatifs (qualité)
      - llama-3.3-70b-versatile  : génération de prompts d'images (vitesse)

    Utilisation :
        provider = GroqProvider(api_key="gsk_...")
        # ou via build_llm(provider="groq")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_GROQ_MODEL,
        base_url: str = GROQ_BASE_URL,
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key or _get_env_key("GROQ_API_KEY") or ""
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        if not self._api_key:
            logger.warning("GroqProvider: aucune clé API fournie.")

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Appel à l'API Groq (compatible OpenAI).

        Note : Groq supporte le json_mode mais peut nécessiter
        un prompt qui demande explicitement du JSON.
        """
        model_name = model or self._model

        # Vérification explicite : pas d'appel sans clé
        if not self._api_key:
            elapsed = 0
            logger.error("Groq API: clé API manquante — configurez GROQ_API_KEY")
            return LLMResponse(
                content="[Groq API Error: clé API manquante — définissez GROQ_API_KEY]",
                model=model_name,
                provider_name="groq",
                finish_reason="error",
                time_ms=elapsed,
            )

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            body["response_format"] = {"type": "json_object"}

        # Retry avec backoff progressif (pour rate limiting)
        max_retries = 5          # Augmenté de 3 à 5 pour plus de résilience
        base_delay = 2.0         # secondes
        max_wait = 30            # délai maximum entre 2 tentatives (évite d'attendre 46 min)
        import random

        for attempt in range(1, max_retries + 1):
            start = time.time()
            try:
                resp = self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                elapsed = int((time.time() - start) * 1000)

                # Gestion du rate limiting (429)
                if resp.status_code == 429:
                    # Lire le header retry-after, mais plafonné à max_wait + jitter
                    retry_after_header = resp.headers.get("retry-after", "")
                    raw_wait = 0
                    if retry_after_header.isdigit():
                        raw_wait = int(retry_after_header)
                    wait = min(raw_wait, max_wait) if raw_wait > 0 else min(base_delay * attempt, max_wait)
                    # Jitter : ±30% pour éviter le thundering herd
                    jitter = 1.0 + random.uniform(-0.3, 0.3)
                    wait = max(1.0, wait * jitter)

                    if raw_wait > max_wait:
                        logger.warning(
                            "Groq rate limit (429) — tentative %d/%d, "
                            "retry-after=%ds (plafonné à %ds+variation=%.0fs)",
                            attempt, max_retries, raw_wait, max_wait, wait,
                        )
                    else:
                        logger.warning(
                            "Groq rate limit (429) — tentative %d/%d, attente %.0fs",
                            attempt, max_retries, wait,
                        )
                    if attempt < max_retries:
                        time.sleep(wait)
                        continue
                    else:
                        logger.error("Groq rate limit — toutes les tentatives épuisées")
                        return LLMResponse(
                            content=f"[Groq API Error: Rate limit dépassé après {max_retries} tentatives]",
                            model=model_name,
                            provider_name="groq",
                            finish_reason="error",
                            time_ms=elapsed,
                        )

                resp.raise_for_status()
                data = resp.json()
                break  # Succès

            except httpx.HTTPStatusError as exc:
                elapsed = int((time.time() - start) * 1000)
                if exc.response.status_code == 429 and attempt < max_retries:
                    wait = min(base_delay * attempt, max_wait)
                    jitter = 1.0 + random.uniform(-0.3, 0.3)
                    wait = max(1.0, wait * jitter)
                    logger.warning(
                        "Groq rate limit (429, httpx) — tentative %d/%d, attente %.0fs",
                        attempt, max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error("Groq API error after %d ms: %s", elapsed, exc)
                return LLMResponse(
                    content=f"[Groq API Error: {exc}]",
                    model=model_name,
                    provider_name="groq",
                    finish_reason="error",
                    time_ms=elapsed,
                )
            except Exception as exc:
                elapsed = int((time.time() - start) * 1000)
                logger.error("Groq API error after %d ms: %s", elapsed, exc)
                return LLMResponse(
                    content=f"[Groq API Error: {exc}]",
                    model=model_name,
                    provider_name="groq",
                    finish_reason="error",
                    time_ms=elapsed,
                )

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Groq peut retourner des modèles avec préfixe qu'on nettoie
        actual_model = data.get("model", model_name)

        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason", "stop")

        cost = estimate_cost(model_name, prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=actual_model,
            provider_name="groq",
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_ms=elapsed,
            cost_usd=cost,
            metadata={
                "model": model_name,
                "actual_model": actual_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            },
        )

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()

    def count_tokens(self, text: str) -> int:
        """
        Estimation du nombre de tokens.

        Approximation : 1 token ≈ 4 caractères pour les langues latines.
        """
        return max(1, len(text) // 4)


# ── Factory — build_llm() ────────────────────────────────────────────────────

def build_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory — construit le bon provider à partir de l'environnement.

    Ordre de priorité automatique (si provider non spécifié) :
      1. GROQ_API_KEY       → GroqProvider
      2. ANTHROPIC_API_KEY  → ClaudeProvider
      3. OPENAI_API_KEY     → OpenAIProvider
      4. GEMINI_API_KEY     → GeminiProvider
      5. Sinon              → OpenAIProvider (mode dégradé)

    Args:
        provider : forcer un provider ("openai", "gemini", "claude", "ollama").
        model    : forcer un modèle spécifique.
        **kwargs : paramètres passés au constructeur du provider.

    Exemples :
        # Automatique (priorité ANTHROPIC > OPENAI > GEMINI)
        llm = build_llm()

        # Forcer un provider
        llm = build_llm(provider="openai", model="gpt-4o-mini")

        # Provider avec modèle custom
        llm = build_llm(provider="gemini", model="gemini-1.5-pro")
    """
    provider_name = provider

    if provider_name is None:
        # Détection automatique — DeepSeek prioritaire
        if _get_env_key("DEEPSEEK_API_KEY"):
            provider_name = "deepseek"
            logger.info("build_llm: détection auto → DeepSeek (DEEPSEEK_API_KEY trouvée)")
        elif _get_env_key("GROQ_API_KEY"):
            provider_name = "groq"
            logger.info("build_llm: détection auto → Groq (GROQ_API_KEY trouvée)")
        elif _get_env_key("ANTHROPIC_API_KEY"):
            provider_name = "claude"
            logger.info("build_llm: détection auto → Claude (ANTHROPIC_API_KEY trouvée)")
        elif _get_env_key("OPENAI_API_KEY"):
            provider_name = "openai"
            logger.info("build_llm: détection auto → OpenAI (OPENAI_API_KEY trouvée)")
        elif _get_env_key("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            provider_name = "gemini"
            logger.info("build_llm: détection auto → Gemini (GEMINI_API_KEY trouvée)")
        else:
            provider_name = "openai"
            logger.warning(
                "build_llm: aucune clé API trouvée. "
                "Fallback OpenAIProvider (mode dégradé). "
                "Configurez DEEPSEEK_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY ou GEMINI_API_KEY dans .env"
            )

    provider_name = provider_name.lower()

    if provider_name == "openai":
        m = model or _DEFAULT_OPENAI_MODEL
        return OpenAIProvider(model=m, **kwargs)

    elif provider_name == "gemini":
        m = model or _DEFAULT_GEMINI_MODEL
        return GeminiProvider(model=m, **kwargs)

    elif provider_name == "claude":
        m = model or _DEFAULT_CLAUDE_MODEL
        return ClaudeProvider(model=m, **kwargs)

    elif provider_name == "ollama":
        m = model or "llama3"
        return OllamaProvider(model=m, **kwargs)

    elif provider_name == "deepseek":
        m = model or "deepseek-chat"
        return DeepSeekProvider(model=m, **kwargs)

    elif provider_name == "groq":
        m = model or _DEFAULT_GROQ_MODEL
        return GroqProvider(model=m, **kwargs)

    else:
        logger.warning(
            "Provider '%s' inconnu. Fallback OpenAI. "
            "Providers disponibles: openai, gemini, claude, ollama, deepseek",
            provider_name,
        )
        return OpenAIProvider(model=model or _DEFAULT_OPENAI_MODEL, **kwargs)
