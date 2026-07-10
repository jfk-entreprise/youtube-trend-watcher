"""
Script de test et démonstration du LLM Provider (Sprint 16).

Envoie "Réponds uniquement : OK" à chaque provider configuré
et affiche :
  - provider
  - modèle
  - temps de réponse
  - coût estimé
  - contenu reçu
  - tokens

Usage :
    python scripts/test_llm.py                        # Test auto (provider prioritaire)
    python scripts/test_llm.py --all                   # Test tous les providers configurés
    python scripts/test_llm.py --provider openai       # Test un provider spécifique
    python scripts/test_llm.py --model gpt-4o          # Forcer un modèle
    python scripts/test_llm.py --prompt "Dis bonjour"  # Prompt personnalisé
    python scripts/test_llm.py --json                  # Tester le mode JSON
    python scripts/test_llm.py --verbose               # Réponse complète (JSON)

Ce script permet de vérifier rapidement que l'infrastructure LLM
est fonctionnelle avant d'intégrer les générateurs LLM (templates,
scripts, etc.).

Aucun moteur du projet n'est importé — test isolé du LLM Provider.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

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
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Provider — Test de l'infrastructure IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all", action="store_true",
                        help="Tester tous les providers configurés")
    parser.add_argument("--provider", type=str, default=None,
                        help="Provider à tester (openai, gemini, claude)")
    parser.add_argument("--model", type=str, default=None,
                        help="Forcer un modèle spécifique")
    parser.add_argument("--prompt", type=str, default="Réponds uniquement : OK",
                        help="Prompt à envoyer (défaut: 'Réponds uniquement : OK')")
    parser.add_argument("--json", action="store_true",
                        help="Tester le mode JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Afficher la réponse complète (JSON)")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  LLM PROVIDER — Test Sprint 16")
    print("=" * 72)

    message = LLMMessage(role="user", content=args.prompt)

    if args.all:
        providers_to_test = _get_all_available_providers()
    else:
        providers_to_test = {args.provider: None} if args.provider else {}

    if not providers_to_test:
        providers_to_test = {"auto": None}

    for prov_name, prov_model in providers_to_test.items():
        model_override = prov_model or args.model

        if prov_name == "auto":
            _test_provider(build_llm(model=model_override), message, args)
        else:
            try:
                provider = build_llm(provider=prov_name, model=model_override)
                _test_provider(provider, message, args)
            except Exception as e:
                print(f"\n  ✗ {prov_name.upper():12s} → Erreur d'initialisation : {e}")

    print()
    print("=" * 72)
    print("  TEST TERMINÉ")
    print("=" * 72)
    print()


def _test_provider(
    provider: LLMProvider,
    message: LLMMessage,
    args: argparse.Namespace,
) -> None:
    """Teste un provider et affiche les résultats."""
    print(f"\n  -- {provider.name.upper()} / {provider.model} --")

    messages = [message]

    if args.json:
        messages = [
            LLMMessage(role="system", content=(
                "Tu es un assistant qui répond UNIQUEMENT en JSON valide. "
                "Ta réponse doit être un objet JSON avec une clé 'response'."
            )),
            message,
        ]

    start = time.time()
    response = provider.generate(
        messages,
        temperature=0.1,
        max_tokens=50,
        json_mode=args.json,
    )
    elapsed = int((time.time() - start) * 1000)

    status = "OK" if response.finish_reason == "stop" else "ERR"

    print(f"    [{status}] Modele          : {response.model}")
    print(f"      Provider        : {response.provider_name}")
    print(f"      Temps           : {response.time_ms} ms ({elapsed} ms reel)")
    print(f"      Tokens          : {response.prompt_tokens} -> {response.completion_tokens} (total: {response.total_tokens})")
    print(f"      Cout            : ${response.cost_usd:.6f} USD")
    print(f"      Raison          : {response.finish_reason}")

    if args.verbose:
        print(f"      Reponse (brute)  :")
        print(f"        {json.dumps(response.content, ensure_ascii=False)}")
    else:
        # Troncature pour l'affichage
        preview = response.content[:100].replace("\n", " ")
        print(f"      Reponse courte   : {preview}")

    if args.json:
        try:
            parsed = json.loads(response.content)
            print(f"      JSON valide      : OK -> {json.dumps(parsed, ensure_ascii=False)[:80]}")
        except json.JSONDecodeError:
            print(f"      JSON valide      : ERR (pas du JSON valide)")


def _get_all_available_providers() -> dict:
    """
    Retourne tous les providers pour lesquels une clé API est disponible.
    """
    import os
    providers = {}

    # Vérifier les clés dans l'ordre de priorité
    if os.environ.get("DEEPSEEK_API_KEY"):
        providers["deepseek"] = None
    if os.environ.get("GROQ_API_KEY"):
        providers["groq"] = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers["claude"] = None
    if os.environ.get("OPENAI_API_KEY"):
        providers["openai"] = None
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        providers["gemini"] = None

    if not providers:
        print("  [!] Aucune cle API trouvee dans l'environnement.")
        print("    Test en mode dégradé (OpenAIProvider sans clé).")
        providers["openai"] = None

    return providers


if __name__ == "__main__":
    main()
