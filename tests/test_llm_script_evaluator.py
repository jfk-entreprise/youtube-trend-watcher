"""
Tests unitaires pour le LLM Script Evaluator (Sprint 21).

Teste :
  1. LLMScriptScore — création, frozen, global_score
  2. LLMScriptEvaluator — création, nom, stats
  3. build_user_prompt — prompt bien formé
  4. extract_json — extraction robuste dans tous les cas
  5. validate_json_structure — validation stricte
  6. build_score_from_json — reconstruction complète, recalcul du global_score
  7. evaluate — échec LLM (pas de provider configuré en environnement de test)
  8. BaseEvaluator — polymorphisme avec ScriptEvaluator
  9. Découplage — n'importe aucun moteur interne
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.llm_script_evaluator import LLMScriptEvaluator, LLMScriptScore, _CRITERIA
from src.script_evaluator import BaseEvaluator, ScriptEvaluator
from src.script_engine import Script, ScriptScene


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_script() -> Script:
    return Script(
        title="Ce jeu de camouflage casse les regles de l'esport",
        hook="Et si le pire spot de camouflage devenait une arme secrete ?",
        introduction="C'est ce qui vient de se passer dans la scene esport.",
        scenes=[
            ScriptScene(order=1, title="Hook", narration="Et si le pire spot devenait une arme secrete ?",
                        visual_description="Plan choc", image_prompt="Bold composition",
                        animation_notes="Fade-in", sound_effects="Whoosh", duration_seconds=8),
            ScriptScene(order=2, title="Contexte", narration="Voici ce que personne n'avait remarque.",
                        visual_description="Tete parlante", image_prompt="Clean setup",
                        animation_notes="Crossfade", sound_effects="Music", duration_seconds=12),
            ScriptScene(order=3, title="Developpement", narration="Le camouflage change tout sur le terrain.",
                        visual_description="Demo gameplay", image_prompt="Gameplay overlay",
                        animation_notes="Zoom", sound_effects="Impact", duration_seconds=14),
        ],
        conclusion="Ce detail change la meta du jeu.",
        call_to_action="Quel spot debile as-tu deja vu ? Dis-le en commentaire.",
        estimated_duration=105,
        language="fr",
        target_audience="Joueurs esport",
        style="Innovant",
        metadata={"generator": "llm_v1"},
    )


@pytest.fixture
def valid_llm_json():
    return {
        "hook": 8,
        "curiosity": 7,
        "storytelling": 8,
        "rhythm": 7,
        "clarity": 9,
        "cta": 8,
        "retention": 8,
        "viral": 7,
        "global_score": 62,
        "strengths": ["Hook percutant", "CTA specifique"],
        "weaknesses": ["Rythme un peu lent au milieu"],
        "suggestions": ["Raccourcir la scene 2"],
    }


# ── Tests : LLMScriptScore ────────────────────────────────────────────────────

class TestLLMScriptScore:
    def test_creation(self):
        score = LLMScriptScore(
            hook=8, curiosity=7, storytelling=8, rhythm=7, clarity=9,
            cta=8, retention=8, viral=7, global_score=62,
        )
        assert score.hook == 8
        assert score.global_score == 62
        assert score.strengths == []

    def test_frozen(self):
        score = LLMScriptScore(
            hook=8, curiosity=7, storytelling=8, rhythm=7, clarity=9,
            cta=8, retention=8, viral=7, global_score=62,
        )
        with pytest.raises(FrozenInstanceError):
            score.hook = 10

    def test_is_polymorphic_with_base_evaluator_contract(self):
        """LLMScriptScore expose .global_score, comme ScriptScore."""
        score = LLMScriptScore(
            hook=8, curiosity=7, storytelling=8, rhythm=7, clarity=9,
            cta=8, retention=8, viral=7, global_score=62,
        )
        assert hasattr(score, "global_score")


# ── Tests : création LLMScriptEvaluator ───────────────────────────────────────

class TestCreation:
    def test_default_creation(self):
        ev = LLMScriptEvaluator()
        assert ev is not None
        assert "llm_judge" in ev.name
        assert ev.stats["llm_calls"] == 0

    def test_custom_provider(self):
        ev = LLMScriptEvaluator(provider_name="claude")
        assert "claude" in ev.name

    def test_custom_model(self):
        ev = LLMScriptEvaluator(model="gpt-4o-mini")
        assert "gpt-4o-mini" in ev.name

    def test_low_default_temperature(self):
        """Temperature basse par defaut pour un jugement consistant."""
        ev = LLMScriptEvaluator()
        assert ev._temperature <= 0.5

    def test_stats_immutable_copy(self):
        ev = LLMScriptEvaluator()
        stats = ev.stats
        stats["llm_calls"] = 999
        assert ev.stats["llm_calls"] == 0

    def test_is_base_evaluator(self):
        ev = LLMScriptEvaluator()
        assert isinstance(ev, BaseEvaluator)


# ── Tests : build_user_prompt ─────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_prompt_contains_script_content(self, sample_script):
        prompt = LLMScriptEvaluator._build_user_prompt(sample_script)
        assert sample_script.title in prompt
        assert sample_script.hook in prompt
        assert sample_script.call_to_action in prompt
        assert "Developpement" in prompt

    def test_prompt_contains_all_scenes(self, sample_script):
        prompt = LLMScriptEvaluator._build_user_prompt(sample_script)
        for scene in sample_script.scenes:
            assert scene.narration in prompt


# ── Tests : extract_json ──────────────────────────────────────────────────────

class TestExtractJson:
    def test_raw_json(self):
        text = '{"hook": 8}'
        assert LLMScriptEvaluator._extract_json(text) == text

    def test_markdown_json_block(self):
        text = '```json\n{"hook": 8}\n```'
        assert LLMScriptEvaluator._extract_json(text) == '{"hook": 8}'

    def test_text_around_json(self):
        text = 'Voici le resultat:\n{"hook": 8}\nCest tout.'
        assert LLMScriptEvaluator._extract_json(text) == '{"hook": 8}'


# ── Tests : validate_json_structure ───────────────────────────────────────────

class TestValidateJsonStructure:
    def test_valid_json(self, valid_llm_json):
        LLMScriptEvaluator._validate_json_structure(valid_llm_json)

    def test_missing_criterion(self, valid_llm_json):
        data = dict(valid_llm_json)
        del data["hook"]
        with pytest.raises(ValueError, match="hook"):
            LLMScriptEvaluator._validate_json_structure(data)

    def test_criterion_not_a_number(self, valid_llm_json):
        data = dict(valid_llm_json)
        data["hook"] = "huit"
        with pytest.raises(ValueError, match="nombre"):
            LLMScriptEvaluator._validate_json_structure(data)

    def test_criterion_out_of_range(self, valid_llm_json):
        data = dict(valid_llm_json)
        data["hook"] = 15
        with pytest.raises(ValueError, match="entre 0 et 10"):
            LLMScriptEvaluator._validate_json_structure(data)

    def test_missing_strengths(self, valid_llm_json):
        data = dict(valid_llm_json)
        del data["strengths"]
        with pytest.raises(ValueError, match="strengths"):
            LLMScriptEvaluator._validate_json_structure(data)

    def test_weaknesses_not_a_list(self, valid_llm_json):
        data = dict(valid_llm_json)
        data["weaknesses"] = "pas une liste"
        with pytest.raises(ValueError, match="liste"):
            LLMScriptEvaluator._validate_json_structure(data)

    def test_all_criteria_present(self):
        assert set(_CRITERIA) == {
            "hook", "curiosity", "storytelling", "rhythm",
            "clarity", "cta", "retention", "viral",
        }


# ── Tests : build_score_from_json ─────────────────────────────────────────────

class TestBuildScoreFromJson:
    def test_full_reconstruction(self, valid_llm_json):
        score = LLMScriptEvaluator._build_score_from_json(
            valid_llm_json, llm_provider="deepseek", llm_model="deepseek-chat",
            response_time_ms=1200, response_tokens=500, response_cost=0.002,
        )
        assert isinstance(score, LLMScriptScore)
        assert score.hook == 8
        assert score.curiosity == 7
        assert score.strengths == ["Hook percutant", "CTA specifique"]
        assert score.metadata["llm_provider"] == "deepseek"

    def test_global_score_is_recomputed_not_trusted(self, valid_llm_json):
        """global_score est toujours recalculé côté serveur (somme des 8 critères)."""
        data = dict(valid_llm_json)
        data["global_score"] = 999  # valeur aberrante fournie par le LLM
        score = LLMScriptEvaluator._build_score_from_json(data)
        expected = sum(data[c] for c in _CRITERIA)
        assert score.global_score == expected
        assert score.global_score != 999


# ── Tests : evaluate (échec sans provider configuré) ─────────────────────────

class TestEvaluateFailure:
    def test_evaluate_raises_without_provider(self, sample_script):
        """
        Sans clé API dans l'environnement de test, build_llm() échoue et
        evaluate() lève une RuntimeError après max_retries tentatives
        (pas de fallback heuristique — ce module EST la variante LLM).
        """
        ev = LLMScriptEvaluator(max_retries=2)
        with pytest.raises(RuntimeError):
            ev.evaluate(sample_script)
        assert ev.stats["llm_failures"] == 2

    def test_evaluate_custom_retries(self, sample_script):
        ev = LLMScriptEvaluator(max_retries=1)
        with pytest.raises(RuntimeError):
            ev.evaluate(sample_script)
        assert ev.stats["llm_failures"] == 1


# ── Tests : polymorphisme avec ScriptEvaluator ────────────────────────────────

class TestPolymorphism:
    def test_both_evaluators_share_base(self):
        assert isinstance(ScriptEvaluator(), BaseEvaluator)
        assert isinstance(LLMScriptEvaluator(), BaseEvaluator)

    def test_both_expose_global_score(self, sample_script):
        heuristic_score = ScriptEvaluator().evaluate(sample_script)
        llm_score = LLMScriptScore(
            hook=8, curiosity=7, storytelling=8, rhythm=7, clarity=9,
            cta=8, retention=8, viral=7, global_score=62,
        )
        assert isinstance(heuristic_score.global_score, float)
        assert isinstance(llm_score.global_score, (int, float))

    def test_both_expose_name(self):
        assert isinstance(ScriptEvaluator().name, str)
        assert isinstance(LLMScriptEvaluator().name, str)


# ── Tests : Découplage ────────────────────────────────────────────────────────

class TestDecoupling:
    def test_no_video_snapshot(self):
        with pytest.raises(ImportError):
            from src.llm_script_evaluator import VideoSnapshot  # type: ignore

    def test_no_virality_engine(self):
        with pytest.raises(ImportError):
            from src.llm_script_evaluator import ViralityEngine  # type: ignore

    def test_no_opportunity_engine(self):
        with pytest.raises(ImportError):
            from src.llm_script_evaluator import OpportunityEngine  # type: ignore

    def test_no_creative_engine(self):
        with pytest.raises(ImportError):
            from src.llm_script_evaluator import CreativeEngine  # type: ignore

    def test_no_brand_engine(self):
        with pytest.raises(ImportError):
            from src.llm_script_evaluator import BrandProfile  # type: ignore

    def test_only_contracts_used(self):
        mod_src = Path(__file__).resolve().parent.parent / "src" / "llm_script_evaluator.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "from src.script_engine import Script" in content
        assert "from src.script_evaluator import BaseEvaluator" in content
        assert "from src.opportunity_engine" not in content
        assert "from src.virality_engine" not in content
        assert "from src.learning_engine" not in content
        assert "from src.brand_engine" not in content
        assert "from src.creative_engine" not in content

    def test_llm_only_dependency(self):
        mod_src = Path(__file__).resolve().parent.parent / "src" / "llm_script_evaluator.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "from src.llm import" in content
        assert "build_llm" in content
