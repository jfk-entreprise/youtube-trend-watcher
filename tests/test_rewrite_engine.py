"""
Tests unitaires pour le Rewrite Engine (Sprint 22).

Teste :
  1. RewriteEngine — création, nom, stats
  2. build_user_prompt — contient la critique et les contraintes
  3. extract_json — extraction robuste
  4. build_script_from_json — reconstruction stricte (sujet/marque/durée/
     scènes préservés), rejets sur divergence de structure
  5. rewrite — garde la version améliorée si le score augmente
  6. rewrite — garde l'originale si le score n'augmente pas
  7. rewrite — jamais d'exception, même en cas d'échec LLM
  8. Découplage — n'importe aucun moteur interne
"""

from pathlib import Path

import pytest

from src.rewrite_engine import RewriteEngine
from src.llm_script_evaluator import LLMScriptScore
from src.script_engine import Script, ScriptScene


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_score(global_score: float, weaknesses=None, suggestions=None) -> LLMScriptScore:
    return LLMScriptScore(
        hook=6, curiosity=6, storytelling=6, rhythm=6, clarity=6,
        cta=6, retention=6, viral=6, global_score=global_score,
        weaknesses=weaknesses or ["Hook trop generique"],
        suggestions=suggestions or ["Ouvrir sur une question directe"],
    )


class _StubEvaluator:
    """Évaluateur factice, retourne un score fixe sans appel LLM."""
    def __init__(self, score_value: float) -> None:
        self.score_value = score_value

    def evaluate(self, script: Script) -> LLMScriptScore:
        return _make_score(self.score_value)


@pytest.fixture
def sample_script() -> Script:
    return Script(
        title="5 secrets de production que les studios cachent",
        hook="Voici un secret que les studios ne veulent pas que tu saches.",
        introduction="On va decortiquer ce que personne ne dit jamais.",
        scenes=[
            ScriptScene(order=1, title="Hook", narration="Voici un secret que les studios cachent.",
                        visual_description="Plan choc", image_prompt="Bold composition",
                        animation_notes="Fade-in", sound_effects="Whoosh", duration_seconds=8),
            ScriptScene(order=2, title="Contexte", narration="Ce secret concerne le montage.",
                        visual_description="Tete parlante", image_prompt="Clean setup",
                        animation_notes="Crossfade", sound_effects="Music", duration_seconds=12),
            ScriptScene(order=3, title="Developpement", narration="Voici comment ca fonctionne en pratique.",
                        visual_description="Demo", image_prompt="Overlay demo",
                        animation_notes="Zoom", sound_effects="Impact", duration_seconds=14),
        ],
        conclusion="Ce detail change la facon de voir un film.",
        call_to_action="Quel autre secret veux-tu qu'on decortique ? Dis-le en commentaire.",
        estimated_duration=105,
        language="fr",
        target_audience="Cinephiles curieux",
        style="Innovant",
        metadata={"generator": "llm_v1", "niche": "Cinema"},
    )


@pytest.fixture
def valid_rewrite_json(sample_script):
    return {
        "hook": "Et si ce detail de montage changeait tout le film ?",
        "introduction": "Personne n'en parle, pourtant c'est partout.",
        "scenes": [
            {"order": s.order, "narration": f"Nouvelle narration pour la scene {s.order}."}
            for s in sample_script.scenes
        ],
        "conclusion": "Desormais tu ne regarderas plus un film pareil.",
        "call_to_action": "Quel film veux-tu qu'on decortique ensuite ? Dis-le en commentaire.",
    }


# ── Tests : création ───────────────────────────────────────────────────────────

class TestCreation:
    def test_default_creation(self):
        engine = RewriteEngine()
        assert engine is not None
        assert "rewrite_engine" in engine.name
        assert engine.stats["rewrite_attempts"] == 0

    def test_custom_provider(self):
        engine = RewriteEngine(provider_name="claude")
        assert "claude" in engine.name

    def test_custom_evaluator_injected(self):
        stub = _StubEvaluator(50)
        engine = RewriteEngine(evaluator=stub)
        assert engine._evaluator is stub

    def test_stats_immutable_copy(self):
        engine = RewriteEngine()
        stats = engine.stats
        stats["rewrite_attempts"] = 999
        assert engine.stats["rewrite_attempts"] == 0


# ── Tests : build_user_prompt ──────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_prompt_contains_critique(self, sample_script):
        evaluation = _make_score(60, weaknesses=["Hook faible"], suggestions=["Ouvre par un chiffre"])
        prompt = RewriteEngine._build_user_prompt(sample_script, evaluation)
        assert "Hook faible" in prompt
        assert "Ouvre par un chiffre" in prompt
        assert "60" in prompt

    def test_prompt_lists_scene_orders(self, sample_script):
        evaluation = _make_score(60)
        prompt = RewriteEngine._build_user_prompt(sample_script, evaluation)
        assert "[1, 2, 3]" in prompt

    def test_prompt_forbids_changing_structure(self, sample_script):
        evaluation = _make_score(60)
        prompt = RewriteEngine._build_user_prompt(sample_script, evaluation)
        assert "sujet" in prompt.lower()
        assert "nombre de scenes" in prompt.lower() or "nombre de scènes" in prompt.lower()


# ── Tests : extract_json ────────────────────────────────────────────────────────

class TestExtractJson:
    def test_raw_json(self):
        text = '{"hook": "x"}'
        assert RewriteEngine._extract_json(text) == text

    def test_markdown_block(self):
        text = '```json\n{"hook": "x"}\n```'
        assert RewriteEngine._extract_json(text) == '{"hook": "x"}'


# ── Tests : build_script_from_json ──────────────────────────────────────────────

class TestBuildScriptFromJson:
    def test_valid_rewrite_preserves_subject_brand_duration(self, sample_script, valid_rewrite_json):
        result = RewriteEngine._build_script_from_json(valid_rewrite_json, sample_script)
        assert isinstance(result, Script)
        # Sujet / identité inchangés
        assert result.title == sample_script.title
        # Marque inchangée
        assert result.language == sample_script.language
        assert result.style == sample_script.style
        assert result.target_audience == sample_script.target_audience
        # Durée inchangée
        assert result.estimated_duration == sample_script.estimated_duration
        assert len(result.scenes) == len(sample_script.scenes)
        for original, new in zip(sample_script.scenes, result.scenes):
            assert new.order == original.order
            assert new.duration_seconds == original.duration_seconds
            assert new.title == original.title
            assert new.visual_description == original.visual_description
            assert new.image_prompt == original.image_prompt
            assert new.animation_notes == original.animation_notes
            assert new.sound_effects == original.sound_effects
            # Seule la narration change
            assert new.narration != original.narration

    def test_rewrite_updates_hook_intro_conclusion_cta(self, sample_script, valid_rewrite_json):
        result = RewriteEngine._build_script_from_json(valid_rewrite_json, sample_script)
        assert result.hook == valid_rewrite_json["hook"]
        assert result.introduction == valid_rewrite_json["introduction"]
        assert result.conclusion == valid_rewrite_json["conclusion"]
        assert result.call_to_action == valid_rewrite_json["call_to_action"]

    def test_missing_field_raises(self, sample_script, valid_rewrite_json):
        data = dict(valid_rewrite_json)
        del data["hook"]
        with pytest.raises(ValueError, match="hook"):
            RewriteEngine._build_script_from_json(data, sample_script)

    def test_scene_count_mismatch_raises(self, sample_script, valid_rewrite_json):
        data = dict(valid_rewrite_json)
        data["scenes"] = data["scenes"][:-1]  # une scène en moins
        with pytest.raises(ValueError, match="scènes"):
            RewriteEngine._build_script_from_json(data, sample_script)

    def test_scene_order_mismatch_raises(self, sample_script, valid_rewrite_json):
        data = dict(valid_rewrite_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        data["scenes"][0]["order"] = 99  # ordre invalide
        with pytest.raises(ValueError, match="[Oo]rdre"):
            RewriteEngine._build_script_from_json(data, sample_script)

    def test_scenes_not_a_list_raises(self, sample_script, valid_rewrite_json):
        data = dict(valid_rewrite_json)
        data["scenes"] = "not a list"
        with pytest.raises(ValueError, match="liste"):
            RewriteEngine._build_script_from_json(data, sample_script)


# ── Tests : rewrite (décision garder/rejeter) ──────────────────────────────────

class TestRewriteDecision:
    def test_keeps_improved_version(self, sample_script, valid_rewrite_json):
        engine = RewriteEngine(evaluator=_StubEvaluator(75))
        engine._try_rewrite = lambda script, evaluation: RewriteEngine._build_script_from_json(
            valid_rewrite_json, script
        )
        original_eval = _make_score(60)
        result = engine.rewrite(sample_script, original_eval)

        assert result.hook == valid_rewrite_json["hook"]
        assert result.metadata["rewritten"] is True
        assert result.metadata["rewrite_score_before"] == 60
        assert result.metadata["rewrite_score_after"] == 75
        assert engine.stats["rewrites_applied"] == 1
        assert engine.stats["rewrites_rejected"] == 0

    def test_keeps_original_if_not_improved(self, sample_script, valid_rewrite_json):
        engine = RewriteEngine(evaluator=_StubEvaluator(55))
        engine._try_rewrite = lambda script, evaluation: RewriteEngine._build_script_from_json(
            valid_rewrite_json, script
        )
        original_eval = _make_score(60)
        result = engine.rewrite(sample_script, original_eval)

        assert result is sample_script
        assert "rewritten" not in result.metadata
        assert engine.stats["rewrites_applied"] == 0
        assert engine.stats["rewrites_rejected"] == 1

    def test_equal_score_keeps_original(self, sample_script, valid_rewrite_json):
        """En cas d'égalité, on ne remplace pas (l'amélioration doit être stricte)."""
        engine = RewriteEngine(evaluator=_StubEvaluator(60))
        engine._try_rewrite = lambda script, evaluation: RewriteEngine._build_script_from_json(
            valid_rewrite_json, script
        )
        original_eval = _make_score(60)
        result = engine.rewrite(sample_script, original_eval)
        assert result is sample_script

    def test_preserves_subject_brand_duration_end_to_end(self, sample_script, valid_rewrite_json):
        """Le pipeline complet garde le sujet/marque/durée même sur une version retenue."""
        engine = RewriteEngine(evaluator=_StubEvaluator(80))
        engine._try_rewrite = lambda script, evaluation: RewriteEngine._build_script_from_json(
            valid_rewrite_json, script
        )
        result = engine.rewrite(sample_script, _make_score(60))
        assert result.title == sample_script.title
        assert result.language == sample_script.language
        assert result.estimated_duration == sample_script.estimated_duration
        assert len(result.scenes) == len(sample_script.scenes)


# ── Tests : rewrite (résilience) ────────────────────────────────────────────────

class TestRewriteResilience:
    def test_never_raises_without_provider_configured(self, sample_script):
        """
        Sans clé API dans l'environnement de test, build_llm() échoue —
        rewrite() ne doit jamais lever d'exception, seulement retomber
        sur le script original.
        """
        engine = RewriteEngine(max_retries=1)
        result = engine.rewrite(sample_script, _make_score(60))
        assert result is sample_script
        assert engine.stats["rewrite_attempts"] == 1
        assert engine.stats["llm_failures"] == 1
        assert engine.stats["rewrites_rejected"] == 1

    def test_invalid_llm_output_falls_back_to_original(self, sample_script):
        engine = RewriteEngine(max_retries=1)
        engine._try_rewrite = lambda script, evaluation: (_ for _ in ()).throw(ValueError("JSON invalide"))
        result = engine.rewrite(sample_script, _make_score(60))
        assert result is sample_script
        assert engine.stats["rewrites_rejected"] == 1


# ── Tests : Découplage ──────────────────────────────────────────────────────────

class TestDecoupling:
    def test_no_virality_engine(self):
        with pytest.raises(ImportError):
            from src.rewrite_engine import ViralityEngine  # type: ignore

    def test_no_opportunity_engine(self):
        with pytest.raises(ImportError):
            from src.rewrite_engine import OpportunityEngine  # type: ignore

    def test_no_creative_engine(self):
        with pytest.raises(ImportError):
            from src.rewrite_engine import CreativeEngine  # type: ignore

    def test_no_brand_engine(self):
        with pytest.raises(ImportError):
            from src.rewrite_engine import BrandProfile  # type: ignore

    def test_only_contracts_used(self):
        mod_src = Path(__file__).resolve().parent.parent / "src" / "rewrite_engine.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "from src.script_engine import Script" in content
        assert "from src.llm_script_evaluator import LLMScriptEvaluator" in content
        assert "from src.llm import" in content
        assert "build_llm" in content
        assert "from src.opportunity_engine" not in content
        assert "from src.virality_engine" not in content
        assert "from src.brand_engine" not in content
        assert "from src.creative_engine" not in content
