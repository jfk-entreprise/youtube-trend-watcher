"""
Tests unitaires pour le Rewrite Engine (Sprint 22, migré Sprint 31.1 —
réécriture des répliques par scène, plus de hook/introduction/conclusion/
call_to_action séparés : la première scène joue le rôle du hook, la dernière
celui du CTA).

Teste :
  1. RewriteEngine — création, nom, stats
  2. build_user_prompt — contient la critique et les contraintes
  3. extract_json — extraction robuste
  4. build_script_from_json — reconstruction stricte (sujet/marque/durée/
     scènes/personnages préservés), rejets sur divergence de structure
  5. rewrite — garde la version améliorée si le score augmente
  6. rewrite — garde l'originale si le score n'augmente pas
  7. rewrite — jamais d'exception, même en cas d'échec LLM
  8. Découplage — n'importe aucun moteur interne
"""

from pathlib import Path

import pytest

from src.rewrite_engine import RewriteEngine
from src.llm_script_evaluator import LLMScriptScore
from src.script_engine import Dialogue, Scene, SceneDescription, Script, ScriptScene, estimate_scene_duration


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


def _description(setting: str) -> SceneDescription:
    return SceneDescription(
        setting=setting,
        composition="Composition equilibree.",
        characters="Narrateur uniquement.",
        lighting="Lumiere dure et contrastee.",
        camera="Plan fixe.",
        mood="Tension.",
        symbolism="Le secret cache derriere le decor.",
        director_notes="Garder le rythme, ne pas trainer.",
        viewer_emotion="Curiosite.",
    )


@pytest.fixture
def sample_script() -> Script:
    return Script(
        title="5 secrets de production que les studios cachent",
        scenes=[
            ScriptScene(
                scene=Scene(number=1, type="hook", description=_description("Plan choc, gros plan, lumiere dure.")),
                dialogues=[Dialogue(personnage="NARRATEUR", replique="Voici un secret que les studios cachent.")],
                transition="Fade-in", duration_seconds=8,
            ),
            ScriptScene(
                scene=Scene(number=2, type="development", description=_description("Tete parlante, fond neutre.")),
                dialogues=[Dialogue(personnage="NARRATEUR", replique="Ce secret concerne le montage.")],
                transition="Crossfade", duration_seconds=12,
            ),
            ScriptScene(
                scene=Scene(number=3, type="cta", description=_description("Demo a l'ecran, overlay.")),
                dialogues=[Dialogue(personnage="NARRATEUR", replique="Voici comment ca fonctionne en pratique.")],
                transition="Zoom", duration_seconds=14,
            ),
        ],
        estimated_duration=105,
        language="fr",
        target_audience="Cinephiles curieux",
        style="Innovant",
        metadata={"generator": "llm_v1", "niche": "Cinema"},
    )


@pytest.fixture
def multi_dialogue_script() -> Script:
    """Script avec plusieurs dialogues (personnages) dans une même scène."""
    return Script(
        title="Le roi et le conseiller",
        scenes=[
            ScriptScene(
                scene=Scene(number=1, type="scene", description=_description("Salle du trone.")),
                dialogues=[
                    Dialogue(personnage="Roi", replique="Que se passe-t-il ?"),
                    Dialogue(personnage="Conseiller", replique="Sire, une nouvelle grave."),
                ],
                transition="Cut", duration_seconds=10,
            ),
        ],
        estimated_duration=10,
        language="fr",
        target_audience="Tous",
        style="Dramatique",
        metadata={},
    )


@pytest.fixture
def valid_rewrite_json(sample_script):
    return {
        "scenes": [
            {
                "order": s.order,
                "dialogues": [
                    {"personnage": d.personnage, "replique": f"Nouvelle replique pour la scene {s.order}."}
                    for d in s.dialogues
                ],
            }
            for s in sample_script.scenes
        ],
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
        assert "subject" in prompt.lower()
        assert "scene count" in prompt.lower()

    def test_prompt_contains_scene_dialogues(self, sample_script):
        evaluation = _make_score(60)
        prompt = RewriteEngine._build_user_prompt(sample_script, evaluation)
        assert "Voici un secret que les studios cachent." in prompt

    def test_prompt_language_follows_script_language(self, sample_script):
        """Sprint 34 — la langue des repliques réécrites suit script.language,
        elle n'est plus hardcodée en français."""
        import dataclasses
        evaluation = _make_score(60)

        fr_prompt = RewriteEngine._build_user_prompt(sample_script, evaluation)
        assert "Write repliques in French." in fr_prompt

        en_script = dataclasses.replace(sample_script, language="en")
        en_prompt = RewriteEngine._build_user_prompt(en_script, evaluation)
        assert "Write repliques in English." in en_prompt


# ── Tests : extract_json ────────────────────────────────────────────────────────

class TestExtractJson:
    def test_raw_json(self):
        text = '{"scenes": []}'
        assert RewriteEngine._extract_json(text) == text

    def test_markdown_block(self):
        text = '```json\n{"scenes": []}\n```'
        assert RewriteEngine._extract_json(text) == '{"scenes": []}'


# ── Tests : build_script_from_json ──────────────────────────────────────────────

class TestBuildScriptFromJson:
    def test_valid_rewrite_preserves_subject_brand(self, sample_script, valid_rewrite_json):
        result = RewriteEngine._build_script_from_json(valid_rewrite_json, sample_script)
        assert isinstance(result, Script)
        # Sujet / identité inchangés
        assert result.title == sample_script.title
        # Marque inchangée
        assert result.language == sample_script.language
        assert result.style == sample_script.style
        assert result.target_audience == sample_script.target_audience
        assert len(result.scenes) == len(sample_script.scenes)
        for original, new in zip(sample_script.scenes, result.scenes):
            assert new.order == original.order
            assert new.scene == original.scene
            assert new.transition == original.transition
            assert len(new.dialogues) == len(original.dialogues)
            for orig_d, new_d in zip(original.dialogues, new.dialogues):
                assert new_d.personnage == orig_d.personnage  # jamais modifié

    def test_duration_recomputed_from_new_dialogues_not_preserved(self, sample_script, valid_rewrite_json):
        """
        Sprint 37 — duration_seconds/estimated_duration ne sont plus
        recopiés depuis l'original : ils reflètent le texte RÉÉCRIT
        (plafonné à MAX_SCENE_DURATION_SECONDS/scène), pour ne jamais
        mentir sur la durée réelle d'une scène réécrite.
        """
        result = RewriteEngine._build_script_from_json(valid_rewrite_json, sample_script)
        for new in result.scenes:
            expected = estimate_scene_duration(new.dialogues)
            assert new.duration_seconds == expected
        assert result.estimated_duration == sum(s.duration_seconds for s in result.scenes)

    def test_rewrite_replaces_replique_text(self, sample_script, valid_rewrite_json):
        result = RewriteEngine._build_script_from_json(valid_rewrite_json, sample_script)
        for original, new in zip(sample_script.scenes, result.scenes):
            for orig_d, new_d in zip(original.dialogues, new.dialogues):
                assert new_d.replique != orig_d.replique  # seule la replique change

    def test_rewrite_updates_hook_and_cta(self, sample_script, valid_rewrite_json):
        """hook/call_to_action sont dérivés des scènes première/dernière —
        elles doivent refléter les nouvelles répliques."""
        result = RewriteEngine._build_script_from_json(valid_rewrite_json, sample_script)
        assert result.hook == "Nouvelle replique pour la scene 1."
        assert result.call_to_action == "Nouvelle replique pour la scene 3."

    def test_multi_dialogue_scene_preserves_personnage_and_count(self, multi_dialogue_script):
        rewrite = {
            "scenes": [{
                "order": 1,
                "dialogues": [
                    {"personnage": "Roi", "replique": "Nouvelle question du roi ?"},
                    {"personnage": "Conseiller", "replique": "Nouvelle reponse du conseiller."},
                ],
            }],
        }
        result = RewriteEngine._build_script_from_json(rewrite, multi_dialogue_script)
        assert len(result.scenes[0].dialogues) == 2
        assert result.scenes[0].dialogues[0].personnage == "Roi"
        assert result.scenes[0].dialogues[0].replique == "Nouvelle question du roi ?"
        assert result.scenes[0].dialogues[1].personnage == "Conseiller"

    def test_missing_field_raises(self, sample_script):
        with pytest.raises(ValueError, match="scenes"):
            RewriteEngine._build_script_from_json({}, sample_script)

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

    def test_dialogue_count_mismatch_raises(self, multi_dialogue_script):
        rewrite = {
            "scenes": [{
                "order": 1,
                "dialogues": [{"personnage": "Roi", "replique": "Une seule replique."}],  # 1 au lieu de 2
            }],
        }
        with pytest.raises(ValueError, match="dialogues"):
            RewriteEngine._build_script_from_json(rewrite, multi_dialogue_script)


# ── Tests : rewrite (décision garder/rejeter) ──────────────────────────────────

class TestRewriteDecision:
    def test_keeps_improved_version(self, sample_script, valid_rewrite_json):
        engine = RewriteEngine(evaluator=_StubEvaluator(75))
        engine._try_rewrite = lambda script, evaluation: RewriteEngine._build_script_from_json(
            valid_rewrite_json, script
        )
        original_eval = _make_score(60)
        result = engine.rewrite(sample_script, original_eval)

        assert result.hook == "Nouvelle replique pour la scene 1."
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

    def test_preserves_subject_brand_end_to_end(self, sample_script, valid_rewrite_json):
        """Le pipeline complet garde le sujet/marque même sur une version retenue."""
        engine = RewriteEngine(evaluator=_StubEvaluator(80))
        engine._try_rewrite = lambda script, evaluation: RewriteEngine._build_script_from_json(
            valid_rewrite_json, script
        )
        result = engine.rewrite(sample_script, _make_score(60))
        assert result.title == sample_script.title
        assert result.language == sample_script.language
        assert len(result.scenes) == len(sample_script.scenes)
        # Sprint 37 — la durée est recalculée depuis le texte réécrit, pas préservée
        assert result.estimated_duration == sum(s.duration_seconds for s in result.scenes)


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
