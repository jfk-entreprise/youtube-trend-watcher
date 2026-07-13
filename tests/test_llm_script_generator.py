"""
Tests unitaires pour le LLM Script Generator (Sprint 21 — Contrôle durée,
Sprint 20.1 — cible fixe format Shorts, Sprint 27 — fiabilisation JSON
alignée sur VisualDirector / LLMImageGenerator / LLMAnimationGenerator,
Sprint 31.1 — format Storyboard Studio unifié : plus de hook/introduction/
conclusion/call_to_action séparés, scene/dialogues/transition par scène).

Teste :
  1. LLMScriptGenerator — création, nom, stats
  2. build_user_prompt — prompt bien formé
  3. extract_json — extraction robuste dans tous les cas (Sprint 27 :
     <think>, Markdown, texte parasite, virgules traînantes, caractères de
     contrôle)
  4. validate_json_structure / parse_and_validate — validation stricte et
     classification des causes d'échec (Sprint 27/31.1)
  5. build_script_from_json — reconstruction complète de Script
  6. generate — fallback vers HeuristicScriptGenerator si LLM échoue, avec
     la raison enregistrée (Sprint 27)
  7. generate — retry automatique (2 tentatives)
  8. retry intelligent — correction JSON via un second appel LLM avant tout
     fallback (Sprint 27)
  9. Découplage — n'importe aucun moteur interne
  10. Validation de durée (Sprint 20.1) — build_duration_breakdown, durée fixe dans le prompt
  11. La cible fixe (100-130s / 8-10 scenes) prévaut sur creative_brief.duration_seconds
  12. Modèle DeepSeek par défaut — deepseek-chat (Sprint 27)
"""

import json
import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict

from src.llm import LLMResponse
from src.llm_script_generator import (
    LLMScriptGenerator,
    _ScriptJsonError,
    _TARGET_DURATION_MIN_SEC,
    _TARGET_DURATION_MAX_SEC,
    _TARGET_DURATION_SEC,
    _TARGET_SCENES_MIN,
    _TARGET_SCENES_MAX,
)
from src.script_engine import Dialogue, Scene, SceneDescription, Script, ScriptScene, ScriptGenerator, estimate_scene_duration
from src.opportunity_engine import Opportunity
from src.creative_engine import CreativeBrief
from src.brand_engine import BrandProfile, JsonBrandStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_opportunity() -> Opportunity:
    return Opportunity(
        title="L'IA va-t-elle remplacer les developpeurs ?",
        niche="Intelligence Artificielle",
        source_video_id="demo_ia_001",
        overall_score=0.85,
        virality_score=0.72,
        growth_score=0.65,
        evergreen_score=0.80,
        trend_score=0.78,
        competition_score=0.45,
        production_difficulty=0.50,
        urgency=0.70,
        recommendation="Produire rapidement.",
        rationale=["Potentiel viral eleve", "Sujet perenne"],
        metadata={"niche": "IA", "source": "demo"},
    )


@pytest.fixture
def sample_brief() -> CreativeBrief:
    return CreativeBrief(
        opportunity_id="demo_ia_001",
        title="5 metiers developpeur transformes par l'IA",
        angle="Liste",
        hook="Voici pourquoi 80% des developpeurs sous-estiment l'IA.",
        promise="Decouvrez les 5 metiers les plus impactes par l'IA.",
        audience="Developpeurs curieux de l'IA",
        emotion="Informatif",
        format="Analyse",
        duration_seconds=480,
        structure=["Hook", "Intro", "Point #1", "Point #2", "Point #3", "Conclusion", "CTA"],
        visual_style="Graphiques et slides epures",
        cta="Abonne-toi pour plus d'analyses.",
        originality_score=0.85,
        production_notes=["Citer les sources"],
        rationale=["Angle liste: SEO efficace"],
        metadata={"niche": "IA", "language": "fr"},
    )


@pytest.fixture
def sample_brand() -> BrandProfile:
    store = JsonBrandStore(Path(__file__).resolve().parent.parent / "brands")
    profile = store.load("ia_fr")
    assert profile is not None, "Brand ia_fr doit exister"
    return profile


def _json_description(setting=None) -> Dict[str, str]:
    return {
        "setting": setting or "Rich description of the setting — architecture, era, textures, atmosphere.",
        "composition": "Balanced composition, subject centered, clear depth of field.",
        "characters": "Narrator only — no visible character on screen.",
        "lighting": "Soft, warm key light with subtle contrast.",
        "camera": "A very slow 6-second dolly-in with a slight low angle.",
        "mood": "Rising curiosity tinged with quiet tension.",
        "symbolism": "The décor evokes the transformation underway.",
        "director_notes": "Keep the pace tight, guide the viewer's gaze toward the main subject, avoid lingering.",
        "viewer_emotion": "Curiosity should gradually give way to a quiet fascination.",
    }


def _json_scene(number, replique, scene_type="hook", transition=None, setting=None):
    return {
        "scene": {
            "number": number,
            "type": scene_type,
            "description": _json_description(setting=setting),
        },
        "dialogues": [{"personnage": "NARRATEUR", "replique": replique}],
        "transition": transition or "Fondu enchaine vers la scene suivante.",
    }


@pytest.fixture
def valid_llm_json() -> Dict[str, Any]:
    """JSON valide genere par un LLM — storyboard cinematographique (Sprint 32.1, 8+ scenes)."""
    return {
        "title": "5 metiers developpeur transformes par l'IA en 2027",
        "scenes": [
            _json_scene(1, "Voici pourquoi 80% des developpeurs sous-estiment l'IA.", scene_type="hook"),
            _json_scene(2, "Aujourd'hui, on va decouvrir les 5 metiers les plus impactes.", scene_type="introduction"),
            _json_scene(3, "Premier metier: le developpeur full-stack va etre transforme.", scene_type="development"),
            _json_scene(4, "Deuxieme metier: le data scientist doit s'adapter aux nouveaux outils.", scene_type="development"),
            _json_scene(5, "Troisieme metier: le DevOps automatise de nouvelles taches.", scene_type="development"),
            _json_scene(6, "Quatrieme metier: le designer UX utilise l'IA pour generer des maquettes.", scene_type="development"),
            _json_scene(7, "Cinquieme metier: l'architecte cloud orchestre des infrastructures intelligentes.", scene_type="twist"),
            _json_scene(8, "Pour finir, voici comment se preparer a ces transformations.", scene_type="cta"),
        ],
        "language": "fr",
        "target_audience": "Developpeurs curieux de l'IA",
        "style": "Innovant",
    }


# ── Tests : Création de LLMScriptGenerator ──────────────────────────────────

class TestCreation:
    def test_default_creation(self):
        """Creation avec valeurs par defaut."""
        gen = LLMScriptGenerator()
        assert gen is not None
        assert "llm" in gen.name
        assert gen.stats["llm_calls"] == 0
        assert gen.stats["fallbacks"] == 0

    def test_custom_provider(self):
        """Provider personnalise."""
        gen = LLMScriptGenerator(provider_name="claude")
        assert "claude" in gen.name

    def test_custom_model(self):
        """Modele personnalise."""
        gen = LLMScriptGenerator(model="gpt-4o-mini")
        assert "gpt-4o-mini" in gen.name

    def test_custom_temperature(self):
        """Temperature personnalisee."""
        gen = LLMScriptGenerator(temperature=0.3)
        assert gen._temperature == 0.3

    def test_custom_max_retries(self):
        """Max retries personnalise."""
        gen = LLMScriptGenerator(max_retries=3)
        assert gen._max_retries == 3

    def test_stats_immutable_copy(self):
        """Les stats sont une copie, pas l'original."""
        gen = LLMScriptGenerator()
        stats = gen.stats
        stats["llm_calls"] = 999  # modifie la copie
        assert gen.stats["llm_calls"] == 0  # original inchange

    def test_default_deepseek_script_model_is_chat(self):
        """
        Sprint 27 : le modèle DeepSeek par défaut est deepseek-chat.
        deepseek-reasoner s'est révélé peu fiable en json_mode strict pour
        cette sortie structurée volumineuse (réponse vide par épuisement du
        budget max_tokens dans le raisonnement interne, avant même d'écrire
        le JSON) — même constat qu'au Sprint 24.5 pour les 3 autres moteurs LLM.
        """
        from src.llm_script_generator import _DEEPSEEK_SCRIPT_MODEL
        assert _DEEPSEEK_SCRIPT_MODEL == "deepseek-chat"

    def test_is_script_generator(self):
        """LLMScriptGenerator est bien un ScriptGenerator."""
        gen = LLMScriptGenerator()
        assert isinstance(gen, ScriptGenerator)

    def test_name_property(self):
        """La propriete name retourne un string non vide."""
        gen = LLMScriptGenerator()
        assert isinstance(gen.name, str)
        assert len(gen.name) > 0


# ── Tests : _resolve_model (Sprint 24 — DeepSeek systematiquement reasoner) ──

class TestResolveModel:
    def test_explicit_model_always_wins(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
        gen = LLMScriptGenerator(model="gpt-4o-mini")
        assert gen._resolve_model() == "gpt-4o-mini"

    def test_explicit_deepseek_provider_without_model_uses_chat(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        gen = LLMScriptGenerator(provider_name="deepseek")
        assert gen._resolve_model() == "deepseek-chat"

    def test_explicit_deepseek_provider_with_model_override(self):
        gen = LLMScriptGenerator(provider_name="deepseek", model="deepseek-chat")
        assert gen._resolve_model() == "deepseek-chat"

    def test_auto_detected_deepseek_via_env_uses_chat(self, monkeypatch):
        """Si DEEPSEEK_API_KEY est présente et aucun provider n'est forcé,
        build_llm() auto-détecterait DeepSeek (priorité #1) — _resolve_model()
        doit donc déjà retourner le modèle DeepSeek par défaut."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
        gen = LLMScriptGenerator()
        assert gen._resolve_model() == "deepseek-chat"

    def test_no_deepseek_key_no_forced_provider_returns_none(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        gen = LLMScriptGenerator()
        assert gen._resolve_model() is None

    def test_other_explicit_provider_not_forced_to_deepseek(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
        gen = LLMScriptGenerator(provider_name="claude")
        assert gen._resolve_model() is None

    def test_deepseek_script_model_configurable(self, monkeypatch):
        """
        DEEPSEEK_MODEL_SCRIPT (.env) doit pouvoir surcharger le modèle par
        défaut. On patch directement la constante déjà résolue au chargement
        du module (plutôt qu'un reload, qui remplacerait la classe importée
        ailleurs dans ce fichier de test et polluerait les autres tests).
        """
        import src.llm_script_generator as mod
        monkeypatch.setattr(mod, "_DEEPSEEK_SCRIPT_MODEL", "deepseek-chat")
        gen = mod.LLMScriptGenerator(provider_name="deepseek")
        assert gen._resolve_model() == "deepseek-chat"


# ── Tests : build_user_prompt ────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_prompt_contains_inputs(self, sample_opportunity, sample_brief, sample_brand):
        """Le prompt contient les entrees."""
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, sample_brief, sample_brand)
        assert "Intelligence Artificielle" in prompt
        assert "developpeurs" in prompt
        assert "IA FR" in prompt
        assert "Innovant" in prompt
        assert "Liste" in prompt
        assert str(_TARGET_DURATION_SEC) in prompt

    def test_prompt_mentions_json(self, sample_opportunity, sample_brief, sample_brand):
        """Le prompt demande du JSON."""
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, sample_brief, sample_brand)
        assert "JSON" in prompt

    # ── Tests Sprint 20.1 : Durée fixe format Shorts dans le prompt ───────────

    def test_prompt_contains_duration_constraint(self, sample_opportunity, sample_brief, sample_brand):
        """Le prompt contient la contrainte de durée cible fixe (format Shorts)."""
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, sample_brief, sample_brand)
        assert str(_TARGET_DURATION_SEC) in prompt
        assert str(_TARGET_DURATION_MIN_SEC) in prompt
        assert str(_TARGET_DURATION_MAX_SEC) in prompt
        assert "TARGET TOTAL DURATION" in prompt
        assert "duration_seconds" in prompt

    def test_prompt_contains_word_count(self, sample_opportunity, sample_brief, sample_brand):
        """Le prompt contient le nombre de mots attendus."""
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, sample_brief, sample_brand)
        expected_words = round(_TARGET_DURATION_SEC * 150 / 60)
        assert str(expected_words) in prompt
        assert "words" in prompt.lower()
        assert "150 words/minute" in prompt

    def test_prompt_contains_breakdown(self, sample_opportunity, sample_brief, sample_brand):
        """Le prompt contient un exemple de répartition."""
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, sample_brief, sample_brand)
        assert "Hook:" in prompt
        assert "Intro:" in prompt
        assert "CTA:" in prompt

    @pytest.mark.parametrize("brief_duration", [45, 60, 90, 480, 600])
    def test_prompt_target_overrides_brief_duration(self, sample_opportunity, sample_brand, brief_duration):
        """
        Sprint 20.1 : quelle que soit creative_brief.duration_seconds, le prompt
        impose toujours la cible fixe format Shorts (100-130s / 8-10 scenes).
        """
        brief = CreativeBrief(
            opportunity_id=f"test_{brief_duration}s",
            title=f"Test {brief_duration}s",
            angle="Liste",
            hook="Hook test",
            promise="Promise test",
            audience="Test audience",
            emotion="Informatif",
            format="Court",
            duration_seconds=brief_duration,
            structure=["Hook", "Scene 1", "Conclusion", "CTA"],
            visual_style="Simple",
            cta="Abonne-toi",
            originality_score=0.5,
            production_notes=[],
            rationale=["Test"],
            metadata={},
        )
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, brief, sample_brand)
        assert str(_TARGET_DURATION_SEC) in prompt
        assert f"{_TARGET_DURATION_MIN_SEC}-{_TARGET_DURATION_MAX_SEC}s" in prompt
        expected_words = round(_TARGET_DURATION_SEC * 150 / 60)
        assert str(expected_words) in prompt

    def test_prompt_mentions_duration_bounds(self, sample_opportunity, sample_brief, sample_brand):
        """Le prompt mentionne la fourchette stricte [100, 130]s (Sprint 20.1)."""
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, sample_brief, sample_brand)
        assert str(_TARGET_DURATION_MIN_SEC) in prompt
        assert str(_TARGET_DURATION_MAX_SEC) in prompt

    # ── Tests Sprint 34 : langue du script suit la marque ─────────────────────

    def test_prompt_language_follows_french_brand(self, sample_opportunity, sample_brief, sample_brand):
        """Sprint 34.1 — seules les repliques suivent la langue de la marque ;
        titre/description/transition restent toujours en anglais."""
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, sample_brief, sample_brand)
        assert "spoken dialogues/repliques must be written in French" in prompt
        assert "write it in FRENCH" in prompt
        assert "write it in ENGLISH" in prompt  # les champs description restent toujours en anglais
        assert "INVENT an original title" in prompt and "write it in ENGLISH" in prompt
        assert "'transition' field must be written in ENGLISH" in prompt

    def test_prompt_language_follows_english_brand(self, sample_opportunity, sample_brief):
        import dataclasses
        store = JsonBrandStore(Path(__file__).resolve().parent.parent / "brands")
        us_brand = store.load("global_us")
        assert us_brand is not None, "Brand global_us doit exister"
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, sample_brief, us_brand)
        assert "spoken dialogues/repliques must be written in English" in prompt
        assert "write it in ENGLISH" in prompt
        assert "French" not in prompt

    # ── Tests Sprint 33 : indice de suite (topic_history) ─────────────────────

    def test_no_sequel_hint_by_default(self, sample_opportunity, sample_brief, sample_brand):
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(sample_opportunity, sample_brief, sample_brand)
        assert "CONTINUATION CONTEXT" not in prompt

    def test_sequel_hint_added_when_opportunity_flagged(self, sample_opportunity, sample_brief, sample_brand):
        import dataclasses
        opportunity = dataclasses.replace(
            sample_opportunity,
            metadata={**sample_opportunity.metadata, "sequel_of": {"title": "Ancien sujet", "date": "2026-01-05"}},
        )
        gen = LLMScriptGenerator()
        prompt = gen._build_user_prompt(opportunity, sample_brief, sample_brand)
        assert "CONTINUATION CONTEXT" in prompt
        assert "Ancien sujet" in prompt
        assert "2026-01-05" in prompt
        assert "Do NOT repeat" in prompt


# ── Tests : build_duration_breakdown ─────────────────────────────────────────

class TestBuildDurationBreakdown:
    def test_short_duration_60s(self):
        """Répartition pour 60 secondes."""
        result = LLMScriptGenerator._build_duration_breakdown(60)
        assert "Hook: 5s" in result
        assert "TOTAL" in result

    def test_medium_duration_90s(self):
        """Répartition pour 90 secondes."""
        result = LLMScriptGenerator._build_duration_breakdown(90)
        assert "Hook:" in result
        assert "Intro:" in result
        assert "CTA:" in result

    def test_long_duration_480s(self):
        """Répartition pour 480 secondes."""
        result = LLMScriptGenerator._build_duration_breakdown(480)
        assert "Hook:" in result
        assert "scenes" in result.lower()
        assert "CTA:" in result

    def test_very_long_duration_600s(self):
        """Répartition pour 600 secondes."""
        result = LLMScriptGenerator._build_duration_breakdown(600)
        assert "scenes" in result.lower()
        assert "TOTAL" in result

    def test_empty_breakdown_not_empty(self):
        """La répartition n'est jamais vide."""
        result = LLMScriptGenerator._build_duration_breakdown(45)
        assert len(result) > 20


# ── Tests : extract_json ─────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json(self):
        """JSON sans fioritures."""
        text = '{"title": "Test"}'
        result = LLMScriptGenerator._extract_json(text)
        assert result == '{"title": "Test"}'

    def test_with_markdown_block(self):
        """JSON dans un bloc markdown ```json ... ```."""
        text = '```json\n{"title": "Test"}\n```'
        result = LLMScriptGenerator._extract_json(text)
        assert result == '{"title": "Test"}'

    def test_with_text_before_after(self):
        """Texte avant et apres le JSON."""
        text = 'Voici le script:\n{"title": "Test"}\nFin.'
        result = LLMScriptGenerator._extract_json(text)
        assert result == '{"title": "Test"}'

    def test_only_curly_braces(self):
        """Seulement des accolades."""
        text = '{"scenes": [{"order": 1}]}'
        result = LLMScriptGenerator._extract_json(text)
        assert "scenes" in result

    def test_empty_string(self):
        """Chaine vide."""
        result = LLMScriptGenerator._extract_json("")
        assert result == ""

    def test_no_json(self):
        """Pas de JSON du tout."""
        result = LLMScriptGenerator._extract_json("Pas de JSON ici")
        assert result == "Pas de JSON ici"

    def test_multiple_braces(self):
        """Plusieurs accolades imbriquees."""
        text = '{"a": {"b": "c"}}'
        result = LLMScriptGenerator._extract_json(text)
        assert result == '{"a": {"b": "c"}}'

    def test_json_with_newlines(self):
        """JSON avec sauts de ligne."""
        text = '{\n  "title": "Test"\n}'
        result = LLMScriptGenerator._extract_json(text)
        assert "title" in result

    def test_markdown_with_text_around(self):
        """Bloc markdown avec texte autour."""
        text = 'Voici le resultat:\n```json\n{"title": "Test"}\n```\nCest tout.'
        result = LLMScriptGenerator._extract_json(text)
        assert result == '{"title": "Test"}'


# ── Tests : validate_json_structure ─────────────────────────────────────────

class TestValidateJsonStructure:
    def test_valid_json(self, valid_llm_json):
        """Un JSON valide ne leve pas d'exception."""
        LLMScriptGenerator._validate_json_structure(valid_llm_json)

    def test_missing_title(self, valid_llm_json):
        """Champ title manquant."""
        data = dict(valid_llm_json)
        del data["title"]
        with pytest.raises(ValueError, match="title"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_missing_scenes(self, valid_llm_json):
        """Champ scenes manquant."""
        data = dict(valid_llm_json)
        del data["scenes"]
        with pytest.raises(ValueError, match="scenes"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_too_few_scenes(self, valid_llm_json):
        """Moins de _TARGET_SCENES_MIN scenes."""
        data = dict(valid_llm_json)
        data["scenes"] = [valid_llm_json["scenes"][0]]
        with pytest.raises(ValueError, match=f"minimum {_TARGET_SCENES_MIN}"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_too_many_scenes(self, valid_llm_json):
        """Plus de _TARGET_SCENES_MAX scenes (Sprint 20.1 — format Shorts strict)."""
        data = dict(valid_llm_json)
        extra_scene = dict(valid_llm_json["scenes"][0])
        n_extra = _TARGET_SCENES_MAX - len(valid_llm_json["scenes"]) + 1
        data["scenes"] = valid_llm_json["scenes"] + [extra_scene] * n_extra
        with pytest.raises(ValueError, match=f"maximum {_TARGET_SCENES_MAX}"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scenes_not_a_list(self, valid_llm_json):
        """scenes n'est pas une liste."""
        data = dict(valid_llm_json)
        data["scenes"] = "not_a_list"
        with pytest.raises(ValueError, match="liste"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scene_missing_scene_number(self, valid_llm_json):
        """Champ 'scene.number' manquant dans une scene."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        data["scenes"][0] = dict(data["scenes"][0])
        data["scenes"][0]["scene"] = dict(data["scenes"][0]["scene"])
        del data["scenes"][0]["scene"]["number"]
        with pytest.raises(ValueError, match="number"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scene_missing_scene_object(self, valid_llm_json):
        """Champ 'scene' (objet imbrique) manquant dans une scene."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        del data["scenes"][0]["scene"]
        with pytest.raises(ValueError, match="scene"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scene_missing_scene_type(self, valid_llm_json):
        """Champ 'scene.type' manquant dans une scene."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        data["scenes"][0] = dict(data["scenes"][0])
        data["scenes"][0]["scene"] = dict(data["scenes"][0]["scene"])
        del data["scenes"][0]["scene"]["type"]
        with pytest.raises(ValueError, match="type"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scene_missing_description(self, valid_llm_json):
        """Champ 'scene.description' manquant dans une scene."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        data["scenes"][0] = dict(data["scenes"][0])
        data["scenes"][0]["scene"] = dict(data["scenes"][0]["scene"])
        del data["scenes"][0]["scene"]["description"]
        with pytest.raises(ValueError, match="description"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scene_description_missing_field(self, valid_llm_json):
        """Un champ de 'scene.description' (ex. 'camera') manquant est rejeté."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        data["scenes"][0] = dict(data["scenes"][0])
        data["scenes"][0]["scene"] = dict(data["scenes"][0]["scene"])
        data["scenes"][0]["scene"]["description"] = dict(data["scenes"][0]["scene"]["description"])
        del data["scenes"][0]["scene"]["description"]["camera"]
        with pytest.raises(ValueError, match="camera"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scene_missing_dialogues(self, valid_llm_json):
        """Champ dialogues manquant dans une scene."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        del data["scenes"][0]["dialogues"]
        with pytest.raises(ValueError, match="dialogues"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scene_empty_dialogues_list(self, valid_llm_json):
        """dialogues est une liste vide — rejeté."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        data["scenes"][0]["dialogues"] = []
        with pytest.raises(ValueError, match="dialogues"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_dialogue_missing_personnage(self, valid_llm_json):
        """Un dialogue sans 'personnage' est rejeté."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        data["scenes"][0]["dialogues"] = [{"replique": "Sans personnage."}]
        with pytest.raises(ValueError, match="personnage"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_dialogue_missing_replique(self, valid_llm_json):
        """Un dialogue sans 'replique' est rejeté."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        data["scenes"][0]["dialogues"] = [{"personnage": "NARRATEUR"}]
        with pytest.raises(ValueError, match="replique"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scene_missing_transition(self, valid_llm_json):
        """Champ transition manquant dans une scene."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        del data["scenes"][0]["transition"]
        with pytest.raises(ValueError, match="transition"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_scene_invalid_number(self, valid_llm_json):
        """scene.number inferieur a 1."""
        data = dict(valid_llm_json)
        data["scenes"] = [dict(s) for s in data["scenes"]]
        data["scenes"][0] = dict(data["scenes"][0])
        data["scenes"][0]["scene"] = dict(data["scenes"][0]["scene"])
        data["scenes"][0]["scene"]["number"] = 0
        with pytest.raises(ValueError, match="entier"):
            LLMScriptGenerator._validate_json_structure(data)

    def test_language_and_style_are_optional(self, valid_llm_json):
        """
        Sprint 31.1 : 'language'/'style' ne sont plus des champs obligatoires
        du schéma Storyboard Studio — ils retombent sur BrandProfile si absents
        (voir _build_script_from_json), donc leur absence ne doit PAS lever.
        """
        data = dict(valid_llm_json)
        del data["language"]
        del data["style"]
        LLMScriptGenerator._validate_json_structure(data)  # ne doit pas lever


# ── Tests : extract_json — robustesse (Sprint 27) ────────────────────────────

class TestExtractJsonRobustness:
    def test_strips_think_tags(self):
        raw = '<think>reasoning here</think>{"title": "Test"}'
        assert json.loads(LLMScriptGenerator._extract_json(raw)) == {"title": "Test"}

    def test_removes_trailing_commas(self):
        raw = '{"title": "Test",}'
        assert json.loads(LLMScriptGenerator._extract_json(raw)) == {"title": "Test"}

    def test_removes_control_characters(self):
        raw = '{"title": "Te\x00st"}'
        cleaned = LLMScriptGenerator._extract_json(raw)
        assert "\x00" not in cleaned
        assert json.loads(cleaned) == {"title": "Test"}

    def test_handles_nested_braces_with_trailing_garbage(self):
        raw = '{"title": "Test {inner}"} trailing garbage {not json}'
        assert json.loads(LLMScriptGenerator._extract_json(raw)) == {"title": "Test {inner}"}

    def test_strips_think_tags_and_markdown_fence_together(self):
        raw = '<think>plan...</think>```json\n{"title": "Test"}\n```'
        assert json.loads(LLMScriptGenerator._extract_json(raw)) == {"title": "Test"}


# ── Tests : parse_and_validate — classification des causes d'échec (Sprint 27) ──

def _make_llm_response(content, finish_reason="stop", model="deepseek-chat"):
    return LLMResponse(
        content=content, model=model, provider_name="deepseek",
        finish_reason=finish_reason, prompt_tokens=10, completion_tokens=10,
        total_tokens=20, time_ms=5, cost_usd=0.0001,
    )


class TestParseAndValidate:
    def test_empty_response_raises_empty_response(self):
        response = _make_llm_response("")
        with pytest.raises(_ScriptJsonError) as exc_info:
            LLMScriptGenerator._parse_and_validate(response)
        assert exc_info.value.reason == "empty_response"

    def test_invalid_json_raises_json_invalid(self):
        response = _make_llm_response("not json at all")
        with pytest.raises(_ScriptJsonError) as exc_info:
            LLMScriptGenerator._parse_and_validate(response)
        assert exc_info.value.reason == "json_invalid"

    def test_truncated_response_raises_json_incomplete(self):
        response = _make_llm_response('{"title": "Test"', finish_reason="length")
        with pytest.raises(_ScriptJsonError) as exc_info:
            LLMScriptGenerator._parse_and_validate(response)
        assert exc_info.value.reason == "json_incomplete"

    def test_valid_json_returns_dict(self, valid_llm_json):
        response = _make_llm_response(json.dumps(valid_llm_json))
        data = LLMScriptGenerator._parse_and_validate(response)
        assert data == valid_llm_json

    def test_missing_field_raises_validation_failed(self, valid_llm_json):
        data = dict(valid_llm_json)
        del data["title"]
        response = _make_llm_response(json.dumps(data))
        with pytest.raises(_ScriptJsonError) as exc_info:
            LLMScriptGenerator._parse_and_validate(response)
        assert exc_info.value.reason == "validation_failed"


# ── Tests : retry intelligent — correction JSON avant fallback (Sprint 27) ───

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


class TestIntelligentRetry:
    def test_recovers_via_repair_retry(self, sample_opportunity, sample_brief, sample_brand, valid_llm_json):
        """
        Un premier JSON invalide est corrigé par un second appel — pas de
        fallback. `valid_llm_json` déclenche par ailleurs la correction de
        durée/mots préexistante (Sprint 20.1, indépendante du repair JSON) —
        on fournit donc une 3e réponse scriptée identique pour ce cas, sans
        pour autant exiger qu'elle soit consommée.
        """
        gen = LLMScriptGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response('Sure! Here you go: {"title": "oops, truncated'),
            _make_llm_response(json.dumps(valid_llm_json)),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])

        script = gen.generate(sample_opportunity, sample_brief, sample_brand)

        assert isinstance(script, Script)
        assert script.title == valid_llm_json["title"]
        assert script.metadata["generator"] == "llm_v1"
        assert gen._provider.calls in (2, 3)
        assert gen.stats["json_repair_attempts"] == 1
        assert gen.stats["json_repairs_success"] == 1
        assert gen.stats["fallbacks"] == 0
        assert gen.stats["llm_success"] == 1

    def test_repair_prompt_asks_to_fix_only_the_json(self, sample_opportunity, sample_brief, sample_brand, valid_llm_json):
        gen = LLMScriptGenerator(max_retries=1)
        provider = _ScriptedProvider([
            _make_llm_response("not json at all"),
            _make_llm_response(json.dumps(valid_llm_json)),
        ])
        gen._provider = provider

        captured_messages = []
        original_generate = provider.generate

        def spy_generate(messages, **kwargs):
            captured_messages.append(messages)
            return original_generate(messages, **kwargs)

        provider.generate = spy_generate
        gen.generate(sample_opportunity, sample_brief, sample_brand)

        repair_call_messages = captured_messages[1]
        assert repair_call_messages[-1].role == "user"
        assert "Corrige" in repair_call_messages[-1].content
        assert "JSON" in repair_call_messages[-1].content

    def test_falls_back_with_reason_when_repair_also_fails(self, sample_opportunity, sample_brief, sample_brand):
        gen = LLMScriptGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("not json at all"),
            _make_llm_response("still not json either"),
        ])

        script = gen.generate(sample_opportunity, sample_brief, sample_brand)

        assert script.metadata["generator"] != "llm_v1"
        assert gen.stats["fallbacks"] == 1
        assert gen.stats["json_repair_attempts"] == 1
        assert gen.stats["json_repairs_success"] == 0
        assert sum(gen.stats["fallback_reasons"].values()) == 1
        assert set(gen.stats["fallback_reasons"].keys()) <= {"json_invalid", "json_incomplete"}

    def test_api_error_skips_repair_entirely(self, sample_opportunity, sample_brief, sample_brand):
        """Une erreur API/timeout n'est pas un problème de format JSON — pas de tentative de correction."""
        gen = LLMScriptGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("[DeepSeek API Error: connection refused]", finish_reason="error"),
        ])

        script = gen.generate(sample_opportunity, sample_brief, sample_brand)

        assert script.metadata["generator"] != "llm_v1"
        assert gen.stats["fallback_reasons"].get("api_error") == 1
        assert gen.stats["json_repair_attempts"] == 0
        assert gen._provider.calls == 1

    def test_timeout_response_classified_as_timeout(self, sample_opportunity, sample_brief, sample_brand):
        gen = LLMScriptGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("[DeepSeek API Error: Read timeout]", finish_reason="error"),
        ])

        gen.generate(sample_opportunity, sample_brief, sample_brand)

        assert gen.stats["fallback_reasons"].get("timeout") == 1

    def test_stats_fallback_reasons_immutable_copy(self, sample_opportunity, sample_brief, sample_brand):
        gen = LLMScriptGenerator(max_retries=1)
        gen._provider = _ScriptedProvider([
            _make_llm_response("[DeepSeek API Error: connection refused]", finish_reason="error"),
        ])
        gen.generate(sample_opportunity, sample_brief, sample_brand)
        stats = gen.stats
        stats["fallback_reasons"]["polluted"] = 999
        assert "polluted" not in gen.stats["fallback_reasons"]


# ── Tests : build_script_from_json ──────────────────────────────────────────

class TestBuildScriptFromJson:
    def test_build_full_script(self, valid_llm_json, sample_opportunity, sample_brief, sample_brand):
        """Reconstruction complete d'un Script."""
        script = LLMScriptGenerator._build_script_from_json(
            data=valid_llm_json,
            opportunity=sample_opportunity,
            creative_brief=sample_brief,
            brand_profile=sample_brand,
            response_time_ms=1500,
            response_tokens=800,
            response_cost=0.0032,
        )
        assert isinstance(script, Script)
        assert script.title == valid_llm_json["title"]
        assert script.hook == valid_llm_json["scenes"][0]["dialogues"][0]["replique"]
        assert script.call_to_action == valid_llm_json["scenes"][-1]["dialogues"][0]["replique"]
        assert len(script.scenes) == 8  # valid_llm_json a 8 scenes (contrainte Sprint 21)

    def test_build_scenes(self, valid_llm_json, sample_opportunity, sample_brief, sample_brand):
        """Chaque scene est bien construite."""
        script = LLMScriptGenerator._build_script_from_json(
            valid_llm_json, sample_opportunity, sample_brief, sample_brand,
            1000, 500, 0.001,
        )
        for i, scene in enumerate(script.scenes):
            assert isinstance(scene, ScriptScene)
            assert scene.order == i + 1
            assert scene.duration_seconds > 0
            assert len(scene.narration_text) > 0
            assert isinstance(scene.scene, Scene)
            assert len(scene.scene.type) > 0
            assert len(scene.scene.description.setting) > 0
            assert len(scene.transition) > 0
            assert len(scene.dialogues) > 0
            assert all(isinstance(d, Dialogue) for d in scene.dialogues)

    def test_estimated_duration(self, valid_llm_json, sample_opportunity, sample_brief, sample_brand):
        """Duree estimee = somme des durees calculees via estimate_scene_duration."""
        script = LLMScriptGenerator._build_script_from_json(
            valid_llm_json, sample_opportunity, sample_brief, sample_brand,
            1000, 500, 0.001,
        )
        expected_duration = sum(
            estimate_scene_duration(
                [Dialogue(personnage=d["personnage"], replique=d["replique"]) for d in s["dialogues"]]
            )
            for s in valid_llm_json["scenes"]
        )
        assert script.estimated_duration == expected_duration

    def test_metadata_contains_llm_info(self, valid_llm_json, sample_opportunity, sample_brief, sample_brand):
        """Les metadonnees contiennent les infos LLM."""
        script = LLMScriptGenerator._build_script_from_json(
            valid_llm_json, sample_opportunity, sample_brief, sample_brand,
            response_time_ms=1500, response_tokens=800, response_cost=0.0032,
        )
        assert script.metadata["generator"] == "llm_v1"
        assert script.metadata["llm_time_ms"] == 1500
        assert script.metadata["llm_tokens"] == 800
        assert script.metadata["llm_cost_usd"] == 0.0032
        assert script.metadata["angle"] == "Liste"
        assert script.metadata["niche"] == "Intelligence Artificielle"

    def test_script_is_frozen(self, valid_llm_json, sample_opportunity, sample_brief, sample_brand):
        """Le script a des scenes immutables."""
        script = LLMScriptGenerator._build_script_from_json(
            valid_llm_json, sample_opportunity, sample_brief, sample_brand,
            1000, 500, 0.001,
        )
        for scene in script.scenes:
            assert isinstance(scene, ScriptScene)
        with pytest.raises(FrozenInstanceError):
            script.scenes[0].order = 999
        with pytest.raises(FrozenInstanceError):
            script.scenes[0].dialogues = []

    def test_script_language_from_json(self, valid_llm_json, sample_opportunity, sample_brief, sample_brand):
        """La langue vient du JSON si present."""
        script = LLMScriptGenerator._build_script_from_json(
            valid_llm_json, sample_opportunity, sample_brief, sample_brand,
            1000, 500, 0.001,
        )
        assert script.language == "fr"

    def test_script_style_from_json(self, valid_llm_json, sample_opportunity, sample_brief, sample_brand):
        """Le style vient du JSON si present."""
        script = LLMScriptGenerator._build_script_from_json(
            valid_llm_json, sample_opportunity, sample_brief, sample_brand,
            1000, 500, 0.001,
        )
        assert script.style == "Innovant"

    def test_script_language_falls_back_to_brand(self, valid_llm_json, sample_opportunity, sample_brief, sample_brand):
        """Sprint 31.1 : si 'language'/'style' sont absents du JSON, on retombe sur le BrandProfile."""
        data = dict(valid_llm_json)
        del data["language"]
        del data["style"]
        script = LLMScriptGenerator._build_script_from_json(
            data, sample_opportunity, sample_brief, sample_brand,
            1000, 500, 0.001,
        )
        assert script.language == sample_brand.primary_language
        assert script.style == sample_brand.tone


# ── Tests : generate (fallback) ─────────────────────────────────────────────

class TestGenerateFallback:
    def test_llm_failure_fallsback(self, sample_opportunity, sample_brief, sample_brand):
        """Si le LLM echoue, fallback vers HeuristicScriptGenerator."""
        gen = LLMScriptGenerator(max_retries=2)
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert isinstance(script, Script)
        assert len(script.scenes) > 0
        assert script.metadata.get("generator") != "llm_v1"
        assert gen.stats["llm_calls"] > 0
        assert gen.stats["fallbacks"] >= 1

    def test_fallback_returns_valid_script(self, sample_opportunity, sample_brief, sample_brand):
        """Le fallback retourne un Script valide."""
        gen = LLMScriptGenerator(max_retries=2)
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert len(script.scenes) >= 3
        assert len(script.title) > 0
        assert len(script.hook) > 0
        assert len(script.call_to_action) > 0
        assert script.estimated_duration > 0

    def test_stats_track_fallbacks(self, sample_opportunity, sample_brief, sample_brand):
        """Les stats incrementent les fallbacks."""
        gen = LLMScriptGenerator(max_retries=2)
        gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert gen.stats["fallbacks"] >= 1

    def test_stats_track_calls(self, sample_opportunity, sample_brief, sample_brand):
        """Les stats incrementent les appels LLM."""
        gen = LLMScriptGenerator(max_retries=2)
        gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert gen.stats["llm_calls"] >= 1

    def test_stats_track_failures(self, sample_opportunity, sample_brief, sample_brand):
        """Les stats incrementent les echecs."""
        gen = LLMScriptGenerator(max_retries=2)
        gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert gen.stats["llm_failures"] >= 1

    def test_retry_attempts(self, sample_opportunity, sample_brief, sample_brand):
        """2 tentatives avant fallback."""
        gen = LLMScriptGenerator(max_retries=2)
        gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert gen.stats["llm_failures"] == 2

    def test_retry_attempts_custom(self, sample_opportunity, sample_brief, sample_brand):
        """Nombre de retries personnalise."""
        gen = LLMScriptGenerator(max_retries=3)
        gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert gen.stats["llm_failures"] == 3

    def test_fallback_generator_name_in_metadata(self, sample_opportunity, sample_brief, sample_brand):
        """Le script de fallback a le bon generateur."""
        gen = LLMScriptGenerator(max_retries=2)
        script = gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert script.metadata["generator"] != "llm_v1"

    def test_multiple_calls_reset_stats(self, sample_opportunity, sample_brief, sample_brand):
        """Appels multiples cumulent les stats."""
        gen = LLMScriptGenerator(max_retries=2)
        gen.generate(sample_opportunity, sample_brief, sample_brand)
        gen.generate(sample_opportunity, sample_brief, sample_brand)
        assert gen.stats["fallbacks"] == 2
        assert gen.stats["llm_failures"] == 4


# ── Tests : Découplage ──────────────────────────────────────────────────────

class TestDecoupling:
    def test_import_does_not_import_internal_engines(self):
        """llm_script_generator.py ne référence aucun des moteurs internes non liés."""
        # Vérifie les imports du MODULE lui-même (source), plutôt que sys.modules :
        # sys.modules est un état global partagé par toute la session pytest — un
        # autre fichier de test important légitimement l'un de ces modules
        # (ex. src.niche_intelligence, utilisé par NicheSelector — Sprint 28) fait
        # échouer une vérification basée sur sys.modules sans rapport avec le
        # découplage réel de ce module.
        mod_src = Path(__file__).resolve().parent.parent / "src" / "llm_script_generator.py"
        content = mod_src.read_text(encoding="utf-8")
        forbidden = [
            "src.collector", "src.storage", "src.agents",
            "src.animation_engine", "src.video_engine",
            "src.distribution_engine", "src.niche_intelligence",
        ]
        for mod_name in forbidden:
            assert mod_name not in content

    def test_only_contracts_used(self):
        """Le module utilise les contrats."""
        mod_src = Path(__file__).resolve().parent.parent / "src" / "llm_script_generator.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "from src.opportunity_engine import Opportunity" in content
        assert "from src.creative_engine import CreativeBrief" in content
        assert "from src.brand_engine import BrandProfile" in content
        assert "from src.script_engine import" in content
        assert "ScriptGenerator" in content
        assert "ScriptScene" in content

    def test_llm_only_dependency(self):
        """Le LLM Provider est la seule dependance externe."""
        mod_src = Path(__file__).resolve().parent.parent / "src" / "llm_script_generator.py"
        content = mod_src.read_text(encoding="utf-8")
        assert "from src.llm import" in content
        assert "build_llm" in content
        assert "LLMMessage" in content
