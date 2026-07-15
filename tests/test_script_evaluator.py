"""
Tests unitaires pour ScriptEvaluator (Sprint 20, storyboard cinematographique
Sprint 32.1).

Couvre :
  - ScriptScore : création, frozen
  - ScriptEvaluator : évaluation d'un script heuristique
  - ScriptEvaluator : évaluation d'un script LLM
  - ScriptEvaluator : comparaison multiple
  - ScriptEvaluator : cas limites (script vide, 1 scène, CTA manquant)
  - ScriptEvaluator : rapport Markdown
  - Découplage : n'importe aucun moteur interdit
"""

from dataclasses import FrozenInstanceError
from typing import Any, Dict, List

import pytest

from src.script_evaluator import ScriptEvaluator, ScriptScore
from src.script_engine import Dialogue, Scene, SceneDescription, Script, ScriptScene


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _narrated(text: str) -> List[Dialogue]:
    """Raccourci : une scène avec un unique narrateur."""
    return [Dialogue(personnage="NARRATEUR", replique=text)]


def _description(setting: str) -> SceneDescription:
    """Description de storyboard riche (9 champs) — au moins 30 mots
    cumulés pour ne pas être pénalisée comme scène 'thin' (Sprint 32.1)."""
    return SceneDescription(
        setting=setting,
        composition="Composition équilibrée, sujet centré, profondeur de champ nette, lignes directrices claires.",
        characters="Narrateur en voix off, présence discrète, ton assuré et posture confiante.",
        lighting="Éclairage doux et cinématographique avec des contrastes maîtrisés et une ambiance chaleureuse.",
        camera="Mouvement de caméra fluide, léger dolly-in, cadrage stable et précis.",
        mood="Ambiance immersive et engageante qui capte l'attention du spectateur.",
        symbolism="Le décor et la lumière renforcent le propos de la scène et son message.",
        director_notes="Garder un rythme soutenu, guider le regard du spectateur vers l'élément clé et maintenir la tension narrative sans lasser.",
        viewer_emotion="Curiosité et attention soutenues jusqu'à la fin de la scène.",
    )


def _scene(order, scene_desc, replique, transition, duration_seconds, scene_type="scene"):
    return ScriptScene(
        scene=Scene(number=order, type=scene_type, description=_description(scene_desc)),
        dialogues=_narrated(replique),
        transition=transition,
        duration_seconds=duration_seconds,
    )


@pytest.fixture
def heuristic_script() -> Script:
    """Script heuristique typique (angle 'Liste', 8 scènes).

    scenes[0] = hook, scenes[1] = introduction, scenes[-2] = conclusion,
    scenes[-1] = call_to_action (propriétés dérivées, Sprint 31.1).
    """
    return Script(
        title="Les 5 secrets de l'IA en 2025",
        scenes=[
            _scene(1, "Plan d'accroche dynamique, composition abstraite qui capte l'attention immédiatement.",
                   "Vous pensez tout savoir sur l'IA ? Détrompez-vous.",
                   "Fondu entrant depuis le noir.", 8, scene_type="hook"),
            _scene(2, "Tête parlante dans un espace de travail propre et lumineux.",
                   "Aujourd'hui, on va parler des tendances IA.",
                   "Fondu enchaîné.", 12, scene_type="introduction"),
            _scene(3, "Infographie qui anime les chiffres clés de la croissance de l'IA.",
                   "Premier point : l'IA générative explose.",
                   "Glissement vers le haut.", 16, scene_type="development"),
            _scene(4, "Comparaison visuelle claire entre modèles propriétaires et open source.",
                   "Deuxième point : les modèles open source.",
                   "Coupe franche.", 14, scene_type="development"),
            _scene(5, "Point culminant avec zoom dramatique sur la visualisation médicale.",
                   "Troisième point : l'IA dans la santé.",
                   "Zoom avant.", 18, scene_type="development"),
            _scene(6, "Carte bonus qui glisse à l'écran, ton plus léger et amusant.",
                   "Un bonus : les outils gratuits.",
                   "Dissolution douce.", 10, scene_type="development"),
            _scene(7, "Résumé visuel des points clés avec retour à la promesse initiale.",
                   "Pour conclure, l'IA est partout.",
                   "Fondu enchaîné.", 12, scene_type="conclusion"),
            _scene(8, "Écran de fin avec bouton d'abonnement animé et liens visibles.",
                   "Abonne-toi pour ne rien rater des prochaines vidéos.",
                   "Fondu sortant au noir.", 10, scene_type="cta"),
        ],
        estimated_duration=100,
        language="fr",
        target_audience="Développeurs et passionnés d'IA",
        style="Expert, dynamique",
        metadata={
            "generator": "heuristic_v1",
            "angle": "Liste",
            "niche": "IA",
            "scene_count": 8,
        },
    )


@pytest.fixture
def groq_script() -> Script:
    """Script Groq typique avec un style plus créatif (7 scènes)."""
    return Script(
        title="L'IA va tout changer — voici pourquoi",
        scenes=[
            _scene(1, "Question choc en plein écran avec éclairage dramatique à fort contraste.",
                   "Et si l'IA était déjà plus intelligente que vous ?",
                   "Zoom avant rapide.", 6, scene_type="hook"),
            _scene(2, "Graphique de croissance exponentielle aux couleurs néon qui capte l'oeil.",
                   "L'IA générative a progressé de 300% en un an.",
                   "Barres qui montent.", 10, scene_type="introduction"),
            _scene(3, "Écran divisé façon révélation comparant modèles open source et propriétaires.",
                   "Ce que personne ne vous dit : les modèles open source sont déjà meilleurs.",
                   "Révélation avec fondu.", 14, scene_type="development"),
            _scene(4, "Enregistrement d'écran avec éditeur de code et démonstration en direct.",
                   "Regardez ce que j'ai fait avec un modèle gratuit.",
                   "Split screen code + résultat.", 20, scene_type="development"),
            _scene(5, "Bureau futuriste avec réalité augmentée illustrant l'impact sur la carrière.",
                   "Les implications sont immenses pour votre carrière.",
                   "Transition futuriste.", 15, scene_type="development"),
            _scene(6, "Gros plan sur le visage du présentateur, regard confiant vers la caméra.",
                   "L'IA ne va pas vous remplacer. Mais quelqu'un qui l'utilise, oui.",
                   "Ralentissement progressif.", 12, scene_type="conclusion"),
            _scene(7, "Écran de fin avec bouton d'abonnement et lien vers le guide gratuit.",
                   "Alors, prêt à maîtriser l'IA ? Abonne-toi et télécharge le guide gratuit en description.",
                   "Liens qui apparaissent.", 10, scene_type="cta"),
        ],
        estimated_duration=87,
        language="fr",
        target_audience="Professionnels du numérique",
        style="Audacieux, direct",
        metadata={
            "generator": "llm_groq_v1",
            "angle": "Histoire",
            "niche": "IA",
            "scene_count": 7,
            "temperature": 0.7,
        },
    )


@pytest.fixture
def empty_script() -> Script:
    """Script minimal (cas limite) — hook et CTA vides (scène unique)."""
    return Script(
        title="Vidéo test",
        scenes=[
            _scene(1, "Plan minimal, aucune action particulière à l'écran.", "",
                   "Coupe franche.", 10),
        ],
        estimated_duration=10,
        language="fr",
        target_audience="Test",
        style="Simple",
        metadata={"generator": "test"},
    )


# ── ScriptScore ──────────────────────────────────────────────────────────────

class TestScriptScore:

    def test_creation(self):
        score = ScriptScore(
            hook_score=7.5,
            curiosity_score=6.0,
            clarity_score=8.0,
            rhythm_score=5.5,
            cta_score=7.0,
            retention_score=6.5,
            emotion_score=4.0,
            originality_score=5.0,
            composite_score=49.5,
        )
        assert score.hook_score == 7.5
        assert score.composite_score == 49.5

    def test_frozen(self):
        score = ScriptScore(
            hook_score=5.0, curiosity_score=5.0, clarity_score=5.0,
            rhythm_score=5.0, cta_score=5.0, retention_score=5.0,
            emotion_score=5.0, originality_score=5.0, composite_score=40.0,
        )
        with pytest.raises(FrozenInstanceError):
            score.hook_score = 10.0

    def test_details_optional(self):
        score = ScriptScore(
            hook_score=0, curiosity_score=0, clarity_score=0,
            rhythm_score=0, cta_score=0, retention_score=0,
            emotion_score=0, originality_score=0, composite_score=0,
            details={"test": "value"},
        )
        assert score.details["test"] == "value"


# ── ScriptEvaluator — Évaluation simple ──────────────────────────────────────

class TestScriptEvaluatorEvaluate:

    def test_evaluate_heuristic(self, heuristic_script):
        eval = ScriptEvaluator()
        score = eval.evaluate(heuristic_script)

        assert isinstance(score, ScriptScore)
        assert 0 <= score.hook_score <= 10
        assert 0 <= score.composite_score <= 80
        assert score.hook_score > 0
        assert score.cta_score > 0

    def test_evaluate_groq(self, groq_script):
        eval = ScriptEvaluator()
        score = eval.evaluate(groq_script)

        assert isinstance(score, ScriptScore)
        assert 0 <= score.hook_score <= 10
        assert score.hook_score > 0
        assert score.curiosity_score > 0

    def test_evaluate_empty(self, empty_script):
        eval = ScriptEvaluator()
        score = eval.evaluate(empty_script)

        # Hook vide → score très faible mais pas forcément 0 (base score)
        assert score.hook_score < 4.0  # Pénalité pour hook vide
        assert score.cta_score == 1.0   # Pas de CTA → score minimal garanti
        assert score.composite_score < 40  # Très faible

    def test_evaluate_reproducible(self, heuristic_script):
        """Même script → mêmes scores (déterministe)."""
        eval = ScriptEvaluator()
        s1 = eval.evaluate(heuristic_script)
        s2 = eval.evaluate(heuristic_script)
        assert s1.composite_score == s2.composite_score
        assert s1.hook_score == s2.hook_score

    def test_all_criteria_present(self, heuristic_script):
        eval = ScriptEvaluator()
        score = eval.evaluate(heuristic_script)

        assert hasattr(score, 'hook_score')
        assert hasattr(score, 'curiosity_score')
        assert hasattr(score, 'clarity_score')
        assert hasattr(score, 'rhythm_score')
        assert hasattr(score, 'cta_score')
        assert hasattr(score, 'retention_score')
        assert hasattr(score, 'emotion_score')
        assert hasattr(score, 'originality_score')
        assert hasattr(score, 'composite_score')

    def test_composite_is_sum(self, heuristic_script):
        eval = ScriptEvaluator()
        score = eval.evaluate(heuristic_script)

        expected = (
            score.hook_score + score.curiosity_score + score.clarity_score
            + score.rhythm_score + score.cta_score + score.retention_score
            + score.emotion_score + score.originality_score
        )
        # composite_score est arrondi séparément de la somme des 8 critères
        # (chacun déjà arrondi à 1 décimale) : un écart résiduel de l'ordre
        # de l'arrondi (<= 0.1) est normal, pas un bug.
        assert abs(score.composite_score - expected) < 0.15

    def test_details_propagated(self, heuristic_script):
        eval = ScriptEvaluator()
        score = eval.evaluate(heuristic_script)

        assert "scene_count" in score.details
        assert score.details["scene_count"] == 8
        assert "generator" in score.details
        assert score.details["generator"] == "heuristic_v1"


# ── ScriptEvaluator — Comparaison ────────────────────────────────────────────

class TestScriptEvaluatorCompare:

    def test_compare_two_scripts(self, heuristic_script, groq_script):
        eval = ScriptEvaluator()
        result = eval.compare(
            [heuristic_script, groq_script],
            labels=["Heuristique", "Groq"],
        )

        assert result["total_scripts"] == 2
        assert len(result["ranked"]) == 2
        assert result["ranked"][0]["score"].composite_score >= result["ranked"][1]["score"].composite_score

    def test_compare_three_scripts(self, heuristic_script, groq_script, empty_script):
        eval = ScriptEvaluator()
        result = eval.compare(
            [heuristic_script, groq_script, empty_script],
        )

        assert result["total_scripts"] == 3
        assert len(result["ranked"]) == 3
        # Le script vide devrait être dernier
        assert result["ranked"][2]["title"] == "Vidéo test"

    def test_comparison_criteria(self, heuristic_script, groq_script):
        eval = ScriptEvaluator()
        result = eval.compare([heuristic_script, groq_script])

        assert "Hook" in result["comparison"]
        assert "Rythme" in result["comparison"]
        assert "CTA" in result["comparison"]

        # Vérifier que chaque critère a un best
        for cname, cdata in result["comparison"].items():
            assert "best" in cdata
            assert "scores" in cdata

    def test_generator_averages(self, heuristic_script, groq_script):
        eval = ScriptEvaluator()
        result = eval.compare([heuristic_script, groq_script])

        assert "heuristic_v1" in result["generator_averages"]
        assert "llm_groq_v1" in result["generator_averages"]

    def test_auto_labels(self, heuristic_script, groq_script):
        eval = ScriptEvaluator()
        result = eval.compare([heuristic_script, groq_script])

        for r in result["ranked"]:
            assert r["label"].startswith("Script #") or "Groq" in r["label"] or "Heuristique" in r["label"]

    def test_labels_mismatch(self, heuristic_script, groq_script):
        """Si labels.length != scripts.length, on génère des labels auto."""
        eval = ScriptEvaluator()
        result = eval.compare(
            [heuristic_script, groq_script],
            labels=["Un seul label"],
        )
        # Doit avoir rattrapé avec des labels auto
        for r in result["ranked"]:
            assert r["label"]  # Non vide


# ── ScriptEvaluator — Rapport Markdown ──────────────────────────────────────

class TestScriptEvaluatorMarkdown:

    def test_generate_markdown_report(self, heuristic_script, groq_script):
        eval = ScriptEvaluator()
        comparison = eval.compare([heuristic_script, groq_script])
        markdown = eval.generate_markdown_report(comparison)

        assert isinstance(markdown, str)
        assert len(markdown) > 100
        assert "# Rapport de Benchmark" in markdown
        assert "Classement général" in markdown
        assert "Comparatif par critère" in markdown
        assert "Conclusion" in markdown

    def test_markdown_contains_scores(self, heuristic_script, groq_script):
        eval = ScriptEvaluator()
        comparison = eval.compare([heuristic_script, groq_script])
        markdown = eval.generate_markdown_report(comparison)

        best = comparison["ranked"][0]
        assert str(best["score"].composite_score) in markdown

    def test_markdown_contains_generator_names(self, heuristic_script, groq_script):
        eval = ScriptEvaluator()
        comparison = eval.compare([heuristic_script, groq_script])
        markdown = eval.generate_markdown_report(comparison)

        assert "heuristic_v1" in markdown
        assert "llm_groq_v1" in markdown


# ── Découplage ───────────────────────────────────────────────────────────────

class TestDecoupling:

    def test_no_video_snapshot(self):
        with pytest.raises(ImportError):
            from src.script_evaluator import VideoSnapshot  # type: ignore

    def test_no_virality_engine(self):
        with pytest.raises(ImportError):
            from src.script_evaluator import ViralityEngine  # type: ignore

    def test_no_opportunity_engine(self):
        with pytest.raises(ImportError):
            from src.script_evaluator import OpportunityEngine  # type: ignore

    def test_no_creative_engine(self):
        with pytest.raises(ImportError):
            from src.script_evaluator import CreativeEngine  # type: ignore

    def test_depends_only_on_script_engine(self):
        """Le module n'importe que Script (et standard lib)."""
        import src.script_evaluator as se
        mod_src = se.__file__
        with open(mod_src, "r", encoding="utf-8") as f:
            content = f.read()
        # Doit importer Script (et uniquement depuis script_engine)
        assert "from src.script_engine import" in content
        assert "Script" in content
        assert "ScriptScore" in content
        # Mais pas les moteurs internes
        assert "from src.opportunity_engine" not in content
        assert "from src.virality_engine" not in content
        assert "from src.learning_engine" not in content
        assert "from src.collector" not in content


# ── Cas limites ──────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_score_range_bounds(self, heuristic_script):
        """Tous les scores doivent être dans [0, 10] (ou [0, 80] pour composite)."""
        eval = ScriptEvaluator()
        score = eval.evaluate(heuristic_script)

        for attr in ["hook_score", "curiosity_score", "clarity_score",
                      "rhythm_score", "cta_score", "retention_score",
                      "emotion_score", "originality_score"]:
            val = getattr(score, attr)
            assert 0 <= val <= 10, f"{attr} = {val} hors range [0, 10]"

        assert 0 <= score.composite_score <= 80

    def test_script_without_hook(self):
        """Script sans hook doit avoir un hook_score très faible.

        scenes[0] joue le rôle du hook et scenes[-1] celui du CTA (propriétés
        dérivées) : il faut donc deux scènes distinctes pour avoir un hook
        vide et un CTA renseigné en même temps.
        """
        script = Script(
            title="Test",
            scenes=[
                _scene(1, "Écran titre sobre, sans accroche particulière visible.", "",
                       "Fondu enchaîné.", 5),
                _scene(2, "Écran de fin avec bouton d'abonnement animé.", "Like et abonne-toi",
                       "Fondu sortant au noir.", 5),
            ],
            estimated_duration=10,
            language="fr",
            target_audience="Test",
            style="Simple",
            metadata={"generator": "test"},
        )
        eval = ScriptEvaluator()
        score = eval.evaluate(script)
        # Hook vide → score de base pénalisé mais pas forcément 0
        assert score.hook_score < 4.0  # Pénalité pour hook vide
        assert score.hook_score >= 0.0

    def test_script_without_cta(self):
        """Script sans CTA doit avoir cta_score = 1.0 (minimum).

        3 scènes : scenes[0] = hook, scenes[1] = conclusion (= scenes[-2]),
        scenes[-1] = CTA vide.
        """
        script = Script(
            title="Test",
            scenes=[
                _scene(1, "Plan d'accroche avec question dramatique à l'écran.",
                       "Super accroche pour attirer l'attention",
                       "Fondu entrant depuis le noir.", 8),
                _scene(2, "Retour au calme, musique de conclusion en fond.",
                       "Merci d'avoir regardé.",
                       "Fondu au noir.", 6),
                _scene(3, "Écran final neutre, sans appel à l'action visible.", "",
                       "Fondu sortant au noir.", 4),
            ],
            estimated_duration=18,
            language="fr",
            target_audience="Test",
            style="Simple",
            metadata={"generator": "test"},
        )
        eval = ScriptEvaluator()
        score = eval.evaluate(script)
        assert score.cta_score == 1.0  # Score minimum garanti

    def test_single_scene_script(self):
        """Script avec une seule scène doit survivre sans erreur.

        Avec une seule scène, hook/introduction/conclusion/call_to_action
        dérivent tous de cette même scène (cas limite structurel).
        """
        script = Script(
            title="Mini",
            scenes=[
                _scene(1, "Plan unique, résumant hook et appel à l'action.",
                       "Hook court. Fin. Bye.",
                       "Fondu enchaîné.", 30),
            ],
            estimated_duration=30,
            language="fr",
            target_audience="Test",
            style="Simple",
            metadata={"generator": "test"},
        )
        eval = ScriptEvaluator()
        score = eval.evaluate(script)
        assert score.composite_score > 0  # Pas d'erreur

    def test_very_long_script(self):
        """Script très long (20+ scènes).

        scenes[0] = hook, scenes[1] = introduction, scenes[-2] = conclusion,
        scenes[-1] = call_to_action ; le reste sont des points génériques.
        """
        scenes = [
            _scene(1, "Plan d'accroche qui interpelle directement le spectateur.",
                   "Vous voulez tout savoir ?",
                   "Fondu entrant depuis le noir.", 15),
            _scene(2, "Écran titre qui annonce le plan des 22 points.",
                   "Voici 22 points importants.",
                   "Fondu enchaîné.", 15),
        ]
        for i in range(2, 20):
            scenes.append(_scene(
                i + 1, "Standard",
                f"Point numéro {i + 1}.",
                "Standard", 15,
            ))
        scenes.append(_scene(21, "Retour au calme, résumé visuel de tous les points abordés.",
                              "Voilà, c'est tout.",
                              "Fondu au noir.", 15))
        scenes.append(_scene(22, "Écran de fin avec bouton d'abonnement animé.",
                              "Abonne-toi pour la suite.",
                              "Fondu sortant au noir.", 15))
        script = Script(
            title="Très long script de test",
            scenes=scenes,
            estimated_duration=330,
            language="fr",
            target_audience="Test",
            style="Standard",
            metadata={"generator": "test"},
        )
        eval = ScriptEvaluator()
        score = eval.evaluate(script)
        assert score.composite_score > 0
        assert score.rhythm_score < 8  # Pénalité pour trop de scènes
