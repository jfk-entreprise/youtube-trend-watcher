"""
Tests unitaires pour ScriptEvaluator (Sprint 20).

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
from src.script_engine import Script, ScriptScene


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def heuristic_script() -> Script:
    """Script heuristique typique (angle 'Liste', 8 scènes)."""
    return Script(
        title="Les 5 secrets de l'IA en 2025",
        hook="Vous pensez tout savoir sur l'IA ? Détrompez-vous.",
        introduction="Aujourd'hui, on va parler des tendances IA.",
        scenes=[
            ScriptScene(order=1, title="Hook", narration="Vous pensez tout savoir sur l'IA ? Détrompez-vous.",
                        visual_description="Plan d'accroche dynamique",
                        image_prompt="Dynamic abstract composition",
                        animation_notes="Fade-in from black",
                        sound_effects="Whoosh",
                        duration_seconds=8),
            ScriptScene(order=2, title="Introduction", narration="Aujourd'hui, on va parler des tendances IA.",
                        visual_description="Tête parlante",
                        image_prompt="Clean workspace",
                        animation_notes="Crossfade",
                        sound_effects="Music",
                        duration_seconds=12),
            ScriptScene(order=3, title="Point #1", narration="Premier point : l'IA générative explose.",
                        visual_description="Infographie",
                        image_prompt="Infographic composition",
                        animation_notes="Number flies in",
                        sound_effects="Chime",
                        duration_seconds=16),
            ScriptScene(order=4, title="Point #2", narration="Deuxième point : les modèles open source.",
                        visual_description="Comparaison",
                        image_prompt="Comparison chart",
                        animation_notes="Slide transition",
                        sound_effects="Whoosh",
                        duration_seconds=14),
            ScriptScene(order=5, title="Point #3", narration="Troisième point : l'IA dans la santé.",
                        visual_description="Point culminant",
                        image_prompt="Medical visualization",
                        animation_notes="Dramatic zoom",
                        sound_effects="Drum roll",
                        duration_seconds=18),
            ScriptScene(order=6, title="Point bonus", narration="Un bonus : les outils gratuits.",
                        visual_description="Bonus card",
                        image_prompt="Bonus content card",
                        animation_notes="Slide in",
                        sound_effects="Bell",
                        duration_seconds=10),
            ScriptScene(order=7, title="Conclusion", narration="Pour conclure, l'IA est partout.",
                        visual_description="Summary",
                        image_prompt="Summary card",
                        animation_notes="Crossfade",
                        sound_effects="Music resolves",
                        duration_seconds=12),
            ScriptScene(order=8, title="CTA", narration="Abonne-toi pour ne rien rater.",
                        visual_description="Bouton abonnement",
                        image_prompt="Subscribe frame",
                        animation_notes="Screen compresses",
                        sound_effects="Subscribe sound",
                        duration_seconds=10),
        ],
        conclusion="L'IA transforme tout ce qu'on connaît.",
        call_to_action="Abonne-toi pour ne rien rater des prochaines vidéos.",
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
    """Script Groq typique avec un style plus créatif."""
    return Script(
        title="L'IA va tout changer — voici pourquoi",
        hook="Et si l'IA était déjà plus intelligente que vous ?",
        introduction="On entend tout et son contraire sur l'IA. Alors, vrai ou fake ?",
        scenes=[
            ScriptScene(order=1, title="Hook", narration="Et si l'IA était déjà plus intelligente que vous ?",
                        visual_description="Question choc en plein écran",
                        image_prompt="Question mark dramatic lighting high contrast",
                        animation_notes="Zoom avant rapide",
                        sound_effects="Impact + silence",
                        duration_seconds=6),
            ScriptScene(order=2, title="Contexte", narration="L'IA générative a progressé de 300% en un an.",
                        visual_description="Graphique de croissance",
                        image_prompt="Exponential growth chart neon colors",
                        animation_notes="Barres qui montent",
                        sound_effects="Tension drone",
                        duration_seconds=10),
            ScriptScene(order=3, title="Révélation", narration="Ce que personne ne vous dit : les modèles open source sont déjà meilleurs.",
                        visual_description="Split screen comparaison",
                        image_prompt="Split screen comparison open source vs closed source",
                        animation_notes="Révélation avec fondu",
                        sound_effects="Révélation chord",
                        duration_seconds=14),
            ScriptScene(order=4, title="Démonstration", narration="Regardez ce que j'ai fait avec un modèle gratuit.",
                        visual_description="Démonstration en direct",
                        image_prompt="Screen recording with code editor",
                        animation_notes="Split screen code + résultat",
                        sound_effects="Clavier rapide",
                        duration_seconds=20),
            ScriptScene(order=5, title="Impact", narration="Les implications sont immenses pour votre carrière.",
                        visual_description="Scène futuriste",
                        image_prompt="Futuristic office AI augmented reality",
                        animation_notes="Transition futuriste",
                        sound_effects="Musique épique",
                        duration_seconds=15),
            ScriptScene(order=6, title="Conclusion", narration="L'IA ne va pas vous remplacer. Mais quelqu'un qui l'utilise, oui.",
                        visual_description="Tête parlante, regard caméra",
                        image_prompt="Close up speaker confident expression",
                        animation_notes="Ralentissement progressif",
                        sound_effects="Piano",
                        duration_seconds=12),
            ScriptScene(order=7, title="CTA", narration="Alors, prêt à maîtriser l'IA ? Abonne-toi et télécharge le guide gratuit en description.",
                        visual_description="Écran de fin avec liens",
                        image_prompt="CTA screen with subscribe button and link",
                        animation_notes="Liens qui apparaissent",
                        sound_effects="Notification + musique fin",
                        duration_seconds=10),
        ],
        conclusion="L'IA est un outil, pas une menace.",
        call_to_action="Abonne-toi et télécharge le guide gratuit en description.",
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
    """Script minimal (cas limite)."""
    return Script(
        title="Vidéo test",
        hook="",
        introduction="",
        scenes=[
            ScriptScene(order=1, title="Scene 1", narration="Bonjour.",
                        visual_description="Plan",
                        image_prompt="Clean",
                        animation_notes="Rien",
                        sound_effects="Rien",
                        duration_seconds=10),
        ],
        conclusion="",
        call_to_action="",
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
        assert score.composite_score < 35  # Très faible

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
        assert abs(score.composite_score - expected) < 0.01

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
        # Doit importer Script
        assert "from src.script_engine import Script" in content
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
        """Script sans hook doit avoir un hook_score très faible."""
        script = Script(
            title="Test",
            hook="",
            introduction="",
            scenes=[
                ScriptScene(order=1, title="Intro", narration="Test.",
                            visual_description="", image_prompt="",
                            animation_notes="", sound_effects="",
                            duration_seconds=10),
            ],
            conclusion="",
            call_to_action="Like et abonne-toi",
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
        """Script sans CTA doit avoir cta_score = 1.0 (minimum)."""
        script = Script(
            title="Test",
            hook="Super accroche pour attirer l'attention",
            introduction="",
            scenes=[
                ScriptScene(order=1, title="Hook", narration="Super accroche.",
                            visual_description="", image_prompt="",
                            animation_notes="", sound_effects="",
                            duration_seconds=8),
            ],
            conclusion="Merci d'avoir regardé.",
            call_to_action="",
            estimated_duration=8,
            language="fr",
            target_audience="Test",
            style="Simple",
            metadata={"generator": "test"},
        )
        eval = ScriptEvaluator()
        score = eval.evaluate(script)
        assert score.cta_score == 1.0  # Score minimum garanti

    def test_single_scene_script(self):
        """Script avec une seule scène doit survivre sans erreur."""
        script = Script(
            title="Mini",
            hook="Hook court.",
            introduction="",
            scenes=[
                ScriptScene(order=1, title="Uniq", narration="Uniq.",
                            visual_description="", image_prompt="",
                            animation_notes="", sound_effects="",
                            duration_seconds=30),
            ],
            conclusion="Fin.",
            call_to_action="Bye.",
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
        """Script très long (20+ scènes)."""
        scenes = []
        for i in range(22):
            scenes.append(ScriptScene(
                order=i+1, title=f"Point #{i+1}",
                narration=f"Point numéro {i+1}.",
                visual_description="Standard",
                image_prompt="Standard",
                animation_notes="Standard",
                sound_effects="Standard",
                duration_seconds=15,
            ))
        script = Script(
            title="Très long script de test",
            hook="Vous voulez tout savoir ?",
            introduction="Voici 22 points importants.",
            scenes=scenes,
            conclusion="Voilà, c'est tout.",
            call_to_action="Abonne-toi pour la suite.",
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
