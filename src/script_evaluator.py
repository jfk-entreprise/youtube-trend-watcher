"""
Script Evaluator v1 — Moteur de notation et comparaison de scripts (Sprint 20).

Objectif :
  Valider que les scripts générés par Groq sont meilleurs que les scripts
  heuristiques, en produisant une note sur 8 critères objectifs.

Critères de notation (chacun sur /10) :
  1. Hook          : qualité de l'accroche d'ouverture
  2. Curiosité     : capacité à piquer la curiosité du spectateur
  3. Clarté        : intelligibilité et simplicité du message
  4. Rythme        : alternance des durées de scènes, variété
  5. CTA           : qualité et pertinence de l'appel à l'action
  6. Rétention     : potentiel de rétention (hook fort, progression narrative)
  7. Émotion       : charge émotionnelle (surprise, rire, tension, inspiration)
  8. Originalité   : angle différenciant, créativité du propos

Utilisation :
    evaluator = ScriptEvaluator()
    scores = evaluator.evaluate(script)  # Dict[str, float]
    report = evaluator.compare(heuristic_script, groq_script)
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.script_engine import MAX_SCENE_DURATION_SECONDS, Script

logger = logging.getLogger(__name__)


# ── Interface commune (Sprint 21) ───────────────────────────────────────────

class BaseEvaluator(ABC):
    """
    Interface commune à tout évaluateur de Script.

    Permet de comparer des implémentations hétérogènes (heuristique,
    LLM-as-judge...) de façon polymorphe : chaque évaluateur retourne un
    objet exposant un `.global_score` (float) et une propriété `name`,
    utilisés pour classer/comparer sans connaître le type concret.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(self, script: Script) -> Any:
        """Évalue un script et retourne un objet exposant `.global_score`."""
        ...


# ── Dataclass de résultats ────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScriptScore:
    """Score d'un script sur les 8 critères + score composite."""
    hook_score: float          # /10
    curiosity_score: float     # /10
    clarity_score: float       # /10
    rhythm_score: float        # /10
    cta_score: float           # /10
    retention_score: float     # /10
    emotion_score: float       # /10
    originality_score: float   # /10
    composite_score: float     # /80  (somme des 8)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def global_score(self) -> float:
        """Alias polymorphe (voir BaseEvaluator) — identique à composite_score."""
        return self.composite_score


# ── ScriptEvaluator ───────────────────────────────────────────────────────────

class ScriptEvaluator(BaseEvaluator):
    """
    Évaluateur de scripts — note chaque script sur 8 critères.

    Utilise des heuristiques linguistiques et structurelles (V1).
    Pas d'appel LLM pour ne pas introduire de biais d'évaluation.

    Voir LLMScriptEvaluator (src/llm_script_evaluator.py) pour la variante
    utilisant un LLM comme juge.
    """

    @property
    def name(self) -> str:
        return "heuristic_evaluator_v1"

    def __init__(self) -> None:
        # Mots-clés par critère pour les heuristiques
        self._hook_markers = [
            r"\?",                     # Question rhétorique
            r"vous (\w+ ){0,3}\?",     # "Vous savez quoi ?"
            r"(imagine|suppose|si vous)",  # Conditionnel immersif
            r"(jamais|toujours|personne)", # Absolu provocateur
            r"(incroyable|choquant|révolutionnaire|clé|secret)",
            r"(va vous|allez-vous|avez-vous)",
            r"\d+\s+(choses|raisons|secrets|méthodes|techniques|étapes|façons|signes|astuces|idées|questions|erreurs|leçons|conseils|trucs|points)",
            r"(pourquoi|comment)",     # Question ouverte
        ]
        
        self._curiosity_markers = [
            r"(suite|la suite|plus tard|révèle|découvrir)",
            r"(ce que|ce qui|ce dont)",
            r"(voilà|voici) (pourquoi|comment|ce que)",
            r"(ne (manquez|ratez) pas|à ne pas rater)",
            r"(spoiler|alerte|attention|surprise)",
        ]
        
        self._emotion_markers = [
            r"(incroyable|impressionnant|extraordinaire|fascinant)",
            r"(triste|émouvant|touchant|bouleversant)",
            r"(drôle|hilare|amusant|comique)",
            r"(frustrant|énervant|agaçant|exaspérant)",
            r"(inspirant|encourageant|motivant)",
            r"(choquant|effrayant|terrifiant|inquiétant)",
        ]
        
        self._cta_markers = [
            r"(abonne|like|commente|partage)",
            r"(souscri|subscribe|follow|suis)",
            r"(clique|lien|description|link)",
            r"(prochaine|à venir|next)",
            r"(rejoins|rejoignez)",
        ]
        
        self._originality_markers = [
            r"(différemment|autrement|nouvelle approche)",
            r"(contre-intuitif|paradoxe|inattendu)",
            r"(personne ne|jamais entendu|négligé)",
            r"(angle|perspective|prisme) (original|unique|différent|nouveau)",
            r"(mythe|légende urbaine|idée reçue|préjugé)",
        ]

    # ── Interface publique ─────────────────────────────────────────────────────

    def evaluate(self, script: Script) -> ScriptScore:
        """
        Évalue un script et retourne ses scores.

        Args:
            script: Script à évaluer (HeuristicScript ou LLMScript).

        Returns:
            ScriptScore avec les 8 critères + composite.
        """
        # Extraire tous les textes du script
        all_text = self._collect_text(script)

        hook = script.hook
        cta = script.call_to_action
        conclusion = script.conclusion
        title = script.title

        # ── 1. Hook (/10) ────────────────────────────────────────────────────
        hook_score = self._score_hook(hook, title)

        # ── 2. Curiosité (/10) ───────────────────────────────────────────────
        curiosity_score = self._score_curiosity(all_text, hook)

        # ── 3. Clarté (/10) ──────────────────────────────────────────────────
        clarity_score = self._score_clarity(all_text, script)

        # ── 4. Rythme (/10) ──────────────────────────────────────────────────
        rhythm_score = self._score_rhythm(script)

        # ── 5. CTA (/10) ─────────────────────────────────────────────────────
        cta_score = self._score_cta(cta, conclusion)

        # ── 6. Rétention (/10) ──────────────────────────────────────────────
        retention_score = self._score_retention(script, hook_score, rhythm_score)

        # ── 7. Émotion (/10) ─────────────────────────────────────────────────
        emotion_score = self._score_emotion(all_text)

        # ── 8. Originalité (/10) ─────────────────────────────────────────────
        originality_score = self._score_originality(all_text, script)

        composite = (
            hook_score + curiosity_score + clarity_score + rhythm_score
            + cta_score + retention_score + emotion_score + originality_score
        )

        details = {
            "hook_length": len(hook),
            "total_length": len(all_text),
            "scene_count": len(script.scenes),
            "avg_scene_duration": (
                sum(s.duration_seconds for s in script.scenes) / max(len(script.scenes), 1)
            ),
            "duration_seconds": script.estimated_duration,
            "generator": script.metadata.get("generator", "unknown"),
        }

        return ScriptScore(
            hook_score=round(hook_score, 1),
            curiosity_score=round(curiosity_score, 1),
            clarity_score=round(clarity_score, 1),
            rhythm_score=round(rhythm_score, 1),
            cta_score=round(cta_score, 1),
            retention_score=round(retention_score, 1),
            emotion_score=round(emotion_score, 1),
            originality_score=round(originality_score, 1),
            composite_score=round(composite, 1),
            details=details,
        )

    def compare(
        self,
        scripts: List[Script],
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare plusieurs scripts et produit un rapport structuré.

        Args:
            scripts: Liste de scripts à comparer.
            labels: Noms d'affichage (ex: ["Heuristique", "Groq #1", "Groq #2"]).

        Returns:
            Dict avec scores individuels, classement, et comparatif.
        """
        if labels is None:
            labels = [f"Script #{i+1}" for i in range(len(scripts))]

        if len(labels) != len(scripts):
            labels = [f"Script #{i+1}" for i in range(len(scripts))]

        results = []
        for script, label in zip(scripts, labels):
            score = self.evaluate(script)
            results.append({
                "label": label,
                "generator": script.metadata.get("generator", "unknown"),
                "title": script.title,
                "score": score,
                "hook": script.hook[:80],
            })

        # Classement par score composite
        ranked = sorted(results, key=lambda r: r["score"].composite_score, reverse=True)

        # Comparatif critère par critère
        criteria_names = [
            "hook_score", "curiosity_score", "clarity_score",
            "rhythm_score", "cta_score", "retention_score",
            "emotion_score", "originality_score",
        ]
        criteria_labels = [
            "Hook", "Curiosité", "Clarté", "Rythme",
            "CTA", "Rétention", "Émotion", "Originalité",
        ]

        comparison = {}
        for cname, clabel in zip(criteria_names, criteria_labels):
            values = {r["label"]: getattr(r["score"], cname) for r in results}
            best_label = max(values, key=values.get)
            comparison[clabel] = {
                "scores": values,
                "best": best_label,
                "best_value": values[best_label],
            }

        # Score moyen par générateur
        from collections import defaultdict
        gen_scores: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            gen = r["generator"]
            gen_scores[gen].append(r["score"].composite_score)

        gen_avg = {
            gen: round(sum(scores) / len(scores), 1)
            for gen, scores in gen_scores.items()
        }

        return {
            "ranked": ranked,
            "comparison": comparison,
            "generator_averages": gen_avg,
            "total_scripts": len(scripts),
        }

    # ── Méthodes privées de scoring ───────────────────────────────────────────

    def _collect_text(self, script: Script) -> str:
        """Concatène tout le texte du script pour analyse."""
        parts = [
            script.title or "",
            *(s.narration_text or "" for s in script.scenes),
        ]
        return " ".join(parts)

    def _score_hook(self, hook: str, title: str) -> float:
        """
        Score du hook (/10).

        - Longueur idéale : 20-80 caractères (pénalité si trop court/long)
        - Contient un marqueur de hook (question, nombre, mot fort)
        - Contient un élément de curiosité
        - Pas de répétition avec le titre
        """
        score = 4.0  # Base

        # Longueur
        hlen = len(hook)
        if 20 <= hlen <= 80:
            score += 1.0
        elif hlen > 150:
            score -= 1.0
        elif hlen < 10:
            score -= 2.0

        # Marqueurs de hook
        hook_matches = sum(1 for p in self._hook_markers if re.search(p, hook, re.I))
        score += min(hook_matches * 1.0, 3.0)

        # Différenciation avec le titre
        title_words = set(title.lower().split())
        hook_words = set(hook.lower().split())
        overlap = len(title_words & hook_words) / max(len(title_words), 1)
        if overlap > 0.6:
            score -= 1.0  # Trop similaire au titre
        elif overlap < 0.2:
            score += 0.5  # Hook bien différencié

        return max(0.0, min(10.0, score))

    def _score_curiosity(self, text: str, hook: str) -> float:
        """
        Score de curiosité (/10).

        - Marqueurs lexicaux de curiosité
        - Questions ouvertes dans le texte
        - Éléments de suspens ou de révélation
        """
        score = 3.0

        # Marqueurs lexicaux
        matches = sum(1 for p in self._curiosity_markers if re.search(p, text, re.I))
        score += min(matches * 1.0, 4.0)

        # Questions ouvertes
        questions = len(re.findall(r"\?", text))
        score += min(questions * 0.5, 2.0)

        # Points de suspension / ellipses
        ellipses = len(re.findall(r"\.{3,}", text))
        score += min(ellipses * 0.5, 1.0)

        return max(0.0, min(10.0, score))

    def _score_clarity(self, text: str, script: Script) -> float:
        """
        Score de clarté (/10).

        - Longueur moyenne des phrases (idéal : 8-20 mots)
        - Vocabulaire simple (pas de jargon technique excessif)
        - Structure claire (introduction, développement, conclusion)
        - Scènes nommées clairement
        """
        score = 5.0

        # Phrases : découpage approximatif
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if sentences:
            avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
            if 8 <= avg_words <= 20:
                score += 1.5
            elif avg_words > 30:
                score -= 1.0
            elif avg_words <= 5:
                score -= 0.5
        else:
            score -= 1.0

        # Richesse du storyboard cinématographique (Sprint 32.1) — chaque scène
        # doit avoir un décor/composition/personnages/lumière/caméra/notes de
        # réalisateur substantiels, pas de placeholder générique à un mot.
        thin_scenes = sum(
            1 for s in script.scenes
            if self._scene_description_word_count(s) < 30
        )
        if thin_scenes > len(script.scenes) * 0.3:
            score -= 1.0
        else:
            score += 1.0

        # Structure narrative
        has_intro = bool(script.introduction.strip())
        has_conclusion = bool(script.conclusion.strip())
        if has_intro and has_conclusion:
            score += 1.5
        elif has_intro or has_conclusion:
            score += 0.5

        return max(0.0, min(10.0, score))

    def _score_rhythm(self, script: Script) -> float:
        """
        Score de rythme (/10).

        - Variété des durées de scènes
        - Pas de scène trop longue (>45s) ou trop courte (<3s)
        - Progression des durées (les scènes clés plus longues)
        - Nombre de scènes optimal (5-12)
        """
        score = 4.0

        scenes = script.scenes
        if not scenes:
            return 1.0

        durations = [s.duration_seconds for s in scenes]

        # Variété des durées
        unique_durations = len(set(durations))
        if unique_durations >= len(durations) * 0.5:
            score += 2.0
        elif unique_durations >= 3:
            score += 1.0

        # Pas de durée aberrante — Sprint 37 : plafond Shorts strict (6s/scène)
        outliers = sum(1 for d in durations if d > MAX_SCENE_DURATION_SECONDS)
        score -= min(outliers * 0.5, 2.0)

        # Nombre de scènes optimal
        n = len(scenes)
        if 5 <= n <= 12:
            score += 2.0
        elif n < 3:
            score -= 2.0
        elif n > 15:
            score -= 1.0

        # Progression : la durée max devrait être au milieu (scène clé)
        max_idx = durations.index(max(durations))
        mid = len(durations) / 2
        if abs(max_idx - mid) <= 2:
            score += 1.0

        # Hook court, CTA court
        first_dur = durations[0]
        last_dur = durations[-1]
        if first_dur <= 10:
            score += 0.5
        if last_dur <= 12:
            score += 0.5

        return max(0.0, min(10.0, score))

    def _score_cta(self, cta: str, conclusion: str) -> float:
        """
        Score du CTA (/10).

        - Marqueurs d'action (abonne-toi, like, commentaire)
        - Position naturelle (dans le flow du contenu)
        - Créatif et engageant (pas générique)
        - Urgence ou bénéfice
        """
        score = 4.0

        if not cta:
            return 1.0

        # Longueur du CTA
        if 15 <= len(cta) <= 120:
            score += 1.0
        elif len(cta) > 200:
            score -= 1.0

        # Marqueurs d'action
        matches = sum(1 for p in self._cta_markers if re.search(p, cta, re.I))
        score += min(matches * 1.5, 4.0)

        # Spécificité au contenu (pas que "like et abonne-toi")
        if len(cta.split()) > 8:
            score += 1.0

        # Lien avec la conclusion
        if conclusion and cta:
            conc_words = set(conclusion.lower().split()[-10:])
            cta_words = set(cta.lower().split()[:5])
            if len(conc_words & cta_words) > 0:
                score += 0.5  # Transition naturelle

        return max(0.0, min(10.0, score))

    def _score_retention(self, script: Script, hook_score: float, rhythm_score: float) -> float:
        """
        Score de potentiel de rétention (/10).

        - Qualité du hook (lié au score hook)
        - Structure narrative (progression, climax)
        - Variété des scènes (pas de répétition)
        - Durée totale optimale
        """
        score = 3.0

        # Hook fort = bonne rétention
        score += hook_score * 0.3

        # Rythme varié = bonne rétention
        score += rhythm_score * 0.2

        # Descriptions de scènes variées (pas de répétition)
        descriptions = [self._scene_description_text(s).lower().strip() for s in script.scenes]
        if len(set(descriptions)) >= len(descriptions) * 0.7:
            score += 1.0

        # Durée optimale — Sprint 37.5 : budget Shorts strict (90s max, 10s/scène)
        duration = script.estimated_duration
        if 45 <= duration <= 90:  # cœur de cible Shorts
            score += 2.0
        elif duration > 90:  # dépasse le budget de production
            score -= 1.0
        elif duration < 20:  # trop court pour développer une idée
            score -= 0.5

        # Progression narrative : présence de climax (recherché dans l'ensemble
        # des champs du storyboard — decor, ambiance, notes de realisateur —
        # pas dans un simple titre de scene, Sprint 32.1)
        has_climax = any(
            "révél" in self._scene_description_text(s).lower()
            or "rebondissement" in self._scene_description_text(s).lower()
            or "clé" in self._scene_description_text(s).lower()
            or "reveal" in self._scene_description_text(s).lower()
            or "twist" in self._scene_description_text(s).lower()
            for s in script.scenes
        )
        if has_climax:
            score += 1.0

        return max(0.0, min(10.0, score))

    # ── Storyboard cinématographique (Sprint 32.1) ────────────────────────────

    _DESCRIPTION_FIELDS = (
        "setting", "composition", "characters", "lighting", "camera",
        "mood", "symbolism", "director_notes", "viewer_emotion",
    )

    @classmethod
    def _scene_description_text(cls, scene) -> str:
        """Concatène les 9 champs de SceneDescription — pour les heuristiques
        de richesse/variété/climax qui portaient auparavant sur un simple
        texte de scène (Sprint 32.1)."""
        desc = scene.scene.description
        return " ".join(getattr(desc, field) for field in cls._DESCRIPTION_FIELDS)

    @classmethod
    def _scene_description_word_count(cls, scene) -> int:
        """Nombre de mots cumulés sur les 9 champs de SceneDescription —
        mesure de richesse du storyboard (decor, composition, personnages,
        lumière, mise en scène, notes de réalisateur...)."""
        return len(cls._scene_description_text(scene).split())

    def _score_emotion(self, text: str) -> float:
        """
        Score d'émotion (/10).

        - Marqueurs émotionnels dans le texte
        - Variété des émotions (rires, tension, inspiration)
        - Expressions fortes
        """
        score = 3.0

        # Marqueurs émotionnels
        for pattern in self._emotion_markers:
            matches = len(re.findall(pattern, text, re.I))
            score += min(matches * 0.5, 3.0)

        # Ponctuation expressive
        exclamations = len(re.findall(r"!", text))
        score += min(exclamations * 0.3, 2.0)

        # Variété émotionnelle (mots de différentes catégories)
        return max(0.0, min(10.0, score))

    def _score_originality(self, text: str, script: Script) -> float:
        """
        Score d'originalité (/10).

        - Marqueurs d'originalité lexicale
        - Angle non générique
        - Pas de template évident
        """
        score = 3.0

        # Marqueurs d'originalité
        matches = sum(1 for p in self._originality_markers if re.search(p, text, re.I))
        score += min(matches * 1.5, 4.0)

        # Métadonnées : angle du brief
        angle = script.metadata.get("angle", "")
        generic_angles = {"liste", "introduction", "tutoriel", "review", "avis"}
        if angle.lower() not in generic_angles:
            score += 1.0

        # Le générateur est-il LLM ?
        generator = script.metadata.get("generator", "")
        if "llm" in generator:
            score += 2.0  # Les LLM produisent généralement plus original

        return max(0.0, min(10.0, score))

    # ── Méthode utilitaire pour rapport Markdown ──────────────────────────────

    def generate_markdown_report(self, comparison: Dict[str, Any]) -> str:
        """
        Génère un rapport Markdown formaté à partir des résultats de compare().

        Args:
            comparison: Résultat de compare().

        Returns:
            Texte du rapport Markdown.
        """
        lines: List[str] = []
        lines.append("# Rapport de Benchmark — Scripts Heuristique vs Groq")
        lines.append("")
        lines.append(f"**Total scripts évalués :** {comparison['total_scripts']}")
        lines.append("")

        # ── Classement général ────────────────────────────────────────────────
        lines.append("## Classement général")
        lines.append("")
        lines.append("| Rang | Script | Générateur | Score (/80) | Hook |")
        lines.append("|------|--------|------------|-------------|------|")

        for rank, result in enumerate(comparison["ranked"], 1):
            s = result["score"]
            short_title = result["title"][:45]
            short_hook = result["hook"][:35]
            lines.append(
                f"| {rank} | {short_title} | {result['generator']} "
                f"| **{s.composite_score}** | {short_hook} |"
            )

        lines.append("")

        # ── Scores moyennes par générateur ────────────────────────────────────
        lines.append("## Performance par générateur")
        lines.append("")
        for gen, avg in sorted(
            comparison["generator_averages"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"- **{gen}** : **{avg}/80** en moyenne")
        lines.append("")

        # ── Comparatif détaillé par critère ────────────────────────────────────
        lines.append("## Comparatif par critère")
        lines.append("")
        lines.append("| Critère | " + " | ".join(
            f"{r['label']}" for r in comparison["ranked"]
        ) + " | Meilleur |")
        lines.append("|---------|" + "|".join(
            "---" for _ in comparison["ranked"]
        ) + "|----------|")

        for cname, cdata in comparison["comparison"].items():
            scores_str = " | ".join(
                f"{cdata['scores'][r['label']]}" for r in comparison["ranked"]
            )
            lines.append(f"| {cname} | {scores_str} | {cdata['best']} ({cdata['best_value']}) |")

        lines.append("")

        # ── Détail par script ─────────────────────────────────────────────────
        lines.append("## Détail par script")
        lines.append("")

        for rank, result in enumerate(comparison["ranked"], 1):
            s = result["score"]
            lines.append(f"### #{rank} — {result['label']} ({result['generator']})")
            lines.append("")
            lines.append(f"- **Titre :** {result['title']}")
            lines.append(f"- **Score total :** {s.composite_score}/80")
            lines.append("- **Détail :**")
            lines.append(f"  - Hook : {s.hook_score}/10")
            lines.append(f"  - Curiosité : {s.curiosity_score}/10")
            lines.append(f"  - Clarté : {s.clarity_score}/10")
            lines.append(f"  - Rythme : {s.rhythm_score}/10")
            lines.append(f"  - CTA : {s.cta_score}/10")
            lines.append(f"  - Rétention : {s.retention_score}/10")
            lines.append(f"  - Émotion : {s.emotion_score}/10")
            lines.append(f"  - Originalité : {s.originality_score}/10")
            lines.append("")

        # ── Conclusion ─────────────────────────────────────────────────────────
        lines.append("## Conclusion")
        lines.append("")

        ranked = comparison["ranked"]
        if len(ranked) >= 2:
            best = ranked[0]
            second = ranked[1]
            diff = best["score"].composite_score - second["score"].composite_score
            lines.append(
                f"**{best['generator']}** ({best['label']}) "
                f"devance **{second['generator']}** ({second['label']}) "
                f"de **{diff:.1f} points**."
            )
            lines.append("")

            # Meilleur par critère
            best_criteria = []
            for cname, cdata in comparison["comparison"].items():
                if "groq" in cdata["best"].lower() or "llm" in cdata["best"].lower():
                    best_criteria.append(cname)

            groq_wins = len(best_criteria)
            lines.append(
                f"Groq remporte **{groq_wins}/8** critères : "
                f"{', '.join(best_criteria)}."
            )
            lines.append("")

            if diff > 0 and groq_wins >= 4:
                lines.append("✅ **Validation : les scripts Groq surclassent les scripts heuristiques.**")
            elif diff <= 0 and groq_wins < 4:
                lines.append("⚠️ **Les scripts heuristiques tiennent encore la comparaison.**")
            else:
                lines.append("📊 **Résultats mitigés — voir le détail par critère.**")

        lines.append("")
        lines.append("---")
        lines.append(f"*Rapport généré par ScriptEvaluator v1 — {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}*")

        return "\n".join(lines)
