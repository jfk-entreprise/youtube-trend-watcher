"""
Creative Engine v1 — Directeur Créatif automatisé.

Transforme une Opportunity en plusieurs CreativeBrief exploitables.
Ne rédige PAS de scripts — prépare uniquement le travail du Script Engine.

Composants :
  - CreativeBrief                : contrat officiel avec le Script Engine.
  - CreativeGenerator            : interface abstraite (ABC).
  - HeuristicCreativeGenerator   : implémentation V1 sans LLM.
  - CreativeEngine               : orchestrateur.

Extensibilité Sprint 13 :
  CreativeGenerator
          │
          ├── HeuristicCreativeGenerator   (V1, heuristique)
          ├── ClaudeCreativeGenerator      (Sprint 13)
          ├── GPTCreativeGenerator         (Sprint 13+)
          └── GeminiCreativeGenerator      (Sprint 13+)

  L'orchestrateur CreativeEngine ne connaît jamais l'implémentation réelle :
      engine = CreativeEngine(generator=ClaudeCreativeGenerator())
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.opportunity_engine import Opportunity

logger = logging.getLogger(__name__)


# ── CreativeBrief ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CreativeBrief:
    """
    Concept éditorial complet — contrat officiel avec le Script Engine.

    Un CreativeBrief = un angle de création unique pour une Opportunity.
    Plusieurs briefs sont générés par Opportunity (3 à 5 variantes distinctes).
    Le Script Engine consomme exclusivement des CreativeBrief.
    """

    opportunity_id: str       # source_video_id de l'Opportunity source
    title: str                # titre YouTube proposé pour cet angle
    angle: str                # nom de la variante (ex. "Liste", "Histoire")
    hook: str                 # accroche d'ouverture (template paramétré)
    promise: str              # promesse de valeur pour le spectateur
    audience: str             # public cible (hérité de l'Opportunity)
    emotion: str              # tonalité émotionnelle
    format: str               # format de production (Short, Tutorial, etc.)
    duration_seconds: int     # durée cible estimée en secondes
    structure: List[str]      # arc narratif, étape par étape
    visual_style: str         # recommandations visuelles de tournage/montage
    cta: str                  # call-to-action de fin de vidéo
    originality_score: float  # diversité vis-à-vis des autres briefs [0.0–1.0]
    production_notes: List[str]  # notes pratiques de production
    rationale: List[str]      # justification de cet angle
    metadata: Dict[str, Any]  # données de débogage et d'extension


# ── CreativeGenerator ─────────────────────────────────────────────────────────

class CreativeGenerator(ABC):
    """
    Interface abstraite pour tous les générateurs de CreativeBrief.

    Pour intégrer un LLM (Sprint 13) :
      1. Sous-classer CreativeGenerator
      2. Implémenter name et generate()
      3. Injecter dans CreativeEngine(generator=MonGenerateur())
      Le reste du système ne change pas.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(self, opportunity: Opportunity) -> List[CreativeBrief]:
        """
        Génère 3 à 5 CreativeBrief, chacun proposant un angle distinct.
        Ne produit jamais un script complet.
        """
        ...


# ── Données heuristiques ──────────────────────────────────────────────────────

_ANGLES: Dict[str, Dict[str, Any]] = {
    "Liste": {
        "structure": [
            "Hook accrocheur",
            "Introduction — pourquoi ce sujet compte",
            "Point #1 (le plus impactant en premier)",
            "Point #2",
            "Point #3",
            "Point bonus (surprise ou nuance)",
            "Synthèse rapide",
            "CTA",
        ],
        "duration_factor": 1.1,
        "visual": "Titres numérotés animés, transitions rapides, B-roll illustratif par point",
    },
    "Histoire": {
        "structure": [
            "Hook émotionnel — in medias res",
            "Contexte et présentation du protagoniste",
            "Conflit ou défi central",
            "Tentative et obstacle",
            "Rebondissement / découverte clé",
            "Résolution",
            "Leçon et takeaway actionnable",
            "CTA",
        ],
        "duration_factor": 1.2,
        "visual": "Narration visuelle immersive, images d'ambiance, voix off posée, B-roll narratif",
    },
    "Erreurs fréquentes": {
        "structure": [
            "Hook choc — la pire erreur révélée d'emblée",
            "Pourquoi tout le monde fait ces erreurs",
            "Erreur #1 + impact réel + correction",
            "Erreur #2 + impact réel + correction",
            "Erreur #3 + impact réel + correction",
            "La bonne approche étape par étape",
            "Récapitulatif express",
            "CTA",
        ],
        "duration_factor": 1.0,
        "visual": "Split-screen mauvais/bon, icônes ✗/✓, texte d'impact rouge/vert",
    },
    "Comparaison": {
        "structure": [
            "Hook — le grand débat posé clairement",
            "Présentation des deux options (neutre)",
            "Critère #1 : analyse comparative + verdict partiel",
            "Critère #2 : analyse comparative + verdict partiel",
            "Critère #3 : analyse comparative + verdict partiel",
            "Notre verdict global avec nuances",
            "Cas particuliers et recommandations personnalisées",
            "CTA",
        ],
        "duration_factor": 1.15,
        "visual": "Tableau comparatif animé, barres de score, overlay A vs B côte à côte",
    },
    "Challenge": {
        "structure": [
            "Hook — le défi annoncé avec enjeu clair",
            "Règles du challenge (simple et compréhensible)",
            "Tentative #1 — premiers résultats",
            "Tentative #2 — progression ou échec",
            "Tentative #3 — climax du challenge",
            "Résultat final révélé",
            "Ce qu'on a vraiment appris",
            "CTA",
        ],
        "duration_factor": 0.9,
        "visual": "Caméra action, chronomètre à l'écran, réactions authentiques, montage dynamique",
    },
}

_HOOK_TEMPLATES: Dict[str, List[str]] = {
    "curiosity": [
        "Personne ne parle de {topic}... et c'est une erreur.",
        "Ce que personne ne te dit sur {topic}.",
        "J'ai découvert quelque chose de surprenant sur {topic}.",
        "Tu n'as jamais vu {topic} sous cet angle.",
    ],
    "error": [
        "90 % des gens font cette erreur avec {topic}.",
        "Tu fais probablement ça avec {topic}... et tu as tort.",
        "L'erreur numéro 1 sur {topic} que tout le monde fait.",
        "Arrête de faire ça avec {topic} immédiatement.",
    ],
    "authority": [
        "Voici pourquoi {topic} va tout changer.",
        "La vérité sur {topic} que les experts ne disent pas.",
        "Après des années dans {topic}, voici ce que j'ai appris.",
        "{topic} : ce que les meilleurs font différemment.",
    ],
    "direct": [
        "Tu fais probablement ça avec {topic}...",
        "Si tu t'intéresses à {topic}, regarde ça.",
        "Tout ce que tu dois savoir sur {topic} en {duration}.",
        "{topic} expliqué simplement — sans jargon.",
    ],
    "challenge": [
        "J'ai testé {topic} pendant 30 jours. Voici ce qui s'est passé.",
        "Peut-on vraiment réussir avec {topic} en une semaine ?",
        "Le défi ultime autour de {topic} — j'ai accepté.",
        "J'ai tout misé sur {topic}. Résultat...",
    ],
    "comparison": [
        "{topic} : quelle est vraiment la meilleure option ?",
        "J'ai tout comparé pour toi sur {topic}.",
        "{topic} A vs B — le match honnête.",
        "Quel est le meilleur choix pour {topic} en 2024 ?",
    ],
}

_ANGLE_HOOK_TYPE: Dict[str, str] = {
    "Liste": "authority",
    "Histoire": "direct",
    "Erreurs fréquentes": "error",
    "Comparaison": "comparison",
    "Challenge": "challenge",
}

_TITLE_TEMPLATES: Dict[str, str] = {
    "Liste": "Top 5 : tout ce qu'il faut savoir sur {topic}",
    "Histoire": "{topic} : l'histoire que personne ne raconte",
    "Erreurs fréquentes": "{topic} — les 3 erreurs que tout le monde fait",
    "Comparaison": "{topic} : le comparatif complet (et honnête)",
    "Challenge": "J'ai testé {topic} pendant 30 jours — voici ce qui s'est passé",
}

_PROMISE_TEMPLATES: Dict[str, str] = {
    "Liste": "Tu vas découvrir les points essentiels sur {topic} — directement actionnables.",
    "Histoire": "Une histoire vraie autour de {topic} qui va changer ta façon de voir les choses.",
    "Erreurs fréquentes": "Tu vas éviter les erreurs que font 90 % des gens avec {topic}.",
    "Comparaison": "Tu vas savoir exactement quelle option choisir pour {topic}, sans te tromper.",
    "Challenge": "Tu vas voir si {topic} est vraiment à la hauteur — testé en conditions réelles.",
}

_BASE_DURATION: Dict[str, int] = {
    "Short": 45,
    "Clip Musical": 240,
    "Divertissement": 360,
    "Actualité": 420,
    "Gameplay": 600,
    "Standard": 600,
    "Tutorial": 720,
    "Analyse": 900,
    "Long": 1200,
}

_VISUAL_BY_FORMAT: Dict[str, str] = {
    "Short": "Vertical 9:16, sous-titres en surimpression obligatoires, rythme 1 idée/3s",
    "Tutorial": "Écran de présentation clair, gros plan sur les étapes, callouts annotés",
    "Analyse": "Graphiques et données visuelles, slides épurés, voix off posée et documentée",
    "Actualité": "Images d'actualité, texte d'impact, transitions nettes, couleurs sobres",
    "Gameplay": "Capture 1080p60, face cam optionnelle en overlay, effets sonores synchronisés",
    "Clip Musical": "Couleurs saturées, chorégraphie centrale, éclairage dramatique, cuts rythmés",
    "Divertissement": "Énergie haute, jump cuts rapides, réactions expressives, musique de fond",
    "Long": "Qualité cinématographique, B-roll riche et varié, narration structurée en actes",
    "Standard": "Format YouTube classique, miniature testée, intro max 30s, chapitres si >8min",
}

_CTA_BY_EMOTION: Dict[str, List[str]] = {
    "Inspirant": [
        "Abonne-toi pour plus de contenu qui change ta vision.",
        "Partage cette vidéo avec quelqu'un qui en a besoin.",
        "Dis-moi en commentaire quelle idée t'a le plus marqué.",
    ],
    "Informatif": [
        "Like si tu as appris quelque chose de nouveau.",
        "Abonne-toi pour ne rater aucune mise à jour.",
        "Pose ta question en commentaire — je réponds à tout.",
    ],
    "Divertissant": [
        "Like si tu as aimé — ça m'aide énormément !",
        "Abonne-toi pour la suite des aventures.",
        "Partage avec un ami qui adorerait ça.",
    ],
    "Motivant": [
        "Commence aujourd'hui — dis-moi en commentaire ton objectif.",
        "Partage cette vidéo à quelqu'un qui hésite encore.",
        "Abonne-toi pour ne pas manquer la prochaine étape.",
    ],
    "Choc/Controverse": [
        "Tu es d'accord ou pas ? Dis-le en commentaire.",
        "Partage si tu penses que les gens doivent entendre ça.",
        "Abonne-toi — la prochaine vidéo va te surprendre encore plus.",
    ],
    "Neutre": [
        "Abonne-toi pour plus de contenu sur ce sujet.",
        "Like si la vidéo t'a été utile.",
        "Dis-moi en commentaire ce que tu veux voir ensuite.",
    ],
}

_PRODUCTION_NOTES_BASE: Dict[str, List[str]] = {
    "Short": [
        "Format vertical 1080×1920",
        "Durée cible : 30–60 secondes — couper sans pitié",
        "Sous-titres automatiques obligatoires (accessibilité + algorithme)",
        "Pas de miniature personnalisée (frame automatique)",
    ],
    "Tutorial": [
        "Préparer un script écrit avant le tournage",
        "Screen recording si contenu numérique (OBS ou Loom)",
        "Ajouter des chapitres YouTube pour faciliter la navigation",
        "Ressources en description (liens, PDF, outils)",
    ],
    "Analyse": [
        "Citer les sources à l'écran pour la crédibilité",
        "Préparer les graphiques et visuels à l'avance",
        "Script complet obligatoire — pas d'improvisation",
        "Prévoir 2 sessions de montage (durée longue)",
    ],
    "Gameplay": [
        "Capture 1080p60 minimum (OBS ou logiciel natif)",
        "Face cam en overlay recommandée pour l'engagement",
        "Highlights uniquement — supprimer les temps morts",
        "Effets sonores et musique libres de droits",
    ],
    "Actualité": [
        "Publier dans les 24–48h suivant l'événement",
        "Sourcer chaque information à l'écran",
        "Miniature avec texte d'impact et visuel de l'événement",
        "Durée courte — aller à l'essentiel",
    ],
    "Divertissement": [
        "Énergie haute dès les 5 premières secondes",
        "Musique de fond dynamique (libres de droits)",
        "Jump cuts pour maintenir le rythme",
        "Réactions authentiques — ne pas surjouer",
    ],
    "Long": [
        "Qualité image professionnelle (4K recommandé)",
        "B-roll riche et varié pour illustrer chaque segment",
        "Chapitres obligatoires pour la navigation",
        "Prévoir 3–5 jours de post-production",
    ],
    "Standard": [
        "Miniature A/B testée (préparer 2 versions)",
        "Intro max 30 secondes — valeur dès le début",
        "Chapitres si durée > 8 minutes",
        "Plan de tournage réalisable en 1 journée",
    ],
}

_PREFERRED_ANGLES: Dict[str, List[str]] = {
    "Short": ["Liste", "Erreurs fréquentes", "Challenge"],
    "Tutorial": ["Liste", "Erreurs fréquentes", "Comparaison", "Histoire"],
    "Analyse": ["Comparaison", "Histoire", "Liste", "Erreurs fréquentes"],
    "Actualité": ["Histoire", "Comparaison", "Erreurs fréquentes", "Liste"],
    "Gameplay": ["Challenge", "Comparaison", "Histoire", "Liste"],
    "Divertissement": ["Challenge", "Histoire", "Liste", "Erreurs fréquentes"],
    "Clip Musical": ["Histoire", "Challenge", "Liste"],
    "Long": ["Histoire", "Comparaison", "Erreurs fréquentes", "Liste", "Challenge"],
    "Standard": ["Liste", "Histoire", "Erreurs fréquentes", "Comparaison", "Challenge"],
}

_ANGLE_RATIONALE: Dict[str, str] = {
    "Liste": "Format liste : lisibilité maximale, partage facile, SEO efficace sur les requêtes 'top N'.",
    "Histoire": "Format histoire : engagement émotionnel fort, rétention améliorée, mémorable.",
    "Erreurs fréquentes": "Format erreurs : fort taux de clic (curiosité + FOMO), SEO naturel sur les pain points.",
    "Comparaison": "Format comparaison : décision facilitée pour l'audience, valeur perçue élevée, shareability.",
    "Challenge": "Format challenge : authenticité perçue, FOMO, fort engagement en commentaires.",
}


# ── HeuristicCreativeGenerator ────────────────────────────────────────────────

class HeuristicCreativeGenerator(CreativeGenerator):
    """
    Générateur heuristique — aucun appel LLM.

    Produit 3 à 5 CreativeBrief par Opportunity en combinant :
      - sélection d'angles adaptée au format du contenu
      - hooks paramétrés par templates
      - structures narratives prédéfinies par angle
      - durée calculée depuis content_type × facteur d'angle
      - score d'originalité basé sur la diversité interne des briefs

    Pour remplacer par un LLM : sous-classer CreativeGenerator.
    """

    def __init__(self, min_variants: int = 3, max_variants: int = 5) -> None:
        self._min = max(1, min_variants)
        self._max = max(self._min, max_variants)

    @property
    def name(self) -> str:
        return "heuristic_v1"

    def generate(self, opportunity: Opportunity) -> List[CreativeBrief]:
        angles = self._select_angles(opportunity)
        raw = [self._build_brief(opportunity, angle) for angle in angles]
        return self._attach_originality(raw)

    # ── Sélection des angles ───────────────────────────────────────────────────

    def _select_angles(self, opportunity: Opportunity) -> List[str]:
        content_type = opportunity.metadata.get("content_type", "Standard")
        preferred = _PREFERRED_ANGLES.get(content_type, _PREFERRED_ANGLES["Standard"])
        n = max(self._min, min(self._max, len(preferred)))
        return preferred[:n]

    # ── Construction d'un brief ────────────────────────────────────────────────

    def _build_brief(self, opportunity: Opportunity, angle: str) -> CreativeBrief:
        angle_def = _ANGLES[angle]
        content_type = opportunity.metadata.get("content_type", "Standard")
        emotion = opportunity.metadata.get("emotion", "Neutre")
        audience = opportunity.metadata.get("target_audience", "Grand public")
        language = opportunity.metadata.get("language", "fr")
        topic = opportunity.niche

        hook = self._build_hook(angle, topic, content_type)
        title = _TITLE_TEMPLATES.get(angle, f"{topic} — {angle}").format(topic=topic)
        promise = _PROMISE_TEMPLATES.get(angle, f"Tout sur {topic}.").format(topic=topic)
        duration = self._compute_duration(content_type, angle_def["duration_factor"])
        visual = self._build_visual(angle, content_type, angle_def["visual"])
        cta = _CTA_BY_EMOTION.get(emotion, _CTA_BY_EMOTION["Neutre"])[0]
        notes = self._build_production_notes(content_type, angle, duration)
        rationale = self._build_rationale(opportunity, angle)

        return CreativeBrief(
            opportunity_id=opportunity.source_video_id,
            title=title,
            angle=angle,
            hook=hook,
            promise=promise,
            audience=audience,
            emotion=emotion,
            format=content_type,
            duration_seconds=duration,
            structure=list(angle_def["structure"]),
            visual_style=visual,
            cta=cta,
            originality_score=0.0,   # attaché en post-traitement
            production_notes=notes,
            rationale=rationale,
            metadata={
                "niche": topic,
                "language": language,
                "opportunity_score": opportunity.overall_score,
                "urgency": opportunity.urgency,
                "angle_duration_factor": angle_def["duration_factor"],
            },
        )

    # ── Hook ──────────────────────────────────────────────────────────────────

    def _build_hook(self, angle: str, topic: str, content_type: str) -> str:
        hook_type = _ANGLE_HOOK_TYPE.get(angle, "direct")
        templates = _HOOK_TEMPLATES.get(hook_type, _HOOK_TEMPLATES["direct"])
        idx = list(_ANGLES.keys()).index(angle) % len(templates)
        duration_label = self._duration_label(content_type)
        return templates[idx].format(topic=topic, duration=duration_label)

    @staticmethod
    def _duration_label(content_type: str) -> str:
        d = _BASE_DURATION.get(content_type, 600)
        if d < 120:
            return "60 secondes"
        return f"{d // 60} minutes"

    # ── Durée ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_duration(content_type: str, factor: float) -> int:
        return max(30, int(_BASE_DURATION.get(content_type, 600) * factor))

    # ── Style visuel ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_visual(angle: str, content_type: str, angle_visual: str) -> str:
        base = _VISUAL_BY_FORMAT.get(content_type, "Format YouTube standard")
        if content_type == "Short":
            return f"Vertical 9:16 — {angle_visual}"
        return f"{base} | {angle_visual}"

    # ── Notes de production ───────────────────────────────────────────────────

    @staticmethod
    def _build_production_notes(content_type: str, angle: str, duration: int) -> List[str]:
        notes = list(_PRODUCTION_NOTES_BASE.get(content_type, _PRODUCTION_NOTES_BASE["Standard"]))
        if angle == "Erreurs fréquentes":
            notes.append("Préparer des exemples concrets et visuels pour chaque erreur")
        elif angle == "Comparaison":
            notes.append("Définir les critères de comparaison avant le tournage")
        elif angle == "Challenge":
            notes.append("Filmer en continu — authenticité prioritaire sur la qualité")
        elif angle == "Histoire":
            notes.append("Story-board visuel recommandé avant tournage")
        if duration > 900:
            notes.append(f"Durée longue ({duration // 60} min) — planifier 2 jours de montage")
        return notes

    # ── Justification ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_rationale(opportunity: Opportunity, angle: str) -> List[str]:
        rationale = [_ANGLE_RATIONALE.get(angle, f"Angle '{angle}' sélectionné.")]
        if opportunity.competition_score < 0.25:
            rationale.append("Faible concurrence — cet angle différenciant maximise la visibilité.")
        if opportunity.evergreen_score > 0.65:
            rationale.append(f"Sujet evergreen (score {opportunity.evergreen_score:.2f}) — longue durée de vie.")
        if opportunity.urgency > 0.75:
            rationale.append("Opportunité urgente — privilégier un angle court et rapide à produire.")
        if opportunity.virality_score > 0.70:
            rationale.append(f"Fort potentiel viral (score {opportunity.virality_score:.2f}) — hook percutant recommandé.")
        return rationale

    # ── Score d'originalité ────────────────────────────────────────────────────

    def _attach_originality(self, briefs: List[CreativeBrief]) -> List[CreativeBrief]:
        """
        Mesure la diversité de chaque brief par rapport aux autres.

        Overlap dimensions (pénalités cumulées) :
          - Même émotion           : -0.10
          - Hook identique (15 car): -0.20
          - Durée proche (< 60s)   : -0.10
          - Style visuel similaire : -0.15  (30 premiers car.)

        Score final = 1.0 − overlap moyen avec les autres briefs.
        """
        n = len(briefs)
        if n <= 1:
            return [self._set_originality(b, 1.0) for b in briefs]

        scores: List[float] = []
        for i, b in enumerate(briefs):
            total_overlap = 0.0
            for j, other in enumerate(briefs):
                if i == j:
                    continue
                overlap = 0.0
                if b.emotion == other.emotion:
                    overlap += 0.10
                if b.hook[:15] == other.hook[:15]:
                    overlap += 0.20
                if abs(b.duration_seconds - other.duration_seconds) < 60:
                    overlap += 0.10
                if b.visual_style[:30] == other.visual_style[:30]:
                    overlap += 0.15
                total_overlap += overlap
            avg = total_overlap / (n - 1)
            scores.append(round(max(0.0, min(1.0, 1.0 - avg)), 3))

        return [self._set_originality(b, s) for b, s in zip(briefs, scores)]

    @staticmethod
    def _set_originality(brief: CreativeBrief, score: float) -> CreativeBrief:
        return CreativeBrief(
            opportunity_id=brief.opportunity_id,
            title=brief.title,
            angle=brief.angle,
            hook=brief.hook,
            promise=brief.promise,
            audience=brief.audience,
            emotion=brief.emotion,
            format=brief.format,
            duration_seconds=brief.duration_seconds,
            structure=brief.structure,
            visual_style=brief.visual_style,
            cta=brief.cta,
            originality_score=score,
            production_notes=brief.production_notes,
            rationale=brief.rationale,
            metadata=brief.metadata,
        )


# ── CreativeEngine ─────────────────────────────────────────────────────────────

class CreativeEngine:
    """
    Orchestrateur du Creative Engine.

    Transforme une liste d'Opportunity en un mapping
    {source_video_id: list[CreativeBrief]}.

    Exemple minimal :
        engine = CreativeEngine()
        briefs_map = engine.generate_all(opportunities)

    Avec générateur LLM (Sprint 13) :
        engine = CreativeEngine(generator=ClaudeCreativeGenerator())
        briefs_map = engine.generate_all(opportunities)
    """

    def __init__(self, generator: Optional[CreativeGenerator] = None) -> None:
        self._generator = generator or HeuristicCreativeGenerator()

    @property
    def generator_name(self) -> str:
        return self._generator.name

    def generate_all(
        self, opportunities: List[Opportunity]
    ) -> Dict[str, List[CreativeBrief]]:
        """Génère les CreativeBrief pour chaque Opportunity."""
        result: Dict[str, List[CreativeBrief]] = {}
        total_briefs = 0
        for opp in opportunities:
            try:
                briefs = self._generator.generate(opp)
                result[opp.source_video_id] = briefs
                total_briefs += len(briefs)
                logger.info(
                    "%d brief(s) → '%s'",
                    len(briefs),
                    opp.title[:55],
                )
            except Exception as exc:
                logger.warning("Échec creative '%s' : %s", opp.source_video_id, exc)
                result[opp.source_video_id] = []
        logger.info(
            "%d opportunités → %d CreativeBrief (générateur: %s)",
            len(opportunities),
            total_briefs,
            self.generator_name,
        )
        return result
