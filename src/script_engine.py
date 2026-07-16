"""
Script Engine v1 — Générateur de scripts vidéo complets.

Transforme un CreativeBrief + BrandProfile en un Script structuré en scènes,
prêt à être tourné ou envoyé aux futurs Visual / Animation / Video Engine.

Architecture à responsabilité unique :
  - Creative Engine  → prépare le cadre créatif (CreativeBrief)
  - Brand Engine     → fournit l'identité éditoriale (BrandProfile)
  - Script Engine    → produit le Script complet, découpé en scènes

Découplage :
  - Le Script Engine ne lit jamais VideoSnapshot, KnowledgeEngine,
    ViralityEngine, Collector, ni Storage.
  - Il ne dépend que de Opportunity (via brief), CreativeBrief, BrandProfile.
  - Interchangeable : HeuristicScriptGenerator → ClaudeScriptGenerator → etc.

Sprint 32.1 — Cinematic Storyboard Contract (final_script.json v2) :
  `ScriptScene.scene` n'est plus un texte libre — c'est un `Scene` structuré
  {number, type, description}, où `description` (`SceneDescription`) porte
  9 champs distincts (setting, composition, characters, lighting, camera,
  mood, symbolism, director_notes, viewer_emotion), chacun destiné à un
  usage précis en aval (Visual Director, Image/Animation Generator lisent
  directement ces champs, sans reconstruction depuis un texte libre).

  `duration_seconds` n'est plus décidé par le LLM : il est calculé après
  génération par `estimate_scene_duration()`, à partir du nombre de mots
  des répliques et d'une vitesse de narration centralisée
  (`NARRATION_WORDS_PER_MINUTE`) — modifiable sans toucher au code appelant.

  Les propriétés dérivées `Script.hook/.introduction/.conclusion/
  .call_to_action` et `ScriptScene.narration_text` (Sprint 31.1) sont
  conservées à l'identique.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.brand_engine import BrandProfile
from src.creative_engine import CreativeBrief
from src.opportunity_engine import Opportunity

logger = logging.getLogger(__name__)


# ── Configuration centralisée (Sprint 32.1) ─────────────────────────────────
# Vitesse de narration utilisée pour estimer duration_seconds à partir du
# nombre de mots d'une scène — modifiable ici sans toucher au code appelant
# (LLMScriptGenerator, HeuristicScriptGenerator utilisent tous les deux
# estimate_scene_duration() ci-dessous).

NARRATION_WORDS_PER_MINUTE: float = 150.0
_MIN_SCENE_DURATION_SECONDS = 2

# Sprint 37 — budget vidéo 1 minute max ; Sprint 37.3 — 10s/scène, 6 scènes
# max. SOURCE UNIQUE DE VÉRITÉ pour le plafond par scène : llm_script_generator.py
# (prompt/validation du LLM) importe CETTE constante au lieu d'en redéfinir
# une localement — Sprint 37.1 avait laissé deux constantes distinctes
# (MAX_SCENE_DURATION_SECONDS ici à 6, MAX_SCENE_DURATION_SEC à 10 côté LLM),
# et cap_dialogues_to_duration() (qui tronque RÉELLEMENT le texte) utilisait
# encore la valeur par défaut d'ici (6) alors que le LLM visait 10 — chaque
# scène était donc coupée à 6s même quand le LLM en écrivait 10. Une seule
# constante, importée partout, élimine ce risque de divergence.
MAX_SCENE_DURATION_SECONDS = 10


def _cap_narration_to_duration(
    text: str, max_seconds: int, words_per_minute: float = NARRATION_WORDS_PER_MINUTE,
) -> str:
    """Tronque un texte aux premiers mots tenant dans max_seconds de parole."""
    words = text.split()
    max_words = max(1, int(max_seconds * words_per_minute / 60.0))
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def estimate_scene_duration(
    dialogues: List["Dialogue"],
    words_per_minute: float = NARRATION_WORDS_PER_MINUTE,
) -> int:
    """
    Estime la durée d'une scène (en secondes entiers) à partir du nombre de
    mots parlés dans ses dialogues et d'une vitesse de narration.

    Sprint 32.1 : le LLM ne décide plus jamais de duration_seconds — cette
    fonction est la SEULE source de vérité, appelée après génération par
    tous les générateurs de Script (LLM et heuristique), pour que la durée
    reste toujours cohérente avec le texte réellement prononcé.
    """
    text = " ".join(d.replique for d in dialogues if d.replique)
    word_count = len(text.split())
    seconds = (word_count / words_per_minute) * 60.0
    return max(_MIN_SCENE_DURATION_SECONDS, round(seconds))


# ── Dialogue ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Dialogue:
    """
    Une réplique unique dans une scène.

    Une narration classique (voix off) est représentée par un unique
    Dialogue avec `personnage="NARRATEUR"` — le narrateur est traité comme
    un personnage normal, pas comme un cas spécial.
    """
    personnage: str
    replique: str


def cap_dialogues_to_duration(
    dialogues: List[Dialogue],
    max_seconds: int = MAX_SCENE_DURATION_SECONDS,
    words_per_minute: float = NARRATION_WORDS_PER_MINUTE,
) -> List[Dialogue]:
    """
    Tronque une liste de dialogues (dans l'ordre) pour qu'elle tienne
    strictement dans max_seconds de parole — garantie APPLIQUÉE, pas
    seulement demandée au LLM (Sprint 37) : un LLM qui ignore la consigne de
    6s/scène ne doit jamais produire un ScriptScene qui la dépasse quand
    même. Coupe un dialogue individuel si besoin (dernier recours), jamais
    au-delà du budget de mots restant.

    Appelée par TOUS les points de construction d'un ScriptScene (LLM,
    heuristique, traduction FR) — voir MAX_SCENE_DURATION_SECONDS.
    """
    if estimate_scene_duration(dialogues, words_per_minute) <= max_seconds:
        return dialogues

    max_words = max(1, int(max_seconds * words_per_minute / 60.0))
    result: List[Dialogue] = []
    words_used = 0
    for d in dialogues:
        words = d.replique.split()
        remaining = max_words - words_used
        if remaining <= 0:
            break
        if len(words) <= remaining:
            result.append(d)
            words_used += len(words)
        else:
            result.append(Dialogue(personnage=d.personnage, replique=" ".join(words[:remaining])))
            break
    return result


# ── SceneDescription / Scene (Sprint 32.1 — storyboard cinématographique) ───

@dataclass(frozen=True)
class SceneDescription:
    """
    Description cinématographique complète d'une scène — les notes de
    pré-production d'un studio, pas une légende.

    Champs (chacun lu DIRECTEMENT par les moteurs en aval, sans
    reconstruction depuis un texte libre — Sprint 32.1) :
      setting        : lieu, architecture, époque, climat, textures, décor.
      composition    : disposition du plan — premier plan/arrière-plan,
                       lignes de force, perspective, équilibre visuel.
      characters     : apparence, posture, expression, vêtements, regard,
                       émotion, interaction — même s'il n'y a qu'un narrateur.
      lighting       : source, intensité, couleur, contraste, ombres, ambiance.
      camera         : angle, objectif, focale, mouvement, vitesse, hauteur,
                       cadrage — précis (ex: "Very slow 8-second dolly-in
                       with a slight low angle."), jamais un mot-clé seul.
      mood           : ambiance émotionnelle de la scène.
      symbolism      : signification cachée — pourquoi ce décor/lumière/
                       couleur/cadrage.
      director_notes : notes personnelles du réalisateur — pourquoi cette
                       scène existe, ce qu'elle doit provoquer, ce qu'il
                       faut éviter, comment guider le regard, maintenir le
                       rythme, quels détails mettre en avant.
      viewer_emotion : ce que le spectateur doit ressentir précisément,
                       phrase complète (jamais un simple mot comme "suspense").
    """
    setting: str
    composition: str
    characters: str
    lighting: str
    camera: str
    mood: str
    symbolism: str
    director_notes: str
    viewer_emotion: str


@dataclass(frozen=True)
class Scene:
    """
    Identité + description d'une scène dans le storyboard.

    Champs :
      number      : index de la scène (1-based), stable après réécriture.
      type        : rôle narratif de la scène (ex: "hook", "development",
                   "twist", "cta") — pilote les décisions heuristiques par
                   défaut (VisualEngine) sans dépendre d'un texte de titre.
      description : SceneDescription complète (9 champs).
    """
    number: int
    type: str
    description: SceneDescription


# ── ScriptScene ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScriptScene:
    """
    Une scène individuelle dans un Script — format storyboard cinématographique
    (Sprint 32.1).

    Notes :
      - scene            : Scene (number, type, description structurée).
      - dialogues        : répliques de la scène, dans l'ordre. Une narration
                           classique = un seul Dialogue(personnage="NARRATEUR", ...).
      - transition       : transition cinématographique vers la scène suivante.
      - duration_seconds : durée calculée par estimate_scene_duration()
                           (jamais décidée par le LLM — Sprint 32.1).
    """

    scene: Scene
    dialogues: List[Dialogue]
    transition: str
    duration_seconds: int

    @property
    def order(self) -> int:
        """Alias de compatibilité — équivaut à `scene.number` (Sprint 31.1 → 32.1)."""
        return self.scene.number

    @property
    def narration_text(self) -> str:
        """
        Concatène les répliques de la scène, dans l'ordre — le texte parlé
        unique dont ont besoin les moteurs qui ne raisonnent pas personnage
        par personnage (évaluateur, continuité narrative, sous-titres).
        """
        return " ".join(d.replique for d in self.dialogues if d.replique)


# ── Script ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Script:
    """
    Script vidéo complet — contrat officiel entre le Script Engine
    et tous les moteurs en aval (Visual Engine, Animation Engine,
    Video Engine, Voice Engine, Distribution Engine).

    Champs principaux :
      - title              : titre YouTube (hérité du CreativeBrief)
      - scenes             : liste ordonnée de ScriptScene — la première
                             joue le rôle du hook, la dernière celui du CTA
                             (plus de champs top-level dédiés, Sprint 31.1)
      - estimated_duration : somme des durées de scènes (secondes)
      - language           : langue du script (héritée du BrandProfile)
      - target_audience    : public cible (hérité)
      - style              : ton / style rédactionnel (hérité du BrandProfile)
      - metadata           : données extensibles pour le débogage/reporting —
                             jamais écrites dans final_script.json (voir
                             ProductionPackageBuilder), uniquement dans le
                             rapport de production.
    """

    title: str
    scenes: List[ScriptScene]
    estimated_duration: int
    language: str
    target_audience: str
    style: str
    metadata: Dict[str, Any]

    # ── Propriétés dérivées (compatibilité moteurs internes, Sprint 31.1) ────
    # Jamais sérialisées par dataclasses.asdict() — uniquement des vues
    # calculées sur `scenes`, pour que ScriptEvaluator/RewriteEngine/
    # LearningEngine continuent de raisonner en termes de hook/CTA sans
    # dupliquer cette logique partout.

    @property
    def hook(self) -> str:
        """Texte de la première scène — joue le rôle du hook d'ouverture."""
        return self.scenes[0].narration_text if self.scenes else ""

    @property
    def introduction(self) -> str:
        """Texte de la deuxième scène, si elle existe."""
        return self.scenes[1].narration_text if len(self.scenes) > 1 else ""

    @property
    def conclusion(self) -> str:
        """Texte de l'avant-dernière scène, si elle existe."""
        return self.scenes[-2].narration_text if len(self.scenes) > 1 else ""

    @property
    def call_to_action(self) -> str:
        """Texte de la dernière scène — joue le rôle du CTA de fin."""
        return self.scenes[-1].narration_text if self.scenes else ""


# ── ScriptGenerator ──────────────────────────────────────────────────────────

class ScriptGenerator(ABC):
    """
    Interface abstraite pour tous les générateurs de Script.

    Pour intégrer un LLM (Sprint 15) :
      1. Sous-classer ScriptGenerator
      2. Implémenter name et generate()
      3. Injecter dans ScriptEngine(generator=MonGenerateur())

    Le système ne change pas — respect du principe ouvert/fermé.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(
        self,
        opportunity: Opportunity,
        creative_brief: CreativeBrief,
        brand_profile: BrandProfile,
    ) -> Script: ...


# ── Données heuristiques ─────────────────────────────────────────────────────

# Mapping angle → structure canonique de scènes
_STRUCTURES: Dict[str, List[str]] = {
    "Liste": [
        "Hook",
        "Introduction",
        "Point #1",
        "Point #2",
        "Point #3",
        "Point bonus",
        "Conclusion",
        "CTA",
    ],
    "Histoire": [
        "Hook",
        "Contexte",
        "Problème/Défi",
        "Tentative",
        "Rebondissement",
        "Résolution",
        "Leçon",
        "CTA",
    ],
    "Erreurs fréquentes": [
        "Hook",
        "Contexte",
        "Erreur #1",
        "Erreur #2",
        "Erreur #3",
        "La bonne approche",
        "Conclusion",
        "CTA",
    ],
    "Comparaison": [
        "Hook",
        "Contexte",
        "Critère #1",
        "Critère #2",
        "Critère #3",
        "Verdict",
        "Conclusion",
        "CTA",
    ],
    "Challenge": [
        "Hook",
        "Contexte",
        "Tentative #1",
        "Tentative #2",
        "Tentative #3",
        "Résultat",
        "Leçon",
        "CTA",
    ],
}

# Section (français, historique) → type storyboard (anglais, Sprint 32.1) —
# pilote VisualEngine sans dépendre d'un texte de titre.
_SECTION_TYPE: Dict[str, str] = {
    "Hook": "hook", "Introduction": "introduction", "Contexte": "context",
    "Développement": "development", "Problème/Défi": "conflict",
    "Tentative": "attempt", "Tentative #1": "attempt", "Tentative #2": "attempt",
    "Tentative #3": "attempt", "Rebondissement": "twist",
    "Point #1": "point", "Point #2": "point", "Point #3": "point", "Point bonus": "point",
    "Erreur #1": "mistake", "Erreur #2": "mistake", "Erreur #3": "mistake",
    "La bonne approche": "resolution", "Critère #1": "criterion",
    "Critère #2": "criterion", "Critère #3": "criterion", "Verdict": "verdict",
    "Résolution": "resolution", "Résultat": "result", "Leçon": "lesson",
    "Conclusion": "conclusion", "CTA": "cta",
}

# Narrations par défaut pour chaque type de scène (templates paramétrés)
_DEFAULT_NARRATIONS: Dict[str, str] = {
    "Hook": "{hook}",
    "Introduction": "Aujourd'hui, on va parler de {topic}. {promise}",
    "Contexte": "Voici pourquoi c'est important. {audience} — ce sujet vous concerne directement.",
    "Développement": "Entrons dans le détail. {topic} cache beaucoup plus qu'on ne le croit.",
    "Problème/Défi": "Le vrai problème avec {topic}, c'est qu'on ne sait pas par où commencer.",
    "Tentative": "Première approche : on teste une méthode, on observe le résultat.",
    "Tentative #1": "Étape 1 : on attaque par le plus simple.",
    "Tentative #2": "Étape 2 : on monte en complexité.",
    "Tentative #3": "Étape 3 : la méthode avancée.",
    "Rebondissement": "Mais voilà ce qu'on ne nous dit pas...",
    "Point #1": "Premier point clé : {topic} — voici ce qu'il faut retenir.",
    "Point #2": "Deuxième point : on creuse un peu plus.",
    "Point #3": "Troisième point : celui qui change tout.",
    "Point bonus": "Un dernier pour la route — celui-ci va vous surprendre.",
    "Erreur #1": "Erreur n°1 : la plus courante. Vous l'avez probablement déjà faite.",
    "Erreur #2": "Erreur n°2 : celle qui coûte le plus cher en temps.",
    "Erreur #3": "Erreur n°3 : la plus subtile — personne n'en parle.",
    "La bonne approche": "Voici la bonne méthode, étape par étape.",
    "Critère #1": "Premier critère de comparaison : regardons les différences.",
    "Critère #2": "Deuxième critère : celui qui fait vraiment la différence.",
    "Critère #3": "Troisième critère : le moins connu mais le plus important.",
    "Verdict": "Le verdict est clair : voici ce qu'il faut choisir.",
    "Résolution": "Finalement, voici ce qu'on retient de tout ça.",
    "Résultat": "Le résultat est là — et il parle de lui-même.",
    "Leçon": "La leçon à tirer de tout ça : {topic}, c'est accessible à tout le monde.",
    "Conclusion": "Pour conclure, retenez ceci : {promise}",
    "CTA": "{cta}",
}

# Instructions visuelles par type de scène — alimentent SceneDescription.setting
# (Sprint 32.1 ; simple filet de secours déterministe, pas un texte hollywoodien).
_SCENE_VISUALS: Dict[str, str] = {
    "Hook": "Plan d'accroche dynamique — visuel choc ou question à l'écran. Texte impactant superposé.",
    "Introduction": "Tête parlante face caméra OU écran titre avec musique douce en fond.",
    "Contexte": "Images d'ambiance ou données contextuelles. Transition douce.",
    "Développement": "Alternance face caméra / B-roll illustratif. Rythme soutenu.",
    "Problème/Défi": "Visuel du problème — graphique, citation, ou mise en scène. Plan resserré.",
    "Tentative": "Montrer l'action en temps réel ou accéléré. Caméra dynamique.",
    "Tentative #1": "Plan large pour montrer l'action. Texte de transition 'Étape 1'.",
    "Tentative #2": "Plan moyen. Chronomètre ou compteur visible. 'Étape 2'.",
    "Tentative #3": "Gros plan. Tension. 'Étape 3' en overley.",
    "Rebondissement": "Arrêt musical. Plan fixe. Texte 'MAIS' en gros.",
    "Point #1": "Infographie ou liste animée. Numéro à l'écran.",
    "Point #2": "Transition vers nouvelle infographie. Changement de couleur.",
    "Point #3": "Point culminant visuel. Donnée choc en plein écran.",
    "Point bonus": "Plan plus léger, musique qui change. Tone shift.",
    "Erreur #1": "Split-screen 'Mauvaise façon / Bonne façon'. Icône ✗ en rouge.",
    "Erreur #2": "Animation d'erreur. Texte d'impact 'Erreur fatale'.",
    "Erreur #3": "Visuel subtil — zoom progressif sur le détail qui coince.",
    "La bonne approche": "Plan de travail, écran partagé ou démonstration pas à pas. Fond clair.",
    "Critère #1": "Tableau comparatif s'affiche. Barres de score animées.",
    "Critère #2": "Nouvelle ligne dans le tableau. Comparaison côte à côte.",
    "Critère #3": "Dernière ligne — celle qui fait pencher la balance. Animation de révélation.",
    "Verdict": "Plan large. Résultat du comparatif affiché. Effet de confirmation (vert ✓).",
    "Résolution": "Musique de conclusion. Retour au calme. Face caméra ou paysage.",
    "Résultat": "Révélation. Donnée finale affichée. Pause de 2 secondes pour l'impact.",
    "Leçon": "Voix off posée. Images d'archives ou de conclusion. Ralenti optionnel.",
    "Conclusion": "Résumé visuel des points clés. Retour sur la promesse initiale.",
    "CTA": "Fond de chaîne ou miniature finale. Boutons abonnement animés. Liens à l'écran.",
}

# Transitions par type de scène (texte descriptif, Sprint 31.1).
_SCENE_TRANSITIONS: Dict[str, str] = {
    "Hook": "Fondu entrant depuis le noir.",
    "Introduction": "Fondu enchaîné.",
    "Contexte": "Dissolution douce.",
    "Développement": "Coupe franche.",
    "Problème/Défi": "Fondu au noir.",
    "Tentative": "Coupe franche.",
    "Tentative #1": "Volet vers la droite.",
    "Tentative #2": "Volet vers la droite.",
    "Tentative #3": "Zoom avant.",
    "Rebondissement": "Coupe sèche façon glitch.",
    "Point #1": "Glissement vers le haut.",
    "Point #2": "Glissement vers le haut.",
    "Point #3": "Zoom avant.",
    "Point bonus": "Dissolution douce.",
    "Erreur #1": "Volet vers la gauche.",
    "Erreur #2": "Volet vers la gauche.",
    "Erreur #3": "Zoom avant.",
    "La bonne approche": "Dissolution douce.",
    "Critère #1": "Glissement vers le haut.",
    "Critère #2": "Glissement vers le haut.",
    "Critère #3": "Poussée vers la gauche.",
    "Verdict": "Fondu enchaîné.",
    "Résolution": "Fondu au noir.",
    "Résultat": "Zoom avant.",
    "Leçon": "Dissolution douce.",
    "Conclusion": "Fondu enchaîné.",
    "CTA": "Fondu sortant au noir.",
}


# ── HeuristicScriptGenerator ─────────────────────────────────────────────────

class HeuristicScriptGenerator(ScriptGenerator):
    """
    Générateur heuristique de scripts — aucun appel LLM.

    Construit un Script complet à partir d'un CreativeBrief et d'un BrandProfile
    en assemblant des templates paramétrés et des structures narratives
    prédéfinies — sert de filet de secours quand le LLM échoue (Sprint 32.1 :
    produit directement le storyboard structuré Scene/SceneDescription).

    Travail effectué :
      1. Résolution structure : angle → liste de scènes.
      2. Paramétrage : topic, hook, promesse, audience, CTA injectés.
      3. Chaque scène reçoit : un Scene structuré (number/type/description),
         une narration portée par un unique personnage NARRATEUR (`dialogues`),
         une transition (`transition`).
      4. Durée calculée par estimate_scene_duration(), pondérée par le
         facteur de durée du BrandProfile (Sprint 32.1 — même logique que
         le LLM, pas une valeur décidée arbitrairement par générateur).
    """

    @property
    def name(self) -> str:
        return "heuristic_v1"

    def generate(
        self,
        opportunity: Opportunity,
        creative_brief: CreativeBrief,
        brand_profile: BrandProfile,
    ) -> Script:
        # ── Résoudre la structure ──────────────────────────────────────────────
        angle = creative_brief.angle
        structure = _STRUCTURES.get(angle, _STRUCTURES["Liste"])

        topic = opportunity.niche
        hook_text = creative_brief.hook
        promise = creative_brief.promise
        audience = creative_brief.audience
        cta_text = creative_brief.cta

        # ── Durée ──────────────────────────────────────────────────────────────
        brand_factor = self._compute_brand_duration_factor(brand_profile)

        scenes: List[ScriptScene] = []
        for idx, section_name in enumerate(structure, 1):
            narration = self._render_narration(
                section_name, topic, hook_text, promise, audience, cta_text,
            )
            narration = _cap_narration_to_duration(narration, MAX_SCENE_DURATION_SECONDS)
            dialogues = [Dialogue(personnage="NARRATEUR", replique=narration)]
            setting = _SCENE_VISUALS.get(section_name, "Plan standard.")
            transition = _SCENE_TRANSITIONS.get(section_name, "Coupe franche.")
            scene_type = _SECTION_TYPE.get(section_name, "scene")

            description = SceneDescription(
                setting=setting,
                composition="Cadrage centré, sujet principal au premier plan, arrière-plan neutre.",
                characters="Voix off uniquement — aucun personnage visible à l'écran.",
                lighting="Éclairage neutre et stable, sans effet dramatique marqué.",
                camera="Plan fixe, cadrage stable, aucun mouvement de caméra complexe.",
                mood="Ton neutre et informatif.",
                symbolism="Aucune symbolique particulière — plan purement informatif.",
                director_notes=(
                    "Scène de secours générée automatiquement (aucun appel LLM disponible) — "
                    "à enrichir par un réalisateur ou un passage LLM ultérieur si possible."
                ),
                viewer_emotion="Le spectateur doit rester attentif et curieux de la suite.",
            )
            duration = min(
                MAX_SCENE_DURATION_SECONDS,
                max(_MIN_SCENE_DURATION_SECONDS, round(estimate_scene_duration(dialogues) * brand_factor)),
            )

            scene = ScriptScene(
                scene=Scene(number=idx, type=scene_type, description=description),
                dialogues=dialogues,
                transition=transition,
                duration_seconds=duration,
            )
            scenes.append(scene)

        # ── Métadonnées du script ──────────────────────────────────────────────
        estimated_duration = sum(s.duration_seconds for s in scenes)

        script = Script(
            title=creative_brief.title,
            scenes=scenes,
            estimated_duration=estimated_duration,
            language=brand_profile.primary_language,
            target_audience=creative_brief.audience,
            style=brand_profile.tone,
            metadata={
                "generator": self.name,
                "angle": angle,
                "niche": topic,
                "brand_id": brand_profile.id,
                "brand_name": brand_profile.name,
                "opportunity_score": opportunity.overall_score,
                "urgency": opportunity.urgency,
                "structure": structure,
                "brand_duration_factor": round(brand_factor, 3),
                "scene_count": len(scenes),
                "opportunity_id": opportunity.source_video_id,
            },
        )

        logger.info(
            "Script '%s' généré : %d scènes, %d s (générateur: %s)",
            creative_brief.title[:50],
            len(scenes),
            estimated_duration,
            self.name,
        )
        return script

    # ── Méthodes auxiliaires ───────────────────────────────────────────────────

    def _render_narration(
        self,
        section: str,
        topic: str,
        hook: str,
        promise: str,
        audience: str,
        cta: str,
    ) -> str:
        """Remplit le template de narration avec les paramètres réels."""
        if section == "Hook":
            return hook
        if section == "CTA":
            return cta

        template = _DEFAULT_NARRATIONS.get(section, "Section {section} : {topic}.")
        text = template.format(
            section=section,
            topic=topic,
            hook=hook,
            promise=promise,
            audience=audience,
            cta=cta,
        )
        return text

    @staticmethod
    def _compute_brand_duration_factor(brand_profile: BrandProfile) -> float:
        """
        Calcule un facteur de durée basé sur le profil de marque.

        Si la marque préfère des vidéos longues → facteur > 1.0
        Si la marque préfère des vidéos courtes → facteur < 1.0

        Référence : creative_engine calcule la durée cible du brief.
        Ici on applique un ajustement secondaire pour la granularité des scènes.
        """
        pref = brand_profile.preferred_video_duration
        # Durée de référence : 600s (10min)
        # Facteur plafonné à [0.6, 1.5] pour éviter des durées aberrantes
        return max(0.6, min(1.5, pref / 600.0))


# ── ScriptEngine ─────────────────────────────────────────────────────────────

class ScriptEngine:
    """
    Orchestrateur du Script Engine.

    Transforme un pipeline complet Opportunity + CreativeBrief + BrandProfile
    en Script structuré, découpé en scènes.

    Exemple minimal (HeuristicScriptGenerator automatique) :
        engine = ScriptEngine()
        script = engine.generate_single(opportunity, brief, brand)

    Avec un générateur LLM (Sprint 15) :
        engine = ScriptEngine(generator=ClaudeScriptGenerator())
        script = engine.generate_single(opportunity, brief, brand)

    Génération multiple (tous les briefs de toutes les opportunités) :
        scripts = engine.generate_all(opportunities, briefs_map, brand)

    Le moteur ne connaît aucun autre moteur du système.
    Il ne manipule que : Opportunity, CreativeBrief, BrandProfile → Script.
    """

    def __init__(self, generator: Optional[ScriptGenerator] = None) -> None:
        self._generator = generator or HeuristicScriptGenerator()

    @property
    def generator_name(self) -> str:
        return self._generator.name

    # ── Interface publique ─────────────────────────────────────────────────────

    def generate_single(
        self,
        opportunity: Opportunity,
        creative_brief: CreativeBrief,
        brand_profile: BrandProfile,
    ) -> Script:
        """
        Génère un Script pour un triplet (opportunity, brief, brand).

        C'est l'API principale du moteur. Appelée par generate_all() et
        utilisable directement pour un test unitaire.

        Returns:
            Script complet, découpé en scènes.
        """
        return self._generator.generate(opportunity, creative_brief, brand_profile)

    def generate_all(
        self,
        opportunities: List[Opportunity],
        briefs_map: Dict[str, List[CreativeBrief]],
        brand_profile: BrandProfile,
    ) -> Dict[str, List[Script]]:
        """
        Génère des Scripts pour un lot d'Opportunity × CreativeBrief.

        briefs_map est typiquement le retour de CreativeEngine.generate_all() :
            {source_video_id: [CreativeBrief, ...]}

        Returns:
            Mapping {source_video_id: [Script, ...]}.
        """
        opp_map = {opp.source_video_id: opp for opp in opportunities}
        result: Dict[str, List[Script]] = {}
        total_scripts = 0

        for video_id, briefs in briefs_map.items():
            opp = opp_map.get(video_id)
            if opp is None:
                logger.warning("Opportunity '%s' introuvable dans la liste.", video_id)
                continue
            scripts: List[Script] = []
            for brief in briefs:
                try:
                    script = self._generator.generate(opp, brief, brand_profile)
                    scripts.append(script)
                    total_scripts += 1
                except Exception as exc:
                    logger.warning(
                        "Échec script '%s' / brief '%s' : %s",
                        video_id, brief.angle, exc,
                    )
            result[video_id] = scripts

        logger.info(
            "%d opportunité(s) → %d script(s) (générateur: %s)",
            len([v for v in result.values() if v]),
            total_scripts,
            self.generator_name,
        )
        return result
