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

Contrat pour les prochains moteurs :
  - Visual Engine    → lit Script.image_prompt, Script.visual_description
  - Animation Engine → lit Script.animation_notes
  - Video Engine     → lit Script.scenes + Script.metadata
  - Voice Engine     → lit Script.narration + BrandProfile.voice_*
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.brand_engine import BrandProfile
from src.creative_engine import CreativeBrief
from src.opportunity_engine import Opportunity

logger = logging.getLogger(__name__)


# ── ScriptScene ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScriptScene:
    """
    Une scène individuelle dans un Script.

    Chaque scène est l'unité atomique de production :
      - Les futurs moteurs (Visual, Animation, Video) liront ces champs
        pour générer des instructions visuelles et techniques.
      - Le Voice Engine lira narration + ton / rythme.

    Notes :
      - order           : index de la scène (1-based)
      - narration       : texte parlé ou voix off pour cette scène
      - visual_description : description de ce qu'on voit à l'écran
      - image_prompt    : prompt générateur d'image (DALL-E, Midjourney, etc.)
      - animation_notes : instructions pour l'animateur / motion designer
      - sound_effects   : effets sonores ou ambiance suggérés
      - duration_seconds: durée estimée de la scène (pose / rythme)
    """

    order: int
    title: str
    narration: str
    visual_description: str
    image_prompt: str
    animation_notes: str
    sound_effects: str
    duration_seconds: int


# ── Script ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Script:
    """
    Script vidéo complet — contrat officiel entre le Script Engine
    et tous les moteurs en aval (Visual Engine, Animation Engine,
    Video Engine, Voice Engine, Distribution Engine).

    Structure canonique :
        Hook → Contexte → Développement → Exemple → Conclusion → CTA

    Champs principaux :
      - title              : titre YouTube (hérité du CreativeBrief)
      - hook               : accroche d'ouverture (héritée et développée)
      - introduction       : transition hook → corps du script
      - scenes             : liste ordonnée de ScriptScene
      - conclusion         : dernier message avant le CTA
      - call_to_action     : CTA de fin de vidéo (hérité et développé)
      - estimated_duration : somme des durées de scènes (secondes)
      - language           : langue du script (héritée du BrandProfile)
      - target_audience    : public cible (hérité)
      - style              : ton / style rédactionnel (hérité du BrandProfile)
      - metadata           : données extensibles pour le débogage

    Pour les prochains moteurs :
      Visual Engine     → script.scenes[].visual_description + image_prompt
      Animation Engine  → script.scenes[].animation_notes
      Video Engine      → script.scenes (montage), script.estimated_duration
      Voice Engine      → script.scenes[].narration + script.style + brand.voice_*
      Distribution      → script.title, script.language, script.metadata
    """

    title: str
    hook: str
    introduction: str
    scenes: List[ScriptScene]
    conclusion: str
    call_to_action: str
    estimated_duration: int
    language: str
    target_audience: str
    style: str
    metadata: Dict[str, Any]


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

# Instructions visuelles par type de scène
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

# Prompts image génériques par type de scène
_SCENE_IMAGE_PROMPTS: Dict[str, str] = {
    "Hook": "Dynamic abstract composition with bold typography, high contrast lighting, cinematic depth of field",
    "Introduction": "Clean professional workspace with warm ambient lighting, shallow depth of field",
    "Contexte": "Contextual environment illustration, atmospheric lighting, muted color palette",
    "Développement": "Detailed infographic style composition, organized information hierarchy, data visualization aesthetic",
    "Problème/Défi": "Dramatic lighting revealing a problem or obstacle, high contrast, moody atmosphere",
    "Tentative": "Action shot showing process in motion, dynamic lighting, motion blur effect",
    "Tentative #1": "First step visual, clean composition, directional lighting from left",
    "Tentative #2": "Mid-process action shot, increased complexity visible, warm lighting",
    "Tentative #3": "Climax of process, dramatic lighting, intense focus on subject",
    "Rebondissement": "Plot twist visual, unexpected angle, surprise element in frame",
    "Point #1": "Numbered infographic #1, clean design, accent color highlighting",
    "Point #2": "Information card #2, continued visual theme, secondary color accent",
    "Point #3": "Key insight reveal #3, strongest visual hierarchy, primary color impact",
    "Point bonus": "Bonus content card, playful lighter design, star or sparkle accent",
    "Erreur #1": "Warning sign style visual, red accent, clear 'before/after' implication",
    "Erreur #2": "Cautionary visual, amber warning tones, 'avoid this' composition",
    "Erreur #3": "Subtle trap visualization, zoom-in reveal style, hidden detail emphasized",
    "La bonne approche": "Step-by-step guide visual, clean instructional design, green/blue accents",
    "Critère #1": "Comparison chart starting point, clean typography, measurement scale visible",
    "Critère #2": "Comparison chart mid-point, balanced visual weight, animated indicator",
    "Critère #3": "Final comparison element, decisive factor highlighted, conclusion building",
    "Verdict": "Clear winner display, checkmark visual, confident resolution aesthetic",
    "Résolution": "Peaceful resolution scene, warm golden hour lighting, calm atmosphere",
    "Résultat": "Final result display, data visualization, success indicators, clean design",
    "Leçon": "Wisdom moment visual, soft lighting, reflective mood, open space",
    "Conclusion": "Summary card with key takeaways, clean organized layout, brand colors",
    "CTA": "Subscription call-to-action frame, brand colors, button visual, inviting composition",
}

# Notes d'animation par type de scène
_SCENE_ANIMATIONS: Dict[str, str] = {
    "Hook": "Fade-in from black. Bold text animation (scale up + stabilize). 0.5s buildup.",
    "Introduction": "Crossfade transition. Gentle parallax on background. Soft text reveal.",
    "Contexte": "Slow zoom on establishing shot. Text overlay fades in line by line.",
    "Développement": "Cut to medium pace. B-roll has gentle pan. Text callouts pop in.",
    "Problème/Défi": "Color grade shifts cooler. Slow push-in on subject. Tension build.",
    "Tentative": "Speed ramping — 2× real time then slow for result. Dynamic cuts.",
    "Tentative #1": "Simple wipe transition. Step counter animates from 1.",
    "Tentative #2": "Quick zoom transition. Step counter updates. Confidence builds.",
    "Tentative #3": "Dramatic zoom. Step counter pulses. Suspense buildup.",
    "Rebondissement": "Hard cut. Beat of silence. Then reveal — zoom out fast.",
    "Point #1": "Number flies in from left. Content fades below. Staggered bullet reveal.",
    "Point #2": "Number transition — slide right, new number enters from left.",
    "Point #3": "Full screen number reveal. Particles or sparkle on '3'. Energetic.",
    "Point bonus": "Slide in from right. Lighter animation style. Bouncy text.",
    "Erreur #1": "Red flash frame. Shake effect. Error symbol scales up fast.",
    "Erreur #2": "Warning pulse. Slow camera zoom out. List grows organically.",
    "Erreur #3": "Focus pull — blur to sharp on hidden detail. Subtle zoom.",
    "La bonne approche": "Green glow transition. Visual checklist animates. Confident build.",
    "Critère #1": "Table draws in from top. Column headers animate sequentially.",
    "Critère #2": "Row slides in from right. Score bars fill left to right.",
    "Critère #3": "Rows highlight. Final column fades in. Dramatic pause before reveal.",
    "Verdict": "Winning side pulses gently. Losing side fades to 50% opacity. Stamps 'WINNER'.",
    "Résolution": "Crossfade to calmer scene. Music resolves. Gentle camera pull back.",
    "Résultat": "Reveal animation — curtain or radial wipe. Hold for 2s. Then subtle bounce.",
    "Leçon": "Text types out slowly. Background fades to warm tone. Reflective pace.",
    "Conclusion": "Summary cards stack. Check marks appear. Music swells slightly.",
    "CTA": "Screen compresses to show subscribe button. Bell icon shakes. Links pulse gently.",
}

# Effets sonores par type de scène
_SCENE_SOUNDS: Dict[str, str] = {
    "Hook": "Whoosh + impact sound. Music starts strong then drops to background.",
    "Introduction": "Background music at speaking volume. Subtle room tone.",
    "Contexte": "Ambient pad. Gentle underscore. Soft transition swoosh.",
    "Développement": "Rhythmic background beat. Subtle click on bullet points.",
    "Problème/Défi": "Tension building drone. Single piano note. Clock tick optional.",
    "Tentative": "Upbeat action music. Speed-up whoosh. Result reveal — cymbal crash.",
    "Tentative #1": "Button click sound. Music builds slightly in energy.",
    "Tentative #2": "Progress tone. Brief riser. Music energy increases.",
    "Tentative #3": "Drum roll build. Climactic chord on reveal. Silence then punch.",
    "Rebondissement": "Needle scratch. Complete silence for 1s. Then bass drop.",
    "Point #1": "Soft chime on number reveal. Pop sound for text line.",
    "Point #2": "Different pitched chime. Content swoosh for new info.",
    "Point #3": "Triumphant chord. Sparkle sound effect. Crowd cheer optional.",
    "Point bonus": "Bell sound — lighter, higher pitch. 'Ta-da' flourish.",
    "Erreur #1": "Buzzer sound. Error alert tone. Scratching record.",
    "Erreur #2": "Warning beep. Low rumble. Tension tone.",
    "Erreur #3": "Subtle 'miss' sound. Piano wrong note. Disappointed sigh.",
    "La bonne approche": "Correct answer chime. Success tone. 'Aha' moment sound.",
    "Critère #1": "Gentle click as table appears. Soft ping for first data point.",
    "Critère #2": "Whoosh for new row. Rising tone as score fills.",
    "Critère #3": "Deeper whoosh. Suspense tone. Drum hit for final reveal.",
    "Verdict": "Winner fanfare. Applause optional. Confident chord resolution.",
    "Résolution": "Music resolves to tonic. Warm ambient pad. Gentle exhale sound.",
    "Résultat": "Reveal impact sound. Sustained chord holds. Then satisfied exhale.",
    "Leçon": "Soft piano. Gentle string pad. Reflective silence between sentences.",
    "Conclusion": "Music swells slightly. Warm reverb on final words. Soft button click.",
    "CTA": "Subscribe sound effect. Bell ding. Music lift then fade to end.",
}

# Durée estimée par scène (secondes)
_SCENE_DURATIONS: Dict[str, int] = {
    "Hook": 8,
    "Introduction": 12,
    "Contexte": 10,
    "Développement": 15,
    "Problème/Défi": 10,
    "Tentative": 12,
    "Tentative #1": 12,
    "Tentative #2": 14,
    "Tentative #3": 16,
    "Rebondissement": 6,
    "Point #1": 16,
    "Point #2": 14,
    "Point #3": 18,
    "Point bonus": 10,
    "Erreur #1": 14,
    "Erreur #2": 14,
    "Erreur #3": 16,
    "La bonne approche": 20,
    "Critère #1": 14,
    "Critère #2": 14,
    "Critère #3": 16,
    "Verdict": 10,
    "Résolution": 12,
    "Résultat": 10,
    "Leçon": 15,
    "Conclusion": 12,
    "CTA": 10,
}


# ── HeuristicScriptGenerator ─────────────────────────────────────────────────

class HeuristicScriptGenerator(ScriptGenerator):
    """
    Générateur heuristique de scripts — aucun appel LLM.

    Construit un Script complet à partir d'un CreativeBrief et d'un BrandProfile
    en assemblant des templates paramétrés, des structures narratives prédéfinies,
    et des instructions audio/visuelles préparées pour les futurs moteurs.

    Le texte produit est simple (V1 d'architecture) ; la qualité rédactionnelle
    sera améliorée par les générateurs LLM dans les sprints suivants.

    Travail effectué :
      1. Résolution structure : angle → liste de scènes.
      2. Paramétrage : topic, hook, promesse, audience, CTA injectés.
      3. Enrichissement pour futurs moteurs : image_prompt, animation_notes,
         sound_effects, visual_description sur chaque scène.
      4. Ajustement de la durée depuis le BrandProfile (facteur multiplicateur).
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
            visual = _SCENE_VISUALS.get(section_name, "Plan standard.")
            image_prompt = _SCENE_IMAGE_PROMPTS.get(section_name, "Clean minimal composition, soft lighting.")
            animation = _SCENE_ANIMATIONS.get(section_name, "Standard cut. No animation.")
            sound = _SCENE_SOUNDS.get(section_name, "Background music continues.")
            duration = max(
                4,
                int(_SCENE_DURATIONS.get(section_name, 10) * brand_factor),
            )

            scene = ScriptScene(
                order=idx,
                title=section_name,
                narration=narration,
                visual_description=visual,
                image_prompt=image_prompt,
                animation_notes=animation,
                sound_effects=sound,
                duration_seconds=duration,
            )
            scenes.append(scene)

        # ── Métadonnées du script ──────────────────────────────────────────────
        estimated_duration = sum(s.duration_seconds for s in scenes)

        # Extraction de l'intro et conclusion
        intro_idx = self._find_scene_index(scenes, {"Introduction", "Contexte"})
        conclusion_idx = self._find_scene_index(scenes, {"Conclusion", "Leçon", "Résolution", "Verdict"})

        introduction = scenes[intro_idx].narration if intro_idx < len(scenes) else ""
        conclusion = scenes[conclusion_idx].narration if conclusion_idx < len(scenes) else ""

        script = Script(
            title=creative_brief.title,
            hook=hook_text,
            introduction=introduction,
            scenes=scenes,
            conclusion=conclusion,
            call_to_action=cta_text,
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
    def _find_scene_index(scenes: List[ScriptScene], candidates: set) -> int:
        """Trouve l'index de la première scène dont le titre correspond."""
        for i, s in enumerate(scenes):
            if s.title in candidates:
                return i
        return 0

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
