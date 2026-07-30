"""
Production Package Builder — Sprint 28 (Studio de production autonome),
mis à jour Sprint 31.1 (Storyboard JSON + nettoyage des métadonnées techniques),
Sprint 34.6 (prompts image/vidéo au format riche "mega-prompt"), Sprint 35
(1 niche/jour déclinée en 2 langues, visuels partagés), puis Sprint 38
(prompts texte brut fluides, format Google Veo3/Imagen, plus de JSON/labels).

Construit, pour LA niche/histoire du jour, le package de production "propre"
attendu en sortie quotidienne du pipeline :

    outputs/YYYY-MM-DD/niche_01/
        final_script_en.json
        final_script_fr.json
        image_prompts/            (UNIQUE, partagé — les visuels ne changent pas)
        animation_prompts_en/
        animation_prompts_fr/
        report.md
        story.txt                 (Sprint 37.4 — résumé narratif lisible, scène par scène)

Les dossiers techniques internes (shot_plans, .cache, benchmark.json) restent
écrits ailleurs par le pipeline (scripts/run_daily_pipeline.py) — ce module ne
les duplique jamais : seul ce qui est nécessaire à la production réelle de la
vidéo se retrouve dans niche_XX/.

Sprint 31.1 :
  - final_script_*.json adopte le format Storyboard Studio unifié
    (title + scenes[{order, scene, dialogues, transition, duration_seconds}])
    — aucun champ interne (metadata, language, style...) n'y est écrit.

Sprint 38 (remplace le format "mega-prompt" JSON du Sprint 34.6) :
  - image_prompts/scene_XX.txt et animation_prompts_*/scene_XX.txt sont du
    texte brut — UN paragraphe fluide unique, déjà écrit et garanti conforme
    par LLMImageGenerator/LLMAnimationGenerator (voir leurs
    _finalize_final_prompt : format Google Veo3/Imagen/Nano Banana, sans
    JSON, sans negative_prompt, sans mots bannis "8K"/"HDR"/"AAA quality"/
    "photorealistic", ≤80 mots image / ≤70 mots vidéo). Ce module se limite
    à écrire ce texte tel quel, y ajouter la ligne Audio/Narration
    déterministe (dialogues verbatim, voir _build_audio_line) pour la vidéo,
    et la note de référence de personnage récurrent le cas échéant — aucune
    reformulation, aucune concaténation de labels ici.

Sprint 35 :
  - Une seule niche/histoire est produite chaque jour, déclinée en 2 vidéos
    (anglais + français) qui partagent EXACTEMENT le même contenu visuel —
    un seul `image_prompts/`, deux `animation_prompts_en/`/`animation_prompts_fr/`
    identiques sauf la ligne Audio/Narration et la langue.

Ne dépend d'aucun autre moteur créatif : il consomme uniquement les objets
déjà produits (Script, ImagePrompt, AnimationPrompt, ShotPlan) via
NicheProductionResult.
"""

import dataclasses
import json
import logging
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.brand_engine import BrandProfile
from src.niche_intelligence import Niche
from src.script_engine import Dialogue, Script, estimate_scene_duration, split_dialogues_by_duration

logger = logging.getLogger(__name__)

# Sprint 36 — l'outil de génération vidéo de l'utilisateur ne produit jamais
# plus de 8 secondes par clip. Une scène plus longue est donc exportée en
# plusieurs fichiers animation_prompts_*/scene_XXa.txt, scene_XXb.txt...
# qui réutilisent tous la même image (même sujet/décor/style) mais couvrent
# chacun une tranche de dialogue distincte, à assembler bout à bout au montage.
#
# Sprint 37 — le budget de production (coût par génération de clip) impose
# désormais une cible native par scène dès l'écriture du script (voir
# MAX_SCENE_DURATION_SEC dans llm_script_generator.py) : ce découpage ne
# devrait donc plus jamais se déclencher en pratique — il reste ici comme
# filet de sécurité si un script dépasse malgré tout la cible.
#
# Sprint 37.3 — l'outil de génération vidéo accepte désormais des clips de
# 10s (au lieu de 8s). Sprint 37.5 — budget total porté à 90s (jusqu'à
# 9 scènes de 10s), pour une histoire plus développée/cohérente.
MAX_CLIP_DURATION_SECONDS = 10
_CLIP_SUFFIXES = string.ascii_lowercase


# ── Contrat d'entrée ──────────────────────────────────────────────────────────

@dataclass
class NicheProductionResult:
    """
    Résultat complet de production pour LA niche du jour (Sprint 35 — une
    seule niche, déclinée en 2 vidéos EN/FR). Regroupe ce qui est nécessaire
    au package final.
    """
    niche: Niche
    brand_en: BrandProfile              # packaging de la vidéo anglaise (ex: global_us)
    brand_fr: BrandProfile              # marque FR qui pilote le ton du script + packaging FR
    final_script_en: Script
    final_script_fr: Script
    images: List[Dict[str, Any]]           # [{"scene_order": int, "image_prompt": ImagePrompt, ...}] — partagé
    animations_en: List[Dict[str, Any]]    # [{"scene_order": int, "animation_prompt": AnimationPrompt}]
    animations_fr: List[Dict[str, Any]]    # mêmes AnimationPrompt que animations_en, dialogues/duration substitués
    rewrite_result: Optional[Dict[str, Any]] = None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _serialize_script(script: Script) -> Dict[str, Any]:
    """
    Projette un Script sur le contrat storyboard cinématographique (Sprint 32.1) :
    UNIQUEMENT {title, scenes[{scene: {number, type, description{9 champs}},
    dialogues, transition, duration_seconds}]}. Les champs internes du
    pipeline (metadata, language, style, target_audience, estimated_duration)
    ne font pas partie du contrat de production — ils restent sur l'objet
    Python pour les besoins internes (évaluateur, rapport) mais ne sont
    jamais écrits sur disque ici.
    """
    return {
        "title": script.title,
        "scenes": [
            {
                "scene": {
                    "number": scene.scene.number,
                    "type": scene.scene.type,
                    "description": {
                        "setting": scene.scene.description.setting,
                        "composition": scene.scene.description.composition,
                        "characters": scene.scene.description.characters,
                        "lighting": scene.scene.description.lighting,
                        "camera": scene.scene.description.camera,
                        "mood": scene.scene.description.mood,
                        "symbolism": scene.scene.description.symbolism,
                        "director_notes": scene.scene.description.director_notes,
                        "viewer_emotion": scene.scene.description.viewer_emotion,
                    },
                },
                "dialogues": [
                    {"personnage": d.personnage, "replique": d.replique}
                    for d in scene.dialogues
                ],
                "transition": scene.transition,
                "duration_seconds": scene.duration_seconds,
            }
            for scene in script.scenes
        ],
    }


# ── Cohérence des personnages récurrents (Sprint 37.3) ──────────────────────
# Sur un format court à peu de scènes, un même personnage nommé réapparaît
# souvent (ex: "Maya Hart" en scène 2, 3, 5, 7, 10). Les générateurs
# d'image/vidéo texte-vers-image n'ont aucune mémoire d'une scène à l'autre :
# sans référence explicite, chaque scène réinvente un visage différent pour
# le "même" personnage. On repère ici, dans l'ordre des scènes, la première
# apparition de chaque personnage nommé et on ajoute un renvoi explicite dans
# les scènes suivantes ("Character Reference: utiliser l'image de la scène
# XX comme référence visuelle") — l'utilisateur fournit alors cette image en
# entrée de son outil de génération, en plus du prompt, pour garder le même
# visage/coiffure/tenue sur tout le personnage.

_NAME_TOKEN_RE = re.compile(r"\b[A-Z][a-zA-Z'’-]+(?:\s+[A-Z][a-zA-Z'’-]+){0,2}\b")
_GENERIC_NAME_LEAD_WORDS = {
    "young", "old", "elderly", "middle-aged", "teenage", "a", "an", "the",
    "male", "female", "man", "woman", "boy", "girl", "narrator", "n/a", "none",
}


def _character_name_tokens(character_desc: str) -> set:
    """
    Extrait, depuis une entrée de la liste "characters" d'un ImagePrompt
    (ex: "Maya Hart, late 40s, short gray hair..."), les mots qui composent
    probablement un nom propre — le premier groupe de mots capitalisés du
    texte, en écartant les adjectifs descriptifs capitalisés en début de
    phrase (ex: une entrée commençant par "Young woman..." n'a pas de nom).
    """
    match = _NAME_TOKEN_RE.search(character_desc or "")
    if not match:
        return set()
    words = [w for w in match.group(0).split() if len(w) >= 3]
    if not words or words[0].lower() in _GENERIC_NAME_LEAD_WORDS:
        return set()
    return {w.lower() for w in words}


def _track_character_references(images: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Parcourt les images dans l'ordre des scènes et construit, pour chaque
    scene_order, le texte "Character Reference" à inclure dans le prompt —
    vide si aucun personnage de cette scène n'est déjà apparu avant.
    """
    known: List[Dict[str, Any]] = []  # [{"tokens": set, "name": str, "scene": int}]
    references: Dict[int, str] = {}

    for entry in sorted(images, key=lambda e: e["scene_order"]):
        scene_order = entry["scene_order"]
        meta = entry["image_prompt"].metadata or {}
        characters = meta.get("characters") or []
        notes: List[str] = []

        for character_desc in characters:
            tokens = _character_name_tokens(character_desc)
            if not tokens:
                continue
            match = next((k for k in known if k["tokens"] & tokens), None)
            if match is not None:
                notes.append(
                    f"{match['name']} already appeared in scene_{match['scene']:02d} — "
                    f"use the image generated for scene_{match['scene']:02d} "
                    "(image_prompts/scene_"
                    f"{match['scene']:02d}.txt) as the visual reference for this "
                    "character: keep the exact same face, hairstyle, clothing, and body type."
                )
            else:
                name = " ".join(w.capitalize() for w in sorted(tokens, key=lambda w: character_desc.lower().index(w)))
                known.append({"tokens": tokens, "name": name, "scene": scene_order})

        references[scene_order] = (
            " ".join(notes) if notes else "None (no recurring named character in this scene)."
        )
    return references


def _build_image_prompt_file(
    image_prompt: Any,
    character_reference: str = "None (no recurring named character in this scene).",
) -> str:
    """
    Construit le contenu texte brut de image_prompts/scene_XX.txt (Sprint 38
    — format Google Veo3/Imagen/Nano Banana) : le paragraphe fluide unique
    déjà écrit par LLMImageGenerator (image_prompt.prompt — déjà garanti
    conforme : format 9:16, identité stylisée, sans mots bannis, ≤80 mots,
    voir llm_image_generator._finalize_final_prompt), complété par la note
    de référence de personnage récurrent en une phrase naturelle. Aucun
    JSON, aucun negative_prompt, aucun label — un texte prêt à être copié
    tel quel dans l'outil de génération.
    """
    text = image_prompt.prompt.strip()
    if character_reference and not character_reference.startswith("None"):
        text = f"{text} {character_reference}"
    return text


def _build_audio_line(dialogues: List[Any], language: str) -> str:
    """
    Construit la ligne Audio/Narration finale (Sprint 38 — format Google
    Veo3) à partir des dialogues de la scène, VERBATIM — jamais reformulée
    ni générée par un LLM. Un narrateur devient "Narrator voiceover in
    {language}: ...", un personnage nommé devient "{Personnage} says: ..."
    — sans guillemets autour du texte parlé, comme l'exige le format Google.
    """
    lines: List[str] = []
    for d in dialogues:
        speaker = (d.personnage or "").strip()
        text = (d.replique or "").strip()
        if not text:
            continue
        if not speaker or speaker.upper() in ("NARRATEUR", "NARRATOR"):
            lines.append(f"Narrator voiceover in {language}: {text}")
        else:
            lines.append(f"{speaker} says: {text}")
    return " ".join(lines)


def _build_animation_prompt_file(
    animation_prompt: Any, language: str,
    character_reference: str = "None (no recurring named character in this scene).",
) -> str:
    """
    Construit le contenu texte brut de animation_prompts_*/scene_XX.txt
    (Sprint 38 — format Google Veo3) : le paragraphe de mouvement/son déjà
    écrit par LLMAnimationGenerator (animation_prompt.prompt — caméra/sujet/
    lumière/son UNIQUEMENT, jamais le dialogue — déjà garanti conforme, voir
    llm_animation_generator._finalize_final_prompt), suivi de la ligne
    Audio/Narration construite déterministiquement depuis les dialogues
    VERBATIM (_build_audio_line — jamais reformulés par un LLM), puis de la
    note de référence de personnage récurrent le cas échéant. Aucun JSON,
    aucun label.
    """
    audio_line = _build_audio_line(animation_prompt.dialogues, language)
    parts = [animation_prompt.prompt.strip()]
    if audio_line:
        parts.append(audio_line)
    if character_reference and not character_reference.startswith("None"):
        parts.append(character_reference)
    return " ".join(parts)


def _split_dialogues_for_clip_limit(
    dialogues: List[Dialogue], max_seconds: int = MAX_CLIP_DURATION_SECONDS,
) -> List[List[Dialogue]]:
    """
    Regroupe les répliques d'une scène en clips consécutifs dont la durée
    estimée ne dépasse jamais max_seconds — nécessaire car l'outil de
    génération vidéo cible ne produit que des clips de max_seconds maximum.
    Délègue à script_engine.split_dialogues_by_duration() (source unique de
    la logique de découpage sentence-safe, Sprint 37.6).
    """
    return split_dialogues_by_duration(dialogues, max_seconds)


def _split_animation_for_clip_limit(animation_prompt: Any, max_seconds: int = MAX_CLIP_DURATION_SECONDS) -> List[Any]:
    """
    Décline un AnimationPrompt (scène complète, potentiellement > 8s) en une
    liste d'AnimationPrompt "clips", chacun ≤ max_seconds. Réutilise
    intégralement tous les champs de mouvement/son/style de la scène — seuls
    dialogues/duration/transition diffèrent par clip. Seul le DERNIER clip
    porte la vraie transition vers la scène suivante ; les clips
    intermédiaires indiquent une continuité (même scène, à recoller au montage).
    """
    groups = _split_dialogues_for_clip_limit(animation_prompt.dialogues, max_seconds)
    if len(groups) <= 1:
        return [animation_prompt]

    clips = []
    last_index = len(groups) - 1
    for idx, group in enumerate(groups):
        is_last = idx == last_index
        clips.append(
            dataclasses.replace(
                animation_prompt,
                dialogues=group,
                duration=estimate_scene_duration(group) if group else 0,
                transition=(
                    animation_prompt.transition if is_last
                    else "Continuous shot — hard cut directly to the next clip of the same scene."
                ),
            )
        )
    return clips


# ── ProductionPackageBuilder ─────────────────────────────────────────────────

class ProductionPackageBuilder:
    """
    Exemple minimal :
        builder = ProductionPackageBuilder()
        package_dir = builder.build(output_dir, niche_index=1, result=niche_result)
    """

    def build(self, output_dir: Path, niche_index: int, result: NicheProductionResult) -> Path:
        package_dir = Path(output_dir) / f"niche_{niche_index:02d}"
        image_dir = package_dir / "image_prompts"
        animation_dir_en = package_dir / "animation_prompts_en"
        animation_dir_fr = package_dir / "animation_prompts_fr"
        package_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        animation_dir_en.mkdir(parents=True, exist_ok=True)
        animation_dir_fr.mkdir(parents=True, exist_ok=True)

        _write_json(package_dir / "final_script_en.json", _serialize_script(result.final_script_en))
        _write_json(package_dir / "final_script_fr.json", _serialize_script(result.final_script_fr))

        character_references = _track_character_references(result.images)

        for entry in sorted(result.images, key=lambda e: e["scene_order"]):
            _write_text(
                image_dir / f"scene_{entry['scene_order']:02d}.txt",
                _build_image_prompt_file(
                    entry["image_prompt"],
                    character_references.get(entry["scene_order"], "None (no recurring named character in this scene)."),
                ),
            )

        clip_counts: Dict[str, int] = {"English": 0, "French": 0}
        for animation_dir, animations, language in (
            (animation_dir_en, result.animations_en, "English"),
            (animation_dir_fr, result.animations_fr, "French"),
        ):
            for entry in sorted(animations, key=lambda e: e["scene_order"]):
                clips = _split_animation_for_clip_limit(entry["animation_prompt"])
                clip_counts[language] += len(clips)
                if len(clips) > 1:
                    logger.info(
                        "Scène %d (%ds, %s) découpée en %d clips de %ds max : %s",
                        entry["scene_order"], entry["animation_prompt"].duration, language,
                        len(clips), MAX_CLIP_DURATION_SECONDS,
                        ", ".join(f"{c.duration}s" for c in clips),
                    )
                character_reference = character_references.get(
                    entry["scene_order"], "None (no recurring named character in this scene)."
                )
                for idx, clip in enumerate(clips):
                    suffix = _CLIP_SUFFIXES[idx] if len(clips) > 1 else ""
                    _write_text(
                        animation_dir / f"scene_{entry['scene_order']:02d}{suffix}.txt",
                        _build_animation_prompt_file(clip, language, character_reference),
                    )

        (package_dir / "report.md").write_text(self._build_report(result, clip_counts), encoding="utf-8")
        (package_dir / "story.txt").write_text(self._build_story_text(result), encoding="utf-8")

        logger.info("Package de production créé : %s", package_dir)
        return package_dir

    @staticmethod
    def _scene_metrics_cells(metadata: Optional[Dict[str, Any]]) -> tuple:
        """Extrait (provider, statut, temps, coût) depuis un metadata NON strippé."""
        if not metadata:
            return ("—", "—", "—", "—")
        provider = str(metadata.get("provider", "—"))
        fallback_reason = metadata.get("fallback_reason")
        status = f"fallback ({fallback_reason})" if fallback_reason else "LLM"
        time_ms = metadata.get("time_ms", 0)
        cost = metadata.get("cost_usd", 0.0)
        return (provider, status, f"{time_ms} ms", f"${cost:.6f}")

    @staticmethod
    def _build_story_text(result: NicheProductionResult) -> str:
        """
        Construit story.txt (Sprint 37.4) : uniquement l'histoire, en
        français, telle que racontée par la narration/les dialogues du
        script FR — aucun élément technique du storyboard (caméra,
        transition, durée, type de scène...) n'y figure. Construit à partir
        des dialogues déjà traduits (DialogueTranslator) : aucune nouvelle
        génération ni traduction ici.
        """
        paragraphs = [
            scene.narration_text
            for scene in result.final_script_fr.scenes
            if scene.narration_text
        ]
        return "\n\n".join(paragraphs)

    @staticmethod
    def _build_report(result: NicheProductionResult, clip_counts: Optional[Dict[str, int]] = None) -> str:
        clip_counts = clip_counts or {}
        script_en = result.final_script_en
        script_fr = result.final_script_fr
        lines = [
            f"# Package de production — {result.niche.name}",
            "",
            f"**Chaîne EN :** {result.brand_en.name} ({result.brand_en.id})  ",
            f"**Chaîne FR :** {result.brand_fr.name} ({result.brand_fr.id})  ",
            f"**Niche :** {result.niche.name} (score={result.niche.niche_score:.3f})  ",
            f"**Titre :** {script_en.title}  ",
            f"**Hook (EN) :** {script_en.hook}  ",
            f"**Hook (FR) :** {script_fr.hook}  ",
            f"**Durée estimée EN :** {script_en.estimated_duration}s — **FR :** {script_fr.estimated_duration}s  ",
            f"**Scènes :** {len(script_en.scenes)}  ",
        ]
        if result.rewrite_result is not None:
            applied = result.rewrite_result.get("rewrite_applied")
            lines.append(f"**Réécriture :** {'appliquée' if applied else 'non appliquée'}  ")
        lines += [
            "",
            f"- Prompts image générés : {len(result.images)} (partagés entre les 2 langues)",
            f"- Scènes → clips vidéo (limite {MAX_CLIP_DURATION_SECONDS}s/clip) : "
            f"{len(result.animations_en)} scènes → {clip_counts.get('English', len(result.animations_en))} "
            f"clips EN / {clip_counts.get('French', len(result.animations_fr))} clips FR "
            "(une scène plus longue que "
            f"{MAX_CLIP_DURATION_SECONDS}s est exportée en plusieurs fichiers scene_XXa/b/c.txt, "
            "même image, à recoller au montage)",
            "",
            "## Métriques techniques par scène",
            "",
            "Source unique des informations techniques (provider, modèle, temps, "
            "coût, statut, fallback) — ces champs n'apparaissent plus dans "
            "`image_prompts/*.txt` ni `animation_prompts_*/*.txt` (textes bruts, Sprint 38). "
            "Les métriques d'animation ci-dessous portent sur la génération anglaise "
            "(seule à faire un appel LLM — la version française réutilise ses résultats).",
            "",
            "| Scène | Image — provider | Image — statut | Image — temps | Image — coût "
            "| Animation — provider | Animation — statut | Animation — temps | Animation — coût |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        images_by_order = {e["scene_order"]: e["image_prompt"] for e in result.images}
        animations_by_order = {e["scene_order"]: e["animation_prompt"] for e in result.animations_en}
        all_orders = sorted(set(images_by_order) | set(animations_by_order))

        for order in all_orders:
            img = images_by_order.get(order)
            anim = animations_by_order.get(order)
            img_cells = ProductionPackageBuilder._scene_metrics_cells(
                img.metadata if img is not None else None
            )
            anim_cells = ProductionPackageBuilder._scene_metrics_cells(
                anim.metadata if anim is not None else None
            )
            lines.append(
                f"| {order} | {img_cells[0]} | {img_cells[1]} | {img_cells[2]} | {img_cells[3]} "
                f"| {anim_cells[0]} | {anim_cells[1]} | {anim_cells[2]} | {anim_cells[3]} |"
            )

        lines.append("")
        return "\n".join(lines)
