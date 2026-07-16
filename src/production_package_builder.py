"""
Production Package Builder — Sprint 28 (Studio de production autonome),
mis à jour Sprint 31.1 (Storyboard JSON + nettoyage des métadonnées techniques),
Sprint 34.6 (prompts image/vidéo au format riche "mega-prompt"), puis
Sprint 35 (1 niche/jour déclinée en 2 langues, visuels partagés).

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

Sprint 34.6 :
  - image_prompts/scene_XX.json et animation_prompts_*/scene_XX.json adoptent
    un format "mega-prompt" à 3 clés {prompt, negative_prompt, instruction_format}
    — le champ "prompt" concatène des libellés riches ("Subject: ... Clothing:
    ... Camera Angle: ...") construits à partir du contenu déjà généré
    (ImagePrompt/AnimationPrompt, ShotPlan, SceneDescription, BrandProfile) —
    aucune nouvelle génération LLM ici, uniquement une reformulation.

Sprint 35 :
  - Une seule niche/histoire est produite chaque jour, déclinée en 2 vidéos
    (anglais + français) qui partagent EXACTEMENT le même contenu visuel —
    un seul `image_prompts/`, deux `animation_prompts_en/`/`animation_prompts_fr/`
    identiques sauf Dialogue/Speaker/Narration/Language/Scene Duration.

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
from src.script_engine import Dialogue, Script, estimate_scene_duration

logger = logging.getLogger(__name__)

_INSTRUCTION_FORMAT = "Respond STRICTLY in valid JSON. Do not include any explanation or markdown."

# Sprint 36 — l'outil de génération vidéo de l'utilisateur ne produit jamais
# plus de 8 secondes par clip. Une scène plus longue est donc exportée en
# plusieurs fichiers animation_prompts_*/scene_XXa.json, scene_XXb.json...
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

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


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


def _s(value: Any, default: str = "unspecified") -> str:
    """Normalise une valeur texte potentiellement absente/vide."""
    text = str(value).strip() if value is not None else ""
    return text or default


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
                    f"{match['scene']:02d}.json) as the visual reference for this "
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
    image_prompt: Any, shot_plan: Optional[Any], description: Any, brand: BrandProfile,
    character_reference: str = "None (no recurring named character in this scene).",
) -> Dict[str, Any]:
    """
    Construit le fichier image_prompts/scene_XX.json (Sprint 34.6) : un
    "mega-prompt" texte unique regroupant des libellés riches, à partir du
    contenu déjà généré par LLMImageGenerator (ImagePrompt), le VisualDirector
    (ShotPlan, si disponible) et l'identité de marque (BrandProfile) — aucune
    nouvelle génération ici, uniquement une reformulation.
    """
    meta = image_prompt.metadata or {}
    camera_angle = shot_plan.camera_angle if shot_plan else description.camera
    lens = shot_plan.lens if shot_plan else "unspecified"
    composition = shot_plan.composition if shot_plan else description.composition
    color_palette = shot_plan.color_palette if shot_plan else ", ".join(brand.color_palette)
    details = " ".join(
        part for part in (description.symbolism, description.director_notes) if part
    )

    fields = [
        ("Subject", image_prompt.subject),
        ("Appearance", meta.get("appearance")),
        ("Clothing", meta.get("clothing")),
        ("Accessories", meta.get("accessories")),
        ("Pose", meta.get("pose")),
        ("Action", image_prompt.prompt),
        ("Facial Expression", meta.get("facial_expression")),
        ("Emotion", meta.get("emotion")),
        ("Environment", description.setting),
        ("Background", meta.get("background")),
        ("Weather", meta.get("weather")),
        ("Time of Day", meta.get("time_of_day")),
        ("Lighting", description.lighting),
        ("Camera Angle", camera_angle),
        ("Lens", lens),
        ("Composition", composition),
        ("Style", image_prompt.style),
        ("Color Palette", color_palette),
        ("Character Reference", character_reference),
        ("Details", details),
        ("Text (optional)", "None"),
        ("Language", "None (no on-screen text)"),
    ]
    prompt = " ".join(f"{label}: {_s(value)}." for label, value in fields)

    return {
        "prompt": prompt,
        "negative_prompt": image_prompt.negative_prompt,
        "instruction_format": _INSTRUCTION_FORMAT,
    }


def _dialogue_fields(dialogues: List[Any]) -> Dict[str, str]:
    """Dérive Dialogue/Speaker/Narration (verbatim) depuis les répliques de la scène."""
    if not dialogues:
        return {"dialogue": "None", "speaker": "None", "narration": "None"}
    speakers = [
        "NARRATOR" if not d.personnage.strip() or d.personnage.strip().upper() in ("NARRATEUR", "NARRATOR")
        else d.personnage.strip()
        for d in dialogues
    ]
    lines = [d.replique for d in dialogues]
    narration = " ".join(
        d.replique for d in dialogues if (d.personnage or "").strip().upper() in ("", "NARRATEUR", "NARRATOR")
    )
    return {
        "dialogue": " / ".join(lines),
        "speaker": " / ".join(speakers),
        "narration": narration or "None",
    }


def _build_animation_prompt_file(
    animation_prompt: Any, image_prompt: Any, shot_plan: Optional[Any], description: Any, language: str,
    character_reference: str = "None (no recurring named character in this scene).",
) -> Dict[str, Any]:
    """
    Construit le fichier animation_prompts/scene_XX.json (Sprint 34.6) — même
    principe que _build_image_prompt_file, en réutilisant en plus l'ImagePrompt
    de la même scène (apparence/vêtements déjà établis) et l'AnimationPrompt
    (mouvement/son/transition déjà générés) : aucune nouvelle génération ici.
    """
    meta = animation_prompt.metadata or {}
    img_meta = image_prompt.metadata or {}
    camera_angle = shot_plan.camera_angle if shot_plan else description.camera
    lens = shot_plan.lens if shot_plan else "unspecified"
    composition = shot_plan.composition if shot_plan else description.composition
    shot_type = shot_plan.shot_type if shot_plan else "unspecified"
    dialogue_fields = _dialogue_fields(animation_prompt.dialogues)

    fields = [
        ("Subject", image_prompt.subject),
        ("Appearance", img_meta.get("appearance")),
        ("Clothing", img_meta.get("clothing")),
        ("Accessories", img_meta.get("accessories")),
        ("Initial Pose", img_meta.get("pose")),
        ("Character Action", animation_prompt.subject_motion),
        ("Secondary Actions", animation_prompt.environment_motion),
        ("Facial Expression", img_meta.get("facial_expression")),
        ("Emotion", meta.get("emotion")),
        ("Environment", description.setting),
        ("Background", img_meta.get("background")),
        ("Weather", img_meta.get("weather")),
        ("Time of Day", img_meta.get("time_of_day")),
        ("Lighting", animation_prompt.lighting_changes),
        ("Camera Shot", shot_type),
        ("Camera Angle", camera_angle),
        ("Camera Movement", animation_prompt.camera_motion),
        ("Lens", lens),
        ("Composition", composition),
        ("Visual Style", image_prompt.style),
        ("Character Reference", character_reference),
        ("Animation Style", meta.get("animation_style")),
        ("Scene Duration", f"{animation_prompt.duration}s"),
        ("Frame Rate", "24 fps"),
        ("Dialogue", dialogue_fields["dialogue"]),
        ("Speaker", dialogue_fields["speaker"]),
        ("Narration", dialogue_fields["narration"]),
        ("Language", language),
        ("Voice", meta.get("voice")),
        ("Lip Sync", "Synced to spoken dialogue audio"),
        ("Sound Effects", meta.get("sound_effects")),
        ("Ambient Sounds", animation_prompt.sound_design),
        ("Background Music", meta.get("background_music")),
        ("Atmosphere", description.mood),
        ("Ending Scene", animation_prompt.transition),
    ]
    prompt = " ".join(f"{label}: {_s(value)}." for label, value in fields)

    return {
        "prompt": prompt,
        "negative_prompt": image_prompt.negative_prompt,
        "instruction_format": _INSTRUCTION_FORMAT,
    }


def _split_single_dialogue(dialogue: Dialogue, max_seconds: int) -> List[Dialogue]:
    """
    Scinde UNE réplique trop longue pour tenir seule dans max_seconds, en
    coupant sur les frontières de phrases (jamais au milieu d'une phrase),
    et en dernier recours sur les mots si une phrase unique dépasse déjà
    max_seconds à elle seule.
    """
    if estimate_scene_duration([dialogue]) <= max_seconds:
        return [dialogue]

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(dialogue.replique.strip()) if s]
    if len(sentences) <= 1:
        words = dialogue.replique.split()
        words_per_second = estimate_scene_duration.__globals__["NARRATION_WORDS_PER_MINUTE"] / 60.0
        max_words = max(1, int(max_seconds * words_per_second))
        return [
            Dialogue(personnage=dialogue.personnage, replique=" ".join(words[i : i + max_words]))
            for i in range(0, len(words), max_words)
        ] or [dialogue]

    parts: List[Dialogue] = []
    current: List[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and estimate_scene_duration([Dialogue(dialogue.personnage, candidate)]) > max_seconds:
            parts.append(Dialogue(personnage=dialogue.personnage, replique=" ".join(current)))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        parts.append(Dialogue(personnage=dialogue.personnage, replique=" ".join(current)))
    return parts


def _split_dialogues_for_clip_limit(
    dialogues: List[Dialogue], max_seconds: int = MAX_CLIP_DURATION_SECONDS,
) -> List[List[Dialogue]]:
    """
    Regroupe les répliques d'une scène en clips consécutifs dont la durée
    estimée ne dépasse jamais max_seconds — nécessaire car l'outil de
    génération vidéo cible ne produit que des clips de 8s maximum. Une seule
    réplique déjà trop longue est elle-même scindée (voir _split_single_dialogue).
    """
    atomic: List[Dialogue] = []
    for d in dialogues:
        atomic.extend(_split_single_dialogue(d, max_seconds))

    if not atomic:
        return [[]]

    groups: List[List[Dialogue]] = []
    current: List[Dialogue] = []
    for d in atomic:
        candidate = current + [d]
        if current and estimate_scene_duration(candidate) > max_seconds:
            groups.append(current)
            current = [d]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


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

        # Les descriptions de scène (setting/lighting/camera/mood...) sont
        # partagées entre les 2 langues — seules les répliques diffèrent
        # (final_script_en/final_script_fr ont les mêmes scene.number).
        scenes_by_number = {s.scene.number: s for s in result.final_script_en.scenes}
        images_by_order = {e["scene_order"]: e for e in result.images}
        character_references = _track_character_references(result.images)

        for entry in sorted(result.images, key=lambda e: e["scene_order"]):
            script_scene = scenes_by_number.get(entry["scene_order"])
            description = script_scene.scene.description if script_scene else None
            _write_json(
                image_dir / f"scene_{entry['scene_order']:02d}.json",
                _build_image_prompt_file(
                    entry["image_prompt"], entry.get("shot_plan"), description, result.brand_en,
                    character_references.get(entry["scene_order"], "None (no recurring named character in this scene)."),
                ),
            )

        clip_counts: Dict[str, int] = {"English": 0, "French": 0}
        for animation_dir, animations, language in (
            (animation_dir_en, result.animations_en, "English"),
            (animation_dir_fr, result.animations_fr, "French"),
        ):
            for entry in sorted(animations, key=lambda e: e["scene_order"]):
                script_scene = scenes_by_number.get(entry["scene_order"])
                description = script_scene.scene.description if script_scene else None
                image_entry = images_by_order.get(entry["scene_order"], {})
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
                    _write_json(
                        animation_dir / f"scene_{entry['scene_order']:02d}{suffix}.json",
                        _build_animation_prompt_file(
                            clip, image_entry.get("image_prompt"),
                            image_entry.get("shot_plan"), description, language,
                            character_reference,
                        ),
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
            f"{MAX_CLIP_DURATION_SECONDS}s est exportée en plusieurs fichiers scene_XXa/b/c.json, "
            "même image, à recoller au montage)",
            "",
            "## Métriques techniques par scène",
            "",
            "Source unique des informations techniques (provider, modèle, temps, "
            "coût, statut, fallback) — ces champs n'apparaissent plus dans "
            "`image_prompts/*.json` ni `animation_prompts_*/*.json` (Sprint 31.1). "
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
