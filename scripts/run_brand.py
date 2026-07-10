"""
Brand Engine — Script de validation et d'affichage.
Usage : python scripts/run_brand.py

Charge tous les profils de marque depuis brands/,
les valide et affiche un résumé complet.
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.brand_engine import BrandEngine, BrandProfile, validate_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BRANDS_DIR = ROOT / "brands"


# ── Formatage ─────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    return f"{m}m{s:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m"


def _level_bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def _level_label(value: float) -> str:
    if value >= 0.80:
        return "Très élevé"
    if value >= 0.60:
        return "Élevé"
    if value >= 0.40:
        return "Modéré"
    if value >= 0.20:
        return "Faible"
    return "Très faible"


def build_report(profiles: list[BrandProfile], issues_map: dict[str, list[str]]) -> str:
    SEP = "=" * 72
    THIN = "-" * 72
    lines = [
        SEP,
        "  BRAND ENGINE v1 — PROFILS DE MARQUE",
        f"  Répertoire  : brands/",
        f"  Profils     : {len(profiles)} marque(s) chargée(s)",
        f"  Valides     : {sum(1 for v in issues_map.values() if not v)}",
        f"  Avec erreurs: {sum(1 for v in issues_map.values() if v)}",
        SEP,
    ]

    for profile in profiles:
        issues = issues_map.get(profile.id, [])
        status = "✓ VALIDE" if not issues else f"✗ {len(issues)} ERREUR(S)"

        lines += [
            "",
            f"  ┌── {profile.name.upper()}  [{status}]",
            THIN,
            f"  │  ID          : {profile.id}",
            f"  │  Description : {profile.description[:65]}",
            f"  │  Niche       : {profile.niche}",
            f"  │  Audience    : {profile.target_audience[:60]}",
            f"  │  Langue      : {profile.primary_language}",
            f"  │  Ton         : {profile.tone}",
            f"  │  Vitesse     : {profile.voice_speed}",
            f"  │  Durée cible : {_fmt_duration(profile.preferred_video_duration)}",
            "  │",
            "  │  NIVEAUX ÉDITORIAUX",
            f"  │  Émotion       {profile.emotion_level:.2f}  {_level_bar(profile.emotion_level)}  {_level_label(profile.emotion_level)}",
            f"  │  Humour        {profile.humor_level:.2f}  {_level_bar(profile.humor_level)}  {_level_label(profile.humor_level)}",
            f"  │  Autorité      {profile.authority_level:.2f}  {_level_bar(profile.authority_level)}  {_level_label(profile.authority_level)}",
            f"  │  Curiosité     {profile.curiosity_level:.2f}  {_level_bar(profile.curiosity_level)}  {_level_label(profile.curiosity_level)}",
            f"  │  Storytelling  {profile.storytelling_level:.2f}  {_level_bar(profile.storytelling_level)}  {_level_label(profile.storytelling_level)}",
            "  │",
            f"  │  FORMATS ({len(profile.preferred_formats)})",
        ]
        for fmt in profile.preferred_formats:
            lines.append(f"  │    • {fmt}")

        lines += [
            "  │",
            f"  │  HOOKS ({len(profile.preferred_hooks)})",
        ]
        for hook in profile.preferred_hooks:
            lines.append(f"  │    → {hook[:68]}")

        lines += [
            "  │",
            f"  │  CTA ({len(profile.preferred_cta)})",
        ]
        for cta in profile.preferred_cta:
            lines.append(f"  │    → {cta[:68]}")

        lines += [
            "  │",
            f"  │  MOTS INTERDITS ({len(profile.forbidden_words)})",
            f"  │    {', '.join(profile.forbidden_words)}",
            "  │",
            "  │  IDENTITÉ VISUELLE",
            f"  │  Style    : {profile.visual_style[:65]}",
            f"  │  Couleurs : {', '.join(profile.color_palette[:4])}",
            f"  │  Typo     : {profile.typography_style[:65]}",
            f"  │  Miniature: {profile.thumbnail_style[:65]}",
        ]

        pillars = profile.metadata.get("content_pillars", [])
        freq = profile.metadata.get("posting_frequency", "—")
        platforms = profile.metadata.get("platform_priority", [])
        lines += [
            "  │",
            "  │  STRATÉGIE",
            f"  │  Fréquence  : {freq}",
            f"  │  Plateformes: {', '.join(platforms)}",
            f"  │  Piliers    : {', '.join(pillars)}",
        ]

        if issues:
            lines += ["  │", "  │  ERREURS DE VALIDATION"]
            for issue in issues:
                lines.append(f"  │    ✗ {issue}")

        lines.append(f"  └{'─' * 60}")

    # ── Résumé comparatif ──────────────────────────────────────────────────────
    if len(profiles) > 1:
        lines += ["", SEP, "  COMPARATIF DES MARQUES", THIN]
        header = f"  {'Marque':<22} {'Émotion':>8} {'Humour':>8} {'Autorité':>9} {'Curiosité':>10} {'Story':>7} {'Durée':>8}"
        lines += [header, f"  {'─'*22} {'─'*8} {'─'*8} {'─'*9} {'─'*10} {'─'*7} {'─'*8}"]
        for p in profiles:
            lines.append(
                f"  {p.name:<22}"
                f" {p.emotion_level:>8.2f}"
                f" {p.humor_level:>8.2f}"
                f" {p.authority_level:>9.2f}"
                f" {p.curiosity_level:>10.2f}"
                f" {p.storytelling_level:>7.2f}"
                f" {_fmt_duration(p.preferred_video_duration):>8}"
            )

    lines += ["", SEP, "  FIN DU RAPPORT", SEP]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    engine = BrandEngine(brands_dir=BRANDS_DIR)
    profiles = engine.list()

    if not profiles:
        logger.error("Aucun profil trouvé dans brands/")
        sys.exit(1)

    logger.info("%d marque(s) chargée(s) depuis brands/", len(profiles))

    issues_map: dict[str, list[str]] = {}
    for profile in profiles:
        issues = engine.validate(profile)
        issues_map[profile.id] = issues
        if issues:
            logger.warning("Profil '%s' : %d problème(s) détecté(s).", profile.id, len(issues))
        else:
            logger.info("Profil '%s' : valide.", profile.id)

    report = build_report(profiles, issues_map)
    print("\n" + report)


if __name__ == "__main__":
    main()
