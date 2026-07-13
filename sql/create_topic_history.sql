-- ============================================================
-- Sprint 33 — Initialisation de la table topic_history
-- À exécuter dans l'éditeur SQL de Supabase (une seule fois)
-- ============================================================
--
-- Persiste les sujets/histoires déjà produits par le pipeline quotidien
-- (scripts/run_daily_pipeline.py), pour éviter de reproduire un sujet
-- quasi identique dans une même niche d'un jour à l'autre — et pour
-- pouvoir traiter un sujet proche comme une SUITE explicite plutôt qu'un
-- remake silencieux (voir src/topic_history.py — TopicHistoryFilter).

CREATE TABLE IF NOT EXISTS topic_history (
    id                    BIGSERIAL    PRIMARY KEY,

    -- Sujet produit (titre du Script final — Script.title)
    title                 TEXT         NOT NULL,
    niche                 TEXT         NOT NULL,
    brand_id              TEXT         NOT NULL,
    source_video_id       TEXT         NOT NULL,

    -- Date de production (comparée à une fenêtre glissante, ex. 5 jours)
    produced_date         DATE         NOT NULL,

    metadata              JSONB        DEFAULT '{}'::jsonb
);

-- Index sur (niche, produced_date) : c'est exactement le filtre utilisé par
-- TopicHistoryFilter.classify() à chaque run (niche courante × fenêtre
-- glissante de N derniers jours).
CREATE INDEX IF NOT EXISTS idx_topic_history_niche_date
    ON topic_history (niche, produced_date);
