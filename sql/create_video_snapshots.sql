-- ============================================================
-- Sprint 6 — Initialisation de la table video_snapshots
-- À exécuter dans l'éditeur SQL de Supabase (une seule fois)
-- ============================================================

CREATE TABLE IF NOT EXISTS video_snapshots (
    id               BIGSERIAL    PRIMARY KEY,

    -- Identité de la vidéo
    video_id         TEXT         NOT NULL,
    title            TEXT         NOT NULL,
    channel_id       TEXT         NOT NULL,
    channel_title    TEXT         NOT NULL,
    published_at     TIMESTAMPTZ  NOT NULL,
    description      TEXT,

    -- Durée
    duration_iso     TEXT         NOT NULL,
    duration_seconds INTEGER      NOT NULL CHECK (duration_seconds >= 0),

    -- Statistiques (nullable : certains créateurs les masquent)
    view_count       BIGINT,
    like_count       BIGINT,
    comment_count    BIGINT,

    -- Contexte de collecte
    keyword          TEXT         NOT NULL,
    source           TEXT         NOT NULL DEFAULT 'keyword'
                         CHECK (source IN ('keyword', 'trending')),
    collected_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Index sur video_id (jointures, lookup par vidéo)
CREATE INDEX IF NOT EXISTS idx_vsnap_video_id
    ON video_snapshots (video_id);

-- Index sur collected_at DESC (requêtes "les N plus récents")
CREATE INDEX IF NOT EXISTS idx_vsnap_collected_at
    ON video_snapshots (collected_at DESC);

-- Index sur source (filtre keyword vs trending)
CREATE INDEX IF NOT EXISTS idx_vsnap_source
    ON video_snapshots (source);

-- Index composite pour les timelines (virality engine : groupe + tri)
CREATE INDEX IF NOT EXISTS idx_vsnap_timeline
    ON video_snapshots (video_id, collected_at DESC);
