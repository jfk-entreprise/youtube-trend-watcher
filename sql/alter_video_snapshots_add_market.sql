-- ============================================================
-- Sprint 34 — Ajout de la colonne market à video_snapshots
-- À exécuter dans l'éditeur SQL de Supabase (une seule fois)
-- ============================================================
--
-- Corrige un bug de fond : TrendingAgent stockait le code région
-- ("US", "FR", "CI") dans la colonne `keyword` faute de colonne dédiée,
-- ce qui polluait la détection de niche (NicheAnalyzer dérive le nom de
-- niche depuis `keyword` — voir src/niche_intelligence.py::_niche_name).
-- `market` devient la source de vérité du marché d'une vidéo collectée
-- (Sprint 34 — segmentation US/FR), `keyword` redevient un vrai sujet.

ALTER TABLE video_snapshots ADD COLUMN IF NOT EXISTS market TEXT NOT NULL DEFAULT 'FR';

CREATE INDEX IF NOT EXISTS idx_video_snapshots_market
    ON video_snapshots (market);

-- Backfill/nettoyage des données historiques polluées par le bug ci-dessus :
-- les lignes 'trending' dont keyword est en fait un code région migrent ce
-- code vers market, et keyword redevient vide (bascule sur "(trending)").
UPDATE video_snapshots
SET market = keyword, keyword = ''
WHERE source = 'trending' AND keyword IN ('US', 'FR', 'CI');

-- Les lignes source='keyword' gardent market='FR' (valeur par défaut),
-- cohérent avec le comportement historique du collecteur (toujours
-- region_code='FR' avant ce sprint).
