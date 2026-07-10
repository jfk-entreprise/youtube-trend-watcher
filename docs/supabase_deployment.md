# Guide de déploiement — Sprint 6 : Migration Supabase

## Prérequis

- Un projet Supabase actif (gratuit suffisant)
- Python 3.11+ avec l'environnement virtuel activé
- Le fichier `.env` à la racine du projet

---

## Étape 1 — Installer la dépendance

```bash
pip install supabase>=2.0.0
# ou si vous utilisez requirements.txt :
pip install -r requirements.txt
```

---

## Étape 2 — Configurer le fichier `.env`

Ajoutez (ou vérifiez) ces deux variables dans `.env` :

```env
SUPABASE_URL=https://<votre-project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...   # Clé service_role (Settings > API)
```

> **Où trouver ces valeurs :**  
> Supabase Dashboard → votre projet → Settings → API  
> Copiez **Project URL** et la clé **service_role** (pas `anon`).

---

## Étape 3 — Créer la table dans Supabase

1. Dans le Dashboard Supabase, ouvrez **SQL Editor**.
2. Collez le contenu de `sql/create_video_snapshots.sql`.
3. Cliquez **Run**.

Vous devriez voir `Success. No rows returned.` — la table et les 4 index sont créés.

### Sprint 28 — Table `active_niches` (Studio de production)

Répétez la même procédure avec `sql/create_active_niches.sql` : cette table
persiste les niches actives d'un jour sur l'autre pour `run_daily_pipeline.py`
(voir `src/niche_selector.py`). Sans elle (ou sans credentials Supabase), le
pipeline bascule automatiquement sur un fichier JSON local
(`.cache/active_niches.json`).

---

## Étape 4 — Valider la connexion

```bash
python scripts/validate_supabase.py
```

Le script insère un snapshot de test, le relit, puis le supprime.  
Un encadré `SUCCES` confirme que tout fonctionne.

---

## Étape 5 — Lancer la collecte

Les scripts `sprint2_collect.py` et `test_agents.py` détectent automatiquement
Supabase si les variables sont présentes :

```bash
# Collecte par mots-clés
python scripts/sprint2_collect.py

# Collecte multi-agents (keyword + trending)
python scripts/test_agents.py
```

Les logs afficheront `Backend actif : Supabase` à chaque démarrage.  
En cas de panne Supabase, les données basculent automatiquement vers `data/videos.csv`.

---

## Vérifier les données dans Supabase

Dans **Table Editor** → `video_snapshots`, ou via SQL :

```sql
-- Les 10 snapshots les plus récents
SELECT video_id, title, source, view_count, collected_at
FROM video_snapshots
ORDER BY collected_at DESC
LIMIT 10;

-- Nombre de snapshots par source
SELECT source, COUNT(*) FROM video_snapshots GROUP BY source;

-- Timeline d'une vidéo spécifique
SELECT view_count, like_count, collected_at
FROM video_snapshots
WHERE video_id = '<video_id>'
ORDER BY collected_at;
```

---

## Architecture du backend de stockage

```
build_storage(csv_fallback_path)
       │
       ├─ SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY présents ?
       │       │
       │       YES → FallbackStorage
       │               ├─ primary  : SupabaseStorage  (Supabase cloud)
       │               └─ fallback : CsvStorage       (data/videos.csv)
       │
       └─ NO → CsvStorage (data/videos.csv)
```

La logique métier (agents, virality engine) n'est **pas modifiée**.
