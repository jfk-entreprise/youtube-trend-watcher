# Guide d'exploitation — GitHub Actions

Ce guide décrit comment configurer, déclencher et adapter le workflow d'automatisation du pipeline de veille YouTube.

---

## 1. Configuration des GitHub Secrets

Les clés d'API ne doivent **jamais** être écrites en dur dans le dépôt. Le workflow les lit exclusivement via les Secrets GitHub.

### Procédure pas à pas

1. Ouvrir le dépôt sur GitHub.
2. Cliquer sur **Settings** (onglet supérieur, visible uniquement pour les administrateurs).
3. Dans le menu latéral gauche, aller dans **Secrets and variables → Actions**.
4. Cliquer sur **New repository secret** pour chaque variable ci-dessous.

### Secrets à configurer

| Nom du secret | Obligatoire | Description |
|---------------|:-----------:|-------------|
| `YOUTUBE_API_KEY` | **Oui** | Clé d'API YouTube Data API v3 (Google Cloud Console) |
| `SUPABASE_URL` | Non | URL du projet Supabase (ex. : `https://xxx.supabase.co`) |
| `SUPABASE_SERVICE_ROLE_KEY` | Non | Clé service role Supabase (onglet API du dashboard) |
| `SLACK_WEBHOOK_URL` | Non | URL du webhook Slack pour les notifications |

> Sans `SUPABASE_URL`, le pipeline opère en mode **CSV uniquement** — les données sont sauvegardées localement dans `data/videos.csv` pendant l'exécution du runner, mais ne persistent pas entre deux runs. Configurer Supabase est recommandé pour accumuler de l'historique et activer les métriques temporelles du Virality Engine.

---

## 2. Déclenchement manuel du workflow

### Depuis l'interface GitHub

1. Aller dans l'onglet **Actions** du dépôt.
2. Dans le menu latéral gauche, sélectionner le workflow **YouTube Trend Watcher — Pipeline 6h**.
3. Cliquer sur le bouton **Run workflow** (affiché à droite de la liste des exécutions).
4. Choisir la branche cible (généralement `main`) et confirmer avec **Run workflow**.

L'exécution démarre dans les secondes suivantes et est visible en temps réel dans la console du runner.

### Depuis la CLI GitHub (`gh`)

```bash
gh workflow run collect.yml --ref main
```

---

## 3. Modification de la fréquence d'exécution

La planification est définie par la syntaxe **cron** dans `.github/workflows/collect.yml` :

```yaml
on:
  schedule:
    - cron: '0 */6 * * *'   # Toutes les 6 heures
```

### Syntaxe cron (5 champs)

```
┌──────────── minute       (0–59)
│  ┌─────────── heure        (0–23, UTC)
│  │  ┌──────────── jour du mois (1–31)
│  │  │  ┌─────────── mois         (1–12)
│  │  │  │  ┌──────────── jour de semaine (0–7, 0=dimanche)
│  │  │  │  │
*  *  *  *  *
```

### Exemples courants

| Fréquence souhaitée | Expression cron |
|---------------------|-----------------|
| Toutes les 6 heures | `0 */6 * * *` |
| Toutes les 12 heures | `0 */12 * * *` |
| Une fois par jour à 08h00 UTC | `0 8 * * *` |
| Toutes les heures | `0 * * * *` |
| Lundi, mercredi, vendredi à 07h00 | `0 7 * * 1,3,5` |

> GitHub Actions exécute les crons en **UTC**. Adapter l'heure en conséquence (UTC+1 en hiver, UTC+2 en été pour la France).

> La fréquence minimale supportée par GitHub Actions est **toutes les 5 minutes** (`*/5 * * * *`). En pratique, les crons fréquents peuvent subir un délai de quelques minutes selon la charge des runners.

---

## 4. Récupération des rapports (artefacts)

Après chaque exécution, les rapports sont sauvegardés comme artefacts et conservés **30 jours**.

1. Aller dans l'onglet **Actions**.
2. Cliquer sur l'exécution souhaitée.
3. En bas de la page, section **Artifacts**, télécharger `trend-reports-runXXX-...`.
4. Le zip contient les fichiers `reports/*.md` et `reports/virality_*.txt`.

---

## 5. Interprétation des rapports

| Fichier | Contenu |
|---------|---------|
| `reports/YYYY-MM-DD_HH-MM.md` | Rapport de synthèse : stats brutes, Top 10 Markdown |
| `reports/virality_YYYYMMDD_HHMMSS.txt` | Rapport complet du Virality Engine (Top 20, détail critères) |

---

## 6. Dépannage

### Le workflow échoue à l'étape "Validation des secrets requis"

`YOUTUBE_API_KEY` n'est pas configuré ou son nom est mal orthographié. Vérifier dans **Settings → Secrets and variables → Actions**.

### `run_virality.py` échoue avec "Fichier CSV introuvable"

En mode CSV uniquement (sans Supabase), `data/videos.csv` doit être créé par `test_agents.py` dans le même run. Si le fichier est absent, `test_agents.py` a probablement échoué lors de l'étape précédente — consulter les logs de l'étape 1.

### Les métriques temporelles du Virality Engine ne s'activent pas

Le Time Engine nécessite **≥ 2 snapshots** pour la même vidéo. Sans Supabase, chaque run repart d'un CSV vide. Configurer `SUPABASE_URL` et `SUPABASE_SERVICE_ROLE_KEY` pour accumuler l'historique entre les runs.
