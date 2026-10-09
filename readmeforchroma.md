# Base Chroma : sauvegarde et restauration

Base indexée le 08/10/2026 : 2 311 publications ANSD du catalogue, 283 837 fragments, collection `ansd_publications`.

## Versions à épingler

Ajoutez-les à `backend/requirements.txt`. Sinon, une autre machine installera des versions plus récentes, et une version différente de Chroma pourrait ne pas relire cette base.

```
chromadb==1.5.9
fastembed==0.9.0
pymupdf==1.28.2
langchain-text-splitters==1.1.3
```

Le modèle d'embedding (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, défini dans `config.py`) ne doit pas changer non plus : avec un autre modèle, les vecteurs de la base ne seraient plus comparables à ceux des questions.

## Sauvegarde

Archive : `ansd-corpus-2026-10-08.tar.gz` (1 322 Mo), contenant `storage/chroma/` et `data/manifest.csv`.

SHA-256 : `b9f2185f3c2c6d9974ac5d21e1d3313cdb891fc85b8d2954edbbec155ceb5c70`

Pour refaire une sauvegarde, depuis `backend` (arrêtez le backend avant, pour une copie cohérente) :

```bash
docker compose stop backend
tar -czf ../ansd-corpus-AAAA-MM-JJ.tar.gz storage/chroma data/manifest.csv
docker compose start backend
```

`Compress-Archive` de PowerShell 5.1 échoue au-delà de 2 Go : utilisez `tar`.
Ne mettez pas l'archive dans Git (2,6 Go, limite de 100 Mo par fichier sur GitHub), ni le `.env`.

## Restauration sur une autre machine

1. Clonez le dépôt du backend et vérifiez que `requirements.txt` épingle les versions ci-dessus.
2. Extrayez l'archive dans `backend` : `tar -xzf ansd-corpus-2026-10-08.tar.gz`
3. Recréez `backend/.env` avec une clé `OPENROUTER_API_KEY` valide.
4. Lancez le backend (`docker compose up --build -d`).
5. Vérifiez : `/api/health` doit afficher `"indexed_chunks": 283837`.
