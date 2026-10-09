# ANSD RAG

Systeme de questions-reponses base sur les publications publiques de l'ANSD
(Agence Nationale de la Statistique et de la Demographie du Senegal),
reproduisant en Python pur l'architecture RAG decrite dans le memoire de
Mouhamad KOUNTA (ESP/UCAD, 2024-2025).

Les PDF ne sont **jamais stockes sur disque** : ils sont recuperes en
memoire, decoupes, vectorises et indexes a la volee, puis les octets sont
abandonnes. Seuls le texte decoupe (dans Chroma) et les metadonnees
(`data/manifest.csv` : titre, URL, taille) persistent localement.

## Correspondance avec le memoire original

| Etape du memoire | Outil original | Ce projet |
|---|---|---|
| Extraction PDF | NiFi + PyMuPDF | fetch en memoire depuis ansd.sn + PyMuPDF |
| Stockage brut | Supabase | aucun (traitement a la volee) |
| Decoupage | RecursiveCharacterTextSplitter (LangChain) | identique |
| Embeddings | Ollama (nomic-embed-text) | fastembed, local, multilingue |
| Base vectorielle | Pinecone (cloud) | Chroma (local, persistant) |
| Generation | GPT-4.1-nano via OpenRouter | identique |
| Orchestration | n8n | scripts Python (`scraper.py` / `ingest.py` / `rag.py`) |
| Interface | Streamlit | identique |

Le site de l'ANSD reference environ **2500 documents PDF** au total. Pour
garder un temps d'indexation raisonnable, le catalogue se limite par defaut
aux 100 premiers documents (les plus recents) ; ajuste `--max-docs`,
`--category` ou `--type` pour elargir ou cibler le corpus.

Note technique : le certificat SSL de `ansd.sn` a une chaine incomplete
(`unable to get local issuer certificate`). Les navigateurs la completent
automatiquement, mais pas la librairie `requests` : la verification SSL est
donc desactivee dans `scraper.py`/`ingest.py` pour ce site public en lecture
seule.

## Installation

```bash
cd ansd-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Renseigne `OPENROUTER_API_KEY` dans `.env` avec une cle obtenue sur
https://openrouter.ai/keys.

## Utilisation

1. Cataloguer les publications ANSD (titre + URL uniquement, pas de telechargement) :

```bash
python scraper.py --max-docs 100
```

Options utiles :
- `--max-docs N` : nombre de documents a referencer (defaut 100)
- `--category {economie,demographie,societe,autres}` : filtrer par theme
- `--type {rapports,donnees,reference}` : filtrer par type de document
- `--delay 0.3` : delai entre deux pages du catalogue (politesse envers le serveur)

2. Indexer les documents (fetch en memoire, extraction, decoupage, embeddings) :

```bash
python ingest.py
```

Ajoute `--reset` pour repartir de zero si tu relances une indexation.

3. Interroger le systeme :

```bash
# en ligne de commande
python rag.py "Quel est le taux de scolarisation au Senegal ?"

# ou via l'interface web
streamlit run app.py
```

## Structure

```
ansd-rag/
  config.py       parametres centraux (chemins, chunking, modeles)
  scraper.py      catalogue les PDF depuis ansd.sn/toutes-les-publications (titre + URL)
  ingest.py       fetch en memoire + extraction texte + decoupage + embeddings + indexation Chroma
  rag.py          recuperation vectorielle + generation via OpenRouter
  app.py          interface chatbot Streamlit
  data/manifest.csv  catalogue des documents references (titre, url, taille)
  storage/chroma/ base vectorielle persistante (texte + embeddings, pas les PDF)
```

## Limites (heritees du memoire original)

- Pas de score de confiance sur les reponses generees.
- Les questions sans indication temporelle peuvent renvoyer une donnee
  perimee si plusieurs annees sont couvertes par le corpus.
- La qualite depend de l'extraction texte des PDF (mise en page complexe,
  tableaux scannes non-OCRises).
- Chaque relance d'`ingest.py` retelecharge en memoire les PDF non encore
  indexes (aucun cache disque) : un corpus large induit plus de trafic
  reseau a chaque ajout de documents.

## API REST (pour le frontend Next.js)

`api.py` expose le RAG en FastAPI pour le frontend `ANSD-RAG-main` :

```bash
uvicorn api:app --reload --port 8000   # docs : http://localhost:8000/docs
```

ou en Docker : `docker compose up --build`. Variable `CORS_ORIGINS` : URL(s)
du frontend autorisées. Guide complet de lancement : `../../LANCEMENT.md`.

## Tests

Tout se lance dans l'image Docker du backend (memes dependances qu'en production) :

```bash
./run_tests.sh               # unitaires + integration : ~120 tests, quelques secondes, sans reseau ni cout
./run_tests.sh unit          # regles (conversation, conseils, mises en forme, sources), cache, chiffres, analytique, voix
./run_tests.sh integration   # toutes les routes de l'API, modele de langage et recherche simules
./run_tests.sh e2e           # bout en bout contre la plateforme lancee (docker compose up -d, backend + frontend)
./run_tests.sh all           # tout
./run_tests.sh unit -k cache # options pytest en plus
```

- `tests/unit/` et `tests/integration/` n'appellent aucun service externe et n'ecrivent jamais
  dans `storage/analytics.sqlite3` (base temporaire).
- `tests/e2e/` interroge la vraie plateforme avec le vrai modele (~30 s, quelques centimes) : il
  verifie la forme des reponses (sources officielles, types de reponse, flux, pages du frontend),
  pas un texte exact. Variables : `E2E_API_URL`, `E2E_FRONTEND_URL`.

Tests du frontend : `npm test` dans `ANSD-RAG-main`.

## Montée en charge

Ce qui rend le backend capable de tenir une forte charge :

| Mécanisme | Effet |
|---|---|
| Cache des réponses (`cache.py`, Redis) | une question déjà posée (même langue, à la casse/accents près) est servie sans appeler le modèle |
| Regroupement des questions identiques | 500 questions identiques simultanées → 1 à 2 appels au modèle |
| Appels au modèle asynchrones (`httpx`, pool de connexions) | un processus gère des milliers de requêtes en attente ; nouvelles tentatives sur 429/5xx |
| Plusieurs processus (`WEB_CONCURRENCY`) | un par cœur CPU environ (~600 Mo de RAM chacun, modèle d'embedding inclus) |
| Limitation de débit | `RATE_LIMIT_PER_MINUTE` par utilisateur, `RATE_LIMIT_IP_PER_MINUTE` par IP |
| Journal d'utilisation en arrière-plan | écritures groupées, SQLite en WAL ; agrégats du tableau de bord calculés en SQL et mis en cache 30 s |

Mesures sur un MacBook Air (Docker, 1 instance, 4 processus) :

| Charge | Avant | Après |
|---|---|---|
| Question déjà posée (500 en parallèle) | ~11 req/s, 2 s | ~4 450 req/s, 56 ms (médiane) |
| `/api/sources` | 221 req/s | ~9 300 req/s |
| Question nouvelle × 500 simultanées | 500 appels au modèle | 2 appels au modèle |

Plusieurs instances derrière Nginx (Redis partagé) :

```bash
docker compose -f docker-compose.scale.yml up --build -d
docker compose -f docker-compose.scale.yml up -d --scale backend=6
```

Limites de cette configuration (à traiter en production à très grande échelle) :
les questions *nouvelles* restent bornées par les quotas du fournisseur du
modèle ; le journal SQLite et l'index Chroma sont locaux à une machine (passer
à PostgreSQL et à une base vectorielle partagée pour plusieurs serveurs).
