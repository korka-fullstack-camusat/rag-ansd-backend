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
