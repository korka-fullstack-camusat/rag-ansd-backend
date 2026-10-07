import os

import chromadb
import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4.1-nano")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "Tu es un assistant qui repond a des questions en te basant uniquement sur "
    "les extraits de rapports de l'ANSD (Agence Nationale de la Statistique et de la "
    "Demographie du Senegal) fournis en contexte. Si le contexte ne contient pas "
    "l'information demandee, dis-le clairement au lieu d'inventer une reponse. "
    "Cite systematiquement le document et la page source de chaque affirmation."
)

_embedder = None
_collection = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = [vec.tolist() for vec in get_embedder().embed([question])]
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    hits = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "url": meta.get("url"),
            "score": 1 - distance,
        })
    return hits


LANGUAGE_NAMES = {"fr": "francais", "en": "anglais", "wo": "wolof"}


def build_prompt(question: str, hits: list[dict], language: str = "fr") -> str:
    context = "\n\n".join(
        f"[Source {i+1} - {h['source']}, p.{h['page']}]\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return (
        f"Contexte extrait des rapports ANSD :\n\n{context}\n\n"
        f"Question : {question}\n\n"
        f"Reponds en {LANGUAGE_NAMES.get(language, 'francais')}, de maniere claire et structuree, "
        "en t'appuyant uniquement sur le contexte ci-dessus."
    )


def call_llm(question: str, hits: list[dict], language: str = "fr") -> dict:
    """Appel brut a OpenRouter : renvoie le JSON complet (contenu, modele, usage)."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY manquant. Copie .env.example vers .env et renseigne ta cle."
        )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits, language)},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def generate_answer(question: str, hits: list[dict]) -> str:
    return call_llm(question, hits)["choices"][0]["message"]["content"]


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    hits = retrieve(question, top_k=top_k)
    if not hits:
        return {
            "answer": "Aucun document n'est indexe pour le moment. Lance scraper.py puis ingest.py.",
            "sources": [],
        }

    answer = generate_answer(question, hits)
    return {"answer": answer, "sources": hits}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Quel est le taux de scolarisation au Senegal ?"
    result = answer_question(q)
    print(f"\nQuestion : {q}\n")
    print(result["answer"])
    print("\nSources :")
    for h in result["sources"]:
        print(f"  - {h['source']} (p.{h['page']}, score={h['score']:.2f})")
