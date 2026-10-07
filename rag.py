import asyncio
import os
import random

import chromadb
import httpx
import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4.1-nano")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Marqueur renvoye par le modele quand les extraits ne permettent pas de
# repondre : l'API le remplace par un message simple (« pas encore de donnees »).
NO_DATA_MARKER = "INDISPONIBLE"

SYSTEM_PROMPT = (
    "Tu es un assistant qui repond a des questions en te basant uniquement sur "
    "les extraits de rapports de l'ANSD (Agence Nationale de la Statistique et de la "
    "Demographie du Senegal) fournis en contexte. Si le contexte ne contient pas "
    f"l'information demandee, reponds uniquement par le mot {NO_DATA_MARKER}, sans rien "
    "ajouter : n'invente jamais de reponse. "
    "Sois bref et precis : donne directement le chiffre ou le fait demande (avec "
    "son unite et sa periode) en 1 a 3 phrases, sans introduction, sans repeter la "
    "question et sans conclusion. N'ecris pas de references aux sources dans le "
    "texte (elles sont affichees a part) et n'utilise pas de mise en forme Markdown."
)

EXPLAIN_PROMPT = (
    "Tu es un assistant qui explique des statistiques de l'ANSD (Agence Nationale "
    "de la Statistique et de la Demographie du Senegal) en te basant uniquement sur "
    "les extraits fournis en contexte. On te donne une question et la reponse courte "
    "deja donnee : developpe-la pour qui veut en savoir plus — detail des chiffres "
    "(composantes, evolutions, comparaisons), definitions utiles, periode et "
    "publication concernees. Reste factuel et concis : 2 a 4 courts paragraphes, ou "
    "une liste a puces (lignes commencant par \"- \") si c'est plus clair. Tu peux "
    "mettre en **gras** les chiffres cles. N'invente rien : si le contexte n'apporte "
    "rien de plus, dis-le en une phrase. N'ecris pas de references aux sources dans le texte."
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


LANGUAGE_NAMES = {
    "fr": "francais",
    "en": "anglais",
    "wo": "wolof",
    "ff": "pulaar (peul du Senegal)",
    "srr": "serere (seereer)",
    "dyo": "diola (joola-fonyi)",
}


def build_prompt(question: str, hits: list[dict], language: str = "fr") -> str:
    context = "\n\n".join(
        f"[Source {i+1} - {h['source']}, p.{h['page']}]\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return (
        f"Contexte extrait des rapports ANSD :\n\n{context}\n\n"
        f"Question : {question}\n\n"
        f"Reponds en {LANGUAGE_NAMES.get(language, 'francais')}, de maniere breve et precise, "
        "en t'appuyant uniquement sur le contexte ci-dessus."
    )


def _check_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY manquant. Copie .env.example vers .env et renseigne ta cle."
        )


def _chat(messages: list[dict], timeout: int = 60, **options) -> dict:
    """Appel brut (synchrone) a OpenRouter — scripts CLI et Streamlit."""
    _check_key()
    payload = {"model": OPENROUTER_MODEL, "messages": messages, **options}
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# --- Appels asynchrones (API) : un seul pool de connexions HTTP par processus,
# reutilise entre requetes, et nouvelles tentatives quand le fournisseur sature.
LLM_MAX_CONNECTIONS = int(os.environ.get("LLM_MAX_CONNECTIONS", "200"))
LLM_RETRIES = int(os.environ.get("LLM_RETRIES", "3"))
_async_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60, connect=10),
            limits=httpx.Limits(max_connections=LLM_MAX_CONNECTIONS, max_keepalive_connections=LLM_MAX_CONNECTIONS),
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        )
    return _async_client


async def _achat(messages: list[dict], timeout: float = 60, **options) -> dict:
    _check_key()
    payload = {"model": OPENROUTER_MODEL, "messages": messages, **options}
    for attempt in range(LLM_RETRIES + 1):
        try:
            resp = await _client().post(OPENROUTER_URL, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = isinstance(exc, httpx.TransportError) or (
                exc.response.status_code == 429 or exc.response.status_code >= 500
            )
            if not retryable or attempt == LLM_RETRIES:
                raise
            # Attente exponentielle avec gigue (ou Retry-After si le fournisseur l'indique).
            retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.5 * 2**attempt
            await asyncio.sleep(min(delay, 8) + random.random() * 0.3)
    raise RuntimeError("unreachable")


def _answer_messages(question: str, hits: list[dict], language: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits, language)},
    ]


def call_llm(question: str, hits: list[dict], language: str = "fr") -> dict:
    return _chat(_answer_messages(question, hits, language), temperature=0.2, max_tokens=300)


async def acall_llm(question: str, hits: list[dict], language: str = "fr") -> dict:
    return await _achat(_answer_messages(question, hits, language), temperature=0.2, max_tokens=300)


def _explain_messages(question: str, answer: str, hits: list[dict], language: str) -> list[dict]:
    prompt = (
        build_prompt(question, hits, language).rsplit("Reponds en", 1)[0]
        + f"Reponse courte deja donnee : {answer}\n\n"
        + f"Developpe cette reponse en {LANGUAGE_NAMES.get(language, 'francais')}, "
        "en t'appuyant uniquement sur le contexte ci-dessus."
    )
    return [
        {"role": "system", "content": EXPLAIN_PROMPT},
        {"role": "user", "content": prompt},
    ]


def explain(question: str, answer: str, hits: list[dict], language: str = "fr") -> str:
    """Explication detaillee d'une reponse courte (bouton « Voir plus »)."""
    completion = _chat(_explain_messages(question, answer, hits, language), temperature=0.2)
    return completion["choices"][0]["message"]["content"].strip()


async def aexplain(question: str, answer: str, hits: list[dict], language: str = "fr") -> str:
    completion = await _achat(_explain_messages(question, answer, hits, language), temperature=0.2, max_tokens=700)
    return completion["choices"][0]["message"]["content"].strip()


TITLE_PROMPT = (
    "Tu nommes des discussions dans un historique. Donne le sujet de la question "
    "en 1 a 3 mots, en francais, comme un titre court (exemples : \"Population\", "
    "\"Croissance du PIB\", \"Espérance de vie\", \"Chômage des jeunes\"). "
    "Reponds uniquement par le titre, sans guillemets ni ponctuation finale."
)


def _title_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": TITLE_PROMPT}, {"role": "user", "content": question}]


def _clean_title(completion: dict) -> str:
    title = completion["choices"][0]["message"]["content"].strip().strip("\"'«»“”.!?:;").strip()
    words = title.split()
    if not words:
        raise ValueError("titre vide")
    title = " ".join(words[:4])
    return title[:1].upper() + title[1:40]


def short_title(question: str) -> str:
    """Titre de 1 a 3 mots resumant le sujet d'une question (historique des discussions)."""
    return _clean_title(_chat(_title_messages(question), timeout=15, temperature=0, max_tokens=12))


async def ashort_title(question: str) -> str:
    return _clean_title(await _achat(_title_messages(question), timeout=15, temperature=0, max_tokens=12))


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
