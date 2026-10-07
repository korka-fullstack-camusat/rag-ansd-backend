"""API REST consommee par le frontend Next.js (ANSD-RAG-main/lib/api.ts).

Les formes de reponse suivent exactement les types TypeScript du frontend
(QueryResponse, Citation, SourceDocument). Lancement :

    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import csv
import hashlib
import logging
import hmac
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Literal

import analytics
from cache import normalize, store
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from config import MANIFEST_PATH, TOP_K
from rag import (
    NO_DATA_MARKER,
    OPENROUTER_MODEL,
    acall_llm,
    aexplain,
    ashort_title,
    get_collection,
    get_embedder,
    retrieve,
)

logger = logging.getLogger("ansd-api")


class _HideHealthcheck(logging.Filter):
    """Le healthcheck Docker appelle /api/health toutes les 15 s : on ne le
    journalise pas pour garder des logs lisibles."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_HideHealthcheck())

GENERIC_ERROR_MESSAGE = "Un bug est survenu. Veuillez réessayer."
NO_INDEX_MESSAGE = (
    "Aucun document n'est indexé pour le moment. Lancez scraper.py puis ingest.py."
)
# Reponse affichee quand les publications indexees ne couvrent pas la question.
NO_DATA_MESSAGES = {
    "fr": "Nous n'avons pas encore de données sur cette demande.",
    "en": "We don't have data on this request yet.",
}

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Prechauffage : modele d'embedding et base vectorielle charges des le
    # demarrage, pas a la premiere question d'un utilisateur.
    await run_in_threadpool(lambda: list(get_embedder().embed(["warmup"])))
    await run_in_threadpool(get_collection)
    yield


app = FastAPI(title="ANSD RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ schemas

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    language: Literal["fr", "wo", "en", "ff", "srr", "dyo"] = "fr"
    # Comment la question a ete posee (statistiques d'usage uniquement).
    mode: Literal["text", "voice"] = "text"


class Citation(BaseModel):
    document_id: str
    document_title: str
    quote: str
    page_start: int | None
    page_end: int | None
    verified: bool


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class QueryResponse(BaseModel):
    question: str
    language: str
    answered: bool
    answer: str
    citations: list[Citation]
    sources_used: list[str]
    model: str
    usage: Usage


class SourceDocument(BaseModel):
    id: str
    title: str
    publisher: str
    publication_date: str
    filename: str
    description: str


# ------------------------------------------------------------------ helpers

def document_id(url: str) -> str:
    """Identifiant stable d'un document, derive de son URL ansd.sn."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def publication_date(url: str) -> str:
    # Les PDF ansd.sn sont ranges sous /sites/default/files/AAAA-MM/...
    match = re.search(r"/files/(\d{4}-\d{2})/", url)
    return match.group(1) if match else ""


def read_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def indexed_urls() -> set[str]:
    collection = get_collection()
    if collection.count() == 0:
        return set()
    return {m["url"] for m in collection.get(include=["metadatas"])["metadatas"] if m.get("url")}


_NUMBER_RE = re.compile(r"(?<![\w.])(?:p\.\s*|page\s+|source\s+)?\d+(?:[.,]\d+)*", re.IGNORECASE)


def _normalize_numbers(text: str) -> str:
    # "18 126 390" -> "18126390" (espaces, insecables et fines insecables)
    return re.sub(r"(?<=\d)[   ](?=\d)", "", text)


def _figures(text: str) -> list[str]:
    figures = [
        m.group(0)
        for m in _NUMBER_RE.finditer(_normalize_numbers(text))
        if not re.match(r"(p\.|page|source)", m.group(0), re.IGNORECASE)
    ]
    return [f for f in figures if len(f.replace(",", "").replace(".", "")) > 1]


def figures_found_in_context(answer: str, hits: list[dict], question: str = "") -> bool:
    """Vrai si chaque chiffre apporte par la reponse (hors numeros de
    page/source et hors nombres deja presents dans la question, ex. l'annee
    demandee) apparait mot pour mot dans les extraits recuperes. Une reponse
    sans chiffre propre (ex. « information non disponible ») n'est pas verifiee."""
    context = _normalize_numbers(" ".join(h["text"] for h in hits))
    from_question = set(_figures(question))
    figures = [f for f in _figures(answer) if f not in from_question]
    return bool(figures) and all(f in context for f in figures)


# ------------------------------------------------------------------ routes

# ------------------------------------------------------------------ montee en charge

RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))
# Par adresse IP : volontairement large — au Senegal, de nombreux utilisateurs
# partagent la meme IP publique (operateurs mobiles, universites, entreprises).
RATE_LIMIT_IP_PER_MINUTE = int(os.environ.get("RATE_LIMIT_IP_PER_MINUTE", "600"))
RATE_LIMIT_MESSAGE = "Trop de questions en peu de temps. Patientez une minute puis réessayez."

_corpus = {"version": 0, "checked": 0.0}
_sources_cache: dict = {"version": None, "items": []}
_retrieval_cache: OrderedDict[str, list[dict]] = OrderedDict()


def corpus_version() -> int:
    """Nombre de fragments indexes, relu au plus une fois par minute : sert de
    version du corpus dans les cles de cache (une reindexation invalide tout)."""
    now = time.time()
    if now - _corpus["checked"] > 60:
        _corpus["version"] = get_collection().count()
        _corpus["checked"] = now
    return _corpus["version"]


async def aretrieve(question: str) -> list[dict]:
    """Recherche vectorielle hors de la boucle d'evenements, avec un petit
    cache par processus (la meme question sert a la reponse puis au « Voir plus »)."""
    key = f"{corpus_version()}:{normalize(question)}"
    if key in _retrieval_cache:
        _retrieval_cache.move_to_end(key)
        return _retrieval_cache[key]
    hits = await run_in_threadpool(retrieve, question, TOP_K)
    _retrieval_cache[key] = hits
    while len(_retrieval_cache) > 2000:
        _retrieval_cache.popitem(last=False)
    return hits


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def enforce_rate_limit(request: Request, client_id: str | None, per_client: int = RATE_LIMIT_PER_MINUTE) -> None:
    if not await store.allow(f"ip:{client_ip(request)}", RATE_LIMIT_IP_PER_MINUTE) or (
        client_id and not await store.allow(f"client:{client_id}", per_client)
    ):
        raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE, headers={"Retry-After": "60"})


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "indexed_chunks": corpus_version(), "cache": store.backend}


@app.get("/api/sources", response_model=list[SourceDocument])
async def sources() -> list[SourceDocument]:
    """Liste des publications indexees — recalculee seulement si le corpus change."""
    version = corpus_version()
    if _sources_cache["version"] != version:
        _sources_cache["items"] = await run_in_threadpool(_build_sources)
        _sources_cache["version"] = version
    return _sources_cache["items"]


def _build_sources() -> list[SourceDocument]:
    rows = read_manifest()
    indexed = indexed_urls()
    if indexed:
        rows = [r for r in rows if r["url"] in indexed]
    return [
        SourceDocument(
            id=document_id(r["url"]),
            title=r["title"],
            publisher="ANSD",
            publication_date=publication_date(r["url"]),
            filename=document_id(r["url"]),
            description=" · ".join(p for p in (r.get("extension", "").upper(), r.get("size", "")) if p),
        )
        for r in rows
    ]


@app.post("/api/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    request: Request,
    x_client_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
) -> QueryResponse:
    """Repond a la question (depuis le cache si elle a deja ete posee) et
    journalise l'utilisation (tableau de bord admin)."""
    started = time.perf_counter()
    question = req.question.strip()
    response: QueryResponse | None = None
    cached = False
    try:
        await enforce_rate_limit(request, x_client_id)
    except HTTPException:
        raise  # trop de requetes : non journalise comme une question
    try:
        key = f"answer:v{corpus_version()}:{req.language}:{normalize(question)}"
        data, cached = await store.cached(key, lambda: _answer(question, req.language))
        response = QueryResponse(**{**data, "question": question})
        return response
    finally:
        analytics.log_event(
            "query",
            client_id=x_client_id,
            session_id=x_session_id,
            question=question,
            language=req.language,
            mode=req.mode,
            answered=response.answered if response else None,
            error=response is None,
            cached=cached,
            latency_ms=round((time.perf_counter() - started) * 1000),
            # Une reponse servie depuis le cache n'a rien consomme.
            prompt_tokens=0 if cached else (response.usage.prompt_tokens if response else None),
            completion_tokens=0 if cached else (response.usage.completion_tokens if response else None),
            sources=list(dict.fromkeys(c.document_title for c in response.citations)) if response else None,
        )


async def _answer(question: str, language: str) -> dict:
    try:
        hits = await aretrieve(question)
    except Exception:
        logger.exception("retrieval failed")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MESSAGE)

    no_data = QueryResponse(
        question=question, language=language, answered=False,
        answer=NO_DATA_MESSAGES.get(language, NO_DATA_MESSAGES["fr"]),
        citations=[], sources_used=[], model=OPENROUTER_MODEL, usage=Usage(),
    ).model_dump()
    if not hits:
        return no_data

    try:
        completion = await acall_llm(question, hits, language)
        answer = completion["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=GENERIC_ERROR_MESSAGE)

    if NO_DATA_MARKER in answer.upper():
        return no_data

    verified = figures_found_in_context(answer, hits, question)
    citations = [
        Citation(
            document_id=document_id(h["url"]) if h.get("url") else h["source"],
            document_title=h["source"],
            quote=h["text"][:400],
            page_start=h["page"],
            page_end=h["page"],
            verified=verified,
        )
        for h in hits
    ]
    usage = completion.get("usage") or {}
    return QueryResponse(
        question=question,
        language=language,
        answered=True,
        answer=answer,
        citations=citations,
        sources_used=list(dict.fromkeys(c.document_id for c in citations)),
        model=completion.get("model", OPENROUTER_MODEL),
        usage=Usage(**{k: usage.get(k, 0) for k in Usage.model_fields}),
    ).model_dump()


class ExplainRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=8000)
    language: Literal["fr", "wo", "en", "ff", "srr", "dyo"] = "fr"


class ExplainResponse(BaseModel):
    details: str


@app.post("/api/explain", response_model=ExplainResponse)
async def explain_answer(
    req: ExplainRequest,
    request: Request,
    x_client_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
) -> ExplainResponse:
    """Explication detaillee d'une reponse deja donnee (bouton « Voir plus »),
    preparee en arriere-plan par le frontend et mise en cache."""
    question = req.question.strip()
    await enforce_rate_limit(request, x_client_id)
    ok = False
    try:
        key = f"explain:v{corpus_version()}:{req.language}:{normalize(question)}"
        data, _ = await store.cached(key, lambda: _explain(question, req.answer.strip(), req.language))
        ok = True
        return ExplainResponse(**data)
    finally:
        analytics.log_event(
            "explain", client_id=x_client_id, session_id=x_session_id,
            question=question, language=req.language, error=not ok,
        )


async def _explain(question: str, answer: str, language: str) -> dict:
    try:
        hits = await aretrieve(question)
        if not hits:
            raise HTTPException(status_code=404, detail=NO_INDEX_MESSAGE)
        return {"details": await aexplain(question, answer, hits, language)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("explain failed")
        raise HTTPException(status_code=502, detail=GENERIC_ERROR_MESSAGE)


class TitleRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class TitleResponse(BaseModel):
    title: str


@app.post("/api/title", response_model=TitleResponse)
async def title(req: TitleRequest, request: Request, x_session_id: str | None = Header(default=None)) -> TitleResponse:
    """Titre court (1 a 3 mots) d'une discussion, pour l'historique du frontend
    (et les « sujets les plus demandes » du tableau de bord)."""
    await enforce_rate_limit(request, None)
    question = req.question.strip()
    try:
        data, _ = await store.cached(
            f"title:{normalize(question)}",
            lambda: _title(question),
            ttl=7 * 24 * 3600,
        )
    except Exception:
        logger.exception("title generation failed")
        raise HTTPException(status_code=502, detail=GENERIC_ERROR_MESSAGE)
    await run_in_threadpool(analytics.log_session_title, x_session_id, data["title"])
    return TitleResponse(**data)


async def _title(question: str) -> dict:
    return {"title": await ashort_title(question)}


@app.get("/files/{doc_id}")
def publication_file(doc_id: str) -> RedirectResponse:
    """Les PDF ne sont jamais stockes localement : on redirige vers ansd.sn."""
    for row in read_manifest():
        if document_id(row["url"]) == doc_id:
            return RedirectResponse(row["url"], status_code=307)
    raise HTTPException(status_code=404, detail="Publication introuvable.")


class TrackRequest(BaseModel):
    event: Literal["details_open", "listen"]


@app.post("/api/track", status_code=204)
async def track(
    req: TrackRequest,
    x_client_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
) -> None:
    """Interactions de l'utilisateur (« Voir plus » deplie, « Écouter ») pour
    le tableau de bord admin."""
    analytics.log_event(req.event, client_id=x_client_id, session_id=x_session_id)


# ------------------------------------------------------------------ admin

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def require_admin(authorization: str | None = Header(default=None)) -> None:
    """Acces au tableau de bord : `Authorization: Bearer <ADMIN_TOKEN>`."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Tableau de bord désactivé : définissez ADMIN_TOKEN dans le .env du backend.",
        )
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token.encode(), ADMIN_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Mot de passe administrateur incorrect.")


@app.get("/api/admin/stats", dependencies=[Depends(require_admin)])
def admin_stats(days: int = 30) -> dict:
    """Indicateurs d'utilisation sur les `days` derniers jours (0 = tout)."""
    return analytics.stats(days if days > 0 else None)


@app.get("/api/admin/questions", dependencies=[Depends(require_admin)])
def admin_questions(
    days: int = 30,
    status: Literal["all", "answered", "no_data", "error"] = "all",
    limit: int = 25,
    offset: int = 0,
) -> dict:
    """Journal des questions, le plus recent en premier."""
    return analytics.recent_questions(
        days if days > 0 else None,
        limit=max(1, min(limit, 200)),
        offset=max(0, offset),
        status=None if status == "all" else status,
    )


VOICE_UNAVAILABLE = "Le mode vocal n'est pas encore disponible sur ce serveur."


@app.post("/api/voice/transcribe")
def voice_transcribe() -> None:
    raise HTTPException(status_code=501, detail=VOICE_UNAVAILABLE)


@app.post("/api/voice/speak")
def voice_speak() -> None:
    raise HTTPException(status_code=501, detail=VOICE_UNAVAILABLE)
