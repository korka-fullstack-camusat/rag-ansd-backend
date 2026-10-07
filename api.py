"""API REST consommee par le frontend Next.js (ANSD-RAG-main/lib/api.ts).

Les formes de reponse suivent exactement les types TypeScript du frontend
(QueryResponse, Citation, SourceDocument). Lancement :

    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import csv
import hashlib
import logging
import os
import re
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from config import MANIFEST_PATH, TOP_K
from rag import OPENROUTER_MODEL, call_llm, get_collection, retrieve

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

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]

app = FastAPI(title="ANSD RAG API")
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


def figures_found_in_context(answer: str, hits: list[dict]) -> bool:
    """Vrai si chaque chiffre de la reponse (hors numeros de page/source)
    apparait mot pour mot dans les extraits recuperes."""
    context = _normalize_numbers(" ".join(h["text"] for h in hits))
    figures = [
        m.group(0)
        for m in _NUMBER_RE.finditer(_normalize_numbers(answer))
        if not re.match(r"(p\.|page|source)", m.group(0), re.IGNORECASE)
    ]
    figures = [f for f in figures if len(f.replace(",", "").replace(".", "")) > 1]
    return bool(figures) and all(f in context for f in figures)


# ------------------------------------------------------------------ routes

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": get_collection().count()}


@app.get("/api/sources", response_model=list[SourceDocument])
def sources() -> list[SourceDocument]:
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
def query(req: QueryRequest) -> QueryResponse:
    question = req.question.strip()
    try:
        hits = retrieve(question, top_k=TOP_K)
    except Exception:
        logger.exception("retrieval failed")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MESSAGE)

    if not hits:
        return QueryResponse(
            question=question, language=req.language, answered=False, answer=NO_INDEX_MESSAGE,
            citations=[], sources_used=[], model=OPENROUTER_MODEL, usage=Usage(),
        )

    try:
        completion = call_llm(question, hits, req.language)
        answer = completion["choices"][0]["message"]["content"]
    except Exception:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=GENERIC_ERROR_MESSAGE)

    verified = figures_found_in_context(answer, hits)
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
        language=req.language,
        answered=True,
        answer=answer,
        citations=citations,
        sources_used=list(dict.fromkeys(c.document_id for c in citations)),
        model=completion.get("model", OPENROUTER_MODEL),
        usage=Usage(**{k: usage.get(k, 0) for k in Usage.model_fields}),
    )


@app.get("/files/{doc_id}")
def publication_file(doc_id: str) -> RedirectResponse:
    """Les PDF ne sont jamais stockes localement : on redirige vers ansd.sn."""
    for row in read_manifest():
        if document_id(row["url"]) == doc_id:
            return RedirectResponse(row["url"], status_code=307)
    raise HTTPException(status_code=404, detail="Publication introuvable.")


VOICE_UNAVAILABLE = "Le mode vocal n'est pas encore disponible sur ce serveur."


@app.post("/api/voice/transcribe")
def voice_transcribe() -> None:
    raise HTTPException(status_code=501, detail=VOICE_UNAVAILABLE)


@app.post("/api/voice/speak")
def voice_speak() -> None:
    raise HTTPException(status_code=501, detail=VOICE_UNAVAILABLE)
