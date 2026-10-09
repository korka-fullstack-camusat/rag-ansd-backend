"""API REST consommee par le frontend Next.js (ANSD-RAG-main/lib/api.ts).

Les formes de reponse suivent exactement les types TypeScript du frontend
(QueryResponse, Citation, SourceDocument). Lancement :

    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import csv
import hashlib
import logging
import hmac
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Literal

import analytics
from cache import normalize, store
import voice
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field

from config import MANIFEST_PATH, TOP_K
from rag import (
    NO_DATA_MARKER,
    OPENROUTER_MODEL,
    acall_llm,
    aexplain,
    ashort_title,
    get_collection,
    corpus_index,
    aguidance,
    corpus_reply,
    acondense,
    get_embedder,
    retrieve,
    format_only_clause,
    period_only,
    with_period,
    guidance_question,
    small_talk_reply,
    split_used_sources,
    with_format,
    unknown_acronyms,
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
    # Index du corpus (titres, URL, sigles) : environ 2 minutes la premiere fois,
    # puis lu depuis storage/corpus_cache.json. Lance en arriere-plan pour ne pas
    # retarder le demarrage ; les requetes qui en dependent attendent la fin.
    threading.Thread(target=corpus_index, daemon=True).start()
    yield


app = FastAPI(title="ANSD RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ schemas

class SourceRef(BaseModel):
    title: str = Field(max_length=300)
    page: int | None = None


class HistoryTurn(BaseModel):
    """Echange precedent de la discussion (question de suite : « et ces chiffres ? »)."""
    question: str = Field(max_length=2000)
    answer: str = Field(max_length=4000)
    sources: list[SourceRef] = Field(default_factory=list, max_length=6)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # « Relancer » : nouvelle reponse, sans resservir celle en cache.
    regenerate: bool = False
    # Derniers echanges avec leurs sources (seuls les deux derniers sont utilises).
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)
    language: Literal["fr", "wo", "en", "ff", "srr", "dyo"] = "fr"
    # Comment la question a ete posee (statistiques d'usage uniquement).
    mode: Literal["text", "voice"] = "text"


class Citation(BaseModel):
    document_id: str
    document_title: str
    # Adresse officielle du PDF sur ansd.sn (lien partageable, ex. dans une copie).
    url: str | None = None
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
    # answer : reponse tiree des publications ; no_data : rien dans le corpus ;
    # chat : conversation courante (« bonjour », « merci »…), sans recherche.
    kind: Literal["answer", "no_data", "chat"] = "answer"
    # Question reformulee de maniere autonome (questions de suite), utilisee pour
    # la recherche ; reprise par « Voir plus ».
    standalone_question: str | None = None
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
    """URL des documents indexes, relues seulement si le nombre de fragments change."""
    return corpus_index()["urls"]


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


async def aretrieve(
    question: str, prefer: list[tuple[str, int]] | None = None, top_k: int = TOP_K
) -> list[dict]:
    """Recherche vectorielle hors de la boucle d'evenements, avec un petit
    cache par processus (la meme question sert a la reponse puis au « Voir plus »)."""
    key = f"{corpus_version()}:{normalize(question)}:{sorted(set(prefer or []))}:{top_k}"
    if key in _retrieval_cache:
        _retrieval_cache.move_to_end(key)
        return _retrieval_cache[key]
    hits = await run_in_threadpool(retrieve, question, top_k, prefer)
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


async def _to_french(text: str, timeout: float = 25) -> str:
    """Wolof -> francais ; le texte tel quel s'il n'est pas du wolof (le service traduirait
    alors dans l'autre sens) ou si la traduction echoue (la recherche sera moins bonne, mais
    l'utilisateur obtient une reponse)."""
    if not voice.looks_wolof(text):
        return text
    try:
        return await voice.translate(text, "wo", "fr", timeout=timeout)
    except voice.VoiceError:
        return text


@app.post("/api/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    request: Request,
    x_client_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
) -> QueryResponse:
    """Point d'entree. Anglais et francais : traites directement. Wolof : la question et
    l'historique sont traduits en francais (Soynade), traites comme une question francaise
    (meme recherche, meme cache), puis la reponse est traduite en wolof."""
    if req.language != "wo":
        return await _query(req, request, x_client_id, x_session_id)
    chat = small_talk_reply(req.question, "wo")  # « Nanga def ? », « Jërëjëf »… sont reconnus tels quels
    if chat:
        await enforce_rate_limit(request, x_client_id)
        analytics.log_event("chat", client_id=x_client_id, session_id=x_session_id, question=req.question, language="wo")
        # Reponse ecrite en wolof si elle existe ; sinon le francais, traduit.
        if chat == small_talk_reply(req.question, "fr"):
            chat = await voice.to_wolof(chat)
        return QueryResponse(
            question=req.question, language="wo", kind="chat", answered=False, answer=chat,
            citations=[], sources_used=[], model="", usage=Usage(),
        )
    question_fr, *history_fr = await asyncio.gather(
        _to_french(req.question),
        *[_to_french(t.answer, timeout=10) for t in req.history[-2:]],
        *[_to_french(t.question, timeout=10) for t in req.history[-2:]],
    )
    turns = req.history[-2:]
    history = [
        HistoryTurn(question=history_fr[len(turns) + i], answer=history_fr[i], sources=t.sources)
        for i, t in enumerate(turns)
    ]
    fr_req = req.model_copy(update={"language": "fr", "question": question_fr, "history": history})
    resp = await _query(fr_req, request, x_client_id, x_session_id)
    answer = await voice.to_wolof(resp.answer) if resp.answer else resp.answer
    return resp.model_copy(update={"language": "wo", "question": req.question, "answer": answer})


async def _query(
    req: QueryRequest,
    request: Request,
    x_client_id: str | None,
    x_session_id: str | None,
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
    chat = small_talk_reply(question, req.language)
    if chat:
        analytics.log_event("chat", client_id=x_client_id, session_id=x_session_id, question=question, language=req.language)
        return QueryResponse(
            question=question, language=req.language, kind="chat", answered=False, answer=chat,
            citations=[], sources_used=[], model="", usage=Usage(),
        )
    try:
        about_corpus = await run_in_threadpool(corpus_reply, question, req.language)
    except Exception:
        logger.exception("corpus reply failed")
        about_corpus = None
    if about_corpus:
        analytics.log_event("chat", client_id=x_client_id, session_id=x_session_id, question=question, language=req.language)
        return QueryResponse(
            question=question, language=req.language, kind="chat", answered=False, answer=about_corpus,
            citations=[], sources_used=[], model="", usage=Usage(),
        )
    if guidance_question(question):
        # Demande de conseil (« que me conseillez-vous pour recuperer les donnees ? ») :
        # reponse de guide, sans recherche documentaire ni chiffre.
        history = [t.model_dump() for t in req.history if t.answer.strip()][-1:]
        try:
            completion = await aguidance(question, req.language, history)
            advice = completion["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.exception("guidance failed")
            advice = ""
        if advice:
            analytics.log_event("chat", client_id=x_client_id, session_id=x_session_id, question=question, language=req.language)
            return QueryResponse(
                question=question, language=req.language, kind="chat", answered=False, answer=advice,
                citations=[], sources_used=[], model=completion.get("model", OPENROUTER_MODEL), usage=Usage(),
            )
    try:
        history = [t.model_dump() for t in req.history if t.answer.strip()][-2:]
        standalone = question
        clause = format_only_clause(question) if history else None
        periods = period_only(question) if history and not clause else None
        if clause:
            # Simple demande de mise en forme : meme sujet que la question precedente.
            standalone = with_format(history[-1]["question"], clause)
        elif periods:
            # Simple relance de periode (« et pour 2025 ? ») : meme sujet, autre periode —
            # meme si la question precedente n'a pas eu de reponse.
            standalone = with_period(history[-1]["question"], periods)
        elif history:
            try:
                standalone = await acondense(question, history)
            except Exception:
                logger.warning("reformulation impossible, question telle quelle", exc_info=True)
        # Les sources et la reponse precedentes ne servent QUE si la question est une suite
        # (mise en forme, autre periode, ou reformulee a partir de l'historique). Une question
        # sur un autre sujet repart de zero : sinon les pages du tour precedent sont forcees
        # dans les resultats et affichees comme sources de la nouvelle reponse.
        follow_up = bool(clause or periods) or normalize(standalone) != normalize(question)
        prefer = (
            [(src["title"], src["page"]) for t in history[-1:] for src in t["sources"] if src["page"] is not None]
            if follow_up else []
        )
        previous = history[-1]["answer"] if history and follow_up else None
        # La reponse precedente fait partie de la cle : « mets ca en tableau » apres deux
        # reponses differentes ne doit pas resservir la meme reponse en cache.
        prev_key = hashlib.sha1(previous.encode()).hexdigest()[:12] if previous else "-"
        key = f"answer:v17:{corpus_version()}:{req.language}:{normalize(standalone)}:{sorted(set(prefer))}:{prev_key}"
        data, cached = await store.cached(
            key,
            lambda: _answer(standalone, req.language, prefer, previous, temperature=0.6 if req.regenerate else 0.2),
            refresh=req.regenerate,
        )
        response = QueryResponse(
            **{**data, "question": question, "standalone_question": standalone if standalone != question else None}
        )
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


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
YEAR_CANDIDATES_FACTOR = 8  # candidats recuperes avant filtre par annee = TOP_K x 8


def filter_by_years(question: str, hits: list[dict]) -> list[dict]:
    """Question qui cite une ou des annees : seuls les extraits qui les mentionnent
    (texte ou titre du document) sont donnes au modele. Evite qu'il lise un chiffre
    d'une autre annee dans un tableau (colonnes d'annees perdues a l'extraction du
    PDF). Aucun extrait pour ces annees => liste vide => « pas de donnees »."""
    years = set(YEAR_RE.findall(question))
    if not years:
        return hits
    return [h for h in hits if any(y in h["text"] or y in h["source"] for y in years)]


_EMPTY_ANSWER_RE = re.compile(
    r"(n'est|ne sont|n'a|n'ont|ne figure|ne figurent|ne contient|ne contiennent)\s+(pas|aucun)"
    r"[^.]{0,60}(indiqu|mentionn|pr[ée]cis|disponible|fourni|pr[ée]sent|donn)",
    re.IGNORECASE,
)


def _is_empty_answer(answer: str, question: str) -> bool:
    """Reponse du type « l'information n'est pas explicitement indiquee », sans
    aucun chiffre propre : c'est une absence de donnees, affichee comme telle."""
    return bool(_EMPTY_ANSWER_RE.search(answer)) and not [
        f for f in _figures(answer) if f not in set(_figures(question))
    ]


_YEAR_ONLY_RE = re.compile(r"(?:19|20)\d{2}")


def pick_sources(answer: str, hits: list[dict], used: list[int]) -> list[int]:
    """Indices des extraits a afficher comme sources de la reponse.

    Une source n'est affichee que si son texte contient reellement un chiffre de la
    reponse : les annees sont ignorees (« 2010 » apparait dans n'importe quel
    document de la serie) et le chiffre doit etre un nombre entier dans l'extrait
    (« 12 » ne correspond pas a « 2012 »). Les extraits cites par le modele passent
    en premier. Reponse sans chiffre propre ou rien de verifiable : on s'en tient a
    ce que le modele a indique, a defaut a l'extrait le plus pertinent."""
    figures = [f for f in _figures(answer) if not _YEAR_ONLY_RE.fullmatch(f)]
    if not figures:
        return used or [0]
    patterns = [re.compile(rf"(?<![\d,.]){re.escape(f)}(?!\d)") for f in figures]
    texts = [_normalize_numbers(h["text"]) for h in hits]
    supporting = [i for i, t in enumerate(texts) if any(p.search(t) for p in patterns)]
    if not supporting:
        return used or [0]
    ordered = [i for i in used if i in supporting] + [i for i in supporting if i not in used]
    return ordered[:3]


async def _answer(
    question: str,
    language: str,
    prefer: list[tuple[str, int]] | None = None,
    previous: str | None = None,
    temperature: float = 0.2,
) -> dict:
    no_data = QueryResponse(
        question=question, language=language, answered=False, kind="no_data",
        answer=NO_DATA_MESSAGES.get(language, NO_DATA_MESSAGES["fr"]),
        citations=[], sources_used=[], model=OPENROUTER_MODEL, usage=Usage(),
    ).model_dump()
    # Sigle inconnu de tout le corpus (ex. « DR ») : rien a chercher, et surtout
    # rien a laisser inventer au modele.
    if await run_in_threadpool(unknown_acronyms, question):
        return no_data
    try:
        # Question datee : on cherche plus large avant de filtrer par annee, sinon les
        # 6 meilleurs extraits (souvent d'autres annees d'une meme serie) ne laissent rien.
        wide = TOP_K * YEAR_CANDIDATES_FACTOR if YEAR_RE.search(question) else TOP_K
        hits = filter_by_years(question, await aretrieve(question, prefer, wide))[:TOP_K]
    except Exception:
        logger.exception("retrieval failed")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MESSAGE)

    if not hits:
        return no_data

    try:
        completion = await acall_llm(question, hits, language, previous, temperature)
        raw = completion["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=GENERIC_ERROR_MESSAGE)

    answer, used = split_used_sources(raw, len(hits))
    if not answer or answer.strip(" .").upper() == NO_DATA_MARKER or _is_empty_answer(answer, question):
        return no_data

    verified = figures_found_in_context(answer, hits, question)
    # Sources affichees : seulement les extraits reellement utilises pour repondre.
    cited = [hits[i] for i in pick_sources(answer, hits, used)]
    cited = list({(h["source"], h["page"]): h for h in cited}.values())
    citations = [
        Citation(
            document_id=document_id(h["url"]) if h.get("url") else h["source"],
            document_title=h["source"],
            url=h.get("url"),
            quote=h["text"][:400],
            page_start=h["page"],
            page_end=h["page"],
            verified=verified,
        )
        for h in cited
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
    # Sources de la reponse courte : l'explication s'appuie d'abord sur elles.
    sources: list[SourceRef] = Field(default_factory=list, max_length=6)
    language: Literal["fr", "wo", "en", "ff", "srr", "dyo"] = "fr"


class DetailSource(BaseModel):
    n: int
    title: str
    page: int | None = None
    url: str | None = None


class ExplainResponse(BaseModel):
    # Texte avec des references [[n]] placees apres chaque passage, renvoyant a `sources`.
    details: str
    sources: list[DetailSource] = Field(default_factory=list)


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
        prefer = [(src.title, src.page) for src in req.sources if src.page is not None]
        answer_key = hashlib.sha1(req.answer.strip().encode()).hexdigest()[:12]
        key = f"explain:v8:{corpus_version()}:{req.language}:{normalize(question)}:{sorted(set(prefer))}:{answer_key}"
        data, _ = await store.cached(key, lambda: _explain(question, req.answer.strip(), req.language, prefer))
        ok = True
        return ExplainResponse(**data)
    finally:
        analytics.log_event(
            "explain", client_id=x_client_id, session_id=x_session_id,
            question=question, language=req.language, error=not ok,
        )


async def _explain(question: str, answer: str, language: str, prefer: list[tuple[str, int]] | None = None) -> dict:
    try:
        hits = filter_by_years(question, await aretrieve(question, prefer)) or await aretrieve(question, prefer)
        if not hits:
            raise HTTPException(status_code=404, detail=NO_INDEX_MESSAGE)
        details, sources = await aexplain(question, answer, hits, language)
        return {"details": details, "sources": sources}
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


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    language: Literal["fr", "wo", "en", "ff", "srr", "dyo"] = "wo"


@app.post("/api/voice/transcribe")
async def voice_transcribe(
    request: Request,
    audio: UploadFile = File(...),
    language: str = Form("wo"),
    x_client_id: str | None = Header(default=None),
) -> dict:
    """Question posee a voix haute -> texte (Soynade, voir voice.py)."""
    await enforce_rate_limit(request, x_client_id)
    try:
        text = await voice.transcribe(await audio.read(), audio.filename or "", language)
    except voice.VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail)
    return {"text": text}


@app.post("/api/voice/speak")
async def voice_speak(
    req: SpeakRequest, request: Request, x_client_id: str | None = Header(default=None)
) -> Response:
    """Lecture a voix haute d'une reponse (Soynade, voir voice.py). 501 pour les langues
    sans voix : le frontend retombe alors sur la synthese du navigateur."""
    await enforce_rate_limit(request, x_client_id)
    try:
        content, media_type = await voice.synthesize(req.text, req.language)
    except voice.VoiceError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail)
    return Response(content=content, media_type=media_type)
