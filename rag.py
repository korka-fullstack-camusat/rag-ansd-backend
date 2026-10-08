import asyncio
import os
import random
import re
import unicodedata
from dataclasses import dataclass

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
    "ajouter : n'invente jamais de reponse, et n'affirme jamais qu'une donnee existe si "
    "elle n'apparait pas explicitement dans les extraits. Si les extraits contiennent des "
    "chiffres en rapport avec la demande, donne les plus pertinents (valeur, unite, periode). "
    "Si l'indicateur exact demande n'y figure pas mais que des chiffres sur le meme sujet et la "
    "meme periode y sont, donne-les en precisant ce qu'ils mesurent. Ne reponds jamais seulement "
    f"que « l'information n'est pas indiquee » : sans aucun chiffre utile, reponds {NO_DATA_MARKER}. "
    "Pour un montant tire d'un tableau, precise sa nature telle qu'indiquee dans le titre du "
    "tableau (prix courants ou volumes chaines, unite) et lis la valeur dans la colonne de "
    "l'annee demandee ; dans une suite de questions, garde la meme base que la reponse "
    "precedente si elle est disponible. "
    "Chaque chiffre doit garder exactement le sens qu'il a dans l'extrait : ne presente jamais "
    "un taux de reponse, une part, un poids ou un indice comme une evolution (ou l'inverse). "
    "Si une valeur demandee ne figure pas dans les extraits, dis-le pour cette valeur au lieu "
    "de la remplacer par un autre chiffre. "
    "FORMAT : par defaut, sois bref et precis — donne directement le chiffre ou le fait "
    "demande (avec son unite et sa periode) en 1 a 3 phrases, sans introduction, sans "
    "repeter la question, sans conclusion et sans mise en forme. Mais si l'utilisateur "
    "demande un format, respecte-le : « point par point », « en liste » → une ligne par "
    "point commencant par « - » ; « numerote », « etapes » → lignes « 1. », « 2. »… ; "
    "« tableau » → tableau Markdown simple (| colonne | colonne |) ; « plus court », « en une "
    "phrase » → une seule phrase ; « plus de details », « explique » → 1 a 3 courts "
    "paragraphes. Tu peux mettre en **gras** les chiffres cles d'une liste ou d'un tableau. "
    "Si la demande porte seulement sur la forme de la reponse precedente, reprends ses "
    "informations dans le nouveau format, sans en ajouter qui ne soient pas dans les extraits. "
    "N'ecris pas de references aux sources dans le texte (elles sont affichees a part). "
    "Termine par une ligne separee « SOURCES: » suivie des numeros des seuls extraits "
    "que tu as reellement utilises (exemple : SOURCES: 2, 5)."
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
    "rien de plus, dis-le en une phrase. Chaque chiffre garde exactement le sens qu'il a dans "
    "l'extrait (un taux de reponse n'est pas une evolution). N'ecris pas de references aux "
    "sources dans le texte."
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


# ------------------------------------------------------------------ portee de la question
# Une question qui cite une page (« page 3 », « p. 12 », « pages 3 a 5 ») ou un
# document (« ICAS, T2 2026 », « comptes nationaux provisoires 2025 ») est
# limitee a ceux-ci : la recherche ne va pas chercher ailleurs.

PAGE_RE = re.compile(
    r"\b(?:pages?|p\.)\s*(\d{1,4})(?:\s*(?:-|–|à|a|au|et|to|and)\s*(\d{1,4}))?",
    re.IGNORECASE,
)
_sources_cache: dict = {"count": None, "titles": []}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def known_sources() -> list[str]:
    """Titres des documents indexes (relus seulement si le corpus change)."""
    collection = get_collection()
    count = collection.count()
    if _sources_cache["count"] != count:
        metas = collection.get(include=["metadatas"])["metadatas"] if count else []
        _sources_cache["titles"] = sorted({m["source"] for m in metas})
        _sources_cache["count"] = count
    return _sources_cache["titles"]


@dataclass
class Scope:
    sources: list[str]
    pages: list[int]

    def describe(self) -> str:
        parts = []
        if self.sources:
            parts.append("document " + " / ".join(f"« {s} »" for s in self.sources))
        if self.pages:
            parts.append(("page " if len(self.pages) == 1 else "pages ") + ", ".join(map(str, self.pages)))
        return ", ".join(parts)


def parse_scope(question: str) -> Scope:
    q = f" {_norm(question)} "

    # Document : titre complet cite, sinon son sigle (ICAS, ICAI…) — en departageant
    # les documents de meme sigle par les autres mots du titre (T2, 2026…).
    full = [t for t in known_sources() if f" {_norm(t)} " in q]
    sources = full
    if not full:
        by_acronym = []
        for title in known_sources():
            first = title.split()[0].strip(",;:")
            if len(first) >= 3 and first.isupper() and f" {first.lower()} " in q:
                extra = [w for w in _norm(title).split()[1:] if f" {w} " in q]
                by_acronym.append((len(extra), title))
        if by_acronym:
            best = max(score for score, _ in by_acronym)
            sources = [t for score, t in by_acronym if score == best]

    pages: list[int] = []
    for m in PAGE_RE.finditer(question):
        first = int(m.group(1))
        last = int(m.group(2)) if m.group(2) else first
        if last < first:
            first, last = last, first
        pages.extend(range(first, min(last, first + 19) + 1))
    return Scope(sources=sources, pages=sorted(set(pages)))


def _where(scope: Scope) -> dict | None:
    clauses = []
    if scope.sources:
        clauses.append({"source": {"$in": scope.sources}})
    if scope.pages:
        clauses.append({"page": {"$in": scope.pages}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# Sommaires et listes de tableaux (« Analyse du PIB ........ 9 ») : leurs numeros
# de page seraient lus comme des chiffres. On les ecarte des resultats.
_TOC_LEADERS = re.compile(r"(?:\.\s?){6,}\s*\d")


def _is_table_of_contents(text: str) -> bool:
    return len(_TOC_LEADERS.findall(text)) >= 2


def _hits(results) -> list[dict]:
    return [
        hit
        for hit in _raw_hits(results)
        if not _is_table_of_contents(hit["text"])
    ]


def _raw_hits(results) -> list[dict]:
    return [
        {
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "url": meta.get("url"),
            "score": 1 - distance,
        }
        for text, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def retrieve(question: str, top_k: int = TOP_K, prefer: list[tuple[str, int]] | None = None) -> list[dict]:
    """Extraits les plus pertinents. `prefer` : (document, page) deja cites dans la
    discussion — une question de suite (« donne-moi les chiffres ») y cherche
    d'abord, puis complete avec le reste du corpus."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    scope = parse_scope(question)
    where = _where(scope)
    query_embedding = [vec.tolist() for vec in get_embedder().embed([question])]
    # Page(s) demandee(s) : on prend tous leurs fragments (dans une limite raisonnable),
    # classes par pertinence ; sinon les `top_k` plus proches (dans le document cite le cas echeant).
    n_results = 12 if scope.pages else top_k + 3
    hits = _hits(collection.query(query_embeddings=query_embedding, n_results=n_results, where=where))
    if not scope.pages:
        hits = hits[:top_k]

    if prefer and where is None:
        pairs = [{"$and": [{"source": s}, {"page": p}]} for s, p in dict.fromkeys(prefer)][:8]
        prior = _hits(
            collection.query(
                query_embeddings=query_embedding,
                n_results=4,
                where=pairs[0] if len(pairs) == 1 else {"$or": pairs},
            )
        )
        seen = {h["text"] for h in prior}
        hits = prior + [h for h in hits if h["text"] not in seen][: max(0, top_k - 2)]
    return hits


# --- garde-fou : sigle absent de tout le corpus (ex. « DR ») => pas de donnees,
# plutot que de laisser le modele affirmer que l'information existe.
ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_acronyms_cache: dict = {"count": None, "set": set()}


def corpus_acronyms() -> set[str]:
    collection = get_collection()
    count = collection.count()
    if _acronyms_cache["count"] != count:
        docs = collection.get(include=["documents", "metadatas"]) if count else {"documents": [], "metadatas": []}
        found: set[str] = set()
        for text, meta in zip(docs["documents"], docs["metadatas"]):
            found.update(ACRONYM_RE.findall(text))
            found.update(ACRONYM_RE.findall(meta.get("source", "")))
        _acronyms_cache.update(count=count, set=found)
    return _acronyms_cache["set"]


def unknown_acronyms(question: str) -> list[str]:
    known = corpus_acronyms()
    return [a for a in ACRONYM_RE.findall(question) if a not in known]


# --- conversation courante (salutations, remerciements, presentation)
SMALL_TALK = [
    (
        r"^(bonjour|bonsoir|salut|hello|hi|hey|coucou|salam|salamalekum|salaam aleykoum|asalaa?m? ?(maa)?lekum|nanga ?def|na nga def)\b",
        {
            "fr": "Bonjour ! Je suis l'assistant de l'ANSD. Posez-moi une question sur les statistiques du Sénégal : croissance, prix, emploi, chiffre d'affaires des entreprises…",
            "en": "Hello! I'm the ANSD assistant. Ask me about Senegal's statistics: growth, prices, employment, business turnover…",
        },
    ),
    (
        r"^(merci|thanks?|thank you|jerej[eë]f|jerejef)\b",
        {
            "fr": "Avec plaisir ! N'hésitez pas si vous avez une autre question.",
            "en": "You're welcome! Feel free to ask another question.",
        },
    ),
    (
        r"^(au revoir|bye|goodbye|a bientot|à bientôt|ba beneen|ba benn yoon)\b",
        {
            "fr": "Au revoir et à bientôt !",
            "en": "Goodbye, see you soon!",
        },
    ),
    (
        r"^(qui es[- ]tu|tu es qui|c'?est quoi (cet|ce) (assistant|chatbot)|que (sais|peux)[- ]tu faire|who are you|what can you do)\b",
        {
            "fr": "Je suis l'assistant de l'ANSD. Je réponds à vos questions à partir des publications officielles de l'ANSD, en citant à chaque fois le document et la page. Vous pouvez aussi me demander une réponse point par point, en tableau, plus courte ou plus détaillée.",
            "en": "I'm the ANSD assistant. I answer your questions from ANSD's official publications, always citing the document and page. You can also ask for a bullet-point, table, shorter or more detailed answer.",
        },
    ),
]


def small_talk_reply(question: str, language: str) -> str | None:
    """Reponse directe aux messages de conversation courante (« bonjour », « merci »…),
    sans recherche documentaire. None si c'est une vraie question. Langues sans
    traduction validee (wolof, pulaar, sereer, diola) : reponse en francais."""
    text = question.strip().lower().rstrip(" !?.")
    if len(text) > 40:  # une phrase longue est une vraie question
        return None
    for pattern, replies in SMALL_TALK:
        if re.match(pattern, text):
            return replies.get(language) or replies["fr"]
    return None


# --- demandes de mise en forme seules (« point par point », « en tableau »…)
# Reconnues par regle, sans le modele : le sujet est celui de la question precedente.
# (consigne, mots qui la declenchent — sans accents, debut de mot)
FORMAT_REQUESTS = [
    ("sous forme de tableau", ("tableau", "tableaux", "table")),
    ("sous forme de liste numérotée", ("numero", "etape", "numbered")),
    ("point par point (liste à puces)", ("point", "points", "liste", "listes", "puce", "puces", "bullet", "list")),
    ("en une seule phrase", ("court", "simple", "resum", "phrase", "bref", "brievement", "shorter", "summar")),
    ("de façon plus détaillée", ("detail", "developpe", "explique", "elabor")),
]
_FORMAT_FILLERS = set(
    (
        "affiche afficher mets mettre met ca ce cela ceci le la les l un une des de du d en sous forme dans "
        "moi me te tu peux pourrais pourriez stp svp s il plait plais vous voulez donne donner fais faire "
        "presente presenter montre montrer reponse reponses plus encore juste et avec par a au aux "
        "ecris ecrire redige rediger reformule reformuler version maintenant aussi oui merci "
        "show give make put it this that as in a the please answer can you could more"
    ).split()
)
_FORMAT_SUFFIX = re.compile(r"\s*— réponds .*$")


def format_only_clause(question: str) -> str | None:
    """Consigne de forme si le message n'est QU'une demande de mise en forme
    (court, sans nouveau sujet) ; None sinon — « explique-moi le PIB » est une
    vraie question, « mets ça dans un tableau » une mise en forme."""
    words = _norm(question).split()
    if not words or len(words) > 10:
        return None
    clause = None
    for word in words:
        if word in _FORMAT_FILLERS:
            continue
        match = next((c for c, stems in FORMAT_REQUESTS if any(word.startswith(st) for st in stems)), None)
        if match is None:
            return None  # mot porteur de sens : nouvelle question
        clause = clause or match
    return clause


# --- relances de periode seules (« et pour 2025 ? », « je veux juste 2026 »…)
# Regle fixe : sujet de la question precedente, seule la periode change.
_PERIOD_FILLERS = _FORMAT_FILLERS | set(
    (
        "et pour en l annee annees an ans sur concernant les donnees donnee chiffres chiffre "
        "je veux voudrais juste seulement uniquement alors quid que qu est ce qui quel quelle "
        "for year years what about and data only just"
    ).split()
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_QUARTER = re.compile(r"\b(?:t[1-4]|[1-4](?:er|e|eme|ème)? trimestre|q[1-4])\b", re.IGNORECASE)


def period_only(question: str) -> list[str] | None:
    """Periode(s) citee(s) si le message n'est QU'une relance de periode ; None sinon."""
    text = _norm(question)
    years = _YEAR.findall(text)
    quarters = [q.upper() for q in _QUARTER.findall(text)]
    if not years and not quarters:
        return None
    rest = _QUARTER.sub(" ", _YEAR.sub(" ", text))
    leftover = [w for w in rest.split() if w not in _PERIOD_FILLERS and w not in ("trimestre", "trimestres")]
    return None if leftover else years + quarters


def with_period(previous_question: str, periods: list[str]) -> str:
    """Question precedente avec la nouvelle periode : l'annee citee est remplacee
    si la question n'en contenait qu'une, sinon la periode est precisee a la fin."""
    base = _FORMAT_SUFFIX.sub("", previous_question).rstrip(" ?.")
    years = [p for p in periods if _YEAR.fullmatch(p)]
    if len(years) == 1 and len(set(_YEAR.findall(base))) == 1 and len(periods) == 1:
        return f"{_YEAR.sub(years[0], base)} ?"
    return f"{base} — période : {' '.join(periods)} ?"


def with_format(previous_question: str, clause: str) -> str:
    """Question precedente (sans son eventuelle ancienne consigne) + nouvelle consigne."""
    return f"{_FORMAT_SUFFIX.sub('', previous_question).rstrip(' ?.')} — réponds {clause}."


# --- reformulation des questions de suite
CONDENSE_PROMPT = (
    "Tu reformules la derniere question d'une discussion en une question autonome, "
    "comprehensible sans l'historique : remplace les references implicites (« ces "
    "chiffres », « cette page », « et en 2024 ? », « donne-moi plus de details ») par ce "
    "qu'elles designent (sujet, document, page, periode). Garde le sujet tel que l'utilisateur "
    "l'a formule (par exemple « les emplois du PIB ») : ne le remplace pas par un detail tire "
    "d'une reponse ; si l'utilisateur recentre la discussion (« on va parler de X »), X devient "
    "le sujet. N'ajoute jamais une annee ou une periode que l'utilisateur n'a pas citee. Pour « et pour 2025 ? », « et en 2024 ? », reprends le sujet de la discussion "
    "et change seulement la periode, meme si la question precedente n'a pas eu de reponse. "
    "Si la derniere question contient une "
    "consigne de forme, reprends-la avec ses propres mots ; n'en ajoute aucune qui vienne des "
    "questions precedentes. Reste fidele a la question : "
    "n'ajoute aucun detail, liste ou indicateur que l'utilisateur n'a pas demande, et reste "
    "court. Garde la langue de la question. Si la question est deja autonome, renvoie-la "
    "telle quelle. Reponds uniquement par la question."
)


def _condense_messages(question: str, history: list[dict]) -> list[dict]:
    lines = []
    for turn in history[-2:]:
        sources = ", ".join(f"{s['title']} p.{s['page']}" for s in turn.get("sources", [])[:4])
        answer = turn["answer"][:600] if turn.get("sources") else f"{turn['answer'][:200]} (aucune donnee trouvee)"
        lines.append(f"Question : {turn['question']}\nReponse : {answer}")
        if sources:
            lines.append(f"Sources de cette reponse : {sources}")
    return [
        {"role": "system", "content": CONDENSE_PROMPT},
        {"role": "user", "content": "\n".join(lines) + f"\n\nDerniere question : {question}"},
    ]


SOURCES_RE = re.compile(r"\n?[ \t]*\**SOURCES?\**\s*:\s*([\d,;\set&]*)\s*$", re.IGNORECASE)


def split_used_sources(answer: str, n_hits: int) -> tuple[str, list[int]]:
    """Retire la ligne « SOURCES: 2, 5 » de la reponse et renvoie les indices
    (0-based) des extraits que le modele dit avoir utilises."""
    match = SOURCES_RE.search(answer)
    if not match:
        return answer.strip(), []
    indices = []
    for n in re.findall(r"\d+", match.group(1)):
        i = int(n) - 1
        if 0 <= i < n_hits and i not in indices:
            indices.append(i)
    return answer[: match.start()].strip(), indices


LANGUAGE_NAMES = {
    "fr": "francais",
    "en": "anglais",
    "wo": "wolof",
    "ff": "pulaar (peul du Senegal)",
    "srr": "serere (seereer)",
    "dyo": "diola (joola-fonyi)",
}


def build_prompt(question: str, hits: list[dict], language: str = "fr", previous: str | None = None) -> str:
    context = "\n\n".join(
        f"[Source {i+1} - {h['source']}, p.{h['page']}]\n{h['text']}"
        for i, h in enumerate(hits)
    )
    scope = parse_scope(question)
    focus = (
        f"La question porte precisement sur : {scope.describe()}. "
        "Reponds uniquement a partir des extraits correspondants.\n\n"
        if scope.sources or scope.pages
        else ""
    )
    earlier = (
        f"Reponse precedente dans la discussion (a reprendre si l'utilisateur demande "
        f"seulement de la reformuler ou de la presenter autrement) :\n{previous[:1500]}\n\n"
        if previous
        else ""
    )
    return (
        f"Contexte extrait des rapports ANSD :\n\n{context}\n\n"
        f"{earlier}{focus}Question : {question}\n\n"
        f"Reponds en {LANGUAGE_NAMES.get(language, 'francais')}, en t'appuyant uniquement "
        "sur le contexte ci-dessus, dans le format demande (bref par defaut)."
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


def _answer_messages(question: str, hits: list[dict], language: str, previous: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits, language, previous)},
    ]


def call_llm(question: str, hits: list[dict], language: str = "fr") -> dict:
    return _chat(_answer_messages(question, hits, language), temperature=0.2, max_tokens=300)


async def acondense(question: str, history: list[dict]) -> str:
    """Question autonome a partir d'une question de suite et de l'historique."""
    if not history:
        return question
    completion = await _achat(_condense_messages(question, history), timeout=15, temperature=0, max_tokens=120)
    rewritten = completion["choices"][0]["message"]["content"].strip().strip("«»\"")
    return rewritten or question


async def acall_llm(
    question: str, hits: list[dict], language: str = "fr", previous: str | None = None, temperature: float = 0.2
) -> dict:
    # max_tokens assez large pour une liste ou un tableau demandes explicitement.
    return await _achat(_answer_messages(question, hits, language, previous), temperature=temperature, max_tokens=600)


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
