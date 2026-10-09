"""Index de recherche rapide, construit une fois a partir de la base Chroma.

Chroma reste la base de reference (ingestion), mais ses requetes coutent 0,3 a 4 s, surtout avec un
filtre (annee, document). Cet index fait la meme recherche en quelques dizaines de millisecondes :

  - vecteurs : tous les embeddings normalises, en memoire (numpy, ~440 Mo pour 284 000 passages) ;
    la similarite est calculee exactement pour tous les passages (produit matrice-vecteur), et les
    filtres annee / document sont de simples masques ;
  - plein texte : SQLite FTS5 (BM25, sans accents), pour les mots exacts que l'embedding classe mal
    (« chomage des jeunes », sigles, noms de region) ;
  - fusion des deux classements (Reciprocal Rank Fusion), puis fraicheur de la publication.

Fichiers (volume Docker dedie, FASTINDEX_DIR) : vectors.npy, rows.npz, passages.sqlite3, info.json.
L'index est reconstruit automatiquement si le nombre de passages de Chroma change (~2 minutes).
"""

import datetime
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

import rag
from config import CHROMA_DIR, ROOT_DIR

logger = logging.getLogger("ansd-fastindex")
if not logger.handlers:  # construction (~2 min) visible dans « docker logs »
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Dossier racine (volume Docker) : l'index courant est dans « current », reconstruit dans « building »
# puis renomme (le point de montage lui-meme ne peut pas etre remplace).
INDEX_ROOT = Path(os.environ.get("FASTINDEX_DIR", str(ROOT_DIR / "fastindex")))
INDEX_DIR = INDEX_ROOT / "current"
BATCH = 5000
RRF_K = 60
CANDIDATES = 150  # passages retenus par chaque classement avant fusion
RECENT_CANDIDATES = 50
RELEVANCE_FLOOR = 0.45  # part du meilleur score de pertinence pour rester candidat
RECENCY_WEIGHT = 0.35  # bonus maximal (publication de l'annee en cours), score de pertinence ramene a 1
VALUE_WEIGHT = 0.05

# ------------------------------------------------------------------ dates des publications

_MONTHS = ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre", "novembre", "decembre"]


def pub_date(url: str | None) -> tuple[int, int]:
    """(annee, mois) de mise en ligne, lus dans l'adresse ansd.sn (/files/AAAA-MM/…)."""
    m = re.search(r"/files/(\d{4})-(\d{2})/", url or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def doc_year(title: str, url: str | None) -> int:
    """Annee d'une publication : la plus recente citee dans son titre (« SES 2011 », « ENES, T3-2025 »,
    « BADIS 2014-2018 »), a defaut l'annee de mise en ligne. La date de mise en ligne seule ne
    convient pas : beaucoup d'anciens rapports ont ete deposes en bloc sur le site en 2022."""
    this_year = datetime.date.today().year
    years = [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", title or "") if int(y) <= this_year]
    return max(years) if years else pub_date(url)[0]


def period_label(title: str, url: str | None) -> str:
    """Periode d'une publication, lisible : « juin 2026 », « 2e trimestre 2026 », « 2025 »."""
    year = doc_year(title, url)
    if not year:
        return "date inconnue"
    norm = rag._norm(title or "")
    for i, name in enumerate(_MONTHS, 1):
        if re.search(rf"\b{name}\b", norm):
            return f"{_MONTHS_FR[i - 1]} {year}"
    m = re.search(r"\bt([1-4])\b", norm) or re.search(r"\b([1-4]) ?(?:er|e|eme)? trimestre\b|\btrimestre ([1-4])\b", norm)
    if m:
        q = int(next(g for g in m.groups() if g))
        return f"{q}{'er' if q == 1 else 'e'} trimestre {year}"
    return str(year)


_MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]


def doc_order(title: str, url: str | None) -> int:
    """Cle de fraicheur : annee * 100 + mois (ou fin du trimestre) lus dans le titre
    (« ENES, T2-2026 » > « ENES, T1-2026 », « IHPC, aout 2026 » > « IHPC, mars 2026 »)."""
    year = doc_year(title, url)
    norm = rag._norm(title or "")
    month = 0
    for i, name in enumerate(_MONTHS, 1):
        if re.search(rf"\b{name}\b", norm):
            month = i
    if not month:
        m = re.search(r"\bt([1-4])\b", norm) or re.search(r"\b([1-4]) ?(?:er|e|eme)? trimestre\b|\btrimestre ([1-4])\b", norm)
        if m:
            month = int(next(g for g in m.groups() if g)) * 3
    return year * 100 + month


# ------------------------------------------------------------------ mots-cles

_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "au", "aux", "en", "et", "ou", "a", "sur", "dans",
    "par", "pour", "avec", "que", "qui", "quel", "quelle", "quels", "quelles", "est", "sont", "ete", "etait", "ce",
    "cet", "cette", "ces", "son", "sa", "ses", "leur", "il", "elle", "the", "of", "in", "and", "or", "is", "are",
    "what", "which", "how", "to", "for", "senegal", "ansd", "plus", "moins", "selon", "entre", "combien", "donne",
    "moi", "recent", "recente", "dernier", "derniere", "actuel", "actuelle", "niveau",
    # politesse : « merci beaucoup », « bonjour » ne declenchent pas de recherche
    "merci", "beaucoup", "bonjour", "bonsoir", "salut", "super", "parfait", "genial", "hello", "thanks", "thank",
    "you", "okay", "cool", "bien", "tres", "nanga", "def", "jerejef",
}


def keywords(query: str) -> list[str]:
    """Mots porteurs de sens de la requete (sans accents, sans annees, singuliers approximatifs)."""
    out = []
    for w in rag._norm(query).split():
        if w in _STOPWORDS or len(w) < 3 or re.fullmatch(r"(?:19|20)\d{2}", w):
            continue
        if len(w) > 4 and w[-1] in "sx":
            w = w[:-1]
        if w not in out:
            out.append(w)
    return out


def key_phrase(query: str) -> str:
    """Mots porteurs de sens tels qu'ecrits (accents compris : l'embedding y est sensible),
    sans mots-outils ni annees : « Quel est le taux de chômage au Sénégal ? » -> « taux chômage »."""
    out = []
    for w in re.findall(r"[\w'’-]+", query):
        word = re.split(r"['’]", w)[-1]
        norm = rag._norm(word)
        if not norm or norm in _STOPWORDS or len(norm) < 3 or re.fullmatch(r"(?:19|20)\d{2}", norm):
            continue
        out.append(word)
    return " ".join(out)


_VALUE_RE = re.compile(r"\d+[,.]\d+\s?%|\b\d{1,3}(?:[\s  ]\d{3}){2,}\b")


def has_value(text: str) -> bool:
    return bool(_VALUE_RE.search(text))


# ------------------------------------------------------------------ construction


def _chroma_ids() -> list[str]:
    con = sqlite3.connect(f"file:{CHROMA_DIR / 'chroma.sqlite3'}?mode=ro", uri=True, timeout=60)
    try:
        return [r[0] for r in con.execute("SELECT embedding_id FROM embeddings ORDER BY id")]
    finally:
        con.close()


def build(target: Path = INDEX_DIR) -> None:
    """Exporte Chroma vers l'index rapide (dossier temporaire puis remplacement atomique)."""
    started = time.time()
    collection = rag.get_collection()
    ids = _chroma_ids()
    n = len(ids)
    if n == 0:
        raise RuntimeError("base Chroma vide")
    tmp = target.with_name("building")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    db = sqlite3.connect(tmp / "passages.sqlite3")
    db.executescript(
        """
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE docs(id INTEGER PRIMARY KEY, title TEXT, url TEXT, year INTEGER, ord INTEGER);
        CREATE TABLE passages(id INTEGER PRIMARY KEY, doc INTEGER, page INTEGER, text TEXT);
        CREATE VIRTUAL TABLE fts USING fts5(text, content='passages', content_rowid='id',
                                            tokenize='unicode61 remove_diacritics 2');
        """
    )
    vectors = None
    doc_of_row = np.zeros(n, dtype=np.int32)
    toc = np.zeros(n, dtype=bool)
    docs: dict[str, int] = {}
    doc_rows = []
    for start in range(0, n, BATCH):
        batch = collection.get(ids=ids[start : start + BATCH], include=["embeddings", "documents", "metadatas"])
        emb = np.asarray(batch["embeddings"], dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        if vectors is None:
            vectors = np.lib.format.open_memmap(tmp / "vectors.npy", mode="w+", dtype=np.float32, shape=(n, emb.shape[1]))
        vectors[start : start + len(emb)] = emb
        rows = []
        for j, (text, meta) in enumerate(zip(batch["documents"], batch["metadatas"])):
            text = text or ""
            meta = meta or {}
            title = meta.get("source") or ""
            if title not in docs:
                url = meta.get("url")
                docs[title] = len(docs)
                doc_rows.append((docs[title], title, url, doc_year(title, url), doc_order(title, url)))
            row = start + j
            doc_of_row[row] = docs[title]
            toc[row] = rag._is_table_of_contents(text)
            rows.append((row, docs[title], int(meta.get("page") or 0), text))
        db.executemany("INSERT INTO passages VALUES (?,?,?,?)", rows)
        logger.info("index rapide : %d / %d passages", min(start + BATCH, n), n)
    db.executemany("INSERT INTO docs VALUES (?,?,?,?,?)", doc_rows)
    db.execute("INSERT INTO fts(fts) VALUES('rebuild')")
    db.execute("INSERT INTO fts(fts) VALUES('optimize')")
    db.commit()
    db.close()
    vectors.flush()
    del vectors
    np.savez(tmp / "rows.npz", doc=doc_of_row, toc=toc)
    (tmp / "info.json").write_text(json.dumps({"count": n, "built_at": time.time()}), encoding="utf-8")
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    logger.info("index rapide construit : %d passages, %d documents, %.0f s", n, len(docs), time.time() - started)


# ------------------------------------------------------------------ recherche


class FastIndex:
    def __init__(self, path: Path = INDEX_DIR) -> None:
        self.path = path
        self.count = json.loads((path / "info.json").read_text(encoding="utf-8"))["count"]
        # Projete en memoire : les processus d'un meme conteneur partagent les memes pages (cache systeme).
        self.vectors = np.load(path / "vectors.npy", mmap_mode="r")
        rows = np.load(path / "rows.npz")
        self.doc_of_row = rows["doc"]
        self.toc = rows["toc"]
        self._local = threading.local()
        con = self._db()
        docs = con.execute("SELECT id, title, url, year, ord FROM docs ORDER BY id").fetchall()
        self.titles = [d[1] for d in docs]
        self.urls = [d[2] for d in docs]
        self.doc_years = np.array([d[3] for d in docs], dtype=np.int32)
        self.doc_orders = np.array([d[4] for d in docs], dtype=np.int32)
        self.title_ids = {t: i for i, t in enumerate(self.titles)}
        # Fraicheur au mois pres (T2-2026 > T1-2026), de 0 (2012 ou avant) a 1 (aujourd'hui).
        now = datetime.date.today()
        months = (self.doc_orders // 100) * 12 + np.clip(self.doc_orders % 100, 0, 12)
        span = (now.year * 12 + now.month) - 2012 * 12
        recency = np.clip((months - 2012 * 12) / span, 0, 1).astype(np.float32)
        self.row_recency = recency[self.doc_of_row]
        self.row_recent = (self.doc_years >= now.year - 1)[self.doc_of_row]
        self._year_masks: OrderedDict[int, np.ndarray] = OrderedDict()
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def _db(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.path / 'passages.sqlite3'}?mode=ro", uri=True, check_same_thread=False)
            con.execute("PRAGMA mmap_size=1073741824")
            self._local.con = con
        return con

    def embed(self, query: str) -> np.ndarray:
        with self._lock:
            cached = self._embed_cache.get(query)
            if cached is not None:
                self._embed_cache.move_to_end(query)
                return cached
        vec = np.asarray(next(iter(rag.get_embedder().embed([query]))), dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        with self._lock:
            self._embed_cache[query] = vec
            while len(self._embed_cache) > 2000:
                self._embed_cache.popitem(last=False)
        return vec

    def year_mask(self, year: int) -> np.ndarray:
        """Passages qui citent l'annee dans leur texte, ou dont la publication porte cette annee."""
        with self._lock:
            mask = self._year_masks.get(year)
        if mask is not None:
            return mask
        mask = (self.doc_years == year)[self.doc_of_row]
        rowids = [r for (r,) in self._db().execute("SELECT rowid FROM fts WHERE fts MATCH ?", (f'"{year}"',))]
        if rowids:
            mask = mask.copy()
            mask[np.asarray(rowids, dtype=np.int64)] = True
        with self._lock:
            self._year_masks[year] = mask
            while len(self._year_masks) > 64:
                self._year_masks.popitem(last=False)
        return mask

    def docs_matching(self, document: str) -> np.ndarray:
        wanted = rag._norm(document)
        ids = [i for i, t in enumerate(self.titles) if wanted and wanted in rag._norm(t)]
        return np.asarray(ids, dtype=np.int32)

    def search(
        self,
        query: str,
        k: int = 8,
        year: int | None = None,
        document: str | None = None,
        latest: bool = True,
        max_per_doc: int = 3,
    ) -> list[dict]:
        if not query.strip():
            return []
        allowed = ~self.toc
        if document and document.strip():
            doc_ids = self.docs_matching(document)
            if len(doc_ids):
                allowed = allowed & np.isin(self.doc_of_row, doc_ids)
        if year:
            allowed = allowed & self.year_mask(int(year))
        if not allowed.any():
            return []

        # 1. Similarite semantique exacte sur tous les passages autorises.
        sims = self.vectors @ self.embed(query)
        sims = np.where(allowed, sims, -2.0)
        top = np.argpartition(-sims, min(CANDIDATES, len(sims) - 1))[:CANDIDATES]
        vec_rank = top[np.argsort(-sims[top])]
        vec_rank = vec_rank[sims[vec_rank] > -1.0]
        # Meme recherche limitee aux publications des deux dernieres annees : sans elle, une serie
        # ancienne et volumineuse (SES 2011-2016…) occupe toutes les places et le dernier chiffre manque.
        recent_rank = np.array([], dtype=np.int64)
        if latest and not year:
            recent_sims = np.where(self.row_recent, sims, -2.0)
            top = np.argpartition(-recent_sims, min(RECENT_CANDIDATES, len(sims) - 1))[:RECENT_CANDIDATES]
            recent_rank = top[np.argsort(-recent_sims[top])]
            recent_rank = recent_rank[recent_sims[recent_rank] > -1.0]

        # 2. Plein texte BM25 sur les mots de la requete (le premier passe le filtre en Python).
        bm25_rank: list[int] = []
        words = keywords(query)
        if words:
            match = " OR ".join(f"{w}*" for w in words)
            limit = CANDIDATES if allowed.all() else 2000
            try:
                for (rowid,) in self._db().execute(
                    "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?", (match, limit)
                ):
                    if allowed[rowid]:
                        bm25_rank.append(rowid)
                        if len(bm25_rank) >= CANDIDATES:
                            break
            except sqlite3.OperationalError:
                logger.warning("requete plein texte invalide : %r", match)

        # 2 bis. Plein texte limite aux publications recentes (meme raison que recent_rank).
        recent_bm25: list[int] = []
        if latest and not year and words:
            try:
                for (rowid,) in self._db().execute(
                    "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT 3000", (" OR ".join(f"{w}*" for w in words),)
                ):
                    if allowed[rowid] and self.row_recent[rowid]:
                        recent_bm25.append(rowid)
                        if len(recent_bm25) >= RECENT_CANDIDATES:
                            break
            except sqlite3.OperationalError:
                pass

        # 3. Fusion des classements (Reciprocal Rank Fusion).
        relevance: dict[int, float] = {}
        for rank, row in enumerate(vec_rank):
            relevance[int(row)] = relevance.get(int(row), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, row in enumerate(bm25_rank):
            relevance[row] = relevance.get(row, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, row in enumerate(recent_rank):
            relevance[int(row)] = relevance.get(int(row), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, row in enumerate(recent_bm25):
            relevance[row] = relevance.get(row, 0.0) + 1.0 / (RRF_K + rank + 1)
        if not relevance:
            return []
        top_relevance = max(relevance.values())
        # 4. Parmi les passages vraiment pertinents, les publications recentes et ceux qui contiennent
        #    une valeur chiffree passent devant (une serie annuelle renverrait sinon ses vieux numeros).
        pool = [r for r in sorted(relevance, key=relevance.get, reverse=True)[:CANDIDATES]
                if relevance[r] >= RELEVANCE_FLOOR * top_relevance]
        placeholders = ",".join("?" * len(pool))
        rows_data = {
            rid: (page, text)
            for rid, page, text in self._db().execute(
                f"SELECT id, page, text FROM passages WHERE id IN ({placeholders})", pool
            )
        }
        scores = {}
        for row in pool:
            score = relevance[row] / top_relevance
            if latest:
                score += RECENCY_WEIGHT * float(self.row_recency[row])
            if has_value(rows_data.get(row, (0, ""))[1]):
                score += VALUE_WEIGHT
            scores[row] = score
        best = sorted(scores, key=scores.get, reverse=True)
        texts = {r: d[1] for r, d in rows_data.items()}
        pages = {r: d[0] for r, d in rows_data.items()}

        picked: list[dict] = []
        per_doc: dict[int, int] = {}
        seen: set[str] = set()
        for row in best:
            doc = int(self.doc_of_row[row])
            text = texts.get(row, "")
            if per_doc.get(doc, 0) >= max_per_doc or text[:200] in seen:
                continue
            per_doc[doc] = per_doc.get(doc, 0) + 1
            seen.add(text[:200])
            picked.append({
                "text": text,
                "source": self.titles[doc],
                "page": pages.get(row),
                "url": self.urls[doc],
                "score": round(scores[row], 5),
                "order": int(self.doc_orders[doc]),
            })
            if len(picked) >= k:
                break
        return picked  # du plus pertinent au moins pertinent (fraicheur comprise)


# ------------------------------------------------------------------ cycle de vie

_index: FastIndex | None = None
_state = {"status": "absent", "error": None}
_ensure_lock = threading.Lock()


def get() -> FastIndex | None:
    """L'index s'il est pret, sinon None (la recherche retombe alors sur Chroma)."""
    return _index


def status() -> dict:
    return {**_state, "count": _index.count if _index else None}


def ensure() -> None:
    """Charge l'index, ou le (re)construit s'il manque ou ne correspond plus a Chroma. Bloquant :
    a lancer dans un thread au demarrage."""
    global _index
    with _ensure_lock:
        try:
            count = rag.get_collection().count()
            INDEX_ROOT.mkdir(parents=True, exist_ok=True)
            with _FileLock(INDEX_ROOT / ".lock"):  # plusieurs processus : un seul construit, les autres attendent
                info = INDEX_DIR / "info.json"
                fresh = info.exists() and json.loads(info.read_text(encoding="utf-8")).get("count") == count
                if not fresh:
                    _state["status"] = "building"
                    logger.info("construction de l'index rapide (%d passages)…", count)
                    build()
            _state["status"] = "loading"
            started = time.time()
            index = FastIndex()
            index.search("taux de chomage", k=1)  # charge les pages utiles en memoire
            _index = index
            _state["status"] = "ready"
            logger.info("index rapide pret en %.1f s (%d passages)", time.time() - started, _index.count)
        except Exception as exc:
            _state.update(status="error", error=f"{type(exc).__name__}: {exc}")
            logger.exception("index rapide indisponible : recherche via Chroma")


class _FileLock:
    """Verrou exclusif entre processus (fcntl, Linux) ; sans effet ailleurs (developpement Windows)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self):
        try:
            import fcntl
        except ImportError:
            return self
        self.handle = open(self.path, "w")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc) -> None:
        if self.handle:
            import fcntl

            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
