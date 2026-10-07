"""Journal d'utilisation de l'assistant + agregats pour le tableau de bord admin.

Base SQLite dans storage/ (volume Docker deja persistant). On enregistre,
pour chaque question : date, texte, langue, mode (ecrit/vocal), reponse
obtenue ou non, temps de reponse, tokens, publications citees. Les
utilisateurs ne sont identifies que par un identifiant anonyme aleatoire
genere par leur navigateur (aucun nom, e-mail ni adresse IP).
"""

import json
import queue
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

from config import ROOT_DIR

DB_PATH = ROOT_DIR / "storage" / "analytics.sqlite3"

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
        _conn.row_factory = sqlite3.Row
        # WAL : lectures (tableau de bord) et ecritures concurrentes, y compris
        # depuis plusieurs processus du backend.
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                type TEXT NOT NULL,            -- query | explain (preparation) | details_open | listen
                client_id TEXT,
                session_id TEXT,
                question TEXT,
                language TEXT,
                mode TEXT,                     -- text | voice
                answered INTEGER,              -- 1 reponse, 0 pas de donnees
                error INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                sources TEXT,                  -- JSON : titres des publications citees
                cached INTEGER NOT NULL DEFAULT 0 -- 1 = reponse servie depuis le cache
            );
            CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
            CREATE INDEX IF NOT EXISTS events_type_ts ON events(type, ts);
            CREATE TABLE IF NOT EXISTS session_titles (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                ts REAL NOT NULL
            );
            """
        )
        # Bases creees avant l'ajout de la colonne `cached`.
        columns = {row[1] for row in _conn.execute("PRAGMA table_info(events)")}
        if "cached" not in columns:
            _conn.execute("ALTER TABLE events ADD COLUMN cached INTEGER NOT NULL DEFAULT 0")
            _conn.commit()
    return _conn


# Ecritures en arriere-plan : une requete ne fait que deposer son evenement
# dans une file (instantane) ; un fil dedie les insere par lots.
_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=100_000)
_writer_started = False

_INSERT = """INSERT INTO events (ts, type, client_id, session_id, question, language, mode,
   answered, error, latency_ms, prompt_tokens, completion_tokens, sources, cached)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


def _writer() -> None:
    while True:
        batch = [_queue.get()]
        try:
            while len(batch) < 500:
                batch.append(_queue.get_nowait())
        except queue.Empty:
            pass
        try:
            with _lock:
                _db().executemany(_INSERT, batch)
                _db().commit()
        except Exception:
            pass
        time.sleep(0.05)  # laisse s'accumuler un lot sous forte charge


def _ensure_writer() -> None:
    global _writer_started
    if not _writer_started:
        with _lock:
            if not _writer_started:
                threading.Thread(target=_writer, name="analytics-writer", daemon=True).start()
                _writer_started = True


def log_event(
    type: str,
    *,
    client_id: str | None = None,
    session_id: str | None = None,
    question: str | None = None,
    language: str | None = None,
    mode: str | None = None,
    answered: bool | None = None,
    error: bool = False,
    cached: bool = False,
    latency_ms: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    sources: list[str] | None = None,
) -> None:
    """N'echoue et ne bloque jamais : un probleme de journal ne doit pas casser
    ni ralentir une reponse (file pleine = evenement ignore)."""
    try:
        _ensure_writer()
        _queue.put_nowait(
            (
                time.time(), type, _clip(client_id, 64), _clip(session_id, 64), _clip(question, 2000),
                language, mode if mode in ("text", "voice") else None,
                None if answered is None else int(answered), int(error), latency_ms,
                prompt_tokens, completion_tokens, json.dumps(sources or [], ensure_ascii=False), int(cached),
            )
        )
    except Exception:
        pass


def log_session_title(session_id: str | None, title: str) -> None:
    if not session_id:
        return
    try:
        with _lock:
            _db().execute(
                "INSERT OR REPLACE INTO session_titles (session_id, title, ts) VALUES (?, ?, ?)",
                (_clip(session_id, 64), title, time.time()),
            )
            _db().commit()
    except Exception:
        pass


def _clip(value: str | None, size: int) -> str | None:
    return value[:size] if value else value


def _since(days: int | None) -> float:
    return 0.0 if not days else time.time() - days * 86400


def _pct(part: int, total: int) -> float:
    return round(100 * part / total, 1) if total else 0.0


_stats_cache: dict[int | None, tuple[float, dict]] = {}
STATS_CACHE_SECONDS = 30


def stats(days: int | None) -> dict:
    """Tous les agregats du tableau de bord, sur les `days` derniers jours
    (None = depuis le debut). Calcules par SQLite (pas de chargement des
    lignes en memoire) et gardes 30 s : le cout ne depend pas du nombre de
    visites du tableau de bord."""
    cached = _stats_cache.get(days)
    if cached and time.time() - cached[0] < STATS_CACHE_SECONDS:
        return cached[1]
    result = _compute_stats(days)
    _stats_cache[days] = (time.time(), result)
    return result


def _compute_stats(days: int | None) -> dict:
    since = _since(days)
    ok = "type = 'query' AND error = 0 AND ts >= :since"
    p = {"since": since}
    with _lock:
        db = _db()
        one = lambda sql: db.execute(sql, p).fetchone()  # noqa: E731
        k = one(
            f"""SELECT COUNT(*) AS questions,
                       SUM(answered = 1) AS answered,
                       SUM(answered = 0) AS no_data,
                       SUM(mode = 'voice') AS voice,
                       SUM(cached) AS cached,
                       COUNT(DISTINCT client_id) AS users,
                       COUNT(DISTINCT session_id) AS sessions,
                       AVG(latency_ms) AS avg_latency,
                       COUNT(latency_ms) AS n_latency,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens
                FROM events WHERE {ok}"""
        )
        all_queries, errors = one(
            "SELECT COUNT(*), COALESCE(SUM(error), 0) FROM events WHERE type = 'query' AND ts >= :since"
        )
        p90 = None
        if k["n_latency"]:
            p90 = db.execute(
                f"SELECT latency_ms FROM events WHERE {ok} AND latency_ms IS NOT NULL "
                "ORDER BY latency_ms LIMIT 1 OFFSET :off",
                {**p, "off": int(0.9 * (k["n_latency"] - 1))},
            ).fetchone()[0]
        # Clics reels (« Voir plus » deplie, « Écouter ») — l'explication elle-meme
        # est preparee en arriere-plan avant tout clic, elle ne compte donc pas.
        interactions = dict(
            db.execute(
                "SELECT type, COUNT(*) FROM events WHERE type IN ('details_open', 'listen') AND ts >= :since GROUP BY type",
                p,
            ).fetchall()
        )
        day_rows = db.execute(
            f"""SELECT date(ts, 'unixepoch') AS d, COUNT(*) AS total, SUM(answered = 1) AS answered
                FROM events WHERE {ok} GROUP BY d""",
            p,
        ).fetchall()
        first_ts = one(f"SELECT MIN(ts) FROM events WHERE {ok}")[0]
        hour_rows = db.execute(
            f"SELECT CAST(strftime('%H', ts, 'unixepoch') AS INTEGER) AS h, COUNT(*) FROM events WHERE {ok} GROUP BY h",
            p,
        ).fetchall()
        languages = db.execute(
            f"SELECT language, COUNT(*) AS n FROM events WHERE {ok} GROUP BY language ORDER BY n DESC", p
        ).fetchall()
        modes = db.execute(
            f"SELECT COALESCE(mode, 'text') AS m, COUNT(*) AS n FROM events WHERE {ok} GROUP BY m ORDER BY n DESC", p
        ).fetchall()
        topics = db.execute(
            "SELECT title, COUNT(*) AS n FROM session_titles WHERE ts >= :since GROUP BY title ORDER BY n DESC LIMIT 10",
            p,
        ).fetchall()
        sources = db.execute(
            """SELECT j.value AS title, COUNT(*) AS n
                FROM events AS e, json_each(e.sources) AS j
                WHERE e.type = 'query' AND e.error = 0 AND e.ts >= :since AND e.answered = 1
                GROUP BY j.value ORDER BY n DESC LIMIT 10""",
            p,
        ).fetchall()

        def top_questions(extra: str) -> list[dict]:
            rows = db.execute(
                f"""SELECT lower(trim(rtrim(trim(question), '?'))) AS k, MAX(question) AS q,
                           COUNT(*) AS n, MAX(ts) AS last
                    FROM events WHERE {ok} {extra} AND question IS NOT NULL
                    GROUP BY k ORDER BY n DESC, last DESC LIMIT 10""",
                p,
            ).fetchall()
            return [{"question": r["q"], "count": r["n"], "last_ts": r["last"]} for r in rows]

        top = top_questions("")
        unanswered = top_questions("AND answered = 0")

    questions = k["questions"] or 0
    answered = k["answered"] or 0

    # Serie quotidienne continue (jours sans activite a 0), en heure UTC (= Dakar).
    today = datetime.now(timezone.utc).date()
    if days:
        start = today - timedelta(days=days - 1)
    elif first_ts:
        start = datetime.fromtimestamp(first_ts, timezone.utc).date()
    else:
        start = today
    by_day = {r["d"]: r for r in day_rows}
    per_day = []
    d = start
    while d <= today:
        r = by_day.get(d.isoformat())
        total = r["total"] if r else 0
        done = (r["answered"] or 0) if r else 0
        per_day.append({"date": d.isoformat(), "total": total, "answered": done, "no_data": total - done})
        d += timedelta(days=1)
    hours = dict(hour_rows)

    return {
        "period_days": days,
        "generated_at": time.time(),
        "kpis": {
            "questions": questions,
            "users": k["users"] or 0,
            "sessions": k["sessions"] or 0,
            "answer_rate": _pct(answered, questions),
            "no_data": k["no_data"] or 0,
            "voice_share": _pct(k["voice"] or 0, questions),
            "avg_latency_ms": round(k["avg_latency"]) if k["avg_latency"] is not None else None,
            "p90_latency_ms": p90,
            "cache_rate": _pct(k["cached"] or 0, questions),
            "details_opened": interactions.get("details_open", 0),
            "details_rate": _pct(interactions.get("details_open", 0), answered),
            "listens": interactions.get("listen", 0),
            "errors": errors,
            "error_rate": _pct(errors, all_queries),
            "prompt_tokens": k["prompt_tokens"],
            "completion_tokens": k["completion_tokens"],
        },
        "per_day": per_day,
        "per_hour": [{"hour": h, "total": hours.get(h, 0)} for h in range(24)],
        "languages": [{"language": r["language"], "total": r["n"]} for r in languages],
        "modes": [{"mode": r["m"], "total": r["n"]} for r in modes],
        "topics": [{"topic": r["title"], "total": r["n"]} for r in topics],
        "top_questions": top,
        "unanswered": unanswered,
        "top_sources": [{"title": r["title"], "total": r["n"]} for r in sources],
    }


def recent_questions(days: int | None, limit: int = 50, offset: int = 0, status: str | None = None) -> dict:
    since = _since(days)
    where = "type = 'query' AND ts >= ?"
    params: list = [since]
    if status == "answered":
        where += " AND error = 0 AND answered = 1"
    elif status == "no_data":
        where += " AND error = 0 AND answered = 0"
    elif status == "error":
        where += " AND error = 1"
    with _lock:
        db = _db()
        total = db.execute(f"SELECT COUNT(*) FROM events WHERE {where}", params).fetchone()[0]
        rows = db.execute(
            f"""SELECT ts, question, language, mode, answered, error, latency_ms
                FROM events WHERE {where} ORDER BY ts DESC LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()
    return {"total": total, "items": [dict(r) for r in rows]}
