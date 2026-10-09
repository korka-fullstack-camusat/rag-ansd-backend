"""analytics.py : journal d'utilisation et statistiques du tableau de bord (base temporaire)."""

import time

import analytics


def _flush(timeout: float = 5.0) -> None:
    """Attend que le fil d'ecriture ait vide la file d'evenements."""
    deadline = time.time() + timeout
    while not analytics._queue.empty() and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)


def test_la_base_de_test_est_isolee():
    assert "ansd-tests-" in str(analytics.DB_PATH)


def test_log_event_et_statistiques():
    analytics.log_event("query", client_id="c1", session_id="s1", question="Taux de chômage ?", language="fr",
                        mode="text", answered=True, latency_ms=800, prompt_tokens=100, completion_tokens=20,
                        sources=["Rapport A"])
    analytics.log_event("query", client_id="c2", session_id="s2", question="Question sans données ?",
                        language="en", mode="voice", answered=False, latency_ms=400)
    analytics.log_event("details_open", client_id="c1", session_id="s1")
    _flush()

    stats = analytics._compute_stats(days=1)
    kpis = stats["kpis"]
    assert kpis["questions"] >= 2
    assert kpis["users"] >= 2
    assert kpis["no_data"] >= 1
    assert kpis["details_opened"] >= 1
    assert {"per_day", "per_hour", "languages", "top_questions", "unanswered"} <= stats.keys()
    assert len(stats["per_hour"]) == 24

    recent = analytics.recent_questions(days=1, limit=10, status="no_data")
    assert any(item["question"] == "Question sans données ?" for item in recent["items"])


def test_log_event_ne_leve_jamais(monkeypatch):
    monkeypatch.setattr(analytics._queue, "put_nowait", lambda *_: (_ for _ in ()).throw(RuntimeError("plein")))
    analytics.log_event("query", question="x")  # ne doit pas lever


def test_log_session_title():
    analytics.log_session_title("s1", "Chômage")
    analytics.log_session_title(None, "ignoré")  # sans session : rien a faire
    row = analytics._db().execute("SELECT title FROM session_titles WHERE session_id = 's1'").fetchone()
    assert row is not None and row[0] == "Chômage"
