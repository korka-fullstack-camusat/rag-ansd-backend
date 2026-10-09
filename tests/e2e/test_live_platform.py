"""Tests de bout en bout : la plateforme lancee (docker compose up -d, backend et frontend)
repond-elle correctement, avec la vraie base documentaire et le vrai modele ?

Lancement : ./run_tests.sh e2e  (environ 2 minutes, quelques centimes d'appels au modele).
Les reponses du modele varient : on verifie la forme et le bon type de reponse, pas le texte exact.
"""

import json
import os
import uuid

import httpx
import pytest

API = os.environ.get("E2E_API_URL", "http://localhost:8000").rstrip("/")
FRONTEND = os.environ.get("E2E_FRONTEND_URL", "http://localhost:3000").rstrip("/")
TIMEOUT = httpx.Timeout(180, connect=10)

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def http():
    # Identifiant propre aux tests : leurs questions sont reperables dans le tableau de bord.
    headers = {"X-Client-Id": f"e2e-{uuid.uuid4()}"}
    with httpx.Client(timeout=TIMEOUT, headers=headers) as client:
        try:
            client.get(f"{API}/api/health").raise_for_status()
        except httpx.HTTPError as exc:
            pytest.skip(f"API injoignable sur {API} ({exc}) : lancez « docker compose up -d »")
        yield client


def ask(http, question, language="fr", **extra):
    res = http.post(f"{API}/api/query", json={"question": question, "language": language, **extra})
    assert res.status_code == 200, res.text
    return res.json()


def test_api_en_bonne_sante_et_corpus_indexe(http):
    body = http.get(f"{API}/api/health").json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 1000, "la base Chroma semble vide (storage/ manquant ?)"


def test_liste_des_publications(http):
    sources = http.get(f"{API}/api/sources").json()
    assert len(sources) > 100
    assert {"id", "title", "publisher", "publication_date"} <= sources[0].keys()


def test_question_chiffree_avec_sources_officielles(http):
    body = ask(http, "Quel est le taux de chômage au Sénégal en 2023 ?", regenerate=True)
    assert body["kind"] == "answer" and body["answered"] is True, body["answer"]
    assert any(ch.isdigit() for ch in body["answer"])
    assert body["citations"], "une reponse chiffree doit citer au moins une source"
    for citation in body["citations"]:
        assert citation["url"].startswith("https://www.ansd.sn/")
        assert citation["page_start"] is None or citation["page_start"] >= 1


def test_question_en_anglais(http):
    body = ask(http, "What was the consumer price index trend in 2025?", language="en")
    assert body["language"] == "en"
    assert body["kind"] in {"answer", "no_data", "chat"}
    assert body["answer"].strip()


def test_salutation(http):
    body = ask(http, "Bonjour")
    assert body["kind"] == "chat" and body["citations"] == []
    assert body["answer"].strip()


def test_hors_sujet_sans_chiffre_invente(http):
    body = ask(http, "Combien de girafes vivent sur la planète Mars en 2023 ?")
    assert body["answered"] is False and body["citations"] == []


def test_question_de_suite(http):
    first = ask(http, "Quel est le taux de chômage au Sénégal en 2023 ?")
    history = [{"question": first["question"], "answer": first["answer"],
                "sources": [{"title": c["document_title"], "page": c["page_start"]} for c in first["citations"]]}]
    body = ask(http, "et pour 2022 ?", history=history)
    assert body["answer"].strip()
    assert body["kind"] in {"answer", "no_data", "chat"}


def test_reponse_en_flux(http):
    events = []
    with http.stream("POST", f"{API}/api/query/stream",
                     json={"question": "Quelle est la population du Sénégal ?", "language": "fr"}) as res:
        assert res.status_code == 200
        for line in res.iter_lines():
            if line.strip():
                events.append(json.loads(line))
    assert events[-1]["type"] == "done", events[-1]
    assert events[-1]["response"]["answer"].strip()


def test_explication_detaillee(http):
    first = ask(http, "Quel est le taux de chômage au Sénégal en 2023 ?")
    if not first["answered"]:
        pytest.skip("pas de reponse chiffree a detailler")
    res = http.post(f"{API}/api/explain", json={
        "question": first["question"], "answer": first["answer"], "language": "fr",
        "sources": [{"title": c["document_title"], "page": c["page_start"]} for c in first["citations"]],
    })
    assert res.status_code == 200, res.text
    assert len(res.json()["details"]) > len(first["answer"]) / 2


def test_titre_de_discussion(http):
    res = http.post(f"{API}/api/title", json={"question": "Quel est le taux de chômage des jeunes en 2023 ?"})
    assert res.status_code == 200
    assert 0 < len(res.json()["title"].split()) <= 4


def test_lien_vers_une_publication(http):
    sources = http.get(f"{API}/api/sources").json()
    res = http.get(f"{API}/files/{sources[0]['id']}", follow_redirects=False)
    assert res.status_code == 307 and res.headers["location"].startswith("https://www.ansd.sn/")


def test_admin_protege(http):
    assert http.get(f"{API}/api/admin/stats").status_code in {401, 503}
    token = os.environ.get("ADMIN_TOKEN")
    if token:
        res = http.get(f"{API}/api/admin/stats", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200 and "kpis" in res.json()


@pytest.mark.parametrize("path", ["/accueil", "/admin"])
def test_pages_du_frontend(path):
    try:
        res = httpx.get(f"{FRONTEND}{path}", timeout=30, follow_redirects=True)
    except httpx.HTTPError as exc:
        pytest.skip(f"frontend injoignable sur {FRONTEND} ({exc})")
    assert res.status_code == 200
    assert "<html" in res.text.lower()
