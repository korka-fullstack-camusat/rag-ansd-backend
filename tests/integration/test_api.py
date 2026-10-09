"""Tests d'integration de l'API (FastAPI) : chaque route est appelee comme le fait le frontend,
avec la recherche documentaire et le modele de langage remplaces par des doublures. Aucun appel
reseau, aucun cout, resultats reproductibles."""

import json
import types

import pytest
from fastapi.testclient import TestClient

import agent
import api
import cache

URL_A = "https://www.ansd.sn/sites/default/files/2024-03/ENES_T4_2023.pdf"
HITS = [
    {
        "source": "Enquête nationale sur l'emploi T4 2023",
        "page": 12,
        "url": URL_A,
        "text": "Au quatrième trimestre 2023, le taux de chômage élargi est de 22,3 %.",
        "score": 0.9,
    },
    {
        "source": "Repères statistiques 2022",
        "page": 4,
        "url": "https://www.ansd.sn/sites/default/files/2022-12/REPERES_2022.pdf",
        "text": "En 2022, le taux de chômage élargi était de 21,0 %.",
        "score": 0.7,
    },
]


def completion(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        "model": "modele-de-test",
    }


class Fakes:
    """Doublures enregistrant leurs appels."""

    def __init__(self):
        self.llm_answer = "Le taux de chômage élargi est de 22,3 % au quatrième trimestre 2023.\nSOURCES: 1"
        self.llm_calls: list[str] = []
        self.retrieve_calls: list[str] = []
        self.hits = HITS
        self.llm_error: Exception | None = None

    async def aretrieve(self, question, prefer=None, top_k=6):
        self.retrieve_calls.append(question)
        return self.hits

    async def acall_llm(self, question, hits, language="fr", previous=None, temperature=0.2):
        self.llm_calls.append(question)
        if self.llm_error:
            raise self.llm_error
        return completion(self.llm_answer)


@pytest.fixture
def fakes(monkeypatch):
    f = Fakes()
    monkeypatch.setattr(api, "store", cache.Store())  # cache et limites de debit neufs a chaque test
    monkeypatch.setattr(api, "corpus_version", lambda: 1)
    monkeypatch.setattr(api, "corpus_reply", lambda question, language: None)
    monkeypatch.setattr(api, "unknown_acronyms", lambda question: [])
    monkeypatch.setattr(api, "aretrieve", f.aretrieve)
    monkeypatch.setattr(api, "acall_llm", f.acall_llm)

    async def acondense(question, history):
        return question

    async def aguidance(question, language="fr", history=None):
        return (
            "1. Définissez votre question.\n2. Consultez les Repères statistiques[[1]].",
            [{"n": 1, "title": "Repères statistiques, avril 2026", "page": None, "url": "https://www.ansd.sn/r.pdf"}],
            "modele-de-test",
        )

    async def aexplain(question, answer, hits, language="fr"):
        return "Explication détaillée[[1]].", [{"n": 1, "title": hits[0]["source"], "page": hits[0]["page"], "url": hits[0]["url"]}]

    async def ashort_title(question):
        return "Chômage"

    monkeypatch.setattr(api, "acondense", acondense)
    monkeypatch.setattr(api, "aguidance", aguidance)
    monkeypatch.setattr(api, "aexplain", aexplain)
    monkeypatch.setattr(api, "ashort_title", ashort_title)
    monkeypatch.setattr(api, "ADMIN_TOKEN", "jeton-de-test")
    monkeypatch.setattr(agent, "AGENT_ENABLED", False)  # circuit a regles par defaut ; l'agent a ses propres tests
    api._retrieval_cache.clear()
    return f


@pytest.fixture
def client(fakes):
    # Sans « with » : le demarrage (chargement du modele d'embedding, index) n'est pas lance.
    return TestClient(api.app)


def ask(client, question, **extra):
    return client.post("/api/query", json={"question": question, "language": "fr", **extra})


# ------------------------------------------------------------------ sante et validation

def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "indexed_chunks": 1, "cache": "memory"}


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "x" * 2001}, {"question": "Taux ?", "language": "de"}, {}],
)
def test_query_refuse_une_requete_invalide(client, payload):
    assert client.post("/api/query", json=payload).status_code == 422


def test_cors_autorise_le_frontend(client):
    res = client.options(
        "/api/query",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ------------------------------------------------------------------ reponses chiffrees

def test_question_chiffree_reponse_avec_sources(client, fakes):
    res = ask(client, "Quel est le taux de chômage en 2023 ?")
    assert res.status_code == 200
    body = res.json()
    assert body["kind"] == "answer" and body["answered"] is True
    assert "22,3 %" in body["answer"] and "SOURCES" not in body["answer"]
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["url"] == URL_A and citation["page_start"] == 12
    assert citation["verified"] is True  # le chiffre de la reponse figure dans l'extrait
    assert body["sources_used"] == [api.document_id(URL_A)]
    assert body["usage"]["total_tokens"] == 150


def test_question_identique_servie_depuis_le_cache(client, fakes):
    ask(client, "Quel est le taux de chômage en 2023 ?")
    ask(client, "  quel est le TAUX de chomage en 2023 ")
    assert len(fakes.llm_calls) == 1


def test_relancer_ignore_le_cache(client, fakes):
    ask(client, "Quel est le taux de chômage en 2023 ?")
    ask(client, "Quel est le taux de chômage en 2023 ?", regenerate=True)
    assert len(fakes.llm_calls) == 2


def test_annee_absente_des_extraits_pas_de_donnees(client, fakes):
    body = ask(client, "Quel est le taux de chômage en 1990 ?").json()
    assert body["kind"] == "no_data" and body["answered"] is False
    assert body["answer"] == api.NO_DATA_MESSAGES["fr"]
    assert fakes.llm_calls == []  # aucun extrait de 1990 : le modele n'est pas appele


def test_modele_sans_reponse_pas_de_donnees(client, fakes):
    fakes.llm_answer = api.NO_DATA_MARKER
    body = ask(client, "Quel est le nombre de girafes au Sénégal ?").json()
    assert body["kind"] == "no_data"


def test_sigle_inconnu_pas_de_donnees(client, fakes, monkeypatch):
    monkeypatch.setattr(api, "unknown_acronyms", lambda question: ["DR"])
    body = ask(client, "Quel est le taux de DR ?").json()
    assert body["kind"] == "no_data"
    assert fakes.retrieve_calls == [] and fakes.llm_calls == []


def test_reponse_en_anglais(client, fakes):
    fakes.hits = []
    body = client.post("/api/query", json={"question": "What is the unemployment rate?", "language": "en"}).json()
    assert body["answer"] == api.NO_DATA_MESSAGES["en"]


def test_panne_du_modele_message_generique(client, fakes):
    fakes.llm_error = RuntimeError("fournisseur indisponible")
    res = ask(client, "Quel est le taux de chômage en 2023 ?")
    assert res.status_code == 502
    assert res.json()["detail"] == api.GENERIC_ERROR_MESSAGE


# ------------------------------------------------------------------ conversation et suites

def test_salutation_sans_recherche(client, fakes):
    body = ask(client, "Bonjour").json()
    assert body["kind"] == "chat"
    assert body["answer"].startswith("Bonjour !")
    assert fakes.retrieve_calls == [] and fakes.llm_calls == []


def test_salutation_en_wolof_sans_traduction(client, fakes):
    body = client.post("/api/query", json={"question": "Nanga def", "language": "wo"}).json()
    assert body["kind"] == "chat" and body["language"] == "wo"
    assert body["answer"].startswith("Nanga def")


def test_demande_de_mise_en_forme_reprend_le_sujet_precedent(client, fakes):
    history = [{"question": "Quel est le taux de chômage en 2023 ?", "answer": "22,3 %", "sources": []}]
    ask(client, "mets ça dans un tableau", history=history)
    assert fakes.llm_calls == ["Quel est le taux de chômage en 2023 — réponds sous forme de tableau."]


def test_relance_de_periode_reprend_le_sujet_precedent(client, fakes):
    history = [{"question": "Quel est le taux de chômage en 2023 ?", "answer": "22,3 %", "sources": []}]
    ask(client, "et pour 2022 ?", history=history)
    assert fakes.retrieve_calls[-1] == "Quel est le taux de chômage en 2022 ?"


# ------------------------------------------------------------------ guide de recherche

def test_demande_de_conseil_reponse_guide_avec_liens(client, fakes):
    body = ask(client, "Qu'est ce que vous me conseillez en tant que debutant pour la recuperation des donnees ?").json()
    assert body["kind"] == "guide" and body["answered"] is False
    assert "[[1]]" in body["answer"]
    assert body["sources"][0]["url"] == "https://www.ansd.sn/r.pdf"
    assert fakes.llm_calls == []  # pas de recherche de chiffres


def test_precision_apres_le_guide_continue_l_accompagnement(client, fakes):
    history = [{"question": "Donnez-moi les étapes à suivre pour mes recherches", "answer": "1. …", "sources": []}]
    body = ask(client, "Sur la santé des enfants", history=history).json()
    assert body["kind"] == "guide"
    assert body["standalone_question"] == "Donnez-moi les étapes à suivre pour mes recherches — Sur la santé des enfants"


# ------------------------------------------------------------------ agent conversationnel

AGENT_RESULT = {
    "answer": "Le taux de chômage élargi est de 22,3 % au T4 2023.",
    "citations": [
        {"document_title": HITS[0]["source"], "url": URL_A, "quote": HITS[0]["text"],
         "page_start": 12, "page_end": 12, "verified": True}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    "model": "agent-de-test",
    "searched": True,
}


def test_agent_reponse_avec_sources(client, fakes, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_ENABLED", True)
    calls = []

    async def run_agent(question, language="fr", history=None, temperature=0.2):
        calls.append(question)
        return AGENT_RESULT

    monkeypatch.setattr(agent, "run_agent", run_agent)
    body = ask(client, "Quel est le taux de chômage en 2023 ?").json()
    assert body["kind"] == "answer" and body["model"] == "agent-de-test"
    assert body["citations"][0]["document_id"] == api.document_id(URL_A)
    ask(client, "Quel est le taux de chômage en 2023 ?")
    assert len(calls) == 1  # seconde fois : cache


def test_agent_en_panne_repli_sur_le_circuit_a_regles(client, fakes, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_ENABLED", True)

    async def run_agent(*args, **kwargs):
        raise RuntimeError("agent indisponible")

    monkeypatch.setattr(agent, "run_agent", run_agent)
    body = ask(client, "Quel est le taux de chômage en 2023 ?").json()
    assert body["kind"] == "answer" and "22,3 %" in body["answer"]
    assert len(fakes.llm_calls) == 1


def _stream(client, question):
    res = client.post("/api/query/stream", json={"question": question, "language": "fr"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in res.text.splitlines() if line.strip()]


def test_flux_envoie_le_texte_puis_la_reponse_finale(client, fakes, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_ENABLED", True)

    async def agent_events(question, language="fr", history=None, temperature=0.2):
        yield {"type": "status", "text": "search"}
        yield {"type": "delta", "text": "Le taux "}
        yield {"type": "delta", "text": "est de 22,3 %."}
        yield {"type": "result", "data": AGENT_RESULT}

    monkeypatch.setattr(agent, "agent_events", agent_events)
    events = _stream(client, "Quel est le taux de chômage en 2023 ?")
    assert [e["type"] for e in events] == ["status", "delta", "delta", "done"]
    assert events[-1]["response"]["kind"] == "answer"


def test_flux_agent_en_panne_reinitialise_puis_repond(client, fakes, monkeypatch):
    monkeypatch.setattr(agent, "AGENT_ENABLED", True)

    async def agent_events(question, language="fr", history=None, temperature=0.2):
        yield {"type": "delta", "text": "Début de réponse"}
        raise RuntimeError("coupure")

    monkeypatch.setattr(agent, "agent_events", agent_events)
    events = _stream(client, "Quel est le taux de chômage en 2023 ?")
    assert [e["type"] for e in events] == ["delta", "reset", "done"]
    assert "22,3 %" in events[-1]["response"]["answer"]


def test_flux_sans_agent_une_seule_reponse(client, fakes):
    events = _stream(client, "Bonjour")
    assert [e["type"] for e in events] == ["done"]
    assert events[0]["response"]["kind"] == "chat"


def test_finalize_de_l_agent_retient_les_sources_citees():
    run = types.SimpleNamespace(passages={1: HITS[0], 2: HITS[1]})
    result = agent.finalize("Le taux est de 22,3 % [1].\nSOURCES: 1", run, "Taux de chômage en 2023 ?")
    assert result["answer"] == "Le taux est de 22,3 %."
    assert [c["document_title"] for c in result["citations"]] == [HITS[0]["source"]]
    assert result["citations"][0]["verified"] is True


def test_finalize_retrouve_la_source_d_un_chiffre_non_reference():
    run = types.SimpleNamespace(passages={1: HITS[1], 2: HITS[0]})
    result = agent.finalize("Le taux est de 22,3 %.", run, "Taux ?")
    assert [c["document_title"] for c in result["citations"]] == [HITS[0]["source"]]


# ------------------------------------------------------------------ autres routes

def test_explication_detaillee(client, fakes):
    res = client.post(
        "/api/explain",
        json={"question": "Taux de chômage 2023 ?", "answer": "22,3 %", "language": "fr",
              "sources": [{"title": HITS[0]["source"], "page": 12}]},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["details"] == "Explication détaillée[[1]]."
    assert body["sources"][0]["url"] == URL_A


def test_explication_sans_extrait_404(client, fakes):
    fakes.hits = []
    res = client.post("/api/explain", json={"question": "x ?", "answer": "y", "language": "fr"})
    assert res.status_code == 404


def test_titre_de_discussion(client):
    res = client.post("/api/title", json={"question": "Quel est le taux de chômage des jeunes ?"})
    assert res.status_code == 200 and res.json() == {"title": "Chômage"}


def test_suivi_des_interactions(client):
    assert client.post("/api/track", json={"event": "details_open"}).status_code == 204
    assert client.post("/api/track", json={"event": "inconnu"}).status_code == 422


def test_limite_de_debit_par_client(client):
    headers = {"X-Client-Id": "client-presse"}
    codes = [
        client.post("/api/query", json={"question": "Bonjour"}, headers=headers).status_code
        for _ in range(api.RATE_LIMIT_PER_MINUTE + 1)
    ]
    assert codes[:-1] == [200] * api.RATE_LIMIT_PER_MINUTE
    assert codes[-1] == 429


def test_publication_redirige_vers_ansd(client, monkeypatch):
    monkeypatch.setattr(api, "read_manifest", lambda: [{"url": URL_A, "title": "ENES"}])
    res = client.get(f"/files/{api.document_id(URL_A)}", follow_redirects=False)
    assert res.status_code == 307 and res.headers["location"] == URL_A
    assert client.get("/files/inconnu", follow_redirects=False).status_code == 404


def test_liste_des_publications(client, monkeypatch):
    monkeypatch.setattr(api, "read_manifest", lambda: [
        {"url": URL_A, "title": "ENES T4 2023", "extension": "pdf", "size": "2 Mo"},
        {"url": "https://www.ansd.sn/autre.pdf", "title": "Non indexé"},
    ])
    monkeypatch.setattr(api, "indexed_urls", lambda: {URL_A})
    api._sources_cache["version"] = None
    body = client.get("/api/sources").json()
    assert [s["title"] for s in body] == ["ENES T4 2023"]
    assert body[0]["publication_date"] == "2024-03" and body[0]["description"] == "PDF · 2 Mo"


# ------------------------------------------------------------------ administration

def test_admin_sans_jeton_refuse(client):
    assert client.get("/api/admin/stats").status_code == 401
    assert client.get("/api/admin/stats", headers={"Authorization": "Bearer mauvais"}).status_code == 401


def test_admin_desactive_sans_jeton_configure(client, monkeypatch):
    monkeypatch.setattr(api, "ADMIN_TOKEN", "")
    assert client.get("/api/admin/stats").status_code == 503


def test_admin_statistiques_et_journal(client):
    headers = {"Authorization": "Bearer jeton-de-test"}
    stats = client.get("/api/admin/stats?days=7", headers=headers)
    assert stats.status_code == 200 and "kpis" in stats.json()
    questions = client.get("/api/admin/questions?status=no_data&limit=5", headers=headers)
    assert questions.status_code == 200 and {"total", "items"} <= questions.json().keys()


# ------------------------------------------------------------------ voix

def test_lecture_audio_langue_sans_voix(client, monkeypatch):
    monkeypatch.setattr(api.voice, "_api_key", lambda: "cle-de-test")
    res = client.post("/api/voice/speak", json={"text": "Bonjour", "language": "fr"})
    assert res.status_code == 501


def test_lecture_audio_renvoie_le_son(client, monkeypatch):
    async def synthesize(text, language):
        return b"RIFF....", "audio/wav"

    monkeypatch.setattr(api.voice, "synthesize", synthesize)
    res = client.post("/api/voice/speak", json={"text": "Nanga def", "language": "wo"})
    assert res.status_code == 200 and res.content == b"RIFF...." and res.headers["content-type"] == "audio/wav"


def test_transcription(client, monkeypatch):
    async def transcribe(audio, filename, language):
        assert audio == b"son" and language == "wo"
        return "Ñaata nit ñoo am ci Senegaal ?"

    monkeypatch.setattr(api.voice, "transcribe", transcribe)
    res = client.post("/api/voice/transcribe", files={"audio": ("q.webm", b"son", "audio/webm")}, data={"language": "wo"})
    assert res.status_code == 200 and res.json() == {"text": "Ñaata nit ñoo am ci Senegaal ?"}
