"""Regles de rag.py appliquees sans le modele de langage : conversation courante,
demandes de conseil, mises en forme et relances de periode, references aux sources."""

import pytest

import rag


# ------------------------------------------------------------------ conversation courante

@pytest.mark.parametrize(
    "message, language, expected_start",
    [
        ("Bonjour", "fr", "Bonjour !"),
        ("bonjour !", "fr", "Bonjour !"),
        ("Hello", "en", "Hello!"),
        ("Merci beaucoup", "fr", "Avec plaisir"),
        ("Nanga def", "wo", "Nanga def"),
        ("Au revoir", "fr", "Au revoir"),
    ],
)
def test_small_talk_reply_reconnait_les_salutations(message, language, expected_start):
    assert rag.small_talk_reply(message, language).startswith(expected_start)


def test_small_talk_reply_langue_sans_traduction_repond_en_francais():
    assert rag.small_talk_reply("Bonjour", "ff").startswith("Bonjour !")


@pytest.mark.parametrize(
    "question",
    [
        "Quel est le taux de chômage en 2023 ?",
        "Bonjour, quel est le taux d'inflation au Sénégal en 2025 selon l'ANSD ?",
    ],
)
def test_small_talk_reply_ignore_les_vraies_questions(question):
    assert rag.small_talk_reply(question, "fr") is None


# ------------------------------------------------------------------ demandes de conseil

@pytest.mark.parametrize(
    "question",
    [
        "Qu'est ce que vous me conseillez en tant que debutant pour la recuperation des donnees ?",
        "Donner moi une etape a suivre sur mes recherches et aussi pour chaque ressources donner les acces au sources",
        "Où trouver les données sur l'emploi ?",
        "Comment télécharger les bulletins mensuels ?",
        "Quels conseils pour analyser les données de l'EDS ?",
        "Par où commencer ?",
        "How can I access ANSD survey data?",
    ],
)
def test_guidance_question_reconnait_les_demandes_de_conseil(question):
    assert rag.guidance_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "Quel est le taux de chômage en 2023 ?",
        "Quelle a été l'inflation en 2025 ?",
        "Combien de ménages au Sénégal ?",
        "Conseil économique social et environnemental budget 2023",
        "Comment est calculé l'IHPC ?",
        "Comment évolue le PIB ?",
        "",
    ],
)
def test_guidance_question_laisse_passer_les_questions_chiffrees(question):
    assert not rag.guidance_question(question)


def test_guidance_follow_up_apres_une_demande_de_conseil():
    history = [{"question": "Donnez-moi les étapes à suivre pour mes recherches", "answer": "1. …", "sources": []}]
    assert rag.guidance_follow_up("Sur la santé des enfants", history)
    # Une question chiffree reprend le circuit normal, meme juste apres le guide.
    assert not rag.guidance_follow_up("Quel est le taux de chômage en 2023 ?", history)


def test_guidance_follow_up_sans_guide_precedent():
    history = [{"question": "Quel est le taux de chômage en 2023 ?", "answer": "22,3 %", "sources": []}]
    assert not rag.guidance_follow_up("Sur la santé des enfants", history)
    assert not rag.guidance_follow_up("Sur la santé des enfants", [])


# ------------------------------------------------------------------ mise en forme et periode

@pytest.mark.parametrize(
    "message, clause",
    [
        ("mets ça dans un tableau", "sous forme de tableau"),
        ("point par point", "point par point (liste à puces)"),
        ("plus court stp", "en une seule phrase"),
        ("plus de détails", "de façon plus détaillée"),
    ],
)
def test_format_only_clause_reconnait_les_mises_en_forme(message, clause):
    assert rag.format_only_clause(message) == clause


def test_format_only_clause_ignore_une_nouvelle_question():
    assert rag.format_only_clause("explique-moi le PIB du Sénégal") is None


def test_with_format_remplace_l_ancienne_consigne():
    first = rag.with_format("Quel est le PIB en 2024 ?", "sous forme de tableau")
    assert first == "Quel est le PIB en 2024 — réponds sous forme de tableau."
    second = rag.with_format(first, "en une seule phrase")
    assert second == "Quel est le PIB en 2024 — réponds en une seule phrase."


@pytest.mark.parametrize(
    "message, periods",
    [
        ("et pour 2025 ?", ["2025"]),
        ("je veux juste 2026", ["2026"]),
        ("et au T2 2024 ?", ["2024", "T2"]),
    ],
)
def test_period_only_reconnait_les_relances_de_periode(message, periods):
    assert rag.period_only(message) == periods


def test_period_only_ignore_un_nouveau_sujet():
    assert rag.period_only("et le chômage en 2025 ?") is None
    assert rag.period_only("et le chômage ?") is None


def test_with_period_remplace_l_annee():
    assert rag.with_period("Quel est le taux de chômage en 2023 ?", ["2025"]) == "Quel est le taux de chômage en 2025 ?"


def test_with_period_precise_la_periode_si_pas_d_annee():
    assert rag.with_period("Quel est le taux de chômage ?", ["T2"]) == "Quel est le taux de chômage — période : T2 ?"


# ------------------------------------------------------------------ sources de la reponse

def test_split_used_sources_retire_la_ligne_sources():
    answer, used = rag.split_used_sources("Le taux est de 22,3 %.\nSOURCES: 2, 5, 2, 9", n_hits=6)
    assert answer == "Le taux est de 22,3 %."
    assert used == [1, 4]  # indices 0-based, sans doublon, hors limites ignores


def test_split_used_sources_sans_ligne_sources():
    assert rag.split_used_sources("  Réponse simple.  ", 3) == ("Réponse simple.", [])


HITS = [
    {"source": "Rapport A", "page": 3, "url": "https://www.ansd.sn/a.pdf"},
    {"source": "Rapport B", "page": 7, "url": "https://www.ansd.sn/b.pdf"},
    {"source": "Rapport A", "page": 3, "url": "https://www.ansd.sn/a.pdf"},
]


def test_link_references_numerote_les_sources_dans_l_ordre():
    text, sources = rag.link_references("Premier fait [2]. Second fait [1].", HITS)
    assert text == "Premier fait[[1]]. Second fait[[2]]."
    assert [(s["n"], s["title"], s["page"]) for s in sources] == [(1, "Rapport B", 7), (2, "Rapport A", 3)]


def test_link_references_fusionne_les_extraits_d_une_meme_page():
    text, sources = rag.link_references("Fait [1]. Autre fait [3].", HITS)
    assert len(sources) == 1
    assert text.count("[[1]]") == 1  # une seule reference en fin de suite de passages de la meme source


def test_link_references_retire_les_numeros_inconnus():
    text, sources = rag.link_references("Fait [9].", HITS)
    assert text == "Fait."
    assert sources == []


def test_link_references_accepte_les_doubles_crochets_du_modele():
    text, _ = rag.link_references("A « X »[[1]] et B [2].", HITS[:2])
    assert text == "A « X »[[1]] et B[[2]]."
    assert "[[[" not in text


# ------------------------------------------------------------------ portee de la question

def test_parse_scope_pages_et_document(monkeypatch):
    monkeypatch.setattr(rag, "known_sources", lambda: ["ICAS T2 2026", "ICAS T1 2026", "Repères statistiques"])
    scope = rag.parse_scope("Que dit l'ICAS T2 2026 aux pages 3 à 5 ?")
    assert scope.sources == ["ICAS T2 2026"]
    assert scope.pages == [3, 4, 5]
    assert scope.describe() == "document « ICAS T2 2026 », pages 3, 4, 5"


def test_parse_scope_sans_portee(monkeypatch):
    monkeypatch.setattr(rag, "known_sources", lambda: ["Repères statistiques"])
    scope = rag.parse_scope("Quel est le taux de chômage ?")
    assert scope.sources == [] and scope.pages == []


def test_build_prompt_contient_le_contexte_et_la_consigne(monkeypatch):
    monkeypatch.setattr(rag, "known_sources", lambda: [])
    hits = [{"source": "Rapport A", "page": 3, "text": "Le taux est de 22,3 %."}]
    prompt = rag.build_prompt("Quel est le taux ?", hits, "en")
    assert "[Source 1 - Rapport A, p.3]" in prompt
    assert "Reponds en anglais" in prompt
    assert rag.NO_DATA_MARKER in prompt


def test_clean_title():
    completion = {"choices": [{"message": {"content": "  « chômage des jeunes au sénégal en 2024 ». "}}]}
    assert rag._clean_title(completion) == "Chômage des jeunes au"
