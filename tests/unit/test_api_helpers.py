"""Fonctions utilitaires de api.py (sans serveur)."""

import api


def test_document_id_stable():
    url = "https://www.ansd.sn/sites/default/files/2026-07/REPERES.pdf"
    assert api.document_id(url) == api.document_id(url)
    assert len(api.document_id(url)) == 16
    assert api.document_id(url) != api.document_id(url + "x")


def test_publication_date():
    assert api.publication_date("https://www.ansd.sn/sites/default/files/2023-07/BULLETIN.pdf") == "2023-07"
    assert api.publication_date("https://example.org/doc.pdf") == ""


HITS = [
    {"source": "Rapport 2023", "page": 1, "text": "En 2023, le taux de chômage élargi est de 22,3 %."},
    {"source": "Rapport 2022", "page": 2, "text": "En 2022, le taux était de 21,0 %."},
    {"source": "Note", "page": 3, "text": "Population totale : 18 126 390 habitants."},
]


def test_filter_by_years():
    assert [h["source"] for h in api.filter_by_years("Taux de chômage en 2023 ?", HITS)] == ["Rapport 2023"]
    assert api.filter_by_years("Taux de chômage ?", HITS) == HITS
    assert api.filter_by_years("Taux en 1990 ?", HITS) == []


def test_figures_found_in_context():
    assert api.figures_found_in_context("Le taux est de 22,3 % en 2023.", HITS, "Taux en 2023 ?")
    assert api.figures_found_in_context("La population est de 18 126 390 habitants.", HITS)
    assert not api.figures_found_in_context("Le taux est de 25,1 %.", HITS)
    assert not api.figures_found_in_context("Information non disponible.", HITS)  # sans chiffre : non verifie


def test_is_empty_answer():
    assert api._is_empty_answer("Le taux n'est pas explicitement indiqué dans les extraits.", "Taux ?")
    assert not api._is_empty_answer("Le taux est de 22,3 %.", "Taux ?")


def test_pick_sources_garde_les_extraits_qui_contiennent_les_chiffres():
    assert api.pick_sources("Le taux est de 22,3 %.", HITS, used=[]) == [0]
    assert api.pick_sources("Réponse sans chiffre.", HITS, used=[2]) == [2]
    assert api.pick_sources("Réponse sans chiffre.", HITS, used=[]) == [0]
