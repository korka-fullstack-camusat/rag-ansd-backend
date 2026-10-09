"""figures.py : un meme nombre ecrit de plusieurs facons est reconnu comme identique."""

import pytest

import figures


@pytest.mark.parametrize(
    "token, canonical",
    [
        ("18 126 390", "18126390"),
        ("18,126,390", "18126390"),
        ("18.126.390", "18126390"),
        ("20,3", "20.3"),
        ("1.6", "1.6"),
        ("1 234,5", "1234.5"),
    ],
)
def test_canon(token, canonical):
    assert figures.canon(token) == canonical


def test_figures_ignore_les_annees_et_les_chiffres_isoles():
    assert figures.figures("En 2023, le taux était de 22,3 % pour 5 régions et 18 126 390 habitants.") == [
        "22.3",
        "18126390",
    ]


def test_figure_set_compare_les_formats_francais_et_anglais():
    assert figures.figure_set("1,6 %") == figures.figure_set("1.6%")


def test_locate():
    text = "Population : 18 126 390 habitants"
    assert figures.locate(text, "18126390") == text.index("18")
    assert figures.locate(text, "42") == -1
