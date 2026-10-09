"""voice.py : verifications des traductions et detection du wolof (sans appel au service)."""

import voice


def test_numbers_preserved():
    assert voice.numbers_preserved("Le taux est de 22,3 % en 2023.", "Taux bi mooy 22,3 % ci 2023.")
    assert not voice.numbers_preserved("Le taux est de 22,3 %.", "Taux bi mooy 22 %.")
    assert voice.numbers_preserved("18 126 390 habitants", "18126390 nit")


def test_looks_wolof():
    assert voice.looks_wolof("Nanga def ? Ñaata nit ñoo am ci Senegaal ?")
    assert not voice.looks_wolof("Quel est le taux de chômage en 2023 ?")
    assert not voice.looks_wolof("What is the unemployment rate?")


def test_french_ratio():
    assert voice._french_ratio("") == 0.0
    assert voice._french_ratio("le taux de la population") > 0.5
    assert voice._french_ratio("Nanga def ñaata nit") == 0.0


def test_acceptable_refuse_une_traduction_identique_ou_qui_perd_les_chiffres():
    source = "Le taux de chômage est de 22,3 % en 2023."
    assert not voice._acceptable(source, source)
    assert not voice._acceptable(source, "Taux bi mooy 20 % ci 2023.")


def test_truncate_coupe_en_fin_de_phrase():
    text = "Première phrase. Deuxième phrase un peu plus longue. Troisième."
    assert voice._truncate(text, 30) == "Première phrase."
    assert voice._truncate("court", 30) == "court"
