"""Comparaison des chiffres d'une reponse avec ceux des publications, quel que soit leur format.

« 18 126 390 », « 18,126,390 » et « 18126390 » sont le meme nombre ; « 1,6 % » (francais) et
« 1.6%» (anglais) aussi. Les nombres sont ramenes a une forme canonique (chiffres, point decimal)
des deux cotes avant d'etre compares. Les annees seules (2025) ne comptent pas comme des chiffres.
"""

import re

_SPACES = "\\s\u00a0\u202f"
# Nombre ecrit avec groupes de milliers (espaces, virgules ou points) et partie decimale eventuelle.
_TOKEN_RE = re.compile(
    rf"\d{{1,3}}(?:[{_SPACES},.]\d{{3}})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def canon(token: str) -> str:
    """Forme canonique d'un nombre ecrit : « 18 126 390 » -> « 18126390 », « 20,3 » -> « 20.3 »."""
    t = re.sub(rf"[{_SPACES}]", "", token)
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", t):  # 18,126,390 ; 18.126.390 ; 2,345 -> milliers
        return re.sub(r"[.,]", "", t)
    m = re.fullmatch(r"(\d{1,3}(?:[.,]\d{3})*)[.,](\d+)", t)  # milliers puis decimales
    if m:
        return re.sub(r"[.,]", "", m.group(1)) + "." + m.group(2)
    return t.replace(",", ".")


def numbers(text: str) -> list[tuple[str, int]]:
    """[(forme canonique, position)] des nombres du texte, hors annees seules et chiffres isoles."""
    out = []
    for m in _TOKEN_RE.finditer(text):
        c = canon(m.group(0))
        digits = c.replace(".", "")
        if len(digits) > 1 and not _YEAR_RE.fullmatch(c):
            out.append((c, m.start()))
    return out


def figures(text: str) -> list[str]:
    return [c for c, _ in numbers(text)]


def figure_set(text: str) -> set[str]:
    return set(figures(text))


def locate(text: str, figure: str) -> int:
    """Position du nombre `figure` (canonique) dans `text`, ou -1."""
    for c, pos in numbers(text):
        if c == figure:
            return pos
    return -1
