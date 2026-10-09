"""Voix : reconnaissance (ASR) et synthese (TTS) du wolof via Soynade.

Contrat avec le frontend (lib/api.ts) :
  POST /api/voice/transcribe  multipart « audio » (webm/ogg/mp4 du navigateur) + « language »
                              -> {"text": "..."}
  POST /api/voice/speak       {"text", "language"} -> audio (mp3)

Soynade n'accepte que wav/mp3/flac : l'enregistrement du navigateur est converti en wav
mono 16 kHz avec ffmpeg (binaire fourni par le paquet imageio-ffmpeg). Seules les langues de
SOYNADE_*_LANGUAGES sont traitees ; pour les autres le frontend lit la reponse avec la voix du
navigateur (francais, anglais). Ni l'audio ni le texte ne sont journalises.

Variables d'environnement : SOYNADE_API_KEY (obligatoire), SOYNADE_BASE_URL.
"""

import asyncio
import logging
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path

import httpx

logger = logging.getLogger("ansd-voice")

SOYNADE_BASE_URL = os.environ.get("SOYNADE_BASE_URL", "https://api.soynade.ai").rstrip("/")
ASR_LANGUAGES = {"wo", "fr", "en"}  # langues du modele de reconnaissance Soynade
TTS_LANGUAGES = {"wo"}  # synthese : seul le wolof est documente chez Soynade
TTS_MAX_CHARS = int(os.environ.get("SOYNADE_TTS_MAX_CHARS", "500"))
MAX_UPLOAD_BYTES = 25_000_000  # bien sous la limite Soynade (50 Mo d'audio)
ASR_TIMEOUT = 60
TTS_TIMEOUT = 90

UNAVAILABLE = "Le mode vocal n'est pas disponible dans cette langue sur ce serveur."
NOT_CONFIGURED = "Le mode vocal n'est pas encore configuré sur ce serveur."
SERVICE_ERROR = "Le service vocal est momentanément indisponible. Veuillez réessayer."
BAD_AUDIO = "L'enregistrement n'a pas pu être lu. Veuillez réessayer."
EMPTY_TEXT = "Aucun texte à lire."


class VoiceError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(limits=httpx.Limits(max_connections=20))
    return _client


def _api_key() -> str:
    key = os.environ.get("SOYNADE_API_KEY", "").strip()
    if not key:
        raise VoiceError(501, NOT_CONFIGURED)
    return key


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


async def _to_wav(data: bytes, filename: str) -> bytes:
    """Convertit l'enregistrement du navigateur en wav mono 16 kHz."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".webm", ".ogg", ".mp4", ".m4a", ".wav", ".mp3", ".flac"}:
        suffix = ".webm"
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / f"in{suffix}", Path(tmp) / "out.wav"
        src.write_bytes(data)
        proc = await asyncio.create_subprocess_exec(
            _ffmpeg(), "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            raise VoiceError(400, BAD_AUDIO)
        if proc.returncode != 0 or not dst.exists() or dst.stat().st_size < 1000:
            logger.warning("ffmpeg: conversion impossible (code %s)", proc.returncode)
            raise VoiceError(400, BAD_AUDIO)
        return dst.read_bytes()


def _check_response(resp: httpx.Response) -> None:
    """Traduit une erreur Soynade en erreur pour le frontend (sans rien divulguer)."""
    if resp.status_code < 400:
        return
    request_id = None
    try:
        request_id = (resp.json().get("error") or {}).get("request_id")
    except (ValueError, AttributeError):
        pass
    logger.warning("soynade: HTTP %s request_id=%s", resp.status_code, request_id)
    if resp.status_code == 429:
        raise VoiceError(429, "Trop de demandes vocales. Patientez un instant puis réessayez.")
    if resp.status_code in (413, 415, 422, 400):
        raise VoiceError(400, BAD_AUDIO)
    # 401/402/403 (cle, credits) et 5xx : un probleme de service, pas de l'utilisateur.
    raise VoiceError(502, SERVICE_ERROR)


async def transcribe(audio: bytes, filename: str, language: str) -> str:
    key = _api_key()
    if language not in ASR_LANGUAGES:
        raise VoiceError(501, UNAVAILABLE)
    if not audio:
        raise VoiceError(400, BAD_AUDIO)
    if len(audio) > MAX_UPLOAD_BYTES:
        raise VoiceError(413, "Enregistrement trop long.")
    wav = await _to_wav(audio, filename)
    try:
        resp = await _http().post(
            f"{SOYNADE_BASE_URL}/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": ("recording.wav", wav, "audio/wav")},
            data={"language": language, "response_format": "json", "temperature": "0"},
            timeout=ASR_TIMEOUT,
        )
    except httpx.HTTPError:
        logger.warning("soynade: transcription injoignable")
        raise VoiceError(502, SERVICE_ERROR)
    _check_response(resp)
    try:
        body = resp.json()
    except ValueError:
        raise VoiceError(502, SERVICE_ERROR)
    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str):
        logger.warning("soynade: reponse de transcription inattendue (cles: %s)", sorted(body) if isinstance(body, dict) else type(body))
        raise VoiceError(502, SERVICE_ERROR)
    return text.strip()


_SENTENCE_END = re.compile(r"[.!?…]\s")


def _truncate(text: str, limit: int) -> str:
    """Coupe a la fin d'une phrase plutot qu'au milieu d'un mot."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [m.end() for m in _SENTENCE_END.finditer(head)]
    return head[: ends[-1]].strip() if ends else head.rsplit(" ", 1)[0]


async def synthesize(text: str, language: str) -> tuple[bytes, str]:
    """Audio (octets, type MIME) de `text` lu en `language`."""
    key = _api_key()
    if language not in TTS_LANGUAGES:
        raise VoiceError(501, UNAVAILABLE)
    text = _truncate(text, TTS_MAX_CHARS)
    if not text:
        raise VoiceError(400, EMPTY_TEXT)
    try:
        resp = await _http().post(
            f"{SOYNADE_BASE_URL}/v1/text-to-speech",
            headers={"Authorization": f"Bearer {key}"},
            json={"text": text, "language": language, "output_format": "mp3"},
            timeout=TTS_TIMEOUT,
        )
    except httpx.HTTPError:
        logger.warning("soynade: synthese injoignable")
        raise VoiceError(502, SERVICE_ERROR)
    _check_response(resp)
    return resp.content, resp.headers.get("content-type", "audio/mpeg")


# ---------------------------------------------------------------- traduction (wolof)
# Le wolof passe par le francais : la question est traduite wo -> fr avant la recherche, la
# reponse fr -> wo ensuite. Soynade traduit bien wo -> fr, mais fr -> wo est irregulier (les phrases
# a chiffres reviennent souvent en francais, ou avec les annees ecrites en lettres) : toute
# traduction qui ne conserve pas exactement les chiffres est refusee par `numbers_preserved`.
TRANSLATE_TIMEOUT = 25
_CACHE_MAX = 2000
_tr_cache: "OrderedDict[tuple[str, str, str], str]" = OrderedDict()
_NUMBER_RE = re.compile(r"\d[\d\s  .,]*\d|\d")


def _cache_put(source: str, target: str, text: str, out: str) -> None:
    _tr_cache[(source, target, text)] = out
    if out != text:  # le sens inverse est connu : l'historique de la discussion ne coute rien
        _tr_cache[(target, source, out)] = text
    while len(_tr_cache) > _CACHE_MAX:
        _tr_cache.popitem(last=False)


def _numbers(text: str) -> list[str]:
    return [re.sub(r"[\s  ]", "", m.group(0)).strip(".,") for m in _NUMBER_RE.finditer(text)]


def numbers_preserved(source_text: str, translated: str) -> bool:
    """Tous les nombres du texte d'origine se retrouvent, a l'identique, dans la traduction."""
    flat = re.sub(r"[\s  ]", "", translated)
    return all(n in flat for n in _numbers(source_text))


async def translate(text: str, source: str, target: str, timeout: float = TRANSLATE_TIMEOUT) -> str:
    """Traduit `text` (wo, fr ou en). Leve VoiceError si le service est injoignable ou refuse."""
    text = text.strip()
    if not text or source == target:
        return text
    key = (source, target, text)
    if key in _tr_cache:
        _tr_cache.move_to_end(key)
        return _tr_cache[key]
    api_key = _api_key()
    for attempt in (0, 1):
        try:
            resp = await _http().post(
                f"{SOYNADE_BASE_URL}/v1/translations",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"text": text, "source_language": source, "target_language": target, "temperature": 0.1},
                timeout=timeout,
            )
        except httpx.HTTPError:
            logger.warning("soynade: traduction injoignable ou trop lente")
            raise VoiceError(502, SERVICE_ERROR)
        # Erreur passagere (503, limite de debit) : une seconde tentative apres une courte pause.
        if attempt == 0 and resp.status_code in (429, 502, 503):
            await asyncio.sleep(1.5)
            continue
        break
    _check_response(resp)
    try:
        out = resp.json().get("translated_text")
    except (ValueError, AttributeError):
        out = None
    if not isinstance(out, str) or not out.strip():
        logger.warning("soynade: reponse de traduction inattendue")
        raise VoiceError(502, SERVICE_ERROR)
    out = out.strip()
    _cache_put(source, target, text, out)
    return out


def _acceptable(source_fr: str, out: str) -> bool:
    """Traduction utilisable : differente de la source, chiffres conserves, peu de mots francais."""
    return out.strip() != source_fr.strip() and numbers_preserved(source_fr, out) and _french_ratio(out) <= 0.2


# Soynade ne traduit pas le vocabulaire statistique (« taux », « population »… reviennent tels quels)
# et limite le debit (429) : un modele de langage prend alors le relais. Gemini Flash donne le wolof
# le plus fidele des modeles essayes (gpt-4.1 et claude contresens sur « jëfandikoo »).
WOLOF_LLM_MODEL = os.environ.get("WOLOF_LLM_MODEL", "google/gemini-2.5-flash")
WOLOF_LLM_TIMEOUT = 40
_WOLOF_SYSTEM = (
    "Tu es un traducteur professionnel francais -> wolof (orthographe officielle du CLAD, alphabet latin : "
    "ñ, ŋ, ë, à, é, ó, x, c, j). Traduis fidelement, en wolof naturel et simple, tel qu'on le parle au Senegal. "
    "Garde EXACTEMENT tels quels tous les chiffres et nombres, les annees, les pourcentages, les noms propres, "
    "les sigles (ANSD, RGPH-5…) et les titres de documents. Reponds uniquement par la traduction, sans commentaire."
)


async def _llm_translate_to_wolof(text_fr: str) -> str | None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        resp = await _http().post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": WOLOF_LLM_MODEL,
                "temperature": 0.2,
                "max_tokens": 1500,
                "messages": [{"role": "system", "content": _WOLOF_SYSTEM}, {"role": "user", "content": text_fr}],
            },
            timeout=WOLOF_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        out = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        logger.warning("traduction wolof par LLM impossible", exc_info=True)
        return None
    return out.strip() if isinstance(out, str) and out.strip() else None


async def to_wolof(text_fr: str) -> str:
    """Traduction francais -> wolof d'une reponse : Soynade d'abord, puis un modele de langage s'il
    echoue ou renvoie du francais. Le francais d'origine si aucune traduction ne conserve les chiffres."""
    try:
        out = await translate(text_fr, "fr", "wo", timeout=20)
        if _acceptable(text_fr, out):
            return out
    except VoiceError:
        pass
    key = ("fr", "wo-llm", text_fr)
    if key in _tr_cache:
        _tr_cache.move_to_end(key)
        return _tr_cache[key]
    out = await _llm_translate_to_wolof(text_fr)
    if out and _acceptable(text_fr, out):
        _tr_cache[key] = out
        while len(_tr_cache) > _CACHE_MAX:
            _tr_cache.popitem(last=False)
        return out
    return text_fr


# Detection grossiere du wolof : le service de traduction retourne le sens inverse (francais ->
# wolof) quand on lui donne du francais, ce qui rendrait une question francaise illisible.
_WOLOF_WORDS = {
    "nga", "ngi", "nanga", "def", "naka", "lan", "ñaata", "ñaar", "ñett", "fan", "kan", "ci", "bi", "yi", "mi",
    "si", "dafa", "bëgg", "begg", "xam", "mën", "laaj", "jërëjëf", "jerejef", "waa", "nit", "ñi", "ñoo", "nekk",
    "dokimaa", "kayit", "amul", "am", "ak", "ba", "boo", "moom", "man", "yow", "ma", "mangi", "maa", "ngir",
    "sa", "ay", "léegi", "leegi", "atum", "at", "weer", "bés", "yoon", "wax", "wone", "tuma",
}
_FRENCH_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "est", "sont", "quel", "quelle", "quels", "quelles",
    "combien", "qui", "que", "quoi", "pour", "dans", "en", "et", "ou", "au", "aux", "sur", "avec", "tu", "as",
    "je", "vous", "avez", "nous", "il", "elle", "ce", "cette", "ont", "par", "comment", "quand", "pourquoi",
    "population", "taux", "evolution", "évolution", "ans", "annee", "année", "donne", "moi", "peux",
}


def looks_wolof(text: str) -> bool:
    """Vrai si le texte est plutot du wolof que du francais ou de l'anglais."""
    low = text.lower()
    words = re.findall(r"[\wëñŋ']+", low)
    wolof = sum(w in _WOLOF_WORDS for w in words) + (2 if re.search(r"[ñëŋ]", low) else 0)
    french = sum(w in _FRENCH_WORDS for w in words)
    return wolof >= french and wolof > 0


def _french_ratio(text: str) -> float:
    words = re.findall(r"[\wëñŋ']+", text.lower())
    return sum(w in _FRENCH_WORDS for w in words) / len(words) if words else 0.0
