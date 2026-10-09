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
import time
from collections import OrderedDict
from pathlib import Path

import httpx

import figures

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
QUOTA_EXHAUSTED = (
    "La lecture audio en wolof est momentanément indisponible (limite quotidienne du service vocal atteinte). "
    "Réessayez dans quelques heures."
)
_tts_blocked_until = 0.0  # quota de synthese vocale epuise : pas d'appel avant cet instant (time.monotonic)


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


# Voix wolof locale (service tts-wolof, voir tts/app.py) : prend le relais quand Soynade est
# indisponible (quota epuise, panne, cle absente). Vide = pas de secours.
LOCAL_TTS_URL = os.environ.get("LOCAL_TTS_URL", "").strip().rstrip("/")
LOCAL_TTS_TIMEOUT = 120


async def synthesize(text: str, language: str) -> tuple[bytes, str, str]:
    """Audio (octets, type MIME, fournisseur « soynade » ou « local ») de `text` lu en `language`."""
    if language not in TTS_LANGUAGES:
        raise VoiceError(501, UNAVAILABLE)
    try:
        content, media_type = await _soynade_synthesize(text, language)
        return content, media_type, "soynade"
    except VoiceError as exc:
        if exc.status == 400 or not LOCAL_TTS_URL:
            raise
        try:
            resp = await _http().post(f"{LOCAL_TTS_URL}/speak", json={"text": text}, timeout=LOCAL_TTS_TIMEOUT)
        except httpx.HTTPError:
            logger.warning("voix wolof locale injoignable")
            raise exc
        if resp.status_code != 200:
            logger.warning("voix wolof locale indisponible (HTTP %s)", resp.status_code)
            raise exc
        return resp.content, resp.headers.get("content-type", "audio/wav"), "local"


async def _soynade_synthesize(text: str, language: str) -> tuple[bytes, str]:
    key = _api_key()
    text = _truncate(text, TTS_MAX_CHARS)
    if not text:
        raise VoiceError(400, EMPTY_TEXT)
    global _tts_blocked_until
    if time.monotonic() < _tts_blocked_until:
        raise VoiceError(503, QUOTA_EXHAUSTED)
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
    if resp.status_code == 429 and _retry_after(resp) > 30:
        # Quota (journalier) epuise : plus d'appel jusqu'a l'heure indiquee, message clair a l'utilisateur.
        _tts_blocked_until = time.monotonic() + _retry_after(resp)
        logger.warning("soynade: quota de synthese vocale epuise, suspendu %.0f min", _retry_after(resp) / 60)
        raise VoiceError(503, QUOTA_EXHAUSTED)
    _check_response(resp)
    return resp.content, resp.headers.get("content-type", "audio/mpeg")


def _retry_after(resp: httpx.Response) -> float:
    try:
        return float(resp.headers.get("retry-after") or 0)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------- traduction (wolof)
# Le wolof passe par le francais : la question est traduite wo -> fr avant la recherche, la
# reponse fr -> wo ensuite. Soynade traduit bien wo -> fr, mais fr -> wo est irregulier (les phrases
# a chiffres reviennent souvent en francais, ou avec les annees ecrites en lettres) : toute
# traduction qui ne conserve pas exactement les chiffres est refusee par `numbers_preserved`.
TRANSLATE_TIMEOUT = 25
_blocked_until = 0.0  # Soynade a annonce un quota epuise : pas d'appel avant cet instant (time.monotonic)
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
    global _blocked_until
    if time.monotonic() < _blocked_until:
        raise VoiceError(429, SERVICE_ERROR)  # quota epuise : inutile d'attendre une reponse
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
        if resp.status_code == 429:
            try:
                retry_after = float(resp.headers.get("retry-after") or 0)
            except ValueError:
                retry_after = 0
            if retry_after > 30:
                # Quota (journalier) epuise : plus aucun appel jusqu'a l'heure indiquee, le modele de
                # langage prend le relais sans faire attendre l'utilisateur.
                _blocked_until = time.monotonic() + retry_after
                logger.warning("soynade: quota de traduction epuise, suspendu %.0f min", retry_after / 60)
                raise VoiceError(429, SERVICE_ERROR)
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


# Reponses (francais -> wolof) : un modele de langage, directement. Soynade ne traduit pas le vocabulaire
# statistique (« taux », « population »… reviennent tels quels) : l'appeler d'abord coutait du quota et
# plusieurs secondes pour rien. Gemini Flash donne le wolof le plus fidele des modeles essayes (gpt-4.1
# et claude font des contresens, ex. « jëfandikoo »). Questions (wolof -> francais) : Soynade, qui les
# traduit bien, et ce modele en secours (quota epuise, panne).
WOLOF_LLM_MODEL = os.environ.get("WOLOF_LLM_MODEL", "google/gemini-2.5-flash")
WOLOF_LLM_TIMEOUT = 40
_WOLOF_SYSTEM = (
    "Tu es un traducteur professionnel francais -> wolof (orthographe officielle du CLAD, alphabet latin : "
    "ñ, ŋ, ë, à, é, ó, x, c, j). Traduis fidelement, en wolof naturel et simple, tel qu'on le parle au Senegal. "
    "Garde EXACTEMENT tels quels tous les chiffres et nombres, les annees, les pourcentages, les noms propres, "
    "les sigles (ANSD, RGPH-5…) et les titres de documents. Reponds uniquement par la traduction, sans commentaire."
)
_FRENCH_SYSTEM = (
    "Tu es un traducteur professionnel wolof -> francais. Le texte est une question posee a l'assistant "
    "statistique de l'ANSD (Senegal). Traduis-le fidelement en francais clair, en gardant tels quels les "
    "chiffres, les annees, les noms de lieux et les sigles. Si le texte est deja en francais, renvoie-le tel quel. "
    "Reponds uniquement par la traduction, sans commentaire."
)


async def _llm_translate(system: str, text: str) -> str | None:
    """Traduction par le modele de langage (OpenRouter). None en cas d'echec."""
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
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
            },
            timeout=WOLOF_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        out = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        logger.warning("traduction par LLM impossible", exc_info=True)
        return None
    return out.strip() if isinstance(out, str) and out.strip() else None


def _cached(key: tuple[str, str, str]) -> str | None:
    if key in _tr_cache:
        _tr_cache.move_to_end(key)
        return _tr_cache[key]
    return None


async def to_wolof(text_fr: str) -> str:
    """Traduction francais -> wolof d'une reponse, par le modele de langage. Le francais d'origine si
    la traduction echoue, ne conserve pas les chiffres ou reste trop francaise."""
    key = ("fr", "wo-llm", text_fr)
    if (hit := _cached(key)) is not None:
        return hit
    out = await _llm_translate(_WOLOF_SYSTEM, text_fr)
    if out and _acceptable(text_fr, out):
        _cache_put("fr", "wo-llm", text_fr, out)
        return out
    return text_fr


# ---------------------------------------------------------------- traduction (pulaar)
# Modele local (service local-models : NLLB-600M + adaptateur LoRA pulaar). Les reponses sont
# traduites phrase par phrase, en un seul lot ; une phrase dont un chiffre ne se retrouve pas dans la
# traduction reste en francais (le modele, experimental, peut deformer ou inventer).
LOCAL_MT_URL = os.environ.get("LOCAL_MT_URL", "").strip().rstrip("/")
LOCAL_MT_TIMEOUT = 120
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LIST_PREFIX = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)?(.*)$")


async def _local_translate(texts: list[str], source: str, target: str) -> list[str] | None:
    if not LOCAL_MT_URL or not texts:
        return None
    try:
        resp = await _http().post(
            f"{LOCAL_MT_URL}/translate", json={"texts": texts, "source": source, "target": target}, timeout=LOCAL_MT_TIMEOUT
        )
    except httpx.HTTPError:
        logger.warning("traduction locale injoignable")
        return None
    if resp.status_code != 200:
        logger.warning("traduction locale indisponible (HTTP %s)", resp.status_code)
        return None
    out = resp.json().get("texts")
    return out if isinstance(out, list) and len(out) == len(texts) else None


def _same_figures(source: str, translated: str) -> bool:
    return figures.figure_set(source) <= figures.figure_set(translated)


async def to_pulaar(text_fr: str) -> str:
    """Traduction francais -> pulaar d'une reponse ; listes et lignes conservees, tableaux laisses en
    francais. Le francais d'origine si le service local est indisponible."""
    key = ("fr", "ff", text_fr)
    if (hit := _cached(key)) is not None:
        return hit
    lines = text_fr.replace("**", "").split("\n")
    layout: list[tuple[str, int, int] | str] = []  # (prefixe, debut, nombre) ou ligne gardee telle quelle
    sentences: list[str] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("|"):
            layout.append(line)
            continue
        prefix, body = _LIST_PREFIX.match(line).groups()
        parts = [s for s in _SENTENCE_SPLIT.split(body) if s.strip()]
        layout.append((prefix or "", len(sentences), len(parts)))
        sentences.extend(parts)
    translated = await _local_translate(sentences, "fr", "ff")
    if translated is None:
        return text_fr
    kept = [t if t.strip() and _same_figures(s, t) else s for s, t in zip(sentences, translated)]
    out = "\n".join(
        item if isinstance(item, str) else item[0] + " ".join(kept[item[1] : item[1] + item[2]]) for item in layout
    )
    _cache_put("fr", "ff", text_fr, out)
    return out


_PULAAR_FRENCH_SYSTEM = (
    "Tu es un traducteur professionnel pulaar (peul du Senegal, Fuuta Tooro) -> francais. Le texte est un message "
    "ecrit a l'assistant statistique de l'ANSD (Senegal), souvent en orthographe libre et sans accents "
    "(ex. « mido falla andu » = « je veux savoir »). Traduis-le fidelement en francais clair, en gardant tels quels "
    "les chiffres, noms de lieux et sigles. Si le texte est deja en francais, renvoie-le tel quel. "
    "Reponds uniquement par la traduction."
)


_TITLE_LANGUAGES = {
    "en": "anglais",
    "wo": "wolof du Senegal (orthographe officielle du CLAD : à, é, ë, ó, ñ, ŋ, x)",
    "ff": "pulaar du Senegal (Fuuta Tooro, orthographe officielle : ɓ, ɗ, ƴ, ŋ, ñ)",
}


async def translate_title(title_fr: str, language: str) -> str:
    """Titre court de discussion (1 a 3 mots) traduit pour l'historique ; le francais en cas d'echec."""
    target = _TITLE_LANGUAGES.get(language)
    if not target:
        return title_fr
    out = await _llm_translate(
        f"Traduis ce titre de discussion (1 a 3 mots, sujet statistique) du francais vers le {target}. "
        "Reponds uniquement par le titre traduit, sans guillemets ni ponctuation finale.",
        title_fr,
    )
    out = (out or "").strip().strip("\"'«»“”.").strip()
    return out[:60] if out and len(out.split()) <= 6 else title_fr


async def pulaar_to_french(text: str) -> str:
    """Question pulaar -> francais : le modele de langage (Gemini), qui comprend le pulaar ecrit librement
    (« mido falla andu… ») bien mieux que NLLB ; NLLB local en secours. Le texte tel quel s'il est deja
    en francais ou si les deux traductions echouent."""
    if _french_ratio(text) > 0.25:
        return text
    key = ("ff", "fr", text)
    if (hit := _cached(key)) is not None:
        return hit
    out = await _llm_translate(_PULAAR_FRENCH_SYSTEM, text)
    if not out or not numbers_preserved(text, out):
        translated = await _local_translate([text], "ff", "fr")
        out = translated[0] if translated and translated[0].strip() else None
    if not out:
        return text
    _cache_put("ff", "fr", text, out)
    return out


async def to_french(text_wo: str, timeout: float = TRANSLATE_TIMEOUT) -> str:
    """Traduction wolof -> francais d'une question : Soynade, puis le modele de langage si Soynade est
    indisponible (quota, panne). Leve VoiceError si les deux echouent."""
    try:
        return await translate(text_wo, "wo", "fr", timeout=timeout)
    except VoiceError:
        pass
    key = ("wo", "fr-llm", text_wo)
    if (hit := _cached(key)) is not None:
        return hit
    out = await _llm_translate(_FRENCH_SYSTEM, text_wo)
    if not out or not numbers_preserved(text_wo, out):
        raise VoiceError(502, SERVICE_ERROR)
    _cache_put("wo", "fr-llm", text_wo, out)
    return out


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
