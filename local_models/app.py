"""Modeles locaux (CPU) : voix wolof de secours et traduction francais <-> pulaar.

TRADUCTION PULAAR (POST /translate) : facebook/nllb-200-distilled-600M (pulaar = « fuv_Latn ») avec
l'adaptateur LoRA kawkumputer/pulaar-ai-nllb-600m-v4 pour francais -> pulaar ; le modele de base seul
(adaptateur desactive) pour pulaar -> francais. Un seul modele en memoire (~2,4 Go) pour les deux sens.
Licence : CC-BY-NC 4.0 (NLLB et l'adaptateur) ; l'auteur presente l'adaptateur comme experimental.

VOIX WOLOF (POST /speak) : synthese vocale de secours quand Soynade est indisponible (quota, panne).

Modele : bilalfaye/speecht5_tts-wolof-v0.2 (SpeechT5 affine sur le wolof et le francais, licence MIT),
vocodeur microsoft/speecht5_hifigan (MIT), empreinte de voix : un x-vector du jeu CMU Arctic (MIT),
enregistre dans speaker.npy. Tourne sur CPU.

Rapidite : le texte est coupe en segments courts (phrases coupees aux virgules), calcules en parallele
par plusieurs processus (TTS_WORKERS, chacun avec sa copie du modele, ~350 Mo).

Les chiffres sont lus en francais (« dix-neuf virgule un pour cent »), comme on les dit couramment au
Senegal ; le modele lit les deux langues.

    POST /speak  {"text": "..."}  -> audio/wav (16 kHz, mono)
    GET  /health                  -> {"status": "loading" | "ready" | "error"}
"""

import asyncio
import io
import logging
import multiprocessing
import os
import re
import threading
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

import imageio_ffmpeg
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from num2words import num2words
from pydantic import BaseModel, Field

logger = logging.getLogger("tts-wolof")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

MODEL = os.environ.get("TTS_MODEL", "bilalfaye/speecht5_tts-wolof-v0.2")
VOCODER = os.environ.get("TTS_VOCODER", "microsoft/speecht5_hifigan")
MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "600"))
MAX_WORDS = int(os.environ.get("TTS_MAX_WORDS", "10"))  # mots par segment calcule
WORKERS = int(os.environ.get("TTS_WORKERS", "6"))
THREADS = int(os.environ.get("TTS_THREADS", "3"))  # par processus
SPEED = float(os.environ.get("TTS_SPEED", "1.25"))  # debit de lecture (1 = vitesse du modele)
STRETCH = os.environ.get("TTS_STRETCH", "rubberband").strip().lower()  # « rubberband » ou « atempo »
DENOISE = os.environ.get("TTS_DENOISE", "0").strip() in {"1", "true", "yes"}  # debruitage leger
SAMPLE_RATE = 16000

_state = {"status": "loading", "error": None, "translation": "loading"}
_pool: ProcessPoolExecutor | None = None

# ------------------------------------------------------------------ processus de calcul

_worker: dict = {}


def _init_worker() -> None:
    """Charge le modele dans chaque processus de calcul (une fois)."""
    import torch
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    torch.set_num_threads(THREADS)
    _worker["torch"] = torch
    _worker["processor"] = SpeechT5Processor.from_pretrained(MODEL)
    _worker["model"] = SpeechT5ForTextToSpeech.from_pretrained(MODEL).eval()
    _worker["vocoder"] = SpeechT5HifiGan.from_pretrained(VOCODER).eval()
    _worker["speaker"] = torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "speaker.npy"))).unsqueeze(0)


def _segment_audio(segment: str) -> np.ndarray:
    torch, processor, model = _worker["torch"], _worker["processor"], _worker["model"]
    inputs = processor(text=segment, return_tensors="pt", truncation=True, max_length=model.config.max_text_positions)
    with torch.inference_mode():
        # Synthese de base : les options de la fiche du modele (faisceaux, penalites de repetition)
        # le faisaient deborder (2,5 fois la duree normale) et doublaient le temps de calcul.
        speech = model.generate_speech(inputs["input_ids"], _worker["speaker"], vocoder=_worker["vocoder"])
    return speech.numpy()


# ------------------------------------------------------------------ traduction (pulaar)

NLLB_MODEL = os.environ.get("MT_BASE_MODEL", "facebook/nllb-200-distilled-600M")
PULAAR_ADAPTER = os.environ.get("MT_PULAAR_ADAPTER", "kawkumputer/pulaar-ai-nllb-600m-v4")
MT_THREADS = int(os.environ.get("MT_THREADS", "8"))
NLLB_CODES = {"fr": "fra_Latn", "ff": "fuv_Latn"}
_mt: dict = {}
_mt_pool: ProcessPoolExecutor | None = None


def _init_mt() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(MT_THREADS)
    _mt["torch"] = torch
    _mt["tokenizer"] = AutoTokenizer.from_pretrained(NLLB_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    _mt["model"] = PeftModel.from_pretrained(base, PULAAR_ADAPTER).eval()


def _translate_batch(texts: list[str], source: str, target: str) -> list[str]:
    torch, tok, model = _mt["torch"], _mt["tokenizer"], _mt["model"]
    tok.src_lang = NLLB_CODES[source]
    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=200)
    options = {"forced_bos_token_id": tok.convert_tokens_to_ids(NLLB_CODES[target]), "num_beams": 4, "max_new_tokens": 200}
    with torch.inference_mode():
        if target == "ff":
            # Reglages de la fiche de l'adaptateur (evite les repetitions).
            out = model.generate(**inputs, **options, repetition_penalty=1.3, no_repeat_ngram_size=3)
        else:
            with model.disable_adapter():  # l'adaptateur n'a appris que le sens francais -> pulaar
                out = model.generate(**inputs, **options)
    return tok.batch_decode(out, skip_special_tokens=True)


def _load_mt() -> None:
    global _mt_pool
    try:
        started = time.time()
        _mt_pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=_init_mt)
        _mt_pool.submit(_translate_batch, ["Bonjour."], "fr", "ff").result()
        _state["translation"] = "ready"
        logger.info("traduction pulaar prete en %.0f s", time.time() - started)
    except Exception as exc:
        _state["translation"] = f"error: {type(exc).__name__}: {exc}"
        logger.exception("chargement de la traduction pulaar impossible")


def _load() -> None:
    global _pool
    try:
        started = time.time()
        # « spawn » : PyTorch supporte mal la copie (fork) d'un processus qui a deja des threads.
        _pool = ProcessPoolExecutor(
            max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker
        )
        # Prechauffage : un segment par processus, pour que chacun charge son modele des maintenant.
        list(_pool.map(_segment_audio, ["Na nga def"] * WORKERS))
        _state["status"] = "ready"
        logger.info("voix wolof prete en %.0f s (%s, %d processus x %d threads)", time.time() - started, MODEL, WORKERS, THREADS)
    except Exception as exc:
        _state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        logger.exception("chargement de la voix wolof impossible")


# ------------------------------------------------------------------ texte


def _say_number(match: re.Match) -> str:
    raw = match.group(0)
    compact = re.sub(r"[\s  ]", "", raw)
    percent = compact.endswith("%")
    compact = compact.rstrip("%")
    try:
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", compact):  # milliers
            value = int(re.sub(r"[.,]", "", compact))
        elif re.search(r"[.,]", compact):
            value = float(compact.replace(",", "."))
        else:
            value = int(compact)
        words = num2words(value, lang="fr")
    except (ValueError, OverflowError, NotImplementedError):
        return raw
    return f" {words}{' pour cent' if percent else ''} "


def prepare(text: str) -> list[tuple[str, float]]:
    """Texte lisible par le modele, coupe en segments courts [(segment, pause apres en s)].
    Des segments courts (phrases coupees aux virgules) sont lus plus vite et plus surement : sur une
    longue phrase, le modele ralentit et finit parfois par deborder apres la fin du texte."""
    text = re.sub(r"\*\*|__|#{1,6}\s*|`", "", text)
    text = re.sub(r"\[\d+(?:[,;]\s*\d+)*\]", "", text)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)
    text = re.sub(r"\d[\d\s  ]*(?:[.,]\d+)?\s?%?", _say_number, text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split()))[:MAX_CHARS]
    out: list[tuple[str, float]] = []
    for sentence in (s.strip() for s in re.split(r"(?<=[.!?;:])\s+", text) if s.strip()):
        parts = [p.strip() for p in sentence.split(",") if p.strip()]
        merged: list[str] = []
        for p in parts:  # un fragment de moins de 3 mots reste attache au precedent
            if merged and len(p.split()) < 3:
                merged[-1] += ", " + p
            else:
                merged.append(p)
        # Segments de plus de MAX_WORDS mots coupes en parts egales : ils sont calcules en parallele,
        # c'est le plus long qui fixe le temps d'attente.
        pieces: list[tuple[str, float]] = []
        for i, p in enumerate(merged):
            words = p.split()
            n = -(-len(words) // MAX_WORDS)
            size = -(-len(words) // n)
            chunks = [" ".join(words[j : j + size]) for j in range(0, len(words), size)]
            for k, chunk in enumerate(chunks):
                last_of_part = k == len(chunks) - 1
                pause = (0.35 if i == len(merged) - 1 else 0.15) if last_of_part else 0.05
                pieces.append((chunk[:200], pause))
        out.extend(pieces)
    return out


# ------------------------------------------------------------------ synthese


async def synthesize(text: str) -> np.ndarray:
    """Segments calcules en parallele par les processus, puis remis dans l'ordre avec leurs pauses."""
    segments = prepare(text)
    if not segments:
        raise ValueError("texte vide")
    loop = asyncio.get_running_loop()
    audios = await asyncio.gather(*[loop.run_in_executor(_pool, _segment_audio, s) for s, _ in segments])
    pieces = []
    fade_in, fade_out = int(SAMPLE_RATE * 0.005), int(SAMPLE_RATE * 0.02)
    for audio, (_, pause) in zip(audios, segments):
        audio = audio.astype(np.float32).copy()
        # Fondus aux bords de chaque segment : le modele s'arrete net (dernier echantillon non nul),
        # ce qui faisait un petit clic a chaque jonction.
        if len(audio) > fade_in + fade_out:
            audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
            audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
        pieces.append(audio)
        pieces.append(np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32))
    return np.concatenate(pieces)


def _wav(audio: np.ndarray) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _audio_filters() -> str:
    """Traitement de la voix : acceleration sans changer la hauteur (rubberband, plus propre qu'atempo
    qui donne un leger effet metallique), puis coupure du souffle aigu du vocodeur, attenuation des
    sifflantes et volume normalise (le modele parle tres bas, ~ -27 dB)."""
    if STRETCH == "atempo":
        stretch = f"atempo={SPEED}"
    else:
        stretch = f"rubberband=tempo={SPEED}:pitchq=quality:formant=preserved:window=standard:transients=smooth"
    chain = [stretch]
    if DENOISE:
        chain.append("afftdn=nf=-30")
    chain += ["highpass=f=70", "lowpass=f=7000", "deesser=i=0.4", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    return ",".join(chain)


async def _mp3(audio: np.ndarray) -> bytes:
    """MP3 (~10 fois plus leger que le WAV), accelere de SPEED sans changer la hauteur de la voix :
    le modele parle lentement (~2,5 mots/s contre ~3,2 a l'oral)."""
    proc = await asyncio.create_subprocess_exec(
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "wav", "-i", "pipe:0",
        "-filter:a", _audio_filters(), "-ac", "1", "-ar", "22050", "-codec:a", "libmp3lame", "-q:a", "2", "-f", "mp3", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(_wav(audio))
    if proc.returncode != 0 or not out:
        raise RuntimeError(f"ffmpeg : {err.decode(errors='ignore')[:200]}")
    return out


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Au premier demarrage, telechargement des modeles (~3 Go) : plusieurs minutes.
    threading.Thread(target=_load, daemon=True).start()
    threading.Thread(target=_load_mt, daemon=True).start()
    yield
    for pool in (_pool, _mt_pool):
        if pool:
            pool.shutdown(cancel_futures=True)


app = FastAPI(title="Voix wolof locale", lifespan=lifespan)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class TranslateRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=60)
    source: str
    target: str


@app.get("/health")
def health() -> dict:
    return _state


@app.post("/translate")
async def translate(req: TranslateRequest) -> dict:
    """Traduit une liste de phrases (un lot : plus rapide que phrase par phrase). Sens : fr <-> ff."""
    if (req.source, req.target) not in {("fr", "ff"), ("ff", "fr")}:
        raise HTTPException(status_code=400, detail="sens de traduction non pris en charge")
    if _state.get("translation") != "ready":
        raise HTTPException(status_code=503, detail=f"traduction pulaar : {_state.get('translation')}")
    texts = [t.strip()[:600] for t in req.texts]
    loop = asyncio.get_running_loop()
    out = await loop.run_in_executor(_mt_pool, _translate_batch, texts, req.source, req.target)
    return {"texts": out}


@app.post("/speak")
async def speak(req: SpeakRequest) -> Response:
    if _state["status"] != "ready":
        raise HTTPException(status_code=503, detail=f"voix wolof locale : {_state['status']}")
    try:
        audio = await synthesize(req.text)
    except ValueError:
        raise HTTPException(status_code=400, detail="texte vide")
    try:
        return Response(content=await _mp3(audio), media_type="audio/mpeg")
    except RuntimeError:
        logger.warning("encodage mp3 impossible, envoi en wav", exc_info=True)
        return Response(content=_wav(audio), media_type="audio/wav")
