"""Cache des reponses et limitation de debit, pour tenir la charge.

- Avec REDIS_URL : cache et compteurs partages entre tous les processus et
  toutes les instances du backend (deploiement multi-serveurs).
- Sans REDIS_URL : repli en memoire, propre a chaque processus (suffisant
  pour un seul serveur).

Le « single-flight » regroupe les questions identiques posees au meme
moment : un seul appel au modele, tous les demandeurs recoivent la reponse.
"""

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Awaitable, Callable

logger = logging.getLogger("ansd-cache")

REDIS_URL = os.environ.get("REDIS_URL", "")
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", str(24 * 3600)))
MEMORY_CACHE_SIZE = int(os.environ.get("MEMORY_CACHE_SIZE", "5000"))


def normalize(text: str) -> str:
    """Cle de cache d'une question : sans accents, casse, ponctuation ni espaces superflus."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w%,.]+", " ", text)
    return " ".join(text.split()).strip(" .,")


class _MemoryStore:
    def __init__(self, size: int):
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._size = size
        self._counters: dict[str, tuple[float, int]] = {}

    async def get(self, key: str) -> str | None:
        item = self._data.get(key)
        if not item:
            return None
        expires, value = item
        if expires < time.time():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: str, ttl: int) -> None:
        self._data[key] = (time.time() + ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._size:
            self._data.popitem(last=False)

    async def incr(self, key: str, window: int) -> int:
        now = time.time()
        expires, count = self._counters.get(key, (now + window, 0))
        if expires < now:
            expires, count = now + window, 0
        self._counters[key] = (expires, count + 1)
        if len(self._counters) > 100_000:  # menage occasionnel
            self._counters = {k: v for k, v in self._counters.items() if v[0] >= now}
        return count + 1


class _RedisStore:
    def __init__(self, url: str):
        import redis.asyncio as redis

        self._r = redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)

    async def get(self, key: str) -> str | None:
        value = await self._r.get(key)
        return value.decode() if value else None

    async def set(self, key: str, value: str, ttl: int) -> None:
        await self._r.set(key, value, ex=ttl)

    async def incr(self, key: str, window: int) -> int:
        pipe = self._r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window, nx=True)
        count, _ = await pipe.execute()
        return int(count)


class Store:
    """Redis si configure et joignable, sinon memoire. Une panne de Redis ne
    casse jamais une reponse : on bascule en memoire."""

    def __init__(self):
        self._memory = _MemoryStore(MEMORY_CACHE_SIZE)
        self._redis = _RedisStore(REDIS_URL) if REDIS_URL else None
        self._inflight: dict[str, asyncio.Future] = {}

    @property
    def backend(self) -> str:
        return "redis" if self._redis else "memory"

    async def _call(self, method: str, *args):
        if self._redis:
            try:
                return await getattr(self._redis, method)(*args)
            except Exception as exc:
                logger.warning("redis indisponible (%s), repli memoire", exc)
        return await getattr(self._memory, method)(*args)

    async def get_json(self, key: str):
        raw = await self._call("get", key)
        return json.loads(raw) if raw else None

    async def set_json(self, key: str, value, ttl: int = CACHE_TTL) -> None:
        await self._call("set", key, json.dumps(value, ensure_ascii=False), ttl)

    async def cached(self, key: str, compute: Callable[[], Awaitable], ttl: int = CACHE_TTL, cache_if=None):
        """Renvoie (valeur, depuis_le_cache). Les appels concurrents sur une meme
        cle partagent un seul calcul (single-flight)."""
        hit = await self.get_json(key)
        if hit is not None:
            return hit, True
        if key in self._inflight:
            return await asyncio.shield(self._inflight[key]), True

        future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            value = await compute()
            if cache_if is None or cache_if(value):
                await self.set_json(key, value, ttl)
            future.set_result(value)
            return value, False
        except BaseException as exc:
            future.set_exception(exc)
            future.exception()  # evite l'avertissement « exception never retrieved »
            raise
        finally:
            self._inflight.pop(key, None)

    async def allow(self, key: str, limit: int, window: int = 60) -> bool:
        """Limitation de debit a fenetre fixe : vrai si l'appel est autorise."""
        if limit <= 0:
            return True
        return await self._call("incr", f"rl:{key}:{int(time.time() // window)}", window) <= limit


store = Store()
