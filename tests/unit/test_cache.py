"""cache.py : cle de cache, cache memoire, calcul unique (single-flight) et limitation de debit."""

import asyncio

import pytest

import cache


def run(coro):
    return asyncio.run(coro)


def test_normalize_ignore_accents_casse_et_ponctuation():
    assert cache.normalize("  Quel est le TAUX de chômage ?? ") == "quel est le taux de chomage"
    assert cache.normalize("Évolution du PIB, 2024.") == "evolution du pib, 2024"


def test_store_en_memoire_sans_redis():
    store = cache.Store()
    assert store.backend == "memory"

    async def scenario():
        assert await store.get_json("absent") is None
        await store.set_json("k", {"a": 1, "texte": "Sénégal"})
        assert await store.get_json("k") == {"a": 1, "texte": "Sénégal"}

    run(scenario())


def test_memoire_expiration_et_taille_maximale():
    mem = cache._MemoryStore(size=2)

    async def scenario():
        await mem.set("a", "1", ttl=-1)  # deja expire
        assert await mem.get("a") is None
        await mem.set("b", "2", ttl=60)
        await mem.set("c", "3", ttl=60)
        await mem.set("d", "4", ttl=60)  # depasse la taille : la plus ancienne part
        assert await mem.get("b") is None
        assert await mem.get("d") == "4"

    run(scenario())


def test_cached_calcule_une_seule_fois_pour_des_appels_simultanes():
    store = cache.Store()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {"valeur": 42}

    async def scenario():
        results = await asyncio.gather(*[store.cached("same", compute) for _ in range(5)])
        assert all(value == {"valeur": 42} for value, _ in results)
        assert sum(1 for _, from_cache in results if not from_cache) == 1
        # Ensuite servi depuis le cache, et `refresh` force un nouveau calcul.
        assert (await store.cached("same", compute))[1] is True
        assert (await store.cached("same", compute, refresh=True))[1] is False

    run(scenario())
    assert calls == 2


def test_cached_ne_garde_pas_une_erreur():
    store = cache.Store()

    async def failing():
        raise RuntimeError("panne")

    async def ok():
        return {"ok": True}

    async def scenario():
        with pytest.raises(RuntimeError):
            await store.cached("k", failing)
        assert await store.cached("k", ok) == ({"ok": True}, False)

    run(scenario())


def test_cached_cache_if():
    store = cache.Store()

    async def scenario():
        await store.cached("k", lambda: asyncio.sleep(0, result={"answered": False}), cache_if=lambda v: v["answered"])
        assert await store.get_json("k") is None

    run(scenario())


def test_allow_limite_le_debit():
    store = cache.Store()

    async def scenario():
        allowed = [await store.allow("client:x", limit=3) for _ in range(5)]
        assert allowed == [True, True, True, False, False]
        assert await store.allow("client:y", limit=3)  # autre client, autre compteur
        assert await store.allow("client:x", limit=0)  # 0 = illimite

    run(scenario())
