"""The response cache, its keys, and the spend cap."""

from __future__ import annotations

import hashlib
import inspect
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eval.retrieval import config
from eval.retrieval.cache import DiskCache, make_key
from eval.retrieval.spend import SpendLimitExceeded, SpendTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
WAIT_S = 10  # upper bound for anything that would otherwise hang a broken test


@pytest.fixture
def cache(tmp_path):
    store = DiskCache(tmp_path / "cache.sqlite")
    yield store
    store.close()


# --------------------------------------------------------------------------- #
# make_key                                                                     #
# --------------------------------------------------------------------------- #
def test_make_key_is_the_sha256_of_the_canonical_json():
    # Pinned byte for byte: a change here orphans every response already cached.
    canonical = b'["llm",{"a":[1,2.5,null,true],"b":"caf\\u00e9"}]'
    key = make_key("llm", {"b": "café", "a": [1, 2.5, None, True]})
    assert key == hashlib.sha256(canonical).hexdigest()


def test_make_key_ignores_dict_order():
    first = make_key("jev", {"query": "q", "nested": {"a": 1, "b": 2}})
    second = make_key("jev", {"nested": {"b": 2, "a": 1}, "query": "q"})
    assert first == second


def test_make_key_separates_values_types_and_part_boundaries():
    keys = [
        make_key("a", "b"),
        make_key("ab"),
        make_key("a", "b", ""),
        make_key(["a", "b"]),
        make_key("a", 1),
        make_key("a", "1"),
        make_key("a", None),
        make_key("a", [1, 2]),
        make_key("a", [2, 1]),
    ]
    assert len(set(keys)) == len(keys)


def test_make_key_rejects_what_is_not_plain_json():
    with pytest.raises(TypeError):
        make_key("llm", object())
    with pytest.raises(ValueError):
        make_key("llm", float("nan"))


# --------------------------------------------------------------------------- #
# DiskCache                                                                    #
# --------------------------------------------------------------------------- #
def test_miss_then_hit_round_trips_json(cache):
    assert cache.get("k") is None

    value = {
        "answers": {"c0": {"type": "noul", "noul": 0.96}, "c1": {"type": "refusal"}},
        "text": "naïve ☕ \ud83d",  # models do emit lone surrogates; storage must cope
        "numbers": [1, 2.5, None, True],
    }
    cache.set("k", value)

    assert cache.get("k") == value
    assert list(cache.get("k")["answers"]) == ["c0", "c1"]  # object order is kept


def test_set_replaces_the_earlier_value(cache):
    cache.set("k", {"version": 1})
    cache.set("k", {"version": 2})
    assert cache.get("k") == {"version": 2}


def test_a_stored_null_reads_as_a_miss(cache):
    # The clients store the payload objects they got, so None is always a miss to them.
    cache.set("k", None)
    assert cache.get("k") is None
    cache.set("k", {"after": "null"})
    assert cache.get("k") == {"after": "null"}


def test_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "not" / "yet" / "there" / "cache.sqlite"
    store = DiskCache(path)
    try:
        store.set("k", 1)
    finally:
        store.close()
    assert path.is_file()


def test_default_path_is_in_the_gitignored_cache_dir():
    assert config.CACHE_DIR.name == ".cache"
    assert config.CACHE_DIR.relative_to(REPO_ROOT) == Path("eval/retrieval/.cache")
    assert inspect.signature(DiskCache).parameters["path"].default is None


def test_default_path_follows_a_redirected_cache_dir(tmp_path, monkeypatch):
    # Resolved when the cache is built, not at import: a test that redirects
    # config.CACHE_DIR (as the CLI smoke tests do) must never write the real cache.
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "redirected")

    store = DiskCache()
    try:
        store.set("k", 1)
    finally:
        store.close()

    assert store.path == tmp_path / "redirected" / "api_cache.sqlite"
    assert store.path.is_file()


def test_a_later_instance_reads_what_an_earlier_one_wrote(tmp_path):
    path = tmp_path / "cache.sqlite"
    first = DiskCache(path)
    first.set("k", {"from": "first"})
    first.close()

    second = DiskCache(path)
    try:
        assert second.get("k") == {"from": "first"}
        second.set("k2", {"from": "second"})
        assert second.get("k2") == {"from": "second"}
    finally:
        second.close()


def test_entries_survive_a_process_that_died_without_closing(tmp_path):
    # An interrupted run never closes its cache; the resumed run must find its work.
    path = tmp_path / "cache.sqlite"
    script = (
        "import os, sys\n"
        "from eval.retrieval.cache import DiskCache\n"
        "cache = DiskCache(sys.argv[1])\n"
        "cache.set('from-the-dead', {'n': 1})\n"
        "os._exit(0)\n"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(path)], check=True, cwd=REPO_ROOT, timeout=60
    )

    store = DiskCache(path)
    try:
        assert store.get("from-the-dead") == {"n": 1}
    finally:
        store.close()


def test_threads_share_one_cache(cache):
    workers, items = 8, 25

    def work(worker: int) -> None:
        for item in range(items):
            key = make_key("smoke", worker, item)
            cache.set(key, {"worker": worker, "item": item})
            assert cache.get(key) == {"worker": worker, "item": item}
            cache.set("contended", {"last": worker})
            assert cache.get("contended")["last"] in range(workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(work, worker) for worker in range(workers)]:
            future.result(timeout=60)  # re-raises anything a worker hit

    for worker in range(workers):
        for item in range(items):
            assert cache.get(make_key("smoke", worker, item)) == {"worker": worker, "item": item}


def test_lock_lets_one_thread_compute_a_missing_entry(cache):
    workers = 8
    start = threading.Barrier(workers)
    computed: list[int] = []

    def get_or_compute(worker: int) -> dict:
        start.wait(timeout=WAIT_S)
        with cache.lock("k"):
            value = cache.get("k")
            if value is None:
                computed.append(worker)
                value = {"computed_by": worker}
                cache.set("k", value)
        return value

    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(get_or_compute, range(workers)))

    assert len(computed) == 1
    assert values == [{"computed_by": computed[0]}] * workers
    assert cache._key_locks == {}  # a key's lock is dropped once nobody holds or awaits it


def test_lock_does_not_make_different_keys_wait(cache):
    entered = threading.Event()

    def take_other_key() -> None:
        with cache.lock("b"):
            entered.set()

    with cache.lock("a"):
        thread = threading.Thread(target=take_other_key)
        thread.start()
        assert entered.wait(timeout=WAIT_S)
    thread.join(timeout=WAIT_S)


def test_lock_is_released_when_its_body_raises(cache):
    with pytest.raises(RuntimeError, match="call failed"):
        with cache.lock("k"):
            raise RuntimeError("call failed")

    acquired = threading.Event()

    def take_it_again() -> None:
        with cache.lock("k"):
            acquired.set()

    thread = threading.Thread(target=take_it_again, daemon=True)
    thread.start()
    assert acquired.wait(timeout=WAIT_S)
    thread.join(timeout=WAIT_S)


# --------------------------------------------------------------------------- #
# SpendTracker                                                                 #
# --------------------------------------------------------------------------- #
def test_spend_tracker_raises_once_the_total_passes_the_limit():
    spend = SpendTracker(1.0)
    spend.add(0.5)
    spend.add(0.5)
    spend.check()  # exactly at the limit has not passed it
    assert spend.total == 1.0

    with pytest.raises(SpendLimitExceeded, match="EVAL_MAX_SPEND_USD"):
        spend.add(0.25)

    assert spend.total == 1.25  # the money is spent whether or not the cap complains
    with pytest.raises(SpendLimitExceeded):
        spend.check()
    with pytest.raises(SpendLimitExceeded):
        spend.add(0.0)


def test_spend_tracker_with_a_zero_limit_allows_only_free_calls():
    spend = SpendTracker(0)
    spend.check()
    spend.add(0.0)
    with pytest.raises(SpendLimitExceeded):
        spend.add(0.000001)


@pytest.mark.parametrize("limit", [-1.0, float("nan")])
def test_spend_tracker_rejects_a_limit_that_cannot_cap(limit):
    with pytest.raises(ValueError):
        SpendTracker(limit)


@pytest.mark.parametrize("cost", [-0.01, float("nan"), float("inf")])
def test_spend_tracker_rejects_a_cost_that_would_corrupt_the_total(cost):
    spend = SpendTracker(10.0)
    with pytest.raises(ValueError):
        spend.add(cost)
    assert spend.total == 0.0


def test_spend_tracker_counts_every_add_from_many_threads():
    spend = SpendTracker(1e9)
    workers, adds = 8, 500

    def work(_: int) -> None:
        for _ in range(adds):
            spend.add(0.25)  # sums exactly in binary

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, range(workers)))

    assert spend.total == workers * adds * 0.25
