"""Build the benchmark's live components from settings, lazily, in one place.

Every CLI under ``eval/retrieval`` gets its store, models and hosted clients
from a :class:`Runtime`, so a command that only needs the LLM never loads a
model or opens a database connection, and tests can swap any piece for a fake
by assigning the attribute before first use.
"""

from __future__ import annotations

import json
import sys
from functools import cached_property
from typing import Any

from . import config
from .interfaces import ChatLLM, CorpusStore, DecisionClient, Embedder, PairScorer, Ranker

# Written by the dev-set study (``python -m eval.retrieval.jev_study``) and
# committed: how Jev is applied (and which version) for the reported numbers.
JEV_SCORING_PATH = config.RETRIEVAL_ROOT / "jev_scoring.json"
# The pin's fields that say which Jev answers (the model id, on which gateway,
# and the dated snapshots the API served the study); every other field is the
# ``JevScoringConfig`` of how it is asked.
JEV_PIN_KEYS: frozenset[str] = frozenset({"model", "gateway", "served_models"})


def load_jev_pin() -> dict[str, Any]:
    """The committed Jev application config, or an empty dict before the study has run."""

    if not JEV_SCORING_PATH.exists():
        return {}
    return json.loads(JEV_SCORING_PATH.read_text(encoding="utf-8"))


class Runtime:
    """Lazily constructed store, local models, hosted clients and rankers."""

    def __init__(
        self,
        settings: config.Settings | None = None,
        *,
        store_kind: str = "pg",
        use_cache: bool = True,
    ) -> None:
        if store_kind not in ("pg", "memory"):
            raise ValueError(f"store_kind must be 'pg' or 'memory', got {store_kind!r}")
        self.settings = settings or config.get_settings()
        self.store_kind = store_kind
        self.use_cache = use_cache

    @cached_property
    def spend(self):
        from .spend import SpendTracker

        return SpendTracker(self.settings.max_spend_usd)

    @cached_property
    def cache(self):
        if not self.use_cache:
            return None
        from .cache import DiskCache

        return DiskCache()

    @cached_property
    def llm(self) -> ChatLLM:
        from .llm import build_llm

        return build_llm(self.settings, cache=self.cache, spend=self.spend)

    @cached_property
    def jev_model(self) -> str:
        """An explicit choice, else the version the dev study pinned on this gateway, else the default.

        The choice is explicit when the settings name a model other than the
        gateway's default (``EVAL_JEV_MODEL`` reaches them through
        ``config.get_settings``). A model id is a gateway's own spelling, so a
        pin written on another gateway is not applied: the gateway's default
        is used and a warning says so.
        """

        if not self.settings.jev_url:
            return ""  # a gateway without Jev: nothing to pin, nothing to warn about
        gateway = self.settings.gateway
        default = config.GATEWAYS.get(gateway, {}).get("jev_model")
        if default is not None and self.settings.jev_model != default:
            return self.settings.jev_model
        pin = load_jev_pin()
        pinned = pin.get("model")
        if not pinned:
            return self.settings.jev_model
        pinned_gateway = pin.get("gateway")
        if pinned_gateway in (None, gateway):
            return pinned
        print(
            f"warning: {JEV_SCORING_PATH} pins Jev {pinned!r} on the {pinned_gateway} gateway, "
            f"whose model ids differ from {gateway}'s; using {self.settings.jev_model!r} "
            "(rerun the study on this gateway, or set EVAL_JEV_MODEL)",
            file=sys.stderr,
            flush=True,
        )
        return self.settings.jev_model

    @cached_property
    def jev(self) -> DecisionClient:
        from dataclasses import replace

        from .jev import build_jev

        settings = replace(self.settings, jev_model=self.jev_model)
        return build_jev(settings, cache=self.cache, spend=self.spend)

    @cached_property
    def embedder(self) -> Embedder:
        from .local_models import BgeM3Embedder

        return BgeM3Embedder(device=self.settings.device)

    @cached_property
    def scorer(self) -> PairScorer:
        from .local_models import BgeReranker

        return BgeReranker(device=self.settings.device)

    @cached_property
    def store(self) -> CorpusStore:
        if self.store_kind == "pg":
            from .store import PgCorpusStore

            return PgCorpusStore(self.settings.database_url)
        from .store import InMemoryCorpusStore

        return InMemoryCorpusStore.from_dataset(embedder=self.embedder)

    def jev_scoring(self, salt: str = ""):
        """The pinned Jev scoring config (defaults before the study), with a per-run salt."""

        from .pipelines.jev_scoring import JevScoringConfig

        pinned = {key: value for key, value in load_jev_pin().items() if key not in JEV_PIN_KEYS}
        base = JevScoringConfig.from_json(pinned) if pinned else JevScoringConfig()
        from dataclasses import replace

        return replace(base, concurrency=self.settings.jev_concurrency, salt=salt)

    def ranker(
        self,
        name: str,
        *,
        k: int = 30,
        salt: str = "",
        jev_scoring=None,
        jev: DecisionClient | None = None,
    ) -> Ranker:
        """Build one ranker, constructing only the components that ranker needs.

        A Jev ranker asks ``jev`` when one is given (the study builds one client
        per model id), else this runtime's client; ``jev_scoring`` likewise
        replaces the pinned config with ``salt``.
        """

        from .pipelines import build_ranker

        if name not in config.RANKER_NAMES:
            raise ValueError(f"unknown ranker {name!r}; expected one of {config.RANKER_NAMES}")
        uses_jev = name in config.JEV_RANKERS
        return build_ranker(
            name,
            store=self.store,
            embedder=None if name == "jev_full_pool" else self.embedder,
            scorer=self.scorer if name == "bge_m3_reranker" else None,
            jev=(self.jev if jev is None else jev) if uses_jev else None,
            jev_scoring=(jev_scoring or self.jev_scoring(salt)) if uses_jev else None,
            k=k,
        )
