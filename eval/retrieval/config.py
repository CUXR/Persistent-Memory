"""Constants and environment-driven settings for the retrieval benchmark.

Everything that must stay fixed for the numbers to reproduce (seed, reference
"now", dataset shape, k grids) is a module constant. Everything that depends on
the machine or account (database URL, gateway keys, model ids) comes from the
environment, optionally via ``eval/.env`` (gitignored).

The default gateway is OpenAI's own API, the one hosted service the project
pays for. Jev is served by the routers alone, so its rankers are opt-in.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from .schema import QUERY_TYPES

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
EVAL_ROOT = REPO_ROOT / "eval"
RETRIEVAL_ROOT = EVAL_ROOT / "retrieval"
DATA_DIR = RETRIEVAL_ROOT / "data"
RESULTS_DIR = RETRIEVAL_ROOT / "results"
CACHE_DIR = RETRIEVAL_ROOT / ".cache"  # gitignored: API response cache, embeddings, raw run outputs

PEOPLE_PATH = DATA_DIR / "people.jsonl"
CONVERSATIONS_PATH = DATA_DIR / "conversations.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
# The judge-validation summary ``judge_validation score`` writes and the report includes.
JUDGE_VALIDATION_PATH = DATA_DIR / "judge_validation.json"

# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #
SEED = 20261104
# The benchmark's fixed "now" (a Wednesday). Relative time expressions in
# queries, time-filter padding, and the dataset's last day all use this.
NOW = datetime(2026, 11, 4, 18, 0, 0, tzinfo=timezone.utc)
DATASET_START = datetime(2025, 11, 5, 0, 0, 0, tzinfo=timezone.utc)  # ~12 months before NOW
ID_NAMESPACE = uuid.UUID("6f1d2c1e-7a55-4f0b-9c0e-2b1f4a6d9e10")


def stable_id(kind: str, key: str | int) -> str:
    """Deterministic UUID for a dataset entity (same seed, same ids)."""

    return str(uuid.uuid5(ID_NAMESPACE, f"{SEED}:{kind}:{key}"))


# --------------------------------------------------------------------------- #
# Dataset shape                                                                #
# --------------------------------------------------------------------------- #
OWNER_ID = stable_id("owner", 0)
OWNER_FIRST_NAME = "Riley"
OWNER_LAST_NAME = "Morgan"
OWNER_DISPLAY_NAME = "Riley Morgan"
OWNER_USERNAME = "eval-riley-morgan"

N_PEOPLE = 40
N_CONVERSATIONS_TARGET = 800
HEAVY_PERSON_COUNTS = (150, 148, 146, 142)  # conversations for the four "heavy" people
LIGHT_PERSON_RANGE = (1, 10)  # conversations for everyone else
SUMMARY_WORD_RANGE = (20, 100)

QUERY_TYPE_COUNTS: dict[str, int] = {
    "lookup": 50,
    "paraphrase": 45,
    "detail_only": 45,
    "multi_conversation": 40,
    "cross_person": 35,
    "temporal": 40,
    "unanswerable": 30,  # ~1 in 10
    "unknown_name": 15,  # ~1 in 20
}
# The counts name the schema's query types, in the schema's order: one vocabulary.
assert tuple(QUERY_TYPE_COUNTS) == QUERY_TYPES, (tuple(QUERY_TYPE_COUNTS), QUERY_TYPES)
N_QUERIES = sum(QUERY_TYPE_COUNTS.values())  # 300
DEV_FRACTION = 0.10  # 30 dev / 270 test, stratified by type

# --------------------------------------------------------------------------- #
# Query flow                                                                   #
# --------------------------------------------------------------------------- #
UNKNOWN_NAME_TEMPLATE = "There is no one in your memory named {name}."
AMBIGUOUS_NAME_TEMPLATE = "More than one person in your memory matches {name}: {candidates}."
DECLINE_ANSWER = "I don't know."
PAD_FRACTION = 0.20
PAD_MIN_HOURS = 4

# --------------------------------------------------------------------------- #
# Rankers and evaluation grid                                                  #
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
STAGE1_TOP_K = 100
RANKER_NAMES: tuple[str, ...] = ("bge_m3", "bge_m3_reranker", "bge_m3_jev", "jev_full_pool")
JEV_RANKERS: tuple[str, ...] = ("bge_m3_jev", "jev_full_pool")
# The rankers that run on this machine alone: no hosted call, no spend.
LOCAL_RANKERS: tuple[str, ...] = tuple(name for name in RANKER_NAMES if name not in JEV_RANKERS)
RECALL_KS: tuple[int, ...] = (5, 10, 20, 30)
SYSTEM_KS: tuple[int, ...] = (5, 10, 20)
FALLBACK_KS: tuple[int, ...] = (3, 5, 10)
LABEL_POOL_DEPTH = 30  # top-N per ranker pooled for the missed-gold check
JEV_VARIANCE_RUNS = 3
BOOTSTRAP_RESAMPLES = 5000
CI_LEVEL = 0.95

# USD per million tokens (input, output). Used only when the gateway does not
# report a cost for a call, which OpenAI's own API never does. A model id is a
# gateway's own spelling: "gpt-5-nano" on OpenAI, "openai/gpt-5-nano" on a router.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # OpenRouter list prices on 2026-11-06.
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "anthropic/claude-opus-5.5": (4.00, 20.00),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "openai/gpt-5-nano": (0.05, 0.40),
    "openai/gpt-5-mini": (0.25, 2.00),
    "openai/gpt-6-luna": (0.10, 0.50),
    "openai/gpt-6.1-terra": (1.00, 5.00),
    # OpenAI standard-tier list prices on 2026-11-15 (cached input is cheaper:
    # charging the full input price keeps the spend figure an upper bound).
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-4o-mini": (0.15, 0.60),
}
JEV_PRICE_USD_PER_MTOK_INPUT = 0.042  # output tokens are free

# The chat models a router serves by default, and the ones OpenAI's own API does.
_ROUTER_MODELS: dict[str, str] = {
    "gen_model": "anthropic/claude-sonnet-5.5",
    "summary_model": "openai/gpt-5-nano",
    "system_model": "openai/gpt-5-nano",
    "judge_model": "anthropic/claude-sonnet-5.5",
}
_OPENAI_MODELS: dict[str, str] = {
    "gen_model": "gpt-6.1-terra",
    "summary_model": "gpt-5-nano",
    "system_model": "gpt-5-nano",
    "judge_model": "gpt-6.1-terra",
}

DEFAULT_GATEWAY = "openai"
# Per gateway: the chat-completions base URL, the environment variable that
# holds its API key (``key_env``), the decisions endpoint, the Jev model id and
# the default chat models. OpenAI does not serve Jev (an empty ``jev_url``), so
# on that gateway only ``LOCAL_RANKERS`` run; the routers bill separately.
GATEWAYS: dict[str, dict[str, str]] = {
    "openai": {
        "llm_base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "jev_url": "",
        "jev_model": "",
        **_OPENAI_MODELS,
    },
    "openrouter": {
        "llm_base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "jev_url": "https://openrouter.ai/api/alpha/decisions",
        "jev_model": "typesafe/jev-1.14",
        **_ROUTER_MODELS,
    },
    "vercel": {
        "llm_base_url": "https://ai-gateway.vercel.sh/v1",
        "key_env": "AI_GATEWAY_API_KEY",
        "jev_url": "https://ai-gateway.vercel.sh/v1/decisions",
        "jev_model": "typesafe-ai/jev-1.14.0",
        **_ROUTER_MODELS,
    },
}


@dataclass(frozen=True)
class Settings:
    """Machine- and account-specific settings (see ``eval/.env.example``)."""

    database_url: str | None  # Neon eval branch, direct (non-pooler) connection string
    gateway: str  # "openai" | "openrouter" | "vercel"
    llm_base_url: str
    llm_api_key: str | None
    gen_model: str  # personas, dialogues, queries
    summary_model: str  # the 20-100 word conversation summary (separate call)
    system_model: str  # extraction + answering (the system under test)
    judge_model: str  # answer judge + relevance judge for label pooling
    jev_url: str  # empty on a gateway that does not serve Jev
    jev_api_key: str | None
    jev_model: str  # pinned Jev version
    llm_concurrency: int
    jev_concurrency: int
    device: str | None  # torch device for local models (None = auto: mps/cuda/cpu)
    max_spend_usd: float  # hard cap on hosted-API spend per command

    @property
    def rankers(self) -> tuple[str, ...]:
        """The rankers this gateway can run: the Jev ones need its decisions endpoint."""

        return RANKER_NAMES if self.jev_url else LOCAL_RANKERS


def _load_dotenv() -> None:
    if os.environ.get("EVAL_DISABLE_DOTENV"):
        return
    env_path = EVAL_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(env_path, override=False)


@lru_cache
def get_settings() -> Settings:
    """Read settings from the environment (and ``eval/.env`` when present)."""

    _load_dotenv()
    env = os.environ
    gateway = env.get("EVAL_GATEWAY", DEFAULT_GATEWAY).strip().lower()
    if gateway not in GATEWAYS:
        raise ValueError(f"EVAL_GATEWAY must be one of {sorted(GATEWAYS)}, got {gateway!r}")
    defaults = GATEWAYS[gateway]
    api_key = env.get(defaults["key_env"]) or None
    return Settings(
        database_url=env.get("EVAL_DATABASE_URL") or None,
        gateway=gateway,
        llm_base_url=env.get("EVAL_LLM_BASE_URL", defaults["llm_base_url"]),
        llm_api_key=env.get("EVAL_LLM_API_KEY") or api_key,
        gen_model=env.get("EVAL_GEN_MODEL", defaults["gen_model"]),
        summary_model=env.get("EVAL_SUMMARY_MODEL", defaults["summary_model"]),
        system_model=env.get("EVAL_SYSTEM_MODEL", defaults["system_model"]),
        judge_model=env.get("EVAL_JUDGE_MODEL", defaults["judge_model"]),
        # EVAL_JEV_URL moves a gateway's decisions endpoint; it cannot give one to
        # a gateway without Jev, whose request shape the client would not know.
        jev_url=env.get("EVAL_JEV_URL", defaults["jev_url"]) if defaults["jev_url"] else "",
        jev_api_key=(env.get("EVAL_JEV_API_KEY") or api_key) if defaults["jev_url"] else None,
        jev_model=env.get("EVAL_JEV_MODEL", defaults["jev_model"]),
        llm_concurrency=int(env.get("EVAL_LLM_CONCURRENCY", "8")),
        jev_concurrency=int(env.get("EVAL_JEV_CONCURRENCY", "8")),
        device=env.get("EVAL_DEVICE") or None,
        max_spend_usd=float(env.get("EVAL_MAX_SPEND_USD", "150")),
    )
