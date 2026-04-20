from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import re
from typing import Any, Protocol, Sequence, cast
from uuid import UUID

from ..core.config import get_settings
from ..crud.memory_store import MemoryStore
from ..schema.memory import EdgeOut, FactOut, RetrievedPersonContext, SummaryOut

settings = get_settings()

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "with",
}


class BiEncoder(Protocol):
    """Input: one query string and many candidate texts. Output: one score per text for the first retrieval pass."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        ...


class CrossEncoderReranker(Protocol):
    """Input: one query string and many candidate texts. Output: one score per text for the final reranking pass."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        ...


@dataclass(frozen=True)
class MemoryCandidate:
    """One retrievable memory item. Input to ranking is its text; output is the original fact, summary, or edge payload."""

    kind: str
    text: str
    payload: FactOut | SummaryOut | EdgeOut


class LexicalBiEncoder:
    """Fallback scorer. Input: query plus candidate texts. Output: simple lexical similarity scores."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        query_weights = _token_weights(query)
        if not query_weights:
            return [0.0 for _ in texts]

        return [_weighted_overlap_score(query_weights, _token_weights(text)) for text in texts]


class LexicalCrossEncoder:
    """Fallback reranker. Input: query plus candidate texts. Output: slightly stricter lexical scores for final ordering."""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        normalized_query = query.strip().lower()
        query_weights = _token_weights(query)
        if not normalized_query and not query_weights:
            return [0.0 for _ in texts]

        scores: list[float] = []
        for text in texts:
            text_weights = _token_weights(text)
            overlap = _weighted_overlap_score(query_weights, text_weights)
            phrase_bonus = 0.25 if normalized_query and normalized_query in text.lower() else 0.0
            scores.append(overlap + phrase_bonus)
        return scores


class BGEM3BiEncoder:
    """Model-backed first-pass retriever. Input: query and candidate texts. Output: coarse relevance scores."""

    def __init__(self, model_name: str = settings.retrieval_bi_encoder_model) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(
            [query, *texts],
            normalize_embeddings=True,
        )
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        return [
            float(sum(float(a) * float(b) for a, b in zip(query_embedding, candidate)))
            for candidate in candidate_embeddings
        ]


class BGEReranker:
    """Model-backed reranker. Input: query and candidate texts. Output: final relevance scores."""

    def __init__(self, model_name: str = settings.retrieval_reranker_model) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []

        model = self._load_model()
        pairs = [(query, text) for text in texts]
        predictions = model.predict(pairs)
        return [float(score) for score in predictions]


def retrieve_person_context(
    person_id: UUID,
    query: str,
    store: MemoryStore | None = None,
    bi_encoder: BiEncoder | None = None,
    reranker: CrossEncoderReranker | None = None,
    bi_encoder_top_k: int = settings.retrieval_bi_encoder_top_k,
    bi_encoder_min_score: float = settings.retrieval_bi_encoder_min_score,
    reranker_top_k: int = settings.retrieval_reranker_top_k,
    reranker_min_score: float = settings.retrieval_reranker_min_score,
) -> RetrievedPersonContext:
    """Input: a person ID, the user's question, and optional store/scorers.

    Output: a ``RetrievedPersonContext`` containing the most relevant facts,
    summaries, and outgoing edges for that person.
    """

    owns_store = store is None
    memory_store = store or MemoryStore()
    if owns_store:
        memory_store.initialize()

    try:
        profile = memory_store.get_profile_context(person_id)
        candidates = _build_candidates(profile.facts, profile.summaries, profile.edges_from)

        if not query.strip():
            top_candidates = candidates[:reranker_top_k]
        else:
            bi_encoder_impl = bi_encoder or _default_bi_encoder()
            reranker_impl = reranker or _default_reranker()
            try:
                top_candidates = _retrieve_top_candidates(
                    query=query,
                    candidates=candidates,
                    bi_encoder=bi_encoder_impl,
                    reranker=reranker_impl,
                    bi_encoder_top_k=bi_encoder_top_k,
                    bi_encoder_min_score=bi_encoder_min_score,
                    reranker_top_k=reranker_top_k,
                    reranker_min_score=reranker_min_score,
                )
            except Exception:
                if bi_encoder is not None or reranker is not None:
                    raise
                top_candidates = _retrieve_top_candidates(
                    query=query,
                    candidates=candidates,
                    bi_encoder=LexicalBiEncoder(),
                    reranker=LexicalCrossEncoder(),
                    bi_encoder_top_k=bi_encoder_top_k,
                    bi_encoder_min_score=bi_encoder_min_score,
                    reranker_top_k=reranker_top_k,
                    reranker_min_score=reranker_min_score,
                )

        facts: list[FactOut] = []
        summaries: list[SummaryOut] = []
        edges: list[EdgeOut] = []

        for candidate in top_candidates:
            if candidate.kind == "fact":
                facts.append(cast(FactOut, candidate.payload))
            elif candidate.kind == "summary":
                summaries.append(cast(SummaryOut, candidate.payload))
            elif candidate.kind == "edge":
                edges.append(cast(EdgeOut, candidate.payload))

        return RetrievedPersonContext(
            person_id=person_id,
            facts=facts,
            summaries=summaries,
            edges=edges,
        )
    finally:
        if owns_store:
            memory_store.close()


def _build_candidates(
    facts: Sequence[FactOut],
    summaries: Sequence[SummaryOut],
    edges: Sequence[EdgeOut],
) -> list[MemoryCandidate]:
    """Input: separate fact, summary, and edge lists. Output: one combined candidate list used for ranking."""
    candidates: list[MemoryCandidate] = []

    candidates.extend(
        MemoryCandidate(
            kind="fact",
            text=f"{fact.fact_category} {fact.fact_text}".strip(),
            payload=fact,
        )
        for fact in facts
    )
    candidates.extend(
        MemoryCandidate(kind="summary", text=summary.summary_text, payload=summary)
        for summary in summaries
    )
    candidates.extend(
        MemoryCandidate(
            kind="edge",
            text=f"{edge.relation} {edge.dst_name}".strip(),
            payload=edge,
        )
        for edge in edges
    )

    return candidates


def _retrieve_top_candidates(
    query: str,
    candidates: Sequence[MemoryCandidate],
    bi_encoder: BiEncoder,
    reranker: CrossEncoderReranker,
    bi_encoder_top_k: int,
    bi_encoder_min_score: float,
    reranker_top_k: int,
    reranker_min_score: float,
) -> list[MemoryCandidate]:
    """Input: query plus all candidates and scoring settings. Output: the highest-ranked candidates after both retrieval stages."""
    if not candidates:
        return []

    bi_scores = bi_encoder.score(query, [candidate.text for candidate in candidates])
    coarse_candidates = _take_top(
        candidates,
        bi_scores,
        bi_encoder_top_k,
        min_score=bi_encoder_min_score,
    )
    if not coarse_candidates:
        return []

    rerank_scores = reranker.score(query, [candidate.text for candidate in coarse_candidates])
    return _take_top(
        coarse_candidates,
        rerank_scores,
        reranker_top_k,
        min_score=reranker_min_score,
    )


def _take_top(
    candidates: Sequence[MemoryCandidate],
    scores: Sequence[float],
    limit: int,
    min_score: float | None = None,
) -> list[MemoryCandidate]:
    """Input: candidates with scores. Output: the best candidates up to ``limit``, optionally dropping low-score items."""
    ranked = list(zip(candidates, scores, range(len(candidates))))
    if min_score is not None:
        ranked = [row for row in ranked if row[1] > min_score]
    ranked.sort(key=lambda row: (row[1], -row[2]), reverse=True)
    return [candidate for candidate, _, _ in ranked[:limit]]


def _default_bi_encoder() -> BiEncoder:
    if importlib.util.find_spec("sentence_transformers") is None:
        return LexicalBiEncoder()
    return BGEM3BiEncoder()


def _default_reranker() -> CrossEncoderReranker:
    if importlib.util.find_spec("sentence_transformers") is None:
        return LexicalCrossEncoder()
    return BGEReranker()


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("is", "ss")):
        return token[:-1]
    return token


def _token_weights(text: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    for raw_token in re.findall(r"[a-z0-9]+", text.lower()):
        if raw_token in _STOP_WORDS:
            continue
        token = _normalize_token(raw_token)
        counts[token] = counts.get(token, 0) + 1

    return {
        token: 1.0 + math.log(count)
        for token, count in counts.items()
    }


def _weighted_overlap_score(
    query_weights: dict[str, float],
    text_weights: dict[str, float],
) -> float:
    if not query_weights or not text_weights:
        return 0.0

    overlap = set(query_weights) & set(text_weights)
    if not overlap:
        return 0.0

    numerator = sum(min(query_weights[token], text_weights[token]) for token in overlap)
    denominator = math.sqrt(sum(weight * weight for weight in query_weights.values()))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
