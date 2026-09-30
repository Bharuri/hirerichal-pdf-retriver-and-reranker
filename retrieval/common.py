"""Shared query normalization and source-traceable retrieval results."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from ingestion.chunker import RetrievalChunk


_QUERY_TERM = re.compile(r"[\w+#]+(?:[./:-][\w+#]+)*", re.UNICODE)
_KEYWORD_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "may",
        "might",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
    }
)


class RetrievalError(RuntimeError):
    """Raised when a configured local retrieval operation cannot complete."""


@dataclass(frozen=True)
class ContextEvidence:
    """A related source chunk added as context, with its structural relation."""

    chunk: RetrievalChunk
    relation: Literal["parent", "child", "neighbor"]


@dataclass(frozen=True)
class RetrievalHit:
    """Ranked chunk evidence with source metadata and optional channel scores."""

    chunk: RetrievalChunk
    source_filename: str
    source_path: str
    rank: int
    score: float
    semantic_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float | None = None
    reranker_score: float | None = None
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    context_chunks: tuple[ContextEvidence, ...] = ()

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def page_start(self) -> int:
        return self.chunk.page_start

    @property
    def page_end(self) -> int:
        return self.chunk.page_end

    @property
    def section_path(self) -> tuple[str, ...]:
        return self.chunk.section_path

    @property
    def text(self) -> str:
        return self.chunk.text


def normalize_query(query: str) -> str:
    """Apply Unicode compatibility normalization, case-folding and whitespace collapse."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def query_terms(normalized_query: str) -> tuple[str, ...]:
    """Extract distinct technical terms without stemming."""
    return tuple(
        term
        for term in dict.fromkeys(_QUERY_TERM.findall(normalized_query))
        if term not in _KEYWORD_STOP_WORDS
    )