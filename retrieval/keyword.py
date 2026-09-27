"""Keyword retrieval over chunk text and hierarchy metadata."""

from __future__ import annotations

import re

from config import Settings
from db.duckdb import DuckDBStore
from retrieval.common import RetrievalHit, normalize_query, query_terms


_WORD_BOUNDARY = r"(^|[^[:alnum:]_]){}([^[:alnum:]_]|$)"


def keyword_search(
	store: DuckDBStore,
	query: str,
	*,
	top_k: int | None = None,
	settings: Settings | None = None,
) -> tuple[RetrievalHit, ...]:
	"""Rank exact query-term matches in chunk text, section title and section path.

	Chunk text contributes one point per matched term; section titles and paths
	contribute two points. Terms are matched case-insensitively with token
	boundaries, so a technical identifier is not matched as a substring of a
	different identifier.
	"""
	limit = top_k if top_k is not None else (settings.keyword_top_k if settings else 20)
	if limit <= 0:
		raise ValueError("top_k must be greater than zero")
	normalized = normalize_query(query)
	terms = query_terms(normalized)
	if not terms:
		return ()

	patterns = tuple(
		_WORD_BOUNDARY.format(re.escape(term)) for term in terms
	)
	rows = store.search_keyword_rows(patterns, limit)
	return tuple(
		RetrievalHit(
			chunk=stored.chunk,
			source_filename=stored.source_filename,
			source_path=stored.source_path,
			rank=rank,
			score=score,
			keyword_score=score,
			keyword_rank=rank,
		)
		for rank, (stored, score) in enumerate(rows, start=1)
	)
