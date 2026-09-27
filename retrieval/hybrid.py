"""Configurable, deduplicated fusion of semantic and keyword candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from config import Settings
from db.duckdb import DuckDBStore
from ingestion.indexer import EmbeddingProvider
from retrieval.common import RetrievalHit
from retrieval.keyword import keyword_search
from retrieval.semantic import semantic_search


@dataclass
class _Channels:
	hit: RetrievalHit
	keyword: RetrievalHit | None = None
	semantic: RetrievalHit | None = None


def fuse_results(
	keyword_results: Sequence[RetrievalHit],
	semantic_results: Sequence[RetrievalHit],
	*,
	top_k: int | None = None,
	method: str | None = None,
	rrf_k: int | None = None,
	semantic_weight: float | None = None,
	keyword_weight: float | None = None,
	settings: Settings | None = None,
) -> tuple[RetrievalHit, ...]:
	"""Fuse ranked lists using weighted RRF by default or normalized scores."""
	limit = top_k if top_k is not None else (settings.hybrid_top_k if settings else 10)
	selected_method = method or (settings.hybrid_fusion_method if settings else "rrf")
	rank_constant = rrf_k if rrf_k is not None else (settings.rrf_k if settings else 60)
	sem_weight = (
		semantic_weight
		if semantic_weight is not None
		else (settings.hybrid_semantic_weight if settings else 1.0)
	)
	key_weight = (
		keyword_weight
		if keyword_weight is not None
		else (settings.hybrid_keyword_weight if settings else 1.0)
	)
	_validate_fusion_parameters(limit, selected_method, rank_constant, sem_weight, key_weight)

	channels: dict[str, _Channels] = {}
	keyword_by_id = _deduplicate(keyword_results)
	semantic_by_id = _deduplicate(semantic_results)
	for chunk_id in sorted(set(keyword_by_id) | set(semantic_by_id)):
		keyword_hit = keyword_by_id.get(chunk_id)
		semantic_hit = semantic_by_id.get(chunk_id)
		base_hit = semantic_hit or keyword_hit
		assert base_hit is not None
		channels[chunk_id] = _Channels(
			hit=base_hit, keyword=keyword_hit, semantic=semantic_hit
		)

	if selected_method == "weighted_sum":
		keyword_scores = _normalized_scores(keyword_by_id)
		semantic_scores = _normalized_scores(semantic_by_id)

	ranked: list[tuple[float, _Channels]] = []
	for chunk_id, item in channels.items():
		if selected_method == "rrf":
			score = 0.0
			if item.keyword is not None:
				score += key_weight / (rank_constant + item.keyword.rank)
			if item.semantic is not None:
				score += sem_weight / (rank_constant + item.semantic.rank)
		else:
			score = (
				key_weight * keyword_scores.get(chunk_id, 0.0)
				+ sem_weight * semantic_scores.get(chunk_id, 0.0)
			)
		ranked.append((score, item))

	ranked.sort(key=lambda pair: _fusion_sort_key(pair[0], pair[1]))
	hits: list[RetrievalHit] = []
	for rank, (score, item) in enumerate(ranked[:limit], start=1):
		hits.append(
			RetrievalHit(
				chunk=item.hit.chunk,
				source_filename=item.hit.source_filename,
				source_path=item.hit.source_path,
				rank=rank,
				score=score,
				semantic_score=item.semantic.score if item.semantic else None,
				keyword_score=item.keyword.score if item.keyword else None,
				hybrid_score=score,
				semantic_rank=item.semantic.rank if item.semantic else None,
				keyword_rank=item.keyword.rank if item.keyword else None,
			)
		)
	return tuple(hits)


def hybrid_search(
	store: DuckDBStore,
	query: str,
	embedding_provider: EmbeddingProvider | None = None,
	*,
	top_k: int | None = None,
	keyword_top_k: int | None = None,
	semantic_top_k: int | None = None,
	method: str | None = None,
	rrf_k: int | None = None,
	semantic_weight: float | None = None,
	keyword_weight: float | None = None,
	settings: Settings | None = None,
) -> tuple[RetrievalHit, ...]:
	"""Run available retrieval channels and return fused source-traceable hits.

	Without an embedding provider, the keyword channel remains usable and its
	candidates are returned through the same fusion contract.
	"""
	keywords = keyword_search(
		store, query, top_k=keyword_top_k, settings=settings
	)
	semantics = (
		semantic_search(
			store,
			query,
			embedding_provider,
			top_k=semantic_top_k,
			settings=settings,
		)
		if embedding_provider is not None
		else ()
	)
	return fuse_results(
		keywords,
		semantics,
		top_k=top_k,
		method=method,
		rrf_k=rrf_k,
		semantic_weight=semantic_weight,
		keyword_weight=keyword_weight,
		settings=settings,
	)


def _deduplicate(results: Sequence[RetrievalHit]) -> dict[str, RetrievalHit]:
	selected: dict[str, RetrievalHit] = {}
	for hit in sorted(
		results,
		key=lambda item: (
			item.rank,
			-item.score,
			item.source_path.casefold(),
			item.chunk.sequence,
			item.chunk_id,
		),
	):
		selected.setdefault(hit.chunk_id, hit)
	return selected


def _normalized_scores(results: dict[str, RetrievalHit]) -> dict[str, float]:
	if not results:
		return {}
	scores = [hit.score for hit in results.values()]
	low, high = min(scores), max(scores)
	if math.isclose(low, high):
		return {chunk_id: 1.0 for chunk_id in results}
	return {
		chunk_id: (hit.score - low) / (high - low)
		for chunk_id, hit in results.items()
	}


def _fusion_sort_key(
	score: float, item: _Channels
) -> tuple[float, int, int, int, str, str, int, str]:
	semantic_rank = item.semantic.rank if item.semantic else 2**31
	keyword_rank = item.keyword.rank if item.keyword else 2**31
	return (
		-score,
		min(semantic_rank, keyword_rank),
		semantic_rank,
		keyword_rank,
		item.hit.source_path.casefold(),
		item.hit.document_id,
		item.hit.chunk.sequence,
		item.hit.chunk_id,
	)


def _validate_fusion_parameters(
	top_k: int,
	method: str,
	rrf_k: int,
	semantic_weight: float,
	keyword_weight: float,
) -> None:
	if top_k <= 0:
		raise ValueError("top_k must be greater than zero")
	if method not in {"rrf", "weighted_sum"}:
		raise ValueError("fusion method must be 'rrf' or 'weighted_sum'")
	if rrf_k <= 0:
		raise ValueError("rrf_k must be greater than zero")
	weights = (semantic_weight, keyword_weight)
	if any(not math.isfinite(weight) or weight < 0 for weight in weights):
		raise ValueError("fusion weights must be finite and non-negative")
	if not any(weights):
		raise ValueError("at least one fusion weight must be greater than zero")
