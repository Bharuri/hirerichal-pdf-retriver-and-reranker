"""Semantic cosine retrieval over compatible DuckDB chunk embeddings."""

from __future__ import annotations

import math
from numbers import Real

from config import Settings
from db.duckdb import DuckDBStore, StoredChunk
from ingestion.indexer import EmbeddingProvider
from retrieval.common import RetrievalError, RetrievalHit, normalize_query


def semantic_search(
	store: DuckDBStore,
	query: str,
	provider: EmbeddingProvider,
	*,
	top_k: int | None = None,
	settings: Settings | None = None,
) -> tuple[RetrievalHit, ...]:
	"""Return the highest cosine-similarity chunks for one configured model."""
	limit = top_k if top_k is not None else (settings.semantic_top_k if settings else 20)
	if limit <= 0:
		raise ValueError("top_k must be greater than zero")
	normalized = normalize_query(query)
	if not normalized:
		return ()

	provider_name, model_name, model_version = _provider_identity(provider)
	available_dimensions = store.get_compatible_embedding_dimensions(
		provider_name, model_name, model_version
	)
	if not available_dimensions:
		return ()

	try:
		raw_vectors = provider.embed((normalized,))
		vectors = list(raw_vectors)
	except Exception as error:
		raise RetrievalError("Query embedding failed for the configured model.") from error
	if len(vectors) != 1:
		raise RetrievalError("Embedding provider must return exactly one query vector.")
	query_vector = _validated_vector(vectors[0])
	query_norm = _vector_norm(query_vector)
	if query_norm == 0:
		raise RetrievalError("Query embedding has zero magnitude and cannot be compared.")
	if len(query_vector) not in available_dimensions:
		return ()

	stored_rows = store.get_compatible_embedding_rows(
		provider_name, model_name, model_version, len(query_vector)
	)
	scored: dict[str, tuple[float, StoredChunk]] = {}
	for stored, vector in stored_rows:
		try:
			normalized_vector = _validated_vector(vector)
		except RetrievalError:
			continue
		vector_norm = _vector_norm(normalized_vector)
		if vector_norm == 0 or len(normalized_vector) != len(query_vector):
			continue
		similarity = math.fsum(
			(left / query_norm) * (right / vector_norm)
			for left, right in zip(query_vector, normalized_vector, strict=True)
		)
		similarity = max(-1.0, min(1.0, similarity))
		current = scored.get(stored.chunk.chunk_id)
		if current is None or similarity > current[0]:
			scored[stored.chunk.chunk_id] = (similarity, stored)

	ordered = sorted(
		scored.values(),
		key=lambda item: (
			-item[0],
			item[1].source_path.casefold(),
			item[1].chunk.sequence,
			item[1].chunk.chunk_id,
		),
	)[:limit]
	return tuple(
		RetrievalHit(
			chunk=stored.chunk,
			source_filename=stored.source_filename,
			source_path=stored.source_path,
			rank=rank,
			score=score,
			semantic_score=score,
			semantic_rank=rank,
		)
		for rank, (score, stored) in enumerate(ordered, start=1)
	)


def _provider_identity(provider: EmbeddingProvider) -> tuple[str, str, str]:
	identity: list[str] = []
	for field in ("provider_name", "model_name", "model_version"):
		value = getattr(provider, field, None)
		if not isinstance(value, str) or not value.strip():
			raise RetrievalError(f"Embedding provider {field} must be configured.")
		identity.append(value.strip())
	if not callable(getattr(provider, "embed", None)):
		raise RetrievalError("Embedding provider must implement embed(texts).")
	return identity[0], identity[1], identity[2]


def _validated_vector(vector: object) -> tuple[float, ...]:
	if isinstance(vector, (str, bytes)):
		raise RetrievalError("Embedding provider returned an invalid vector.")
	try:
		values = tuple(vector)  # type: ignore[arg-type]
	except TypeError as error:
		raise RetrievalError("Embedding provider returned an invalid vector.") from error
	if not values or any(
		isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
		for value in values
	):
		raise RetrievalError("Embedding vector must contain finite numeric values.")
	return tuple(float(value) for value in values)


def _vector_norm(vector: tuple[float, ...]) -> float:
	return math.hypot(*vector)
