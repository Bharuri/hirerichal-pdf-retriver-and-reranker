"""Embedding generation and persistence used by the local indexing flow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Protocol, Sequence

from config import Settings
from db.duckdb import DuckDBStore, EmbeddingRecord
from ingestion.chunker import RetrievalChunk


class EmbeddingProvider(Protocol):
	"""Small adapter contract for a configured embedding model."""

	provider_name: str
	model_name: str
	model_version: str

	def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
		"""Return one vector per input text, in the same order."""


class SentenceTransformerProvider:
	"""Local embedding adapter backed by the ``sentence-transformers`` package."""

	provider_name = "sentence-transformers"

	def __init__(
		self,
		model_name: str,
		*,
		revision: str | None = None,
		device: str | None = None,
		model: object | None = None,
	) -> None:
		if not model_name.strip():
			raise ValueError("Sentence Transformer model name must not be empty")
		self.model_name = model_name.strip()
		self._model = (
			model
			if model is not None
			else self._load_model(self.model_name, revision, device)
		)
		self.model_version = (
			self._resolved_revision(self._model) or revision or "unversioned"
		)

	def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
		"""Encode text batches to CPU-side Python float vectors."""
		if not texts:
			return ()
		if any(not isinstance(text, str) for text in texts):
			raise TypeError("Sentence Transformer inputs must all be strings")
		try:
			encoded = self._model.encode(  # type: ignore[attr-defined]
				list(texts),
				convert_to_numpy=True,
				normalize_embeddings=False,
				show_progress_bar=False,
			)
		except Exception as error:
			raise RuntimeError("Sentence Transformer encoding failed.") from error
		try:
			vectors = tuple(
				tuple(float(value) for value in row)
				for row in encoded
			)
		except (TypeError, ValueError) as error:
			raise RuntimeError("Sentence Transformer returned invalid vectors.") from error
		if len(vectors) != len(texts):
			raise RuntimeError("Sentence Transformer returned an unexpected vector count.")
		return vectors

	@staticmethod
	def _load_model(model_name: str, revision: str | None, device: str | None) -> object:
		try:
			from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
		except ImportError as error:
			raise RuntimeError(
				"Sentence Transformers is not installed. Install project requirements first."
			) from error
		try:
			return SentenceTransformer(model_name, revision=revision, device=device)
		except Exception as error:
			raise RuntimeError(
				"The configured Sentence Transformer model could not be loaded."
			) from error

	@staticmethod
	def _resolved_revision(model: object) -> str | None:
		modules = getattr(model, "_modules", {})
		if not isinstance(modules, dict):
			return None
		for module in modules.values():
			auto_model = getattr(module, "auto_model", None)
			config = getattr(auto_model, "config", None)
			commit_hash = getattr(config, "_commit_hash", None)
			if isinstance(commit_hash, str) and commit_hash.strip():
				return commit_hash.strip()
		return None


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
	"""Build the supported provider selected by local embedding settings."""
	if settings.embedding_provider is None or settings.embedding_model is None:
		raise ValueError("embedding provider and model must be configured in settings")
	if settings.embedding_provider != SentenceTransformerProvider.provider_name:
		raise ValueError(
			f"Unsupported embedding provider: {settings.embedding_provider}"
		)
	return SentenceTransformerProvider(
		settings.embedding_model,
		revision=settings.embedding_model_version,
	)


@dataclass(frozen=True)
class EmbeddingFailure:
	"""Safe per-chunk failure summary; provider exception details are not exposed."""

	chunk_id: str
	reason: str


@dataclass(frozen=True)
class EmbeddingResult:
	"""Vectors successfully available after generation and reuse checks."""

	embeddings: tuple[EmbeddingRecord, ...]
	reused_count: int
	failures: tuple[EmbeddingFailure, ...]


def embed_chunks(
	chunks: Sequence[RetrievalChunk],
	store: DuckDBStore,
	provider: EmbeddingProvider | None = None,
	*,
	batch_size: int = 32,
	max_retries: int = 1,
	settings: Settings | None = None,
	on_embedding_start: Callable[[RetrievalChunk, int, int], None] | None = None,
) -> EmbeddingResult:
	"""Generate, reuse, validate, and persist one vector per final retrieval chunk.

	Embedding input combines the chunk's section path and source text. Existing
	vectors are reused only for the same chunk content and provider/model/version.
	Provider failures are retried one chunk at a time so one bad input does not
	discard successful embeddings from the rest of its batch.
	"""
	if settings is None and provider is None:
		settings = Settings.from_environment()
	if provider is None:
		if settings is None:
			raise ValueError("embedding settings are required when no provider is supplied")
		provider = create_embedding_provider(settings)
	_validate_provider(provider)
	if batch_size <= 0:
		raise ValueError("batch_size must be greater than zero")
	if max_retries < 0:
		raise ValueError("max_retries must be non-negative")

	chunk_ids = [chunk.chunk_id for chunk in chunks]
	if len(chunk_ids) != len(set(chunk_ids)):
		raise ValueError("chunk IDs must be unique in an embedding request")

	provider_name = provider.provider_name.strip()
	model_name = provider.model_name.strip()
	model_version = provider.model_version.strip()
	if settings is not None:
		if settings.embedding_provider is None or settings.embedding_model is None:
			raise ValueError("embedding provider and model must be configured in settings")
		if (
			settings.embedding_provider != provider_name
			or settings.embedding_model != model_name
		):
			raise ValueError("embedding provider does not match the configured provider and model")
	result_by_id: dict[str, EmbeddingRecord] = {}
	missing: list[RetrievalChunk] = []
	chunk_positions = {chunk.chunk_id: index for index, chunk in enumerate(chunks, start=1)}
	for chunk in chunks:
		cached = store.get_embedding(
			chunk.chunk_id,
			provider_name,
			model_name,
			model_version,
			chunk.content_hash,
		)
		if cached is None:
			missing.append(chunk)
		else:
			result_by_id[chunk.chunk_id] = cached

	expected_dimension = _cached_dimension(tuple(result_by_id.values()))
	generated: list[EmbeddingRecord] = []
	failures: list[EmbeddingFailure] = []
	for offset in range(0, len(missing), batch_size):
		batch = missing[offset : offset + batch_size]
		if on_embedding_start is not None:
			for chunk in batch:
				on_embedding_start(chunk, chunk_positions[chunk.chunk_id], len(chunks))
		try:
			vectors = _request_vectors(
				provider,
				[_embedding_input(chunk) for chunk in batch],
				expected_dimension,
			)
		except Exception:
			# A batch failure is isolated by retrying its items individually.
			for chunk in batch:
				record, dimension, failure = _embed_one(
					chunk,
					provider,
					provider_name,
					model_name,
					model_version,
					expected_dimension,
					max_retries,
				)
				expected_dimension = expected_dimension or dimension
				if record is not None:
					generated.append(record)
					result_by_id[chunk.chunk_id] = record
				elif failure is not None:
					failures.append(failure)
			continue

		for chunk, vector in zip(batch, vectors, strict=True):
			record = EmbeddingRecord(
				chunk_id=chunk.chunk_id,
				provider_name=provider_name,
				model_name=model_name,
				model_version=model_version,
				content_hash=chunk.content_hash,
				vector=vector,
			)
			generated.append(record)
			result_by_id[chunk.chunk_id] = record
			expected_dimension = expected_dimension or len(vector)

	if generated:
		store.save_embeddings(generated)

	return EmbeddingResult(
		embeddings=tuple(
			result_by_id[chunk.chunk_id]
			for chunk in chunks
			if chunk.chunk_id in result_by_id
		),
		reused_count=len(chunks) - len(missing),
		failures=tuple(failures),
	)


def _validate_provider(provider: EmbeddingProvider) -> None:
	for field in ("provider_name", "model_name", "model_version"):
		value = getattr(provider, field, None)
		if not isinstance(value, str) or not value.strip():
			raise ValueError(f"embedding provider {field} must not be empty")
	if not callable(getattr(provider, "embed", None)):
		raise TypeError("embedding provider must implement embed(texts)")


def _embedding_input(chunk: RetrievalChunk) -> str:
	section_context = " > ".join(part.strip() for part in chunk.section_path if part.strip())
	return f"{section_context}\n\n{chunk.text}" if section_context else chunk.text


def _request_vectors(
	provider: EmbeddingProvider,
	texts: Sequence[str],
	expected_dimension: int | None,
) -> list[tuple[float, ...]]:
	raw_vectors = provider.embed(texts)
	if isinstance(raw_vectors, (str, bytes)):
		raise ValueError("provider returned an invalid batch")
	vectors = list(raw_vectors)
	if len(vectors) != len(texts):
		raise ValueError("provider returned a different number of vectors than inputs")
	validated = [_validate_vector(vector) for vector in vectors]
	dimensions = {len(vector) for vector in validated}
	if len(dimensions) != 1:
		raise ValueError("provider returned vectors with inconsistent dimensions")
	dimension = next(iter(dimensions))
	if expected_dimension is not None and dimension != expected_dimension:
		raise ValueError("provider vector dimension changed for the configured model")
	return validated


def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
	if isinstance(vector, (str, bytes)):
		raise ValueError("provider returned a non-numeric vector")
	values: list[float] = []
	for value in vector:
		if isinstance(value, bool) or not isinstance(value, Real):
			raise ValueError("provider returned a non-numeric vector")
		converted = float(value)
		if not math.isfinite(converted):
			raise ValueError("provider returned a non-finite vector value")
		values.append(converted)
	if not values:
		raise ValueError("provider returned an empty vector")
	return tuple(values)


def _cached_dimension(embeddings: Sequence[EmbeddingRecord]) -> int | None:
	dimensions = {len(item.vector) for item in embeddings}
	if len(dimensions) > 1:
		raise ValueError("stored vectors have inconsistent dimensions for one model")
	return next(iter(dimensions), None)


def _embed_one(
	chunk: RetrievalChunk,
	provider: EmbeddingProvider,
	provider_name: str,
	model_name: str,
	model_version: str,
	expected_dimension: int | None,
	max_retries: int,
) -> tuple[EmbeddingRecord | None, int | None, EmbeddingFailure | None]:
	last_error: Exception | None = None
	for _ in range(max_retries + 1):
		try:
			vectors = _request_vectors(
				provider, [_embedding_input(chunk)], expected_dimension
			)
			vector = vectors[0]
			return (
				EmbeddingRecord(
					chunk_id=chunk.chunk_id,
					provider_name=provider_name,
					model_name=model_name,
					model_version=model_version,
					content_hash=chunk.content_hash,
					vector=vector,
				),
				len(vector),
				None,
			)
		except Exception as error:
			last_error = error

	error_type = type(last_error).__name__ if last_error is not None else "ProviderError"
	return (
		None,
		None,
		EmbeddingFailure(
			chunk.chunk_id,
			f"Embedding failed after {max_retries + 1} attempt(s) ({error_type}).",
		),
	)
