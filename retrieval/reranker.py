"""Replaceable reranking interface with an optional Sentence Transformers adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal, Protocol, Sequence

from config import Settings
from retrieval.context import ContextRepository, expand_context
from retrieval.common import RetrievalHit, normalize_query


class RerankerProvider(Protocol):
	"""Small cross-encoder-like interface for scoring query/candidate pairs."""

	provider_name: str
	model_name: str
	model_version: str

	def score(self, query: str, candidate_texts: Sequence[str]) -> Sequence[float]:
		"""Return one relevance score for each candidate text, in input order."""


@dataclass(frozen=True)
class RerankResult:
	"""Reranked candidates or an explicit safe report that reranking was skipped."""

	candidates: tuple[RetrievalHit, ...]
	status: Literal["applied", "skipped", "failed"]
	provider_name: str | None = None
	model_name: str | None = None
	model_version: str | None = None
	reason: str | None = None


class SentenceTransformerCrossEncoder:
	"""Optional local cross-encoder backed by ``sentence-transformers``."""

	provider_name = "sentence-transformers-cross-encoder"

	def __init__(
		self,
		model_name: str,
		*,
		revision: str | None = None,
		model: object | None = None,
	) -> None:
		if not model_name.strip():
			raise ValueError("Cross-Encoder model name must not be empty")
		self.model_name = model_name.strip()
		self._model = (
			model
			if model is not None
			else self._load_model(self.model_name, revision)
		)
		self.model_version = self._revision(self._model) or revision or "unversioned"

	def score(self, query: str, candidate_texts: Sequence[str]) -> tuple[float, ...]:
		if not candidate_texts:
			return ()
		try:
			values = self._model.predict(  # type: ignore[attr-defined]
				[(query, text) for text in candidate_texts],
				show_progress_bar=False,
			)
		except Exception as error:
			raise RuntimeError("Sentence Transformers reranking failed.") from error
		try:
			rows = list(values)
			scores = tuple(
				float(row[-1] if not isinstance(row, Real) else row)
				for row in rows
			)
		except (TypeError, ValueError, IndexError) as error:
			raise RuntimeError("Sentence Transformers returned invalid reranker scores.") from error
		if len(scores) != len(candidate_texts):
			raise RuntimeError("Sentence Transformers returned an unexpected score count.")
		return scores

	@staticmethod
	def _load_model(model_name: str, revision: str | None) -> object:
		try:
			from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
		except ImportError as error:
			raise RuntimeError(
				"Sentence Transformers is not installed. Install project requirements first."
			) from error
		try:
			return CrossEncoder(model_name, revision=revision)
		except Exception as error:
			raise RuntimeError("The configured Cross-Encoder model could not be loaded.") from error

	@staticmethod
	def _revision(model: object) -> str | None:
		configuration = getattr(getattr(model, "model", None), "config", None)
		commit_hash = getattr(configuration, "_commit_hash", None)
		return commit_hash.strip() if isinstance(commit_hash, str) and commit_hash.strip() else None


def create_reranker(settings: Settings) -> RerankerProvider | None:
	"""Create the optional configured reranker; return None when disabled."""
	if settings.reranker_provider is None and settings.reranker_model is None:
		return None
	if settings.reranker_provider is None or settings.reranker_model is None:
		raise ValueError("Reranker provider and model must be configured together.")
	if settings.reranker_provider != SentenceTransformerCrossEncoder.provider_name:
		raise ValueError(f"Unsupported reranker provider: {settings.reranker_provider}")
	return SentenceTransformerCrossEncoder(
		settings.reranker_model,
		revision=settings.reranker_model_version,
	)


def rerank_candidates(
	query: str,
	candidates: Sequence[RetrievalHit],
	reranker: RerankerProvider | None = None,
	*,
	top_k: int | None = None,
	settings: Settings | None = None,
) -> RerankResult:
	"""Reorder fused candidates while retaining their retrieval scores and metadata.

	No configured reranker leaves the fused order unchanged and reports ``skipped``.
	Provider errors or malformed scores return the original order with a safe failure
	reason rather than dropping valid retrieval evidence.
	"""
	if reranker is None:
		return RerankResult(
			candidates=tuple(candidates),
			status="skipped",
			reason="No reranker configured; fused candidate order was retained.",
		)
	if not candidates:
		return RerankResult(
			candidates=(), status="skipped", reason="No candidates were available to rerank."
		)
	limit = top_k if top_k is not None else (settings.reranker_top_k if settings else len(candidates))
	if limit <= 0:
		raise ValueError("top_k must be greater than zero")
	query_text = normalize_query(query)
	if not query_text:
		return _failed_result(candidates, reranker, "Query is empty; original fused order was retained.")
	identity = _reranker_identity(reranker)
	try:
		scores = tuple(
			float(score)
			for score in reranker.score(
				query_text, tuple(_candidate_text(hit) for hit in candidates)
			)
		)
		if len(scores) != len(candidates):
			raise ValueError("Reranker returned a different number of scores than candidates.")
		if any(not math.isfinite(score) for score in scores):
			raise ValueError("Reranker scores must be finite.")
	except Exception as error:
		return _failed_result(
			candidates,
			reranker,
			f"Reranking failed ({type(error).__name__}); original fused order was retained.",
		)

	ranked = sorted(
		zip(candidates, scores, strict=True),
		key=lambda item: (
			-item[1],
			item[0].rank,
			item[0].source_path.casefold(),
			item[0].document_id,
			item[0].chunk.sequence,
			item[0].chunk_id,
		),
	)
	return RerankResult(
		candidates=tuple(
			replace(hit, rank=rank, reranker_score=score)
			for rank, (hit, score) in enumerate(ranked[:limit], start=1)
		),
		status="applied",
		provider_name=identity[0],
		model_name=identity[1],
		model_version=identity[2],
	)


def rerank_and_expand(
	query: str,
	candidates: Sequence[RetrievalHit],
	repository: ContextRepository,
	reranker: RerankerProvider | None = None,
	*,
	top_k: int | None = None,
	max_context_chunks: int | None = None,
	max_context_characters: int | None = None,
	settings: Settings | None = None,
) -> RerankResult:
	"""Rerank fused candidates if enabled, then append bounded hierarchy context."""
	result = rerank_candidates(
		query, candidates, reranker, top_k=top_k, settings=settings
	)
	return replace(
		result,
		candidates=expand_context(
			result.candidates,
			repository,
			max_context_chunks=max_context_chunks,
			max_context_characters=max_context_characters,
			settings=settings,
		),
	)


def _candidate_text(hit: RetrievalHit) -> str:
	section_path = " > ".join(hit.section_path)
	return f"{section_path}\n\n{hit.text}" if section_path else hit.text


def _reranker_identity(reranker: RerankerProvider) -> tuple[str, str, str]:
	values: list[str] = []
	for field in ("provider_name", "model_name", "model_version"):
		value = getattr(reranker, field, None)
		if not isinstance(value, str) or not value.strip():
			raise ValueError(f"Reranker {field} must be configured.")
		values.append(value.strip())
	if not callable(getattr(reranker, "score", None)):
		raise TypeError("Reranker must implement score(query, candidate_texts).")
	return values[0], values[1], values[2]


def _failed_result(
	candidates: Sequence[RetrievalHit], reranker: RerankerProvider, reason: str
) -> RerankResult:
	return RerankResult(
		candidates=tuple(candidates),
		status="failed",
		provider_name=getattr(reranker, "provider_name", None),
		model_name=getattr(reranker, "model_name", None),
		model_version=getattr(reranker, "model_version", None),
		reason=reason,
	)
