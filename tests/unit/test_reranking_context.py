"""Focused TASK-08 tests for reranking and hierarchy context expansion."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from config import Settings
from ingestion.chunker import RetrievalChunk
from ingestion.hierarchy import HierarchyNode
from retrieval.common import RetrievalHit
from retrieval.context import expand_context
from retrieval.reranker import (
    SentenceTransformerCrossEncoder,
    create_reranker,
	rerank_and_expand,
    rerank_candidates,
)


def _chunk(
    chunk_id: str,
    sequence: int,
    parent_id: str,
    text: str,
    *,
    previous: str | None = None,
    following: str | None = None,
    section_path: tuple[str, ...] = ("Manual", "Section"),
    document_id: str = "doc-1",
) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        parent_id=parent_id,
        level="content",
        section_title="Section",
        section_path=section_path,
        previous_chunk_id=previous,
        next_chunk_id=following,
        page_start=1 + sequence,
        page_end=1 + sequence,
        sequence=sequence,
        text=text,
        chunk_size=len(text),
        size_unit="characters",
        token_count=len(text.split()),
        content_hash=f"hash-{chunk_id}",
        configuration_fingerprint="config",
    )


def _hit(chunk: RetrievalChunk, rank: int, score: float = 0.5) -> RetrievalHit:
    return RetrievalHit(
        chunk=chunk,
        source_filename=f"{chunk.document_id}.pdf",
        source_path=f"manual/{chunk.document_id}.pdf",
        rank=rank,
        score=score,
        hybrid_score=score,
    )


class FakeReranker:

	provider_name = "fake-reranker"
	model_name = "fake-cross-encoder"
	model_version = "v1"

	def __init__(self, scores: tuple[float, ...] | None = None, error: Exception | None = None) -> None:
		self.scores = scores
		self.error = error
		self.calls: list[tuple[str, tuple[str, ...]]] = []

	def score(self, query: str, candidate_texts: tuple[str, ...] | list[str]) -> tuple[float, ...]:
		texts = tuple(candidate_texts)
		self.calls.append((query, texts))
		if self.error:
			raise self.error
		return self.scores or tuple(float(len(text)) for text in texts)


class FakeContextRepository:
	def __init__(self, chunks: tuple[RetrievalChunk, ...], nodes: tuple[HierarchyNode, ...]) -> None:
		self.chunks = chunks
		self.nodes = nodes
		self.chunk_requests: list[str] = []
		self.hierarchy_requests: list[str] = []

	def get_chunks(self, document_id: str) -> tuple[RetrievalChunk, ...]:
		self.chunk_requests.append(document_id)
		return tuple(item for item in self.chunks if item.document_id == document_id)

	def get_hierarchy(self, document_id: str) -> tuple[HierarchyNode, ...]:
		self.hierarchy_requests.append(document_id)
		return tuple(item for item in self.nodes if item.document_id == document_id)


def _node(
	node_id: str,
	parent_id: str | None,
	level: str,
	sequence: int,
) -> HierarchyNode:
	return HierarchyNode(
		node_id=node_id,
		document_id="doc-1",
		parent_id=parent_id,
		level=level,
		section_title=node_id,
		section_path=("Manual", node_id),
		page_start=1,
		page_end=10,
		sequence=sequence,
		detection_method="test",
	)


def test_reranker_reorders_candidates_and_keeps_source_and_retrieval_scores() -> None:
	first = _hit(_chunk("chunk-1", 0, "section", "short"), 1, 0.9)
	second = _hit(_chunk("chunk-2", 1, "section", "more relevant passage"), 2, 0.8)
	reranker = FakeReranker((0.1, 0.95))

	result = rerank_candidates("  MVCC  query ", (first, second), reranker, top_k=1)

	assert result.status == "applied"
	assert result.provider_name == "fake-reranker"
	assert result.model_name == "fake-cross-encoder"
	assert result.model_version == "v1"
	assert [hit.chunk_id for hit in result.candidates] == ["chunk-2"]
	assert result.candidates[0].rank == 1
	assert result.candidates[0].score == 0.8
	assert result.candidates[0].hybrid_score == 0.8
	assert result.candidates[0].reranker_score == 0.95
	assert result.candidates[0].page_start == 2
	assert result.candidates[0].section_path == ("Manual", "Section")
	assert reranker.calls[0][0] == "mvcc query"
	assert "Manual > Section" in reranker.calls[0][1][0]


def test_reranker_skipped_and_failed_paths_retain_original_order() -> None:
	candidates = (
		_hit(_chunk("chunk-1", 0, "section", "one"), 1),
		_hit(_chunk("chunk-2", 1, "section", "two"), 2),
	)

	skipped = rerank_candidates("query", candidates)
	failed = rerank_candidates(
		"query", candidates, FakeReranker(error=RuntimeError("secret provider detail"))
	)

	assert skipped.status == "skipped"
	assert "No reranker" in (skipped.reason or "")
	assert skipped.candidates == candidates
	assert failed.status == "failed"
	assert failed.candidates == candidates
	assert "secret provider detail" not in (failed.reason or "")


def test_reranker_rejects_malformed_and_non_finite_scores_without_dropping_candidates() -> None:
	candidate = _hit(_chunk("chunk-1", 0, "section", "one"), 1)

	wrong_count = rerank_candidates("query", (candidate,), FakeReranker((0.1, 0.2)))
	non_finite = rerank_candidates("query", (candidate,), FakeReranker((float("nan"),)))

	assert wrong_count.status == non_finite.status == "failed"
	assert wrong_count.candidates == non_finite.candidates == (candidate,)


def test_sentence_transformer_cross_encoder_adapter_and_factory() -> None:
	class FakeModel:
		model = SimpleNamespace(config=SimpleNamespace(_commit_hash="commit-9"))

		def __init__(self) -> None:
			self.pairs = None

		def predict(self, pairs, **kwargs):
			self.pairs = (pairs, kwargs)
			return [0.25, 0.75]

	model = FakeModel()
	adapter = SentenceTransformerCrossEncoder("cross-encoder/test", model=model)
	assert adapter.score("q", ("a", "b")) == (0.25, 0.75)
	assert adapter.model_version == "commit-9"
	assert model.pairs[0] == [("q", "a"), ("q", "b")]

	assert create_reranker(Settings.from_environment({}, project_root=Path.cwd())) is None
	configured = Settings.from_environment(
		{
			"PDF_RAG_RERANKER_PROVIDER": "unsupported",
			"PDF_RAG_RERANKER_MODEL": "test-model",
		},
		project_root=Path.cwd(),
	)
	with pytest.raises(ValueError, match="Unsupported reranker"):
		create_reranker(configured)


def test_context_expansion_adds_preferred_parent_neighbor_and_child_with_bounds() -> None:
	root = _node("root", None, "document", 0)
	section = _node("section", "root", "section", 1)
	other_section = _node("other-section", "root", "section", 2)
	selected = _chunk("selected", 1, "section", "selected passage", previous="before", following="after")
	parent_context = _chunk("parent-context", 0, "root", "document-level background")
	neighbor = _chunk("after", 2, "other-section", "continued neighboring text", previous="selected")
	child = _chunk("child-context", 3, "child-section", "child-specific details")
	cross_document = _chunk("foreign", 1, "content-node", "foreign passage", document_id="doc-2")
	foreign_hit = _hit(cross_document, 2)
	chunks = (parent_context, selected, neighbor, child, cross_document)
	nodes = (
		root,
		section,
		other_section,
		_node("child-section", "section", "subsection", 5),
		_node("foreign-root", None, "document", 0),
		_node("foreign-section", "foreign-root", "section", 1),
	)
	repository = FakeContextRepository(chunks, nodes)
	hits = (
		_hit(selected, 1),
		foreign_hit,
	)

	result = expand_context(hits, repository, max_context_chunks=3, max_context_characters=100)

	assert [evidence.chunk.chunk_id for evidence in result[0].context_chunks] == [
		"parent-context", "after", "child-context"
	]
	assert [evidence.relation for evidence in result[0].context_chunks] == [
		"parent", "neighbor", "child"
	]
	assert result[1].context_chunks == ()
	assert repository.chunk_requests == ["doc-1", "doc-2"]
	assert repository.hierarchy_requests == ["doc-1", "doc-2"]


def test_context_expansion_deduplicates_candidates_and_duplicate_text_globally() -> None:
	root = _node("root", None, "document", 0)
	section = _node("section", "root", "section", 1)
	selected_a = _chunk(
		"selected-a", 0, "content-a", "primary content", following="context-duplicate"
	)
	selected_b = _chunk("selected-b", 1, "content-b", "different primary")
	duplicate_text = _chunk(
		"context-duplicate", 2, "content-b", "PRIMARY CONTENT", previous="selected-a"
	)
	duplicate_hit = _hit(selected_a, 2)
	repository = FakeContextRepository(
		(selected_a, selected_b, duplicate_text),
		(
			root,
			section,
			_node("content-a", "section", "content", 2),
			_node("content-b", "section", "content", 3),
		),
	)

	result = expand_context(
		(_hit(selected_a, 1), duplicate_hit, _hit(selected_b, 3)),
		repository,
		max_context_chunks=4,
		max_context_characters=100,
	)

	assert len(result) == 2
	assert result[0].context_chunks == ()
	assert result[1].context_chunks == ()


def test_rerank_and_expand_retains_skipped_status_and_attaches_context() -> None:
	root = _node("root", None, "document", 0)
	section = _node("section", "root", "section", 1)
	selected = _chunk("selected", 0, "section", "selected evidence", following="neighbor")
	neighbor = _chunk(
		"neighbor", 1, "section", "supporting evidence", previous="selected"
	)
	repository = FakeContextRepository(
		(selected, neighbor),
		(root, section),
	)
	candidate = _hit(selected, 1)

	result = rerank_and_expand("query", (candidate,), repository)

	assert result.status == "skipped"
	assert [item.chunk.chunk_id for item in result.candidates[0].context_chunks] == [
		"neighbor"
	]
	assert result.candidates[0].context_chunks[0].relation == "neighbor"


@pytest.mark.parametrize(
	("kwargs", "message"),
	[
		({"max_context_chunks": -1}, "max_context_chunks"),
		({"max_context_characters": -1}, "max_context_characters"),
	],
)
def test_context_expansion_rejects_negative_bounds(
	kwargs: dict[str, int], message: str
) -> None:
	with pytest.raises(ValueError, match=message):
		expand_context((), FakeContextRepository((), ()), **kwargs)