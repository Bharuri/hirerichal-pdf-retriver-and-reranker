"""TASK-07 retrieval integration tests using temporary DuckDB files and fake vectors."""

from pathlib import Path
from dataclasses import replace

import pytest

from db.duckdb import DuckDBStore, EmbeddingRecord
from ingestion.chunker import RetrievalChunk
from ingestion.hierarchy import HierarchyNode, HierarchyResult
from ingestion.pdf_loader import PageText, PdfReadResult, PdfStatus
from retrieval.common import RetrievalError
from retrieval.hybrid import hybrid_search
from retrieval.keyword import keyword_search
from retrieval.semantic import semantic_search


class QueryEmbedder:
    provider_name = "fake"
    model_name = "test-model"
    model_version = "v1"

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.queries: list[tuple[str, ...]] = []

    def embed(self, texts: tuple[str, ...] | list[str]) -> list[list[float]]:
        self.queries.append(tuple(texts))
        return [self.vector]


def _add_document(
    store: DuckDBStore,
    texts: tuple[tuple[str, str], ...],
    *,
    document_id: str = "manual-1",
    source_path: str = "postgres/manual.pdf",
) -> tuple[RetrievalChunk, ...]:
    document = PdfReadResult(
        document_id=document_id,
        source_filename=Path(source_path).name,
        source_path=source_path,
        content_sha256=f"sha-{document_id}",
        page_count=1,
        metadata={"title": "Technical Manual"},
        pages=(PageText(1, "\n".join(text for text, _ in texts)),),
        status=PdfStatus.EXTRACTED,
    )
    root = HierarchyNode(
        node_id=f"{document_id}-root",
        document_id=document_id,
        parent_id=None,
        level="document",
        section_title="Technical Manual",
        section_path=("Technical Manual",),
        page_start=1,
        page_end=1,
        sequence=0,
        detection_method="document_root",
    )
    nodes = [root]
    chunks: list[RetrievalChunk] = []
    for sequence, (text, title) in enumerate(texts):
        section_id = f"{document_id}-section-{sequence}"
        section_path = ("Technical Manual", title)
        nodes.append(
            HierarchyNode(
                node_id=section_id,
                document_id=document_id,
                parent_id=root.node_id,
                level="section",
                section_title=title,
                section_path=section_path,
                page_start=1,
                page_end=1,
                sequence=sequence * 2 + 1,
                detection_method="test_outline",
            )
        )
        content_id = f"{document_id}-content-{sequence}"
        nodes.append(
            HierarchyNode(
                node_id=content_id,
                document_id=document_id,
                parent_id=section_id,
                level="content",
                section_title=None,
                section_path=section_path,
                page_start=1,
                page_end=1,
                sequence=sequence * 2 + 2,
                detection_method="paragraph_content",
                text=text,
            )
        )
        chunks.append(
            RetrievalChunk(
                chunk_id=f"{document_id}-chunk-{sequence}",
                document_id=document_id,
                parent_id=section_id,
                level="content",
                section_title=title,
                section_path=section_path,
                previous_chunk_id=chunks[-1].chunk_id if chunks else None,
                next_chunk_id=None,
                page_start=1,
                page_end=1,
                sequence=sequence,
                text=text,
                chunk_size=max(1, len(text)),
                size_unit="characters",
                token_count=len(text.split()),
                content_hash=f"{document_id}-hash-{sequence}",
                configuration_fingerprint="test-config",
            )
        )
    linked = tuple(
        replace(
            chunk,
            next_chunk_id=chunks[index + 1].chunk_id
            if index + 1 < len(chunks)
            else None,
        )
        for index, chunk in enumerate(chunks)
    )
    store.replace_document_index(
        document, HierarchyResult(document_id, tuple(nodes)), linked
    )
    return linked


def test_keyword_search_matches_exact_technical_terms_and_hierarchy_fields(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "keyword.duckdb")
    store.initialize()
    chunks = _add_document(
        store,
        (
            ("PostgreSQL uses MVCC for concurrent transactions.", "Storage"),
            ("Snapshots control visibility.", "MVCC Snapshots"),
            ("An XMVCCextra identifier is unrelated.", "Unrelated"),
        ),
    )

    hits = keyword_search(store, "  MVCC\t", top_k=10)

    assert [hit.chunk_id for hit in hits] == [
        "manual-1-chunk-1", "manual-1-chunk-0"
    ]
    assert [hit.score for hit in hits] == [4.0, 1.0]
    assert hits[0].keyword_rank == 1
    assert hits[0].source_filename == "manual.pdf"
    assert hits[0].source_path == "postgres/manual.pdf"
    assert hits[0].document_id == "manual-1"
    assert hits[0].page_start == hits[0].page_end == 1
    assert hits[0].section_path == ("Technical Manual", "MVCC Snapshots")
    assert hits[0].chunk == chunks[1]
    assert keyword_search(store, " \n ") == ()
    store.close()


def test_keyword_search_handles_empty_index_and_is_available_without_vectors(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "keyword-empty.duckdb")
    store.initialize()

    assert keyword_search(store, "PostgreSQL") == ()
    _add_document(store, (("PostgreSQL supports MVCC.", "Concurrency"),))
    hits = keyword_search(store, "postgresql")

    assert len(hits) == 1
    assert hits[0].text == "PostgreSQL supports MVCC."
    store.close()


def test_semantic_cosine_search_filters_model_version_dimension_and_stale_vectors(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "semantic.duckdb")
    store.initialize()
    chunks = _add_document(
        store,
        (
            ("Best semantic match.", "One"),
            ("Orthogonal semantic match.", "Two"),
            ("Opposite semantic match.", "Three"),
            ("Wrong vector dimension.", "Four"),
            ("Different provider.", "Five"),
        ),
    )
    store.save_embeddings(
        (
            EmbeddingRecord(chunks[0].chunk_id, "fake", "test-model", "v1", chunks[0].content_hash, (1.0, 0.0)),
            EmbeddingRecord(chunks[1].chunk_id, "fake", "test-model", "v1", chunks[1].content_hash, (0.0, 1.0)),
            EmbeddingRecord(chunks[2].chunk_id, "fake", "test-model", "v1", chunks[2].content_hash, (-1.0, 0.0)),
            EmbeddingRecord(chunks[3].chunk_id, "fake", "test-model", "v1", chunks[3].content_hash, (1.0, 0.0, 0.0)),
            EmbeddingRecord(chunks[4].chunk_id, "other", "test-model", "v1", chunks[4].content_hash, (1.0, 0.0)),
            EmbeddingRecord(chunks[4].chunk_id, "fake", "test-model", "v2", chunks[4].content_hash, (1.0, 0.0)),
        )
    )
    provider = QueryEmbedder([1.0, 0.0])

    hits = semantic_search(store, "  MVCC\tPostgreSQL ", provider, top_k=2)

    assert provider.queries == [("mvcc postgresql",)]
    assert [hit.chunk_id for hit in hits] == [
        "manual-1-chunk-0", "manual-1-chunk-1"
    ]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 0.0])
    assert [hit.semantic_rank for hit in hits] == [1, 2]
    assert hits[0].source_filename == "manual.pdf"
    assert hits[0].section_path == ("Technical Manual", "One")

    wrong_dimension = semantic_search(
        store, "mvcc", QueryEmbedder([1.0, 0.0, 0.0]), top_k=10
    )
    assert [hit.chunk_id for hit in wrong_dimension] == ["manual-1-chunk-3"]
    no_matching_dimension = semantic_search(
        store, "mvcc", QueryEmbedder([1.0, 0.0, 0.0, 0.0]), top_k=10
    )
    assert no_matching_dimension == ()
    store.close()


def test_semantic_search_on_empty_index_does_not_call_provider(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "semantic-empty.duckdb")
    store.initialize()
    provider = QueryEmbedder([1.0, 0.0])

    assert semantic_search(store, "PostgreSQL", provider) == ()
    assert provider.queries == []
    store.close()


def test_semantic_search_rejects_invalid_query_vectors_without_leaking_provider_error(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "semantic-invalid-query.duckdb")
    store.initialize()
    chunks = _add_document(store, (("Existing vector.", "Vectors"),))
    store.save_embeddings(
        (
            EmbeddingRecord(chunks[0].chunk_id, "fake", "test-model", "v1", chunks[0].content_hash, (1.0, 0.0)),
        )
    )
    provider = QueryEmbedder([float("nan"), 0.0])

    with pytest.raises(RetrievalError, match="finite numeric"):
        semantic_search(store, "vectors", provider)
    store.close()


def test_hybrid_search_works_without_embedding_provider_and_respects_top_k(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "hybrid-keyword-only.duckdb")
    store.initialize()
    _add_document(
        store,
        (
            ("PostgreSQL MVCC visibility rules.", "Concurrency"),
            ("PostgreSQL vacuum removes old tuples.", "Maintenance"),
        ),
    )

    hits = hybrid_search(store, "postgresql", top_k=1)

    assert len(hits) == 1
    assert hits[0].rank == 1
    assert hits[0].keyword_rank == 1
    assert hits[0].semantic_rank is None
    assert hits[0].hybrid_score == hits[0].score
    store.close()


def test_keyword_search_preserves_punctuation_in_technical_terms(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "technical-identifiers.duckdb")
    store.initialize()
    _add_document(
        store,
        (
            ("C++ interoperates with pg_catalog.", "Languages"),
            ("The C++17 standard adds language features.", "Standards"),
        ),
    )

    cpp_hits = keyword_search(store, "C++")
    catalog_hits = keyword_search(store, "PG_CATALOG")

    assert [hit.chunk_id for hit in cpp_hits] == ["manual-1-chunk-0"]
    assert [hit.chunk_id for hit in catalog_hits] == ["manual-1-chunk-0"]
    store.close()