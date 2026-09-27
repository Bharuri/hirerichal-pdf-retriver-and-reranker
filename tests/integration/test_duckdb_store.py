"""Focused TASK-05 tests for local DuckDB persistence and re-indexing."""

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from config import Settings
from db.duckdb import DuckDBError, DuckDBStore, EmbeddingRecord, IndexRunRecord
from ingestion.chunker import RetrievalChunk
from ingestion.hierarchy import HierarchyNode, HierarchyResult
from ingestion.indexer import embed_chunks
from ingestion.pdf_loader import PageText, PdfReadResult, PdfStatus


NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


def sample_index(
    *,
    text: str = "PostgreSQL stores row versions.",
    content_hash: str = "source-v1",
    page_two_text: str = "A second source page.",
) -> tuple[PdfReadResult, HierarchyResult, tuple[RetrievalChunk, ...]]:
    document = PdfReadResult(
        document_id="document-1",
        source_filename="postgres.pdf",
        source_path="postgres.pdf",
        content_sha256=content_hash,
        page_count=2,
        metadata={"title": "PostgreSQL Manual", "author": "Documentation"},
        pages=(
            PageText(1, text, ()),
            PageText(2, page_two_text, ("little_extractable_text",)),
        ),
        status=PdfStatus.EXTRACTED,
        warnings=(),
    )
    root = HierarchyNode(
        node_id="root-1",
        document_id="document-1",
        parent_id=None,
        level="document",
        section_title="PostgreSQL Manual",
        section_path=("PostgreSQL Manual",),
        page_start=1,
        page_end=2,
        sequence=0,
        detection_method="document_root",
    )
    section = HierarchyNode(
        node_id="section-1",
        document_id="document-1",
        parent_id="root-1",
        level="section",
        section_title="Concurrency Control",
        section_path=("PostgreSQL Manual", "Concurrency Control"),
        page_start=1,
        page_end=2,
        sequence=1,
        detection_method="fixture_outline",
        confidence=1.0,
    )
    content = HierarchyNode(
        node_id="content-1",
        document_id="document-1",
        parent_id="section-1",
        level="content",
        section_title=None,
        section_path=section.section_path,
        page_start=1,
        page_end=1,
        sequence=2,
        detection_method="paragraph_content",
        text=text,
    )
    hierarchy = HierarchyResult("document-1", (root, section, content))
    chunk = RetrievalChunk(
        chunk_id="chunk-1",
        document_id="document-1",
        parent_id="section-1",
        level="content",
        section_title="Concurrency Control",
        section_path=section.section_path,
        previous_chunk_id=None,
        next_chunk_id=None,
        page_start=1,
        page_end=1,
        sequence=0,
        text=text,
        chunk_size=len(text),
        size_unit="characters",
        token_count=len(text.split()),
        content_hash=f"hash-{content_hash}",
        configuration_fingerprint="chunk-config-v1",
    )
    return document, hierarchy, (chunk,)


def test_initial_schema_creates_expected_tables_and_reopens(tmp_path: Path) -> None:
    database_path = tmp_path / "data" / "retrieval.duckdb"
    store = DuckDBStore(database_path)
    with pytest.raises(DuckDBError, match="not initialized"):
        store.counts()

    store.initialize()
    assert store.is_open
    assert store.counts() == {"documents": 0, "chunks": 0, "failures": 0}
    with store._require_connection() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
    assert {
        "documents",
        "pages",
        "hierarchy_nodes",
        "chunks",
        "embeddings",
        "index_runs",
    } <= tables
    store.close()

    reopened = DuckDBStore(database_path)
    reopened.initialize()
    assert reopened.counts()["documents"] == 0
    reopened.close()


def test_document_pages_hierarchy_chunks_and_embedding_round_trip(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "round-trip.duckdb")
    store.initialize()
    document, hierarchy, chunks = sample_index()
    embedding = EmbeddingRecord(
        "chunk-1", "fake-provider", "local-model", "v1", chunks[0].content_hash,
        (0.25, 0.75), NOW,
    )

    store.replace_document_index(document, hierarchy, chunks, (embedding,))

    stored_document = store.get_document("document-1")
    assert stored_document is not None
    assert stored_document["source_filename"] == "postgres.pdf"
    assert stored_document["source_path"] == "postgres.pdf"
    assert stored_document["content_sha256"] == "source-v1"
    assert stored_document["metadata"]["title"] == "PostgreSQL Manual"
    assert stored_document["status"] == "extracted"
    assert store.get_pages("document-1") == (
        {"page_number": 1, "text": "PostgreSQL stores row versions.", "warnings": ()},
        {
            "page_number": 2,
            "text": "A second source page.",
            "warnings": ("little_extractable_text",),
        },
    )
    assert store.get_hierarchy("document-1") == hierarchy.nodes
    assert store.get_chunks("document-1") == chunks
    assert store.get_embedding(
        "chunk-1", "fake-provider", "local-model", "v1"
    ) == embedding
    assert store.counts() == {"documents": 1, "chunks": 1, "failures": 0}
    store.close()


def test_reindex_replaces_derived_rows_without_duplicate_records(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "replace.duckdb")
    store.initialize()
    first = sample_index()
    second = sample_index(
        text="PostgreSQL uses snapshots for MVCC.", content_hash="source-v2"
    )

    store.replace_document_index(*first)
    store.replace_document_index(*first)
    assert store.counts() == {"documents": 1, "chunks": 1, "failures": 0}

    store.replace_document_index(*second)
    assert store.counts() == {"documents": 1, "chunks": 1, "failures": 0}
    assert store.get_document("document-1")["content_sha256"] == "source-v2"
    assert (
        store.get_pages("document-1")[0]["text"]
        == "PostgreSQL uses snapshots for MVCC."
    )
    assert (
        store.get_chunks("document-1")[0].text == "PostgreSQL uses snapshots for MVCC."
    )
    assert store.get_embedding("chunk-1", "fake-provider", "local-model", "v1") is None
    store.close()


def test_failed_document_replacement_rolls_back_and_preserves_old_index(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "rollback.duckdb")
    store.initialize()
    old_document, old_hierarchy, old_chunks = sample_index()
    store.replace_document_index(old_document, old_hierarchy, old_chunks)

    other = PdfReadResult(
        document_id="another-document",
        source_filename="collision.pdf",
        source_path="postgres.pdf",  # violates unique source path after the old rows are deleted
        content_sha256="other-hash",
        page_count=1,
        metadata={},
        pages=(PageText(1, "Other content."),),
        status=PdfStatus.EXTRACTED,
    )
    other_root = HierarchyNode(
        "other-root",
        "another-document",
        None,
        "document",
        "Other",
        ("Other",),
        1,
        1,
        0,
        "document_root",
    )
    with pytest.raises(DuckDBError, match="rolled back"):
        store.replace_document_index(
            other, HierarchyResult("another-document", (other_root,)), ()
        )

    assert store.get_document("document-1") is not None
    assert store.get_document("another-document") is None
    assert store.get_chunks("document-1") == old_chunks
    assert store.counts() == {"documents": 1, "chunks": 1, "failures": 0}
    store.close()


def test_invalid_references_and_duplicate_ids_are_rejected_before_writes(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "validation.duckdb")
    store.initialize()
    document, hierarchy, chunks = sample_index()

    orphan = EmbeddingRecord(
        "absent-chunk", "fake-provider", "model", "v1", "hash", (0.1, 0.2), NOW
    )
    with pytest.raises(DuckDBError, match="missing chunk"):
        store.replace_document_index(document, hierarchy, chunks, (orphan,))

    with pytest.raises(DuckDBError, match="unique within a document"):
        store.replace_document_index(document, hierarchy, chunks + chunks)

    assert store.counts() == {"documents": 0, "chunks": 0, "failures": 0}
    store.close()


def test_index_run_summary_round_trips_and_validates_counts(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "runs.duckdb")
    store.initialize()
    run = IndexRunRecord("run-1", "partial", 2, 5, 1, NOW, NOW, "one malformed PDF")

    store.save_index_run(run)

    assert store.get_index_run("run-1") == run
    with pytest.raises(ValueError, match="non-negative"):
        store.save_index_run(IndexRunRecord("bad", "failed", -1, 0, 0, NOW))
    store.close()


def test_remove_document_deletes_all_derived_data_and_is_idempotent(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "remove.duckdb")
    store.initialize()
    document, hierarchy, chunks = sample_index()
    vector = EmbeddingRecord(
        "chunk-1", "fake-provider", "model", "v1", chunks[0].content_hash,
        (1.0, 0.0), NOW,
    )
    store.replace_document_index(document, hierarchy, chunks, (vector,))

    assert store.remove_document("document-1") is True
    assert store.remove_document("document-1") is False
    assert store.get_document("document-1") is None
    assert store.get_pages("document-1") == ()
    assert store.get_hierarchy("document-1") == ()
    assert store.get_chunks("document-1") == ()
    assert store.get_embedding("chunk-1", "fake-provider", "model", "v1") is None
    assert store.counts() == {"documents": 0, "chunks": 0, "failures": 0}
    store.close()


def test_prepare_backup_checkpoints_and_closes_only_file_backed_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "backup.duckdb"
    store = DuckDBStore(database_path)
    with pytest.raises(DuckDBError, match="not initialized"):
        store.prepare_backup()

    store.initialize()
    document, hierarchy, chunks = sample_index()
    store.replace_document_index(document, hierarchy, chunks)
    assert store.prepare_backup() == database_path.resolve()
    assert not store.is_open

    reopened = DuckDBStore(database_path)
    reopened.initialize()
    assert reopened.get_chunks("document-1") == chunks
    reopened.close()

    memory_store = DuckDBStore(":memory:")
    memory_store.initialize()
    with pytest.raises(DuckDBError, match="in-memory"):
        memory_store.prepare_backup()
    memory_store.close()


class FakeEmbeddingProvider:
    provider_name = "fake"
    model_name = "test-model"
    model_version = "v1"

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail_on = fail_on

    def embed(self, texts: tuple[str, ...] | list[str]) -> list[list[float]]:
        batch = tuple(texts)
        self.calls.append(batch)
        if self.fail_on and any(self.fail_on in text for text in batch):
            raise RuntimeError("provider detail must not escape")
        return [[float(index + 1), float(len(text))] for index, text in enumerate(batch)]


def _three_chunks() -> tuple[RetrievalChunk, ...]:
    _, _, (first,) = sample_index()
    return (
        first,
        replace(
            first,
            chunk_id="chunk-2",
            sequence=1,
            text="A second technical passage.",
            content_hash="hash-second",
            previous_chunk_id="chunk-1",
        ),
        replace(
            first,
            chunk_id="chunk-3",
            sequence=2,
            text="A third technical passage.",
            content_hash="hash-third",
            previous_chunk_id="chunk-2",
        ),
    )


def test_embedding_batches_preserve_order_context_metadata_and_reuse(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "embedding-batches.duckdb")
    store.initialize()
    document, hierarchy, _ = sample_index()
    chunks = _three_chunks()
    store.replace_document_index(document, hierarchy, chunks)
    provider = FakeEmbeddingProvider()
    progress: list[tuple[str, int, int]] = []
    settings = Settings.from_environment(
        {
            "PDF_RAG_EMBEDDING_PROVIDER": "fake",
            "PDF_RAG_EMBEDDING_MODEL": "test-model",
        },
        project_root=tmp_path,
    )

    first_result = embed_chunks(
        chunks,
        store,
        provider,
        batch_size=2,
        settings=settings,
        on_embedding_start=lambda chunk, position, total: progress.append(
            (chunk.chunk_id, position, total)
        ),
    )

    assert [len(batch) for batch in provider.calls] == [2, 1]
    assert "PostgreSQL Manual > Concurrency Control" in provider.calls[0][0]
    assert [item.chunk_id for item in first_result.embeddings] == [
        "chunk-1", "chunk-2", "chunk-3"
    ]
    assert first_result.reused_count == 0
    assert first_result.failures == ()
    assert progress == [
        ("chunk-1", 1, 3),
        ("chunk-2", 2, 3),
        ("chunk-3", 3, 3),
    ]
    assert all(item.provider_name == "fake" for item in first_result.embeddings)
    assert all(item.model_version == "v1" and len(item.vector) == 2 for item in first_result.embeddings)

    calls_before_reuse = len(provider.calls)
    second_result = embed_chunks(chunks, store, provider, batch_size=2)

    assert len(provider.calls) == calls_before_reuse
    assert second_result.reused_count == 3
    assert [item.chunk_id for item in second_result.embeddings] == [
        item.chunk_id for item in first_result.embeddings
    ]
    assert [item.vector for item in second_result.embeddings] == [
        item.vector for item in first_result.embeddings
    ]
    stored = store.get_embedding(
        "chunk-1", "fake", "test-model", "v1", chunks[0].content_hash
    )
    assert stored is not None
    assert stored.vector == first_result.embeddings[0].vector
    assert stored.provider_name == "fake"
    assert stored.content_hash == chunks[0].content_hash
    assert store.get_embedding(
        "chunk-1", "fake", "test-model", "v1", "stale-content"
    ) is None
    store.close()


def test_failed_chunks_are_reported_and_not_marked_embedded(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "embedding-failures.duckdb")
    store.initialize()
    document, hierarchy, _ = sample_index()
    chunks = _three_chunks()
    bad_chunk = replace(
        chunks[1], text="bad technical passage", content_hash="hash-bad-text"
    )
    chunks = (chunks[0], bad_chunk, chunks[2])
    store.replace_document_index(document, hierarchy, chunks)
    provider = FakeEmbeddingProvider(fail_on="bad technical passage")

    result = embed_chunks(chunks, store, provider, batch_size=3, max_retries=1)

    assert [item.chunk_id for item in result.embeddings] == ["chunk-1", "chunk-3"]
    assert len(result.failures) == 1
    assert result.failures[0].chunk_id == "chunk-2"
    assert "provider detail" not in result.failures[0].reason
    assert store.get_embedding("chunk-2", "fake", "test-model", "v1") is None
    assert store.get_embedding("chunk-1", "fake", "test-model", "v1") is not None

    provider.fail_on = None
    retry = embed_chunks(chunks, store, provider)
    assert retry.reused_count == 2
    assert not retry.failures
    assert len(retry.embeddings) == 3
    store.close()


@pytest.mark.parametrize("invalid_vector", [[], [float("nan")], [float("inf")], [True]])
def test_invalid_vectors_are_reported_and_never_persisted(
    tmp_path: Path, invalid_vector: list[float]
) -> None:
    class InvalidProvider(FakeEmbeddingProvider):
        def embed(self, texts: tuple[str, ...] | list[str]) -> list[list[float]]:
            self.calls.append(tuple(texts))
            return [invalid_vector for _ in texts]

    store = DuckDBStore(tmp_path / "invalid-vectors.duckdb")
    store.initialize()
    document, hierarchy, chunks = sample_index()
    store.replace_document_index(document, hierarchy, chunks)

    result = embed_chunks(chunks, store, InvalidProvider(), max_retries=0)

    assert result.embeddings == ()
    assert result.failures[0].chunk_id == "chunk-1"
    assert store.get_embedding("chunk-1", "fake", "test-model", "v1") is None
    store.close()


def test_embedding_provider_must_match_configured_provider_and_model(
    tmp_path: Path,
) -> None:
    store = DuckDBStore(tmp_path / "embedding-config.duckdb")
    store.initialize()
    _, _, chunks = sample_index()
    settings = Settings.from_environment(
        {
            "PDF_RAG_EMBEDDING_PROVIDER": "another-provider",
            "PDF_RAG_EMBEDDING_MODEL": "test-model",
        },
        project_root=tmp_path,
    )

    with pytest.raises(ValueError, match="does not match"):
        embed_chunks(chunks, store, FakeEmbeddingProvider(), settings=settings)
    store.close()


def test_embed_chunks_uses_configured_sentence_transformer_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ingestion.indexer as indexer

    class ConfiguredProvider:
        provider_name = "sentence-transformers"
        model_name = "sentence-transformers/test-model"
        model_version = "commit-1"

        def __init__(self, model_name: str, *, revision: str | None = None) -> None:
            assert model_name == self.model_name
            assert revision is None

        def embed(self, texts: tuple[str, ...] | list[str]) -> list[list[float]]:
            return [[1.0, float(len(text))] for text in texts]

    monkeypatch.setattr(indexer, "SentenceTransformerProvider", ConfiguredProvider)
    store = DuckDBStore(tmp_path / "configured-provider.duckdb")
    store.initialize()
    document, hierarchy, chunks = sample_index()
    store.replace_document_index(document, hierarchy, chunks)
    settings = Settings.from_environment(
        {
            "PDF_RAG_EMBEDDING_PROVIDER": "sentence-transformers",
            "PDF_RAG_EMBEDDING_MODEL": "sentence-transformers/test-model",
        },
        project_root=tmp_path,
    )

    result = embed_chunks(chunks, store, settings=settings)

    assert len(result.embeddings) == 1
    assert result.embeddings[0].provider_name == "sentence-transformers"
    assert result.embeddings[0].model_version == "commit-1"
    assert store.get_embedding(
        "chunk-1",
        "sentence-transformers",
        "sentence-transformers/test-model",
        "commit-1",
    ) is not None
    store.close()


def test_changed_chunk_content_is_not_reused(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "changed-content.duckdb")
    store.initialize()
    document, hierarchy, original = sample_index()
    store.replace_document_index(document, hierarchy, original)
    provider = FakeEmbeddingProvider()
    first = embed_chunks(original, store, provider)
    changed = replace(
        original[0],
        text="Updated PostgreSQL content.",
        content_hash="updated-content-hash",
    )
    store.replace_document_index(document, hierarchy, (changed,))

    updated = embed_chunks((changed,), store, provider)

    assert len(provider.calls) == 2
    assert updated.reused_count == 0
    assert updated.embeddings[0].content_hash == "updated-content-hash"
    assert updated.embeddings[0].vector != first.embeddings[0].vector
    store.close()


def test_inconsistent_vector_dimensions_are_rejected(tmp_path: Path) -> None:
    class InconsistentProvider(FakeEmbeddingProvider):
        def embed(self, texts: tuple[str, ...] | list[str]) -> list[list[float]]:
            batch = tuple(texts)
            self.calls.append(batch)
            if len(batch) > 1:
                return [[1.0], [1.0, 2.0]]
            return [[1.0]] if "PostgreSQL stores" in batch[0] else [[1.0, 2.0]]

    store = DuckDBStore(tmp_path / "inconsistent-dimensions.duckdb")
    store.initialize()
    document, hierarchy, chunks = sample_index()
    second = replace(
        chunks[0],
        chunk_id="chunk-2",
        sequence=1,
        text="A separate passage.",
        content_hash="hash-separate",
    )
    chunks = (chunks[0], second)
    store.replace_document_index(document, hierarchy, chunks)

    result = embed_chunks(chunks, store, InconsistentProvider(), batch_size=2)

    assert [item.chunk_id for item in result.embeddings] == ["chunk-1"]
    assert [item.chunk_id for item in result.failures] == ["chunk-2"]
    assert store.get_embedding("chunk-2", "fake", "test-model", "v1") is None
    store.close()


def test_old_embedding_schema_is_upgraded_on_initialize(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(database_path))
    connection.execute(
        """CREATE TABLE embeddings (
            chunk_id VARCHAR NOT NULL,
            model_name VARCHAR NOT NULL,
            model_version VARCHAR NOT NULL,
            dimension INTEGER NOT NULL,
            vector DOUBLE[] NOT NULL,
            created_at VARCHAR NOT NULL,
            PRIMARY KEY (chunk_id, model_name, model_version)
        )"""
    )
    connection.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
        ["legacy-chunk", "old-model", "v0", 2, [1.0, 2.0], NOW.isoformat()],
    )
    connection.close()

    store = DuckDBStore(database_path)
    store.initialize()
    columns = {
        row[1]
        for row in store._require_connection()
        .execute("PRAGMA table_info('embeddings')")
        .fetchall()
    }
    assert {"provider_name", "model_name", "model_version", "content_hash"} <= columns
    legacy_embedding = store.get_embedding(
        "legacy-chunk", "legacy", "old-model", "v0"
    )
    assert legacy_embedding is not None
    assert legacy_embedding.vector == (1.0, 2.0)
    assert legacy_embedding.content_hash == ""
    store.close()
