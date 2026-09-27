"""Integration coverage for the manually invoked PDF-to-DuckDB ingestion command."""

from pathlib import Path

from config import Settings
from db.duckdb import DuckDBStore
from ingestion.main import run_ingestion


class FakeEmbeddingProvider:
    provider_name = "sentence-transformers"
    model_name = "test-model"
    model_version = "fake-v1"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail = fail

    def embed(self, texts: tuple[str, ...] | list[str]) -> list[list[float]]:
        batch = tuple(texts)
        self.calls.append(batch)
        if self.fail:
            raise RuntimeError("fake provider detail must stay private")
        return [[float(len(text)), 1.0] for text in batch]


def _write_text_pdf(path: Path, page_text: str) -> None:
    """Write a small valid one-page PDF containing extractable text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Title (Test Manual) >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode())
        pdf.extend(body + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    path.write_bytes(pdf)


def _settings(project_root: Path) -> Settings:
    return Settings.from_environment(
        {
            "PDF_RAG_CORPUS_DIR": "data/pdfs",
            "PDF_RAG_DATABASE_PATH": "data/duckdb/test.duckdb",
            "PDF_RAG_EMBEDDING_PROVIDER": "sentence-transformers",
            "PDF_RAG_EMBEDDING_MODEL": "test-model",
        },
        project_root=project_root,
    )


def test_run_ingestion_extracts_chunks_embeds_and_is_idempotent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_text_pdf(
        settings.corpus_dir / "postgres.pdf",
        "CONCURRENCY CONTROL PostgreSQL uses MVCC to preserve row versions.",
    )
    provider = FakeEmbeddingProvider()

    first = run_ingestion(settings, embedding_provider=provider)

    assert first.status == "completed"
    assert first.discovered_pdf_count == 1
    assert first.indexed_document_count == 1
    assert first.chunk_count > 0
    assert first.embedding_count == first.chunk_count
    assert first.embedding_failure_count == 0
    assert first.reused_document_count == 0
    assert first.failure_count == 0
    first_provider_calls = len(provider.calls)

    second = run_ingestion(settings, embedding_provider=provider)

    assert second.status == "completed"
    assert second.indexed_document_count == 1
    assert second.reused_document_count == 1
    assert second.chunk_count == first.chunk_count
    assert second.embedding_count == first.embedding_count
    assert len(provider.calls) == first_provider_calls

    with DuckDBStore(settings.database_path) as store:
        assert store.counts()["documents"] == 1
        chunks = store.get_chunks(store.list_documents()[0]["document_id"])
        assert len(chunks) == first.chunk_count
        assert "MVCC" in chunks[0].text
        embedding = store.get_embedding(
            chunks[0].chunk_id,
            "sentence-transformers",
            "test-model",
            "fake-v1",
            chunks[0].content_hash,
        )
        assert embedding is not None
        run_record = store.get_index_run(second.run_id)
        assert run_record is not None
        assert run_record.status == "completed"


def test_unreadable_pdf_does_not_prevent_valid_document_indexing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.corpus_dir.mkdir(parents=True)
    (settings.corpus_dir / "broken.pdf").write_bytes(b"not a PDF")
    _write_text_pdf(
        settings.corpus_dir / "valid.pdf",
        "PostgreSQL transactions use MVCC for concurrent row visibility.",
    )

    summary = run_ingestion(settings, embedding_provider=FakeEmbeddingProvider())

    assert summary.status == "partial"
    assert summary.discovered_pdf_count == 2
    assert summary.indexed_document_count == 1
    assert summary.failure_count == 1
    assert summary.failures[0].source_path == "broken.pdf"
    with DuckDBStore(settings.database_path) as store:
        assert store.counts()["documents"] == 1
        assert store.counts()["chunks"] > 0


def test_embedding_failures_are_reported_without_losing_chunk_index(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_text_pdf(
        settings.corpus_dir / "postgres.pdf",
        "PostgreSQL MVCC keeps older row versions visible to transactions.",
    )

    summary = run_ingestion(settings, embedding_provider=FakeEmbeddingProvider(fail=True))

    assert summary.status == "partial"
    assert summary.indexed_document_count == 1
    assert summary.chunk_count > 0
    assert summary.embedding_count == 0
    assert summary.embedding_failure_count == summary.chunk_count
    assert summary.failure_count == summary.chunk_count
    assert all("fake provider detail" not in item.reason for item in summary.failures)
    with DuckDBStore(settings.database_path) as store:
        document_id = store.list_documents()[0]["document_id"]
        assert len(store.get_chunks(document_id)) == summary.chunk_count


def test_no_pdfs_creates_configured_corpus_directory_without_empty_database(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    summary = run_ingestion(settings)

    assert summary.status == "empty"
    assert summary.discovered_pdf_count == 0
    assert settings.corpus_dir.is_dir()
    assert not settings.database_path.exists()


def test_changed_pdf_replaces_derived_index_and_vectors(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    pdf_path = settings.corpus_dir / "postgres.pdf"
    _write_text_pdf(pdf_path, "PostgreSQL MVCC preserves older row versions for readers.")
    provider = FakeEmbeddingProvider()
    first = run_ingestion(settings, embedding_provider=provider)
    _write_text_pdf(pdf_path, "PostgreSQL vacuum removes obsolete row versions from storage.")

    updated = run_ingestion(settings, embedding_provider=provider)

    assert updated.status == "completed"
    assert updated.reused_document_count == 0
    assert updated.chunk_count == first.chunk_count
    assert len(provider.calls) == 2
    with DuckDBStore(settings.database_path) as store:
        document = store.list_documents()[0]
        chunks = store.get_chunks(document["document_id"])
        assert "vacuum" in chunks[0].text.lower()
        assert len(chunks) == updated.chunk_count
