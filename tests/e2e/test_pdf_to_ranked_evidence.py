"""TASK-11 end-to-end tests from generated PDFs through ranked Streamlit evidence."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import search_index
from config import Settings
from db.duckdb import DuckDBStore
from ingestion.main import run_ingestion
from retrieval.common import RetrievalHit


class DeterministicEmbeddingProvider:
    provider_name = "sentence-transformers"
    model_name = "e2e-embedding-model"
    model_version = "e2e-embedding-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: tuple[str, ...] | list[str]) -> list[list[float]]:
        batch = tuple(texts)
        self.calls.append(batch)
        vectors: list[list[float]] = []
        for text in batch:
            if "mvcc" in text.casefold():
                vectors.append([1.0, 0.0])
            elif "vacuum" in text.casefold():
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.0, 0.25])
        return vectors


class DeterministicReranker:
    provider_name = "sentence-transformers-cross-encoder"
    model_name = "e2e-reranker"
    model_version = "e2e-reranker-v1"

    def score(self, query: str, candidate_texts: tuple[str, ...] | list[str]) -> list[float]:
        return [
            1.0 if "mvcc" in text.casefold() else 0.1
            for text in candidate_texts
        ]


def _write_pdf(path: Path, pages: tuple[tuple[str, ...], ...]) -> None:
    """Create a small text PDF with predictable line and page boundaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    page_ids = [3 + page_index * 2 for page_index in range(len(pages))]
    font_id = 3 + len(pages) * 2
    info_id = font_id + 1
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{item} 0 R' for item in page_ids)}] "
            f"/Count {len(pages)} >>"
        ).encode("ascii"),
    ]
    for page_index, lines in enumerate(pages):
        page_id = page_ids[page_index]
        stream_id = page_id + 1
        commands = ["BT /F1 12 Tf 72 720 Td"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({escaped}) Tj 0 -18 Td")
        commands.append("ET")
        stream = " ".join(commands).encode("ascii")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {stream_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Title (PostgreSQL Concurrency Manual) >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(body + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(pdf)


def _settings(root: Path) -> Settings:
    return Settings.from_environment(
        {
            "PDF_RAG_CORPUS_DIR": "data/pdfs",
            "PDF_RAG_DATABASE_PATH": "data/duckdb/e2e.duckdb",
            "PDF_RAG_CHUNK_SIZE": "70",
            "PDF_RAG_CHUNK_OVERLAP": "0",
            "PDF_RAG_SEMANTIC_TOP_K": "10",
            "PDF_RAG_KEYWORD_TOP_K": "10",
            "PDF_RAG_HYBRID_TOP_K": "5",
            "PDF_RAG_RERANKER_TOP_K": "3",
            "PDF_RAG_CONTEXT_MAX_CHUNKS": "2",
            "PDF_RAG_CONTEXT_MAX_CHARACTERS": "300",
            "PDF_RAG_EMBEDDING_PROVIDER": "sentence-transformers",
            "PDF_RAG_EMBEDDING_MODEL": "e2e-embedding-model",
            "PDF_RAG_RERANKER_PROVIDER": "sentence-transformers-cross-encoder",
            "PDF_RAG_RERANKER_MODEL": "e2e-reranker",
        },
        project_root=root,
    )


def test_generated_pdf_ingestion_retrieval_rerank_context_and_repeat_index(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    valid_pdf = settings.corpus_dir / "postgres" / "concurrency.pdf"
    _write_pdf(
        valid_pdf,
        (
            (
                "CONCURRENCY CONTROL",
                "Readers keep a stable snapshot while transactions proceed.",
            ),
            (
                "PostgreSQL MVCC preserves row versions for concurrent transactions.",
                "Vacuum removes obsolete tuples after snapshots release them.",
            ),
        ),
    )
    broken_pdf = settings.corpus_dir / "broken.pdf"
    broken_pdf.parent.mkdir(parents=True, exist_ok=True)
    broken_pdf.write_bytes(b"not a PDF")
    embedder = DeterministicEmbeddingProvider()

    first_run = run_ingestion(settings, embedding_provider=embedder)

    assert first_run.discovered_pdf_count == 2
    assert first_run.indexed_document_count == 1
    assert first_run.chunk_count >= 2
    assert first_run.embedding_count == first_run.chunk_count
    assert first_run.status == "partial"
    assert first_run.failure_count == 1
    assert first_run.failures[0].source_path == "broken.pdf"
    calls_after_first_run = len(embedder.calls)
    with DuckDBStore(settings.database_path) as store:
        counts_after_first_run = store.counts()
        documents = store.list_documents()
        assert len(documents) == 1
        document_id = documents[0]["document_id"]
        stored_chunks = store.get_chunks(document_id)
        assert len(stored_chunks) == first_run.chunk_count
        assert any("MVCC" in chunk.text for chunk in stored_chunks)

    second_run = run_ingestion(settings, embedding_provider=embedder)
    assert second_run.reused_document_count == 1
    assert second_run.chunk_count == first_run.chunk_count
    assert len(embedder.calls) == calls_after_first_run
    with DuckDBStore(settings.database_path) as store:
        assert store.counts() == counts_after_first_run

        ranked, note, reranker_status = search_index(
            settings,
            "How does MVCC preserve row versions?",
            "hybrid",
            1,
            embedding_provider=embedder,
            reranker=DeterministicReranker(),
            use_reranker=True,
            expand_hierarchy=True,
        )

    assert note is None
    assert reranker_status == "applied"
    assert len(ranked) == 1
    evidence: RetrievalHit = ranked[0]
    assert "MVCC" in evidence.text
    assert evidence.source_filename == "concurrency.pdf"
    assert evidence.source_path == "postgres/concurrency.pdf"
    assert evidence.page_start == evidence.page_end == 2
    assert evidence.section_path[-1] == "CONCURRENCY CONTROL"
    assert evidence.semantic_score is not None
    assert evidence.keyword_score is not None
    assert evidence.hybrid_score is not None
    assert evidence.reranker_score == 1.0
    assert evidence.context_chunks
    assert all(item.chunk.document_id == evidence.document_id for item in evidence.context_chunks)
    assert all(item.chunk.chunk_id != evidence.chunk_id for item in evidence.context_chunks)
    assert all(item.chunk.page_start <= item.chunk.page_end for item in evidence.context_chunks)


def test_streamlit_smoke_displays_end_to_end_indexed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    _write_pdf(
        settings.corpus_dir / "iceberg" / "iceberg.pdf",
        (
            (
                "SNAPSHOT ISOLATION",
                "Iceberg snapshots identify a consistent table state for readers.",
                "Snapshot metadata tracks sequence numbers and manifest references.",
            ),
        ),
    )
    (settings.corpus_dir / "iceberg" / "not-a-pdf.pdf").write_bytes(b"broken")
    summary = run_ingestion(settings, embedding_provider=DeterministicEmbeddingProvider())
    assert summary.indexed_document_count == 1

    monkeypatch.setenv("PDF_RAG_CORPUS_DIR", str(settings.corpus_dir))
    monkeypatch.setenv("PDF_RAG_DATABASE_PATH", str(settings.database_path))
    monkeypatch.setenv("PDF_RAG_EMBEDDING_PROVIDER", "")
    monkeypatch.setenv("PDF_RAG_EMBEDDING_MODEL", "")
    monkeypatch.setenv("PDF_RAG_RERANKER_PROVIDER", "")
    monkeypatch.setenv("PDF_RAG_RERANKER_MODEL", "")

    app_path = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    assert not app.exception
    app.selectbox(key="retrieval_method").select("Keyword")
    app.text_input(key="retrieval_query").set_value("snapshots")
    app.button(key="search_button").click().run()

    assert not app.exception
    assert any("Documents" in metric.label and metric.value == "1" for metric in app.metric)
    assert any("iceberg.pdf" in item.label for item in app.expander)
    assert any("Iceberg snapshots identify" in item.value for item in app.text)
    assert any("SNAPSHOT ISOLATION" in item.value for item in app.caption)
    assert any("Related context (neighbor)" in item.value for item in app.caption)
    assert any("Snapshot metadata tracks" in item.value for item in app.text)


def test_generated_pdf_evidence_can_be_searched_again_after_store_reopen(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    _write_pdf(
        settings.corpus_dir / "postgres.pdf",
        (("TRANSACTIONS", "PostgreSQL transactions use MVCC for row visibility."),),
    )
    run_ingestion(settings, embedding_provider=DeterministicEmbeddingProvider())

    first, _, _ = search_index(settings, "MVCC", "keyword", 5)
    reopened_store = DuckDBStore(settings.database_path)
    reopened_store.initialize()
    reopened_store.close()
    second, _, _ = search_index(settings, "MVCC", "keyword", 5)

    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]
    assert second[0].source_filename == "postgres.pdf"
    assert second[0].page_start == 1
    assert "MVCC" in second[0].text
