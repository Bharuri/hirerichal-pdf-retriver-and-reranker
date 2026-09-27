"""TASK-09 tests for the search-only Streamlit application."""

from pathlib import Path
from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

from app import (
    SearchApplicationError,
    get_database_tables,
    get_index_snapshot,
    query_database_table,
    search_index,
)
from config import Settings
from db.duckdb import DuckDBStore
from ingestion.chunker import RetrievalChunk
from ingestion.hierarchy import HierarchyNode, HierarchyResult
from ingestion.pdf_loader import PageText, PdfReadResult, PdfStatus


class QueryProvider:
    provider_name = "fake"
    model_name = "test-model"
    model_version = "v1"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _settings(tmp_path: Path, *, indexed: bool = False) -> Settings:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    environment = {
        "PDF_RAG_CORPUS_DIR": str(corpus_dir),
        "PDF_RAG_DATABASE_PATH": str(tmp_path / "manual-index.duckdb"),
        "PDF_RAG_EMBEDDING_PROVIDER": "" if not indexed else "fake",
        "PDF_RAG_EMBEDDING_MODEL": "" if not indexed else "test-model",
    }
    return Settings.from_environment(environment, project_root=tmp_path)


def _create_manual_index(settings: Settings) -> RetrievalChunk:
    document = PdfReadResult(
        document_id="postgres-doc",
        source_filename="postgres-manual.pdf",
        source_path="postgres/postgres-manual.pdf",
        content_sha256="document-hash",
        page_count=1,
        metadata={"title": "PostgreSQL Manual"},
        pages=(PageText(1, "MVCC preserves concurrent row versions."),),
        status=PdfStatus.EXTRACTED,
    )
    root = HierarchyNode(
        node_id="root",
        document_id="postgres-doc",
        parent_id=None,
        level="document",
        section_title="PostgreSQL Manual",
        section_path=("PostgreSQL Manual",),
        page_start=1,
        page_end=1,
        sequence=0,
        detection_method="document_root",
    )
    section = HierarchyNode(
        node_id="section",
        document_id="postgres-doc",
        parent_id="root",
        level="section",
        section_title="Concurrency Control",
        section_path=("PostgreSQL Manual", "Concurrency Control"),
        page_start=1,
        page_end=1,
        sequence=1,
        detection_method="test_outline",
    )
    content = HierarchyNode(
        node_id="content",
        document_id="postgres-doc",
        parent_id="section",
        level="content",
        section_title=None,
        section_path=section.section_path,
        page_start=1,
        page_end=1,
        sequence=2,
        detection_method="paragraph_content",
        text="MVCC preserves concurrent row versions.",
    )
    chunk = RetrievalChunk(
        chunk_id="postgres-mvcc-chunk",
        document_id="postgres-doc",
        parent_id="section",
        level="content",
        section_title="Concurrency Control",
        section_path=section.section_path,
        previous_chunk_id=None,
        next_chunk_id=None,
        page_start=1,
        page_end=1,
        sequence=0,
        text="MVCC preserves concurrent row versions.",
        chunk_size=40,
        size_unit="characters",
        token_count=5,
        content_hash="chunk-hash",
        configuration_fingerprint="test-config",
    )
    with DuckDBStore(settings.database_path) as store:
        store.replace_document_index(
            document,
            HierarchyResult("postgres-doc", (root, section, content)),
            (chunk,),
        )
    return chunk


def test_index_snapshot_does_not_create_missing_database(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    snapshot = get_index_snapshot(settings)

    assert not snapshot.database_exists
    assert not settings.database_path.exists()


def test_search_index_runs_keyword_retrieval_on_manual_index(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    chunk = _create_manual_index(settings)

    hits, note, reranker_status = search_index(
        settings,
        "  MVCC ",
        "keyword",
        5,
        expand_hierarchy=True,
    )

    assert [hit.chunk_id for hit in hits] == [chunk.chunk_id]
    assert hits[0].source_filename == "postgres-manual.pdf"
    assert hits[0].page_start == hits[0].page_end == 1
    assert hits[0].section_path == ("PostgreSQL Manual", "Concurrency Control")
    assert hits[0].keyword_score is not None
    assert hits[0].context_chunks == ()
    assert note is None
    assert reranker_status is None


def test_search_does_not_require_access_to_the_pdf_corpus(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    chunk = _create_manual_index(settings)
    settings_without_corpus = replace(settings, corpus_dir=tmp_path / "not-mounted")

    hits, _, _ = search_index(
        settings_without_corpus, "MVCC", "keyword", 5
    )

    assert [hit.chunk_id for hit in hits] == [chunk.chunk_id]


def test_database_tab_service_lists_and_queries_tables_read_only(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    _create_manual_index(settings)

    table_names = get_database_tables(settings)
    preview = query_database_table(settings, "chunks", limit=1)

    assert {"documents", "chunks", "pages", "embeddings"} <= set(table_names)
    assert preview.table_name == "chunks"
    assert "chunk_id" in preview.columns
    assert len(preview.rows) == 1
    assert preview.rows[0][preview.columns.index("chunk_id")] == "postgres-mvcc-chunk"
    with DuckDBStore(settings.database_path) as store:
        assert store.counts()["chunks"] == 1


def test_database_tab_lists_names_and_renders_selected_table_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    _create_manual_index(settings)
    monkeypatch.setenv("PDF_RAG_CORPUS_DIR", str(settings.corpus_dir))
    monkeypatch.setenv("PDF_RAG_DATABASE_PATH", str(settings.database_path))
    monkeypatch.setenv("PDF_RAG_EMBEDDING_PROVIDER", "")
    monkeypatch.setenv("PDF_RAG_EMBEDDING_MODEL", "")

    app_path = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    assert not app.exception
    assert "chunks" in app.selectbox(key="database_table").options
    app.selectbox(key="database_table").select("chunks")
    app.button(key="database_query_button").click().run()

    assert not app.exception
    assert any("Showing up to" in item.value for item in app.caption)
    assert len(app.dataframe) == 1


def test_search_index_supports_semantic_retrieval_with_injected_provider(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    chunk = _create_manual_index(settings)
    from db.duckdb import EmbeddingRecord

    with DuckDBStore(settings.database_path) as store:
        store.save_embeddings(
            (
                EmbeddingRecord(
                    chunk.chunk_id,
                    "fake",
                    "test-model",
                    "v1",
                    chunk.content_hash,
                    (1.0, 0.0),
                ),
            )
        )

    hits, _, _ = search_index(
        settings, "mvcc", "semantic", 2, embedding_provider=QueryProvider()
    )

    assert [hit.chunk_id for hit in hits] == [chunk.chunk_id]
    assert hits[0].semantic_score == pytest.approx(1.0)


def test_search_requires_manually_created_index_and_nonempty_query(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(SearchApplicationError, match="Ingest PDFs manually"):
        search_index(settings, "mvcc", "keyword", 5)

    with DuckDBStore(settings.database_path):
        pass
    with pytest.raises(SearchApplicationError, match="no searchable chunks"):
        search_index(settings, "mvcc", "keyword", 5)

    with pytest.raises(SearchApplicationError, match="Enter a search query"):
        search_index(settings, "  ", "keyword", 5)


def test_streamlit_smoke_renders_search_and_read_only_manual_index_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setenv("PDF_RAG_CORPUS_DIR", str(settings.corpus_dir))
    monkeypatch.setenv("PDF_RAG_DATABASE_PATH", str(settings.database_path))
    monkeypatch.setenv("PDF_RAG_EMBEDDING_PROVIDER", "")
    monkeypatch.setenv("PDF_RAG_EMBEDDING_MODEL", "")

    app_path = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(str(app_path)).run()

    assert not app.exception
    assert any("Local PDF Retrieval" in item.value for item in app.title)
    assert any("manual and separate" in item.value for item in app.info)
    assert any("No DuckDB index file" in item.value for item in app.warning)
    assert not settings.database_path.exists()


def test_streamlit_search_form_displays_ranked_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    _create_manual_index(settings)
    monkeypatch.setenv("PDF_RAG_CORPUS_DIR", str(settings.corpus_dir))
    monkeypatch.setenv("PDF_RAG_DATABASE_PATH", str(settings.database_path))
    monkeypatch.setenv("PDF_RAG_EMBEDDING_PROVIDER", "")
    monkeypatch.setenv("PDF_RAG_EMBEDDING_MODEL", "")

    app_path = Path(__file__).resolve().parents[2] / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    app.selectbox(key="retrieval_method").select("Keyword")
    app.text_input(key="retrieval_query").set_value("MVCC")
    app.checkbox(key="expand_hierarchy").uncheck()
    app.button(key="search_button").click().run()

    assert not app.exception
    assert any("postgres-manual.pdf" in item.label for item in app.expander)
    assert any(
        "MVCC preserves concurrent row versions." in item.value for item in app.text
    )