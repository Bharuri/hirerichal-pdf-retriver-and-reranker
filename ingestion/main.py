"""Manual document-to-DuckDB ingestion command; run with ``python -m ingestion.main``."""

from __future__ import annotations

import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from config import Settings
from db.duckdb import DuckDBError, DuckDBStore, IndexRunRecord
from ingestion.chunker import RetrievalChunk, chunk_document
from ingestion.hierarchy import HierarchyResult, detect_hierarchy
from ingestion.indexer import (
    EmbeddingProvider,
    create_embedding_provider,
    embed_chunks,
)
from ingestion.pdf_loader import (
    PdfCorpusError,
    PdfReadResult,
    PdfStatus,
    extract_corpus,
)


logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Raised when the manual ingestion run cannot be started or finalized."""


@dataclass(frozen=True)
class IngestionFailure:
    """Safe failure information for one document or chunk embedding."""

    source_path: str
    reason: str
    chunk_id: str | None = None

@dataclass(frozen=True)
class IngestionSummary:
    """Counts and failures from one manual document indexing run."""

    run_id: str
    status: str
    discovered_pdf_count: int
    indexed_document_count: int
    chunk_count: int
    embedding_count: int
    embedding_failure_count: int
    reused_document_count: int
    database_path: Path
    failures: tuple[IngestionFailure, ...] = ()

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def discovered_document_count(self) -> int:
        """Return the number of discovered PDF and text documents."""
        return self.discovered_pdf_count


def run_ingestion(
    settings: Settings | None = None,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> IngestionSummary:
    """Extract, structure, chunk, store and embed documents from the configured corpus.

    The routine is intended for explicit/manual invocation, not Streamlit startup.
    Existing document indexes and compatible embeddings are reused when their
    extracted pages, hierarchy and deterministic chunks are unchanged.
    """
    active_settings = settings or Settings.from_environment()
    try:
        active_settings.corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus = extract_corpus(active_settings.corpus_dir)
    except (OSError, PdfCorpusError) as error:
        raise IngestionError(
            "The configured PDF corpus could not be read. Check PDF_RAG_CORPUS_DIR."
        ) from error

    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)
    discovered_count = len(corpus.documents)
    logger.info(
        "Manual ingestion discovered %d document(s) in the configured corpus",
        discovered_count,
    )
    if discovered_count == 0:
        return IngestionSummary(
            run_id=run_id,
            status="empty",
            discovered_pdf_count=0,
            indexed_document_count=0,
            chunk_count=0,
            embedding_count=0,
            embedding_failure_count=0,
            reused_document_count=0,
            database_path=active_settings.database_path,
        )

    failures: list[IngestionFailure] = []
    indexed_document_count = 0
    chunk_count = 0
    embedding_count = 0
    embedding_failure_count = 0
    reused_document_count = 0
    provider = embedding_provider
    provider_load_error: str | None = None

    try:
        with DuckDBStore(active_settings.database_path) as store:
            for document in corpus.documents:
                if document.status in {PdfStatus.UNREADABLE, PdfStatus.ENCRYPTED} or not document.pages:
                    reason = f"Document extraction was {document.status.value}; no indexable pages were available."
                    failures.append(IngestionFailure(document.source_path, reason))
                    logger.warning("Skipping %s: %s", document.source_path, reason)
                    continue

                try:
                    hierarchy = detect_hierarchy(document)
                    chunks = chunk_document(
                        document,
                        hierarchy,
                        chunk_size=active_settings.chunk_size,
                        chunk_overlap=active_settings.chunk_overlap,
                        size_unit=active_settings.chunk_size_unit,
                    )
                    unchanged = _existing_index_matches(
                        store, document, hierarchy, chunks
                    )
                    if unchanged:
                        reused_document_count += 1
                        logger.info("Reusing unchanged document index: %s", document.source_path)
                    else:
                        store.replace_document_index(document, hierarchy, chunks)

                    indexed_document_count += 1
                    chunk_count += len(chunks)
                    if not chunks or (
                        provider is None
                        and active_settings.embedding_provider is None
                    ):
                        continue

                    if provider is None and provider_load_error is None:
                        try:
                            provider = create_embedding_provider(active_settings)
                        except Exception as error:
                            provider_load_error = type(error).__name__
                            logger.warning(
                                "Embedding provider unavailable for %s (%s)",
                                document.source_path,
                                provider_load_error,
                            )
                    if provider_load_error is not None:
                        reason = (
                            "Embedding provider could not be initialized "
                            f"({provider_load_error})."
                        )
                        embedding_failure_count += len(chunks)
                        failures.extend(
                            IngestionFailure(document.source_path, reason, chunk.chunk_id)
                            for chunk in chunks
                        )
                        continue

                    embedding_result = embed_chunks(
                        chunks,
                        store,
                        provider,
                        settings=active_settings,
                        on_embedding_start=lambda chunk, position, total: logger.info(
                            "Embedding chunk %d/%d: %s",
                            position,
                            total,
                            chunk.chunk_id,
                        ),
                    )
                    embedding_count += len(embedding_result.embeddings)
                    embedding_failure_count += len(embedding_result.failures)
                    if embedding_result.reused_count:
                        logger.info(
                            "Reused %d/%d stored embeddings for %s",
                            embedding_result.reused_count,
                            len(chunks),
                            document.source_path,
                        )
                    failures.extend(
                        IngestionFailure(
                            document.source_path,
                            item.reason,
                            item.chunk_id,
                        )
                        for item in embedding_result.failures
                    )
                except Exception as error:
                    reason = f"Document indexing failed ({type(error).__name__})."
                    failures.append(IngestionFailure(document.source_path, reason))
                    logger.exception("Could not index document %s", document.source_path)

            finished_at = datetime.now(timezone.utc)
            status = _run_status(indexed_document_count, failures)
            try:
                store.save_index_run(
                    IndexRunRecord(
                        run_id=run_id,
                        status=status,
                        document_count=indexed_document_count,
                        chunk_count=chunk_count,
                        failure_count=len(failures),
                        started_at=started_at,
                        finished_at=finished_at,
                        safe_error_summary=(
                            f"{len(failures)} document/chunk failure(s)"
                            if failures
                            else None
                        ),
                    )
                )
            except (DuckDBError, ValueError) as error:
                raise IngestionError(
                    "The ingestion run summary could not be stored in DuckDB."
                ) from error
    except DuckDBError as error:
        raise IngestionError(
            "DuckDB could not be opened or updated during manual ingestion."
        ) from error

    summary = IngestionSummary(
        run_id=run_id,
        status=status,
        discovered_pdf_count=discovered_count,
        indexed_document_count=indexed_document_count,
        chunk_count=chunk_count,
        embedding_count=embedding_count,
        embedding_failure_count=embedding_failure_count,
        reused_document_count=reused_document_count,
        database_path=active_settings.database_path,
        failures=tuple(failures),
    )
    logger.info(
        "Ingestion %s: documents=%d chunks=%d embeddings=%d failures=%d",
        summary.status,
        summary.indexed_document_count,
        summary.chunk_count,
        summary.embedding_count,
        summary.failure_count,
    )
    return summary


def _existing_index_matches(
    store: DuckDBStore,
    document: PdfReadResult,
    hierarchy: HierarchyResult,
    chunks: Sequence[RetrievalChunk],
) -> bool:
    """Check source and derived records to retain valid vectors on repeat runs."""
    existing = store.get_document(document.document_id)
    if existing is None:
        return False
    if (
        existing["source_filename"] != document.source_filename
        or existing["source_path"] != document.source_path
        or existing["content_sha256"] != document.content_sha256
        or existing["page_count"] != document.page_count
        or existing["status"] != document.status.value
        or existing["metadata"] != dict(document.metadata)
        or existing["warnings"] != tuple(document.warnings)
    ):
        return False

    stored_pages = store.get_pages(document.document_id)
    expected_pages = tuple(
        {
            "page_number": page.page_number,
            "text": page.text,
            "warnings": tuple(page.warnings),
        }
        for page in document.pages
    )
    return (
        stored_pages == expected_pages
        and store.get_hierarchy(document.document_id) == hierarchy.nodes
        and store.get_chunks(document.document_id) == tuple(chunks)
    )


def _run_status(indexed_document_count: int, failures: Sequence[IngestionFailure]) -> str:
    if not failures:
        return "completed"
    return "partial" if indexed_document_count else "failed"


def main() -> int:
    """Run the configured manual ingestion from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        summary = run_ingestion()
    except IngestionError as error:
        logger.error("%s", error)
        return 2

    print(
        "Ingestion "
        f"{summary.status}: documents={summary.discovered_document_count}, "
        f"documents={summary.indexed_document_count}, chunks={summary.chunk_count}, "
        f"embeddings={summary.embedding_count}, failures={summary.failure_count}"
    )
    print(f"DuckDB: {summary.database_path}")
    for failure in summary.failures:
        target = f" [{failure.chunk_id}]" if failure.chunk_id else ""
        print(f"- {failure.source_path}{target}: {failure.reason}")
    return 1 if summary.failure_count else 0


if __name__ == "__main__":
    sys.exit(main())
