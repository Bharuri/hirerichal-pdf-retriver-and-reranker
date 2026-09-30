"""Small local DuckDB store for extracted PDFs and retrieval records."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Sequence
import duckdb

from ingestion.chunker import RetrievalChunk
from ingestion.hierarchy import HierarchyNode, HierarchyResult
from ingestion.pdf_loader import PdfReadResult


class DuckDBError(RuntimeError):
    """Raised when local DuckDB storage cannot complete a safe operation."""


@dataclass(frozen=True)
class EmbeddingRecord:
    """Vector plus provider and content identity supplied by an embedding adapter."""

    chunk_id: str
    provider_name: str
    model_name: str
    model_version: str
    content_hash: str
    vector: tuple[float, ...]
    created_at: datetime | None = None


@dataclass(frozen=True)
class StoredChunk:
    """A persisted retrieval chunk with its source document reference."""

    chunk: RetrievalChunk
    source_filename: str
    source_path: str


@dataclass(frozen=True)
class IndexRunRecord:
    """Small safe summary of a local indexing run."""

    run_id: str
    status: str
    document_count: int
    chunk_count: int
    failure_count: int
    started_at: datetime
    finished_at: datetime | None = None
    safe_error_summary: str | None = None


class DuckDBStore:
    """Single-process DuckDB connection with atomic per-document replacement."""

    def __init__(self, database_path: str | Path) -> None:
        value = str(database_path).strip()
        if not value:
            raise ValueError("database_path must not be empty")
        self.database_path = value
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._lock = threading.RLock()

    @property
    def is_open(self) -> bool:
        return self._connection is not None

    def initialize(self) -> None:
        """Open DuckDB and create missing tables/indexes from schema.sql."""
        with self._lock:
            if self._connection is not None:
                return
            if self.database_path != ":memory:":
                path = Path(self.database_path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                self.database_path = str(path.resolve())
            connection: duckdb.DuckDBPyConnection | None = None
            try:
                connection = duckdb.connect(self.database_path)
                schema = (
                    Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
                )
                connection.execute(schema)
                self._upgrade_embedding_schema(connection)
            except Exception as error:
                if connection is not None:
                    connection.close()
                raise DuckDBError(
                    "DuckDB could not be initialized from the local schema."
                ) from error
            self._connection = connection

    def replace_document_index(
        self,
        document: PdfReadResult,
        hierarchy: HierarchyResult,
        chunks: Sequence[RetrievalChunk],
        embeddings: Sequence[EmbeddingRecord] = (),
    ) -> None:
        """Atomically replace one document and all its derived index records."""
        self._require_connection()
        self._validate_index(document, hierarchy, chunks, embeddings)
        connection = self._require_connection()
        with self._lock:
            try:
                connection.execute("BEGIN TRANSACTION")
                self._delete_derived(connection, document.document_id)
                self._upsert_document(connection, document)
                connection.executemany(
                    "INSERT INTO pages VALUES (?, ?, ?, ?)",
                    [
                        (
                            document.document_id,
                            page.page_number,
                            page.text,
                            _json(list(page.warnings)),
                        )
                        for page in document.pages
                    ],
                )
                self._insert_hierarchy(connection, hierarchy.nodes)
                connection.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            item.chunk_id,
                            item.document_id,
                            item.parent_id,
                            item.level,
                            item.section_title,
                            _json(list(item.section_path)),
                            item.previous_chunk_id,
                            item.next_chunk_id,
                            item.page_start,
                            item.page_end,
                            item.sequence,
                            item.text,
                            item.chunk_size,
                            item.size_unit,
                            item.token_count,
                            item.content_hash,
                            item.configuration_fingerprint,
                            item.chunker_version,
                        )
                        for item in chunks
                    ],
                )
                for item in embeddings:
                    connection.execute(
                        "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            item.chunk_id,
                            item.provider_name,
                            item.model_name,
                            item.model_version,
                            item.content_hash,
                            len(item.vector),
                            list(item.vector),
                            _datetime_string(item.created_at or datetime.now(timezone.utc)),
                        ],
                    )
                connection.execute("COMMIT")
            except Exception as error:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                if isinstance(error, DuckDBError):
                    raise
                raise DuckDBError(
                    "Document index replacement failed and was rolled back."
                ) from error

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        connection = self._require_connection()
        with self._lock:
            row = connection.execute(
                """SELECT document_id, source_filename, source_path, content_sha256,
						  page_count, status, metadata_json, warnings_json, indexed_at, parser_version
				   FROM documents WHERE document_id = ?""",
                [document_id],
            ).fetchone()
        return _document_from_row(row) if row is not None else None

    def list_documents(self) -> tuple[dict[str, Any], ...]:
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """SELECT document_id, source_filename, source_path, content_sha256,
						  page_count, status, metadata_json, warnings_json, indexed_at, parser_version
				   FROM documents ORDER BY lower(source_path), document_id"""
            ).fetchall()
        return tuple(_document_from_row(row) for row in rows)

    def list_table_names(self) -> tuple[str, ...]:
        """List user tables in DuckDB's main schema in stable order."""
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """SELECT table_name
                   FROM information_schema.tables
                   WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                   ORDER BY lower(table_name), table_name"""
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def preview_table(
        self, table_name: str, *, limit: int = 100
    ) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
        """Read a bounded sample from a listed main-schema table only."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if table_name not in self.list_table_names():
            raise DuckDBError("Requested table does not exist in the local index.")
        quoted_name = '"' + table_name.replace('"', '""') + '"'
        connection = self._require_connection()
        with self._lock:
            try:
                cursor = connection.execute(
                    f"SELECT * FROM {quoted_name} LIMIT ?", [limit]
                )
                rows = cursor.fetchall()
                columns = tuple(str(description[0]) for description in cursor.description or ())
            except Exception as error:
                raise DuckDBError("The selected local index table could not be queried.") from error
        return columns, tuple(tuple(row) for row in rows)

    def get_pages(self, document_id: str) -> tuple[dict[str, Any], ...]:
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                "SELECT page_number, text, warnings_json FROM pages WHERE document_id = ? ORDER BY page_number",
                [document_id],
            ).fetchall()
        return tuple(
            {
                "page_number": row[0],
                "text": row[1],
                "warnings": tuple(_decode_json(row[2], list)),
            }
            for row in rows
        )

    def get_hierarchy(self, document_id: str) -> tuple[HierarchyNode, ...]:
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """SELECT node_id, document_id, parent_id, level, section_title,
						  section_path_json, page_start, page_end, sequence,
						  detection_method, confidence, text
				   FROM hierarchy_nodes WHERE document_id = ? ORDER BY sequence, node_id""",
                [document_id],
            ).fetchall()
        return tuple(
            HierarchyNode(
                node_id=row[0],
                document_id=row[1],
                parent_id=row[2],
                level=row[3],
                section_title=row[4],
                section_path=tuple(_decode_json(row[5], list)),
                page_start=row[6],
                page_end=row[7],
                sequence=row[8],
                detection_method=row[9],
                confidence=row[10],
                text=row[11],
            )
            for row in rows
        )

    def get_chunks(self, document_id: str) -> tuple[RetrievalChunk, ...]:
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """SELECT chunk_id, document_id, parent_id, level, section_title,
						  section_path_json, previous_chunk_id, next_chunk_id,
						  page_start, page_end, sequence, text, chunk_size, size_unit,
						  token_count, content_hash, configuration_fingerprint, chunker_version
				   FROM chunks WHERE document_id = ? ORDER BY sequence, chunk_id""",
                [document_id],
            ).fetchall()
        return tuple(_chunk_from_row(row) for row in rows)

    def search_keyword_rows(
        self, term_patterns: Sequence[str], top_k: int
    ) -> tuple[tuple[StoredChunk, float], ...]:
        """Search chunk text and hierarchy fields using parameterized regex terms."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not term_patterns:
            return ()
        score_terms: list[str] = []
        parameters: list[Any] = []
        for pattern in term_patterns:
            score_terms.extend(
                (
                    "CASE WHEN regexp_matches(lower(coalesce(c.text, '')), ?) THEN 1.0 ELSE 0.0 END",
                    "CASE WHEN regexp_matches(lower(coalesce(c.section_title, '')), ?) THEN 2.0 ELSE 0.0 END",
                    "CASE WHEN regexp_matches(lower(c.section_path_json), ?) THEN 2.0 ELSE 0.0 END",
                )
            )
            parameters.extend((pattern, pattern, pattern))
        score_expression = " + ".join(score_terms)
        query = f"""
            WITH scored AS (
                SELECT c.chunk_id, c.document_id, c.parent_id, c.level,
                       c.section_title, c.section_path_json, c.previous_chunk_id,
                       c.next_chunk_id, c.page_start, c.page_end, c.sequence,
                       c.text, c.chunk_size, c.size_unit, c.token_count,
                       c.content_hash, c.configuration_fingerprint, c.chunker_version,
                       d.source_filename, d.source_path,
                       ({score_expression}) AS score
                FROM chunks AS c
                JOIN documents AS d ON d.document_id = c.document_id
            )
            SELECT * FROM scored
            WHERE score > 0
            ORDER BY score DESC, lower(source_path), sequence, chunk_id
            LIMIT ?
        """
        parameters.append(top_k)
        connection = self._require_connection()
        with self._lock:
            try:
                rows = connection.execute(query, parameters).fetchall()
            except Exception as error:
                raise DuckDBError("Keyword retrieval could not query the local index.") from error
        return tuple(
            (_stored_chunk_from_row(row), float(row[20])) for row in rows
        )

    def get_compatible_embedding_dimensions(
        self, provider_name: str, model_name: str, model_version: str
    ) -> tuple[int, ...]:
        """Return dimensions for current chunk vectors with the requested model identity."""
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """SELECT DISTINCT e.dimension
                   FROM embeddings AS e
                   JOIN chunks AS c ON c.chunk_id = e.chunk_id
                   WHERE e.provider_name = ? AND e.model_name = ?
                     AND e.model_version = ? AND e.content_hash = c.content_hash
                   ORDER BY e.dimension""",
                [provider_name, model_name, model_version],
            ).fetchall()
        return tuple(int(row[0]) for row in rows)

    def get_compatible_embedding_rows(
        self,
        provider_name: str,
        model_name: str,
        model_version: str,
        dimension: int,
    ) -> tuple[tuple[StoredChunk, tuple[float, ...]], ...]:
        """Return current chunk vectors matching provider, model, version and dimension."""
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        connection = self._require_connection()
        with self._lock:
            rows = connection.execute(
                """SELECT c.chunk_id, c.document_id, c.parent_id, c.level,
                          c.section_title, c.section_path_json, c.previous_chunk_id,
                          c.next_chunk_id, c.page_start, c.page_end, c.sequence,
                          c.text, c.chunk_size, c.size_unit, c.token_count,
                          c.content_hash, c.configuration_fingerprint, c.chunker_version,
                          d.source_filename, d.source_path, e.vector
                   FROM embeddings AS e
                   JOIN chunks AS c ON c.chunk_id = e.chunk_id
                     AND c.content_hash = e.content_hash
                   JOIN documents AS d ON d.document_id = c.document_id
                   WHERE e.provider_name = ? AND e.model_name = ?
                     AND e.model_version = ? AND e.dimension = ?
                   ORDER BY lower(d.source_path), c.sequence, c.chunk_id""",
                [provider_name, model_name, model_version, dimension],
            ).fetchall()
        return tuple(
            (_stored_chunk_from_row(row), tuple(row[20])) for row in rows
        )

    def get_embedding(
        self,
        chunk_id: str,
        provider_name: str,
        model_name: str,
        model_version: str,
        content_hash: str | None = None,
    ) -> EmbeddingRecord | None:
        connection = self._require_connection()
        content_filter = " AND content_hash = ?" if content_hash is not None else ""
        parameters: list[Any] = [chunk_id, provider_name, model_name, model_version]
        if content_hash is not None:
            parameters.append(content_hash)
        with self._lock:
            row = connection.execute(
                """SELECT chunk_id, provider_name, model_name, model_version,
                        content_hash, vector, created_at
                   FROM embeddings
                   WHERE chunk_id = ? AND provider_name = ? AND model_name = ?
                     AND model_version = ?""" + content_filter,
                parameters,
            ).fetchone()
        return (
            EmbeddingRecord(
                row[0], row[1], row[2], row[3], row[4], tuple(row[5]),
                _parse_datetime(row[6])
            )
            if row
            else None
        )

    def save_embeddings(self, embeddings: Sequence[EmbeddingRecord]) -> None:
        """Atomically persist vectors only for matching, already-stored chunks."""
        if not embeddings:
            return
        connection = self._require_connection()
        keys: set[tuple[str, str, str, str]] = set()
        with self._lock:
            for item in embeddings:
                _validate_embedding_record(item)
                key = (
                    item.chunk_id,
                    item.provider_name,
                    item.model_name,
                    item.model_version,
                )
                if key in keys:
                    raise DuckDBError("Duplicate chunk/provider/model embeddings are not allowed.")
                keys.add(key)
                row = connection.execute(
                    "SELECT content_hash FROM chunks WHERE chunk_id = ?", [item.chunk_id]
                ).fetchone()
                if row is None:
                    raise DuckDBError("Embedding references a missing chunk.")
                if row[0] != item.content_hash:
                    raise DuckDBError("Embedding content hash does not match the stored chunk.")

            try:
                connection.execute("BEGIN TRANSACTION")
                connection.executemany(
                    """INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (chunk_id, provider_name, model_name, model_version)
                       DO UPDATE SET content_hash = excluded.content_hash,
                         dimension = excluded.dimension, vector = excluded.vector,
                         created_at = excluded.created_at""",
                    [
                        (
                            item.chunk_id,
                            item.provider_name,
                            item.model_name,
                            item.model_version,
                            item.content_hash,
                            len(item.vector),
                            list(item.vector),
                            _datetime_string(item.created_at or datetime.now(timezone.utc)),
                        )
                        for item in embeddings
                    ],
                )
                connection.execute("COMMIT")
            except Exception as error:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                if isinstance(error, DuckDBError):
                    raise
                raise DuckDBError("Chunk embeddings could not be stored and were rolled back.") from error

    def save_index_run(self, run: IndexRunRecord) -> None:
        _validate_run(run)
        connection = self._require_connection()
        with self._lock:
            try:
                connection.execute(
                    """INSERT INTO index_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
					   ON CONFLICT (run_id) DO UPDATE SET
						 started_at = excluded.started_at, finished_at = excluded.finished_at,
						 status = excluded.status, document_count = excluded.document_count,
						 chunk_count = excluded.chunk_count, failure_count = excluded.failure_count,
						 safe_error_summary = excluded.safe_error_summary""",
                    [
                        run.run_id,
                        run.started_at.isoformat(),
                        _datetime_string(run.finished_at),
                        run.status,
                        run.document_count,
                        run.chunk_count,
                        run.failure_count,
                        run.safe_error_summary,
                    ],
                )
            except Exception as error:
                raise DuckDBError("Index run summary could not be stored.") from error

    def get_index_run(self, run_id: str) -> IndexRunRecord | None:
        connection = self._require_connection()
        with self._lock:
            row = connection.execute(
                """SELECT run_id, status, document_count, chunk_count, failure_count,
						  started_at, finished_at, safe_error_summary
				   FROM index_runs WHERE run_id = ?""",
                [run_id],
            ).fetchone()
        return (
            IndexRunRecord(
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                _parse_datetime(row[5]),
                _parse_datetime(row[6]),
                row[7],
            )
            if row
            else None
        )

    def prepare_backup(self) -> Path:
        """Checkpoint and close a file database before an external copy.

        The caller remains responsible for making the backup using the selected
        DuckDB version's documented procedure. In-memory databases cannot be
        backed up as a database file.
        """
        if self.database_path == ":memory:":
            raise DuckDBError("An in-memory database cannot be backed up as a file.")
        connection = self._require_connection()
        with self._lock:
            try:
                connection.execute("CHECKPOINT")
                connection.close()
            except Exception as error:
                raise DuckDBError("DuckDB could not be prepared for backup.") from error
            self._connection = None
            return Path(self.database_path)

    def remove_document(self, document_id: str) -> bool:
        """Remove a document and all derived data in one transaction."""
        connection = self._require_connection()
        with self._lock:
            try:
                connection.execute("BEGIN TRANSACTION")
                if (
                    connection.execute(
                        "SELECT 1 FROM documents WHERE document_id = ?", [document_id]
                    ).fetchone()
                    is None
                ):
                    connection.execute("COMMIT")
                    return False
                connection.execute(
                    "DELETE FROM embeddings WHERE chunk_id IN "
                    "(SELECT chunk_id FROM chunks WHERE document_id = ?)",
                    [document_id],
                )
                for table in ("chunks", "hierarchy_nodes", "pages", "documents"):
                    connection.execute(
                        f"DELETE FROM {table} WHERE document_id = ?", [document_id]
                    )
                connection.execute("COMMIT")
                return True
            except Exception as error:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                if isinstance(error, DuckDBError):
                    raise
                raise DuckDBError(
                    "Document removal failed and was rolled back."
                ) from error

    def counts(self) -> dict[str, int]:
        connection = self._require_connection()
        with self._lock:
            documents = connection.execute("SELECT count(*) FROM documents").fetchone()[
                0
            ]
            chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            failures = connection.execute(
                "SELECT count(*) FROM documents WHERE status IN ('unreadable', 'failed', 'encrypted')"
            ).fetchone()[0]
        return {
            "documents": int(documents),
            "chunks": int(chunks),
            "failures": int(failures),
        }

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> DuckDBStore:
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _require_connection(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            raise DuckDBError("DuckDB is not initialized; call initialize() first.")
        return self._connection

    @staticmethod
    def _upgrade_embedding_schema(connection: duckdb.DuckDBPyConnection) -> None:
        """Explicitly upgrade the pre-TASK-06 embedding table without a migration framework."""
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('embeddings')").fetchall()
        }
        required = {
            "provider_name", "model_name", "model_version", "content_hash",
            "dimension", "vector", "created_at",
        }
        if not required <= columns:
            try:
                connection.execute("BEGIN TRANSACTION")
                connection.execute(
                    """CREATE TABLE embeddings_task06 (
                    chunk_id VARCHAR NOT NULL,
                    provider_name VARCHAR NOT NULL,
                    model_name VARCHAR NOT NULL,
                    model_version VARCHAR NOT NULL,
                    content_hash VARCHAR NOT NULL,
                    dimension INTEGER NOT NULL CHECK (dimension > 0),
                    vector DOUBLE[] NOT NULL,
                    created_at VARCHAR NOT NULL,
                    PRIMARY KEY (chunk_id, provider_name, model_name, model_version),
                    CHECK (array_length(vector) = dimension)
                )"""
                )
                if {"provider_name", "content_hash"} <= columns:
                    connection.execute(
                        """INSERT INTO embeddings_task06
                       SELECT chunk_id, provider_name, model_name, model_version,
                          content_hash, dimension, vector, created_at FROM embeddings"""
                    )
                else:
                    connection.execute(
                        """INSERT INTO embeddings_task06
                       SELECT chunk_id, 'legacy', model_name, model_version, '',
                          dimension, vector, created_at FROM embeddings"""
                    )
                connection.execute("DROP TABLE embeddings")
                connection.execute("ALTER TABLE embeddings_task06 RENAME TO embeddings")
                connection.execute("COMMIT")
            except Exception as error:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise DuckDBError("The local embedding table could not be upgraded.") from error
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name, model_version)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_provider_model "
            "ON embeddings(provider_name, model_name, model_version)"
        )

    def _validate_index(
        self,
        document: PdfReadResult,
        hierarchy: HierarchyResult,
        chunks: Sequence[RetrievalChunk],
        embeddings: Sequence[EmbeddingRecord],
    ) -> None:
        if document.document_id != hierarchy.document_id:
            raise DuckDBError("Document and hierarchy IDs do not match.")
        node_by_id = {node.node_id: node for node in hierarchy.nodes}
        if len(node_by_id) != len(hierarchy.nodes):
            raise DuckDBError("Hierarchy node IDs must be unique within a document.")
        for node in hierarchy.nodes:
            if node.document_id != document.document_id:
                raise DuckDBError("Hierarchy node references another document.")
            if node.page_start < 1 or node.page_end < node.page_start:
                raise DuckDBError("Hierarchy node has an invalid page range.")
            if any(
                page not in {item.page_number for item in document.pages}
                for page in range(node.page_start, node.page_end + 1)
            ):
                raise DuckDBError("Hierarchy node refers to a missing extracted page.")
            if node.parent_id is not None and node.parent_id not in node_by_id:
                raise DuckDBError("Hierarchy node references a missing parent.")

        page_numbers = {page.page_number for page in document.pages}
        chunk_by_id: dict[str, RetrievalChunk] = {}
        for chunk in chunks:
            if chunk.document_id != document.document_id:
                raise DuckDBError("Chunk references another document.")
            if chunk.chunk_id in chunk_by_id:
                raise DuckDBError("Chunk IDs must be unique within a document.")
            chunk_by_id[chunk.chunk_id] = chunk
            if chunk.parent_id not in node_by_id:
                raise DuckDBError("Chunk references a missing hierarchy node.")
            if any(
                page not in page_numbers
                for page in range(chunk.page_start, chunk.page_end + 1)
            ):
                raise DuckDBError("Chunk refers to a page missing from extracted data.")

        embedding_keys: set[tuple[str, str, str, str]] = set()
        for embedding in embeddings:
            key = (
                embedding.chunk_id,
                embedding.provider_name,
                embedding.model_name,
                embedding.model_version,
            )
            if embedding.chunk_id not in chunk_by_id:
                raise DuckDBError("Embedding references a missing chunk.")
            if key in embedding_keys:
                raise DuckDBError("Duplicate chunk/provider/model embeddings are not allowed.")
            _validate_embedding_record(embedding)
            if embedding.content_hash != chunk_by_id[embedding.chunk_id].content_hash:
                raise DuckDBError("Embedding content hash does not match its chunk.")
            embedding_keys.add(key)

    @staticmethod
    def _upsert_document(
        connection: duckdb.DuckDBPyConnection, document: PdfReadResult
    ) -> None:
        connection.execute(
            """INSERT INTO documents (
				   document_id, source_filename, source_path, content_sha256, page_count,
				   status, metadata_json, warnings_json, indexed_at, parser_version
			   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			   ON CONFLICT (document_id) DO UPDATE SET
				   source_filename = excluded.source_filename,
				   source_path = excluded.source_path,
				   content_sha256 = excluded.content_sha256,
				   page_count = excluded.page_count,
				   status = excluded.status,
				   metadata_json = excluded.metadata_json,
				   warnings_json = excluded.warnings_json,
				   indexed_at = excluded.indexed_at,
				   parser_version = excluded.parser_version""",
            [
                document.document_id,
                document.source_filename,
                document.source_path,
                document.content_sha256,
                document.page_count,
                document.status.value,
                _json(dict(document.metadata)),
                _json(list(document.warnings)),
                datetime.now(timezone.utc).isoformat(),
                "pypdf",
            ],
        )

    @staticmethod
    def _delete_derived(
        connection: duckdb.DuckDBPyConnection, document_id: str
    ) -> None:
        connection.execute(
            "DELETE FROM embeddings WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE document_id = ?)",
            [document_id],
        )
        for table in ("chunks", "hierarchy_nodes", "pages"):
            connection.execute(
                f"DELETE FROM {table} WHERE document_id = ?", [document_id]
            )

    @staticmethod
    def _insert_hierarchy(
        connection: duckdb.DuckDBPyConnection, nodes: Sequence[HierarchyNode]
    ) -> None:
        pending = list(nodes)
        inserted: set[str] = set()
        while pending:
            ready = [
                node
                for node in pending
                if node.parent_id is None or node.parent_id in inserted
            ]
            if not ready:
                raise DuckDBError("Hierarchy has a missing parent or a parent cycle.")
            for node in sorted(ready, key=lambda item: item.sequence):
                connection.execute(
                    "INSERT INTO hierarchy_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        node.node_id,
                        node.document_id,
                        node.parent_id,
                        node.level,
                        node.section_title,
                        _json(list(node.section_path)),
                        node.page_start,
                        node.page_end,
                        node.sequence,
                        node.detection_method,
                        node.confidence,
                        node.text,
                    ],
                )
                inserted.add(node.node_id)
                pending.remove(node)

    def _insert_pages(
        self, connection: duckdb.DuckDBPyConnection, document: PdfReadResult
    ) -> None:
        connection.executemany(
            "INSERT INTO pages VALUES (?, ?, ?, ?)",
            [
                (
                    document.document_id,
                    page.page_number,
                    page.text,
                    _json(list(page.warnings)),
                )
                for page in document.pages
            ],
        )

    @staticmethod
    def _insert_chunks(
        connection: duckdb.DuckDBPyConnection, chunks: Sequence[RetrievalChunk]
    ) -> None:
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.parent_id,
                    chunk.level,
                    chunk.section_title,
                    _json(list(chunk.section_path)),
                    chunk.previous_chunk_id,
                    chunk.next_chunk_id,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.sequence,
                    chunk.text,
                    chunk.chunk_size,
                    chunk.size_unit,
                    chunk.token_count,
                    chunk.content_hash,
                    chunk.configuration_fingerprint,
                    chunk.chunker_version,
                )
                for chunk in chunks
            ],
        )

    @staticmethod
    def _insert_embeddings(
        connection: duckdb.DuckDBPyConnection, embeddings: Sequence[EmbeddingRecord]
    ) -> None:
        connection.executemany(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    item.chunk_id,
                    item.provider_name,
                    item.model_name,
                    item.model_version,
                    item.content_hash,
                    len(item.vector),
                    list(item.vector),
                    _datetime_string(item.created_at or datetime.now(timezone.utc)),
                )
                for item in embeddings
            ],
        )


def _validate_run(run: IndexRunRecord) -> None:
    if not run.run_id.strip() or not run.status.strip():
        raise ValueError("run_id and status must not be empty")
    if min(run.document_count, run.chunk_count, run.failure_count) < 0:
        raise ValueError("index run counts must be non-negative")
    if run.started_at.tzinfo is None or run.started_at.utcoffset() is None:
        raise ValueError("started_at must be timezone-aware")
    if run.finished_at is not None:
        if run.finished_at.tzinfo is None or run.finished_at.utcoffset() is None:
            raise ValueError("finished_at must be timezone-aware")
        if run.finished_at < run.started_at:
            raise ValueError("finished_at cannot precede started_at")


def _validate_embedding_record(embedding: EmbeddingRecord) -> None:
    if not embedding.chunk_id.strip():
        raise DuckDBError("Embedding chunk ID is required.")
    if not embedding.provider_name.strip():
        raise DuckDBError("Embedding provider name is required.")
    if not embedding.model_name.strip() or not embedding.model_version.strip():
        raise DuckDBError("Embedding model name and version are required.")
    if not embedding.content_hash.strip():
        raise DuckDBError("Embedding content hash is required.")
    if not embedding.vector or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        for value in embedding.vector
    ):
        raise DuckDBError("Embedding vectors must be non-empty, numeric, and finite.")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _datetime_string(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _decode_json(value: str, expected_type: type) -> Any:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise DuckDBError("Stored index metadata is not valid JSON.") from error
    if not isinstance(decoded, expected_type):
        raise DuckDBError("Stored index metadata has an unexpected data shape.")
    return decoded


def _document_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "document_id": row[0],
        "source_filename": row[1],
        "source_path": row[2],
        "content_sha256": row[3],
        "page_count": row[4],
        "status": row[5],
        "metadata": _decode_json(row[6], dict),
        "warnings": tuple(_decode_json(row[7], list)),
        "indexed_at": row[8],
        "parser_version": row[9],
    }


def _chunk_from_row(row: tuple[Any, ...]) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=row[0],
        document_id=row[1],
        parent_id=row[2],
        level=row[3],
        section_title=row[4],
        section_path=tuple(_decode_json(row[5], list)),
        previous_chunk_id=row[6],
        next_chunk_id=row[7],
        page_start=row[8],
        page_end=row[9],
        sequence=row[10],
        text=row[11],
        chunk_size=row[12],
        size_unit=row[13],
        token_count=row[14],
        content_hash=row[15],
        configuration_fingerprint=row[16],
        chunker_version=row[17],
    )


def _stored_chunk_from_row(row: tuple[Any, ...]) -> StoredChunk:
    return StoredChunk(
        chunk=_chunk_from_row(row[:18]),
        source_filename=row[18],
        source_path=row[19],
    )
