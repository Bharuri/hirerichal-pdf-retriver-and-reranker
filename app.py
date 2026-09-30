"""Local Streamlit interface for searching a manually maintained DuckDB index."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal


from config import Settings
from db.duckdb import DuckDBError, DuckDBStore
from ingestion.indexer import EmbeddingProvider, create_embedding_provider
from retrieval.common import RetrievalError, RetrievalHit
from retrieval.context import expand_context
from retrieval.hybrid import hybrid_search
from retrieval.keyword import keyword_search
from retrieval.reranker import (
	RerankerProvider,
	create_reranker,
	rerank_candidates,
)
from retrieval.semantic import semantic_search


logger = logging.getLogger(__name__)
SearchMode = Literal["keyword", "semantic", "hybrid"]


class SearchApplicationError(RuntimeError):
	"""A safe, user-facing search or local-index error."""


@dataclass(frozen=True)
class IndexSnapshot:
	"""Basic index state shown by the UI; this module never indexes PDFs."""

	database_exists: bool
	documents: int = 0
	chunks: int = 0
	failures: int = 0
	document_rows: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class TablePreview:
	"""Column names and rows from a bounded, read-only DuckDB table query."""

	table_name: str
	columns: tuple[str, ...]
	rows: tuple[tuple[object, ...], ...]


def get_index_snapshot(settings: Settings) -> IndexSnapshot:
	"""Read current DuckDB status without creating a missing database file."""
	if not settings.database_path.is_file():
		return IndexSnapshot(database_exists=False)
	try:
		with DuckDBStore(settings.database_path) as store:
			counts = store.counts()
			return IndexSnapshot(
				database_exists=True,
				documents=counts["documents"],
				chunks=counts["chunks"],
				failures=counts["failures"],
				document_rows=tuple(store.list_documents()),
			)
	except DuckDBError as error:
		logger.exception("Could not read local DuckDB index status")
		raise SearchApplicationError(
			"The local index could not be opened. Check the configured DuckDB file."
		) from error


def get_database_tables(settings: Settings) -> tuple[str, ...]:
	"""Return local DuckDB table names without creating a missing index."""
	if not settings.database_path.is_file():
		return ()
	try:
		with DuckDBStore(settings.database_path) as store:
			return store.list_table_names()
	except DuckDBError as error:
		logger.exception("Could not list local DuckDB tables")
		raise SearchApplicationError(
			"The local index tables could not be listed. Check the configured DuckDB file."
		) from error


def query_database_table(
	settings: Settings,
	table_name: str,
	*,
	limit: int = 100,
) -> TablePreview:
	"""Run a bounded, read-only SELECT preview against one existing DuckDB table."""
	if limit <= 0:
		raise ValueError("limit must be greater than zero")
	if not settings.database_path.is_file():
		raise SearchApplicationError(
			"No DuckDB index exists yet. Ingest PDFs manually before querying its tables."
		)
	try:
		with DuckDBStore(settings.database_path) as store:
			columns, rows = store.preview_table(table_name, limit=limit)
		return TablePreview(table_name, columns, rows)
	except DuckDBError as error:
		logger.exception("Could not query local DuckDB table %s", table_name)
		raise SearchApplicationError(
			"The selected table could not be queried from the local index."
		) from error


def search_index(
	settings: Settings,
	query: str,
	mode: SearchMode,
	top_k: int,
	*,
	embedding_provider: EmbeddingProvider | None = None,
	reranker: RerankerProvider | None = None,
	use_reranker: bool = False,
	expand_hierarchy: bool = False,
) -> tuple[tuple[RetrievalHit, ...], str | None, str | None]:
	"""Search the existing index and optionally rerank/expand results.

	This function only reads the configured DuckDB index. It never discovers or
	reads PDFs, creates chunks, or generates document embeddings.
	"""
	normalized_query = " ".join(query.split())
	if not normalized_query:
		raise SearchApplicationError("Enter a search query first.")
	if mode not in {"keyword", "semantic", "hybrid"}:
		raise ValueError("mode must be 'keyword', 'semantic', or 'hybrid'")
	if top_k <= 0:
		raise ValueError("top_k must be greater than zero")
	if not settings.database_path.is_file():
		raise SearchApplicationError(
			"No DuckDB index exists yet. Ingest PDFs manually, then search the created index here."
		)

	note: str | None = None
	reranker_status: str | None = None
	try:
		with DuckDBStore(settings.database_path) as store:
			counts = store.counts()
			if counts["documents"] == 0 or counts["chunks"] == 0:
				raise SearchApplicationError(
					"The DuckDB index has no searchable chunks. Ingest PDFs manually before searching."
				)

			logger.info("Starting %s retrieval", mode)
			if mode == "keyword":
				hits = keyword_search(
					store, normalized_query, top_k=top_k, settings=settings
				)
			elif mode == "semantic":
				provider = embedding_provider or create_embedding_provider(settings)
				hits = semantic_search(
					store, normalized_query, provider, top_k=top_k, settings=settings
				)
				if not hits:
					note = "No compatible embeddings or semantic matches were found in the manual index."
			else:
				provider = embedding_provider
				if provider is None and settings.embedding_provider is not None:
					try:
						provider = create_embedding_provider(settings)
					except Exception as error:
						logger.warning(
							"Semantic channel unavailable; continuing hybrid search with keywords (%s)",
							type(error).__name__,
						)
						note = "Semantic embeddings are unavailable; showing keyword-only results."
				try:
					hits = hybrid_search(
						store,
						normalized_query,
						provider,
						top_k=top_k,
						settings=settings,
					)
				except RetrievalError as error:
					if provider is None:
						raise
					logger.warning(
						"Semantic channel failed; continuing hybrid search with keywords (%s)",
						type(error).__name__,
					)
					hits = hybrid_search(
						store, normalized_query, None, top_k=top_k, settings=settings
					)
					note = "Semantic retrieval failed; showing keyword-only results."

			if mode == "hybrid" and use_reranker and hits:
				selected_reranker = reranker
				if selected_reranker is None:
					try:
						selected_reranker = create_reranker(settings)
					except Exception as error:
						logger.warning(
							"Configured reranker unavailable (%s)", type(error).__name__
						)
						note = "The configured reranker is unavailable; fused ranking was retained."
				rerank_result = rerank_candidates(
					normalized_query,
					hits,
					selected_reranker,
					top_k=top_k,
					settings=settings,
				)
				hits = rerank_result.candidates
				reranker_status = rerank_result.status
				if rerank_result.reason:
					note = rerank_result.reason

			if expand_hierarchy and hits:
				hits = expand_context(hits, store, settings=settings)

			logger.info("Completed %s retrieval with %d ranked results", mode, len(hits))
			return hits, note, reranker_status
	except SearchApplicationError:
		raise
	except (DuckDBError, RetrievalError, OSError, RuntimeError) as error:
		logger.exception("Local retrieval failed")
		raise SearchApplicationError(
			"Search could not be completed. Check the local index and configured model."
		) from error


def main(streamlit_module: object | None = None, settings: Settings | None = None) -> None:
	"""Render search, index status, and read-only DuckDB table browsing."""
	if streamlit_module is None:
		import streamlit as streamlit_module

	st = streamlit_module
	active_settings = settings or Settings.from_environment()
	st.set_page_config(page_title="Local PDF Retrieval", page_icon="📚", layout="wide")
	st.title("Local PDF Retrieval")
	st.caption("Search ranked evidence from your local technical-document index.")
	st.info(
		"PDF ingestion is manual and separate from retrieval. This app never scans or indexes PDFs; "
		"it searches the configured DuckDB index only."
	)

	snapshot: IndexSnapshot | None = None
	snapshot_error: str | None = None
	try:
		snapshot = get_index_snapshot(active_settings)
	except SearchApplicationError as error:
		snapshot_error = str(error)

	search_tab, status_tab, database_tab = st.tabs(
		("Search", "Index status", "Database")
	)
	with status_tab:
		st.subheader("Manually maintained index")
		st.caption(f"DuckDB: {active_settings.database_path}")
		if snapshot_error:
			st.error(snapshot_error)
		elif snapshot is None or not snapshot.database_exists:
			st.warning(
				"No DuckDB index file was found. Ingest documents manually before searching."
			)
		else:
			metric_columns = st.columns(3)
			metric_columns[0].metric("Documents", snapshot.documents)
			metric_columns[1].metric("Chunks", snapshot.chunks)
			metric_columns[2].metric("Failed documents", snapshot.failures)
			if snapshot.documents == 0 or snapshot.chunks == 0:
				st.warning(
					"The index has no searchable chunks. Ingestion is a separate manual step."
				)
			for document in snapshot.document_rows:
				with st.expander(str(document["source_filename"])):
					st.text(f"Status: {document['status']}")
					st.text(f"Source: {document['source_path']}")
					st.text(f"Pages: {document['page_count']}")
					warnings = document["warnings"]
					st.text(f"Warnings: {', '.join(warnings) or 'None'}")

	with database_tab:
		st.subheader("DuckDB tables")
		st.caption("Browse read-only table previews from the configured local DuckDB index.")
		if snapshot_error:
			st.error(snapshot_error)
		elif snapshot is None or not snapshot.database_exists:
			st.warning("No DuckDB index file was found. Ingest documents manually first.")
		else:
			try:
				table_names = get_database_tables(active_settings)
			except SearchApplicationError as error:
				st.error(str(error))
			else:
				if not table_names:
					st.info("The DuckDB index contains no tables.")
				else:
					st.write("Tables:", ", ".join(table_names))
					with st.form("database_query_form"):
						selected_table = st.selectbox(
							"Table to query", table_names, key="database_table"
						)
						row_limit = st.number_input(
							"Maximum rows",
							min_value=1,
							max_value=1000,
							value=100,
							step=10,
							key="database_row_limit",
						)
						query_submitted = st.form_submit_button(
							"Query table", key="database_query_button"
						)
					if query_submitted:
						try:
							preview = query_database_table(
								active_settings,
								selected_table,
								limit=int(row_limit),
							)
						except SearchApplicationError as error:
							st.error(str(error))
						else:
							st.caption(
								f"Showing up to {int(row_limit)} rows from {preview.table_name}."
							)
							if preview.rows:
								st.dataframe(
									[
										{
											column: value
											for column, value in zip(
												preview.columns, row, strict=True
											)
										}
										for row in preview.rows
									],
									use_container_width=True,
								)
							else:
								st.info("The selected table is empty.")

	with search_tab:
		st.subheader("Search evidence")
		with st.form("retrieval_form"):
			query = st.text_input("Search query", key="retrieval_query")
			mode_label = st.selectbox(
				"Retrieval method",
				("Hybrid", "Keyword", "Semantic"),
				index=0,
				key="retrieval_method",
			)
			top_k = st.number_input(
				"Results to show",
				min_value=1,
				max_value=100,
				value=max(1, min(100, active_settings.hybrid_top_k)),
				step=1,
				key="retrieval_top_k",
			)
			reranker_available = (
				active_settings.reranker_provider is not None
				and active_settings.reranker_model is not None
			)
			use_reranker = st.checkbox(
				"Apply configured reranker (hybrid only)",
				value=True,
				disabled=not reranker_available,
				key="use_reranker",
			)
			if not reranker_available:
				st.caption("Reranking is optional and is not configured.")
			expand_hierarchy = st.checkbox(
				"Expand hierarchy context", value=True, key="expand_hierarchy"
			)
			show_scores = st.checkbox(
				"Show retrieval score breakdown",
				value=False,
				key="show_score_breakdown",
			)
			submitted = st.form_submit_button("Search", key="search_button")

		if submitted:
			if snapshot_error:
				st.error(snapshot_error)
			elif snapshot is None or not snapshot.database_exists:
				st.error(
					"No DuckDB index exists. Ingest PDFs manually, then run the search again."
				)
			else:
				mode: SearchMode = mode_label.lower()  # type: ignore[assignment]
				try:
					hits, note, reranker_status = search_index(
						active_settings,
						query,
						mode,
						int(top_k),
						use_reranker=bool(use_reranker),
						expand_hierarchy=bool(expand_hierarchy),
					)
				except SearchApplicationError as error:
					st.error(str(error))
				else:
					if note:
						st.warning(note)
					if reranker_status == "skipped":
						st.caption("Reranker skipped; fused candidate order was retained.")
					elif reranker_status == "failed":
						st.warning("Reranking failed; original fused candidate order was retained.")
					if not hits:
						st.info("No matching evidence was found.")
					for hit in hits:
						title = (
							f"{hit.rank}. {hit.source_filename} — "
							f"pages {hit.page_start}–{hit.page_end}"
						)
						with st.expander(title, expanded=True):
							st.caption(
								f"Section: {' > '.join(hit.section_path) or 'Unspecified'}"
							)
							st.caption(
								f"Chunk ID: {hit.chunk_id} · Document ID: {hit.document_id}"
							)
							st.text(hit.text)
							if show_scores:
								st.caption(
									"Scores — "
									f"final: {hit.score:.6g}; "
									f"semantic: {_format_score(hit.semantic_score)}; "
									f"keyword: {_format_score(hit.keyword_score)}; "
									f"hybrid: {_format_score(hit.hybrid_score)}; "
									f"reranker: {_format_score(hit.reranker_score)}"
								)
							for context in hit.context_chunks:
								with st.container(border=True):
									st.caption(
										f"Related context ({context.relation}) · "
										f"{context.chunk.chunk_id} · pages "
										f"{context.chunk.page_start}–{context.chunk.page_end}"
									)
									st.caption(
										"Section: "
										f"{' > '.join(context.chunk.section_path) or 'Unspecified'}"
									)
									st.text(context.chunk.text)


def _format_score(score: float | None) -> str:
	return "—" if score is None else f"{score:.6g}"

if __name__ == "__main__":
	main()
