Tasks — Local PDF Retrieval Application

This task list translates `requirements.md` and `design.md` into small implementation steps. Each task owns focused tests and can be run without a live embedding or reranking service unless explicitly noted. Use temporary PDFs and DuckDB files in tests, never the local production corpus.

The application ends at retrieval. Its output is a ranked, optionally reranked, hierarchy-expanded evidence set. No answer generation or summarization is performed by the application.

Working rules

Keep the MVP local and single-user. Do not add answer generation, remote deployment, accounts, a vector database, or other out-of-scope features.

Prefer the simple module layout in `design.md`; do not create additional architecture layers unless implementation demonstrates a need.

Complete a task when its focused tests pass and its acceptance checks are met. Run the full test suite and Ruff checks before integrating tasks.

Provider calls must be replaceable with deterministic fakes. Live-provider tests are optional and must not be required for the default test run.

Task order

TASK-01 — Confirm local configuration and setup

Work: Review existing configuration module and notes. Confirm Python 3.10 setup, configured PDF directory, DuckDB file path, generated-artifact directory, safe local-only defaults, documented chunk/retrieval settings. Avoid config framework if existing loader sufficient.

Independent test: injected temporary environment/directories; no DuckDB/provider.

Acceptance: valid local defaults; missing/invalid required settings clear error; secrets not printed; Windows venv and app start instructions documented.

Requirements: REQ-001–002, REQ-056.

TASK-02 — Read PDFs and preserve page metadata

Work: PDF discovery/page-by-page extraction. Capture stable document identity/content hash, source filename/path, PDF metadata, page count, per-page text, warnings/status. One unreadable PDF must not stop others.

Independent: synthetic PDFs/temp corpus; no DuckDB/provider.

Acceptance: page numbers 1-based; missing metadata harmless; empty/unreadable reported; files outside corpus not read; unchanged files same identity/hash.

Requirements: REQ-003–008.

TASK-03 — Detect document hierarchy

Work: PDF outlines first, conservative heading/numbering patterns, page/paragraph fallback. Keep order, parent relationship, section path, page span, detection method, heuristic/uncertain indication.

Independent: fixed extracted-page fixtures.

Acceptance: outlines, numbered headings, missing/ambiguous headings, fallback hierarchy, page spans, parent relationships, deterministic order; no inferred styling parser doesn't expose.

Requirements: REQ-009–012.

TASK-04 — Create deterministic hierarchical chunks

Work: context-aware, page-aware chunks using hierarchy paths and configured size/overlap. Prefer section/paragraph/sentence/line boundaries. Keep source text separate from hierarchy context metadata. Record stable ID, order, parent/neighbor references, document/page range.

Independent: synthetic pages/hierarchy; no PDF parsing/storage/embeddings.

Acceptance: section boundaries, overlap/size, long passages, code/SQL/list text, cross-page chunks, traceable pages, deterministic IDs, duplicate-free.

Requirements: REQ-013–017.

TASK-05 — Store and reuse the index in DuckDB

Work: initial DuckDB schema/storage functions. Store document metadata, extracted pages, hierarchy, chunks, vectors/model metadata when present. Initialize on startup. Parameterized SQL/transactions for replacing a document’s derived index. No generic migration framework.

Independent: temp DuckDB, synthetic records.

Acceptance: schema init/reopen, document/chunk round-trip, page/hierarchy metadata, duplicate prevention, vector dimension/model metadata, referential checks, rollback, changed-document replacement, removal of document derived records.

Requirements: REQ-018–026.

TASK-06 — Generate and persist chunk embeddings

Work: small replaceable embedding-provider interface; configured provider. Embed each final retrieval chunk once by default, bounded batches. Store provider/model/version, dimension, vector. Reuse only if chunk content and model identity match.

Independent: deterministic fake provider; live provider optional.

Acceptance: ordering, dimension/finite validation, batches, failure reporting/retry, model metadata, reuse rules. Failure must not mark chunk embedded.

Requirements: REQ-017, REQ-022–024, REQ-057.

TASK-07 — Implement keyword, semantic, and hybrid retrieval

Work: keyword over chunk text/hierarchy fields, semantic cosine retrieval over compatible DuckDB vectors, configurable fusion (RRF default). Preserve rank/score, dedup by chunk ID. Keyword available without embeddings.

Independent: DuckDB synthetic chunks/vectors, deterministic query embeddings.

Acceptance: exact technical terms, query normalization, empty index, model/dimension filtering, top-k, score ordering, fusion, tie handling, dedup, source/page metadata.

Requirements: REQ-027–033.

TASK-08 — Add optional reranking and hierarchy-aware context expansion

Work: replaceable reranker interface/optional adapter. Rerank hybrid candidates, add useful parent/child/neighbor chunks from same doc within bounds. Preserve IDs/scores/pages/section paths. If no reranker, return fused candidates and report skipped.

Independent: fake candidates/deterministic fake reranker/context repo.

Acceptance: reranking order/failure, disabled reranker, parent/neighbor expansion, duplicate removal, context bounds, same-document constraints, metadata traceability.

Requirements: REQ-034–038.

TASK-09 — Build the Streamlit retrieval application

Work: local Streamlit app. Ingestion is a separate manual operation and is never triggered by the retrieval app. Provide a read-only Index status view for the configured DuckDB index and a Search view for query and retrieval method/top-k selection; do not add ingestion controls. Show counts/status and ranked evidence with scores, PDF name, page range, section path, chunk text. No generated answers.

Independent: rendering/service/module with temp storage/fake providers; no live credentials.

Acceptance: local startup; manual index status/errors; an externally/manual populated index is searchable after restart; missing corpus/index/provider states clear; source references point actual chunks/pages; local content rendered as untrusted text; app startup/search does not scan or ingest PDFs. Run manual ingestion separately with `python -m ingestion.main`.

Requirements: REQ-039–044, REQ-054–056.

TASK-10 — Add retrieval metrics and evaluation runner

Work: dataset loading + independent command/function. Recall@5/@10/@20, MRR, NDCG@5/@10/@20. Record dataset, embedding/reranker identity, retrieval config.

Independent: hand-calculated examples/local eval dataset; no db/provider for metric unit tests.

Acceptance: cutoff, empty relevant sets, duplicate/tied results, per-query/aggregate, config capture, independent invocation.

Requirements: REQ-045–050.

TASK-11 — Complete integration and end-to-end regression tests

Work: PDF → extraction → hierarchy → chunking → DuckDB → retrieval → optional reranking/context and query → ranked evidence. Include doc-level failure recovery and Streamlit smoke test.

Independent: generated sample PDFs, temp db, fake providers. Default suite offline.

Acceptance: evidence maps expected original PDF page/section; bad PDF doesn’t block valid; repeat indexing no duplicates; unit/integration/E2E pass.

Requirements: REQ-051–056.

Completion check

Local MVP done when user can run app, index PostgreSQL/Iceberg PDFs, restart app, search using keyword/configured semantic/hybrid retrieval, optionally rerank and expand hierarchy context, and see ranked evidence with accurate doc/page/section references. Retrieval metrics and automated tests pass. No LLM-generated answer is part of product.

