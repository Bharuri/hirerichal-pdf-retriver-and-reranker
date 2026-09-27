1. Purpose

Build a focused retrieval system for technical documentation PDFs, initially PostgreSQL and Apache Iceberg. The system extracts document structure, creates hierarchical chunks, stores them in DuckDB, retrieves relevant evidence using semantic and keyword search, combines the results, reranks candidates, and expands useful hierarchical context. The project ends at retrieval and evidence presentation; it does not generate answers.

2. Scope

Input: local technical-documentation PDF files.

Initial corpus: PostgreSQL and Apache Iceberg documentation PDFs.

Storage: DuckDB only; no external vector database.

Embeddings: configurable embedding model/provider.

Retrieval: semantic, keyword, hybrid fusion, and reranking.

Context: hierarchical parent/child and neighboring-chunk expansion.

UI: Streamlit retrieval interface showing ranked evidence and metadata.

Evaluation: retrieval metrics only.

Development style: Specification-Driven Development with traceable requirements, design, tasks, and tests.

3. Explicit Non-Goals

No LLM or generative model.

No prompt engineering.

No answer generation or summarization.

No chat-style generated responses.

No generation-quality evaluation.

No SQL execution against PostgreSQL or Iceberg.

No web browsing during retrieval.

No model fine-tuning.

No external vector database.

No multi-user production authentication/authorization.

No production TLS, distributed deployment, SLOs, disaster recovery, or enterprise alerting.

No mandatory OCR in the first version.

No graph visualization; hierarchy is represented as metadata and used for retrieval/context.

4. Functional Requirements

4.1 Configuration

REQ-001 — The system shall load configurable paths for the PDF corpus, DuckDB database, and generated artifacts.

REQ-002 — The system shall allow the embedding model, retrieval top-k values, reranker, and chunking parameters to be configured without changing application logic.

4.2 PDF Discovery

REQ-003 — The system shall discover supported PDF files from the configured corpus directory.

REQ-004 — The system shall assign a stable document identifier based on document identity/content metadata and record the source filename and path.

4.3 PDF Extraction

REQ-005 — The system shall extract text page by page while preserving page numbers.

REQ-006 — The system shall preserve enough source metadata to map every final chunk back to its document and source pages.

REQ-007 — The system shall identify pages with little or no extractable text and report them as extraction warnings.

REQ-008 — The extraction process shall tolerate an unreadable PDF without preventing other valid PDFs from being processed.

4.4 Hierarchy Detection

REQ-009 — The system shall detect document headings and construct a hierarchy of document, section, subsection, and content.

REQ-010 — The hierarchy detector shall use PDF outline information when available.

REQ-011 — When outlines are unavailable, the system shall use heading patterns, numbering, typography metadata when available, and page/paragraph structure as fallback signals.

REQ-012 — The system shall record hierarchy metadata such as section path and hierarchy level for each chunk.

4.5 Hierarchical Chunking

REQ-013 — The system shall create final retrieval chunks from the extracted hierarchical content.

REQ-014 — Chunk size and overlap shall be configurable.

REQ-015 — Chunks shall not intentionally cross unrelated section boundaries when a suitable boundary is available.

REQ-016 — Each chunk shall have a stable chunk identifier and retain document, section, page, and parent/neighbor metadata.

REQ-017 — The system shall embed each final retrieval chunk once by default; hierarchy shall primarily be used for metadata and context expansion rather than repeated multi-level embedding.

4.6 DuckDB Storage

REQ-018 — The system shall store document metadata in DuckDB.

REQ-019 — The system shall store chunk text, hierarchy metadata, page metadata, and embedding data in DuckDB.

REQ-020 — The stored representation shall support semantic retrieval and keyword retrieval without requiring an external database.

REQ-021 — The system shall prevent duplicate chunk records when the same document is indexed repeatedly.

REQ-022 — The schema shall include document/content and embedding-model metadata needed to determine whether an index is reusable.

4.7 Embeddings and Indexing

REQ-023 — The system shall generate embeddings for final retrieval chunks using the configured embedding model.

REQ-024 — The system shall store the embedding model identifier with generated embeddings.

REQ-025 — The indexing pipeline shall process the configured PDF corpus into searchable DuckDB records.

REQ-026 — The indexing pipeline shall report document, chunk, and failure counts after an indexing run.

4.8 Retrieval

REQ-027 — The system shall normalize a user query before retrieval.

REQ-028 — The system shall support semantic retrieval using embedding similarity.

REQ-029 — The system shall support keyword-based retrieval over chunk text.

REQ-030 — The system shall combine semantic and keyword results using a configurable hybrid-ranking method.

REQ-031 — The hybrid retrieval stage shall retain source identifiers and ranking information needed for debugging.

REQ-032 — The system shall support configurable retrieval top-k values.

REQ-033 — The system shall return source metadata including document, page, section path, and chunk identifier.

4.9 Reranking and Context

REQ-034 — The system shall support a reranker after initial hybrid retrieval.

REQ-035 — The reranker shall produce a relevance-ranked candidate list.

REQ-036 — The system shall expand selected chunks with useful parent/child or neighboring context from the same document.

REQ-037 — Context expansion shall preserve source metadata so expanded evidence remains traceable.

REQ-038 — The system shall return the final ranked evidence set to the caller without generating an answer.

4.10 Streamlit UI

REQ-039 — The system shall provide a Streamlit interface for entering a search query.

REQ-040 — The UI shall display ranked retrieval results with relevance scores.

REQ-041 — The UI shall display document name, page range, section path, and chunk text for retrieved evidence.

REQ-042 — The UI shall provide an optional view showing semantic, keyword, hybrid, and reranker scores.

REQ-043 — The UI shall show basic corpus/index information such as document count and indexed chunk count.

REQ-044 — The UI shall report user-friendly errors for missing corpus or unavailable index.

4.11 Retrieval Evaluation

REQ-045 — The system shall support a manually curated retrieval evaluation dataset containing queries and expected relevant chunk/document identifiers.

REQ-046 — Retrieval evaluation shall calculate Recall@5, Recall@10, and Recall@20.

REQ-047 — Retrieval evaluation shall calculate MRR.

REQ-048 — Retrieval evaluation shall calculate NDCG@5, NDCG@10, and NDCG@20.

REQ-049 — Evaluation results shall identify the embedding model, reranker, and retrieval configuration used for the run.

REQ-050 — Retrieval evaluation shall be runnable independently from the Streamlit UI.

4.12 Testing and Basic Reliability

REQ-051 — Core extraction, hierarchy, chunking, storage, retrieval, reranking, and context components shall have focused automated tests.

REQ-052 — The project shall include integration tests covering PDF-to-DuckDB indexing.

REQ-053 — The project shall include an end-to-end test covering query-to-ranked-evidence flow.

REQ-054 — The system shall log major indexing and retrieval steps at an appropriate level for local debugging.

REQ-055 — Recoverable document-level failures shall be recorded without silently hiding the failure.

REQ-056 — The application shall fail with a clear error when required configuration or index data is missing.

4.13 Extensibility

REQ-057 — The embedding implementation shall be replaceable through a small provider interface.

REQ-058 — The reranker implementation shall be replaceable through a small provider interface.

5. Acceptance Baseline

Both PostgreSQL and Iceberg PDFs can be indexed.

Hierarchy is retained in chunk metadata.

A query retrieves relevant chunks through semantic + keyword hybrid retrieval.

Reranking produces a final relevance-ordered evidence set.

Hierarchy-aware context expansion provides traceable supporting context.

Streamlit displays ranked evidence and source metadata.

Retrieval evaluation produces Recall, MRR, and NDCG results.

Automated unit, integration, and end-to-end retrieval tests pass.

The project remains a local/single-user retrieval application rather than a production platform.

6. Change Control

New capabilities should first be added as a requirement, then reflected in design, tasks, implementation, and tests. Features unrelated to document retrieval should remain outside the MVP.