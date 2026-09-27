1. Design Goal

Build a simple, measurable retrieval system whose final output is a ranked and context-expanded set of documentation evidence. No generative answer layer is included.

2. High-Level Architecture

PDFs → extraction → hierarchy detection → hierarchical chunking → DuckDB index → query normalization → semantic + keyword retrieval → hybrid fusion → reranking → hierarchy/context expansion → ranked evidence → Streamlit

3. Project Structure

Use this small, module-oriented layout. Implement modules in task order; files shown for later tasks are planned placeholders until those tasks begin.

```text
PDF-Spliter and reranker/
├── app.py
├── config.py
├── data/
│   ├── pdfs/                 # optional local corpus location; configurable
│   ├── duckdb/               # local database files; generated/ignored
│   └── artifacts/            # generated local artifacts; generated/ignored
├── ingestion/
│   ├── pdf_loader.py         # discovery and page-aware PDF extraction
│   ├── hierarchy.py          # outline/heading hierarchy detection
│   ├── chunker.py            # hierarchical page-aware chunking
│   └── indexer.py            # indexing orchestration
├── retrieval/
│   ├── semantic.py
│   ├── keyword.py
│   ├── hybrid.py
│   ├── reranker.py
│   └── context.py
├── db/
│   ├── duckdb.py
│   └── schema.sql
├── evaluation/
│   ├── retrieval_metrics.py
│   └── runner.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── specs/
  ├── requirements.md
  ├── design.md
  ├── tasks.md
  └── configuration.md
```

The default source corpus is `data/pdfs/`; an existing source corpus may remain elsewhere, such as `doc/`, by configuring `PDF_RAG_CORPUS_DIR`. Keep local databases and generated artifacts out of source control.

4. Ingestion Design

4.1 PDF Discovery

Scan the configured directory and create a document record for each PDF.

4.2 Extraction

Use pypdf for page-level extraction while preserving page numbers and extraction warnings.

4.3 Hierarchy Detection

Prefer PDF outlines/bookmarks. If unavailable, detect numbered and heading-like patterns, then use page/paragraph structure as fallback.

4.4 Chunking

Create final retrieval chunks from the hierarchy. Prefer section and paragraph boundaries and apply configurable size/overlap.

4.5 Stable IDs

Derive stable document and chunk identifiers from document identity/content and stable structural metadata.

5. Hierarchy Model

Hierarchy is represented as metadata rather than a graph.

document_id

level

section_title

section_path

parent_chunk_id

page_start / page_end

chunk_sequence

neighbor identifiers

Example: PostgreSQL → SQL Language → SELECT → JOIN → chunk. The full path is stored with the chunk and can be used during retrieval and context expansion.

6. Hierarchical Embedding Strategy

The MVP uses one embedding per final retrieval chunk. The hierarchy does not require four separate embeddings.

Construct embedding input from section path plus chunk text.

Store the embedding model name/version.

Use chunk embeddings for semantic retrieval.

Use hierarchy metadata for filtering and context expansion.

Optional future experiment: embeddings for section summaries, but not part of the MVP.

7. DuckDB Data Model

documents
---------
document_id
name
source_path
content_hash
page_count
indexed_at

chunks
------
chunk_id
document_id
parent_chunk_id
level
section_title
section_path
page_start
page_end
chunk_sequence
text
embedding
embedding_model

evaluation_cases
----------------
case_id
query
relevant_chunk_ids
expected_document_ids

evaluation_runs
---------------
run_id
run_time
embedding_model
reranker
metrics_json

8. Retrieval Pipeline

1. Normalize the user query while preserving technical terms.

2. Semantic retrieval: embed the query and compare it with stored chunk vectors.

3. Keyword retrieval: match important query terms against chunk text, section title, and section path.

4. Hybrid fusion: combine semantic and keyword result lists using RRF or another simple configurable method.

5. Reranking: apply a cross-encoder or configured reranker to the top hybrid candidates.

6. Context expansion: add useful parent, child, or neighboring chunks from the same document.

7. Return a deduplicated, bounded, traceable evidence set.

9. Suggested Initial Parameters

Semantic candidates: top 20

Keyword candidates: top 20

Hybrid candidates: top 10

Reranked candidates: top 5

Context expansion: parent + nearby chunks where useful

All values remain configurable.

10. Keyword Retrieval

Start with simple DuckDB SQL-based matching. Score matches across chunk text, section title, and section path. DuckDB FTS can be evaluated later but is not required for the MVP.

11. Reranking

The reranker receives the query and hybrid candidates and returns relevance scores. Keep it behind a small interface so different local or hosted rerankers can be substituted.

12. Context Builder

Start from reranked chunks.

Add parent section information when the result lacks context.

Add adjacent chunks when content continues across boundaries.

Add children only when they are relevant to the selected evidence.

Deduplicate repeated text.

Preserve document, page, section, and chunk metadata.

13. Final Retrieval Result

The system returns structured evidence rather than an answer.

{
  query,
  chunk_id,
  document,
  page_start,
  page_end,
  section_path,
  retrieval_score,
  semantic_score,
  keyword_score,
  hybrid_score,
  reranker_score,
  context_chunks
}

14. Streamlit Design

Query input.

Top retrieved evidence list.

Document/page/section metadata.

Chunk text.

Optional score breakdown.

Optional hierarchy/context expansion view.

Index statistics.

PDF ingestion is a separate manual operation outside Streamlit. The UI may
display the existing DuckDB index status, but it does not scan the corpus or
start extraction, chunking, or embedding.

15. Retrieval Evaluation

Recall@5, @10, @20

MRR

NDCG@5, @10, @20

Record embedding/reranker/configuration for every evaluation run.

16. Error Handling

Missing configuration → clear startup error.

Missing PDF directory → clear indexing error.

Unreadable PDF → report document failure and continue.

Empty extraction → warning/document status.

Missing index → UI instructs the user to run indexing.

Provider failure → clear error without exposing credentials.

17. Testing Strategy

Unit: extraction metadata, hierarchy detection, chunk boundaries, IDs, retrieval scoring, fusion, reranking, context expansion.

Integration: sample PDFs → DuckDB → retrieval.

End-to-end: query → retrieval → reranking → context → displayed evidence.

Evaluation regression: maintain a fixed question set and compare retrieval metrics after changes.

18. Deliberately Excluded

LLM/generative answering

Prompt engineering

Answer summarization

Generation evaluation

Microservices

External vector stores

Kubernetes

Enterprise identity

Distributed locking

Production SLO/DR/alerting