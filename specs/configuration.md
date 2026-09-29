# Local Configuration and Setup

This guide configures the single-user local retrieval application described in `requirements.md`. The defaults use `data/pdfs/` as the manually ingested corpus, a DuckDB file under `data/duckdb/`, generated files under `data/artifacts/`, and the local Sentence Transformers model `sentence-transformers/all-MiniLM-L6-v2`. Keyword retrieval remains available if embeddings are not installed or configured.

## Supported local baseline

- Python 3.10
- Windows PowerShell
- Streamlit bound to `127.0.0.1`
- One local DuckDB database file
- PDFs stored under the configured corpus directory

Do not change the host to a LAN or public address. Remote access and multi-user authentication are out of scope.

## Create the Python environment

Run from the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the runtime packages, including Sentence Transformers, with `python -m pip install -r requirements.txt`. The model weights are downloaded from the model repository the first time the provider is initialized, then reused from the local cache. The application does not install packages or download extensions at startup.

If PowerShell activation is unavailable, call `.\.venv\Scripts\python.exe` directly. A machine-wide execution-policy change is not required. Use `deactivate` to leave an activated environment.

## Settings

Settings use `PDF_RAG_` environment variables. Values not supplied use these defaults:

| Variable | Default | Purpose |
|---|---|---|
| `PDF_RAG_CORPUS_DIR` | `data/pdfs` | Local directory containing the PostgreSQL and Apache Iceberg PDFs for manual ingestion. |
| `PDF_RAG_DATABASE_PATH` | `data/duckdb/retrieval.duckdb` | DuckDB database file; created under the project root when opened. |
| `PDF_RAG_ARTIFACTS_DIR` | `data/artifacts` | Local generated/cache artifacts. |
| `PDF_RAG_HOST` | `127.0.0.1` | Local-only Streamlit host. Only loopback values are accepted. |
| `PDF_RAG_PORT` | `8501` | Streamlit port. |
| `PDF_RAG_CHUNK_SIZE` | `1000` | Chunk-size limit. |
| `PDF_RAG_CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks; must be less than chunk size. |
| `PDF_RAG_CHUNK_SIZE_UNIT` | `characters` | Supported values: `characters` or `tokens`. |
| `PDF_RAG_SEMANTIC_TOP_K` | `20` | Semantic candidate count. |
| `PDF_RAG_KEYWORD_TOP_K` | `20` | Keyword candidate count. |
| `PDF_RAG_HYBRID_TOP_K` | `10` | Fused candidate count. |
| `PDF_RAG_HYBRID_FUSION_METHOD` | `rrf` | Fusion method: `rrf` or `weighted_sum`. |
| `PDF_RAG_RRF_K` | `60` | Positive RRF rank constant. |
| `PDF_RAG_HYBRID_SEMANTIC_WEIGHT` | `1.0` | Non-negative semantic contribution to fusion. |
| `PDF_RAG_HYBRID_KEYWORD_WEIGHT` | `1.0` | Non-negative keyword contribution to fusion. |
| `PDF_RAG_RERANKER_TOP_K` | `5` | Final reranked candidate count. |
| `PDF_RAG_CONTEXT_MAX_CHUNKS` | `2` | Maximum supporting context chunks attached to each result. Set to `0` to disable expansion. |
| `PDF_RAG_CONTEXT_MAX_CHARACTERS` | `4000` | Maximum total text characters in supporting chunks attached to each result. |
| `PDF_RAG_EMBEDDING_PROVIDER` | `sentence-transformers` | Embedding adapter identifier. The supported adapter runs locally. |
| `PDF_RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers model identifier; configure together with provider. |
| `PDF_RAG_EMBEDDING_MODEL_VERSION` | unset | Optional Sentence Transformers model revision to pin. If omitted, the resolved model commit is recorded when available. |
| `PDF_RAG_RERANKER_PROVIDER` | `sentence-transformers-cross-encoder` | Reranker adapter. |
| `PDF_RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder model identifier. |
| `PDF_RAG_RERANKER_MODEL_VERSION` | unset | Optional reranker model revision/version metadata. |

Relative paths resolve from the repository root; absolute paths may be supplied locally but shall not be committed to source control. The configuration loader does not read or retain API-key values. Any selected hosted adapter must obtain credentials directly from a protected environment variable and must never log or display them.

## Start the application

The design specifies a root-level `app.py` as the Streamlit entry point. Once that entry point is implemented, start the local app from the repository root with:

```powershell
python -m streamlit run app.py
```

Streamlit prints the local URL (normally `http://localhost:8501`). Stop the process with Ctrl+C. The app only searches an existing DuckDB index and displays its status. PDF discovery, extraction, chunking, and embedding are a separate manual operation. Put PDFs under `data/pdfs/` (or configure another corpus path), then run `python -m ingestion.main` from the repository root to populate DuckDB; the retrieval app does not scan PDFs or start ingestion.

## Verify settings and tests

Run focused configuration tests and then the complete test suite from the activated environment:

```powershell
python -m pytest tests/unit/test_config.py
python -m pytest
```

Tests should use temporary directories and injected environment dictionaries; they must not depend on the real corpus, user-specific absolute paths, or API credentials.

## Run retrieval evaluation

Create a JSON array (or `.jsonl` file with one object per line) of manually curated cases. Each case needs a unique `case_id`, a `query`, and one or both of `relevant_chunk_ids` and `relevant_document_ids`. For example:

```json
[
	{
		"case_id": "postgres-mvcc-001",
		"query": "How does MVCC preserve row visibility?",
		"relevant_chunk_ids": ["known-chunk-id"],
		"relevant_document_ids": ["known-document-id"]
	}
]
```

The independent runner accepts a Python retrieval callable using `module:function` syntax. Run it from the repository root with `python -m evaluation.runner path/to/cases.json --retriever your_module:retrieve --output data/artifacts/evaluation.json`. The callable receives each query and returns ranked chunk IDs or result objects with `chunk_id` (and `document_id` when evaluating document IDs). Equal scores follow the retriever's supplied deterministic order. This command does not launch Streamlit.
