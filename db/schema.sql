-- Initial schema for the local retrieval index. Schema upgrades are handled
-- explicitly in db/duckdb.py; this MVP does not use a generic migration system.
-- Cross-table references are validated by DuckDBStore before each atomic write.
-- Avoid derived-row foreign keys because they prevent reliable DuckDB delete-and-replace.

CREATE TABLE IF NOT EXISTS documents (
	document_id VARCHAR PRIMARY KEY,
	source_filename VARCHAR NOT NULL,
	source_path VARCHAR NOT NULL UNIQUE,
	content_sha256 VARCHAR,
	page_count INTEGER NOT NULL CHECK (page_count >= 0),
	status VARCHAR NOT NULL,
	metadata_json VARCHAR NOT NULL,
	warnings_json VARCHAR NOT NULL,
	indexed_at VARCHAR NOT NULL,
	parser_version VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
	document_id VARCHAR NOT NULL,
	page_number INTEGER NOT NULL CHECK (page_number >= 1),
	text VARCHAR NOT NULL,
	warnings_json VARCHAR NOT NULL,
	PRIMARY KEY (document_id, page_number)
);

CREATE TABLE IF NOT EXISTS hierarchy_nodes (
	node_id VARCHAR PRIMARY KEY,
	document_id VARCHAR NOT NULL,
	parent_id VARCHAR,
	level VARCHAR NOT NULL,
	section_title VARCHAR,
	section_path_json VARCHAR NOT NULL,
	page_start INTEGER NOT NULL CHECK (page_start >= 1),
	page_end INTEGER NOT NULL CHECK (page_end >= page_start),
	sequence INTEGER NOT NULL CHECK (sequence >= 0),
	detection_method VARCHAR NOT NULL,
	confidence DOUBLE,
	text VARCHAR,
	UNIQUE (document_id, sequence)
);

CREATE TABLE IF NOT EXISTS chunks (
	chunk_id VARCHAR PRIMARY KEY,
	document_id VARCHAR NOT NULL,
	parent_id VARCHAR NOT NULL,
	level VARCHAR NOT NULL,
	section_title VARCHAR,
	section_path_json VARCHAR NOT NULL,
	previous_chunk_id VARCHAR,
	next_chunk_id VARCHAR,
	page_start INTEGER NOT NULL CHECK (page_start >= 1),
	page_end INTEGER NOT NULL CHECK (page_end >= page_start),
	sequence INTEGER NOT NULL CHECK (sequence >= 0),
	text VARCHAR NOT NULL CHECK (length(trim(text)) > 0),
	chunk_size INTEGER NOT NULL CHECK (chunk_size > 0),
	size_unit VARCHAR NOT NULL CHECK (size_unit IN ('characters', 'tokens')),
	token_count INTEGER NOT NULL CHECK (token_count >= 0),
	content_hash VARCHAR NOT NULL,
	configuration_fingerprint VARCHAR NOT NULL,
	chunker_version VARCHAR NOT NULL,
	UNIQUE (document_id, sequence),
	UNIQUE (document_id, content_hash, configuration_fingerprint, sequence)
);

CREATE TABLE IF NOT EXISTS embeddings (
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
);

CREATE TABLE IF NOT EXISTS index_runs (
	run_id VARCHAR PRIMARY KEY,
	started_at VARCHAR NOT NULL,
	finished_at VARCHAR,
	status VARCHAR NOT NULL,
	document_count INTEGER NOT NULL CHECK (document_count >= 0),
	chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
	failure_count INTEGER NOT NULL CHECK (failure_count >= 0),
	safe_error_summary VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_pages_document_page
	ON pages(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_hierarchy_document_sequence
	ON hierarchy_nodes(document_id, sequence);
CREATE INDEX IF NOT EXISTS idx_chunks_document_sequence
	ON chunks(document_id, sequence);
CREATE INDEX IF NOT EXISTS idx_embeddings_model
	ON embeddings(model_name, model_version);
