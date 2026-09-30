"""Focused tests for TASK-01 local settings and safe defaults."""

from pathlib import Path

import pytest

from config import ConfigurationError, Settings


def test_local_defaults_are_relative_to_project_root_and_loopback_only(tmp_path: Path) -> None:
    settings = Settings.from_environment({}, project_root=tmp_path)

    assert settings.project_root == tmp_path.resolve()
    assert settings.corpus_dir == (tmp_path / "data/pdfs").resolve()
    assert settings.database_path == (tmp_path / "data/duckdb/retrieval.duckdb").resolve()
    assert settings.artifacts_dir == (tmp_path / "data/artifacts").resolve()
    assert settings.host == "127.0.0.1"
    assert settings.embedding_provider == "sentence-transformers"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_model_version is None
    assert settings.reranker_provider == "sentence-transformers-cross-encoder"
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_chunk_and_retrieval_defaults_match_documented_local_values(tmp_path: Path) -> None:
    settings = Settings.from_environment({}, project_root=tmp_path)

    assert (settings.chunk_size, settings.chunk_overlap, settings.chunk_size_unit) == (1000, 150, "characters")
    assert (settings.semantic_top_k, settings.keyword_top_k) == (20, 20)
    assert (settings.hybrid_top_k, settings.reranker_top_k) == (10, 5)
    assert settings.hybrid_fusion_method == "rrf"
    assert settings.rrf_k == 60
    assert (settings.hybrid_semantic_weight, settings.hybrid_keyword_weight) == (0.75, 0.25)
    assert (settings.context_max_chunks, settings.context_max_characters) == (2, 4000)


def test_paths_and_retrieval_values_can_be_overridden_without_absolute_defaults(tmp_path: Path) -> None:
    settings = Settings.from_environment(
        {
            "PDF_RAG_CORPUS_DIR": "local-pdfs",
            "PDF_RAG_DATABASE_PATH": "state/index.duckdb",
            "PDF_RAG_ARTIFACTS_DIR": "state/artifacts",
            "PDF_RAG_CHUNK_SIZE": "800",
            "PDF_RAG_CHUNK_OVERLAP": "100",
            "PDF_RAG_SEMANTIC_TOP_K": "12",
            "PDF_RAG_KEYWORD_TOP_K": "14",
            "PDF_RAG_HYBRID_TOP_K": "9",
            "PDF_RAG_HYBRID_FUSION_METHOD": "weighted_sum",
            "PDF_RAG_RRF_K": "45",
            "PDF_RAG_HYBRID_SEMANTIC_WEIGHT": "0.7",
            "PDF_RAG_HYBRID_KEYWORD_WEIGHT": "1.3",
            "PDF_RAG_RERANKER_TOP_K": "4",
            "PDF_RAG_CONTEXT_MAX_CHUNKS": "3",
            "PDF_RAG_CONTEXT_MAX_CHARACTERS": "2500",
            "PDF_RAG_EMBEDDING_PROVIDER": "local",
            "PDF_RAG_EMBEDDING_MODEL": "test-embedding-model",
            "PDF_RAG_EMBEDDING_MODEL_VERSION": "revision-123",
            "PDF_RAG_RERANKER_PROVIDER": "cross-encoder",
            "PDF_RAG_RERANKER_MODEL": "test-reranker",
            "PDF_RAG_RERANKER_MODEL_VERSION": "rerank-revision-5",
        },
        project_root=tmp_path,
    )

    assert settings.corpus_dir == (tmp_path / "local-pdfs").resolve()
    assert settings.database_path == (tmp_path / "state/index.duckdb").resolve()
    assert settings.artifacts_dir == (tmp_path / "state/artifacts").resolve()
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.semantic_top_k == 12
    assert settings.keyword_top_k == 14
    assert settings.hybrid_top_k == 9
    assert settings.hybrid_fusion_method == "weighted_sum"
    assert settings.rrf_k == 45
    assert settings.hybrid_semantic_weight == 0.7
    assert settings.hybrid_keyword_weight == 1.3
    assert settings.reranker_top_k == 4
    assert settings.context_max_chunks == 3
    assert settings.context_max_characters == 2500
    assert settings.embedding_model == "test-embedding-model"
    assert settings.embedding_model_version == "revision-123"
    assert settings.reranker_model_version == "rerank-revision-5"


def test_settings_reject_non_loopback_binding(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="loopback"):
        Settings.from_environment({"PDF_RAG_HOST": "0.0.0.0"}, project_root=tmp_path)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PDF_RAG_PORT", "bad", "must be an integer"),
        ("PDF_RAG_PORT", "70000", "between 1 and 65535"),
        ("PDF_RAG_CHUNK_SIZE", "0", "greater than zero"),
        ("PDF_RAG_CHUNK_OVERLAP", "1000", "smaller than"),
        ("PDF_RAG_CHUNK_SIZE_UNIT", "words", "characters.*tokens"),
        ("PDF_RAG_SEMANTIC_TOP_K", "0", "greater than zero"),
        ("PDF_RAG_KEYWORD_TOP_K", "-1", "greater than zero"),
        ("PDF_RAG_HYBRID_FUSION_METHOD", "unknown", "rrf.*weighted_sum"),
        ("PDF_RAG_RRF_K", "0", "greater than zero"),
        ("PDF_RAG_HYBRID_SEMANTIC_WEIGHT", "nan", "finite and non-negative"),
        ("PDF_RAG_HYBRID_KEYWORD_WEIGHT", "-0.2", "finite and non-negative"),
        ("PDF_RAG_CONTEXT_MAX_CHUNKS", "-1", "non-negative"),
        ("PDF_RAG_CONTEXT_MAX_CHARACTERS", "-5", "non-negative"),
    ],
)
def test_invalid_values_raise_clear_configuration_error(
    tmp_path: Path, name: str, value: str, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Settings.from_environment({name: value}, project_root=tmp_path)


def test_provider_and_model_must_be_configured_together(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="must either both be set"):
        Settings.from_environment(
            {
                "PDF_RAG_EMBEDDING_PROVIDER": "local",
                "PDF_RAG_EMBEDDING_MODEL": "",
            },
            project_root=tmp_path,
        )


def test_secret_values_are_not_loaded_or_rendered(tmp_path: Path) -> None:
    secret = "never-display-this-secret"
    settings = Settings.from_environment(
        {"PDF_RAG_API_KEY": secret, "API_KEY": secret},
        project_root=tmp_path,
    )

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert not hasattr(settings, "api_key")


def test_whitespace_only_path_is_rejected_without_echoing_value(tmp_path: Path) -> None:
    secret_path = "   "
    with pytest.raises(ConfigurationError) as error:
        Settings.from_environment({"PDF_RAG_CORPUS_DIR": secret_path}, project_root=tmp_path)

    assert "PDF_RAG_CORPUS_DIR" in str(error.value)
    assert secret_path not in str(error.value)
