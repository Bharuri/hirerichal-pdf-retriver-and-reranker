"""Small, dependency-free configuration for the local retrieval application."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigurationError(ValueError):
    """Raised when local application configuration is invalid."""


@dataclass(frozen=True)
class Settings:
    """Validated paths and retrieval/chunking defaults for one local user."""

    project_root: Path
    corpus_dir: Path
    database_path: Path
    artifacts_dir: Path
    host: str = "127.0.0.1"
    port: int = 8501
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunk_size_unit: str = "characters"
    semantic_top_k: int = 20
    keyword_top_k: int = 20
    hybrid_top_k: int = 10
    reranker_top_k: int = 5
    context_max_chunks: int = 2
    context_max_characters: int = 4000
    hybrid_fusion_method: str = "rrf"
    rrf_k: int = 60
    hybrid_semantic_weight: float = 0.75
    hybrid_keyword_weight: float = 0.25
    embedding_provider: str | None = "sentence-transformers"
    embedding_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_version: str | None = None
    reranker_provider: str | None = "sentence-transformers-cross-encoder"
    reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_model_version: str | None = None

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        project_root: str | Path | None = None,
    ) -> Settings:
        """Load settings from PDF_RAG_* variables with local relative defaults.

        ``environ`` and ``project_root`` are injectable to keep configuration
        tests deterministic. Secret values are neither loaded nor retained.
        """
        values = os.environ if environ is None else environ
        root = Path(project_root).expanduser().resolve() if project_root else Path(__file__).resolve().parent

        settings = cls(
            project_root=root,
            corpus_dir=_resolve_path(
                values.get("PDF_RAG_CORPUS_DIR", "data/pdfs"),
                root,
                "PDF_RAG_CORPUS_DIR",
            ),
            database_path=_resolve_path(
                values.get("PDF_RAG_DATABASE_PATH", "data/duckdb/retrieval.duckdb"),
                root,
                "PDF_RAG_DATABASE_PATH",
            ),
            artifacts_dir=_resolve_path(
                values.get("PDF_RAG_ARTIFACTS_DIR", "data/artifacts"),
                root,
                "PDF_RAG_ARTIFACTS_DIR",
            ),
            host=values.get("PDF_RAG_HOST", "127.0.0.1").strip(),
            port=_integer(values, "PDF_RAG_PORT", 8501),
            chunk_size=_integer(values, "PDF_RAG_CHUNK_SIZE", 1000),
            chunk_overlap=_integer(values, "PDF_RAG_CHUNK_OVERLAP", 150),
            chunk_size_unit=values.get("PDF_RAG_CHUNK_SIZE_UNIT", "characters").strip().lower(),
            semantic_top_k=_integer(values, "PDF_RAG_SEMANTIC_TOP_K", 20),
            keyword_top_k=_integer(values, "PDF_RAG_KEYWORD_TOP_K", 20),
            hybrid_top_k=_integer(values, "PDF_RAG_HYBRID_TOP_K", 10),
            reranker_top_k=_integer(values, "PDF_RAG_RERANKER_TOP_K", 5),
            context_max_chunks=_integer(values, "PDF_RAG_CONTEXT_MAX_CHUNKS", 2),
            context_max_characters=_integer(
                values, "PDF_RAG_CONTEXT_MAX_CHARACTERS", 4000
            ),
            hybrid_fusion_method=values.get(
                "PDF_RAG_HYBRID_FUSION_METHOD", "rrf"
            ).strip().lower(),
            rrf_k=_integer(values, "PDF_RAG_RRF_K", 60),
            hybrid_semantic_weight=_floating(
                values, "PDF_RAG_HYBRID_SEMANTIC_WEIGHT", 0.75
            ),
            hybrid_keyword_weight=_floating(
                values, "PDF_RAG_HYBRID_KEYWORD_WEIGHT", 0.25
            ),
            embedding_provider=_optional(
                values, "PDF_RAG_EMBEDDING_PROVIDER", "sentence-transformers"
            ),
            embedding_model=_optional(
                values,
                "PDF_RAG_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            embedding_model_version=_optional(
                values, "PDF_RAG_EMBEDDING_MODEL_VERSION"
            ),
            reranker_provider=_optional(
                values,
                "PDF_RAG_RERANKER_PROVIDER",
                "sentence-transformers-cross-encoder",
            ),
            reranker_model=_optional(
                values,
                "PDF_RAG_RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            reranker_model_version=_optional(
                values, "PDF_RAG_RERANKER_MODEL_VERSION"
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject unsafe local network defaults and invalid numeric settings."""
        if self.host.lower() not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigurationError("PDF_RAG_HOST must use a loopback address for local mode.")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("PDF_RAG_PORT must be between 1 and 65535.")
        if self.chunk_size <= 0:
            raise ConfigurationError("PDF_RAG_CHUNK_SIZE must be greater than zero.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                "PDF_RAG_CHUNK_OVERLAP must be non-negative and smaller than PDF_RAG_CHUNK_SIZE."
            )
        if self.chunk_size_unit not in {"characters", "tokens"}:
            raise ConfigurationError("PDF_RAG_CHUNK_SIZE_UNIT must be 'characters' or 'tokens'.")
        for name, value in (
            ("PDF_RAG_SEMANTIC_TOP_K", self.semantic_top_k),
            ("PDF_RAG_KEYWORD_TOP_K", self.keyword_top_k),
            ("PDF_RAG_HYBRID_TOP_K", self.hybrid_top_k),
            ("PDF_RAG_RERANKER_TOP_K", self.reranker_top_k),
        ):
            if value <= 0:
                raise ConfigurationError(f"{name} must be greater than zero.")
        if self.context_max_chunks < 0:
            raise ConfigurationError("PDF_RAG_CONTEXT_MAX_CHUNKS must be non-negative.")
        if self.context_max_characters < 0:
            raise ConfigurationError(
                "PDF_RAG_CONTEXT_MAX_CHARACTERS must be non-negative."
            )
        if self.hybrid_fusion_method not in {"rrf", "weighted_sum"}:
            raise ConfigurationError(
                "PDF_RAG_HYBRID_FUSION_METHOD must be 'rrf' or 'weighted_sum'."
            )
        if self.rrf_k <= 0:
            raise ConfigurationError("PDF_RAG_RRF_K must be greater than zero.")
        weights = (self.hybrid_semantic_weight, self.hybrid_keyword_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ConfigurationError("Hybrid retrieval weights must be finite and non-negative.")
        if not any(weights):
            raise ConfigurationError("At least one hybrid retrieval weight must be greater than zero.")
        _paired_settings("embedding", self.embedding_provider, self.embedding_model)
        _paired_settings("reranker", self.reranker_provider, self.reranker_model)


def _resolve_path(value: str, project_root: Path, setting_name: str) -> Path:
    raw_path = Path(value).expanduser()
    if not str(value).strip():
        raise ConfigurationError(f"{setting_name} must not be empty.")
    return (raw_path if raw_path.is_absolute() else project_root / raw_path).resolve()


def _integer(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer.") from error


def _floating(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number.") from error


def _optional(
    values: Mapping[str, str], name: str, default: str = ""
) -> str | None:
    value = values.get(name, default).strip()
    return value or None


def _paired_settings(component: str, provider: str | None, model: str | None) -> None:
    if (provider is None) != (model is None):
        raise ConfigurationError(
            f"PDF_RAG_{component.upper()}_PROVIDER and PDF_RAG_{component.upper()}_MODEL "
            "must either both be set or both be omitted."
        )
