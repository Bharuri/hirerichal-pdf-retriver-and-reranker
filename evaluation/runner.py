"""Independent loader, runner, and CLI for offline retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from config import Settings
from evaluation.retrieval_metrics import RetrievalMetrics, calculate_retrieval_metrics


logger = logging.getLogger(__name__)
_METRIC_NAMES = (
    "recall@5",
    "recall@10",
    "recall@20",
    "mrr",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
)


class EvaluationError(ValueError):
    """Raised when an evaluation dataset or evaluation run is invalid."""


@dataclass(frozen=True)
class EvaluationCase:
    """One manually curated query and its relevant chunk/document identifiers."""

    case_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...] = ()
    relevant_document_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Per-query metrics and number of retrieved chunk identifiers."""

    case_id: str
    query: str
    retrieved_count: int
    metrics: RetrievalMetrics


@dataclass(frozen=True)
class EvaluationRun:
    """Evaluation metrics together with reproducibility metadata."""

    run_id: str
    run_time: str
    dataset_path: str
    dataset_case_count: int
    embedding_provider: str | None
    embedding_model: str | None
    embedding_model_version: str | None
    reranker_provider: str | None
    reranker_model: str | None
    reranker_model_version: str | None
    retrieval_configuration: Mapping[str, Any]
    per_query: tuple[EvaluationCaseResult, ...]
    aggregate: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_time": self.run_time,
            "dataset_path": self.dataset_path,
            "dataset_case_count": self.dataset_case_count,
            "embedding": {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "version": self.embedding_model_version,
            },
            "reranker": {
                "provider": self.reranker_provider,
                "model": self.reranker_model,
                "version": self.reranker_model_version,
            },
            "retrieval_configuration": dict(self.retrieval_configuration),
            "per_query": [
                {
                    "case_id": result.case_id,
                    "query": result.query,
                    "retrieved_count": result.retrieved_count,
                    "metrics": result.metrics.as_dict(),
                }
                for result in self.per_query
            ],
            "aggregate": dict(self.aggregate),
        }


def load_evaluation_dataset(path: str | Path) -> tuple[EvaluationCase, ...]:
    """Load a JSON array or JSON Lines file of manually curated evaluation cases.

    Each case requires a unique ``case_id`` and non-empty ``query``. At least
    one of ``relevant_chunk_ids`` or ``relevant_document_ids`` must be given;
    both may be supplied, in which case chunk identifiers take precedence in
    the standalone scorer.
    """
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_file():
        raise EvaluationError("Evaluation dataset file does not exist.")
    try:
        text = dataset_path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvaluationError("Evaluation dataset file could not be read.") from error

    try:
        if dataset_path.suffix.casefold() == ".jsonl":
            raw_cases = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            raw_cases = json.loads(text)
    except json.JSONDecodeError as error:
        raise EvaluationError("Evaluation dataset contains invalid JSON.") from error
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationError("Evaluation dataset must contain a non-empty list of cases.")

    cases: list[EvaluationCase] = []
    seen_case_ids: set[str] = set()
    for index, value in enumerate(raw_cases):
        if not isinstance(value, dict):
            raise EvaluationError(f"Evaluation case {index + 1} must be an object.")
        case_id = value.get("case_id")
        query = value.get("query")
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvaluationError(f"Evaluation case {index + 1} requires a non-empty case_id.")
        case_id = case_id.strip()
        if case_id in seen_case_ids:
            raise EvaluationError("Evaluation case_id values must be unique.")
        if not isinstance(query, str) or not query.strip():
            raise EvaluationError(f"Evaluation case {case_id} requires a non-empty query.")
        chunks = _identifier_list(value.get("relevant_chunk_ids", []), case_id, "relevant_chunk_ids")
        documents = _identifier_list(
            value.get("relevant_document_ids", []), case_id, "relevant_document_ids"
        )
        if not chunks and not documents:
            raise EvaluationError(
                f"Evaluation case {case_id} requires relevant chunk or document identifiers."
            )
        cases.append(
            EvaluationCase(
                case_id=case_id,
                query=query.strip(),
                relevant_chunk_ids=chunks,
                relevant_document_ids=documents,
            )
        )
        seen_case_ids.add(case_id)
    return tuple(cases)


def evaluate_rankings(
    cases: Sequence[EvaluationCase],
    retrieve: Callable[[str], Sequence[Any]],
    *,
    dataset_path: str | Path,
    settings: Settings,
    embedding_model_version: str | None = None,
    reranker_model_version: str | None = None,
) -> EvaluationRun:
    """Run a supplied retrieval callable for each case and calculate metrics.

    Retrieval output items may be chunk ID strings or objects with
    ``chunk_id`` and ``document_id`` attributes (including ``RetrievalHit``).
    Relevant chunk identifiers are preferred when present; otherwise expected
    relevant document identifiers are evaluated.
    """
    if not cases:
        raise EvaluationError("At least one evaluation case is required.")
    if not callable(retrieve):
        raise TypeError("retrieve must be callable")

    per_query: list[EvaluationCaseResult] = []
    for case in cases:
        try:
            retrieved = tuple(retrieve(case.query))
        except Exception as error:
            logger.exception("Retrieval failed for evaluation case %s", case.case_id)
            raise EvaluationError(
                f"Retrieval failed for evaluation case {case.case_id}."
            ) from error
        if case.relevant_chunk_ids:
            ranked_ids = tuple(_chunk_identifier(item) for item in retrieved)
            relevant_ids = frozenset(case.relevant_chunk_ids)
        else:
            ranked_ids = tuple(_document_identifier(item) for item in retrieved)
            relevant_ids = frozenset(case.relevant_document_ids)
        metrics = calculate_retrieval_metrics(ranked_ids, relevant_ids)
        per_query.append(
            EvaluationCaseResult(
                case_id=case.case_id,
                query=case.query,
                retrieved_count=len(retrieved),
                metrics=metrics,
            )
        )

    aggregate = {
        metric_name: sum(result.metrics.as_dict()[metric_name] for result in per_query)
        / len(per_query)
        for metric_name in _METRIC_NAMES
    }
    retrieval_configuration: dict[str, Any] = {
        "semantic_top_k": settings.semantic_top_k,
        "keyword_top_k": settings.keyword_top_k,
        "hybrid_top_k": settings.hybrid_top_k,
        "reranker_top_k": settings.reranker_top_k,
        "hybrid_fusion_method": settings.hybrid_fusion_method,
        "rrf_k": settings.rrf_k,
        "hybrid_semantic_weight": settings.hybrid_semantic_weight,
        "hybrid_keyword_weight": settings.hybrid_keyword_weight,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunk_size_unit": settings.chunk_size_unit,
        "context_max_chunks": settings.context_max_chunks,
        "context_max_characters": settings.context_max_characters,
    }
    return EvaluationRun(
        run_id=uuid.uuid4().hex,
        run_time=datetime.now(timezone.utc).isoformat(),
        dataset_path=str(Path(dataset_path).expanduser().resolve()),
        dataset_case_count=len(cases),
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        embedding_model_version=embedding_model_version or settings.embedding_model_version,
        reranker_provider=settings.reranker_provider,
        reranker_model=settings.reranker_model,
        reranker_model_version=reranker_model_version or settings.reranker_model_version,
        retrieval_configuration=retrieval_configuration,
        per_query=tuple(per_query),
        aggregate=aggregate,
    )


def run_evaluation(
    dataset_path: str | Path,
    retrieve: Callable[[str], Sequence[Any]],
    *,
    settings: Settings | None = None,
    output_path: str | Path | None = None,
    embedding_model_version: str | None = None,
    reranker_model_version: str | None = None,
) -> EvaluationRun:
    """Load a curated dataset, evaluate rankings, and optionally write JSON output."""
    active_settings = settings or Settings.from_environment()
    cases = load_evaluation_dataset(dataset_path)
    run = evaluate_rankings(
        cases,
        retrieve,
        dataset_path=dataset_path,
        settings=active_settings,
        embedding_model_version=embedding_model_version,
        reranker_model_version=reranker_model_version,
    )
    if output_path is not None:
        destination = Path(output_path).expanduser()
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(run.as_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise EvaluationError("Evaluation results could not be written.") from error
    return run


def _identifier_list(value: object, case_id: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvaluationError(f"{field} for evaluation case {case_id} must be a list.")
    identifiers: list[str] = []
    for identifier in value:
        if not isinstance(identifier, str) or not identifier.strip():
            raise EvaluationError(
                f"{field} for evaluation case {case_id} must contain non-empty strings."
            )
        normalized = identifier.strip()
        if normalized not in identifiers:
            identifiers.append(normalized)
    return tuple(identifiers)


def _chunk_identifier(item: Any) -> str:
    identifier = item if isinstance(item, str) else getattr(item, "chunk_id", None)
    if not isinstance(identifier, str) or not identifier:
        raise EvaluationError("Retrieval results must provide chunk_id values.")
    return identifier


def _document_identifier(item: Any) -> str:
    identifier = getattr(item, "document_id", None)
    if not isinstance(identifier, str) or not identifier:
        raise EvaluationError(
            "Document-level evaluation requires retrieval results with document_id values."
        )
    return identifier


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a retrieval function against a manually curated JSON dataset."
    )
    parser.add_argument("dataset", type=Path, help="JSON array or JSON Lines dataset")
    parser.add_argument(
        "--output", type=Path, help="Optional path for machine-readable JSON metrics"
    )
    parser.add_argument(
        "--retriever",
        required=True,
        help="Python callable as module:function; no Streamlit UI is started",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the standalone evaluation command using a configured retrieval callable."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(arguments)
    try:
        retrieve = _load_retriever(args.retriever)
        run = run_evaluation(
            args.dataset,
            retrieve,
            output_path=args.output,
        )
    except (EvaluationError, ImportError, AttributeError, ValueError) as error:
        logger.error("%s", error)
        return 2
    print(json.dumps(run.as_dict(), indent=2, ensure_ascii=False))
    return 0


def _load_retriever(specification: str) -> Callable[[str], Sequence[Any]]:
    if ":" not in specification:
        raise EvaluationError("Retriever must use module:function syntax.")
    module_name, attribute_name = specification.split(":", 1)
    if not module_name or not attribute_name:
        raise EvaluationError("Retriever must use module:function syntax.")
    import importlib

    module = importlib.import_module(module_name)
    retriever = getattr(module, attribute_name)
    if not callable(retriever):
        raise EvaluationError("Configured retriever entry point is not callable.")
    return retriever


if __name__ == "__main__":
    sys.exit(main())
