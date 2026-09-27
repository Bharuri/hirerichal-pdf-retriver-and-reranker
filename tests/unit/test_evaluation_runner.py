"""Offline dataset and evaluation-run tests for TASK-10."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import Settings
from evaluation.runner import (
    EvaluationError,
    EvaluationCase,
    evaluate_rankings,
    load_evaluation_dataset,
    run_evaluation,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings.from_environment(
        {
            "PDF_RAG_SEMANTIC_TOP_K": "7",
            "PDF_RAG_KEYWORD_TOP_K": "9",
            "PDF_RAG_HYBRID_TOP_K": "4",
            "PDF_RAG_RERANKER_TOP_K": "3",
            "PDF_RAG_HYBRID_FUSION_METHOD": "weighted_sum",
            "PDF_RAG_EMBEDDING_PROVIDER": "sentence-transformers",
            "PDF_RAG_EMBEDDING_MODEL": "test-embedder",
            "PDF_RAG_EMBEDDING_MODEL_VERSION": "embedding-revision",
            "PDF_RAG_RERANKER_PROVIDER": "sentence-transformers-cross-encoder",
            "PDF_RAG_RERANKER_MODEL": "test-reranker",
            "PDF_RAG_RERANKER_MODEL_VERSION": "reranker-revision",
        },
        project_root=tmp_path,
    )


def test_load_json_and_jsonl_manual_evaluation_datasets(tmp_path: Path) -> None:
    json_path = tmp_path / "cases.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1",
                    "query": "MVCC behavior",
                    "relevant_chunk_ids": ["chunk-1", "chunk-1"],
                    "relevant_document_ids": ["doc-1"],
                }
            ]
        ),
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "cases.jsonl"
    jsonl_path.write_text(
        '{"case_id":"case-2","query":"vacuum","relevant_document_ids":["doc-2"]}\n',
        encoding="utf-8",
    )

    cases = load_evaluation_dataset(json_path)
    jsonl_cases = load_evaluation_dataset(jsonl_path)

    assert cases == (
        EvaluationCase("case-1", "MVCC behavior", ("chunk-1",), ("doc-1",)),
    )
    assert jsonl_cases == (EvaluationCase("case-2", "vacuum", (), ("doc-2",)),)


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        "[]",
        '{"case_id":"x","query":"q","relevant_chunk_ids":["c"]}',
        '[{"case_id":"x","query":"q"}]',
        '[{"case_id":"x","query":"q","relevant_chunk_ids":"c"}]',
        '[{"case_id":"x","query":"q","relevant_chunk_ids":["c"]},'
        '{"case_id":"x","query":"q","relevant_chunk_ids":["c"]}]',
    ],
)
def test_dataset_loader_rejects_invalid_dataset_shapes_and_records(
    tmp_path: Path, payload: str
) -> None:
    dataset = tmp_path / "invalid.json"
    dataset.write_text(payload, encoding="utf-8")

    with pytest.raises(EvaluationError):
        load_evaluation_dataset(dataset)


def test_dataset_loader_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="does not exist"):
        load_evaluation_dataset(tmp_path / "missing.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    with pytest.raises(EvaluationError, match="invalid JSON"):
        load_evaluation_dataset(malformed)


def test_evaluate_rankings_reports_per_query_aggregate_and_run_configuration(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "curated.json"
    dataset.write_text(
        json.dumps(
            [
                {"case_id": "chunk-case", "query": "q1", "relevant_chunk_ids": ["c1"]},
                {"case_id": "document-case", "query": "q2", "relevant_document_ids": ["d2"]},
            ]
        ),
        encoding="utf-8",
    )
    cases = load_evaluation_dataset(dataset)
    settings = _settings(tmp_path)
    calls: list[str] = []

    def retrieve(query: str):
        calls.append(query)
        if query == "q1":
            return (SimpleNamespace(chunk_id="c1", document_id="d1"),)
        return (SimpleNamespace(chunk_id="c2", document_id="d2"),)

    run = evaluate_rankings(
        cases,
        retrieve,
        dataset_path=dataset,
        settings=settings,
    )

    assert calls == ["q1", "q2"]
    assert run.dataset_case_count == 2
    assert len(run.run_id) == 32
    assert run.embedding_provider == "sentence-transformers"
    assert run.embedding_model == "test-embedder"
    assert run.embedding_model_version == "embedding-revision"
    assert run.reranker_provider == "sentence-transformers-cross-encoder"
    assert run.reranker_model == "test-reranker"
    assert run.reranker_model_version == "reranker-revision"
    assert run.retrieval_configuration["hybrid_fusion_method"] == "weighted_sum"
    assert run.retrieval_configuration["semantic_top_k"] == 7
    assert run.per_query[0].metrics.mrr == 1.0
    assert run.per_query[1].metrics.recall_at_5 == 1.0
    assert run.aggregate["mrr"] == 1.0
    assert run.aggregate["recall@20"] == 1.0
    assert Path(run.dataset_path) == dataset.resolve()


def test_run_evaluation_serializes_machine_readable_output(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        '[{"case_id":"c1","query":"q","relevant_chunk_ids":["found"]}]',
        encoding="utf-8",
    )
    output = tmp_path / "reports" / "metrics.json"
    run = run_evaluation(
        dataset,
        lambda query: ("found",),
        settings=_settings(tmp_path),
        output_path=output,
        embedding_model_version="resolved-embedding-sha",
        reranker_model_version="resolved-reranker-sha",
    )

    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert run.embedding_model_version == "resolved-embedding-sha"
    assert run.reranker_model_version == "resolved-reranker-sha"
    assert serialized["aggregate"]["mrr"] == 1.0
    assert serialized["per_query"][0]["case_id"] == "c1"


def test_evaluation_uses_stable_order_for_tied_scores_and_deduplicates_hits(
    tmp_path: Path,
) -> None:
    case = EvaluationCase("ties", "query", ("relevant",))
    tied_results = (
        SimpleNamespace(chunk_id="irrelevant", score=0.5),
        SimpleNamespace(chunk_id="relevant", score=0.5),
        SimpleNamespace(chunk_id="relevant", score=0.5),
    )

    run = evaluate_rankings(
        (case,),
        lambda query: tied_results,
        dataset_path=tmp_path / "dataset.json",
        settings=_settings(tmp_path),
    )

    assert run.per_query[0].retrieved_count == 3
    assert run.per_query[0].metrics.mrr == 0.5
    assert run.per_query[0].metrics.recall_at_5 == 1.0


def test_evaluation_cli_loads_retriever_without_streamlit(tmp_path: Path) -> None:
    from evaluation.runner import _load_retriever

    retriever = _load_retriever("retrieval.keyword:keyword_search")

    assert callable(retriever)


def test_evaluate_rankings_reports_retrieval_failure_without_leaking_details(
    tmp_path: Path,
) -> None:
    case = EvaluationCase("failure", "query", ("chunk",))

    def failing_retrieval(query: str):
        raise RuntimeError("private provider response")

    with pytest.raises(EvaluationError, match="Retrieval failed for evaluation case failure") as error:
        evaluate_rankings(
            (case,),
            failing_retrieval,
            dataset_path=tmp_path / "dataset.json",
            settings=_settings(tmp_path),
        )
    assert "private provider response" not in str(error.value)
