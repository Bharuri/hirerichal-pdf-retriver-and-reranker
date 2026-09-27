"""Hand-calculated tests for TASK-10 retrieval metrics."""

import math

import pytest

from evaluation.retrieval_metrics import calculate_retrieval_metrics


def test_metrics_match_hand_calculated_recall_mrr_and_ndcg() -> None:
    metrics = calculate_retrieval_metrics(
        ("relevant-a", "irrelevant", "relevant-b", "other"),
        {"relevant-a", "relevant-b"},
    )

    expected_ndcg = (1.0 + 1.0 / math.log2(4)) / (
        1.0 + 1.0 / math.log2(3)
    )
    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.recall_at_20 == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg_at_5 == pytest.approx(expected_ndcg)
    assert metrics.ndcg_at_10 == pytest.approx(expected_ndcg)
    assert metrics.ndcg_at_20 == pytest.approx(expected_ndcg)
    assert metrics.as_dict()["recall@5"] == metrics.recall_at_5


def test_recall_and_ndcg_apply_requested_cutoffs_and_mrr_uses_first_match() -> None:
    ranked = tuple(f"doc-{index}" for index in range(1, 22))
    metrics = calculate_retrieval_metrics(ranked, {"doc-6", "doc-21"})

    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_10 == 0.5
    assert metrics.recall_at_20 == 0.5
    assert metrics.mrr == pytest.approx(1 / 6)
    assert metrics.ndcg_at_5 == 0.0
    assert metrics.ndcg_at_10 > 0.0
    assert metrics.ndcg_at_20 == metrics.ndcg_at_10


def test_empty_retrieval_and_empty_relevant_set_are_zero() -> None:
    empty_relevant = calculate_retrieval_metrics(("a", "b"), set())
    no_results = calculate_retrieval_metrics((), {"a"})

    assert set(empty_relevant.as_dict().values()) == {0.0}
    assert no_results.recall_at_5 == 0.0
    assert no_results.mrr == 0.0
    assert no_results.ndcg_at_20 == 0.0


def test_duplicate_ranked_ids_count_once_at_first_position() -> None:
    metrics = calculate_retrieval_metrics(
        ("miss", "hit", "hit", "other", "miss"),
        {"hit", "other"},
    )

    assert metrics.recall_at_5 == 1.0
    assert metrics.mrr == 0.5
    expected = (1.0 / math.log2(3) + 1.0 / math.log2(4)) / (
        1.0 + 1.0 / math.log2(3)
    )
    assert metrics.ndcg_at_5 == pytest.approx(expected)


def test_metrics_validate_cutoffs_and_identifier_types() -> None:
    with pytest.raises(ValueError, match="exactly"):
        calculate_retrieval_metrics(("a",), {"a"}, cutoffs=(5, 10))
    with pytest.raises(TypeError, match="ranked identifiers"):
        calculate_retrieval_metrics((1,), {"a"})
    with pytest.raises(TypeError, match="relevant identifiers"):
        calculate_retrieval_metrics(("a",), {1})
