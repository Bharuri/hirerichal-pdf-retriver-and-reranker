"""Focused tests for TASK-07 query normalization and hybrid rank fusion."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from config import Settings
from ingestion.chunker import RetrievalChunk
from ingestion.indexer import SentenceTransformerProvider, create_embedding_provider
from retrieval.common import RetrievalHit, normalize_query, query_terms
from retrieval.hybrid import fuse_results


def _hit(chunk_id: str, rank: int, score: float) -> RetrievalHit:
    chunk = RetrievalChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        parent_id="section-1",
        level="content",
        section_title="Database",
        section_path=("Manual", "Database"),
        previous_chunk_id=None,
        next_chunk_id=None,
        page_start=2,
        page_end=3,
        sequence=int(chunk_id.rsplit("-", 1)[-1]),
        text=f"Evidence for {chunk_id}",
        chunk_size=20,
        size_unit="characters",
        token_count=3,
        content_hash=f"hash-{chunk_id}",
        configuration_fingerprint="config",
    )
    return RetrievalHit(
        chunk=chunk,
        source_filename="manual.pdf",
        source_path="manual.pdf",
        rank=rank,
        score=score,
    )


def test_query_normalization_preserves_technical_identifiers() -> None:
    normalized = normalize_query("  ＰＯＳＴＧＲＥＳＱＬ\tMVCC  HTTP/2 C++ pg_catalog ")

    assert normalized == "postgresql mvcc http/2 c++ pg_catalog"
    assert query_terms(normalized) == (
        "postgresql", "mvcc", "http/2", "c++", "pg_catalog"
    )
    assert normalize_query(" \n\t ") == ""


def test_rrf_fusion_deduplicates_and_preserves_channel_ranks_and_scores() -> None:
    keyword = (_hit("chunk-1", 1, 4.0), _hit("chunk-2", 2, 2.0))
    semantic = (_hit("chunk-2", 1, 0.9), _hit("chunk-1", 2, 0.8))

    fused = fuse_results(keyword, semantic, top_k=2, rrf_k=10)

    # Both candidates have the same RRF value; the semantic rank breaks the tie.
    assert [item.chunk_id for item in fused] == ["chunk-2", "chunk-1"]
    assert [item.rank for item in fused] == [1, 2]
    assert fused[0].keyword_rank == 2
    assert fused[0].semantic_rank == 1
    assert fused[0].keyword_score == 2.0
    assert fused[0].semantic_score == 0.9
    assert fused[0].score == pytest.approx(1 / 11 + 1 / 12)
    assert len({item.chunk_id for item in fused}) == len(fused)
    assert fused[0].source_filename == "manual.pdf"
    assert fused[0].page_start == 2
    assert fused[0].section_path == ("Manual", "Database")


def test_rrf_duplicate_inputs_keep_best_rank_and_configurable_weights() -> None:
    keyword = (_hit("chunk-1", 3, 1.0), _hit("chunk-1", 1, 2.0))
    semantic = (_hit("chunk-1", 2, 0.8),)

    fused = fuse_results(
        keyword,
        semantic,
        rrf_k=5,
        keyword_weight=2.0,
        semantic_weight=1.0,
    )

    assert len(fused) == 1
    assert fused[0].keyword_rank == 1
    assert fused[0].keyword_score == 2.0
    assert fused[0].score == pytest.approx(2 / 6 + 1 / 7)


def test_weighted_sum_fusion_uses_normalized_channel_scores() -> None:
    keyword = (_hit("chunk-1", 1, 10.0), _hit("chunk-2", 2, 0.0))
    semantic = (_hit("chunk-2", 1, 0.8), _hit("chunk-1", 2, 0.2))

    fused = fuse_results(
        keyword, semantic, method="weighted_sum", keyword_weight=2, semantic_weight=1
    )

    assert [item.chunk_id for item in fused] == ["chunk-1", "chunk-2"]
    assert fused[0].score == pytest.approx(2.0)
    assert fused[1].score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "unsupported"}, "fusion method"),
        ({"rrf_k": 0}, "rrf_k"),
        ({"semantic_weight": -1.0}, "weights"),
        ({"semantic_weight": 0.0, "keyword_weight": 0.0}, "at least one"),
    ],
)
def test_fusion_rejects_invalid_parameters(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fuse_results((), (), **kwargs)


def test_sentence_transformer_adapter_encodes_batches_and_records_commit() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], dict[str, object]]] = []
            self._modules = {
                "transformer": SimpleNamespace(
                    auto_model=SimpleNamespace(
                        config=SimpleNamespace(_commit_hash="abc123")
                    )
                )
            }

        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            self.calls.append((texts, kwargs))
            return [[float(index), float(len(text))] for index, text in enumerate(texts)]

    model = FakeModel()
    provider = SentenceTransformerProvider(
        "sentence-transformers/test-model", model=model
    )

    vectors = provider.embed(("first", "second"))

    assert provider.provider_name == "sentence-transformers"
    assert provider.model_version == "abc123"
    assert vectors == ((0.0, 5.0), (1.0, 6.0))
    assert model.calls[0][0] == ["first", "second"]
    assert model.calls[0][1] == {
        "convert_to_numpy": True,
        "normalize_embeddings": False,
        "show_progress_bar": False,
    }


def test_sentence_transformer_adapter_uses_explicit_revision_when_unresolved() -> None:
    provider = SentenceTransformerProvider(
        "sentence-transformers/test-model",
        revision="revision-42",
        model=SimpleNamespace(_modules={}),
    )

    assert provider.model_version == "revision-42"


def test_embedding_provider_factory_uses_configured_sentence_transformer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ingestion.indexer as indexer

    constructed: list[tuple[str, str | None]] = []

    class FakeProvider:
        provider_name = "sentence-transformers"
        model_name = "sentence-transformers/custom"
        model_version = "revision-5"

        def __init__(self, model_name: str, *, revision: str | None = None) -> None:
            constructed.append((model_name, revision))

    monkeypatch.setattr(indexer, "SentenceTransformerProvider", FakeProvider)
    settings = Settings.from_environment(
        {
            "PDF_RAG_EMBEDDING_PROVIDER": "sentence-transformers",
            "PDF_RAG_EMBEDDING_MODEL": "sentence-transformers/custom",
            "PDF_RAG_EMBEDDING_MODEL_VERSION": "revision-5",
        },
        project_root=tmp_path,
    )

    provider = create_embedding_provider(settings)

    assert isinstance(provider, FakeProvider)
    assert constructed == [("sentence-transformers/custom", "revision-5")]