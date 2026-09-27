"""Focused tests for TASK-04 deterministic hierarchical chunking."""

import pytest

from ingestion.chunker import chunk_document
from ingestion.hierarchy import HierarchyNode, HierarchyResult
from ingestion.pdf_loader import PageText, PdfReadResult, PdfStatus


def make_document() -> PdfReadResult:
    return PdfReadResult(
        document_id="doc-1",
        source_filename="manual.pdf",
        source_path="manual.pdf",
        content_sha256="pdf-content-hash",
        page_count=3,
        metadata={"title": "Database Manual"},
        pages=(
            PageText(1, "page one"),
            PageText(2, "page two"),
            PageText(3, "page three"),
        ),
        status=PdfStatus.EXTRACTED,
    )


def node(
    node_id: str,
    *,
    parent_id: str | None,
    level: str,
    title: str | None,
    path: tuple[str, ...],
    page: int,
    sequence: int,
    text: str | None = None,
) -> HierarchyNode:
    return HierarchyNode(
        node_id=node_id,
        document_id="doc-1",
        parent_id=parent_id,
        level=level,
        section_title=title,
        section_path=path,
        page_start=page,
        page_end=page,
        sequence=sequence,
        detection_method="fixture",
        confidence=None,
        text=text,
    )


def make_hierarchy(*content: tuple[str, str, int, str]) -> HierarchyResult:
    root = node(
        "root",
        parent_id=None,
        level="document",
        title="Database Manual",
        path=("Database Manual",),
        page=1,
        sequence=0,
    )
    nodes = [root]
    current_sections: dict[str, HierarchyNode] = {}
    for sequence, (section_id, text, page_number, heading) in enumerate(
        content, start=1
    ):
        if section_id not in current_sections:
            section = node(
                section_id,
                parent_id="root",
                level="section",
                title=heading,
                path=("Database Manual", heading),
                page=page_number,
                sequence=sequence,
            )
            current_sections[section_id] = section
            nodes.append(section)
        nodes.append(
            node(
                f"content-{sequence}",
                parent_id=section_id,
                level="content",
                title=None,
                path=current_sections[section_id].section_path,
                page=page_number,
                sequence=sequence + len(content),
                text=text,
            )
        )
    nodes[0] = node(
        "root",
        parent_id=None,
        level="document",
        title="Database Manual",
        path=("Database Manual",),
        page=1,
        sequence=0,
    )
    return HierarchyResult("doc-1", tuple(nodes))


def test_chunks_keep_hierarchy_source_pages_and_same_section_neighbors() -> None:
    document = make_document()
    hierarchy = make_hierarchy(
        ("section-a", "First paragraph has enough content.", 1, "Transactions"),
        ("section-a", "Second paragraph continues here.", 2, "Transactions"),
        ("section-b", "Separate section must remain separate.", 3, "Indexes"),
    )

    chunks = chunk_document(document, hierarchy, chunk_size=200, chunk_overlap=0)

    assert len(chunks) == 2
    first, second = chunks
    assert (
        first.text
        == "First paragraph has enough content.\n\nSecond paragraph continues here."
    )
    assert first.parent_id == "section-a"
    assert first.section_path == ("Database Manual", "Transactions")
    assert (first.page_start, first.page_end) == (1, 2)
    assert first.next_chunk_id is None
    assert second.parent_id == "section-b"
    assert second.section_path == ("Database Manual", "Indexes")
    assert second.page_start == second.page_end == 3
    assert first.chunk_id != second.chunk_id


def test_chunk_size_is_enforced_with_overlap_and_neighbors_linked() -> None:
    document = make_document()
    hierarchy = make_hierarchy(
        (
            "section-a",
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa.",
            1,
            "Terms",
        ),
    )

    chunks = chunk_document(document, hierarchy, chunk_size=15, chunk_overlap=5)

    assert len(chunks) > 1
    assert all(chunk.chunk_size <= 15 for chunk in chunks)
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id
    assert chunks[0].section_path == chunks[1].section_path
    assert chunks[0].sequence == 0
    assert all(chunk.configuration_fingerprint for chunk in chunks)


def test_long_unbroken_text_is_split_without_exceeding_character_limit() -> None:
    document = make_document()
    hierarchy = make_hierarchy(("section-a", "x" * 23, 1, "Long value"))

    chunks = chunk_document(document, hierarchy, chunk_size=8, chunk_overlap=0)

    assert [chunk.text for chunk in chunks] == ["x" * 8, "x" * 8, "x" * 7]
    assert all(len(chunk.text) <= 8 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == "x" * 23


def test_sql_code_and_list_text_remain_intact_when_under_limit() -> None:
    document = make_document()
    source = "SELECT id, name\nFROM customers\nWHERE active = true;"
    hierarchy = make_hierarchy(("section-a", source, 2, "SQL Example"))

    chunks = chunk_document(document, hierarchy, chunk_size=120, chunk_overlap=0)

    assert len(chunks) == 1
    assert chunks[0].text == source
    assert chunks[0].page_start == chunks[0].page_end == 2


def test_chunk_ids_repeat_for_same_input_and_change_with_configuration() -> None:
    document = make_document()
    hierarchy = make_hierarchy(
        ("section-a", "A stable paragraph to chunk.", 1, "Stable")
    )

    first = chunk_document(document, hierarchy, chunk_size=20, chunk_overlap=2)
    second = chunk_document(document, hierarchy, chunk_size=20, chunk_overlap=2)
    changed_settings = chunk_document(
        document, hierarchy, chunk_size=21, chunk_overlap=2
    )

    assert first == second
    assert [chunk.chunk_id for chunk in first] != [
        chunk.chunk_id for chunk in changed_settings
    ]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_token_size_unit_uses_deterministic_lexical_count_or_injected_tokenizer() -> (
    None
):
    document = make_document()
    hierarchy = make_hierarchy(("section-a", "alpha beta gamma delta", 1, "Tokens"))

    lexical_chunks = chunk_document(
        document, hierarchy, chunk_size=2, chunk_overlap=0, size_unit="tokens"
    )
    exact_chunks = chunk_document(
        document,
        hierarchy,
        chunk_size=4,
        chunk_overlap=0,
        size_unit="tokens",
        tokenizer=lambda text: len(text.split()),
    )

    assert [chunk.text for chunk in lexical_chunks] == ["alpha beta", "gamma delta"]
    assert exact_chunks[0].text == "alpha beta gamma delta"
    assert all(chunk.size_unit == "tokens" for chunk in lexical_chunks)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "greater than zero"),
        ({"chunk_size": 10, "chunk_overlap": 10}, "smaller than"),
        ({"size_unit": "bytes"}, "characters.*tokens"),
    ],
)
def test_invalid_chunk_configuration_is_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_document(
            make_document(), make_hierarchy(("section-a", "text", 1, "Title")), **kwargs
        )


def test_rejects_mismatched_document_or_hierarchy_and_missing_source_pages() -> None:
    document = make_document()
    hierarchy = make_hierarchy(("section-a", "text", 1, "Title"))
    with pytest.raises(ValueError, match="identifiers must match"):
        chunk_document(document, HierarchyResult("other", hierarchy.nodes))

    missing_page_node = node(
        "missing-page-content",
        parent_id="section-a",
        level="content",
        title=None,
        path=("Database Manual", "Title"),
        page=4,
        sequence=10,
        text="Text references a page absent from extraction.",
    )
    invalid_hierarchy = HierarchyResult("doc-1", hierarchy.nodes + (missing_page_node,))
    with pytest.raises(ValueError, match="missing extracted page"):
        chunk_document(document, invalid_hierarchy)
