"""Focused tests for TASK-03 document hierarchy detection."""

from ingestion.hierarchy import detect_hierarchy
from ingestion.pdf_loader import PageText, PdfOutlineEntry, PdfReadResult, PdfStatus


def make_document(
    pages: tuple[PageText, ...],
    *,
    outlines: tuple[PdfOutlineEntry, ...] = (),
    document_id: str = "doc-hash",
    title: str | None = "Manual",
) -> PdfReadResult:
    metadata = {"title": title}
    return PdfReadResult(
        document_id=document_id,
        source_filename="manual.pdf",
        source_path="manual.pdf",
        content_sha256="content-hash",
        page_count=len(pages),
        metadata=metadata,
        pages=pages,
        status=PdfStatus.EXTRACTED
        if any(page.text.strip() for page in pages)
        else PdfStatus.EMPTY,
        outlines=outlines,
    )


def test_numbered_headings_build_nested_sections_and_content_nodes() -> None:
    document = make_document(
        (
            PageText(
                1,
                "1 Introduction\n\nPostgreSQL provides reliable data management.\n\n"
                "1.1 Transactions\n\nTransactions group database operations safely.",
            ),
            PageText(2, "\n\nEach transaction has a defined outcome."),
        )
    )

    result = detect_hierarchy(document)
    sections = [
        node for node in result.nodes if node.level in {"section", "subsection"}
    ]
    contents = [node for node in result.nodes if node.level == "content"]

    assert [node.section_title for node in sections] == [
        "1 Introduction",
        "1.1 Transactions",
    ]
    assert sections[0].parent_id == result.nodes[0].node_id
    assert sections[1].parent_id == sections[0].node_id
    assert sections[1].section_path == ("Manual", "1 Introduction", "1.1 Transactions")
    assert [node.sequence for node in result.nodes] == list(range(len(result.nodes)))
    assert len(contents) == 3
    assert contents[0].parent_id == sections[0].node_id
    assert contents[1].parent_id == sections[1].node_id
    assert contents[2].parent_id == sections[1].node_id
    assert contents[2].page_start == 2
    assert sections[1].page_end == 2
    assert all(node.detection_method == "numbered_heading_pattern" for node in sections)
    assert result.warnings == ()


def test_chapter_and_uppercase_patterns_are_marked_as_heuristic() -> None:
    document = make_document(
        (
            PageText(
                1,
                "Chapter 2: Query Processing\n\nThis chapter describes query processing.\n\n"
                "INDEX MAINTENANCE\n\nDatabase indexes improve lookup performance.",
            ),
        )
    )

    result = detect_hierarchy(document)
    headings = [node for node in result.nodes if node.level in {"chapter", "section"}]

    assert [node.section_title for node in headings] == [
        "Chapter 2 Query Processing",
        "INDEX MAINTENANCE",
    ]
    assert headings[0].level == "chapter"
    assert headings[0].detection_method == "chapter_pattern"
    assert headings[0].confidence == 0.9
    assert headings[1].detection_method == "uppercase_heading_pattern"
    assert headings[1].confidence < headings[0].confidence


def test_pdf_outlines_take_precedence_over_text_heading_patterns() -> None:
    document = make_document(
        (
            PageText(
                1, "1 Text heading should not be inferred.\n\nBody under bookmark."
            ),
            PageText(2, "2 Child heading text."),
        ),
        outlines=(
            PdfOutlineEntry("Introduction", 1, 0, 0),
            PdfOutlineEntry("Details", 2, 1, 1),
        ),
    )

    result = detect_hierarchy(document)
    headings = [
        node for node in result.nodes if node.level in {"section", "subsection"}
    ]

    assert [node.section_title for node in headings] == ["Introduction", "Details"]
    assert headings[0].detection_method == "pdf_outline"
    assert headings[1].parent_id == headings[0].node_id
    assert headings[0].confidence == 1.0


def test_no_headings_falls_back_to_page_and_paragraph_nodes() -> None:
    document = make_document(
        (
            PageText(
                1,
                "First paragraph on page one.\nContinued on the same paragraph.\n\nSecond paragraph.",
            ),
            PageText(2, "Text on page two."),
        )
    )

    result = detect_hierarchy(document)
    page_nodes = [node for node in result.nodes if node.level == "page"]
    content_nodes = [node for node in result.nodes if node.level == "content"]

    assert [node.section_title for node in page_nodes] == ["Page 1", "Page 2"]
    assert [node.page_start for node in page_nodes] == [1, 2]
    assert [node.text for node in content_nodes] == [
        "First paragraph on page one. Continued on the same paragraph.",
        "Second paragraph.",
        "Text on page two.",
    ]
    assert content_nodes[0].parent_id == page_nodes[0].node_id
    assert content_nodes[2].parent_id == page_nodes[1].node_id
    assert "no_headings_detected_page_paragraph_fallback_used" in result.warnings
    assert all(node.detection_method == "paragraph_content" for node in content_nodes)


def test_outline_hierarchy_spans_content_pages_and_preserves_order() -> None:
    document = make_document(
        (
            PageText(1, "Evidence in the first page of the section."),
            PageText(2, "Continued evidence on the second page."),
            PageText(3, "A later section begins here."),
        ),
        outlines=(
            PdfOutlineEntry("First Section", 1, 0, 0),
            PdfOutlineEntry("Nested Topic", 2, 1, 1),
            PdfOutlineEntry("Later Section", 3, 0, 2),
        ),
    )

    result = detect_hierarchy(document)
    headings = [
        node for node in result.nodes if node.level in {"section", "subsection"}
    ]
    content = [node for node in result.nodes if node.level == "content"]

    assert [node.section_title for node in headings] == [
        "First Section",
        "Nested Topic",
        "Later Section",
    ]
    assert headings[0].page_end == 2
    assert headings[1].page_end == 2
    assert headings[2].page_start == headings[2].page_end == 3
    assert headings[1].parent_id == headings[0].node_id
    assert headings[2].parent_id == result.nodes[0].node_id
    assert [node.page_start for node in content] == [1, 2, 3]
    assert [node.sequence for node in result.nodes] == sorted(
        node.sequence for node in result.nodes
    )


def test_parent_page_span_includes_child_section_even_without_paragraph_text() -> None:
    document = make_document(
        (
            PageText(1, "Content under the first section."),
            PageText(2, ""),
        ),
        outlines=(
            PdfOutlineEntry("Parent section", 1, 0, 0),
            PdfOutlineEntry("Child section", 2, 1, 1),
        ),
    )

    result = detect_hierarchy(document)
    parent = next(
        node for node in result.nodes if node.section_title == "Parent section"
    )

    assert parent.page_start == 1
    assert parent.page_end == 2


def test_repeated_detection_produces_stable_node_ids_and_order() -> None:
    document = make_document(
        (PageText(1, "1 Introduction\n\nA paragraph with enough source text."),),
    )

    first = detect_hierarchy(document)
    second = detect_hierarchy(document)

    assert first == second
    assert [node.node_id for node in first.nodes] == [
        node.node_id for node in second.nodes
    ]


def test_ambiguous_or_missing_structure_is_fallback_not_claimed_as_heading() -> None:
    document = make_document(
        (
            PageText(
                1,
                "A regular sentence that should not be treated as a heading.\n\n"
                "Another descriptive sentence follows.",
            ),
        ),
        outlines=(PdfOutlineEntry("", 1, 0, 0),),
    )

    result = detect_hierarchy(document)

    assert [node.level for node in result.nodes] == [
        "document",
        "page",
        "content",
        "content",
    ]
    assert all(node.detection_method != "pdf_outline" for node in result.nodes)
    assert "outline_unusable_fell_back_to_text_patterns" in result.warnings
    assert "no_headings_detected_page_paragraph_fallback_used" in result.warnings


def test_empty_pdf_still_has_document_root_without_invented_content() -> None:
    document = make_document((PageText(1, "", ("no_extractable_text",)),))

    result = detect_hierarchy(document)

    assert len(result.nodes) == 2
    assert result.nodes[0].level == "document"
    assert result.nodes[0].section_title == "Manual"
    assert result.nodes[0].text is None
    assert result.nodes[1].level == "page"
    assert result.nodes[1].text is None
