"""Deterministic best-effort hierarchy detection for extracted PDF pages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ingestion.pdf_loader import PdfReadResult


@dataclass(frozen=True)
class HierarchyNode:
    """One document, chapter, section, subsection, page, or content node.

    ``text`` is set only for content nodes. This module describes source
    structure; it does not create retrieval chunks.
    """

    node_id: str
    document_id: str
    parent_id: str | None
    level: str
    section_title: str | None
    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    sequence: int
    detection_method: str
    confidence: float | None = None
    text: str | None = None


@dataclass(frozen=True)
class HierarchyResult:
    """Ordered hierarchy nodes and warnings for one extracted PDF."""

    document_id: str
    nodes: tuple[HierarchyNode, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Heading:
    title: str
    page_number: int
    depth: int
    detection_method: str
    confidence: float
    line_index: int | None = None
    source_order: int = 0


_CHAPTER_RE = re.compile(
    r"^\s*((?:chapter|part)\s+\d+[A-Za-z]?)\s*[:.\-]?\s*(.*?)\s*$",
    re.IGNORECASE,
)
_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})[.)]?\s+(.{2,120}?)\s*$")
_UPPERCASE_RE = re.compile(r"^[A-Z][A-Z0-9 &/,:()'\-]{2,80}$")
_TERMINAL_SENTENCE_RE = re.compile(r"[.!?;:]$")


def detect_hierarchy(document: PdfReadResult) -> HierarchyResult:
    """Detect PDF outlines first, then conservative text headings, then fallback.

    The page-text extraction contract has no reliable font/layout data, so
    this detector does not infer visual typography. Text-derived headings are
    explicitly marked heuristic with their detection method and confidence.
    """
    headings = _headings_from_outlines(document)
    warnings = list(document.warnings)
    if not headings:
        if document.outlines:
            warnings.append("outline_unusable_fell_back_to_text_patterns")
        headings = _headings_from_text(document)
    if not headings:
        warnings.append("no_headings_detected_page_paragraph_fallback_used")
    return _build_hierarchy(document, headings, tuple(dict.fromkeys(warnings)))


def _headings_from_outlines(document: PdfReadResult) -> list[_Heading]:
    headings = [
        _Heading(
            title=outline.title.strip(),
            page_number=outline.page_number,
            depth=max(1, outline.level + 1),
            detection_method="pdf_outline",
            confidence=1.0,
            source_order=outline.order,
        )
        for outline in document.outlines
        if outline.title.strip() and 1 <= outline.page_number <= document.page_count
    ]
    return sorted(headings, key=lambda item: (item.page_number, item.source_order))


def _headings_from_text(document: PdfReadResult) -> list[_Heading]:
    headings: list[_Heading] = []
    source_order = 0
    for page in document.pages:
        for line_index, raw_line in enumerate(page.text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue

            chapter_match = _CHAPTER_RE.fullmatch(line)
            if chapter_match:
                title = " ".join(
                    part for part in chapter_match.groups() if part
                ).strip()
                if title:
                    headings.append(
                        _Heading(
                            title,
                            page.page_number,
                            1,
                            "chapter_pattern",
                            0.9,
                            line_index,
                            source_order,
                        )
                    )
                    source_order += 1
                continue

            numbered_match = _NUMBERED_RE.fullmatch(line)
            if numbered_match and not _TERMINAL_SENTENCE_RE.search(
                numbered_match.group(2).strip()
            ):
                number, title = numbered_match.groups()
                headings.append(
                    _Heading(
                        f"{number} {title.strip()}",
                        page.page_number,
                        min(number.count(".") + 1, 6),
                        "numbered_heading_pattern",
                        0.85,
                        line_index,
                        source_order,
                    )
                )
                source_order += 1
                continue

            if (
                len(line) <= 81
                and " " in line
                and _UPPERCASE_RE.fullmatch(line)
                and not _TERMINAL_SENTENCE_RE.search(line)
            ):
                headings.append(
                    _Heading(
                        line,
                        page.page_number,
                        1,
                        "uppercase_heading_pattern",
                        0.65,
                        line_index,
                        source_order,
                    )
                )
                source_order += 1
    return headings


def _build_hierarchy(
    document: PdfReadResult,
    headings: list[_Heading],
    warnings: tuple[str, ...],
) -> HierarchyResult:
    root_title = str(document.metadata.get("title") or document.source_filename)
    root = _make_node(
        document,
        parent_id=None,
        level="document",
        title=root_title,
        path=(root_title,),
        page_start=1,
        page_end=max(1, document.page_count),
        sequence=0,
        method="document_root",
        confidence=None,
    )
    nodes = [root]
    page_ends = {root.node_id: max(1, document.page_count)}
    sequence = 1
    pages_by_number = {page.page_number: page for page in document.pages}
    headings_by_page: dict[int, list[_Heading]] = {}
    for heading in headings:
        headings_by_page.setdefault(heading.page_number, []).append(heading)
    for page_headings in headings_by_page.values():
        page_headings.sort(
            key=lambda item: (
                item.line_index if item.line_index is not None else item.source_order,
                item.source_order,
            )
        )

    active_sections: list[HierarchyNode] = []
    for page_number in range(1, document.page_count + 1):
        page = pages_by_number.get(page_number)
        if page is None:
            continue

        page_headings = headings_by_page.get(page_number, [])
        if not headings:
            page_node = _make_node(
                document,
                parent_id=root.node_id,
                level="page",
                title=f"Page {page_number}",
                path=(root_title, f"Page {page_number}"),
                page_start=page_number,
                page_end=page_number,
                sequence=sequence,
                method="page_fallback",
                confidence=None,
            )
            nodes.append(page_node)
            page_ends[page_node.node_id] = page_number
            sequence += 1
            active_sections = [page_node]
        else:
            active_sections = [
                node
                for node in active_sections
                if node.level in {"chapter", "section", "subsection"}
            ]

        for heading in (item for item in page_headings if item.line_index is None):
            parent = _parent_for_depth(active_sections, heading.depth, root)
            node = _make_node(
                document,
                parent_id=parent.node_id,
                level=_heading_level(heading),
                title=heading.title,
                path=parent.section_path + (heading.title,),
                page_start=page_number,
                page_end=page_number,
                sequence=sequence,
                method=heading.detection_method,
                confidence=heading.confidence,
            )
            nodes.append(node)
            page_ends[node.node_id] = page_number
            _extend_parent_pages(nodes, page_ends, parent, page_number)
            sequence += 1
            active_sections = _set_active_depth(active_sections, node, heading.depth)

        heading_by_line = {
            heading.line_index: heading
            for heading in page_headings
            if heading.line_index is not None
        }
        paragraphs: list[str] = []
        for line_index, raw_line in enumerate(page.text.splitlines()):
            heading = heading_by_line.get(line_index)
            if heading is not None:
                sequence = _flush_paragraph(
                    paragraphs,
                    page_number,
                    active_sections,
                    root,
                    document,
                    nodes,
                    page_ends,
                    sequence,
                )
                paragraphs = []
                parent = _parent_for_depth(active_sections, heading.depth, root)
                node = _make_node(
                    document,
                    parent_id=parent.node_id,
                    level=_heading_level(heading),
                    title=heading.title,
                    path=parent.section_path + (heading.title,),
                    page_start=page_number,
                    page_end=page_number,
                    sequence=sequence,
                    method=heading.detection_method,
                    confidence=heading.confidence,
                )
                nodes.append(node)
                page_ends[node.node_id] = page_number
                _extend_parent_pages(nodes, page_ends, parent, page_number)
                sequence += 1
                active_sections = _set_active_depth(
                    active_sections, node, heading.depth
                )
            elif raw_line.strip():
                paragraphs.append(raw_line.strip())
            else:
                sequence = _flush_paragraph(
                    paragraphs,
                    page_number,
                    active_sections,
                    root,
                    document,
                    nodes,
                    page_ends,
                    sequence,
                )
                paragraphs = []

        sequence = _flush_paragraph(
            paragraphs,
            page_number,
            active_sections,
            root,
            document,
            nodes,
            page_ends,
            sequence,
        )

    finalized_nodes = tuple(
        HierarchyNode(
            node_id=node.node_id,
            document_id=node.document_id,
            parent_id=node.parent_id,
            level=node.level,
            section_title=node.section_title,
            section_path=node.section_path,
            page_start=node.page_start,
            page_end=max(node.page_start, page_ends.get(node.node_id, node.page_start)),
            sequence=node.sequence,
            detection_method=node.detection_method,
            confidence=node.confidence,
            text=node.text,
        )
        for node in nodes
    )
    return HierarchyResult(document.document_id, finalized_nodes, warnings)


def _flush_paragraph(
    lines: list[str],
    page_number: int,
    active_sections: list[HierarchyNode],
    root: HierarchyNode,
    document: PdfReadResult,
    nodes: list[HierarchyNode],
    page_ends: dict[str, int],
    sequence: int,
) -> int:
    text = " ".join(line for line in lines if line).strip()
    if not text:
        return sequence
    parent = active_sections[-1] if active_sections else root
    content = _make_node(
        document,
        parent_id=parent.node_id,
        level="content",
        title=None,
        path=parent.section_path,
        page_start=page_number,
        page_end=page_number,
        sequence=sequence,
        method="paragraph_content",
        confidence=None,
        text=text,
    )
    nodes.append(content)
    page_ends[content.node_id] = page_number
    _extend_parent_pages(nodes, page_ends, parent, page_number)
    return sequence + 1


def _extend_parent_pages(
    nodes: list[HierarchyNode],
    page_ends: dict[str, int],
    parent: HierarchyNode,
    page_number: int,
) -> None:
    """Extend all ancestors' page spans through a child heading/content node."""
    current: HierarchyNode | None = parent
    while current is not None:
        page_ends[current.node_id] = max(
            page_ends.get(current.node_id, current.page_start), page_number
        )
        current = next(
            (node for node in nodes if node.node_id == current.parent_id), None
        )


def _parent_for_depth(
    active: list[HierarchyNode], depth: int, root: HierarchyNode
) -> HierarchyNode:
    structural = [
        node for node in active if node.level in {"chapter", "section", "subsection"}
    ]
    parent_depth = max(0, depth - 1)
    if parent_depth == 0 or not structural:
        return root
    return structural[min(parent_depth, len(structural)) - 1]


def _set_active_depth(
    active: list[HierarchyNode], node: HierarchyNode, depth: int
) -> list[HierarchyNode]:
    structural = [
        item for item in active if item.level in {"chapter", "section", "subsection"}
    ]
    return structural[: max(0, depth - 1)] + [node]


def _heading_level(heading: _Heading) -> str:
    if heading.detection_method == "chapter_pattern":
        return "chapter"
    return "section" if heading.depth <= 1 else "subsection"


def _make_node(
    document: PdfReadResult,
    *,
    parent_id: str | None,
    level: str,
    title: str | None,
    path: tuple[str, ...],
    page_start: int,
    page_end: int,
    sequence: int,
    method: str,
    confidence: float | None,
    text: str | None = None,
) -> HierarchyNode:
    stable_source = "\x1f".join(
        (
            document.document_id,
            parent_id or "",
            level,
            title or "",
            "/".join(path),
            str(page_start),
            str(sequence),
            text or "",
        )
    )
    node_id = hashlib.sha256(stable_source.encode("utf-8")).hexdigest()
    return HierarchyNode(
        node_id=node_id,
        document_id=document.document_id,
        parent_id=parent_id,
        level=level,
        section_title=title,
        section_path=path,
        page_start=page_start,
        page_end=page_end,
        sequence=sequence,
        detection_method=method,
        confidence=confidence,
        text=text,
    )
