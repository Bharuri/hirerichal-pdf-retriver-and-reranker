"""Deterministic hierarchy-aware retrieval chunking."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from ingestion.hierarchy import HierarchyNode, HierarchyResult
from ingestion.pdf_loader import PdfReadResult


CHUNKER_VERSION = "1"
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(frozen=True)
class RetrievalChunk:
    """Final source-traceable retrieval chunk.

    ``parent_id`` points to the hierarchy node containing the source text.
    Neighbor IDs link consecutive chunks only when they share a section path.
    """

    chunk_id: str
    document_id: str
    parent_id: str
    level: str
    section_title: str | None
    section_path: tuple[str, ...]
    previous_chunk_id: str | None
    next_chunk_id: str | None
    page_start: int
    page_end: int
    sequence: int
    text: str
    chunk_size: int
    size_unit: str
    token_count: int
    content_hash: str
    configuration_fingerprint: str
    chunker_version: str = CHUNKER_VERSION


@dataclass(frozen=True)
class _TextAtom:
    text: str
    page_number: int
    content_node: HierarchyNode
    separator_before: str


def chunk_document(
    document: PdfReadResult,
    hierarchy: HierarchyResult,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    size_unit: str = "characters",
    tokenizer: Callable[[str], int] | None = None,
) -> tuple[RetrievalChunk, ...]:
    """Build stable final retrieval chunks from extracted pages and hierarchy.

    Content is packed only within the same hierarchy parent and section path.
    ``characters`` applies a strict character limit. ``tokens`` uses the
    supplied tokenizer, or a documented deterministic lexical-token estimate
    (words plus punctuation) when none is supplied.
    """
    if document.document_id != hierarchy.document_id:
        raise ValueError("document and hierarchy identifiers must match")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )
    if size_unit not in {"characters", "tokens"}:
        raise ValueError("size_unit must be 'characters' or 'tokens'")

    measure = _measure_function(size_unit, tokenizer)
    page_numbers = {page.page_number for page in document.pages}
    content_nodes = sorted(
        (
            node
            for node in hierarchy.nodes
            if node.level == "content" and node.text is not None and node.text.strip()
        ),
        key=lambda node: (node.sequence, node.page_start, node.node_id),
    )
    if not content_nodes:
        return ()

    groups: list[list[HierarchyNode]] = []
    for node in content_nodes:
        if node.page_start < 1 or node.page_end < node.page_start:
            raise ValueError(f"content node {node.node_id} has an invalid page range")
        if any(
            page not in page_numbers
            for page in range(node.page_start, node.page_end + 1)
        ):
            raise ValueError(
                f"content node {node.node_id} refers to a missing extracted page"
            )
        if (
            not groups
            or groups[-1][-1].parent_id != node.parent_id
            or groups[-1][-1].section_path != node.section_path
        ):
            groups.append([node])
        else:
            groups[-1].append(node)

    fingerprint = _configuration_fingerprint(
        chunk_size, chunk_overlap, size_unit, tokenizer
    )
    chunks: list[RetrievalChunk] = []
    for group in groups:
        atoms = _atoms_for_group(group, chunk_size, size_unit, measure)
        chunks.extend(
            _chunks_for_group(
                document=document,
                atoms=atoms,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                size_unit=size_unit,
                measure=measure,
                configuration_fingerprint=fingerprint,
                initial_sequence=len(chunks),
            )
        )

    linked = _link_neighbors(chunks)
    identifiers = [chunk.chunk_id for chunk in linked]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("chunk identifiers must be unique")
    return tuple(linked)


def _measure_function(
    size_unit: str, tokenizer: Callable[[str], int] | None
) -> Callable[[str], int]:
    if size_unit == "characters":
        return len
    if tokenizer is not None:
        return tokenizer
    return lambda text: len(_TOKEN_PATTERN.findall(text))


def _configuration_fingerprint(
    chunk_size: int,
    chunk_overlap: int,
    size_unit: str,
    tokenizer: Callable[[str], int] | None,
) -> str:
    tokenizer_identity = "lexical-token-estimate"
    if tokenizer is not None:
        tokenizer_identity = f"{tokenizer.__module__}.{getattr(tokenizer, '__qualname__', type(tokenizer).__name__)}"
    value = f"{CHUNKER_VERSION}|{chunk_size}|{chunk_overlap}|{size_unit}|{tokenizer_identity}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atoms_for_group(
    nodes: list[HierarchyNode],
    chunk_size: int,
    size_unit: str,
    measure: Callable[[str], int],
) -> list[_TextAtom]:
    atoms: list[_TextAtom] = []
    previous_node_id: str | None = None
    for node in nodes:
        text = node.text or ""
        sentences = [
            part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()
        ]
        if not sentences and text.strip():
            sentences = [text.strip()]
        for sentence_index, sentence in enumerate(sentences):
            pieces = _split_oversized(sentence, chunk_size, size_unit, measure)
            for piece_index, piece in enumerate(pieces):
                if piece_index:
                    separator = (
                        ""
                        if size_unit == "characters"
                        and _split_is_contiguous(sentence, pieces)
                        else " "
                    )
                elif sentence_index:
                    separator = " "
                elif previous_node_id is not None and previous_node_id != node.node_id:
                    separator = "\n\n"
                else:
                    separator = ""
                atoms.append(_TextAtom(piece, node.page_start, node, separator))
        previous_node_id = node.node_id
    return atoms


def _split_is_contiguous(original: str, pieces: list[str]) -> bool:
    """Whether pieces came from a single overlong token split by characters."""
    return len(original.split()) == 1 and "".join(pieces) == original


def _split_oversized(
    text: str,
    chunk_size: int,
    size_unit: str,
    measure: Callable[[str], int],
) -> list[str]:
    if measure(text) <= chunk_size:
        return [text]

    words = text.split()
    if len(words) == 1:
        if size_unit == "characters":
            return [
                text[index : index + chunk_size]
                for index in range(0, len(text), chunk_size)
            ]
        pieces: list[str] = []
        current = ""
        for character in text:
            candidate = current + character
            if current and measure(candidate) > chunk_size:
                pieces.append(current)
                current = character
            elif not current and measure(candidate) > chunk_size:
                raise ValueError(
                    "tokenizer counts a single character above the configured chunk size"
                )
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    pieces = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if current and measure(candidate) > chunk_size:
            pieces.extend(_split_oversized(current, chunk_size, size_unit, measure))
            current = word
        else:
            current = candidate
    if current:
        pieces.extend(_split_oversized(current, chunk_size, size_unit, measure))
    return pieces


def _join_atoms(atoms: list[_TextAtom], start: int, end: int) -> str:
    text = atoms[start].text
    for index in range(start + 1, end):
        text += atoms[index].separator_before + atoms[index].text
    return text


def _chunks_for_group(
    *,
    document: PdfReadResult,
    atoms: list[_TextAtom],
    chunk_size: int,
    chunk_overlap: int,
    size_unit: str,
    measure: Callable[[str], int],
    configuration_fingerprint: str,
    initial_sequence: int,
) -> list[RetrievalChunk]:
    chunks: list[RetrievalChunk] = []
    start = 0
    while start < len(atoms):
        end = start
        while (
            end < len(atoms)
            and measure(_join_atoms(atoms, start, end + 1)) <= chunk_size
        ):
            end += 1
        if end == start:
            raise ValueError("an atomic text unit exceeds the configured chunk size")

        selected = atoms[start:end]
        text = _join_atoms(atoms, start, end)
        page_start = min(atom.page_number for atom in selected)
        page_end = max(atom.page_number for atom in selected)
        first_node = selected[0].content_node
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sequence = initial_sequence + len(chunks)
        stable_source = "\x1f".join(
            (
                document.document_id,
                configuration_fingerprint,
                first_node.parent_id or "",
                "/".join(first_node.section_path),
                str(sequence),
                str(page_start),
                str(page_end),
                text,
            )
        )
        chunks.append(
            RetrievalChunk(
                chunk_id=hashlib.sha256(stable_source.encode("utf-8")).hexdigest(),
                document_id=document.document_id,
                parent_id=first_node.parent_id or first_node.node_id,
                level=first_node.level,
                section_title=first_node.section_title,
                section_path=first_node.section_path,
                previous_chunk_id=None,
                next_chunk_id=None,
                page_start=page_start,
                page_end=page_end,
                sequence=sequence,
                text=text,
                chunk_size=measure(text),
                size_unit=size_unit,
                token_count=len(_TOKEN_PATTERN.findall(text)),
                content_hash=content_hash,
                configuration_fingerprint=configuration_fingerprint,
            )
        )

        if end == len(atoms):
            break
        next_start = end
        if chunk_overlap > 0:
            overlap_start = end
            for candidate_start in range(end - 1, start, -1):
                if measure(_join_atoms(atoms, candidate_start, end)) > chunk_overlap:
                    break
                overlap_start = candidate_start
            if overlap_start < end:
                next_start = overlap_start
        start = max(start + 1, next_start)
    return chunks


def _link_neighbors(chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
    linked: list[RetrievalChunk] = []
    for index, chunk in enumerate(chunks):
        previous = chunks[index - 1] if index > 0 else None
        following = chunks[index + 1] if index + 1 < len(chunks) else None
        previous_id = (
            previous.chunk_id
            if previous
            and previous.document_id == chunk.document_id
            and previous.section_path == chunk.section_path
            else None
        )
        next_id = (
            following.chunk_id
            if following
            and following.document_id == chunk.document_id
            and following.section_path == chunk.section_path
            else None
        )
        linked.append(
            RetrievalChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                parent_id=chunk.parent_id,
                level=chunk.level,
                section_title=chunk.section_title,
                section_path=chunk.section_path,
                previous_chunk_id=previous_id,
                next_chunk_id=next_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                sequence=chunk.sequence,
                text=chunk.text,
                chunk_size=chunk.chunk_size,
                size_unit=chunk.size_unit,
                token_count=chunk.token_count,
                content_hash=chunk.content_hash,
                configuration_fingerprint=chunk.configuration_fingerprint,
            )
        )
    return linked
