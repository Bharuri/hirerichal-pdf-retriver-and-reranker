"""Local PDF/text discovery and page-aware extraction for the retrieval MVP."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from pypdf import PdfReader


class PdfCorpusError(ValueError):
    """Raised when the configured local PDF corpus cannot be accessed safely."""


class PdfStatus(str, Enum):
    """Extraction outcome for one discovered PDF."""

    EXTRACTED = "extracted"
    EMPTY = "empty"
    PARTIAL = "partial"
    UNREADABLE = "unreadable"
    ENCRYPTED = "encrypted"


@dataclass(frozen=True)
class PdfOutlineEntry:
    """A PDF outline/bookmark item with 1-based page and nesting level."""

    title: str
    page_number: int
    level: int
    order: int


@dataclass(frozen=True)
class PageText:
    """Extracted text and warnings for one 1-based PDF page."""

    page_number: int
    text: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be a 1-based positive integer")


@dataclass(frozen=True)
class PdfReadResult:
    """One PDF's stable identity, metadata, page text, and extraction status."""

    document_id: str
    source_filename: str
    source_path: str
    content_sha256: str | None
    page_count: int
    metadata: Mapping[str, str | None]
    pages: tuple[PageText, ...]
    status: PdfStatus
    warnings: tuple[str, ...] = ()
    outlines: tuple[PdfOutlineEntry, ...] = ()


@dataclass(frozen=True)
class CorpusReadResult:
    """Results of extracting all discovered documents; failures remain per-document."""

    documents: tuple[PdfReadResult, ...]


def discover_pdfs(corpus_dir: str | Path) -> tuple[Path, ...]:
    """Return PDFs under ``corpus_dir`` in stable relative-path order.

    Symlinks resolving outside the configured corpus are ignored. The returned
    paths are canonical paths, so downstream readers operate on the validated
    location rather than the original symlink spelling.
    """
    root = Path(corpus_dir).expanduser().resolve()
    if not root.exists():
        raise PdfCorpusError("Configured PDF corpus directory does not exist.")
    if not root.is_dir():
        raise PdfCorpusError("Configured PDF corpus path is not a directory.")

    discovered: list[tuple[str, Path]] = []
    for candidate in root.rglob("*"):
        if candidate.suffix.lower() != ".pdf" or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            continue
        discovered.append((resolved.relative_to(root).as_posix().casefold(), resolved))

    discovered.sort(key=lambda item: item[0])
    return tuple(path for _, path in discovered)


def discover_documents(corpus_dir: str | Path) -> tuple[Path, ...]:
    """Return supported PDF and UTF-8 text files in stable relative-path order."""
    root = Path(corpus_dir).expanduser().resolve()
    if not root.exists():
        raise PdfCorpusError("Configured document corpus directory does not exist.")
    if not root.is_dir():
        raise PdfCorpusError("Configured document corpus path is not a directory.")

    supported_suffixes = {".pdf", ".txt"}
    discovered: list[tuple[str, Path]] = []
    for candidate in root.rglob("*"):
        if candidate.suffix.lower() not in supported_suffixes or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            continue
        discovered.append((resolved.relative_to(root).as_posix().casefold(), resolved))

    discovered.sort(key=lambda item: item[0])
    return tuple(path for _, path in discovered)


def extract_pdf(pdf_path: str | Path, corpus_dir: str | Path) -> PdfReadResult:
    """Extract one PDF page by page after verifying it is inside the corpus.

    Parser and per-page errors are returned as safe status/warning values so a
    caller can continue with other documents. No absolute path is included in
    the result object.
    """
    root = Path(corpus_dir).expanduser().resolve()
    source = Path(pdf_path).expanduser().resolve()
    if not source.is_relative_to(root):
        raise PdfCorpusError("PDF path is outside the configured corpus directory.")
    if source.suffix.lower() != ".pdf":
        raise PdfCorpusError("Only PDF files can be extracted.")

    try:
        relative_path = source.relative_to(root).as_posix()
    except ValueError as error:
        raise PdfCorpusError("PDF path is outside the configured corpus directory.") from error

    document_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
    try:
        content_sha256 = _file_sha256(source)
    except OSError:
        return PdfReadResult(
            document_id=document_id,
            source_filename=source.name,
            source_path=relative_path,
            content_sha256=None,
            page_count=0,
            metadata=_empty_metadata(),
            pages=(),
            status=PdfStatus.UNREADABLE,
            warnings=("file_read_failed",),
        )

    warnings: list[str] = []
    pages: list[PageText] = []
    outlines: tuple[PdfOutlineEntry, ...] = ()
    try:
        reader = PdfReader(str(source), strict=False)
        page_count = len(reader.pages)
        metadata = _metadata(reader.metadata)
        outlines, outline_warnings = _outlines(reader)
        warnings.extend(outline_warnings)
        if reader.is_encrypted:
            return PdfReadResult(
                document_id,
                source.name,
                relative_path,
                content_sha256,
                page_count,
                metadata,
                (),
                PdfStatus.ENCRYPTED,
                ("encrypted_pdf",),
                outlines,
            )

        for page_number, page in enumerate(reader.pages, start=1):
            page_warnings: list[str] = []
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
                page_warnings.append("page_text_extraction_failed")
                warnings.append(f"page_{page_number}_text_extraction_failed")
            if not text.strip():
                page_warnings.append("no_extractable_text")
                warnings.append(f"page_{page_number}_no_extractable_text")
            elif len(text.strip()) < 20:
                page_warnings.append("little_extractable_text")
                warnings.append(f"page_{page_number}_little_extractable_text")
            pages.append(PageText(page_number, text, tuple(page_warnings)))
    except Exception:
        return PdfReadResult(
            document_id,
            source.name,
            relative_path,
            content_sha256,
            len(pages),
            _empty_metadata(),
            tuple(pages),
            PdfStatus.UNREADABLE,
            tuple(warnings + ["pdf_parse_failed"]),
            outlines,
        )

    has_text = any(page.text.strip() for page in pages)
    if not has_text:
        status = PdfStatus.EMPTY
    elif warnings:
        status = PdfStatus.PARTIAL
    else:
        status = PdfStatus.EXTRACTED
    return PdfReadResult(
        document_id=document_id,
        source_filename=source.name,
        source_path=relative_path,
        content_sha256=content_sha256,
        page_count=page_count,
        metadata=metadata,
        pages=tuple(pages),
        status=status,
        warnings=tuple(warnings),
        outlines=outlines,
    )


def extract_text(text_path: str | Path, corpus_dir: str | Path) -> PdfReadResult:
    """Extract one UTF-8 text file as a single page-like document."""
    root = Path(corpus_dir).expanduser().resolve()
    source = Path(text_path).expanduser().resolve()
    if not source.is_relative_to(root):
        raise PdfCorpusError("Text path is outside the configured corpus directory.")
    if source.suffix.lower() != ".txt":
        raise PdfCorpusError("Only TXT files can be extracted.")

    relative_path = source.relative_to(root).as_posix()
    document_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
    try:
        content_sha256 = _file_sha256(source)
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError):
        return PdfReadResult(
            document_id=document_id,
            source_filename=source.name,
            source_path=relative_path,
            content_sha256=None,
            page_count=0,
            metadata=_empty_metadata(),
            pages=(),
            status=PdfStatus.UNREADABLE,
            warnings=("text_read_failed",),
        )

    warnings: list[str] = []
    page_warnings: list[str] = []
    if not text.strip():
        status = PdfStatus.EMPTY
        page_warnings.append("no_extractable_text")
        warnings.append("text_file_empty")
    elif len(text.strip()) < 20:
        status = PdfStatus.PARTIAL
        page_warnings.append("little_extractable_text")
        warnings.append("text_file_little_text")
    else:
        status = PdfStatus.EXTRACTED

    return PdfReadResult(
        document_id=document_id,
        source_filename=source.name,
        source_path=relative_path,
        content_sha256=content_sha256,
        page_count=1,
        metadata=_empty_metadata(),
        pages=(PageText(1, text, tuple(page_warnings)),),
        status=status,
        warnings=tuple(warnings),
    )


def extract_corpus(corpus_dir: str | Path) -> CorpusReadResult:
    """Extract supported documents independently so one broken file cannot stop the rest."""
    root = Path(corpus_dir).expanduser().resolve()
    paths = discover_documents(root)
    documents = tuple(
        extract_pdf(path, root) if path.suffix.lower() == ".pdf" else extract_text(path, root)
        for path in paths
    )
    return CorpusReadResult(documents)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_metadata() -> dict[str, str | None]:
    return {
        "title": None,
        "author": None,
        "subject": None,
        "creator": None,
        "producer": None,
        "creation_date": None,
        "modification_date": None,
    }


def _metadata(raw_metadata: object) -> dict[str, str | None]:
    fields = {
        "title": "/Title",
        "author": "/Author",
        "subject": "/Subject",
        "creator": "/Creator",
        "producer": "/Producer",
        "creation_date": "/CreationDate",
        "modification_date": "/ModDate",
    }
    result = _empty_metadata()
    if raw_metadata is None:
        return result
    for name, pdf_key in fields.items():
        value = raw_metadata.get(pdf_key)  # type: ignore[attr-defined]
        if value is not None:
            result[name] = str(value)
    return result


def _outlines(reader: PdfReader) -> tuple[tuple[PdfOutlineEntry, ...], tuple[str, ...]]:
    """Extract outline items without allowing a malformed bookmark to fail the PDF."""
    try:
        outline = reader.outline
    except Exception:
        return (), ("outline_read_failed",)

    entries: list[PdfOutlineEntry] = []
    warnings: list[str] = []

    def visit(items: list[object], level: int) -> None:
        for item in items:
            if isinstance(item, list):
                visit(item, level + 1)
                continue
            title = str(getattr(item, "title", "")).strip()
            if not title:
                warnings.append("outline_item_missing_title")
                continue
            try:
                zero_based_page = reader.get_destination_page_number(item)
            except Exception:
                zero_based_page = None
            if zero_based_page is None or zero_based_page < 0:
                warnings.append("outline_item_page_unavailable")
                continue
            entries.append(PdfOutlineEntry(title, zero_based_page + 1, level, len(entries)))

    try:
        visit(outline, 0)
    except Exception:
        warnings.append("outline_read_failed")
    return tuple(entries), tuple(warnings)
