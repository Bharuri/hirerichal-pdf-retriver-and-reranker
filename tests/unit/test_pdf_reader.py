"""Tests for TASK-02 PDF discovery, identity, metadata, and page extraction."""

from pathlib import Path

import pytest

from ingestion.pdf_loader import PdfCorpusError, PdfStatus, discover_pdfs, extract_corpus, extract_pdf


def make_pdf(path: Path, page_texts: list[str], *, title: str | None = None) -> None:
    """Write a small valid PDF with text content without an external fixture."""
    objects: list[bytes] = []
    page_ids = [3 + (2 * page_index) for page_index in range(len(page_texts))]
    font_id = 3 + (2 * len(page_texts))
    info_id = font_id + 1

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode())

    for page_index, text in enumerate(page_texts):
        page_id = page_ids[page_index]
        stream_id = page_id + 1
        escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {stream_id} 0 R >>"
            ).encode()
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    title_entry = f"/Title ({title})" if title else ""
    objects.append(f"<< {title_entry} /Author (Test Author) >>".encode("ascii"))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode())
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf)


def test_discovery_finds_only_pdfs_recursively_in_stable_order(tmp_path: Path) -> None:
    corpus = tmp_path / "doc"
    make_pdf(corpus / "z.pdf", ["last page text that is long enough"])
    make_pdf(corpus / "nested" / "A.PDF", ["first nested page text that is long enough"])
    (corpus / "notes.txt").write_text("not a pdf", encoding="utf-8")

    discovered = discover_pdfs(corpus)

    assert [path.relative_to(corpus).as_posix() for path in discovered] == ["nested/A.PDF", "z.pdf"]


def test_missing_or_non_directory_corpus_has_a_clear_error_without_path_leak(tmp_path: Path) -> None:
    missing = tmp_path / "private" / "missing"
    with pytest.raises(PdfCorpusError, match="does not exist") as error:
        discover_pdfs(missing)
    assert str(missing) not in str(error.value)

    file_path = tmp_path / "not-a-directory"
    file_path.write_text("file", encoding="utf-8")
    with pytest.raises(PdfCorpusError, match="not a directory"):
        discover_pdfs(file_path)


def test_extraction_preserves_one_based_pages_metadata_and_relative_source(tmp_path: Path) -> None:
    corpus = tmp_path / "doc"
    pdf_path = corpus / "postgres" / "manual.pdf"
    make_pdf(
        pdf_path,
        ["PostgreSQL page one contains enough text for extraction.", "Page two also has enough useful text."],
        title="PostgreSQL Manual",
    )

    result = extract_pdf(pdf_path, corpus)

    assert result.status is PdfStatus.EXTRACTED
    assert result.source_filename == "manual.pdf"
    assert result.source_path == "postgres/manual.pdf"
    assert len(result.document_id) == 64
    assert result.content_sha256 is not None and len(result.content_sha256) == 64
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "PostgreSQL page one" in result.pages[0].text
    assert result.metadata["title"] == "PostgreSQL Manual"
    assert result.metadata["author"] == "Test Author"
    assert "C:\\Users" not in result.source_path


def test_missing_pdf_metadata_does_not_fail_extraction(tmp_path: Path) -> None:
    corpus = tmp_path / "doc"
    pdf_path = corpus / "plain.pdf"
    make_pdf(pdf_path, ["A page with enough extracted text for a normal result."])

    result = extract_pdf(pdf_path, corpus)

    assert result.status is PdfStatus.EXTRACTED
    assert result.metadata["title"] is None
    assert result.metadata["author"] == "Test Author"


def test_empty_and_short_text_pages_are_reported_as_warnings(tmp_path: Path) -> None:
    corpus = tmp_path / "doc"
    empty = corpus / "empty.pdf"
    make_pdf(empty, [""])
    short = corpus / "short.pdf"
    make_pdf(short, ["tiny"])

    empty_result = extract_pdf(empty, corpus)
    short_result = extract_pdf(short, corpus)

    assert empty_result.status is PdfStatus.EMPTY
    assert "page_1_no_extractable_text" in empty_result.warnings
    assert short_result.status is PdfStatus.PARTIAL
    assert "page_1_little_extractable_text" in short_result.warnings
    assert short_result.pages[0].warnings == ("little_extractable_text",)


def test_unreadable_pdf_is_reported_and_does_not_stop_other_pdfs(tmp_path: Path) -> None:
    corpus = tmp_path / "doc"
    corpus.mkdir()
    (corpus / "broken.pdf").write_bytes(b"not a valid PDF")
    make_pdf(corpus / "valid.pdf", ["This valid page contains enough extracted text."])

    result = extract_corpus(corpus)

    assert len(result.documents) == 2
    assert result.documents[0].source_filename == "broken.pdf"
    assert result.documents[0].status is PdfStatus.UNREADABLE
    assert "pdf_parse_failed" in result.documents[0].warnings
    assert result.documents[1].status is PdfStatus.EXTRACTED


def test_changed_content_changes_hash_but_not_identity_for_same_relative_file(tmp_path: Path) -> None:
    corpus = tmp_path / "doc"
    pdf_path = corpus / "manual.pdf"
    make_pdf(pdf_path, ["Original document contents are long enough to extract."])
    first = extract_pdf(pdf_path, corpus)
    make_pdf(pdf_path, ["Changed document contents are also long enough to extract."])
    second = extract_pdf(pdf_path, corpus)

    assert first.document_id == second.document_id
    assert first.content_sha256 != second.content_sha256


def test_repeated_extraction_has_stable_identity_hash_and_text(tmp_path: Path) -> None:
    corpus = tmp_path / "doc"
    pdf_path = corpus / "manual.pdf"
    make_pdf(pdf_path, ["A stable page of documentation content for extraction."])

    first = extract_pdf(pdf_path, corpus)
    second = extract_pdf(pdf_path, corpus)

    assert first.document_id == second.document_id
    assert first.content_sha256 == second.content_sha256
    assert first.pages == second.pages


def test_extractor_rejects_paths_outside_corpus_and_non_pdf_files(tmp_path: Path) -> None:
    corpus = tmp_path / "doc"
    corpus.mkdir()
    external_pdf = tmp_path / "private.pdf"
    external_pdf.write_bytes(b"not read")
    with pytest.raises(PdfCorpusError, match="outside"):
        extract_pdf(external_pdf, corpus)

    text_file = corpus / "notes.txt"
    text_file.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(PdfCorpusError, match="Only PDF"):
        extract_pdf(text_file, corpus)
