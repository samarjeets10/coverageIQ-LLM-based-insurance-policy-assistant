import os
import tempfile
from pathlib import Path
import pytest
import fitz

from ai_service.ingestion.validator import (
    PDFValidator,
    InvalidPDFSignatureError,
    CorruptedPDFError,
    PasswordProtectedPDFError,
    DocumentTooLargeError,
    EmptyDocumentError,
    PathTraversalError
)
from ai_service.ingestion.pdf_loader import PyMuPDFLoader
from ai_service.schemas.ingestion import (
    ExtractionMethod,
    PageClassification,
    BoundingBox
)
from ai_service.ingestion.ocr_engine import MockOCREngine, PaddleOCREngine


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def create_simple_pdf(text: str = "CoverageIQ Policy Document", pages: int = 1) -> bytes:
    doc = fitz.open()
    for p in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 72), f"{text} - Page {p+1}")
        page.insert_text((50, 120), "Section 2.1 Hospitalization expenses are covered up to Sum Insured.")
    data = doc.tobytes()
    doc.close()
    return data


def test_valid_digital_pdf(temp_dir):
    """Test 1: Valid digital PDF extraction."""
    pdf_bytes = create_simple_pdf(pages=1)
    pdf_path = temp_dir / "valid_policy.pdf"
    pdf_path.write_bytes(pdf_bytes)

    loader = PyMuPDFLoader()
    doc = loader.load_document(pdf_path)

    assert doc.doc_id.startswith("doc_")
    assert doc.metadata.page_count == 1
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert doc.pages[0].extraction_method == ExtractionMethod.DIGITAL
    assert len(doc.pages[0].blocks) >= 1
    assert any("Hospitalization" in b.text for b in doc.pages[0].blocks)


def test_invalid_pdf_signature(temp_dir):
    """Test 2: Invalid PDF signature (not starting with %PDF-)."""
    fake_path = temp_dir / "fake.pdf"
    fake_path.write_bytes(b"This is not a PDF file at all, just plain text.")

    loader = PyMuPDFLoader()
    with pytest.raises(InvalidPDFSignatureError):
        loader.load_document(fake_path)


def test_corrupted_pdf(temp_dir):
    """Test 3: Corrupted PDF file."""
    corrupted_path = temp_dir / "corrupted.pdf"
    corrupted_path.write_bytes(b"%PDF-1.4\ncorrupted header %%EOF with invalid xref table data 9999999")

    loader = PyMuPDFLoader()
    with pytest.raises(CorruptedPDFError):
        loader.load_document(corrupted_path)


def test_empty_pdf(temp_dir):
    """Test 4: Empty PDF (0 bytes or 0 pages)."""
    empty_path = temp_dir / "empty.pdf"
    empty_path.write_bytes(b"")

    loader = PyMuPDFLoader()
    with pytest.raises(EmptyDocumentError):
        loader.load_document(empty_path)


def test_password_protected_pdf(temp_dir):
    """Test 5: Password-protected PDF rejection."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Confidential Insurance Schedule")
    encrypted_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="admin_password",
        user_pw="client_password"
    )
    doc.close()

    enc_path = temp_dir / "encrypted.pdf"
    enc_path.write_bytes(encrypted_bytes)

    loader = PyMuPDFLoader()
    with pytest.raises(PasswordProtectedPDFError):
        loader.load_document(enc_path)


def test_multi_page_pdf(temp_dir):
    """Test 6 & 8: Multi-page PDF and strict page number preservation."""
    pdf_bytes = create_simple_pdf(pages=3)
    pdf_path = temp_dir / "multi_policy.pdf"
    pdf_path.write_bytes(pdf_bytes)

    loader = PyMuPDFLoader()
    doc = loader.load_document(pdf_path)

    assert doc.metadata.page_count == 3
    assert len(doc.pages) == 3

    for idx, page in enumerate(doc.pages):
        expected_page = idx + 1
        assert page.page_number == expected_page
        for block in page.blocks:
            assert block.page_number == expected_page


def test_two_column_layout(temp_dir):
    """Test 7: Geometry-aware two-column layout reading order."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)

    # 1. Top Header (spanning)
    page.insert_text((200, 30), "Policy Header - Universal Terms")

    # 2. Left Column (x: 50, y: 100 to 200)
    page.insert_text((50, 100), "LeftCol 1: In-Patient Hospitalization")
    page.insert_text((50, 150), "LeftCol 2: ICU room rent coverage")

    # 3. Right Column (x: 350, y: 100 to 200)
    page.insert_text((350, 100), "RightCol 1: Cosmetic Exclusions")
    page.insert_text((350, 150), "RightCol 2: Self-inflicted injury")

    two_col_path = temp_dir / "two_column.pdf"
    doc.save(str(two_col_path))
    doc.close()

    loader = PyMuPDFLoader()
    extracted = loader.load_document(two_col_path)
    blocks = extracted.pages[0].blocks

    # In two-column reading order, left column blocks should precede right column blocks
    text_sequence = [b.text for b in blocks if b.text]
    full_text_stream = "\n".join(text_sequence)

    left_idx = full_text_stream.find("In-Patient Hospitalization")
    right_idx = full_text_stream.find("Cosmetic Exclusions")

    assert left_idx != -1
    assert right_idx != -1
    assert left_idx < right_idx, "Left column content must precede right column content in reading order"


def test_bounding_box_extraction(temp_dir):
    """Test 9: Bounding box integrity and coordinates."""
    pdf_bytes = create_simple_pdf(pages=1)
    pdf_path = temp_dir / "bbox_test.pdf"
    pdf_path.write_bytes(pdf_bytes)

    loader = PyMuPDFLoader()
    doc = loader.load_document(pdf_path)
    page = doc.pages[0]

    for block in page.blocks:
        bbox = block.bbox
        assert bbox.x0 <= bbox.x1
        assert bbox.y0 <= bbox.y1
        assert bbox.width >= 0.0
        assert bbox.height >= 0.0
        assert bbox.area >= 0.0

        for line in block.lines:
            assert line.bbox.x0 <= line.bbox.x1
            assert line.bbox.y0 <= line.bbox.y1


def test_path_traversal_prevention(temp_dir):
    """Test path traversal check."""
    restricted_root = temp_dir / "allowed_folder"
    restricted_root.mkdir()

    outside_file = temp_dir / "secret.pdf"
    outside_file.write_bytes(create_simple_pdf())

    loader = PyMuPDFLoader(allowed_root=restricted_root)
    with pytest.raises(PathTraversalError):
        loader.load_document(outside_file)


def test_document_size_limit(temp_dir):
    """Test document size limit enforcement."""
    pdf_bytes = create_simple_pdf()
    pdf_path = temp_dir / "large.pdf"
    pdf_path.write_bytes(pdf_bytes)

    # Enforce tight limit of 10 bytes
    loader = PyMuPDFLoader(max_size_bytes=10)
    with pytest.raises(DocumentTooLargeError):
        loader.load_document(pdf_path)


def test_digital_only_processing_without_loading_paddleocr(temp_dir):
    """Test 15: Digital-only processing does not trigger PaddleOCR model loading."""
    pdf_bytes = create_simple_pdf(pages=2)
    pdf_path = temp_dir / "digital_only.pdf"
    pdf_path.write_bytes(pdf_bytes)

    ocr_adapter = PaddleOCREngine()
    loader = PyMuPDFLoader(ocr_engine=ocr_adapter)

    doc = loader.load_document(pdf_path)

    assert len(doc.pages) == 2
    for p in doc.pages:
        assert p.extraction_method == ExtractionMethod.DIGITAL
        assert p.ocr_result is None

    # Crucial: PaddleOCR should remain uninitialized (self._ocr is None)
    assert ocr_adapter._ocr is None, "PaddleOCR must not be initialized for pure digital PDFs"


def test_rotated_pages_reading_order_and_bounds(temp_dir):
    """Test rotated PDF pages (90, 180, 270) retain valid bounds and extract reading order."""
    doc = fitz.open()
    for rot in [90, 180, 270]:
        page = doc.new_page(width=500, height=700)
        page.set_rotation(rot)
        cx, cy = page.rect.width / 2.0, page.rect.height / 2.0
        page.insert_text((cx - 80, cy - 20), f"Rotated Content at {rot} degrees")
        page.insert_text((cx - 80, cy + 20), "Section 3.1 Maternity Benefit clause")

    rot_path = temp_dir / "rotated.pdf"
    doc.save(str(rot_path))
    doc.close()

    loader = PyMuPDFLoader()
    extracted = loader.load_document(rot_path)

    assert len(extracted.pages) == 3
    for idx, (p, expected_rot) in enumerate(zip(extracted.pages, [90, 180, 270])):
        assert p.rotation == expected_rot
        assert p.page_number == idx + 1
        assert len(p.blocks) >= 1
        for block in p.blocks:
            assert block.bbox.x0 <= block.bbox.x1
            assert block.bbox.y0 <= block.bbox.y1
            assert block.bbox.x1 <= p.width + 1e-3
            assert block.bbox.y1 <= p.height + 1e-3


def test_non_zero_cropbox_origin_handling(temp_dir):
    """Test non-zero CropBox origins (e.g. margin cropped PDFs) are handled safely."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    # Set cropbox to an inset box with non-zero origin
    page.set_cropbox(fitz.Rect(36.0, 54.0, 564.0, 746.0))
    page.insert_text((50, 70), "Cropped Margin Content - Section 1 Coverage")
    page.insert_text((50, 120), "In-Patient hospitalization is covered up to Sum Insured.")

    cropped_path = temp_dir / "cropped.pdf"
    doc.save(str(cropped_path))
    doc.close()

    loader = PyMuPDFLoader()
    extracted = loader.load_document(cropped_path)

    assert len(extracted.pages) == 1
    page_res = extracted.pages[0]
    assert page_res.page_number == 1
    assert len(page_res.blocks) >= 1
    for block in page_res.blocks:
        assert block.bbox.x0 <= block.bbox.x1
        assert block.bbox.y0 <= block.bbox.y1


def test_two_column_with_interleaved_mid_page_heading(temp_dir):
    """Test two-column reading order when full-width headings cut across columns."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=900)

    # 1. Section 1 Heading (Full Width)
    page.insert_text((150, 50), "SECTION 1 - INPATIENT HOSPITALIZATION BENEFITS")

    # 2. Section 1 Left Column
    page.insert_text((50, 120), "Sec1_Left: Room rent capped at 1% of Sum Insured")

    # 3. Section 1 Right Column
    page.insert_text((350, 120), "Sec1_Right: ICU charges capped at 2% of Sum Insured")

    # 4. Section 2 Heading (Full Width, Mid-Page)
    page.insert_text((150, 300), "SECTION 2 - GENERAL POLICY EXCLUSIONS")

    # 5. Section 2 Left Column
    page.insert_text((50, 380), "Sec2_Left: Cosmetic surgery is not covered")

    # 6. Section 2 Right Column
    page.insert_text((350, 380), "Sec2_Right: Intentional self-injury is excluded")

    interleaved_path = temp_dir / "interleaved.pdf"
    doc.save(str(interleaved_path))
    doc.close()

    loader = PyMuPDFLoader()
    extracted = loader.load_document(interleaved_path)
    full_text = "\n".join(b.text for b in extracted.pages[0].blocks if b.text)

    # Validate logical flow: Sec 1 Heading -> Sec 1 Columns -> Sec 2 Heading -> Sec 2 Columns
    pos_sec1_heading = full_text.find("SECTION 1 - INPATIENT")
    pos_sec1_left = full_text.find("Sec1_Left")
    pos_sec1_right = full_text.find("Sec1_Right")
    pos_sec2_heading = full_text.find("SECTION 2 - GENERAL")
    pos_sec2_left = full_text.find("Sec2_Left")
    pos_sec2_right = full_text.find("Sec2_Right")

    assert pos_sec1_heading != -1
    assert pos_sec1_left != -1
    assert pos_sec1_right != -1
    assert pos_sec2_heading != -1
    assert pos_sec2_left != -1
    assert pos_sec2_right != -1

    assert pos_sec1_heading < pos_sec1_left < pos_sec1_right < pos_sec2_heading < pos_sec2_left < pos_sec2_right


def test_hybrid_page_preserves_digital_and_ocr_without_duplication(temp_dir):
    """Test that on a hybrid page, digital text is preserved in blocks while OCR captures image regions without duplication."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # 1. Digital text at top (> 150 chars)
    page.insert_text((50, 50), "COVERAGEIQ POLICY DOCUMENT - SECTION 1 BENEFITS")
    page.insert_text((50, 80), "In-patient hospitalization is covered for minimum 24 hours.")
    page.insert_text((50, 110), "Pre-hospitalization expenses for 30 days and post-hospitalization for 60 days.")

    # 2. Large raster image at bottom (representing scanned table/endorsement schedule, > 40% area)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 450, 550), 0)
    pix.set_rect(pix.irect, (210, 210, 210))
    img_bytes = pix.tobytes()
    image_rect = fitz.Rect(50, 200, 500, 750)
    page.insert_image(image_rect, stream=img_bytes)

    hybrid_path = temp_dir / "hybrid_dedup.pdf"
    doc.save(str(hybrid_path))
    doc.close()

    # Mock OCR engine returning text specifically for the scanned region
    mock_boxes = [
        ("Scanned Endorsement Schedule: Maternity Copay 20%", 0.96, (60.0, 420.0, 450.0, 450.0))
    ]
    mock_ocr = MockOCREngine(mock_boxes=mock_boxes)

    loader = PyMuPDFLoader(ocr_engine=mock_ocr)
    extracted = loader.load_document(hybrid_path)
    page_res = extracted.pages[0]

    assert page_res.detection.classification == PageClassification.HYBRID
    assert page_res.extraction_method == ExtractionMethod.HYBRID

    # Digital text preserved in blocks
    digital_text = "\n".join(b.text for b in page_res.blocks)
    assert "In-patient hospitalization is covered" in digital_text

    # OCR text present in ocr_result
    assert page_res.ocr_result is not None
    assert page_res.ocr_result.success
    ocr_text = page_res.ocr_result.full_text
    assert "Scanned Endorsement Schedule" in ocr_text

    # Digital text must NOT be duplicated in ocr_result
    assert "In-patient hospitalization is covered" not in ocr_text
