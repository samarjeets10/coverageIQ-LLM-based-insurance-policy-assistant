import tempfile
from pathlib import Path
import pytest
import fitz

from ai_service.ingestion.ocr_engine import (
    PageScannedDetector,
    PaddleOCREngine,
    MockOCREngine,
    calculate_rectangles_union_area
)
from ai_service.schemas.ingestion import (
    PageClassification,
    ExtractionMethod,
    BoundingBox
)
from ai_service.ingestion.pdf_loader import PyMuPDFLoader


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def create_scanned_pdf_bytes() -> bytes:
    """Generates a PDF page with an embedded full-page raster image and no digital text."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Create solid color pixmap representing a scanned page
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 600), 0)
    pix.set_rect(pix.irect, (240, 240, 240))  # Light grey
    img_bytes = pix.tobytes()

    # Insert image covering 90% of page area
    target_rect = fitz.Rect(50, 50, 545, 792)
    page.insert_image(target_rect, stream=img_bytes)

    data = doc.tobytes()
    doc.close()
    return data


def create_hybrid_pdf_bytes() -> bytes:
    """Generates a PDF page with both rich digital text and a large raster image."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # 1. Digital text (> 150 chars)
    page.insert_text(
        (50, 50),
        "COVERAGEIQ HEALTHCARE POLICY - ENDORSEMENT SCHEDULE AND CONDITIONS.\n"
        "Section 5.1 Pre-existing diseases are covered after 36 months of continuous coverage.\n"
        "Section 5.2 Maternity benefits require 24 months of waiting period.\n"
        "All claims must be submitted within 30 days of hospital discharge."
    )

    # 2. Large raster image (> 40% area)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 400), 0)
    pix.set_rect(pix.irect, (200, 200, 200))
    img_bytes = pix.tobytes()
    page.insert_image(fitz.Rect(50, 200, 500, 750), stream=img_bytes)

    data = doc.tobytes()
    doc.close()
    return data


def test_scanned_page_detection():
    """Test 10: PageScannedDetector correctly flags scanned raster pages."""
    pdf_bytes = create_scanned_pdf_bytes()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    detector = PageScannedDetector()
    result = detector.detect_page(page, page_number=1)
    doc.close()

    assert result.classification == PageClassification.SCANNED
    assert result.page_number == 1
    assert result.image_count >= 1
    assert "Scanned document detected" in result.reason


def test_hybrid_page_detection():
    """Test 11: PageScannedDetector correctly flags hybrid text + raster pages."""
    pdf_bytes = create_hybrid_pdf_bytes()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    detector = PageScannedDetector()
    result = detector.detect_page(page, page_number=1)
    doc.close()

    assert result.classification == PageClassification.HYBRID
    assert result.page_number == 1
    assert result.digital_char_count > 100
    assert result.image_count >= 1
    assert "Hybrid layout detected" in result.reason


def test_ocr_engine_unavailable():
    """Test 12: OCR Engine reports explicit unavailable state without fabricating text."""
    mock_ocr = MockOCREngine(is_available_val=False)
    assert not mock_ocr.is_available()

    doc = fitz.open(stream=create_scanned_pdf_bytes(), filetype="pdf")
    res = mock_ocr.process_page(doc[0], page_number=1)
    doc.close()

    assert not res.success
    assert res.error_message == "Mock OCR Engine unavailable."
    assert res.full_text == ""
    assert len(res.boxes) == 0


def test_ocr_mock_inference():
    """Test 13: Mock OCR Engine returns recognized bounding boxes and confidences."""
    sample_boxes = [
        ("Section 4.1 Cosmetic Exclusions", 0.98, (50.0, 100.0, 350.0, 130.0)),
        ("Plastic surgery is not covered", 0.95, (50.0, 140.0, 320.0, 170.0))
    ]
    mock_ocr = MockOCREngine(mock_boxes=sample_boxes)

    doc = fitz.open(stream=create_scanned_pdf_bytes(), filetype="pdf")
    res = mock_ocr.process_page(doc[0], page_number=1)
    doc.close()

    assert res.success
    assert res.engine == "mock"
    assert len(res.boxes) == 2
    assert res.boxes[0].text == "Section 4.1 Cosmetic Exclusions"
    assert res.boxes[0].confidence == 0.98
    assert res.boxes[0].bbox.x0 == 50.0
    assert res.mean_confidence is not None
    assert 0.96 <= res.mean_confidence <= 0.97


def test_ocr_error_handling():
    """Test 14: OCR Engine handles inference failures gracefully."""
    mock_ocr = MockOCREngine(simulate_error=True)

    doc = fitz.open(stream=create_scanned_pdf_bytes(), filetype="pdf")
    res = mock_ocr.process_page(doc[0], page_number=1)
    doc.close()

    assert not res.success
    assert "Simulated OCR inference failure" in res.error_message
    assert res.boxes == []


def test_loader_scanned_page_with_ocr(temp_dir):
    """End-to-end integration: Loader automatically routes scanned page to OCR engine."""
    scanned_bytes = create_scanned_pdf_bytes()
    scanned_path = temp_dir / "scanned_sample.pdf"
    scanned_path.write_bytes(scanned_bytes)

    sample_boxes = [
        ("Scanned Endorsement Clause 10", 0.94, (50.0, 60.0, 300.0, 90.0))
    ]
    mock_ocr = MockOCREngine(mock_boxes=sample_boxes)

    loader = PyMuPDFLoader(ocr_engine=mock_ocr)
    doc = loader.load_document(scanned_path)

    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert page.detection.classification == PageClassification.SCANNED
    assert page.extraction_method == ExtractionMethod.OCR
    assert page.ocr_result is not None
    assert page.ocr_result.success
    assert page.ocr_result.boxes[0].text == "Scanned Endorsement Clause 10"


def test_overlapping_images_geometric_union():
    """Test geometric union area calculation on overlapping rectangles."""
    # Two identical 100x100 squares at same position: union must be 10,000, not 20,000
    rects_identical = [(10.0, 10.0, 110.0, 110.0), (10.0, 10.0, 110.0, 110.0)]
    area_identical = calculate_rectangles_union_area(rects_identical)
    assert abs(area_identical - 10000.0) < 1e-3

    # Two partially overlapping squares (1600 each, 400 overlap): union = 2800
    rects_overlap = [(10.0, 10.0, 50.0, 50.0), (30.0, 30.0, 70.0, 70.0)]
    area_overlap = calculate_rectangles_union_area(rects_overlap)
    assert abs(area_overlap - 2800.0) < 1e-3

    # Boundary clipping
    rects_clipped = [(0.0, 0.0, 100.0, 100.0)]
    area_clipped = calculate_rectangles_union_area(rects_clipped, boundary=(0.0, 0.0, 50.0, 50.0))
    assert abs(area_clipped - 2500.0) < 1e-3


def test_small_decorative_logos_classified_as_digital():
    """Test digital page with small decorative logo (< 1% area) is classified as DIGITAL, not HYBRID."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # 1. Substantial digital text (> 200 chars)
    page.insert_text((50, 80), "SECTION 2 - IN-PATIENT HOSPITALIZATION COVERAGE")
    page.insert_text((50, 110), "Medical expenses incurred for hospital admission are covered up to Sum Insured.")
    page.insert_text((50, 140), "Pre-hospitalization medical expenses incurred 30 days prior to admission are eligible.")
    page.insert_text((50, 170), "Post-hospitalization medical expenses incurred 60 days post discharge are eligible.")

    # 2. Small logo image (80x30 pt = 2400 pt^2 on 501,000 pt^2 page => ~0.48% coverage)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 80, 30), 0)
    pix.set_rect(pix.irect, (180, 180, 180))
    page.insert_image(fitz.Rect(50, 30, 130, 60), stream=pix.tobytes())

    detector = PageScannedDetector()
    result = detector.detect_page(page, page_number=1)
    doc.close()

    assert result.classification == PageClassification.DIGITAL
    assert result.image_coverage_ratio < 0.05
    assert "Digital page confirmed" in result.reason


def test_ocr_coordinate_transformation_accuracy():
    """Test OCR coordinate inversion and clipping accuracy."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)

    # Region clip
    clip_box = BoundingBox(x0=50.0, y0=100.0, x1=350.0, y1=300.0)
    mock_boxes = [
        ("Clause 1", 0.95, (60.0, 120.0, 200.0, 150.0)),
        ("Clause 2 Outside", 0.90, (10.0, 10.0, 40.0, 40.0))  # Outside clip box
    ]
    engine = MockOCREngine(mock_boxes=mock_boxes)
    result = engine.process_page(page, page_number=1, clip_rect=clip_box)
    doc.close()

    assert result.success
    # Only box inside clip_box should be returned
    assert len(result.boxes) == 1
    box = result.boxes[0]
    assert box.text == "Clause 1"
    assert box.bbox.x0 >= clip_box.x0
    assert box.bbox.y0 >= clip_box.y0
    assert box.bbox.x1 <= clip_box.x1
    assert box.bbox.y1 <= clip_box.y1


def test_empty_ocr_result_handling():
    """Test OCR engine handles empty or zero-sized clipping regions cleanly."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)

    # Empty clip box (x0 == x1)
    empty_clip = BoundingBox(x0=50.0, y0=50.0, x1=50.0, y1=50.0)
    engine = MockOCREngine()
    result = engine.process_page(page, page_number=1, clip_rect=empty_clip)
    doc.close()

    assert result.success
    assert len(result.boxes) == 0
    assert result.full_text == ""


def test_ocr_partial_failure_across_multiple_pages(temp_dir):
    """Test multi-page document where OCR succeeds on Page 1 but fails on Page 2."""
    doc = fitz.open()
    for _ in range(2):
        p = doc.new_page(width=500, height=700)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 400), 0)
        pix.set_rect(pix.irect, (230, 230, 230))
        p.insert_image(fitz.Rect(50, 50, 450, 650), stream=pix.tobytes())

    pdf_path = temp_dir / "multipage_scanned.pdf"
    doc.save(str(pdf_path))
    doc.close()

    # Subclassed Mock engine that fails only on page 2
    class PartialFailureOCREngine(MockOCREngine):
        def process_page(self, page, page_number, dpi=None, clip_rect=None):
            if page_number == 2:
                from ai_service.schemas.ingestion import OCRResult
                return OCRResult(
                    page_number=page_number,
                    success=False,
                    boxes=[],
                    full_text="",
                    mean_confidence=None,
                    error_message="OCR engine timed out on page 2",
                    engine="mock"
                )
            return super().process_page(page, page_number, dpi, clip_rect)

    loader = PyMuPDFLoader(ocr_engine=PartialFailureOCREngine())
    extracted = loader.load_document(pdf_path)

    assert len(extracted.pages) == 2
    # Page 1 succeeded
    assert extracted.pages[0].extraction_method == ExtractionMethod.OCR
    assert extracted.pages[0].ocr_result.success is True

    # Page 2 failed gracefully without crashing document processing
    assert extracted.pages[1].ocr_result is not None
    assert extracted.pages[1].ocr_result.success is False
    assert "timed out" in extracted.pages[1].ocr_result.error_message
