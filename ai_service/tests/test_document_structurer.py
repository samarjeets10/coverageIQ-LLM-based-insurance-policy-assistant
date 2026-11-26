import tempfile
from pathlib import Path
import pytest
import fitz

from ai_service.ingestion.validator import PDFValidator
from ai_service.ingestion.pdf_loader import PyMuPDFLoader
from ai_service.ingestion.document_structurer import DocumentStructurer
from ai_service.schemas.ingestion import (
    BoundingBox,
    ExtractedBlock,
    ExtractedLine,
    ExtractedSpan,
    ExtractedPage,
    ExtractedDocument,
    DocumentMetadata,
    ExtractionMethod,
    PageClassification,
    PageDetectionResult,
    OCRResult,
    OCRTextBox,
    NodeType,
    StructuredDocument,
    StructuredSection,
    StructuredTable,
    StructuredTableCell,
)
from ai_service.schemas.structured_output import ChunkCategory


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


# =============================================================================
# Helper function to generate rich test PDFs
# =============================================================================
def draw_table_on_page(page, start_y: float, rows: list, col_widths: list):
    """Draws a table with borders and text on a fitz page."""
    x_start = 50.0
    row_height = 25.0
    total_width = sum(col_widths)
    num_rows = len(rows)

    # Outer border
    page.draw_rect(fitz.Rect(x_start, start_y, x_start + total_width, start_y + num_rows * row_height))

    # Horizontal lines
    for r in range(1, num_rows):
        y = start_y + r * row_height
        page.draw_line(fitz.Point(x_start, y), fitz.Point(x_start + total_width, y))

    # Vertical lines
    curr_x = x_start
    for w in col_widths[:-1]:
        curr_x += w
        page.draw_line(fitz.Point(curr_x, start_y), fitz.Point(curr_x, start_y + num_rows * row_height))

    # Insert text
    for r_idx, row in enumerate(rows):
        curr_x = x_start
        cell_y = start_y + (r_idx + 1) * row_height - 7.0
        for c_idx, cell_text in enumerate(row):
            if cell_text:
                page.insert_text(fitz.Point(curr_x + 5.0, cell_y), str(cell_text), fontsize=9)
            curr_x += col_widths[c_idx]


# =============================================================================
# Test 1: Numbered Section Detection
# =============================================================================
def test_numbered_section_detection(temp_dir):
    """Scenario 1: Detects numbered sections like 'Section 1', 'Section 2.1'."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 50), "HEALTH POLICY 2026", fontsize=16)
    page.insert_text(fitz.Point(50, 100), "1. Definitions", fontsize=13)
    page.insert_text(fitz.Point(50, 130), "In this policy, words shall have specific meanings.", fontsize=10)
    page.insert_text(fitz.Point(50, 180), "2. Scope of Cover", fontsize=13)
    page.insert_text(fitz.Point(50, 210), "2.1 Hospitalization Expenses", fontsize=11)
    page.insert_text(fitz.Point(50, 240), "Medical expenses incurred during inpatient care are covered.", fontsize=10)

    pdf_path = temp_dir / "numbered_sections.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    clause_refs = [s.clause_reference for s in all_secs if s.clause_reference]

    assert "1" in clause_refs
    assert "2" in clause_refs
    assert "2.1" in clause_refs

    sec_2_1 = next(s for s in all_secs if s.clause_reference == "2.1")
    assert "Hospitalization Expenses" in sec_2_1.title
    assert "Medical expenses incurred" in sec_2_1.content
    assert sec_2_1.node_type == NodeType.SUBSECTION


# =============================================================================
# Test 2: Nested Clause Hierarchy
# =============================================================================
def test_nested_clause_hierarchy(temp_dir):
    """Scenario 2: Parent-child relationship (Major Section -> Section -> Subsection -> Subclause)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 60), "PART I - POLICY TERMS", fontsize=14)
    page.insert_text(fitz.Point(50, 100), "SECTION 2 - COVERAGE", fontsize=12)
    page.insert_text(fitz.Point(50, 140), "2.1 Inpatient Care", fontsize=11)
    page.insert_text(fitz.Point(50, 170), "2.1.1 Room Rent", fontsize=10)
    page.insert_text(fitz.Point(50, 200), "(a) ICU charges up to 2% of sum insured.", fontsize=10)
    page.insert_text(fitz.Point(50, 230), "(b) Specialist doctor consultations.", fontsize=10)

    pdf_path = temp_dir / "nested_hierarchy.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    # Root should contain PART I
    assert len(s_doc.sections) >= 1
    root = s_doc.sections[0]
    assert "PART I" in root.title
    assert root.node_type == NodeType.MAJOR_SECTION

    # Child should be SECTION 2
    assert len(root.children) >= 1
    sec2 = root.children[0]
    assert "SECTION 2" in sec2.title

    # Child of SECTION 2 should be 2.1
    assert len(sec2.children) >= 1
    sec2_1 = sec2.children[0]
    assert "2.1 Inpatient Care" in sec2_1.title

    # Child of 2.1 should be 2.1.1
    assert len(sec2_1.children) >= 1
    subclause = sec2_1.children[0]
    assert "2.1.1 Room Rent" in subclause.title

    # Subclauses (a) and (b) should be children under 2.1.1
    assert len(subclause.children) == 2
    assert "(a)" in subclause.children[0].title
    assert "(b)" in subclause.children[1].title


# =============================================================================
# Test 3: Unnumbered Headings (Font-size & Bold Styling)
# =============================================================================
def test_unnumbered_headings(temp_dir):
    """Scenario 3: Detects unnumbered headings via casing, bold styling, and domain keywords."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Heading 1: Bold, All-Caps
    page.insert_text(fitz.Point(50, 80), "GENERAL EXCLUSIONS", fontsize=12)
    page.insert_text(fitz.Point(50, 110), "The company shall not be liable to make any payment for cosmetic surgery.", fontsize=10)
    # Heading 2: Bold, All-Caps
    page.insert_text(fitz.Point(50, 160), "POLICY CONDITIONS", fontsize=12)
    page.insert_text(fitz.Point(50, 190), "Every notice to the company must be in writing.", fontsize=10)

    pdf_path = temp_dir / "unnumbered_headings.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    titles = [s.title for s in all_secs]

    assert any("GENERAL EXCLUSIONS" in t for t in titles)
    assert any("POLICY CONDITIONS" in t for t in titles)

    excl_sec = next(s for s in all_secs if "GENERAL EXCLUSIONS" in s.title)
    assert excl_sec.category == ChunkCategory.EXCLUSIONS
    assert "cosmetic surgery" in excl_sec.content


# =============================================================================
# Test 4: Font-Size and Body Text Disambiguation
# =============================================================================
def test_font_size_and_body_text_disambiguation(temp_dir):
    """Scenario 4: Large body paragraph with terminal period is NOT misclassified as heading."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 60), "Section 1. Waiting Periods", fontsize=13)
    # Paragraph with 12pt font that is long and ends in period
    long_para = (
        "Under this policy, pre-existing diseases are subject to a mandatory 24-month moratorium "
        "commencing from the date of inception of the policy. Any claim arising during this period "
        "shall be strictly repudiated in accordance with IRDAI guidelines."
    )
    page.insert_text(fitz.Point(50, 100), long_para, fontsize=12)

    pdf_path = temp_dir / "font_disambiguation.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    # The long paragraph should be in content, not a separate section node
    assert len(all_secs) == 1
    assert "Waiting Periods" in all_secs[0].title
    assert "pre-existing diseases are subject" in all_secs[0].content


# =============================================================================
# Test 5: Insurance-Specific Categories
# =============================================================================
def test_insurance_specific_categories(temp_dir):
    """Scenario 5: Accurate mapping of Coverage, Exclusions, Conditions, Definitions, General."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 50), "Section 1: Inpatient Hospitalization Benefits", fontsize=12)
    page.insert_text(fitz.Point(50, 75), "Coverage for room rent and nursing fees.", fontsize=10)

    page.insert_text(fitz.Point(50, 120), "Section 2: 30 Days Waiting Period & Specific Exclusions", fontsize=12)
    page.insert_text(fitz.Point(50, 145), "Excluded treatments include cosmetic and dental care.", fontsize=10)

    page.insert_text(fitz.Point(50, 190), "Section 3: Claims Procedure and Grievance Redressal", fontsize=12)
    page.insert_text(fitz.Point(50, 215), "Submit claim documents within 15 days.", fontsize=10)

    page.insert_text(fitz.Point(50, 260), "Section 4: Meaning of Words and Definitions", fontsize=12)
    page.insert_text(fitz.Point(50, 285), "'Hospital' means any institution established for in-patient care.", fontsize=10)

    pdf_path = temp_dir / "categories.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    cat_by_sec = {s.clause_reference: s.category for s in all_secs if s.clause_reference}

    assert cat_by_sec.get("1") == ChunkCategory.COVERAGE
    assert cat_by_sec.get("2") == ChunkCategory.EXCLUSIONS
    assert cat_by_sec.get("3") == ChunkCategory.CONDITIONS
    assert cat_by_sec.get("4") == ChunkCategory.DEFINITIONS


# =============================================================================
# Test 6: Repeated Headers & Footers Detection and Isolation
# =============================================================================
def test_repeated_headers_and_footers_isolation(temp_dir):
    """Scenario 6: Running headers/footers isolated into headers_and_footers and excluded from clauses."""
    doc = fitz.open()
    for p in range(3):
        page = doc.new_page(width=595, height=842)
        # Header in top margin
        page.insert_text(fitz.Point(50, 30), "STAR HEALTH PREMIER POLICY DOCUMENT", fontsize=8)
        # Body
        page.insert_text(fitz.Point(50, 150), f"Section {p+1}. Core Clause {p+1}", fontsize=12)
        page.insert_text(fitz.Point(50, 180), f"Clause {p+1} operative text.", fontsize=10)
        # Footer in bottom margin
        page.insert_text(fitz.Point(50, 810), f"Page {p+1} of 3", fontsize=8)
        page.insert_text(fitz.Point(400, 810), "UIN: SHAHLIP21023V022021", fontsize=8)

    pdf_path = temp_dir / "running_headers_footers.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    # Verify headers and footers were extracted
    assert len(s_doc.headers_and_footers) >= 3
    hf_types = {hf.item_type for hf in s_doc.headers_and_footers}
    assert "header" in hf_types
    assert "footer" in hf_types

    # Verify header/footer text does NOT pollute narrative section content
    for sec in s_doc.get_all_sections():
        assert "STAR HEALTH PREMIER" not in sec.content
        assert "Page 1 of 3" not in sec.content
        assert "SHAHLIP21023V022021" not in sec.content


# =============================================================================
# Test 7: Simple Table Extraction
# =============================================================================
def test_simple_table_extraction(temp_dir):
    """Scenario 7: Extracts rows, columns, headers, and markdown accurately."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 50), "Section 3. Schedule of Benefits", fontsize=12)

    table_data = [
        ["Benefit Item", "Waiting Period", "Sub-limit"],
        ["Cataract Surgery", "24 Months", "$50,000"],
        ["Joint Replacement", "36 Months", "$150,000"],
    ]
    draw_table_on_page(page, start_y=80.0, rows=table_data, col_widths=[150.0, 120.0, 100.0])

    pdf_path = temp_dir / "simple_table.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    assert len(s_doc.tables) == 1
    tbl = s_doc.tables[0]

    assert tbl.page_number == 1
    assert "Benefit Item" in tbl.headers
    assert "Waiting Period" in tbl.headers
    assert "Sub-limit" in tbl.headers
    assert len(tbl.rows) == 2
    assert "Cataract Surgery" in str(tbl.rows[0])
    assert "| Benefit Item | Waiting Period | Sub-limit |" in tbl.markdown
    assert len(tbl.cells) >= 6


# =============================================================================
# Test 8: Table with Empty Cells
# =============================================================================
def test_table_with_empty_cells(temp_dir):
    """Scenario 8: Handles tables with empty/missing cells without crashing."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 50), "Section 4. Co-payments", fontsize=12)

    table_data = [
        ["Age Band", "Zone 1", "Zone 2"],
        ["Up to 45", "Nil", ""],
        ["46 to 60", "", "10%"],
    ]
    draw_table_on_page(page, start_y=80.0, rows=table_data, col_widths=[120.0, 100.0, 100.0])

    pdf_path = temp_dir / "empty_cells_table.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    assert len(s_doc.tables) == 1
    tbl = s_doc.tables[0]
    assert len(tbl.rows) == 2
    # Verify markdown generates without errors
    assert "| Age Band | Zone 1 | Zone 2 |" in tbl.markdown


# =============================================================================
# Test 9: Multi-Page Table Detection
# =============================================================================
def test_multipage_table_detection(temp_dir):
    """Scenario 9: Flags warnings for contiguous tables across consecutive pages."""
    doc = fitz.open()
    # Page 1: Table near bottom
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(fitz.Point(50, 600), "Section 5. Day Care Procedures", fontsize=12)
    t1_data = [
        ["Procedure Name", "Coverage Category"],
        ["Chemotherapy", "Day Care"],
        ["Radiotherapy", "Day Care"],
    ]
    draw_table_on_page(p1, start_y=630.0, rows=t1_data, col_widths=[200.0, 150.0])

    # Page 2: Table near top
    p2 = doc.new_page(width=595, height=842)
    t2_data = [
        ["Procedure Name", "Coverage Category"],
        ["Dialysis", "Day Care"],
        ["Lithotripsy", "Day Care"],
    ]
    draw_table_on_page(p2, start_y=60.0, rows=t2_data, col_widths=[200.0, 150.0])

    pdf_path = temp_dir / "multipage_table.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    # Assert that continuation tables across pages are merged into 1 unified table
    assert len(s_doc.tables) == 1
    merged_tbl = s_doc.tables[0]
    assert merged_tbl.page_start == 1
    assert merged_tbl.page_end == 2
    assert len(merged_tbl.rows) == 4
    assert ["Chemotherapy", "Day Care"] in merged_tbl.rows
    assert ["Lithotripsy", "Day Care"] in merged_tbl.rows

    # Check warning was logged
    assert any("continuation table" in w.lower() for w in s_doc.extraction_warnings)


# =============================================================================
# Test 10: Table and Narrative Separation
# =============================================================================
def test_table_and_narrative_separation(temp_dir):
    """Scenario 10: Table cells are excluded from narrative clauses (not treated as headings)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 50), "2. Deductible Schedule", fontsize=12)
    page.insert_text(fitz.Point(50, 75), "The following deductibles apply per claim.", fontsize=10)

    table_data = [
        ["Plan Tier", "Deductible Amount"],
        ["Silver", "$1,000"],
        ["Gold", "$500"],
    ]
    draw_table_on_page(page, start_y=100.0, rows=table_data, col_widths=[150.0, 150.0])

    page.insert_text(fitz.Point(50, 220), "Claims below the deductible amount are not payable.", fontsize=10)

    pdf_path = temp_dir / "table_separation.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    titles = [s.title for s in all_secs]

    # "Plan Tier" or "Silver" must NOT be classified as independent section headings
    assert "Plan Tier" not in titles
    assert "Silver" not in titles

    # Table blocks are recorded in table.source_block_ids
    assert len(s_doc.tables) == 1
    assert len(s_doc.tables[0].source_block_ids) >= 1


# =============================================================================
# Test 11: OCR-Derived Text Structuring
# =============================================================================
def test_ocr_derived_text_structuring():
    """Scenario 11: Structures synthetic OCR results with exact page and bbox provenance."""
    metadata = DocumentMetadata(
        file_size_bytes=1024,
        file_hash_sha256="fake_ocr_hash_123",
        page_count=1
    )
    ocr_res = OCRResult(
        page_number=1,
        success=True,
        mean_confidence=0.96,
        boxes=[
            OCRTextBox(text="Section 4. Exclusions", confidence=0.98, bbox=BoundingBox(x0=50, y0=50, x1=200, y1=70)),
            OCRTextBox(text="4.1 Cosmetic Surgery is strictly excluded.", confidence=0.95, bbox=BoundingBox(x0=50, y0=90, x1=400, y1=110)),
        ]
    )

    page = ExtractedPage(
        page_number=1,
        width=595,
        height=842,
        detection=PageDetectionResult(
            page_number=1,
            classification=PageClassification.SCANNED,
            digital_char_count=0,
            block_count=0,
            image_count=1,
            image_coverage_ratio=0.95,
            page_area=500990,
            reason="scanned"
        ),
        blocks=[
            ExtractedBlock(
                block_id="doc_ocr_p1_b0",
                page_number=1,
                raw_block_idx=0,
                reading_order_idx=0,
                block_type="text",
                bbox=BoundingBox(x0=50, y0=50, x1=200, y1=70),
                lines=[ExtractedLine(text="Section 4. Exclusions", bbox=BoundingBox(x0=50, y0=50, x1=200, y1=70))],
                text="Section 4. Exclusions"
            ),
            ExtractedBlock(
                block_id="doc_ocr_p1_b1",
                page_number=1,
                raw_block_idx=1,
                reading_order_idx=1,
                block_type="text",
                bbox=BoundingBox(x0=50, y0=90, x1=400, y1=110),
                lines=[ExtractedLine(text="4.1 Cosmetic Surgery is strictly excluded.", bbox=BoundingBox(x0=50, y0=90, x1=400, y1=110))],
                text="4.1 Cosmetic Surgery is strictly excluded."
            )
        ],
        ocr_result=ocr_res,
        extraction_method=ExtractionMethod.OCR
    )

    doc = ExtractedDocument(doc_id="doc_ocr_001", metadata=metadata, pages=[page])

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(doc)

    all_secs = s_doc.get_all_sections()
    assert len(all_secs) >= 1
    sec4 = all_secs[0]
    assert sec4.clause_reference == "4"
    assert sec4.category == ChunkCategory.EXCLUSIONS
    assert sec4.page_start == 1
    assert "doc_ocr_p1_b0" in sec4.source_block_ids


# =============================================================================
# Test 12: Hybrid Page Structuring
# =============================================================================
def test_hybrid_page_structuring(temp_dir):
    """Scenario 12: Handles hybrid pages with digital narrative and scanned table/raster content."""
    metadata = DocumentMetadata(
        file_size_bytes=2048,
        file_hash_sha256="hybrid_hash_456",
        page_count=1
    )
    page = ExtractedPage(
        page_number=1,
        width=595,
        height=842,
        detection=PageDetectionResult(
            page_number=1,
            classification=PageClassification.HYBRID,
            digital_char_count=150,
            block_count=2,
            image_count=1,
            image_coverage_ratio=0.45,
            page_area=500990,
            reason="hybrid"
        ),
        blocks=[
            ExtractedBlock(
                block_id="hybrid_b0",
                page_number=1,
                raw_block_idx=0,
                reading_order_idx=0,
                block_type="text",
                bbox=BoundingBox(x0=50, y0=50, x1=300, y1=80),
                text="Section 1. Digital Clause\nThis is native vector text."
            ),
            ExtractedBlock(
                block_id="hybrid_b1",
                page_number=1,
                raw_block_idx=1,
                reading_order_idx=1,
                block_type="table",
                bbox=BoundingBox(x0=50, y0=200, x1=400, y1=350),
                text="Tier | Copay\nSilver | 10%\nGold | 5%"
            )
        ],
        extraction_method=ExtractionMethod.HYBRID
    )
    doc = ExtractedDocument(doc_id="doc_hybrid_001", metadata=metadata, pages=[page])

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(doc)

    assert len(s_doc.sections) >= 1
    assert len(s_doc.tables) == 1
    assert s_doc.tables[0].extraction_method == ExtractionMethod.HYBRID
    assert "Silver" in s_doc.tables[0].markdown


# =============================================================================
# Test 13: Page Provenance Preservation
# =============================================================================
def test_page_provenance_preservation(temp_dir):
    """Scenario 13: Preserves exact page_start, page_end, block IDs, and bounding boxes."""
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(fitz.Point(50, 120), "Section 10. Multi-Page Policy Terms", fontsize=12)
    p1.insert_text(fitz.Point(50, 160), "Starting on page 1 with introductory terms.", fontsize=10)

    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(fitz.Point(50, 150), "Continuing clause text on page 2 without new heading.", fontsize=10)

    pdf_path = temp_dir / "provenance.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    sec = s_doc.sections[0]
    assert sec.page_start == 1
    assert sec.page_end == 2
    assert len(sec.source_block_ids) >= 2
    assert len(sec.bounding_boxes) >= 2


# =============================================================================
# Test 14: Uncertain or Malformed Document Fallback
# =============================================================================
def test_uncertain_or_malformed_document_fallback(temp_dir):
    """Scenario 14: Fallback to default Document Body section without crashing on flat text."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 150), "Dear customer, thank you for purchasing our policy.", fontsize=10)
    page.insert_text(fitz.Point(50, 180), "Please find attached the policy schedule and certificate of insurance.", fontsize=10)

    pdf_path = temp_dir / "flat_doc.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    # Must not raise an exception
    assert len(s_doc.sections) == 1
    body_sec = s_doc.sections[0]
    assert "Document Body" in body_sec.title or "Preamble" in body_sec.title
    assert "thank you for purchasing" in body_sec.content
    assert body_sec.category == ChunkCategory.GENERAL


# =============================================================================
# Test 15: Preservation of Original Legal Wording
# =============================================================================
def test_preservation_of_original_legal_wording(temp_dir):
    """Scenario 15: Verbatim preservation of legal text (NO EVIDENCE = NO CLAIM)."""
    exact_legal_text = (
        "Clause 12.3: 'Pre-existing Disease' means any condition, ailment, injury or disease "
        "that is diagnosed by a physician within 48 months prior to the effective date of the policy "
        "issued by the insurer, or for which medical advice or treatment was recommended by, or received "
        "from, a physician within 48 months prior to the effective date."
    )

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 60), "12. Specific Exclusions", fontsize=12)
    page.insert_textbox(fitz.Rect(50, 90, 545, 300), exact_legal_text, fontsize=10)

    pdf_path = temp_dir / "verbatim_text.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)

    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    sec = s_doc.sections[0]
    norm_actual = " ".join(sec.content.split())
    norm_expected = " ".join(exact_legal_text.split())
    assert norm_expected in norm_actual


# =============================================================================
# Test 16: False Footer Detection - Grievance Helpline Protection
# =============================================================================
def test_false_footer_detection_grievance_helpline(temp_dir):
    """Scenario 16: Narrative clause containing helpline or grievance escalation is NOT stripped as footer."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 80), "Section 6. Grievance Redressal Mechanism", fontsize=12)
    clause_text = (
        "For grievance redressal, contact our 24x7 Customer Helpline at 1800-425-2255 or email "
        "grievances@starhealth.in. The grievance escalation procedure is outlined below for all insured persons."
    )
    page.insert_textbox(fitz.Rect(50, 100, 500, 200), clause_text, fontsize=10)

    pdf_path = temp_dir / "grievance_helpline.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    assert len(all_secs) >= 1
    sec = next((s for s in all_secs if s.clause_reference == "6"), all_secs[0])
    assert "24x7 Customer Helpline" in sec.content
    assert "1800-425-2255" in sec.content
    # Confirm it was not sent to headers_and_footers
    assert not any("24x7 Customer Helpline" in hf.text for hf in s_doc.headers_and_footers)


# =============================================================================
# Test 17: False Footer Detection - Complaint Officer Protection
# =============================================================================
def test_false_footer_detection_complaint_officer(temp_dir):
    """Scenario 17: Clause mentioning complaint officer or escalation officer is preserved."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 80), "Section 7. Dispute Escalation", fontsize=12)
    clause_text = (
        "Policyholders may approach the designated Complaint Redressal Officer at the branch "
        "office for escalation of unresolved claims and grievances."
    )
    page.insert_textbox(fitz.Rect(50, 100, 500, 200), clause_text, fontsize=10)

    pdf_path = temp_dir / "complaint_officer.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    assert len(all_secs) >= 1
    assert "Complaint Redressal Officer" in all_secs[0].content
    assert not any("Complaint Redressal Officer" in hf.text for hf in s_doc.headers_and_footers)


# =============================================================================
# Test 18: Standalone Regulatory Footer Isolation
# =============================================================================
def test_standalone_regulatory_footer(temp_dir):
    """Scenario 18: Standalone regulatory footer line at page bottom is isolated."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 100), "Section 1. General", fontsize=12)
    page.insert_text(fitz.Point(50, 130), "General policy operative terms.", fontsize=10)
    # Standalone footer at y=810
    page.insert_text(fitz.Point(50, 810), "IRDAI Regn No: 129 | Policy UIN: SHAHLIP21023V022021", fontsize=8)

    pdf_path = temp_dir / "regulatory_footer.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    assert any("SHAHLIP21023V022021" in hf.text for hf in s_doc.headers_and_footers)
    for sec in s_doc.get_all_sections():
        assert "SHAHLIP21023V022021" not in sec.content


# =============================================================================
# Test 19: OCR Engine Failure Propagation
# =============================================================================
def test_ocr_engine_failure_propagation():
    """Scenario 19: OCR failure is captured in extraction_warnings without silent loss."""
    metadata = DocumentMetadata(file_size_bytes=1024, file_hash_sha256="ocr_fail_hash", page_count=1)
    ocr_res = OCRResult(
        page_number=1,
        success=False,
        error_message="OCR engine memory allocation fault",
        boxes=[]
    )
    page = ExtractedPage(
        page_number=1,
        width=595,
        height=842,
        detection=PageDetectionResult(
            page_number=1,
            classification=PageClassification.SCANNED,
            digital_char_count=0,
            block_count=0,
            image_count=1,
            image_coverage_ratio=0.98,
            page_area=500990,
            reason="scanned"
        ),
        blocks=[],
        ocr_result=ocr_res,
        extraction_method=ExtractionMethod.OCR
    )
    doc = ExtractedDocument(doc_id="doc_fail_001", metadata=metadata, pages=[page])
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(doc)

    assert any("OCR failed on page 1: OCR engine memory allocation fault" in w for w in s_doc.extraction_warnings)


# =============================================================================
# Test 20: Low OCR Confidence Warning
# =============================================================================
def test_low_ocr_confidence_warning():
    """Scenario 20: Low OCR confidence triggers an extraction warning."""
    metadata = DocumentMetadata(file_size_bytes=1024, file_hash_sha256="ocr_low_conf", page_count=1)
    ocr_res = OCRResult(
        page_number=1,
        success=True,
        mean_confidence=0.52,
        boxes=[OCRTextBox(text="Clause 1. Low quality text", confidence=0.52, bbox=BoundingBox(x0=50, y0=50, x1=250, y1=70))]
    )
    page = ExtractedPage(
        page_number=1,
        width=595,
        height=842,
        detection=PageDetectionResult(
            page_number=1,
            classification=PageClassification.SCANNED,
            digital_char_count=0,
            block_count=1,
            image_count=1,
            image_coverage_ratio=0.95,
            page_area=500990,
            reason="scanned"
        ),
        blocks=[
            ExtractedBlock(
                block_id="b0",
                page_number=1,
                raw_block_idx=0,
                reading_order_idx=0,
                block_type="text",
                bbox=BoundingBox(x0=50, y0=50, x1=250, y1=70),
                text="Clause 1. Low quality text"
            )
        ],
        ocr_result=ocr_res,
        extraction_method=ExtractionMethod.OCR
    )
    doc = ExtractedDocument(doc_id="doc_low_001", metadata=metadata, pages=[page])
    structurer = DocumentStructurer(ocr_min_confidence=0.70)
    s_doc = structurer.structure_document(doc)

    assert any("Low OCR confidence (0.52) on page 1" in w for w in s_doc.extraction_warnings)


# =============================================================================
# Test 21: Partial Multi-Page OCR Failure
# =============================================================================
def test_partial_multipage_ocr_failure():
    """Scenario 21: Document with 1 scanned failure page and 1 digital page processes safely."""
    metadata = DocumentMetadata(file_size_bytes=2048, file_hash_sha256="part_fail", page_count=2)
    # Page 1: OCR failed
    p1 = ExtractedPage(
        page_number=1,
        width=595,
        height=842,
        detection=PageDetectionResult(
            page_number=1,
            classification=PageClassification.SCANNED,
            digital_char_count=0,
            block_count=0,
            image_count=1,
            image_coverage_ratio=0.98,
            page_area=500990,
            reason="scanned"
        ),
        blocks=[],
        ocr_result=OCRResult(page_number=1, success=False, error_message="Timeout on page 1", boxes=[]),
        extraction_method=ExtractionMethod.OCR
    )
    # Page 2: Digital
    p2 = ExtractedPage(
        page_number=2,
        width=595,
        height=842,
        detection=PageDetectionResult(
            page_number=2,
            classification=PageClassification.DIGITAL,
            digital_char_count=100,
            block_count=1,
            image_count=0,
            image_coverage_ratio=0.0,
            page_area=500990,
            reason="digital"
        ),
        blocks=[
            ExtractedBlock(
                block_id="p2_b0",
                page_number=2,
                raw_block_idx=0,
                reading_order_idx=0,
                block_type="text",
                bbox=BoundingBox(x0=50, y0=50, x1=300, y1=80),
                text="Section 2. Digital Terms\nValid digital policy text on page 2."
            )
        ],
        extraction_method=ExtractionMethod.DIGITAL
    )
    doc = ExtractedDocument(doc_id="doc_part_001", metadata=metadata, pages=[p1, p2])
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(doc)

    assert any("OCR failed on page 1: Timeout on page 1" in w for w in s_doc.extraction_warnings)
    all_secs = s_doc.get_all_sections()
    assert len(all_secs) >= 1
    assert any("Digital Terms" in s.title for s in all_secs)


# =============================================================================
# Test 22: Heading and Content Separation in Same Block
# =============================================================================
def test_heading_and_content_separation_same_block(temp_dir):
    """Scenario 22: Cleanly separates heading title from inlined clause body in the same block."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Single text block containing both heading and body
    combined_text = (
        "Section 4: AYUSH Treatment\n"
        "Expenses incurred for Inpatient Care treatment under Ayurveda, Yoga and Naturopathy, "
        "Unani, Siddha and Homeopathy systems of medicines are covered up to the limits specified in the Schedule."
    )
    page.insert_textbox(fitz.Rect(50, 50, 500, 200), combined_text, fontsize=10)

    pdf_path = temp_dir / "inlined_heading.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    assert len(all_secs) >= 1
    sec = next((s for s in all_secs if "AYUSH Treatment" in s.title), all_secs[0])
    assert sec.title == "Section 4: AYUSH Treatment"
    assert "Expenses incurred for Inpatient Care treatment under Ayurveda" in sec.content


# =============================================================================
# Test 23: Affirmative Coverage - Pre-existing Diseases
# =============================================================================
def test_affirmative_coverage_pre_existing_diseases(temp_dir):
    """Scenario 23: Affirmative coverage for pre-existing diseases is classified as COVERAGE, not EXCLUSION."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 60), "Section 3. Coverage for Pre-existing Diseases after 24 Months", fontsize=12)
    page.insert_text(fitz.Point(50, 90), "Pre-existing conditions declared in the proposal form are covered after 24 months.", fontsize=10)

    pdf_path = temp_dir / "affirmative_ped.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    assert len(all_secs) >= 1
    sec = next((s for s in all_secs if s.clause_reference == "3"), all_secs[0])
    assert sec.category == ChunkCategory.COVERAGE


# =============================================================================
# Test 24: Waiting Period Waiver Rider Benefit Category
# =============================================================================
def test_waiting_period_waiver_benefit_category(temp_dir):
    """Scenario 24: Waiting period waiver rider is classified as COVERAGE, not EXCLUSION."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 60), "Section 5. Waiting Period Waiver Rider", fontsize=12)
    page.insert_text(fitz.Point(50, 90), "Under this optional add-on, all waiting periods are reduced to nil.", fontsize=10)

    pdf_path = temp_dir / "waiver_rider.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    assert len(all_secs) >= 1
    sec = next((s for s in all_secs if s.clause_reference == "5"), all_secs[0])
    assert sec.category == ChunkCategory.COVERAGE


# =============================================================================
# Test 25: Alphanumeric Clause Reference Preservation (1A, 2.1A)
# =============================================================================
def test_alphanumeric_clause_reference_1a(temp_dir):
    """Scenario 25: Alphanumeric clause references like '1A' and '2.1A' are not truncated."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 60), "Section 1A. Hospital Cash Daily Allowance", fontsize=12)
    page.insert_text(fitz.Point(50, 90), "Daily allowance of Rs 1,000 is payable per day.", fontsize=10)
    page.insert_text(fitz.Point(50, 130), "Clause 2.1A ICU Room Rent Limit", fontsize=12)
    page.insert_text(fitz.Point(50, 160), "ICU room rent is covered up to 2% of Sum Insured.", fontsize=10)

    pdf_path = temp_dir / "alphanumeric_refs.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    refs = [s.clause_reference for s in all_secs if s.clause_reference]
    assert "1A" in refs
    assert "2.1A" in refs


# =============================================================================
# Test 26: Roman Numeral Clause Reference Extraction (I, II, III)
# =============================================================================
def test_roman_numeral_clause_reference_i(temp_dir):
    """Scenario 26: Roman numeral clause references I, II, III are extracted accurately."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 60), "I. DEFINITIONS", fontsize=12)
    page.insert_text(fitz.Point(50, 90), "Policyholder means the insured person named in schedule.", fontsize=10)
    page.insert_text(fitz.Point(50, 130), "II. BENEFITS AND COVERAGE", fontsize=12)
    page.insert_text(fitz.Point(50, 160), "Operative benefits under the policy.", fontsize=10)
    page.insert_text(fitz.Point(50, 200), "III. GENERAL EXCLUSIONS", fontsize=12)
    page.insert_text(fitz.Point(50, 230), "Exclusions applicable to all benefits.", fontsize=10)

    pdf_path = temp_dir / "roman_numerals.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    refs = [s.clause_reference for s in all_secs if s.clause_reference]
    assert "I" in refs
    assert "II" in refs
    assert "III" in refs


# =============================================================================
# Test 27: Table Cell Bounding Box Provenance
# =============================================================================
def test_table_cell_bounding_box_provenance(temp_dir):
    """Scenario 27: Every cell in StructuredTable contains valid BoundingBox coordinates."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 40), "Section 8. Premium Rates", fontsize=12)
    table_data = [
        ["Age Band", "Premium"],
        ["18-35", "5,000"],
        ["36-50", "8,500"],
    ]
    draw_table_on_page(page, start_y=60.0, rows=table_data, col_widths=[120.0, 100.0])

    pdf_path = temp_dir / "cell_bbox.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    assert len(s_doc.tables) >= 1
    tbl = s_doc.tables[0]
    assert len(tbl.cells) >= 6
    for cell in tbl.cells:
        assert cell.bbox is not None
        assert cell.bbox.x1 > cell.bbox.x0
        assert cell.bbox.y1 > cell.bbox.y0
        assert cell.colspan >= 1
        assert cell.rowspan >= 1


# =============================================================================
# Test 28: Merged Cells Colspan and Rowspan Detection
# =============================================================================
def test_merged_cells_colspan_detection():
    """Scenario 28: Validates StructuredTableCell colspan and rowspan attributes."""
    cell = StructuredTableCell(
        row_idx=0,
        col_idx=0,
        text="Coverage Details",
        bbox=BoundingBox(x0=50, y0=50, x1=250, y1=80),
        is_header=True,
        colspan=2,
        rowspan=1
    )
    assert cell.colspan == 2
    assert cell.rowspan == 1
    assert cell.text == "Coverage Details"


# =============================================================================
# Test 29: Scanned OCR Spatial Table Reconstruction
# =============================================================================
def test_scanned_ocr_spatial_table_reconstruction():
    """Scenario 29: Scanned document table is reconstructed from spatial OCR bounding boxes."""
    metadata = DocumentMetadata(file_size_bytes=2048, file_hash_sha256="ocr_tbl_hash", page_count=1)
    ocr_boxes = [
        OCRTextBox(text="Benefit Name", confidence=0.98, bbox=BoundingBox(x0=50, y0=100, x1=150, y1=120)),
        OCRTextBox(text="Max Limit", confidence=0.97, bbox=BoundingBox(x0=200, y0=100, x1=350, y1=120)),
        OCRTextBox(text="ICU Room", confidence=0.96, bbox=BoundingBox(x0=50, y0=130, x1=150, y1=150)),
        OCRTextBox(text="Rs 5,000", confidence=0.95, bbox=BoundingBox(x0=200, y0=130, x1=350, y1=150)),
        OCRTextBox(text="Ambulance", confidence=0.96, bbox=BoundingBox(x0=50, y0=160, x1=150, y1=180)),
        OCRTextBox(text="Rs 2,000", confidence=0.95, bbox=BoundingBox(x0=200, y0=160, x1=350, y1=180)),
    ]
    ocr_res = OCRResult(page_number=1, success=True, mean_confidence=0.96, boxes=ocr_boxes)

    blocks = [
        ExtractedBlock(
            block_id=f"ocr_b{i}",
            page_number=1,
            raw_block_idx=i,
            reading_order_idx=i,
            block_type="text",
            bbox=b.bbox,
            text=b.text
        )
        for i, b in enumerate(ocr_boxes)
    ]

    page = ExtractedPage(
        page_number=1,
        width=595,
        height=842,
        detection=PageDetectionResult(
            page_number=1,
            classification=PageClassification.SCANNED,
            digital_char_count=0,
            block_count=len(blocks),
            image_count=1,
            image_coverage_ratio=0.95,
            page_area=500990,
            reason="scanned"
        ),
        blocks=blocks,
        ocr_result=ocr_res,
        extraction_method=ExtractionMethod.OCR
    )
    doc = ExtractedDocument(doc_id="doc_ocr_tbl", metadata=metadata, pages=[page])
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(doc)

    assert len(s_doc.tables) >= 1
    tbl = s_doc.tables[0]
    assert tbl.page_number == 1
    assert "Benefit Name" in tbl.headers
    assert "Max Limit" in tbl.headers
    assert len(tbl.rows) >= 2
    assert "ICU Room" in str(tbl.rows)


# =============================================================================
# Test 30: Borderless Spatial Table Reconstruction
# =============================================================================
def test_borderless_spatial_table_reconstruction():
    """Scenario 30: Borderless multi-column digital text is detected and structured into a table."""
    metadata = DocumentMetadata(file_size_bytes=2048, file_hash_sha256="borderless_hash", page_count=1)
    # 3 rows of 2 columns
    blocks = [
        ExtractedBlock(
            block_id="b0", page_number=1, raw_block_idx=0, reading_order_idx=0, block_type="text",
            bbox=BoundingBox(x0=50, y0=100, x1=180, y1=120), text="Coverage Type"
        ),
        ExtractedBlock(
            block_id="b1", page_number=1, raw_block_idx=1, reading_order_idx=1, block_type="text",
            bbox=BoundingBox(x0=220, y0=100, x1=350, y1=120), text="Co-Pay Percent"
        ),
        ExtractedBlock(
            block_id="b2", page_number=1, raw_block_idx=2, reading_order_idx=2, block_type="text",
            bbox=BoundingBox(x0=50, y0=130, x1=180, y1=150), text="Zone A"
        ),
        ExtractedBlock(
            block_id="b3", page_number=1, raw_block_idx=3, reading_order_idx=3, block_type="text",
            bbox=BoundingBox(x0=220, y0=130, x1=350, y1=150), text="10%"
        ),
        ExtractedBlock(
            block_id="b4", page_number=1, raw_block_idx=4, reading_order_idx=4, block_type="text",
            bbox=BoundingBox(x0=50, y0=160, x1=180, y1=180), text="Zone B"
        ),
        ExtractedBlock(
            block_id="b5", page_number=1, raw_block_idx=5, reading_order_idx=5, block_type="text",
            bbox=BoundingBox(x0=220, y0=160, x1=350, y1=180), text="20%"
        ),
    ]

    page = ExtractedPage(
        page_number=1,
        width=595,
        height=842,
        detection=PageDetectionResult(
            page_number=1,
            classification=PageClassification.DIGITAL,
            digital_char_count=100,
            block_count=len(blocks),
            image_count=0,
            image_coverage_ratio=0.0,
            page_area=500990,
            reason="digital"
        ),
        blocks=blocks,
        extraction_method=ExtractionMethod.DIGITAL
    )
    doc = ExtractedDocument(doc_id="doc_borderless", metadata=metadata, pages=[page])
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(doc)

    assert len(s_doc.tables) >= 1
    tbl = s_doc.tables[0]
    assert "Coverage Type" in tbl.headers
    assert "Co-Pay Percent" in tbl.headers
    assert len(tbl.rows) == 2


# =============================================================================
# Test 31: Legal Exceptions and Qualifiers Preservation
# =============================================================================
def test_legal_exceptions_and_qualifiers_preservation(temp_dir):
    """Scenario 31: Verbatim preservation of complex legal provisos and qualifiers."""
    proviso_text = (
        "Provided always that the liability of the Company shall not exceed the Sum Insured, "
        "subject to Section 4.2 notwithstanding anything to the contrary contained herein, "
        "and provided further that the Insured Person shall have observed and fulfilled all terms."
    )
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(50, 60), "Section 10. Provisos and Legal Limitations", fontsize=12)
    page.insert_textbox(fitz.Rect(50, 90, 500, 250), proviso_text, fontsize=10)

    pdf_path = temp_dir / "legal_provisos.pdf"
    doc.save(pdf_path)
    doc.close()

    loader = PyMuPDFLoader()
    extracted_doc = loader.load_document(pdf_path)
    structurer = DocumentStructurer()
    s_doc = structurer.structure_document(extracted_doc, pdf_path=pdf_path)

    all_secs = s_doc.get_all_sections()
    sec = next((s for s in all_secs if s.clause_reference == "10"), all_secs[0])
    norm_actual = " ".join(sec.content.split())
    norm_expected = " ".join(proviso_text.split())
    assert norm_expected in norm_actual

