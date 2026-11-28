import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set, Union
import fitz

from ai_service.schemas.ingestion import (
    BoundingBox,
    ExtractedSpan,
    ExtractedLine,
    ExtractedBlock,
    ExtractedPage,
    DocumentMetadata,
    ExtractedDocument,
    ExtractionMethod,
    PageClassification,
    NodeType,
    HeaderFooterItem,
    StructuredTableCell,
    StructuredTable,
    StructuredSection,
    StructuredDocument,
)
from ai_service.schemas.structured_output import ChunkCategory

logger = logging.getLogger("coverageiq.ingestion.document_structurer")


# Domain Regex Patterns for Insurance Clauses & Hierarchy
RE_MAJOR_SECTION = re.compile(
    r"^(PART|SCHEDULE)\s+([I|V|X]+|\d+|[A-Z])(\s*[-:–—]\s*(.*))?$",
    re.IGNORECASE
)
RE_SECTION_WORD = re.compile(
    r"^(SECTION|Section)\s+(\d+[A-Za-z]?|[I|V|X]+)(\.?\s*[-:–—]?\s*(.*))?$",
    re.IGNORECASE
)
RE_NUMBERED_SECTION = re.compile(
    r"^(\d+[A-Za-z]?|[I|V|X]+)\.\s+([A-Z].*)$"
)
RE_SUBSECTION = re.compile(
    r"^(\d+\.\d+[A-Za-z]?)\s+([A-Z].*)$"
)
RE_CLAUSE_WORD = re.compile(
    r"^(Clause|CLAUSE)\s+(\d+(\.\d+)*[A-Za-z]?(\([a-z0-9]+\))*?)(\.?\s*[-:–—]?\s*(.*))?$",
    re.IGNORECASE
)
RE_SUBCLAUSE_NUM = re.compile(
    r"^(\d+\.\d+\.\d+[A-Za-z]?)\s+([A-Z].*)$"
)
RE_SUBCLAUSE_LETTER = re.compile(
    r"^\(([a-z]|\d+|[ivx]+)\)\s+(.*)$",
    re.IGNORECASE
)
RE_LIST_BULLET = re.compile(
    r"^[•\-*–—]\s+(.*)$|^\d+\)\s+(.*)$"
)
RE_DEFINITION = re.compile(
    r"^[\"“']?([A-Za-z0-9\s\-]+)[\"”']?\s+(means|shall mean|refers to)\s+(.*)$",
    re.IGNORECASE
)
RE_DEFINITION_COLON = re.compile(
    r"^([A-Za-z0-9\s\-]{2,40})\s*:\s+(.*)$"
)

# Header/Footer Recurring & Standalone Patterns
RE_STANDALONE_PAGE_NUMBER = re.compile(
    r"^(page\s+\d+(\s*(of|/)\s*\d+)?|\d+(\s*(of|/)\s*\d+)?|-\s*\d+\s*-)$",
    re.IGNORECASE
)
RE_STANDALONE_REGULATORY = re.compile(
    r"^(uin|cin|irdai?(\s*regn?\s*no\.?)?|policy\s*(no\.?|uin))\b",
    re.IGNORECASE
)

# Known Insurance Heading Keywords
KNOWN_COVERAGE_KEYWORDS = {
    "coverage", "coverages", "covered", "benefits", "scope of cover", "what is covered",
    "insured events", "in-patient", "hospitalization", "day care", "domiciliary",
    "pre-hospitalization", "post-hospitalization", "sum insured", "cumulative bonus",
    "ambulance", "daily cash", "critical illness", "add-on covers"
}
KNOWN_EXCLUSION_KEYWORDS = {
    "exclusion", "exclusions", "not covered", "what is not covered", "permanent exclusions",
    "waiting period", "waiting periods", "pre-existing", "pre-existing diseases",
    "specific waiting period", "30 days waiting period", "first year exclusion",
    "non-payable", "non-medical", "excluded", "general exclusions", "standard exclusions"
}
KNOWN_CONDITION_KEYWORDS = {
    "condition", "conditions", "general conditions", "terms and conditions", "claim procedure",
    "claims", "claims process", "notice of claim", "cashless", "reimbursement", "cancellation",
    "renewal", "portability", "free look", "free-look", "grace period", "arbitration",
    "subrogation", "fraud", "grievance", "grievance redressal", "ombudsman", "territorial",
    "duties of insured", "statutory regulations"
}
KNOWN_DEFINITION_KEYWORDS = {
    "definition", "definitions", "meaning of words", "words with special meaning", "interpretation"
}


class DocumentStructurer:
    """
    Document Structuring and Table Extraction Engine.
    Transforms raw extracted blocks and OCR data into a typed, hierarchical document
    tree (StructuredDocument) preserving exact legal provenance.
    """

    def __init__(
        self,
        header_margin_ratio: float = 0.12,
        footer_margin_ratio: float = 0.12,
        table_overlap_threshold: float = 0.50,
        min_header_footer_pages: int = 2,
        ocr_min_confidence: float = 0.70
    ):
        self.header_margin_ratio = header_margin_ratio
        self.footer_margin_ratio = footer_margin_ratio
        self.table_overlap_threshold = table_overlap_threshold
        self.min_header_footer_pages = min_header_footer_pages
        self.ocr_min_confidence = ocr_min_confidence

    def structure_document(
        self,
        extracted_doc: ExtractedDocument,
        pdf_path: Optional[Union[str, Path]] = None,
        doc: Optional[fitz.Document] = None
    ) -> StructuredDocument:
        """
        Structures an ExtractedDocument into a StructuredDocument hierarchy.

        Args:
            extracted_doc: ExtractedDocument from PyMuPDFLoader.
            pdf_path: Optional path to source PDF for native table extraction.
            doc: Optional open PyMuPDF fitz.Document instance.
        """
        doc_id = extracted_doc.doc_id
        warnings: List[str] = []
        should_close_doc = False

        if doc is None and pdf_path is not None:
            try:
                doc = fitz.open(pdf_path)
                should_close_doc = True
            except Exception as e:
                logger.warning(f"Could not open PDF file {pdf_path} for native table extraction: {e}")
                warnings.append(f"Failed to open PDF for table extraction: {str(e)}")

        try:
            # Step 0: Inspect and propagate OCR failures / low-confidence warnings
            self._propagate_ocr_diagnostics(extracted_doc, warnings)

            # Step 1: Detect and isolate recurring headers and footers
            headers_and_footers, hf_masked_ids = self._detect_headers_and_footers(extracted_doc)

            # Step 2: Extract structured tables and mask intersecting text blocks
            tables, table_masked_ids, table_warnings = self._extract_tables_and_mask_blocks(
                extracted_doc, doc=doc
            )
            warnings.extend(table_warnings)

            # Step 2b: Merge multi-page continuation tables
            tables = self._merge_continuation_tables(tables, warnings)

            # Masked block IDs = headers/footers + tables
            all_masked_ids: Set[str] = hf_masked_ids.union(table_masked_ids)

            # Step 3: Compute document-level font statistics for relative sizing
            median_body_size = self._compute_median_body_font_size(extracted_doc, all_masked_ids)

            # Step 4: Detect Document Title
            doc_title, title_block_id = self._detect_document_title(
                extracted_doc, all_masked_ids, median_body_size
            )
            if title_block_id:
                all_masked_ids.add(title_block_id)

            # Step 5: Build Hierarchical Clause Tree
            sections, tree_warnings = self._build_hierarchy_tree(
                extracted_doc,
                masked_block_ids=all_masked_ids,
                median_body_size=median_body_size,
                tables=tables
            )
            warnings.extend(tree_warnings)

            # If document has no sections at all, fallback to a single body section
            if not sections:
                sections = self._create_fallback_sections(extracted_doc, all_masked_ids)
                warnings.append("Document structure could not be hierarchically classified; fell back to default section.")

            return StructuredDocument(
                doc_id=doc_id,
                document_title=doc_title,
                metadata=extracted_doc.metadata,
                sections=sections,
                tables=tables,
                headers_and_footers=headers_and_footers,
                extraction_warnings=warnings,
                source_page_count=len(extracted_doc.pages)
            )

        finally:
            if should_close_doc and doc is not None:
                doc.close()

    # -------------------------------------------------------------------------
    # Step 0: OCR Diagnostics & Uncertainty Propagation
    # -------------------------------------------------------------------------
    def _propagate_ocr_diagnostics(
        self,
        extracted_doc: ExtractedDocument,
        warnings: List[str]
    ) -> None:
        """Captures OCR failures and low confidence warnings for strict auditability."""
        for page in extracted_doc.pages:
            p_num = page.page_number
            ocr_res = page.ocr_result

            if ocr_res is not None:
                if not ocr_res.success:
                    err_msg = ocr_res.error_message or "Unknown OCR engine error"
                    msg = f"OCR failed on page {p_num}: {err_msg}"
                    warnings.append(msg)
                    logger.warning(msg)
                elif ocr_res.mean_confidence is not None and ocr_res.mean_confidence < self.ocr_min_confidence:
                    msg = (
                        f"Low OCR confidence ({ocr_res.mean_confidence:.2f}) on page {p_num} "
                        f"(threshold {self.ocr_min_confidence:.2f}) - text extraction may be degraded."
                    )
                    warnings.append(msg)
                    logger.warning(msg)

            # Check scanned page with no text
            if page.detection.classification == PageClassification.SCANNED:
                if not page.blocks and (ocr_res is None or not ocr_res.success):
                    msg = f"Page {p_num}: Scanned page has no extracted text due to OCR failure or missing OCR results."
                    warnings.append(msg)

    # -------------------------------------------------------------------------
    # Step 1: Header and Footer Detection
    # -------------------------------------------------------------------------
    def _detect_headers_and_footers(
        self,
        extracted_doc: ExtractedDocument
    ) -> Tuple[List[HeaderFooterItem], Set[str]]:
        """
        Identifies running headers and footers based on geometric margins,
        common regulatory/page-number patterns, and multi-page text repetition.
        Guards against stripping narrative legal clauses such as grievance helplines.
        """
        headers_and_footers: List[HeaderFooterItem] = []
        masked_block_ids: Set[str] = set()

        header_text_pages: Dict[str, Set[int]] = {}
        footer_text_pages: Dict[str, Set[int]] = {}

        candidate_blocks: List[Tuple[str, int, ExtractedBlock]] = []

        for page in extracted_doc.pages:
            p_height = page.height
            p_num = page.page_number
            top_bound = p_height * self.header_margin_ratio
            bottom_bound = p_height * (1.0 - self.footer_margin_ratio)

            for block in page.blocks:
                b_text = block.text.strip()
                if not b_text:
                    continue

                norm_text = re.sub(r"\s+", " ", b_text.lower())

                # Check header zone
                if block.bbox.y1 <= top_bound:
                    candidate_blocks.append(("header", p_num, block))
                    header_text_pages.setdefault(norm_text, set()).add(p_num)

                # Check footer zone
                elif block.bbox.y0 >= bottom_bound:
                    candidate_blocks.append(("footer", p_num, block))
                    footer_text_pages.setdefault(norm_text, set()).add(p_num)

        for item_type, p_num, block in candidate_blocks:
            b_text = block.text.strip()
            norm_text = re.sub(r"\s+", " ", b_text.lower())
            is_match = False
            confidence = 0.85

            word_count = len(b_text.split())
            has_operative_legal_keywords = any(
                kw in norm_text for kw in (
                    "grievance", "escalation", "complaint", "officer", "ombudsman",
                    "shall", "insured", "insurer", "company", "claim", "coverage", "payable", "terms"
                )
            )

            # 1. Standalone page numbers or standalone regulatory identifiers
            block_lines = [l.strip() for l in b_text.splitlines() if l.strip()]
            is_standalone_page_num = len(b_text) <= 30 and any(RE_STANDALONE_PAGE_NUMBER.match(l) for l in block_lines)
            has_regulatory_keyword = any(kw in norm_text for kw in ("uin:", "irdai", "policy uin", "cin:"))
            is_standalone_regulatory = (
                (len(b_text) <= 120 and (any(RE_STANDALONE_REGULATORY.match(l) for l in block_lines) or has_regulatory_keyword))
                and not has_operative_legal_keywords
                and word_count <= 15
            )

            if is_standalone_page_num or is_standalone_regulatory:
                is_match = True
                confidence = 1.0

            # 2. Repeated exact text across multiple pages in header/footer margin
            # (Do not classify narrative legal text as footer on a single page)
            elif not has_operative_legal_keywords or word_count <= 8:
                if item_type == "header" and len(header_text_pages.get(norm_text, set())) >= self.min_header_footer_pages:
                    is_match = True
                    confidence = 0.95
                elif item_type == "footer" and len(footer_text_pages.get(norm_text, set())) >= self.min_header_footer_pages:
                    is_match = True
                    confidence = 0.95

            if is_match:
                headers_and_footers.append(
                    HeaderFooterItem(
                        item_type=item_type,
                        page_number=p_num,
                        text=b_text,
                        bbox=block.bbox,
                        confidence=confidence
                    )
                )
                masked_block_ids.add(block.block_id)

        return headers_and_footers, masked_block_ids

    # -------------------------------------------------------------------------
    # Step 2: Table Extraction & Masking
    # -------------------------------------------------------------------------
    def _extract_tables_and_mask_blocks(
        self,
        extracted_doc: ExtractedDocument,
        doc: Optional[fitz.Document] = None
    ) -> Tuple[List[StructuredTable], Set[str], List[str]]:
        """
        Extracts tabular content. Uses PyMuPDF native page.find_tables() when doc
        is provided; handles cell-level bounding boxes and merged cells.
        Reconstructs spatial tables from OCR and borderless tabular blocks.
        """
        tables: List[StructuredTable] = []
        table_masked_ids: Set[str] = set()
        warnings: List[str] = []

        if doc is not None:
            # Native PyMuPDF table extraction
            for page_idx, page in enumerate(extracted_doc.pages):
                if page_idx >= len(doc):
                    continue
                fitz_page = doc[page_idx]

                try:
                    tabs = fitz_page.find_tables()

                    for t_idx, tab in enumerate(tabs.tables):
                        t_bbox = BoundingBox(
                            x0=float(tab.bbox[0]),
                            y0=float(tab.bbox[1]),
                            x1=float(tab.bbox[2]),
                            y1=float(tab.bbox[3])
                        )
                        table_id = f"{extracted_doc.doc_id}_p{page.page_number}_tbl{t_idx + 1}"

                        raw_headers = [str(h).strip() if h is not None else "" for h in (tab.header.names or [])]
                        raw_extract = tab.extract() or []
                        raw_rows = tab.rows if hasattr(tab, "rows") else []

                        # Guard against decorative page border frames detected as pseudo-tables:
                        # Full-page rectangular boxes where PyMuPDF treats paragraph text as column headers
                        page_area = page.width * page.height
                        is_full_page_frame = (t_bbox.area / max(1.0, page_area)) >= 0.70
                        has_prose_header = any(len(h) > 200 or len(h.splitlines()) >= 4 for h in raw_headers if h)
                        if is_full_page_frame and has_prose_header:
                            logger.info(f"Skipping decorative page border frame on page {page.page_number} detected as pseudo-table.")
                            continue

                        num_cols = max([len(raw_headers)] + [len(r) for r in raw_extract], default=0)
                        kept_col_indices: List[int] = []
                        for col_i in range(num_cols):
                            hdr_empty = col_i >= len(raw_headers) or raw_headers[col_i] == ""
                            rows_empty = all(
                                col_i >= len(r) or r[col_i] is None or str(r[col_i]).strip() == ""
                                for r in raw_extract
                            )
                            if hdr_empty and rows_empty:
                                col_widths = [
                                    float(rg.cells[col_i][2]) - float(rg.cells[col_i][0])
                                    for rg in raw_rows
                                    if hasattr(rg, "cells") and col_i < len(rg.cells) and rg.cells[col_i] is not None
                                ]
                                # Drop phantom divider sliver columns (double-drawn vector borders <= 6pt wide)
                                # while preserving genuine empty data columns.
                                if col_widths and max(col_widths) <= 6.0:
                                    continue
                            kept_col_indices.append(col_i)

                        if not kept_col_indices and num_cols > 0:
                            kept_col_indices = list(range(num_cols))

                        filtered_headers = [
                            raw_headers[col_i] if col_i < len(raw_headers) else ""
                            for col_i in kept_col_indices
                        ]
                        headers = filtered_headers if any(filtered_headers) else []
                        data_rows: List[List[Optional[str]]] = []

                        start_row_idx = 0
                        if raw_extract and headers:
                            first_row_cleaned = [
                                str(raw_extract[0][col_i]).strip()
                                if (col_i < len(raw_extract[0]) and raw_extract[0][col_i] is not None)
                                else ""
                                for col_i in kept_col_indices
                            ]
                            if first_row_cleaned == headers:
                                start_row_idx = 1

                        for r in raw_extract[start_row_idx:]:
                            cleaned_row = [
                                str(r[col_i]).strip()
                                if (col_i < len(r) and r[col_i] is not None)
                                else None
                                for col_i in kept_col_indices
                            ]
                            if any(c for c in cleaned_row):
                                data_rows.append(cleaned_row)

                        cells: List[StructuredTableCell] = []
                        kept_set = set(kept_col_indices)

                        # Header cells with geometry (preserve original PyMuPDF col mapping after sliver stripping)
                        if headers:
                            hdr_cells = raw_rows[0].cells if (raw_rows and len(raw_rows) > 0) else []
                            for c_idx, h_text in enumerate(headers):
                                orig_col = kept_col_indices[c_idx]
                                cell_bbox = None
                                colspan = 1
                                if orig_col < len(hdr_cells) and hdr_cells[orig_col] is not None:
                                    coords = hdr_cells[orig_col]
                                    cell_bbox = BoundingBox(x0=float(coords[0]), y0=float(coords[1]), x1=float(coords[2]), y1=float(coords[3]))
                                    for next_c in range(orig_col + 1, len(hdr_cells)):
                                        if next_c not in kept_set:
                                            continue
                                        if hdr_cells[next_c] is None:
                                            colspan += 1
                                        else:
                                            break
                                cells.append(
                                    StructuredTableCell(
                                        row_idx=0,
                                        col_idx=c_idx,
                                        text=h_text,
                                        bbox=cell_bbox,
                                        is_header=True,
                                        colspan=colspan,
                                        rowspan=1
                                    )
                                )

                        # Data cells with bounding boxes and colspans
                        for r_offset, r_data in enumerate(data_rows):
                            row_idx = r_offset + (1 if headers else 0)
                            actual_row_idx = r_offset + start_row_idx
                            row_geom = raw_rows[actual_row_idx].cells if actual_row_idx < len(raw_rows) else []

                            for c_idx, c_val in enumerate(r_data):
                                orig_col = kept_col_indices[c_idx]
                                cell_bbox = None
                                colspan = 1
                                if orig_col < len(row_geom) and row_geom[orig_col] is not None:
                                    coords = row_geom[orig_col]
                                    cell_bbox = BoundingBox(x0=float(coords[0]), y0=float(coords[1]), x1=float(coords[2]), y1=float(coords[3]))
                                    for next_c in range(orig_col + 1, len(row_geom)):
                                        if next_c not in kept_set:
                                            continue
                                        if row_geom[next_c] is None:
                                            colspan += 1
                                        else:
                                            break
                                cells.append(
                                    StructuredTableCell(
                                        row_idx=row_idx,
                                        col_idx=c_idx,
                                        text=c_val or "",
                                        bbox=cell_bbox,
                                        is_header=False,
                                        colspan=colspan,
                                        rowspan=1
                                    )
                                )

                        markdown = self._generate_markdown_table(headers, data_rows)

                        matched_block_ids: List[str] = []
                        for b in page.blocks:
                            inter_area = b.bbox.intersection_area(t_bbox)
                            overlap_ratio = inter_area / max(b.bbox.area, 1e-5)
                            center_x = (b.bbox.x0 + b.bbox.x1) / 2.0
                            center_y = (b.bbox.y0 + b.bbox.y1) / 2.0
                            center_inside = (t_bbox.x0 <= center_x <= t_bbox.x1 and t_bbox.y0 <= center_y <= t_bbox.y1)

                            if overlap_ratio >= self.table_overlap_threshold or center_inside:
                                matched_block_ids.append(b.block_id)
                                table_masked_ids.add(b.block_id)

                        structured_table = StructuredTable(
                            table_id=table_id,
                            page_number=page.page_number,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            bbox=t_bbox,
                            headers=headers,
                            rows=data_rows,
                            cells=cells,
                            markdown=markdown.strip(),
                            source_block_ids=matched_block_ids,
                            extraction_method=page.extraction_method,
                            extraction_warnings=[]
                        )
                        tables.append(structured_table)

                except Exception as e:
                    logger.warning(f"Error extracting tables on page {page.page_number}: {e}")
                    warnings.append(f"Table extraction error on page {page.page_number}: {str(e)}")

        # Step 2c: Scanned table extraction from OCR layout or borderless blocks if no vector tables were found
        for page in extracted_doc.pages:
            p_tables = [t for t in tables if t.page_number == page.page_number]
            if not p_tables:
                spatial_table, spatial_masked = self._reconstruct_spatial_table(page, extracted_doc.doc_id)
                if spatial_table:
                    tables.append(spatial_table)
                    table_masked_ids.update(spatial_masked)

        # Fallback for explicit block_type == 'table'
        if not tables:
            for page in extracted_doc.pages:
                for b_idx, block in enumerate(page.blocks):
                    if block.block_type == "table":
                        table_id = f"{extracted_doc.doc_id}_p{page.page_number}_tbl{b_idx + 1}"
                        headers, rows, cells, md = self._parse_table_block_text(block.text)

                        structured_table = StructuredTable(
                            table_id=table_id,
                            page_number=page.page_number,
                            page_start=page.page_number,
                            page_end=page.page_number,
                            bbox=block.bbox,
                            headers=headers,
                            rows=rows,
                            cells=cells,
                            markdown=md,
                            source_block_ids=[block.block_id],
                            extraction_method=page.extraction_method,
                            extraction_warnings=["Table extracted via block fallback."]
                        )
                        tables.append(structured_table)
                        table_masked_ids.add(block.block_id)

        return tables, table_masked_ids, warnings

    def _reconstruct_spatial_table(
        self,
        page: ExtractedPage,
        doc_id: str
    ) -> Tuple[Optional[StructuredTable], Set[str]]:
        """
        Reconstructs tabular structure from 2D spatial alignment of OCR or borderless blocks.
        Ensures ordinary narrative text is never misclassified as a table.
        """
        blocks = [b for b in page.blocks if b.text.strip()]
        if len(blocks) < 4:
            return None, set()

        # Group blocks into rows based on vertical overlap (> 40%)
        rows_grouped: List[List[ExtractedBlock]] = []
        for b in sorted(blocks, key=lambda x: (x.bbox.y0, x.bbox.x0)):
            placed = False
            for r in rows_grouped:
                first = r[0]
                overlap = min(b.bbox.y1, first.bbox.y1) - max(b.bbox.y0, first.bbox.y0)
                min_h = min(b.bbox.height, first.bbox.height)
                if min_h > 0 and (overlap / min_h) >= 0.40:
                    r.append(b)
                    placed = True
                    break
            if not placed:
                rows_grouped.append([b])

        # A valid table requires >= 2 consecutive rows each having >= 2 aligned columns
        table_rows: List[List[ExtractedBlock]] = []
        for r in rows_grouped:
            if len(r) >= 2:
                table_rows.append(r)

        if len(table_rows) < 2:
            return None, set()

        # Check column count consistency
        col_counts = [len(r) for r in table_rows]
        if max(col_counts) < 2 or min(col_counts) < 2:
            return None, set()

        for r in table_rows:
            r.sort(key=lambda x: x.bbox.x0)

        header_row = table_rows[0]
        headers = [b.text.strip() for b in header_row]
        data_rows: List[List[Optional[str]]] = []
        cells: List[StructuredTableCell] = []
        table_blocks: Set[str] = set()

        for c_idx, b in enumerate(header_row):
            cells.append(
                StructuredTableCell(
                    row_idx=0,
                    col_idx=c_idx,
                    text=b.text.strip(),
                    bbox=b.bbox,
                    is_header=True,
                    colspan=1,
                    rowspan=1
                )
            )
            table_blocks.add(b.block_id)

        for r_offset, r in enumerate(table_rows[1:]):
            row_idx = r_offset + 1
            row_vals: List[Optional[str]] = []
            for c_idx, b in enumerate(r):
                row_vals.append(b.text.strip())
                cells.append(
                    StructuredTableCell(
                        row_idx=row_idx,
                        col_idx=c_idx,
                        text=b.text.strip(),
                        bbox=b.bbox,
                        is_header=False,
                        colspan=1,
                        rowspan=1
                    )
                )
                table_blocks.add(b.block_id)
            data_rows.append(row_vals)

        x0 = min(b.bbox.x0 for b in header_row)
        y0 = min(b.bbox.y0 for b in header_row)
        x1 = max(b.bbox.x1 for r in table_rows for b in r)
        y1 = max(b.bbox.y1 for r in table_rows for b in r)
        tbl_bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

        is_ocr = page.extraction_method == ExtractionMethod.OCR or page.detection.classification == PageClassification.SCANNED
        ext_method = ExtractionMethod.OCR if is_ocr else page.extraction_method
        ext_warning = "Table reconstructed from OCR spatial layout." if is_ocr else "Borderless table detected from spatial alignment."

        md = self._generate_markdown_table(headers, data_rows)
        table = StructuredTable(
            table_id=f"{doc_id}_p{page.page_number}_tbl_spatial",
            page_number=page.page_number,
            page_start=page.page_number,
            page_end=page.page_number,
            bbox=tbl_bbox,
            headers=headers,
            rows=data_rows,
            cells=cells,
            markdown=md,
            source_block_ids=list(table_blocks),
            extraction_method=ext_method,
            extraction_warnings=[ext_warning]
        )
        return table, table_blocks

    def _merge_continuation_tables(
        self,
        tables: List[StructuredTable],
        warnings: List[str]
    ) -> List[StructuredTable]:
        """
        Merges consecutive tables across adjacent pages into unified logical tables.
        """
        if len(tables) <= 1:
            return tables

        merged: List[StructuredTable] = []
        i = 0
        while i < len(tables):
            curr = tables[i]

            # Look ahead for multi-page continuation table on next page
            while i + 1 < len(tables):
                next_tbl = tables[i + 1]
                curr_end = curr.page_end or curr.page_number
                next_start = next_tbl.page_start or next_tbl.page_number

                if next_start == curr_end + 1:
                    c1 = len(curr.headers) if curr.headers else (len(curr.rows[0]) if curr.rows else 0)
                    c2 = len(next_tbl.headers) if next_tbl.headers else (len(next_tbl.rows[0]) if next_tbl.rows else 0)

                    # Check column count alignment
                    if c1 > 0 and c1 == c2:
                        start_offset = len(curr.rows) + (1 if curr.headers else 0)
                        curr.rows.extend(next_tbl.rows)

                        for cell in next_tbl.cells:
                            if not cell.is_header:
                                adjusted_cell = StructuredTableCell(
                                    row_idx=cell.row_idx + start_offset - 1,
                                    col_idx=cell.col_idx,
                                    text=cell.text,
                                    bbox=cell.bbox,
                                    is_header=False,
                                    colspan=cell.colspan,
                                    rowspan=cell.rowspan
                                )
                                curr.cells.append(adjusted_cell)

                        curr.page_end = next_tbl.page_end or next_tbl.page_number
                        curr.source_block_ids = list(set(curr.source_block_ids + next_tbl.source_block_ids))
                        curr.markdown = self._generate_markdown_table(curr.headers, curr.rows)
                        msg = f"Merged multi-page continuation table spanning pages {curr.page_start} to {curr.page_end}."
                        curr.extraction_warnings.append(msg)
                        warnings.append(msg)
                        i += 1
                        continue
                break

            merged.append(curr)
            i += 1

        return merged

    def _parse_table_block_text(
        self,
        text: str
    ) -> Tuple[List[str], List[List[Optional[str]]], List[StructuredTableCell], str]:
        """Parses pipe or tab delimited table block text."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return [], [], [], ""

        headers: List[str] = []
        rows: List[List[Optional[str]]] = []
        cells: List[StructuredTableCell] = []

        is_pipe_table = any("|" in line for line in lines)
        delimiter = "|" if is_pipe_table else "\t"

        for line_idx, line in enumerate(lines):
            if is_pipe_table and re.match(r"^\|?(\s*:?-+:?\s*\|?)+$", line):
                continue

            parts = [p.strip() for p in line.split(delimiter)]
            if is_pipe_table:
                if parts and parts[0] == "":
                    parts.pop(0)
                if parts and parts[-1] == "":
                    parts.pop()

            if line_idx == 0:
                headers = parts
                for c_idx, h in enumerate(headers):
                    cells.append(StructuredTableCell(row_idx=0, col_idx=c_idx, text=h, is_header=True))
            else:
                row_vals = [p if p else None for p in parts]
                rows.append(row_vals)
                r_num = len(rows)
                for c_idx, val in enumerate(row_vals):
                    cells.append(StructuredTableCell(row_idx=r_num, col_idx=c_idx, text=val or "", is_header=False))

        md = self._generate_markdown_table(headers, rows)
        return headers, rows, cells, md

    def _generate_markdown_table(
        self,
        headers: List[str],
        rows: List[List[Optional[str]]]
    ) -> str:
        """Generates clean GitHub-Flavored Markdown table text."""
        if not headers and not rows:
            return ""
        cols = max(len(headers), max((len(r) for r in rows), default=0))
        if cols == 0:
            return ""

        h_padded = [headers[i] if i < len(headers) and headers[i] else f"Col {i+1}" for i in range(cols)]
        md_lines = [
            "| " + " | ".join(h_padded) + " |",
            "| " + " | ".join(["---"] * cols) + " |"
        ]
        for row in rows:
            r_padded = [(str(row[i]).strip() if i < len(row) and row[i] is not None else "") for i in range(cols)]
            md_lines.append("| " + " | ".join(r_padded) + " |")

        return "\n".join(md_lines)

    # -------------------------------------------------------------------------
    # Step 3: Document Title Detection
    # -------------------------------------------------------------------------
    def _detect_document_title(
        self,
        extracted_doc: ExtractedDocument,
        masked_ids: Set[str],
        median_body_size: float
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Detects primary document title from early pages (page 1 or 2).
        Returns (document_title, block_id).
        """
        if not extracted_doc.pages:
            return None, None

        candidates: List[Tuple[float, str, str]] = []

        for page in extracted_doc.pages[:2]:
            for block in page.blocks:
                if block.block_id in masked_ids or not block.text.strip():
                    continue

                text = block.text.strip()
                if RE_NUMBERED_SECTION.match(text) or RE_SUBSECTION.match(text) or RE_MAJOR_SECTION.match(text):
                    continue

                if len(text.splitlines()) > 3 or len(text) > 150:
                    continue
                if block.bbox.y0 > page.height * 0.45:
                    continue

                avg_size, is_bold = self._get_block_font_stats(block)
                if avg_size >= median_body_size * 1.25 or (is_bold and avg_size >= median_body_size * 1.10):
                    score = avg_size + (4.0 if is_bold else 0.0) - (block.bbox.y0 * 0.01)
                    candidates.append((score, text, block.block_id))

        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            best = candidates[0]
            return best[1], best[2]

        if extracted_doc.metadata.title:
            return extracted_doc.metadata.title.strip(), None

        return None, None

    # -------------------------------------------------------------------------
    # Step 4: Font Statistics & Multi-Signal Classification
    # -------------------------------------------------------------------------
    def _compute_median_body_font_size(
        self,
        extracted_doc: ExtractedDocument,
        masked_ids: Set[str]
    ) -> float:
        """Computes the median font size of body text spans across the document."""
        sizes: List[float] = []
        for page in extracted_doc.pages:
            for block in page.blocks:
                if block.block_id in masked_ids:
                    continue
                for line in block.lines:
                    for span in line.spans:
                        if span.text.strip() and span.font_size > 0:
                            sizes.append(span.font_size)

        if not sizes:
            return 10.0
        sizes.sort()
        mid = len(sizes) // 2
        return sizes[mid] if len(sizes) % 2 != 0 else (sizes[mid - 1] + sizes[mid]) / 2.0

    def _get_block_font_stats(self, block: ExtractedBlock) -> Tuple[float, bool]:
        """Calculates average font size and whether the block is predominantly bold."""
        total_len = 0
        weighted_size = 0.0
        bold_len = 0

        for line in block.lines:
            for span in line.spans:
                s_len = len(span.text.strip())
                if s_len > 0:
                    total_len += s_len
                    weighted_size += span.font_size * s_len
                    if span.is_bold:
                        bold_len += s_len

        if total_len == 0:
            return 10.0, False

        avg_size = weighted_size / total_len
        is_bold = (bold_len / total_len) >= 0.50
        return avg_size, is_bold

    def _classify_block(
        self,
        block: ExtractedBlock,
        median_body_size: float
    ) -> Tuple[NodeType, int, Optional[str], Optional[ChunkCategory]]:
        """
        Multi-signal block classifier.
        Returns:
            (node_type, hierarchy_level, clause_reference, chunk_category)
        """
        text = block.text.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        first_line = lines[0] if lines else text

        avg_size, is_bold = self._get_block_font_stats(block)
        is_all_caps = text.isupper() and len(text.split()) <= 15
        is_short = len(lines) <= 2 and len(text) <= 120
        ends_with_terminal_period = text.endswith(".") and not text.endswith("..")

        is_first_line_short = len(first_line) <= 120 and len(first_line.split()) <= 15

        # 1. Regex: Major Section (PART I, SCHEDULE A, PART 1)
        m_major = RE_MAJOR_SECTION.match(first_line)
        if m_major and is_first_line_short:
            prefix = m_major.group(1).upper()
            num = m_major.group(2)
            clause_ref = f"{prefix} {num}"
            category = self._infer_category(first_line) or self._infer_category(text)
            return NodeType.MAJOR_SECTION, 1, clause_ref, category

        # 2. Regex: Section Word (SECTION 2 - COVERAGE, Section 1A)
        m_sec_word = RE_SECTION_WORD.match(first_line)
        if m_sec_word and is_first_line_short:
            clause_ref = m_sec_word.group(2)
            category = self._infer_category(first_line) or self._infer_category(text)
            return NodeType.SECTION, 2, clause_ref, category

        # 3. Regex: Numbered Section (e.g. 1. Definitions, 1A. Extended Benefits, I. PRELIMINARY)
        m_sec = RE_NUMBERED_SECTION.match(first_line)
        if m_sec and is_first_line_short:
            clause_ref = m_sec.group(1)
            category = self._infer_category(first_line) or self._infer_category(text)
            return NodeType.SECTION, 2, clause_ref, category

        # 4. Regex: Subsection (e.g. 2.1 Inpatient Care, 2.1A Cataract)
        m_sub = RE_SUBSECTION.match(first_line)
        if m_sub and is_first_line_short:
            clause_ref = m_sub.group(1)
            category = self._infer_category(first_line) or self._infer_category(text)
            return NodeType.SUBSECTION, 3, clause_ref, category

        # 5. Regex: Clause Word (e.g. Clause 4.1 Waiting Periods, Clause 4A, Clause 4(a))
        m_clause = RE_CLAUSE_WORD.match(first_line)
        if m_clause and is_first_line_short:
            clause_ref = m_clause.group(2)
            category = self._infer_category(first_line) or self._infer_category(text)
            return NodeType.CLAUSE, 3, clause_ref, category

        # 6. Regex: Subclause Number (e.g. 2.1.1 Room Rent, 2.1.1A ICU)
        m_subclause_num = RE_SUBCLAUSE_NUM.match(first_line)
        if m_subclause_num and is_first_line_short:
            clause_ref = m_subclause_num.group(1)
            category = self._infer_category(first_line) or self._infer_category(text)
            return NodeType.SUBCLAUSE, 4, clause_ref, category

        # 7. Regex: Subclause Letter (e.g. (a) Organ donor expenses, (i) Day care)
        m_subclause_letter = RE_SUBCLAUSE_LETTER.match(first_line)
        if m_subclause_letter and is_first_line_short:
            clause_ref = f"({m_subclause_letter.group(1)})"
            category = self._infer_category(first_line) or self._infer_category(text)
            return NodeType.SUBCLAUSE, 5, clause_ref, category

        # 8. Unnumbered Major Headings (e.g. "GENERAL EXCLUSIONS", "POLICY CONDITIONS")
        category = self._infer_category(text)
        is_heading_size = avg_size >= median_body_size * 1.15
        if is_short and not ends_with_terminal_period and (is_bold or is_all_caps or is_heading_size):
            norm_lower = text.lower()
            if any(kw in norm_lower for kw in KNOWN_EXCLUSION_KEYWORDS.union(KNOWN_COVERAGE_KEYWORDS).union(KNOWN_CONDITION_KEYWORDS).union(KNOWN_DEFINITION_KEYWORDS)):
                return NodeType.SECTION, 2, text, category
            if is_all_caps and is_bold and len(text.split()) <= 8:
                return NodeType.SECTION, 2, text, category

        # 9. List items (bullets, numbers)
        if RE_LIST_BULLET.match(first_line):
            return NodeType.LIST_ITEM, 6, None, category

        # 10. Definitions
        if RE_DEFINITION.match(first_line) or RE_DEFINITION_COLON.match(first_line):
            return NodeType.DEFINITION, 6, None, ChunkCategory.DEFINITIONS

        # 11. Narrative paragraph (default)
        return NodeType.PARAGRAPH, 6, None, category

    def _infer_category(self, text: str) -> Optional[ChunkCategory]:
        """Infers insurance ChunkCategory from text content with contextual affirmative priority."""
        lower = text.lower()

        # 1. Affirmative coverage & waiver patterns must be checked first
        # e.g. "Coverage for Pre-existing Diseases", "Waiting Period Waiver Benefit"
        affirmative_coverage = re.search(
            r"\b(coverage\s+(for|of)|benefits?\s+(for|of)|scope\s+of\s+cover|what\s+is\s+covered|"
            r"waiver\s+of\s+waiting\s+period|waiting\s+period\s+waiver|reduction\s+of\s+waiting\s+period)\b",
            lower
        )
        has_exclusion_prefix = re.search(
            r"^(exclusion|exclusions|what\s+is\s+not\s+covered|not\s+covered|permanent\s+exclusions)\b",
            lower
        )
        if affirmative_coverage and not has_exclusion_prefix:
            return ChunkCategory.COVERAGE

        # 2. Exclusions
        if any(kw in lower for kw in KNOWN_EXCLUSION_KEYWORDS):
            return ChunkCategory.EXCLUSIONS

        # 3. Definitions
        if any(kw in lower for kw in KNOWN_DEFINITION_KEYWORDS):
            return ChunkCategory.DEFINITIONS

        # 4. Conditions
        if any(kw in lower for kw in KNOWN_CONDITION_KEYWORDS):
            return ChunkCategory.CONDITIONS

        # 5. General Coverage
        if any(kw in lower for kw in KNOWN_COVERAGE_KEYWORDS):
            return ChunkCategory.COVERAGE

        return None

    # -------------------------------------------------------------------------
    # Step 5: Hierarchy Tree Construction
    # -------------------------------------------------------------------------
    def _build_hierarchy_tree(
        self,
        extracted_doc: ExtractedDocument,
        masked_block_ids: Set[str],
        median_body_size: float,
        tables: List[StructuredTable]
    ) -> Tuple[List[StructuredSection], List[str]]:
        """
        Constructs the nested StructuredSection hierarchy using an active stack.
        Preserves heading and inline clause content in separate fields.
        """
        root_sections: List[StructuredSection] = []
        stack: List[Tuple[int, StructuredSection]] = []  # [(level, section)]
        warnings: List[str] = []
        sec_counter = 0

        tables_by_page: Dict[int, List[StructuredTable]] = {}
        for tbl in tables:
            tables_by_page.setdefault(tbl.page_number, []).append(tbl)
        attached_table_ids: Set[str] = set()

        for page in extracted_doc.pages:
            p_num = page.page_number

            for block in page.blocks:
                if block.block_id in masked_block_ids:
                    continue

                text = block.text.strip()
                if not text:
                    continue

                node_type, level, clause_ref, category = self._classify_block(
                    block, median_body_size=median_body_size
                )

                if level <= 5:
                    # Heading node (Major Section, Section, Subsection, Clause, Subclause)
                    sec_counter += 1
                    section_id = f"{extracted_doc.doc_id}_sec_{sec_counter}"

                    # Separate heading title from inlined subsequent clause text
                    lines = [l.strip() for l in text.splitlines() if l.strip()]
                    title = lines[0] if lines else text
                    inline_content = "\n\n".join(lines[1:]) if len(lines) > 1 else ""

                    # Inherit category from stack if not explicitly determined
                    effective_category = category
                    if effective_category is None and stack:
                        effective_category = stack[-1][1].category
                    if effective_category is None:
                        effective_category = ChunkCategory.GENERAL

                    new_section = StructuredSection(
                        section_id=section_id,
                        title=title,
                        clause_reference=clause_ref,
                        node_type=node_type,
                        category=effective_category,
                        content=inline_content,
                        page_start=p_num,
                        page_end=p_num,
                        source_block_ids=[block.block_id],
                        bounding_boxes=[block.bbox],
                        children=[]
                    )

                    # Pop stack until we find a parent with strictly lower level
                    while stack and stack[-1][0] >= level:
                        stack.pop()

                    if not stack:
                        root_sections.append(new_section)
                    else:
                        parent_sec = stack[-1][1]
                        parent_sec.children.append(new_section)
                        parent_sec.page_end = max(parent_sec.page_end, p_num)

                    stack.append((level, new_section))

                else:
                    # Narrative block (Paragraph, List Item, Definition)
                    if not stack:
                        sec_counter += 1
                        preamble_id = f"{extracted_doc.doc_id}_sec_preamble_{sec_counter}"
                        preamble_sec = StructuredSection(
                            section_id=preamble_id,
                            title="Preamble",
                            clause_reference=None,
                            node_type=NodeType.SECTION,
                            category=ChunkCategory.GENERAL,
                            content=text,
                            page_start=p_num,
                            page_end=p_num,
                            source_block_ids=[block.block_id],
                            bounding_boxes=[block.bbox],
                            children=[]
                        )
                        root_sections.append(preamble_sec)
                        stack.append((2, preamble_sec))
                    else:
                        active_sec = stack[-1][1]
                        if active_sec.content:
                            active_sec.content += "\n\n" + text
                        else:
                            active_sec.content = text

                        active_sec.page_end = max(active_sec.page_end, p_num)
                        active_sec.source_block_ids.append(block.block_id)
                        active_sec.bounding_boxes.append(block.bbox)

                        for _, ancestor in stack[:-1]:
                            ancestor.page_end = max(ancestor.page_end, p_num)

            # Integrate tables occurring on this page into the active section
            if p_num in tables_by_page:
                for tbl in tables_by_page[p_num]:
                    if tbl.table_id in attached_table_ids:
                        continue
                    attached_table_ids.add(tbl.table_id)

                    table_md = tbl.markdown.strip()
                    if not table_md:
                        continue

                    if stack:
                        active_sec = stack[-1][1]
                        table_repr = f"\n\n[Table: {tbl.table_id}]\n{table_md}\n\n"
                        if active_sec.content:
                            active_sec.content += table_repr
                        else:
                            active_sec.content = table_repr.strip()

                        active_sec.page_end = max(active_sec.page_end, tbl.page_end or tbl.page_number)
                        active_sec.bounding_boxes.append(tbl.bbox)
                        for b_id in tbl.source_block_ids:
                            if b_id not in active_sec.source_block_ids:
                                active_sec.source_block_ids.append(b_id)
                        for _, ancestor in stack[:-1]:
                            ancestor.page_end = max(ancestor.page_end, tbl.page_end or tbl.page_number)
                    else:
                        sec_counter += 1
                        tbl_sec = StructuredSection(
                            section_id=f"{extracted_doc.doc_id}_sec_tbl_{sec_counter}",
                            title=f"Table (Page {tbl.page_number})",
                            clause_reference=None,
                            node_type=NodeType.TABLE,
                            category=ChunkCategory.GENERAL,
                            content=table_md,
                            page_start=tbl.page_start or tbl.page_number,
                            page_end=tbl.page_end or tbl.page_number,
                            source_block_ids=tbl.source_block_ids,
                            bounding_boxes=[tbl.bbox],
                            children=[]
                        )
                        root_sections.append(tbl_sec)

        return root_sections, warnings

    def _create_fallback_sections(
        self,
        extracted_doc: ExtractedDocument,
        masked_ids: Set[str]
    ) -> List[StructuredSection]:
        """Creates a single fallback section containing all unmasked blocks."""
        all_blocks: List[ExtractedBlock] = []
        for page in extracted_doc.pages:
            for b in page.blocks:
                if b.block_id not in masked_ids and b.text.strip():
                    all_blocks.append(b)

        if not all_blocks:
            return []

        full_content = "\n\n".join(b.text.strip() for b in all_blocks)
        page_min = min(b.page_number for b in all_blocks)
        page_max = max(b.page_number for b in all_blocks)

        sec = StructuredSection(
            section_id=f"{extracted_doc.doc_id}_sec_body",
            title="Document Body",
            clause_reference=None,
            node_type=NodeType.SECTION,
            category=ChunkCategory.GENERAL,
            content=full_content,
            page_start=page_min,
            page_end=page_max,
            source_block_ids=[b.block_id for b in all_blocks],
            bounding_boxes=[b.bbox for b in all_blocks],
            children=[]
        )
        return [sec]
