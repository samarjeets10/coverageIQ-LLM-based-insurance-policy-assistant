import logging
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import fitz

from ai_service.schemas.ingestion import (
    ExtractedDocument,
    ExtractedPage,
    ExtractedBlock,
    ExtractedLine,
    ExtractedSpan,
    BoundingBox,
    DocumentMetadata,
    ExtractionMethod,
    PageDetectionResult,
    PageClassification,
    OCRResult,
    OCRTextBox
)
from ai_service.ingestion.validator import PDFValidator
from ai_service.ingestion.ocr_engine import PageScannedDetector, BaseOCREngine

logger = logging.getLogger("coverageiq.ingestion.pdf_loader")


class PyMuPDFLoader:
    """
    High-fidelity PDF Document Extractor using PyMuPDF (fitz).
    Preserves:
    - 1-indexed page references on every block
    - Font spans, sizes, and styling flags (bold/italic)
    - Precise bounding boxes
    - Geometry-aware multi-column reading order reconstruction
    - Optional page-level OCR on demand
    """

    def __init__(
        self,
        allowed_root: Optional[Union[str, Path]] = None,
        max_size_bytes: Optional[int] = None,
        detector: Optional[PageScannedDetector] = None,
        ocr_engine: Optional[BaseOCREngine] = None
    ):
        self.allowed_root = allowed_root
        self.max_size_bytes = max_size_bytes
        self.detector = detector or PageScannedDetector()
        self.ocr_engine = ocr_engine

    def load_document(
        self,
        file_path: Union[str, Path],
        doc_id: Optional[str] = None
    ) -> ExtractedDocument:
        """
        Validates, opens, and extracts structured blocks from a PDF file.
        """
        safe_path, sha256_hash, file_size = PDFValidator.validate_file(
            file_path,
            allowed_root=self.allowed_root,
            max_size_bytes=self.max_size_bytes
        )

        document_id = doc_id or f"doc_{sha256_hash[:12]}"
        pages: List[ExtractedPage] = []

        with fitz.open(safe_path) as doc:
            meta = doc.metadata or {}
            metadata = DocumentMetadata(
                title=meta.get("title") or None,
                author=meta.get("author") or None,
                subject=meta.get("subject") or None,
                creator=meta.get("creator") or None,
                producer=meta.get("producer") or None,
                creation_date=meta.get("creationDate") or None,
                mod_date=meta.get("modDate") or None,
                file_size_bytes=file_size,
                file_hash_sha256=sha256_hash,
                page_count=len(doc),
                is_encrypted=doc.is_encrypted
            )

            for page_idx in range(len(doc)):
                page_number = page_idx + 1  # 1-indexed
                page = doc[page_idx]
                extracted_page = self._extract_page(page, page_number, document_id)
                pages.append(extracted_page)

        return ExtractedDocument(
            doc_id=document_id,
            metadata=metadata,
            pages=pages
        )

    def load_bytes(
        self,
        data: bytes,
        doc_id: Optional[str] = None
    ) -> ExtractedDocument:
        """
        Validates and extracts structured blocks from an in-memory PDF byte stream.
        """
        sha256_hash, file_size = PDFValidator.validate_bytes(data, max_size_bytes=self.max_size_bytes)
        document_id = doc_id or f"doc_{sha256_hash[:12]}"
        pages: List[ExtractedPage] = []

        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass or doc.is_encrypted:
                from ai_service.ingestion.validator import PasswordProtectedPDFError
                raise PasswordProtectedPDFError("PDF byte stream is password-protected or encrypted.")
            if len(doc) == 0:
                from ai_service.ingestion.validator import EmptyDocumentError
                raise EmptyDocumentError("PDF document contains 0 pages.")

            meta = doc.metadata or {}
            metadata = DocumentMetadata(
                title=meta.get("title") or None,
                author=meta.get("author") or None,
                subject=meta.get("subject") or None,
                creator=meta.get("creator") or None,
                producer=meta.get("producer") or None,
                creation_date=meta.get("creationDate") or None,
                mod_date=meta.get("modDate") or None,
                file_size_bytes=file_size,
                file_hash_sha256=sha256_hash,
                page_count=len(doc),
                is_encrypted=doc.is_encrypted
            )

            for page_idx in range(len(doc)):
                page_number = page_idx + 1
                page = doc[page_idx]
                extracted_page = self._extract_page(page, page_number, document_id)
                pages.append(extracted_page)

        return ExtractedDocument(
            doc_id=document_id,
            metadata=metadata,
            pages=pages
        )

    def _extract_page(self, page: fitz.Page, page_number: int, document_id: str) -> ExtractedPage:
        rect = page.rect
        width = float(rect.width)
        height = float(rect.height)
        rotation = int(page.rotation)

        raw_dict = page.get_text("dict")
        raw_blocks = raw_dict.get("blocks", [])

        extracted_blocks: List[ExtractedBlock] = []

        for raw_idx, block in enumerate(raw_blocks):
            # Block type 0 is text; block type 1 is image
            block_type = "text" if block.get("type", 0) == 0 else "image"
            bbox_coords = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            bbox = BoundingBox(
                x0=float(bbox_coords[0]),
                y0=float(bbox_coords[1]),
                x1=float(bbox_coords[2]),
                y1=float(bbox_coords[3])
            )

            lines: List[ExtractedLine] = []
            block_text_parts: List[str] = []

            if block_type == "text":
                for line in block.get("lines", []):
                    line_coords = line.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    line_bbox = BoundingBox(
                        x0=float(line_coords[0]),
                        y0=float(line_coords[1]),
                        x1=float(line_coords[2]),
                        y1=float(line_coords[3])
                    )
                    spans: List[ExtractedSpan] = []
                    line_text_parts: List[str] = []

                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if not span_text:
                            continue

                        font_name = span.get("font", "")
                        font_size = float(span.get("size", 10.0))
                        flags = int(span.get("flags", 0))

                        # Font flag 2 = italic; flag 16 = bold
                        is_bold = bool(flags & 16) or ("bold" in font_name.lower()) or ("black" in font_name.lower())
                        is_italic = bool(flags & 2) or ("italic" in font_name.lower()) or ("oblique" in font_name.lower())

                        span_coords = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                        span_bbox = BoundingBox(
                            x0=float(span_coords[0]),
                            y0=float(span_coords[1]),
                            x1=float(span_coords[2]),
                            y1=float(span_coords[3])
                        )

                        spans.append(
                            ExtractedSpan(
                                text=span_text,
                                font_name=font_name,
                                font_size=font_size,
                                is_bold=is_bold,
                                is_italic=is_italic,
                                color=span.get("color"),
                                bbox=span_bbox
                            )
                        )
                        line_text_parts.append(span_text)

                    line_full_text = " ".join(line_text_parts).strip()
                    if line_full_text or spans:
                        lines.append(
                            ExtractedLine(
                                text=line_full_text,
                                bbox=line_bbox,
                                spans=spans
                            )
                        )
                        block_text_parts.append(line_full_text)

            full_block_text = "\n".join(block_text_parts).strip()

            # Ignore empty decorative blocks with no text and no lines
            if block_type == "text" and not full_block_text and not lines:
                continue

            extracted_blocks.append(
                ExtractedBlock(
                    block_id=f"{document_id}_p{page_number}_b{raw_idx}",
                    page_number=page_number,
                    raw_block_idx=raw_idx,
                    reading_order_idx=raw_idx,  # Updated after reconstruction
                    block_type=block_type,
                    bbox=bbox,
                    lines=lines,
                    text=full_block_text,
                    column_idx=0
                )
            )

        # Reconstruct logical reading order
        ordered_blocks = self.reconstruct_reading_order(extracted_blocks, page_width=width, page_height=height)

        # Re-assign reading_order_idx and block_id based on sorted reading order
        for idx, blk in enumerate(ordered_blocks):
            blk.reading_order_idx = idx

        # Page classification using multi-factor detector
        detection = self.detector.detect_page(page, page_number)

        # Determine extraction method and perform OCR if necessary
        extraction_method = ExtractionMethod.DIGITAL
        ocr_result: Optional[Any] = None

        if detection.classification == PageClassification.SCANNED:
            if self.ocr_engine is not None:
                ocr_result = self.ocr_engine.process_page(page, page_number)
                if ocr_result.success:
                    extraction_method = ExtractionMethod.OCR
            else:
                logger.debug(f"Page {page_number} detected as SCANNED, but no OCR engine configured.")

        elif detection.classification == PageClassification.HYBRID:
            if self.ocr_engine is not None:
                # 1. Target image regions for OCR to avoid re-OCRing digital text
                image_infos = page.get_image_info(xrefs=True)
                valid_image_boxes: List[BoundingBox] = []

                if image_infos:
                    for info in image_infos:
                        bbox = info.get("bbox")
                        if bbox and len(bbox) == 4:
                            w = max(0.0, float(bbox[2] - bbox[0]))
                            h = max(0.0, float(bbox[3] - bbox[1]))
                            # Skip tiny decorative icons/lines (< 200 pt^2 or < 15pt dimension)
                            if w >= 15 and h >= 15 and (w * h) >= 200:
                                valid_image_boxes.append(
                                    BoundingBox(
                                        x0=float(bbox[0]),
                                        y0=float(bbox[1]),
                                        x1=float(bbox[2]),
                                        y1=float(bbox[3])
                                    )
                                )

                # Merge heavily overlapping image boxes so we don't OCR duplicate sub-regions
                merged_regions = self._merge_overlapping_regions(valid_image_boxes)

                if merged_regions:
                    # OCR each distinct image region
                    combined_boxes: List[OCRTextBox] = []
                    text_lines: List[str] = []
                    all_confidences: List[float] = []
                    any_success = False

                    for region in merged_regions:
                        res = self.ocr_engine.process_page(page, page_number, clip_rect=region)
                        if res.success:
                            any_success = True
                            for b in res.boxes:
                                combined_boxes.append(b)
                                text_lines.append(b.text)
                                if b.confidence is not None:
                                    all_confidences.append(b.confidence)

                    mean_conf = (sum(all_confidences) / len(all_confidences)) if all_confidences else None
                    ocr_result = OCRResult(
                        page_number=page_number,
                        success=any_success,
                        boxes=combined_boxes,
                        full_text="\n".join(text_lines),
                        mean_confidence=mean_conf,
                        error_message=None if any_success else "Region OCR failed",
                        engine=getattr(self.ocr_engine, "__class__", type(self.ocr_engine)).__name__.lower()
                    )
                else:
                    # Fallback: full-page OCR with spatial deduplication against digital blocks
                    res = self.ocr_engine.process_page(page, page_number)
                    if res.success:
                        digital_bboxes = [b.bbox for b in ordered_blocks if b.block_type == "text" and b.text.strip()]
                        deduped_boxes = []
                        deduped_lines = []
                        deduped_confs = []

                        for obox in res.boxes:
                            overlaps_digital = False
                            for dbox in digital_bboxes:
                                if self._boxes_overlap_significantly(obox.bbox, dbox, threshold=0.25):
                                    overlaps_digital = True
                                    break
                            if not overlaps_digital:
                                deduped_boxes.append(obox)
                                deduped_lines.append(obox.text)
                                if obox.confidence is not None:
                                    deduped_confs.append(obox.confidence)

                        mean_conf = (sum(deduped_confs) / len(deduped_confs)) if deduped_confs else None
                        ocr_result = OCRResult(
                            page_number=page_number,
                            success=True,
                            boxes=deduped_boxes,
                            full_text="\n".join(deduped_lines),
                            mean_confidence=mean_conf,
                            error_message=None,
                            engine=getattr(self.ocr_engine, "__class__", type(self.ocr_engine)).__name__.lower()
                        )
                    else:
                        ocr_result = res

                if ocr_result and ocr_result.success:
                    extraction_method = ExtractionMethod.HYBRID
            else:
                logger.debug(f"Page {page_number} detected as HYBRID, but no OCR engine configured.")

        return ExtractedPage(
            page_number=page_number,
            width=width,
            height=height,
            rotation=rotation,
            detection=detection,
            blocks=ordered_blocks,
            ocr_result=ocr_result,
            extraction_method=extraction_method
        )

    @classmethod
    def reconstruct_reading_order(
        cls,
        blocks: List[ExtractedBlock],
        page_width: float,
        page_height: float
    ) -> List[ExtractedBlock]:
        """
        Geometry-aware reading-order reconstruction.
        Distinguishes:
        - Running headers (top 8% margin)
        - Running footers (bottom 8% margin)
        - Full-width spanning headings
        - Two-column layouts (clusters blocks into left vs right column)
        - Single-column flow
        """
        if not blocks:
            return []

        header_blocks: List[ExtractedBlock] = []
        footer_blocks: List[ExtractedBlock] = []
        body_blocks: List[ExtractedBlock] = []

        header_threshold = 0.08 * page_height
        footer_threshold = 0.92 * page_height

        for b in blocks:
            # Check for header
            if b.bbox.y1 <= header_threshold:
                header_blocks.append(b)
            # Check for footer
            elif b.bbox.y0 >= footer_threshold:
                footer_blocks.append(b)
            else:
                body_blocks.append(b)

        # Detect if body is two-column
        mid_x = page_width / 2.0
        left_col_candidates: List[ExtractedBlock] = []
        right_col_candidates: List[ExtractedBlock] = []
        span_full_candidates: List[ExtractedBlock] = []

        for b in body_blocks:
            # Block spanning > 60% of width centered across midpoint
            if b.bbox.width > 0.60 * page_width and (b.bbox.x0 < 0.35 * page_width and b.bbox.x1 > 0.65 * page_width):
                span_full_candidates.append(b)
            elif b.bbox.x1 <= mid_x + (0.05 * page_width):
                left_col_candidates.append(b)
            elif b.bbox.x0 >= mid_x - (0.05 * page_width):
                right_col_candidates.append(b)
            else:
                # Ambiguous or slightly crossing center
                if (b.bbox.x0 + b.bbox.x1) / 2.0 < mid_x:
                    left_col_candidates.append(b)
                else:
                    right_col_candidates.append(b)

        # Determine if two-column layout is prominent
        total_col_blocks = len(left_col_candidates) + len(right_col_candidates)
        is_two_column = False
        if total_col_blocks >= 4:
            # Check if both columns have significant content
            if len(left_col_candidates) >= 2 and len(right_col_candidates) >= 2:
                is_two_column = True

        ordered_body: List[ExtractedBlock] = []

        if is_two_column:
            # Mark column indices
            for b in left_col_candidates:
                b.column_idx = 0
            for b in right_col_candidates:
                b.column_idx = 1
            for b in span_full_candidates:
                b.column_idx = 0

            # Merge spanning headers and columns in vertical bands
            # Sort full spans and column items by vertical position
            all_segments: List[Tuple[float, str, Any]] = []
            for b in span_full_candidates:
                all_segments.append((b.bbox.y0, "span", b))

            # Group column blocks between spanning blocks
            left_sorted = sorted(left_col_candidates, key=lambda b: (b.bbox.y0, b.bbox.x0))
            right_sorted = sorted(right_col_candidates, key=lambda b: (b.bbox.y0, b.bbox.x0))

            if not span_full_candidates:
                # Pure two-column: Left column first (top-to-bottom), then Right column (top-to-bottom)
                ordered_body.extend(left_sorted)
                ordered_body.extend(right_sorted)
            else:
                # Segmented by full-width banners/headings
                y_cuts = sorted([b.bbox.y0 for b in span_full_candidates])
                # Interleave: spanning heading -> left sub-blocks -> right sub-blocks
                curr_y = 0.0
                for cut_b in sorted(span_full_candidates, key=lambda b: b.bbox.y0):
                    # Collect left and right before this cut
                    sub_left = [b for b in left_sorted if curr_y <= b.bbox.y0 < cut_b.bbox.y0]
                    sub_right = [b for b in right_sorted if curr_y <= b.bbox.y0 < cut_b.bbox.y0]
                    ordered_body.extend(sub_left)
                    ordered_body.extend(sub_right)
                    ordered_body.append(cut_b)
                    curr_y = cut_b.bbox.y1

                # Add remainder
                sub_left = [b for b in left_sorted if b.bbox.y0 >= curr_y]
                sub_right = [b for b in right_sorted if b.bbox.y0 >= curr_y]
                ordered_body.extend(sub_left)
                ordered_body.extend(sub_right)
        else:
            # Single-column: sort primarily by top Y coordinate, then X
            # Use a line-band tolerance (e.g. within 4 points of Y)
            def sort_key(blk: ExtractedBlock):
                y_band = round(blk.bbox.y0 / 5.0) * 5.0
                return (y_band, blk.bbox.x0)

            ordered_body = sorted(body_blocks, key=sort_key)

        # Sort headers and footers
        header_sorted = sorted(header_blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))
        footer_sorted = sorted(footer_blocks, key=lambda b: (b.bbox.y0, b.bbox.x0))

        return header_sorted + ordered_body + footer_sorted

    @staticmethod
    def _boxes_overlap_significantly(b1: BoundingBox, b2: BoundingBox, threshold: float = 0.25) -> bool:
        """
        Determines whether two bounding boxes overlap significantly.
        """
        inter_x0 = max(b1.x0, b2.x0)
        inter_y0 = max(b1.y0, b2.y0)
        inter_x1 = min(b1.x1, b2.x1)
        inter_y1 = min(b1.y1, b2.y1)
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return False
        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        min_area = min(b1.area, b2.area)
        if min_area <= 0:
            return False
        return (inter_area / min_area) >= threshold

    @classmethod
    def _merge_overlapping_regions(cls, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Clusters and merges overlapping candidate image bounding boxes to prevent redundant OCR passes.
        """
        if not boxes:
            return []
        merged: List[BoundingBox] = []
        remaining = list(boxes)
        while remaining:
            curr = remaining.pop(0)
            found_overlap = False
            for i, other in enumerate(remaining):
                if cls._boxes_overlap_significantly(curr, other, threshold=0.10):
                    new_box = BoundingBox(
                        x0=min(curr.x0, other.x0),
                        y0=min(curr.y0, other.y0),
                        x1=max(curr.x1, other.x1),
                        y1=max(curr.y1, other.y1)
                    )
                    remaining.pop(i)
                    remaining.append(new_box)
                    found_overlap = True
                    break
            if not found_overlap:
                merged.append(curr)
        return merged
