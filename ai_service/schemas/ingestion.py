from enum import Enum
from typing import List, Optional, Tuple, Any, Dict
from pydantic import BaseModel, Field, model_validator
import fitz
from ai_service.schemas.structured_output import ChunkCategory


class PageClassification(str, Enum):
    DIGITAL = "DIGITAL"
    SCANNED = "SCANNED"
    HYBRID = "HYBRID"
    EMPTY = "EMPTY"
    UNKNOWN = "UNKNOWN"


class ExtractionMethod(str, Enum):
    DIGITAL = "DIGITAL"
    OCR = "OCR"
    HYBRID = "HYBRID"


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x0 > self.x1:
            self.x0, self.x1 = self.x1, self.x0
        if self.y0 > self.y1:
            self.y0, self.y1 = self.y1, self.y0
        return self

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def intersects(self, other: "BoundingBox") -> bool:
        return not (self.x1 < other.x0 or self.x0 > other.x1 or self.y1 < other.y0 or self.y0 > other.y1)

    def intersection_area(self, other: "BoundingBox") -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0)

    def iou(self, other: "BoundingBox") -> float:
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return (inter / union) if union > 0 else 0.0


class ExtractedSpan(BaseModel):
    text: str
    font_name: str
    font_size: float
    is_bold: bool = False
    is_italic: bool = False
    color: Optional[int] = None
    bbox: BoundingBox


class ExtractedLine(BaseModel):
    text: str
    bbox: BoundingBox
    spans: List[ExtractedSpan] = Field(default_factory=list)


class ExtractedBlock(BaseModel):
    block_id: str
    page_number: int = Field(ge=1, description="1-indexed source PDF page number")
    raw_block_idx: int = Field(ge=0, description="Original PyMuPDF extraction index")
    reading_order_idx: int = Field(ge=0, description="Reconstructed logical reading order index")
    block_type: str = "text"  # text | image | table
    bbox: BoundingBox
    lines: List[ExtractedLine] = Field(default_factory=list)
    text: str
    column_idx: int = Field(default=0, ge=0, description="Detected column index (0=left, 1=right, etc.)")


class PageDetectionResult(BaseModel):
    page_number: int = Field(ge=1, description="1-indexed source PDF page number")
    classification: PageClassification
    digital_char_count: int = Field(ge=0)
    block_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    image_coverage_ratio: float = Field(ge=0.0, description="Fraction of page area covered by raster images")
    page_area: float = Field(gt=0.0)
    reason: str


class OCRTextBox(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class OCRResult(BaseModel):
    page_number: int = Field(ge=1, description="1-indexed source PDF page number")
    success: bool
    boxes: List[OCRTextBox] = Field(default_factory=list)
    full_text: str = ""
    mean_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    error_message: Optional[str] = None
    engine: str = "paddleocr"


class ExtractedPage(BaseModel):
    page_number: int = Field(ge=1, description="1-indexed source PDF page number")
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)
    rotation: int = 0
    detection: PageDetectionResult
    blocks: List[ExtractedBlock] = Field(default_factory=list)
    ocr_result: Optional[OCRResult] = None
    extraction_method: ExtractionMethod = ExtractionMethod.DIGITAL


class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    mod_date: Optional[str] = None
    file_size_bytes: int = Field(ge=0)
    file_hash_sha256: str
    page_count: int = Field(ge=0)
    is_encrypted: bool = False


class ExtractedDocument(BaseModel):
    doc_id: str
    metadata: DocumentMetadata
    pages: List[ExtractedPage] = Field(default_factory=list)


class NodeType(str, Enum):
    DOCUMENT_TITLE = "document_title"
    MAJOR_SECTION = "major_section"
    SECTION = "section"
    SUBSECTION = "subsection"
    CLAUSE = "clause"
    SUBCLAUSE = "subclause"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    DEFINITION = "definition"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"


class HeaderFooterItem(BaseModel):
    item_type: str = "header"  # header | footer
    page_number: int = Field(ge=1)
    text: str
    bbox: BoundingBox
    confidence: float = 1.0


class StructuredTableCell(BaseModel):
    row_idx: int = Field(ge=0)
    col_idx: int = Field(ge=0)
    text: str
    bbox: Optional[BoundingBox] = None
    is_header: bool = False
    colspan: int = 1
    rowspan: int = 1


class StructuredTable(BaseModel):
    table_id: str
    page_number: int = Field(ge=1, description="1-indexed source PDF page number")
    page_start: Optional[int] = Field(default=None, ge=1)
    page_end: Optional[int] = Field(default=None, ge=1)
    bbox: BoundingBox
    headers: List[str] = Field(default_factory=list)
    rows: List[List[Optional[str]]] = Field(default_factory=list)
    cells: List[StructuredTableCell] = Field(default_factory=list)
    markdown: str
    source_block_ids: List[str] = Field(default_factory=list)
    extraction_method: ExtractionMethod = ExtractionMethod.DIGITAL
    extraction_warnings: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def set_page_range(self) -> "StructuredTable":
        if self.page_start is None:
            self.page_start = self.page_number
        if self.page_end is None:
            self.page_end = self.page_number
        return self


class StructuredSection(BaseModel):
    section_id: str
    title: str
    clause_reference: Optional[str] = None
    node_type: NodeType = NodeType.SECTION
    category: Optional[ChunkCategory] = None
    content: str = ""
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    source_block_ids: List[str] = Field(default_factory=list)
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)
    children: List["StructuredSection"] = Field(default_factory=list)

    def get_all_subsections(self) -> List["StructuredSection"]:
        """Recursively collect all descendant sections in depth-first order."""
        result: List["StructuredSection"] = []
        for child in self.children:
            result.append(child)
            result.extend(child.get_all_subsections())
        return result


StructuredSection.model_rebuild()


class StructuredDocument(BaseModel):
    doc_id: str
    document_title: Optional[str] = None
    metadata: DocumentMetadata
    sections: List[StructuredSection] = Field(default_factory=list)
    tables: List[StructuredTable] = Field(default_factory=list)
    headers_and_footers: List[HeaderFooterItem] = Field(default_factory=list)
    extraction_warnings: List[str] = Field(default_factory=list)
    source_page_count: int = Field(ge=0)

    def get_all_sections(self) -> List[StructuredSection]:
        """Return all sections in the document flattened in depth-first order."""
        result: List[StructuredSection] = []
        for sec in self.sections:
            result.append(sec)
            result.extend(sec.get_all_subsections())
        return result


