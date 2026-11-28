import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import fitz

from ai_service.config import settings
from ai_service.schemas.ingestion import (
    PageClassification,
    PageDetectionResult,
    OCRResult,
    OCRTextBox,
    BoundingBox
)

logger = logging.getLogger("coverageiq.ingestion.ocr_engine")


def calculate_rectangles_union_area(
    rectangles: List[Tuple[float, float, float, float]],
    boundary: Optional[Tuple[float, float, float, float]] = None
) -> float:
    """
    Computes exact 2D geometric union area of axis-aligned rectangles.
    Overlapping regions are never counted more than once.
    Clips all rectangles to optional boundary (e.g. page.rect).
    """
    if not rectangles:
        return 0.0

    bx0, by0, bx1, by1 = boundary if boundary else (-float("inf"), -float("inf"), float("inf"), float("inf"))
    clipped: List[Tuple[float, float, float, float]] = []

    for r in rectangles:
        x0, y0, x1, y1 = r
        cx0 = max(bx0, min(float(x0), float(x1)))
        cy0 = max(by0, min(float(y0), float(y1)))
        cx1 = min(bx1, max(float(x0), float(x1)))
        cy1 = min(by1, max(float(y0), float(y1)))
        if cx1 > cx0 and cy1 > cy0:
            clipped.append((cx0, cy0, cx1, cy1))

    if not clipped:
        return 0.0

    # Primary: Fast and exact Shapely unary_union
    try:
        from shapely.geometry import box
        from shapely.ops import unary_union
        boxes = [box(c[0], c[1], c[2], c[3]) for c in clipped]
        return float(unary_union(boxes).area)
    except Exception:
        pass

    # Fallback: Exact 2D sweep-line algorithm (Klee's measure)
    events_x = sorted(list(set([c[0] for c in clipped] + [c[2] for c in clipped])))
    total_area = 0.0
    for i in range(len(events_x) - 1):
        x_left = events_x[i]
        x_right = events_x[i + 1]
        dx = x_right - x_left
        if dx <= 0:
            continue
        # Active y intervals covering [x_left, x_right]
        y_intervals = [(c[1], c[3]) for c in clipped if c[0] <= x_left and c[2] >= x_right]
        if not y_intervals:
            continue
        y_intervals.sort()
        merged_y = []
        cur_y0, cur_y1 = y_intervals[0]
        for ny0, ny1 in y_intervals[1:]:
            if ny0 <= cur_y1:
                cur_y1 = max(cur_y1, ny1)
            else:
                merged_y.append((cur_y0, cur_y1))
                cur_y0, cur_y1 = ny0, ny1
        merged_y.append((cur_y0, cur_y1))
        dy = sum(y1 - y0 for y0, y1 in merged_y)
        total_area += dx * dy

    return total_area


class PageScannedDetector:
    """
    Evaluates whether a PDF page is digital, scanned, hybrid, or empty.
    Combines:
    - Character counts and alphanumeric density
    - Text block distribution
    - Raster image placement and surface area coverage
    - Page area aspect ratio
    """

    def __init__(
        self,
        min_digital_chars: Optional[int] = None,
        image_coverage_threshold: Optional[float] = None,
        min_block_count: Optional[int] = None
    ):
        self.min_digital_chars = min_digital_chars or settings.OCR_MIN_DIGITAL_CHARS
        self.image_coverage_threshold = image_coverage_threshold or settings.OCR_IMAGE_COVERAGE_THRESHOLD
        self.min_block_count = min_block_count or settings.OCR_MIN_BLOCK_COUNT

    def detect_page(self, page: fitz.Page, page_number: int) -> PageDetectionResult:
        """
        Analyzes a single PyMuPDF page and returns a classified PageDetectionResult with reasoning.
        """
        rect = page.rect
        page_area = max(1.0, float(rect.width * rect.height))

        # 1. Digital text metrics
        text = page.get_text("text").strip()
        char_count = len(text)
        blocks = page.get_text("blocks")
        text_block_count = sum(1 for b in blocks if b[4].strip())

        # 2. Image metrics
        image_infos = page.get_image_info(xrefs=True)
        raw_images = page.get_images()
        image_count = max(len(image_infos), len(raw_images))

        # Calculate geometric union area of images
        image_rectangles: List[Tuple[float, float, float, float]] = []
        if image_infos:
            for info in image_infos:
                bbox = info.get("bbox")
                if bbox and len(bbox) == 4:
                    image_rectangles.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
        elif raw_images:
            for img in raw_images:
                img_w = img[2]
                img_h = img[3]
                if img_w >= 600 and img_h >= 600:
                    image_rectangles.append((float(page.rect.x0), float(page.rect.y0), float(page.rect.x1), float(page.rect.y1)))
                    break

        page_bounds = (float(page.rect.x0), float(page.rect.y0), float(page.rect.x1), float(page.rect.y1))
        union_image_area = calculate_rectangles_union_area(image_rectangles, boundary=page_bounds)
        image_coverage_ratio = max(0.0, min(1.0, union_image_area / page_area))

        # 3. Multi-factor Classification
        # Case A: Empty page
        if char_count == 0 and image_count == 0:
            return PageDetectionResult(
                page_number=page_number,
                classification=PageClassification.EMPTY,
                digital_char_count=0,
                block_count=0,
                image_count=0,
                image_coverage_ratio=0.0,
                page_area=page_area,
                reason="Page contains zero text characters and zero embedded images."
            )

        # Case B: Scanned page (very low/zero digital characters and significant image presence)
        if char_count < self.min_digital_chars and (image_coverage_ratio >= self.image_coverage_threshold or (image_count > 0 and char_count == 0)):
            return PageDetectionResult(
                page_number=page_number,
                classification=PageClassification.SCANNED,
                digital_char_count=char_count,
                block_count=text_block_count,
                image_count=image_count,
                image_coverage_ratio=image_coverage_ratio,
                page_area=page_area,
                reason=(
                    f"Scanned document detected: low digital text ({char_count} chars < {self.min_digital_chars}) "
                    f"with {image_coverage_ratio:.1%} image coverage across {image_count} image(s)."
                )
            )

        # Case C: Hybrid page (has meaningful text, but ALSO large raster images such as scanned table/stamp)
        if char_count >= self.min_digital_chars and image_coverage_ratio >= self.image_coverage_threshold:
            return PageDetectionResult(
                page_number=page_number,
                classification=PageClassification.HYBRID,
                digital_char_count=char_count,
                block_count=text_block_count,
                image_count=image_count,
                image_coverage_ratio=image_coverage_ratio,
                page_area=page_area,
                reason=(
                    f"Hybrid layout detected: digital text ({char_count} chars) co-exists with "
                    f"significant embedded raster content ({image_coverage_ratio:.1%} coverage)."
                )
            )

        # Case D: Standard digital page
        if char_count >= self.min_digital_chars:
            return PageDetectionResult(
                page_number=page_number,
                classification=PageClassification.DIGITAL,
                digital_char_count=char_count,
                block_count=text_block_count,
                image_count=image_count,
                image_coverage_ratio=image_coverage_ratio,
                page_area=page_area,
                reason=f"Digital page confirmed with {char_count} text characters across {text_block_count} blocks."
            )

        # Case E: Unresolved / low-content edge case
        return PageDetectionResult(
            page_number=page_number,
            classification=PageClassification.UNKNOWN,
            digital_char_count=char_count,
            block_count=text_block_count,
            image_count=image_count,
            image_coverage_ratio=image_coverage_ratio,
            page_area=page_area,
            reason=f"Indeterminate layout: {char_count} text characters and {image_count} image(s)."
        )


class BaseOCREngine(ABC):
    """
    Abstract base interface for OCR engines.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if OCR engine runtime is installed and operational."""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize models lazily."""
        pass

    @abstractmethod
    def process_page(
        self,
        page: fitz.Page,
        page_number: int,
        dpi: Optional[int] = None,
        clip_rect: Optional[BoundingBox] = None
    ) -> OCRResult:
        """Perform OCR recognition on a PyMuPDF page object, optionally restricted to clip_rect."""
        pass


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR Adapter.
    Lazily loads PaddleOCR to avoid startup penalties when processing purely digital documents.
    If PaddleOCR is not installed or initialization fails, gracefully returns an explicit
    unavailable OCRResult without fabricating text.
    """

    def __init__(
        self,
        lang: Optional[str] = None,
        use_angle_cls: Optional[bool] = None,
        dpi: Optional[int] = None
    ):
        self.lang = lang or settings.OCR_LANG
        self.use_angle_cls = use_angle_cls if use_angle_cls is not None else settings.OCR_USE_ANGLE_CLS
        self.dpi = dpi or settings.OCR_DPI
        self._ocr = None
        self._init_attempted = False
        self._is_available = None

    def is_available(self) -> bool:
        if self._is_available is not None:
            return self._is_available
        try:
            import paddleocr  # noqa: F401
            self._is_available = True
        except ImportError:
            self._is_available = False
        except Exception as e:
            logger.warning(f"PaddleOCR check failed: {e}")
            self._is_available = False
        return self._is_available

    def initialize(self) -> bool:
        if self._ocr is not None:
            return True
        if self._init_attempted:
            return self._ocr is not None

        self._init_attempted = True
        if not self.is_available():
            logger.info("PaddleOCR package is not installed in the environment.")
            return False

        try:
            from paddleocr import PaddleOCR
            logger.info(f"Initializing PaddleOCR (lang={self.lang}, use_angle_cls={self.use_angle_cls})")
            self._ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self._ocr = None
            return False

    def process_page(
        self,
        page: fitz.Page,
        page_number: int,
        dpi: Optional[int] = None,
        clip_rect: Optional[BoundingBox] = None
    ) -> OCRResult:
        """
        Converts page (or clipped sub-region) to raster pixmap and executes OCR.
        Preserves bounding boxes, text, confidence, and 1-indexed page number.
        Accurately transforms coordinates accounting for non-zero CropBox origins and page rotation.
        """
        if not self.initialize():
            return OCRResult(
                page_number=page_number,
                success=False,
                boxes=[],
                full_text="",
                mean_confidence=None,
                error_message="PaddleOCR is not available or failed to initialize.",
                engine="paddleocr"
            )

        raster_dpi = dpi or self.dpi
        try:
            import numpy as np

            # Determine clipping rectangle in page coordinates
            if clip_rect is not None:
                fitz_clip = fitz.Rect(clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1) & page.rect
            else:
                fitz_clip = page.rect

            if fitz_clip.is_empty or fitz_clip.width <= 0 or fitz_clip.height <= 0:
                return OCRResult(
                    page_number=page_number,
                    success=True,
                    boxes=[],
                    full_text="",
                    mean_confidence=None,
                    error_message=None,
                    engine="paddleocr"
                )

            # Rasterize page to RGB pixmap
            pix = page.get_pixmap(dpi=raster_dpi, clip=fitz_clip)
            if pix.width <= 0 or pix.height <= 0:
                return OCRResult(
                    page_number=page_number,
                    success=True,
                    boxes=[],
                    full_text="",
                    mean_confidence=None,
                    error_message=None,
                    engine="paddleocr"
                )

            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # RGBA to RGB
                img_array = img_array[:, :, :3]

            # Scale factors to convert OCR pixel coordinates back to PDF points
            scale_x = float(fitz_clip.width) / float(pix.width)
            scale_y = float(fitz_clip.height) / float(pix.height)
            origin_x = float(fitz_clip.x0)
            origin_y = float(fitz_clip.y0)

            # Inference
            raw_results = self._ocr.ocr(img_array, cls=self.use_angle_cls)

            boxes: List[OCRTextBox] = []
            text_lines: List[str] = []
            confidences: List[float] = []

            if raw_results and raw_results[0]:
                for line in raw_results[0]:
                    coords = line[0]
                    text, conf = line[1]

                    # Map coordinates back to PDF points
                    xs = [origin_x + (p[0] * scale_x) for p in coords]
                    ys = [origin_y + (p[1] * scale_y) for p in coords]

                    # Clamp strictly to page bounds
                    clamped_x0 = max(float(page.rect.x0), min(float(page.rect.x1), min(xs)))
                    clamped_y0 = max(float(page.rect.y0), min(float(page.rect.y1), min(ys)))
                    clamped_x1 = max(float(page.rect.x0), min(float(page.rect.x1), max(xs)))
                    clamped_y1 = max(float(page.rect.y0), min(float(page.rect.y1), max(ys)))

                    bbox = BoundingBox(
                        x0=clamped_x0,
                        y0=clamped_y0,
                        x1=clamped_x1,
                        y1=clamped_y1
                    )

                    confidence_val = max(0.0, min(1.0, float(conf)))
                    boxes.append(
                        OCRTextBox(
                            text=str(text).strip(),
                            confidence=confidence_val,
                            bbox=bbox
                        )
                    )
                    text_lines.append(str(text).strip())
                    confidences.append(confidence_val)

            mean_conf = (sum(confidences) / len(confidences)) if confidences else None
            full_text = "\n".join(text_lines).strip()

            return OCRResult(
                page_number=page_number,
                success=True,
                boxes=boxes,
                full_text=full_text,
                mean_confidence=mean_conf,
                error_message=None,
                engine="paddleocr"
            )

        except Exception as e:
            logger.exception(f"OCR inference failed on page {page_number}: {e}")
            return OCRResult(
                page_number=page_number,
                success=False,
                boxes=[],
                full_text="",
                mean_confidence=None,
                error_message=f"OCR inference exception: {str(e)}",
                engine="paddleocr"
            )

    def close(self) -> None:
        """Release lazily initialized PaddleOCR models cleanly and trigger GC."""
        import gc

        model = self._ocr
        self._ocr = None
        self._init_attempted = False
        if model is not None:
            del model
            gc.collect()


class MockOCREngine(BaseOCREngine):
    """
    Mock OCR Engine for unit testing and CI pipelines.
    Allows simulating OCR success, errors, or custom recognized text boxes without external binaries.
    """

    def __init__(
        self,
        is_available_val: bool = True,
        simulate_error: bool = False,
        mock_boxes: Optional[List[Tuple[str, float, Tuple[float, float, float, float]]]] = None
    ):
        self._is_available = is_available_val
        self.simulate_error = simulate_error
        self.mock_boxes = mock_boxes

    def is_available(self) -> bool:
        return self._is_available

    def initialize(self) -> bool:
        return self._is_available

    def process_page(
        self,
        page: fitz.Page,
        page_number: int,
        dpi: Optional[int] = None,
        clip_rect: Optional[BoundingBox] = None
    ) -> OCRResult:
        if not self._is_available:
            return OCRResult(
                page_number=page_number,
                success=False,
                boxes=[],
                full_text="",
                mean_confidence=None,
                error_message="Mock OCR Engine unavailable.",
                engine="mock"
            )

        if self.simulate_error:
            return OCRResult(
                page_number=page_number,
                success=False,
                boxes=[],
                full_text="",
                mean_confidence=None,
                error_message="Simulated OCR inference failure.",
                engine="mock"
            )

        target_bounds = (
            fitz.Rect(clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1) & page.rect
            if clip_rect is not None
            else page.rect
        )

        if target_bounds.is_empty or target_bounds.width <= 0 or target_bounds.height <= 0:
            return OCRResult(
                page_number=page_number,
                success=True,
                boxes=[],
                full_text="",
                mean_confidence=None,
                error_message=None,
                engine="mock"
            )

        boxes: List[OCRTextBox] = []
        text_lines: List[str] = []
        confidences: List[float] = []

        if self.mock_boxes is not None:
            for item in self.mock_boxes:
                text, conf, (x0, y0, x1, y1) = item
                item_rect = fitz.Rect(x0, y0, x1, y1)
                # If clip_rect is given, only include boxes that intersect target_bounds
                if clip_rect is None or not (item_rect & target_bounds).is_empty:
                    cx0 = max(float(page.rect.x0), min(float(page.rect.x1), x0))
                    cy0 = max(float(page.rect.y0), min(float(page.rect.y1), y0))
                    cx1 = max(float(page.rect.x0), min(float(page.rect.x1), x1))
                    cy1 = max(float(page.rect.y0), min(float(page.rect.y1), y1))
                    boxes.append(
                        OCRTextBox(
                            text=text,
                            confidence=conf,
                            bbox=BoundingBox(x0=cx0, y0=cy0, x1=cx1, y1=cy1)
                        )
                    )
                    text_lines.append(text)
                    confidences.append(conf)
        else:
            text = f"Sample Mock OCR Text for Page {page_number}"
            conf = 0.95
            bx0 = target_bounds.x0 + 5.0
            by0 = target_bounds.y0 + 5.0
            bx1 = min(target_bounds.x1, bx0 + 200.0)
            by1 = min(target_bounds.y1, by0 + 30.0)
            boxes.append(
                OCRTextBox(
                    text=text,
                    confidence=conf,
                    bbox=BoundingBox(x0=bx0, y0=by0, x1=bx1, y1=by1)
                )
            )
            text_lines.append(text)
            confidences.append(conf)

        mean_conf = (sum(confidences) / len(confidences)) if confidences else None
        return OCRResult(
            page_number=page_number,
            success=True,
            boxes=boxes,
            full_text="\n".join(text_lines),
            mean_confidence=mean_conf,
            error_message=None,
            engine="mock"
        )
