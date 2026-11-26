import re
import hashlib
import logging
from typing import List, Optional, Tuple, Dict, Any, Set
from pydantic import BaseModel, Field

from ai_service.schemas.ingestion import (
    StructuredDocument,
    StructuredSection,
    StructuredTable,
    NodeType,
    ExtractionMethod,
)
from ai_service.schemas.structured_output import PolicyChunk, ChunkCategory

logger = logging.getLogger("coverageiq.ingestion.semantic_chunker")


# =============================================================================
# Legal Exception & Qualifier Regex Patterns
# =============================================================================
RE_LEGAL_EXCEPTION_MARKERS = re.compile(
    r"\b("
    r"except\s+(that|for|when|as|in\s+case|where)|"
    r"provided\s+(always|that|further)|"
    r"subject\s+to|"
    r"unless|"
    r"notwithstanding|"
    r"with\s+the\s+exception\s+of|"
    r"saving\s+and\s+except|"
    r"limitation\s*[:\s]|"
    r"sub-limit\s*[:\s]|"
    r"waiting\s+period\s*[:\s]|"
    r"conditions?\s+precedent|"
    r"exclusion\s*[:\s]"
    r")\b",
    re.IGNORECASE,
)

RE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")


# =============================================================================
# Tokenizer Abstraction & Configurable Implementations
# =============================================================================
class BaseTokenizer:
    """Abstract base class for semantic chunking tokenizers."""

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def is_fallback_active(self) -> bool:
        return False


class HuggingFaceTokenizer(BaseTokenizer):
    """
    HuggingFace AutoTokenizer wrapper (configured for BAAI/bge-m3).
    Lazy-loads the model tokenizer on first use.
    In unit tests or offline environments where model weights/cache are not present,
    it activates a calibrated XLM-RoBERTa SentencePiece legal approximation without
    performing unexpected network downloads.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", local_files_only: bool = True):
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._tokenizer = None
        self._is_loaded = False
        self._fallback_active = False

    def _load(self) -> None:
        if self._is_loaded or self._fallback_active:
            return
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._is_loaded = True
            logger.info(f"Loaded HuggingFace tokenizer '{self.model_name}' successfully.")
        except Exception as e:
            self._fallback_active = True
            logger.info(
                f"HuggingFace tokenizer '{self.model_name}' offline/unavailable ({e}). "
                f"Active fallback: Using calibrated BGE-M3 XLM-RoBERTa SentencePiece approximation."
            )

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        self._load()
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception as e:
                logger.warning(f"Error encoding with HuggingFace tokenizer: {e}")

        # Calibrated BGE-M3 (XLM-RoBERTa SentencePiece) approximation:
        # XLM-RoBERTa SentencePiece breaks English legal clauses and punctuation into
        # ~1.25x words, calibrated against the BAAI/bge-m3 vocabulary.
        tokens = re.findall(r"\b[A-Za-z0-9_\.-]+\b|[^\w\s]", text)
        return max(1, int(len(tokens) * 1.25))

    @property
    def name(self) -> str:
        return f"HuggingFaceTokenizer({self.model_name})"

    @property
    def is_fallback_active(self) -> bool:
        self._load()
        return self._fallback_active


class TiktokenTokenizer(BaseTokenizer):
    """
    OpenAI tiktoken BPE tokenizer (explicitly selected fallback / alternative).
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._tiktoken = None
        self._is_loaded = False
        self._fallback_active = False

    def _load(self) -> None:
        if self._is_loaded or self._fallback_active:
            return
        try:
            import tiktoken
            self._tiktoken = tiktoken.get_encoding(self.encoding_name)
            self._is_loaded = True
        except Exception as e:
            self._fallback_active = True
            logger.warning(f"Could not load tiktoken encoding '{self.encoding_name}': {e}")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        self._load()
        if self._tiktoken is not None:
            try:
                return len(self._tiktoken.encode(text, disallowed_special=()))
            except Exception:
                pass
        tokens = re.findall(r"\b[A-Za-z0-9_\.-]+\b|[^\w\s]", text)
        return max(1, int(len(tokens) * 1.15))

    @property
    def name(self) -> str:
        return f"TiktokenTokenizer({self.encoding_name})"

    @property
    def is_fallback_active(self) -> bool:
        self._load()
        return self._fallback_active


# Backward-compatible alias
Tokenizer = HuggingFaceTokenizer


# =============================================================================
# Breadcrumb Node
# =============================================================================
class BreadcrumbItem(BaseModel):
    level_name: str  # e.g., "Policy", "Section", "Clause", "Subclause"
    title: str
    clause_ref: Optional[str] = None


# =============================================================================
# Semantic Chunker Engine
# =============================================================================
class SemanticChunker:
    """
    Structure-Aware Semantic Chunking Engine for Insurance Policies.

    Transforms a StructuredDocument into a flattened list of PolicyChunk objects
    strictly observing the 'NO EVIDENCE = NO CLAIM' principle:
    - Preserves section/clause hierarchy and breadcrumb trails.
    - Preserves legal exceptions, provisos, and condition qualifiers attached to operative rules.
    - Handles tables with row-level integrity and repeated headers; splits oversized rows/cells.
    - Tokenizer abstraction supports HuggingFace BAAI/bge-m3 with lazy loading and dependency injection.
    - Emits stable, deterministic chunk IDs including page_start and section_id.
    - Retains full 1-indexed page provenance, page spans, source block IDs, and OCR confidence.
    """

    def __init__(
        self,
        target_chunk_size: int = 400,
        max_chunk_size: int = 600,
        min_chunk_size: int = 50,
        overlap_tokens: int = 0,
        tokenizer: Optional[BaseTokenizer] = None,
        tokenizer_name: str = "BAAI/bge-m3",
        algorithm_version: str = "v3.1",
    ):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        # By default overlap_tokens=0 to maintain strictly non-overlapping legal evidence boundaries.
        # When overlap_tokens > 0, safe sentence overlap is applied across consecutive parts of oversized clauses.
        self.overlap_tokens = overlap_tokens
        self.algorithm_version = algorithm_version

        # Tokenizer initialization with dependency injection support
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name in ("cl100k_base", "p50k_base", "r50k_base", "tiktoken"):
            self.tokenizer = TiktokenTokenizer(
                encoding_name=tokenizer_name if tokenizer_name != "tiktoken" else "cl100k_base"
            )
        else:
            self.tokenizer = HuggingFaceTokenizer(model_name=tokenizer_name, local_files_only=True)

    def chunk_document(
        self,
        structured_doc: StructuredDocument,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        ocr_confidence_by_page: Optional[Dict[int, float]] = None,
    ) -> List[PolicyChunk]:
        """
        Chunks an entire StructuredDocument into retrieval-ready PolicyChunk objects.

        Args:
            structured_doc: The structured document produced by Phase 2.
            policy_id: Optional policy identifier (defaults to structured_doc.doc_id).
            policy_version: Optional policy version or fingerprint for deterministic IDs.
            ocr_confidence_by_page: Optional mapping of page_number -> mean OCR confidence float.

        Returns:
            List[PolicyChunk] with complete provenance and breadcrumbs.
        """
        eff_policy_id = policy_id or structured_doc.doc_id
        eff_version = policy_version or (
            structured_doc.metadata.file_hash_sha256[:8]
            if structured_doc.metadata.file_hash_sha256
            else "v1"
        )

        chunks: List[PolicyChunk] = []
        emitted_chunk_ids: Set[str] = set()

        # Document-level root breadcrumb
        doc_title = structured_doc.document_title or structured_doc.metadata.title or eff_policy_id
        root_breadcrumb = [BreadcrumbItem(level_name="Policy", title=doc_title)]

        # Map of tables attached to sections via [Table: table_id]
        tables_by_id: Dict[str, StructuredTable] = {t.table_id: t for t in structured_doc.tables}
        processed_table_ids: Set[str] = set()

        # Build OCR confidence lookup from document pages if available
        ocr_conf_map: Dict[int, float] = dict(ocr_confidence_by_page or {})

        # Step 1: Recursively chunk the section hierarchy
        for section in structured_doc.sections:
            sec_chunks = self._chunk_section_tree(
                section=section,
                breadcrumb_trail=list(root_breadcrumb),
                policy_id=eff_policy_id,
                policy_version=eff_version,
                tables_by_id=tables_by_id,
                processed_table_ids=processed_table_ids,
                ocr_conf_map=ocr_conf_map,
            )
            for c in sec_chunks:
                if c.chunk_id not in emitted_chunk_ids:
                    emitted_chunk_ids.add(c.chunk_id)
                    chunks.append(c)

        # Step 2: Chunk any standalone or unattached tables
        for table in structured_doc.tables:
            if table.table_id not in processed_table_ids:
                table_chunks = self._chunk_table(
                    table=table,
                    breadcrumb_trail=list(root_breadcrumb),
                    policy_id=eff_policy_id,
                    policy_version=eff_version,
                    parent_category=ChunkCategory.GENERAL,
                    parent_clause_ref="Table Schedule",
                    parent_section_id=table.table_id,
                    ocr_conf_map=ocr_conf_map,
                )
                for c in table_chunks:
                    if c.chunk_id not in emitted_chunk_ids:
                        emitted_chunk_ids.add(c.chunk_id)
                        chunks.append(c)

        return chunks

    # -------------------------------------------------------------------------
    # Section Tree Traversal & Hierarchical Breadcrumbs
    # -------------------------------------------------------------------------
    def _chunk_section_tree(
        self,
        section: StructuredSection,
        breadcrumb_trail: List[BreadcrumbItem],
        policy_id: str,
        policy_version: str,
        tables_by_id: Dict[str, StructuredTable],
        processed_table_ids: Set[str],
        ocr_conf_map: Dict[int, float],
    ) -> List[PolicyChunk]:
        """
        Recursively chunks a section and its subsections while maintaining breadcrumbs.
        Parent sections with no body text supply breadcrumb context without producing empty chunks.
        """
        chunks: List[PolicyChunk] = []

        level_name = "Section"
        if section.node_type == NodeType.MAJOR_SECTION:
            level_name = "Part"
        elif section.node_type == NodeType.SUBSECTION:
            level_name = "Clause"
        elif section.node_type in (NodeType.SUBCLAUSE, NodeType.LIST_ITEM):
            level_name = "Subclause"

        curr_item = BreadcrumbItem(
            level_name=level_name,
            title=section.title,
            clause_ref=section.clause_reference,
        )
        current_trail = breadcrumb_trail + [curr_item]
        effective_clause_ref = section.clause_reference or section.title

        # Check if this section contains table markers
        raw_content = section.content.strip()
        table_matches = re.findall(r"\[Table:\s*([a-zA-Z0-9_\-]+)\]", raw_content)

        text_content = raw_content
        for t_id in table_matches:
            if t_id in tables_by_id:
                processed_table_ids.add(t_id)
                tbl = tables_by_id[t_id]
                text_content = text_content.replace(f"[Table: {t_id}]", "").replace(tbl.markdown, "").strip()

        # 1. Chunk narrative content if present (Parents with NO body text are skipped here)
        if text_content:
            category = section.category or ChunkCategory.GENERAL
            narrative_chunks = self._chunk_narrative_content(
                text=text_content,
                breadcrumb_trail=current_trail,
                section=section,
                policy_id=policy_id,
                policy_version=policy_version,
                category=category,
                clause_ref=effective_clause_ref,
                ocr_conf_map=ocr_conf_map,
            )
            chunks.extend(narrative_chunks)

        # 2. Chunk any embedded tables that belong to this section
        for t_id in table_matches:
            if t_id in tables_by_id:
                tbl = tables_by_id[t_id]
                tbl_chunks = self._chunk_table(
                    table=tbl,
                    breadcrumb_trail=current_trail,
                    policy_id=policy_id,
                    policy_version=policy_version,
                    parent_category=section.category or ChunkCategory.GENERAL,
                    parent_clause_ref=effective_clause_ref,
                    parent_section_id=section.section_id,
                    ocr_conf_map=ocr_conf_map,
                )
                chunks.extend(tbl_chunks)

        # 3. Recursively process children
        for child in section.children:
            child_chunks = self._chunk_section_tree(
                section=child,
                breadcrumb_trail=current_trail,
                policy_id=policy_id,
                policy_version=policy_version,
                tables_by_id=tables_by_id,
                processed_table_ids=processed_table_ids,
                ocr_conf_map=ocr_conf_map,
            )
            chunks.extend(child_chunks)

        return chunks

    # -------------------------------------------------------------------------
    # Narrative Chunking with Legal Exception Preservation
    # -------------------------------------------------------------------------
    def _chunk_narrative_content(
        self,
        text: str,
        breadcrumb_trail: List[BreadcrumbItem],
        section: StructuredSection,
        policy_id: str,
        policy_version: str,
        category: ChunkCategory,
        clause_ref: str,
        ocr_conf_map: Dict[int, float],
    ) -> List[PolicyChunk]:
        """
        Chunks narrative legal text into token-sized PolicyChunk units.
        Preserves exceptions, provisos, and conditions with the operative rule.
        """
        breadcrumb_header = self._format_breadcrumbs(breadcrumb_trail)
        header_tokens = self.tokenizer.count_tokens(breadcrumb_header)

        available_tokens = self.max_chunk_size - header_tokens - 25
        if available_tokens < self.min_chunk_size:
            available_tokens = self.max_chunk_size // 2

        # Break text into atomic legal units (sentences/paragraphs preserving exceptions)
        atomic_units = self._extract_atomic_legal_units(text)

        # Group atomic units into token-sized chunks
        grouped_slices: List[str] = []
        current_slice: List[str] = []
        current_tokens = 0

        for unit in atomic_units:
            unit_tokens = self.tokenizer.count_tokens(unit)

            if unit_tokens > available_tokens:
                if current_slice:
                    grouped_slices.append("\n\n".join(current_slice))
                    current_slice = []
                    current_tokens = 0

                oversized_parts = self._split_oversized_unit(unit, available_tokens)
                grouped_slices.extend(oversized_parts)
                continue

            if current_tokens + unit_tokens > available_tokens and current_slice:
                grouped_slices.append("\n\n".join(current_slice))
                current_slice = [unit]
                current_tokens = unit_tokens
            else:
                current_slice.append(unit)
                current_tokens += unit_tokens

        if current_slice:
            grouped_slices.append("\n\n".join(current_slice))

        if not grouped_slices:
            return []

        chunks: List[PolicyChunk] = []
        total_parts = len(grouped_slices)
        p_start = max(1, section.page_start)
        p_end = max(1, section.page_end)

        # Determine OCR provenance without inventing confidence for digital text
        is_ocr = any(b_id.startswith(("ocr_", "doc_ocr_")) for b_id in section.source_block_ids)
        ocr_conf = ocr_conf_map.get(p_start) if is_ocr else None
        extraction_method_str = "OCR" if is_ocr else "DIGITAL"

        for part_idx, slice_text in enumerate(grouped_slices, start=1):
            # Apply safe sentence overlap if explicitly requested (overlap_tokens > 0)
            rendered_text = slice_text
            if part_idx > 1 and self.overlap_tokens > 0 and len(grouped_slices) > 1:
                prev_slice = grouped_slices[part_idx - 2]
                prev_sentences = [s.strip() for s in RE_SENTENCE_BOUNDARY.split(prev_slice) if s.strip()]
                if prev_sentences:
                    last_s = prev_sentences[-1]
                    if self.tokenizer.count_tokens(last_s) <= self.overlap_tokens:
                        rendered_text = f"[Context: {last_s}]\n{slice_text}"

            part_suffix = f" [Part {part_idx}/{total_parts}]" if total_parts > 1 else ""
            eff_clause_ref = f"{clause_ref}{part_suffix}"
            full_content = f"{breadcrumb_header}\n---\n{rendered_text}".strip()

            chunk_id = self._generate_chunk_id(
                policy_id=policy_id,
                policy_version=policy_version,
                clause_ref=eff_clause_ref,
                content_text=full_content,
                section_id=section.section_id,
                page_start=p_start,
            )

            metadata_dict: Dict[str, Any] = {
                "page_start": p_start,
                "page_end": p_end,
                "source_block_ids": section.source_block_ids,
                "token_count": self.tokenizer.count_tokens(full_content),
                "is_table": False,
                "part_index": part_idx,
                "total_parts": total_parts,
                "extraction_method": extraction_method_str,
            }
            if ocr_conf is not None:
                metadata_dict["ocr_confidence"] = ocr_conf

            chunk = PolicyChunk(
                chunk_id=chunk_id,
                policy_id=policy_id,
                clause_ref=eff_clause_ref,
                page_number=p_start,
                category=category,
                content_text=full_content,
                embedding=None,
                page_start=p_start,
                page_end=p_end,
                source_block_ids=section.source_block_ids,
                metadata=metadata_dict,
            )
            chunks.append(chunk)

        return chunks

    def _extract_atomic_legal_units(self, text: str) -> List[str]:
        """
        Splits text into paragraphs and sentences while keeping exceptions attached to rules.
        A sentence containing or starting with an exception marker ('Provided always',
        'Except that', 'Subject to', etc.) is bonded to the preceding sentence.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        atomic_units: List[str] = []

        for para in paragraphs:
            if len(para.splitlines()) > 1 and any(
                re.match(r"^(\([a-z0-9]+\)|[•\-*]|\d+\.)", line.strip())
                for line in para.splitlines()
            ):
                atomic_units.append(para)
                continue

            sentences = [s.strip() for s in RE_SENTENCE_BOUNDARY.split(para) if s.strip()]
            if len(sentences) <= 1:
                atomic_units.append(para)
                continue

            bonded_sentences: List[str] = []
            for s in sentences:
                is_exception_modifier = bool(
                    re.match(
                        r"^(provided\s+(always|that|further)|except\s+(that|for|when)|"
                        r"subject\s+to|unless|notwithstanding|with\s+the\s+exception|"
                        r"saving\s+and\s+except)\b",
                        s,
                        re.IGNORECASE,
                    )
                )

                if is_exception_modifier and bonded_sentences:
                    bonded_sentences[-1] = f"{bonded_sentences[-1]} {s}"
                else:
                    bonded_sentences.append(s)

            atomic_units.extend(bonded_sentences)

        return atomic_units

    def _split_oversized_unit(self, text: str, max_tokens: int) -> List[str]:
        """Splits an oversized legal unit across words/clauses without dropping content."""
        words = text.split()
        if not words:
            return [text] if text else []

        parts: List[str] = []
        curr: List[str] = []

        for w in words:
            candidate = " ".join(curr + [w]) if curr else w
            if curr and self.tokenizer.count_tokens(candidate) > max_tokens:
                parts.append(" ".join(curr))
                curr = [w]
            else:
                curr.append(w)

        if curr:
            parts.append(" ".join(curr))

        return parts

    # -------------------------------------------------------------------------
    # Table-Aware Chunking (Header Preservation, Row Integrity & Cell Splitting)
    # -------------------------------------------------------------------------
    def _chunk_table(
        self,
        table: StructuredTable,
        breadcrumb_trail: List[BreadcrumbItem],
        policy_id: str,
        policy_version: str,
        parent_category: ChunkCategory,
        parent_clause_ref: str,
        parent_section_id: str,
        ocr_conf_map: Dict[int, float],
    ) -> List[PolicyChunk]:
        """
        Chunks structured tables preserving headers and row relationships.
        No table may bypass max_chunk_size merely because it has two or fewer rows.
        Oversized cells are safely segmented while repeating Markdown headers.
        """
        breadcrumb_header = self._format_breadcrumbs(breadcrumb_trail)
        p_start = max(1, table.page_start or table.page_number)
        p_end = max(1, table.page_end or table.page_number)

        table_md = table.markdown.strip()
        full_table_text = (
            f"{breadcrumb_header}\n"
            f"[Table: {table.table_id} | Pages {p_start}-{p_end}]\n"
            f"{table_md}"
        )
        total_tokens = self.tokenizer.count_tokens(full_table_text)

        # Extraction method & OCR confidence
        ext_method = (
            table.extraction_method.value
            if hasattr(table.extraction_method, "value")
            else str(table.extraction_method)
        )
        is_ocr = ext_method in ("OCR", "SCANNED", "HYBRID")
        ocr_conf = ocr_conf_map.get(p_start) if is_ocr else None

        # 1. If table strictly fits within max_chunk_size, emit as a single chunk
        if total_tokens <= self.max_chunk_size:
            chunk_clause_ref = f"{parent_clause_ref} [Table: {table.table_id}]"
            chunk_id = self._generate_chunk_id(
                policy_id=policy_id,
                policy_version=policy_version,
                clause_ref=chunk_clause_ref,
                content_text=full_table_text,
                section_id=table.table_id,
                page_start=p_start,
            )
            meta: Dict[str, Any] = {
                "page_start": p_start,
                "page_end": p_end,
                "table_id": table.table_id,
                "source_block_ids": table.source_block_ids,
                "token_count": total_tokens,
                "is_table": True,
                "row_count": len(table.rows),
                "extraction_method": ext_method,
            }
            if ocr_conf is not None:
                meta["ocr_confidence"] = ocr_conf

            return [
                PolicyChunk(
                    chunk_id=chunk_id,
                    policy_id=policy_id,
                    clause_ref=chunk_clause_ref,
                    page_number=p_start,
                    category=parent_category,
                    content_text=full_table_text,
                    embedding=None,
                    page_start=p_start,
                    page_end=p_end,
                    source_block_ids=table.source_block_ids,
                    table_id=table.table_id,
                    metadata=meta,
                )
            ]

        # 2. Oversized table (including 1-row or 2-row tables with huge text):
        # Calculate available budget for table rows after breadcrumbs and headers
        col_count = len(table.headers) if table.headers else (len(table.rows[0]) if table.rows else 1)
        header_lines = self._build_table_header_markdown(table.headers, col_count)
        dummy_header = (
            f"{breadcrumb_header}\n"
            f"[Table: {table.table_id} (Rows 99/99) | Pages {p_start}-{p_end}]\n"
            f"{header_lines}\n"
        )
        base_tokens = self.tokenizer.count_tokens(dummy_header)
        available_row_tokens = max(10, self.max_chunk_size - base_tokens - 25)

        # Break table rows into token-compliant slices
        row_slices = self._slice_table_rows_with_cell_splitting(
            table=table,
            available_row_tokens=available_row_tokens,
        )

        chunks: List[PolicyChunk] = []
        total_parts = len(row_slices)

        for part_idx, slice_rows in enumerate(row_slices, start=1):
            slice_md = self._render_table_slice(table.headers, slice_rows)
            part_suffix = f" [Table Part {part_idx}/{total_parts}]"
            chunk_clause_ref = f"{parent_clause_ref} [Table: {table.table_id}]{part_suffix}"

            content = (
                f"{breadcrumb_header}\n"
                f"[Table: {table.table_id} (Rows {part_idx}/{total_parts}) | Pages {p_start}-{p_end}]\n"
                f"{slice_md}"
            )

            chunk_id = self._generate_chunk_id(
                policy_id=policy_id,
                policy_version=policy_version,
                clause_ref=chunk_clause_ref,
                content_text=content,
                section_id=f"{table.table_id}_p{part_idx}",
                page_start=p_start,
            )

            slice_meta: Dict[str, Any] = {
                "page_start": p_start,
                "page_end": p_end,
                "table_id": table.table_id,
                "source_block_ids": table.source_block_ids,
                "token_count": self.tokenizer.count_tokens(content),
                "is_table": True,
                "part_index": part_idx,
                "total_parts": total_parts,
                "row_count": len(slice_rows),
                "extraction_method": ext_method,
            }
            if ocr_conf is not None:
                slice_meta["ocr_confidence"] = ocr_conf

            chunk = PolicyChunk(
                chunk_id=chunk_id,
                policy_id=policy_id,
                clause_ref=chunk_clause_ref,
                page_number=p_start,
                category=parent_category,
                content_text=content,
                embedding=None,
                page_start=p_start,
                page_end=p_end,
                source_block_ids=table.source_block_ids,
                table_id=table.table_id,
                metadata=slice_meta,
            )
            chunks.append(chunk)

        return chunks

    def _slice_table_rows_with_cell_splitting(
        self,
        table: StructuredTable,
        available_row_tokens: int,
    ) -> List[List[List[Optional[str]]]]:
        """
        Slices table rows respecting available_row_tokens.
        If an individual row (e.g., in a 1-row or 2-row table) exceeds available_row_tokens,
        splits the largest narrative cell safely into sub-rows with '(cont.)' notation.
        """
        row_slices: List[List[List[Optional[str]]]] = []
        current_rows: List[List[Optional[str]]] = []
        current_tokens = 0

        for row in table.rows:
            row_md = self._format_table_row(row)
            r_tokens = self.tokenizer.count_tokens(row_md)

            # Case A: Row fits within single slice budget
            if r_tokens <= available_row_tokens:
                if current_tokens + r_tokens > available_row_tokens and current_rows:
                    row_slices.append(current_rows)
                    current_rows = [row]
                    current_tokens = r_tokens
                else:
                    current_rows.append(row)
                    current_tokens += r_tokens
                continue

            # Case B: Individual row itself exceeds available_row_tokens (e.g. 1-row or 2-row massive table)
            if current_rows:
                row_slices.append(current_rows)
                current_rows = []
                current_tokens = 0

            # Find cell with largest token length
            cell_tokens = [self.tokenizer.count_tokens(str(c or "")) for c in row]
            max_c_idx = cell_tokens.index(max(cell_tokens)) if cell_tokens else 0
            largest_cell_text = str(row[max_c_idx] or "")

            # Split largest cell into sub-units
            other_cells_cost = sum(cell_tokens) - cell_tokens[max_c_idx]
            # Account for markdown row pipes and possible "(cont.)" continuation tag
            cell_budget = max(10, available_row_tokens - other_cells_cost - 8)
            cell_parts = self._split_oversized_unit(largest_cell_text, cell_budget)

            for p_idx, part_text in enumerate(cell_parts, start=1):
                sub_row = list(row)
                sub_row[max_c_idx] = part_text
                # If subsequent part and row has multiple columns, mark first column with (cont.)
                if p_idx > 1 and len(sub_row) > 1 and sub_row[0]:
                    if "(cont.)" not in str(sub_row[0]):
                        sub_row[0] = f"{sub_row[0]} (cont.)"

                row_slices.append([sub_row])

        if current_rows:
            row_slices.append(current_rows)

        return row_slices

    def _build_table_header_markdown(self, headers: List[str], col_count: int) -> str:
        """Constructs markdown table header with separator."""
        cols = max(len(headers), col_count, 1)
        h_padded = [headers[i] if i < len(headers) and headers[i] else f"Col {i+1}" for i in range(cols)]
        return f"| {' | '.join(h_padded)} |\n| {' | '.join(['---'] * cols)} |"

    def _format_table_row(self, row: List[Optional[str]]) -> str:
        """Formats a single row into markdown."""
        return "| " + " | ".join(str(c or "").strip() for c in row) + " |"

    def _render_table_slice(self, headers: List[str], rows: List[List[Optional[str]]]) -> str:
        """Renders header and a slice of rows into clean markdown."""
        cols = max(len(headers), max((len(r) for r in rows), default=0))
        lines = [self._build_table_header_markdown(headers, cols)]
        for r in rows:
            lines.append(self._format_table_row(r))
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Breadcrumbs & Deterministic Chunk IDs
    # -------------------------------------------------------------------------
    def _format_breadcrumbs(self, trail: List[BreadcrumbItem]) -> str:
        """Formats breadcrumb trail into a clean hierarchical context block."""
        lines = []
        for item in trail:
            lines.append(f"{item.level_name}: {item.title}")
        return "\n".join(lines)

    def _generate_chunk_id(
        self,
        policy_id: str,
        policy_version: str,
        clause_ref: str,
        content_text: str,
        section_id: Optional[str] = None,
        page_start: Optional[int] = None,
    ) -> str:
        """
        Generates deterministic chunk IDs derived from:
        (policy_id, policy_version, section_id, clause_ref, page_start, content_text, algorithm_version).
        Guarantees idempotency, page discrimination for boilerplate, and version isolation.
        """
        norm_content = " ".join(content_text.split())
        sec_key = section_id or ""
        page_key = str(page_start) if page_start is not None else ""
        payload = f"{policy_id}|{policy_version}|{sec_key}|{clause_ref}|{page_key}|{norm_content}|{self.algorithm_version}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

        safe_policy = re.sub(r"[^a-zA-Z0-9]+", "_", policy_id).strip("_")[:16] or "pol"
        safe_clause = re.sub(r"[^a-zA-Z0-9]+", "_", clause_ref).strip("_")[:20] or "clause"

        return f"chk_{safe_policy}_{safe_clause}_{digest}"
