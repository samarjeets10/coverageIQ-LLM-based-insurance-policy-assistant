import pytest
from typing import List

from ai_service.schemas.ingestion import (
    BoundingBox,
    DocumentMetadata,
    ExtractionMethod,
    NodeType,
    StructuredDocument,
    StructuredSection,
    StructuredTable,
    StructuredTableCell,
)
from ai_service.schemas.structured_output import PolicyChunk, ChunkCategory, CitationItem
from ai_service.ingestion.semantic_chunker import (
    SemanticChunker,
    Tokenizer,
    BaseTokenizer,
    HuggingFaceTokenizer,
    TiktokenTokenizer,
)
from ai_service.retrieval.bm25_retriever import BM25Retriever
from ai_service.retrieval.dense_retriever import DenseVectorRetriever
from ai_service.database import MockKnowledgeBase
from ai_service.llm.generator import GroundedGenerator


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================
def make_metadata(page_count: int = 1, file_hash: str = "hash_abc_123") -> DocumentMetadata:
    return DocumentMetadata(
        title="Comprehensive Health Insurance Policy",
        file_size_bytes=2048,
        file_hash_sha256=file_hash,
        page_count=page_count,
        is_encrypted=False,
    )


# =============================================================================
# Test 1: Basic Clause Chunking
# =============================================================================
def test_basic_clause_chunking():
    """Scenario 1: Chunks standard sections into clause-level PolicyChunk units."""
    meta = make_metadata()
    sec1 = StructuredSection(
        section_id="sec_1",
        title="1. Definitions",
        clause_reference="1",
        node_type=NodeType.SECTION,
        category=ChunkCategory.DEFINITIONS,
        content="In this policy, words shall have specific meanings as defined herein.",
        page_start=1,
        page_end=1,
        source_block_ids=["b0"],
    )
    sec2 = StructuredSection(
        section_id="sec_2",
        title="2. Scope of Cover",
        clause_reference="2",
        node_type=NodeType.SECTION,
        category=ChunkCategory.COVERAGE,
        content="The company indemnifies reasonable and customary medical expenses.",
        page_start=1,
        page_end=1,
        source_block_ids=["b1"],
    )

    doc = StructuredDocument(
        doc_id="doc_health_01",
        document_title="Health Policy 2026",
        metadata=meta,
        sections=[sec1, sec2],
        tables=[],
        headers_and_footers=[],
        extraction_warnings=[],
        source_page_count=1,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    c1, c2 = chunks[0], chunks[1]

    assert c1.policy_id == "doc_health_01"
    assert c1.clause_ref == "1"
    assert c1.category == ChunkCategory.DEFINITIONS
    assert "words shall have specific meanings" in c1.content_text
    assert c1.page_number == 1

    assert c2.policy_id == "doc_health_01"
    assert c2.clause_ref == "2"
    assert c2.category == ChunkCategory.COVERAGE
    assert "indemnifies reasonable and customary" in c2.content_text
    assert c2.page_number == 1


# =============================================================================
# Test 2: Nested Hierarchy and Breadcrumbs
# =============================================================================
def test_nested_hierarchy_and_breadcrumbs():
    """Scenario 2: Retains full breadcrumb context trail (Policy > Part > Section > Clause)."""
    meta = make_metadata()
    subclause = StructuredSection(
        section_id="sec_2_1_1",
        title="2.1.1 Room Rent",
        clause_reference="2.1.1",
        node_type=NodeType.SUBCLAUSE,
        category=ChunkCategory.CONDITIONS,
        content="Room rent is capped at 1% of the Sum Insured per day.",
        page_start=2,
        page_end=2,
        source_block_ids=["b_sub"],
    )
    clause = StructuredSection(
        section_id="sec_2_1",
        title="2.1 Inpatient Care",
        clause_reference="2.1",
        node_type=NodeType.SUBSECTION,
        category=ChunkCategory.COVERAGE,
        content="Inpatient care requires at least 24 hours hospitalization.",
        page_start=2,
        page_end=2,
        source_block_ids=["b_clause"],
        children=[subclause],
    )
    major_sec = StructuredSection(
        section_id="part_1",
        title="PART I - POLICY BENEFITS",
        clause_reference="PART I",
        node_type=NodeType.MAJOR_SECTION,
        category=ChunkCategory.COVERAGE,
        content="",  # Grouping parent with no direct body text
        page_start=2,
        page_end=2,
        source_block_ids=[],
        children=[clause],
    )

    doc = StructuredDocument(
        doc_id="doc_hierarchy",
        document_title="Premier Health Plan",
        metadata=meta,
        sections=[major_sec],
        source_page_count=2,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    # Major section had no body text, so only 2 chunks for clause and subclause
    assert len(chunks) == 2
    c_sub = next(c for c in chunks if c.clause_ref == "2.1.1")

    # Breadcrumb verification
    assert "Policy: Premier Health Plan" in c_sub.content_text
    assert "Part: PART I - POLICY BENEFITS" in c_sub.content_text
    assert "Clause: 2.1 Inpatient Care" in c_sub.content_text
    assert "Subclause: 2.1.1 Room Rent" in c_sub.content_text
    assert "Room rent is capped at 1%" in c_sub.content_text


# =============================================================================
# Test 3: Parent Sections with No Body Text
# =============================================================================
def test_parent_sections_with_no_body_text():
    """Scenario 3: Empty grouping parent sections supply context without creating vacuous chunks."""
    meta = make_metadata()
    sub1 = StructuredSection(
        section_id="s1",
        title="2.1 ICU",
        clause_reference="2.1",
        node_type=NodeType.SUBSECTION,
        category=ChunkCategory.COVERAGE,
        content="ICU expenses covered up to 2% SI.",
        page_start=3,
        page_end=3,
    )
    sub2 = StructuredSection(
        section_id="s2",
        title="2.2 Ambulance",
        clause_reference="2.2",
        node_type=NodeType.SUBSECTION,
        category=ChunkCategory.COVERAGE,
        content="Road ambulance charges up to Rs 2,000.",
        page_start=3,
        page_end=3,
    )
    parent = StructuredSection(
        section_id="p_root",
        title="SECTION 2 - HOSPITALIZATION BENEFITS",
        clause_reference="2",
        node_type=NodeType.SECTION,
        category=ChunkCategory.COVERAGE,
        content="   \n   ",  # Pure whitespace
        page_start=3,
        page_end=3,
        children=[sub1, sub2],
    )

    doc = StructuredDocument(
        doc_id="doc_empty_parent",
        metadata=meta,
        sections=[parent],
        source_page_count=3,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert all(c.content_text.strip() for c in chunks)
    for c in chunks:
        assert "SECTION 2 - HOSPITALIZATION BENEFITS" in c.content_text


# =============================================================================
# Test 4: Legal Exceptions Preserved
# =============================================================================
def test_legal_exceptions_preserved():
    """Scenario 4: Provisos ('Provided always that', 'Except') remain attached to operative rule."""
    meta = make_metadata()
    legal_rule = (
        "Room rent is covered up to 1% of the Sum Insured per day. "
        "Provided always that if the Insured Person occupies a room with higher tariff, "
        "proportionate deduction shall apply to all associated medical charges."
    )
    sec = StructuredSection(
        section_id="sec_rent",
        title="3.2 Room Rent Sub-limits",
        clause_reference="3.2",
        node_type=NodeType.SUBSECTION,
        category=ChunkCategory.CONDITIONS,
        content=legal_rule,
        page_start=5,
        page_end=5,
    )

    doc = StructuredDocument(
        doc_id="doc_proviso",
        metadata=meta,
        sections=[sec],
        source_page_count=5,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    content = chunks[0].content_text
    assert "Room rent is covered up to 1%" in content
    assert "Provided always that if the Insured Person occupies a room with higher tariff" in content


# =============================================================================
# Test 5: Conditions and Qualifiers Preserved
# =============================================================================
def test_conditions_and_qualifiers_preserved():
    """Scenario 5: Waiting period and eligibility qualifiers stay bonded to benefit statement."""
    meta = make_metadata()
    content = (
        "Maternity delivery expenses are covered up to Rs 50,000 for normal delivery. "
        "Subject to a mandatory waiting period of 24 consecutive months of continuous coverage."
    )
    sec = StructuredSection(
        section_id="sec_maternity",
        title="5.3 Maternity Cover",
        clause_reference="5.3",
        node_type=NodeType.SECTION,
        category=ChunkCategory.CONDITIONS,
        content=content,
        page_start=12,
        page_end=12,
    )

    doc = StructuredDocument(
        doc_id="doc_maternity",
        metadata=meta,
        sections=[sec],
        source_page_count=12,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert "Subject to a mandatory waiting period of 24 consecutive months" in chunks[0].content_text


# =============================================================================
# Test 6: Coverage and Exclusion Separation
# =============================================================================
def test_coverage_and_exclusion_separation():
    """Scenario 6: Coverage clauses and Exclusions are never combined into a single chunk."""
    meta = make_metadata()
    sec_cov = StructuredSection(
        section_id="cov_1",
        title="1. Covered Hospitalization",
        clause_reference="1",
        node_type=NodeType.SECTION,
        category=ChunkCategory.COVERAGE,
        content="All direct inpatient treatments are covered.",
        page_start=1,
        page_end=1,
    )
    sec_excl = StructuredSection(
        section_id="excl_1",
        title="2. Cosmetic Exclusions",
        clause_reference="2",
        node_type=NodeType.SECTION,
        category=ChunkCategory.EXCLUSIONS,
        content="Plastic and cosmetic surgery is strictly excluded.",
        page_start=1,
        page_end=1,
    )

    doc = StructuredDocument(
        doc_id="doc_sep",
        metadata=meta,
        sections=[sec_cov, sec_excl],
        source_page_count=1,
    )

    chunker = SemanticChunker(target_chunk_size=1000)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    c1, c2 = chunks[0], chunks[1]

    assert c1.category == ChunkCategory.COVERAGE
    assert "strictly excluded" not in c1.content_text

    assert c2.category == ChunkCategory.EXCLUSIONS
    assert "treatments are covered" not in c2.content_text


# =============================================================================
# Test 7: Table-Aware Chunk Generation
# =============================================================================
def test_table_aware_chunk_generation():
    """Scenario 7: Generates table chunk preserving headers, markdown, and row relationships."""
    meta = make_metadata()
    tbl = StructuredTable(
        table_id="tbl_daycare",
        page_number=4,
        page_start=4,
        page_end=4,
        bbox=BoundingBox(x0=50, y0=100, x1=400, y1=250),
        headers=["Procedure", "Limit"],
        rows=[
            ["Chemotherapy", "Day Care"],
            ["Radiotherapy", "Day Care"],
        ],
        cells=[
            StructuredTableCell(row_idx=0, col_idx=0, text="Procedure", is_header=True),
            StructuredTableCell(row_idx=0, col_idx=1, text="Limit", is_header=True),
            StructuredTableCell(row_idx=1, col_idx=0, text="Chemotherapy", is_header=False),
            StructuredTableCell(row_idx=1, col_idx=1, text="Day Care", is_header=False),
            StructuredTableCell(row_idx=2, col_idx=0, text="Radiotherapy", is_header=False),
            StructuredTableCell(row_idx=2, col_idx=1, text="Day Care", is_header=False),
        ],
        markdown="| Procedure | Limit |\n| --- | --- |\n| Chemotherapy | Day Care |\n| Radiotherapy | Day Care |",
        source_block_ids=["tbl_b1"],
    )

    doc = StructuredDocument(
        doc_id="doc_tbl",
        metadata=meta,
        sections=[],
        tables=[tbl],
        source_page_count=4,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    t_chunk = chunks[0]
    assert t_chunk.metadata.get("is_table") is True
    assert t_chunk.table_id == "tbl_daycare"
    assert "| Procedure | Limit |" in t_chunk.content_text
    assert "| Chemotherapy | Day Care |" in t_chunk.content_text
    assert t_chunk.page_number == 4


# =============================================================================
# Test 8: Multi-Page Table Provenance
# =============================================================================
def test_multipage_table_provenance():
    """Scenario 8: Preserves page_start and page_end for continuation tables spanning multiple pages."""
    meta = make_metadata(page_count=5)
    tbl = StructuredTable(
        table_id="tbl_multipage",
        page_number=2,
        page_start=2,
        page_end=3,
        bbox=BoundingBox(x0=50, y0=500, x1=450, y1=750),
        headers=["Treatment", "Co-Pay"],
        rows=[
            ["Cataract", "10%"],
            ["Joint Replacement", "20%"],
        ],
        markdown="| Treatment | Co-Pay |\n| --- | --- |\n| Cataract | 10% |\n| Joint Replacement | 20% |",
        source_block_ids=["p2_b1", "p3_b0"],
    )

    doc = StructuredDocument(
        doc_id="doc_multi_tbl",
        metadata=meta,
        sections=[],
        tables=[tbl],
        source_page_count=5,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.page_number == 2
    assert c.page_start == 2
    assert c.page_end == 3
    assert c.metadata.get("page_start") == 2
    assert c.metadata.get("page_end") == 3
    assert "Pages 2-3" in c.content_text


# =============================================================================
# Test 9: Token-Aware Splitting
# =============================================================================
def test_token_aware_splitting():
    """Scenario 9: Respects token boundaries using genuine tokenizer without character heuristics."""
    tokenizer = Tokenizer()
    token_count = tokenizer.count_tokens("Section 1: Inpatient Care Benefits. Coverage up to Sum Insured.")
    assert token_count > 0
    assert isinstance(token_count, int)

    meta = make_metadata()
    sec = StructuredSection(
        section_id="s_tok",
        title="1. Short Clause",
        clause_reference="1",
        node_type=NodeType.SECTION,
        category=ChunkCategory.GENERAL,
        content="Standard short clause text within token limits.",
        page_start=1,
        page_end=1,
    )
    doc = StructuredDocument(doc_id="doc_tok", metadata=meta, sections=[sec], source_page_count=1)

    chunker = SemanticChunker(target_chunk_size=100, max_chunk_size=200)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunker.tokenizer.count_tokens(chunks[0].content_text) <= 200


# =============================================================================
# Test 10: Oversized Clauses Splitting
# =============================================================================
def test_oversized_clauses():
    """Scenario 10: Gracefully splits oversized clauses across parts while preserving breadcrumbs."""
    meta = make_metadata()
    # Generate large narrative text exceeding 100 tokens
    sentences = [f"Sub-provision {i}: Specific medical treatment shall be payable under plan {i} terms." for i in range(1, 20)]
    long_content = " ".join(sentences)

    sec = StructuredSection(
        section_id="sec_huge",
        title="Section 4. Extensive Provisions",
        clause_reference="4",
        node_type=NodeType.SECTION,
        category=ChunkCategory.COVERAGE,
        content=long_content,
        page_start=1,
        page_end=1,
    )
    doc = StructuredDocument(doc_id="doc_oversized", metadata=meta, sections=[sec], source_page_count=1)

    # Use a small max_chunk_size to force splitting
    chunker = SemanticChunker(target_chunk_size=50, max_chunk_size=80)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for i, c in enumerate(chunks, start=1):
        assert f"[Part {i}/" in c.clause_ref
        assert "Section: Section 4. Extensive Provisions" in c.content_text
        assert c.category == ChunkCategory.COVERAGE


# =============================================================================
# Test 11: Deterministic Chunk IDs
# =============================================================================
def test_deterministic_chunk_ids():
    """Scenario 11: The exact same input document produces bitwise identical chunk IDs."""
    meta = make_metadata()
    sec = StructuredSection(
        section_id="sec_det",
        title="2. Deterministic Clause",
        clause_reference="2",
        node_type=NodeType.SECTION,
        category=ChunkCategory.DEFINITIONS,
        content="Exact verbatim text of the definitions clause.",
        page_start=1,
        page_end=1,
    )
    doc = StructuredDocument(doc_id="doc_det", metadata=meta, sections=[sec], source_page_count=1)

    chunker = SemanticChunker()
    run1 = chunker.chunk_document(doc, policy_id="pol_1", policy_version="v1.0")
    run2 = chunker.chunk_document(doc, policy_id="pol_1", policy_version="v1.0")

    assert len(run1) == 1
    assert len(run2) == 1
    assert run1[0].chunk_id == run2[0].chunk_id
    assert run1[0].chunk_id.startswith("chk_pol_1_2_")


# =============================================================================
# Test 12: Different Policy Versions Produce Distinct Chunk IDs
# =============================================================================
def test_different_policy_versions():
    """Scenario 12: Different policy versions produce distinct chunk IDs for identical text."""
    meta = make_metadata()
    sec = StructuredSection(
        section_id="sec_ver",
        title="2. Versioned Clause",
        clause_reference="2",
        node_type=NodeType.SECTION,
        category=ChunkCategory.COVERAGE,
        content="Identical clause wording across versions.",
        page_start=1,
        page_end=1,
    )
    doc = StructuredDocument(doc_id="doc_ver", metadata=meta, sections=[sec], source_page_count=1)

    chunker = SemanticChunker()
    v1_chunks = chunker.chunk_document(doc, policy_id="pol_gold", policy_version="2025.1")
    v2_chunks = chunker.chunk_document(doc, policy_id="pol_gold", policy_version="2026.1")

    assert v1_chunks[0].chunk_id != v2_chunks[0].chunk_id


# =============================================================================
# Test 13: Missing Page References Handled Safely
# =============================================================================
def test_missing_page_references():
    """Scenario 13: Handles sections with zero/missing page numbers without crashing or negative pages."""
    meta = make_metadata()
    sec = StructuredSection(
        section_id="sec_nopage",
        title="Unpaged Section",
        clause_reference="1",
        node_type=NodeType.SECTION,
        category=ChunkCategory.GENERAL,
        content="Clause text missing explicit page number.",
        page_start=1,
        page_end=1,
    )
    doc = StructuredDocument(doc_id="doc_nopage", metadata=meta, sections=[sec], source_page_count=1)

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].page_number >= 1
    assert chunks[0].page_start >= 1


# =============================================================================
# Test 14: OCR-Derived Content Provenance
# =============================================================================
def test_ocr_derived_content():
    """Scenario 14: Preserves provenance from scanned OCR blocks into PolicyChunk."""
    meta = make_metadata()
    sec = StructuredSection(
        section_id="sec_ocr",
        title="Scanned Exclusion",
        clause_reference="4.1",
        node_type=NodeType.SECTION,
        category=ChunkCategory.EXCLUSIONS,
        content="Cosmetic treatments are excluded as per scanned record.",
        page_start=2,
        page_end=2,
        source_block_ids=["ocr_b1", "ocr_b2"],
    )
    doc = StructuredDocument(doc_id="doc_ocr", metadata=meta, sections=[sec], source_page_count=2)

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert "ocr_b1" in chunks[0].source_block_ids
    assert chunks[0].category == ChunkCategory.EXCLUSIONS


# =============================================================================
# Test 15: Hybrid Document Content
# =============================================================================
def test_hybrid_document_content():
    """Scenario 15: Chunks hybrid document containing digital text and OCR-derived tables."""
    meta = make_metadata()
    sec = StructuredSection(
        section_id="sec_hybrid",
        title="Hybrid Section",
        clause_reference="3",
        node_type=NodeType.SECTION,
        category=ChunkCategory.CONDITIONS,
        content="Digital narrative text introducing table below.\n[Table: tbl_hybrid]",
        page_start=1,
        page_end=1,
        source_block_ids=["b_digital"],
    )
    tbl = StructuredTable(
        table_id="tbl_hybrid",
        page_number=1,
        bbox=BoundingBox(x0=50, y0=200, x1=400, y1=350),
        headers=["Tier", "Co-Pay"],
        rows=[["Tier 1", "5%"], ["Tier 2", "10%"]],
        markdown="| Tier | Co-Pay |\n| --- | --- |\n| Tier 1 | 5% |\n| Tier 2 | 10% |",
        source_block_ids=["b_table_ocr"],
        extraction_method=ExtractionMethod.HYBRID,
    )
    doc = StructuredDocument(doc_id="doc_hybrid", metadata=meta, sections=[sec], tables=[tbl], source_page_count=1)

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    types = {c.metadata.get("is_table") for c in chunks}
    assert False in types  # Narrative chunk
    assert True in types   # Table chunk


# =============================================================================
# Test 16: PolicyChunk Backward Compatibility with Retrievers
# =============================================================================
def test_policy_chunk_backward_compatibility(monkeypatch):
    """Scenario 16: Chunks generated by SemanticChunker seamlessly integrate with existing retrievers."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "offline")
    meta = make_metadata()
    sec1 = StructuredSection(
        section_id="sec_ret",
        title="2.1 Inpatient Hospitalization",
        clause_reference="2.1",
        node_type=NodeType.SECTION,
        category=ChunkCategory.COVERAGE,
        content="Medical expenses incurred during inpatient hospitalization are indemnified up to the Sum Insured.",
        page_start=1,
        page_end=1,
    )
    sec2 = StructuredSection(
        section_id="sec_excl",
        title="4.1 Cosmetic Exclusions",
        clause_reference="4.1",
        node_type=NodeType.SECTION,
        category=ChunkCategory.EXCLUSIONS,
        content="Cosmetic and aesthetic plastic surgery is not covered under any circumstances.",
        page_start=1,
        page_end=1,
    )
    sec3 = StructuredSection(
        section_id="sec_mat",
        title="5.3 Maternity Cover",
        clause_reference="5.3",
        node_type=NodeType.SECTION,
        category=ChunkCategory.CONDITIONS,
        content="Maternity delivery medical expenses are subject to waiting periods.",
        page_start=1,
        page_end=1,
    )
    doc = StructuredDocument(doc_id="doc_retrieval_compat", metadata=meta, sections=[sec1, sec2, sec3], source_page_count=1)

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 3

    # 1. BM25 Search
    bm25 = BM25Retriever()
    bm25_res = bm25.search("inpatient hospitalization expenses", chunks, top_n=3)
    assert len(bm25_res) == 3
    assert bm25_res[0][0].chunk_id == chunks[0].chunk_id
    assert bm25_res[0][1] > 0.0

    # 2. Dense Vector Search
    dense = DenseVectorRetriever()
    dense_res = dense.search("inpatient hospitalization expenses", chunks, top_n=3)
    assert len(dense_res) == 3
    assert dense_res[0][0].chunk_id == chunks[0].chunk_id
    assert dense_res[0][1] > 0.0

    # 3. Mock Knowledge Base Storage
    kb = MockKnowledgeBase()
    kb.store_chunks("doc_retrieval_compat", chunks)
    stored = kb.get_chunks("doc_retrieval_compat")
    assert len(stored) == 3
    assert {s.chunk_id for s in stored} == {c.chunk_id for c in chunks}


# =============================================================================
# Test 17: Empty and Malformed Documents Handled Safely
# =============================================================================
def test_empty_and_malformed_documents():
    """Scenario 17: Empty documents return empty chunk lists without raising unhandled exceptions."""
    meta = make_metadata()
    empty_doc = StructuredDocument(
        doc_id="doc_empty",
        metadata=meta,
        sections=[],
        tables=[],
        headers_and_footers=[],
        extraction_warnings=[],
        source_page_count=0,
    )
    chunker = SemanticChunker()
    chunks = chunker.chunk_document(empty_doc)
    assert chunks == []

    # Section with empty whitespace content
    whitespace_sec = StructuredSection(
        section_id="sec_ws",
        title="Empty Title",
        node_type=NodeType.SECTION,
        content="   \n\n\t  ",
        page_start=1,
        page_end=1,
    )
    doc_ws = StructuredDocument(doc_id="doc_ws", metadata=meta, sections=[whitespace_sec], source_page_count=1)
    chunks_ws = chunker.chunk_document(doc_ws)
    assert chunks_ws == []


# =============================================================================
# Test 18: No Duplicate Chunks
# =============================================================================
def test_no_duplicate_chunks():
    """Scenario 18: Complex multi-section and table documents emit 100% unique chunk IDs."""
    meta = make_metadata()
    sections = [
        StructuredSection(
            section_id=f"sec_{i}",
            title=f"Section {i}. General Clause {i}",
            clause_reference=str(i),
            node_type=NodeType.SECTION,
            category=ChunkCategory.GENERAL,
            content=f"Operative clause text for section {i}.",
            page_start=i,
            page_end=i,
        )
        for i in range(1, 6)
    ]
    doc = StructuredDocument(doc_id="doc_unique", metadata=meta, sections=sections, source_page_count=5)

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 5
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# =============================================================================
# Test 19: BGE-M3 Tokenizer Configuration & Fallback
# =============================================================================
def test_bge_m3_tokenizer_configuration():
    """Scenario 19: Supports BGE-M3 configuration with lazy loading and offline fallback."""
    chunker = SemanticChunker(tokenizer_name="BAAI/bge-m3")
    assert isinstance(chunker.tokenizer, HuggingFaceTokenizer)
    assert "BAAI/bge-m3" in chunker.tokenizer.name

    text = "Section 2.1 Inpatient Hospitalization Medical Expenses up to Sum Insured."
    count = chunker.tokenizer.count_tokens(text)
    assert count > 0
    assert isinstance(count, int)


# =============================================================================
# Test 20: Tokenizer Dependency Injection
# =============================================================================
def test_tokenizer_dependency_injection():
    """Scenario 20: Allows custom BaseTokenizer injection for unit tests without model files."""
    class MockCustomTokenizer(BaseTokenizer):
        def count_tokens(self, text: str) -> int:
            return len(text.split())

        @property
        def name(self) -> str:
            return "MockCustomTokenizer"

    custom_tok = MockCustomTokenizer()
    chunker = SemanticChunker(tokenizer=custom_tok)
    assert chunker.tokenizer is custom_tok
    assert chunker.tokenizer.count_tokens("five words in this test") == 5


# =============================================================================
# Test 21: Oversized One-Row Table Splitting
# =============================================================================
def test_oversized_one_row_table():
    """Scenario 21: One-row table with large text (> max_chunk_size) is split without bypassing limits."""
    meta = make_metadata()
    long_narrative = (
        "Under Plan Comprehensive, the insured is entitled to complete reimbursement of medical charges, "
        "including ICU costs, specialist visits, pharmacy bills, diagnostic investigations, advanced robotic "
        "surgeries, organ donor expenses, non-medical items where specified, and emergency ground evacuation "
        "subject to all mandatory waiting periods and territorial jurisdiction terms specified in Part IV."
    )
    tbl = StructuredTable(
        table_id="tbl_onerow_huge",
        page_number=1,
        bbox=BoundingBox(x0=50, y0=100, x1=400, y1=250),
        headers=["Plan Tier", "Terms & Coverage Limits"],
        rows=[["Gold Ultra", long_narrative]],
        markdown=f"| Plan Tier | Terms & Coverage Limits |\n| --- | --- |\n| Gold Ultra | {long_narrative} |",
        source_block_ids=["tbl_b1"],
    )
    doc = StructuredDocument(doc_id="doc_onerow_tbl", metadata=meta, sections=[], tables=[tbl], source_page_count=1)

    # Use max_chunk_size=85 to force splitting of this oversized 1-row table (~110 total tokens)
    chunker = SemanticChunker(max_chunk_size=85)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for c in chunks:
        assert chunker.tokenizer.count_tokens(c.content_text) <= 85
        assert "| Plan Tier | Terms & Coverage Limits |" in c.content_text
        assert c.metadata.get("is_table") is True


# =============================================================================
# Test 22: Oversized Two-Row Table Splitting
# =============================================================================
def test_oversized_two_row_table():
    """Scenario 22: Two-row table with large text does not bypass max_chunk_size."""
    meta = make_metadata()
    long_desc1 = "Paragraph 1 describing Silver plan conditions, sub-limits, proportional deductions, and waiting periods."
    long_desc2 = "Paragraph 2 describing Gold plan conditions, comprehensive copays, room rent caps, and ICU limits."
    tbl = StructuredTable(
        table_id="tbl_tworow_huge",
        page_number=2,
        bbox=BoundingBox(x0=50, y0=100, x1=400, y1=250),
        headers=["Category", "Provisions"],
        rows=[["Silver", long_desc1], ["Gold", long_desc2]],
        markdown=f"| Category | Provisions |\n| --- | --- |\n| Silver | {long_desc1} |\n| Gold | {long_desc2} |",
        source_block_ids=["tbl_b2"],
    )
    doc = StructuredDocument(doc_id="doc_tworow_tbl", metadata=meta, sections=[], tables=[tbl], source_page_count=2)

    # Use max_chunk_size=75 to force splitting of this 2-row table (~100 total tokens)
    chunker = SemanticChunker(max_chunk_size=75)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for c in chunks:
        assert chunker.tokenizer.count_tokens(c.content_text) <= 75
        assert "| Category | Provisions |" in c.content_text


# =============================================================================
# Test 23: Multi-Page Citation Range Generation
# =============================================================================
def test_multipage_citation_range_generation():
    """Scenario 23: Multi-page chunks generate CitationItem with correct page_range and page_end."""
    chunk = PolicyChunk(
        chunk_id="chk_multi_01",
        policy_id="pol_test",
        clause_ref="Section 5. Table of Benefits",
        page_number=6,
        page_start=6,
        page_end=7,
        category=ChunkCategory.COVERAGE,
        content_text="Detailed benefits schedule spanning pages 6 to 7.",
        metadata={"page_start": 6, "page_end": 7},
    )

    cit = CitationItem(
        chunk_id=chunk.chunk_id,
        page=chunk.page_number,
        page_end=chunk.page_end,
        clause=chunk.clause_ref,
        category=chunk.category.value,
        snippet=chunk.content_text,
    )
    assert cit.page == 6
    assert cit.page_end == 7
    assert cit.page_range == "6-7"

    # Test evidence format in GroundedGenerator
    generator = GroundedGenerator()
    evidence_text = generator._format_evidence_chunks([chunk])
    assert "PAGE: 6-7" in evidence_text


# =============================================================================
# Test 24: Single-Page Citation Backward Compatibility
# =============================================================================
def test_single_page_citation_backward_compatibility():
    """Scenario 24: Single-page chunks emit standard page integer and string page_range."""
    cit = CitationItem(
        chunk_id="chk_single_01",
        page=12,
        clause="2.1 Inpatient Care",
        snippet="Coverage text.",
    )
    assert cit.page == 12
    assert cit.page_end is None
    assert cit.page_range == "12"


# =============================================================================
# Test 25: Deterministic Chunk ID Page Discrimination
# =============================================================================
def test_deterministic_chunk_id_page_discrimination():
    """Scenario 25: Identical boilerplate on different pages produces distinct chunk IDs."""
    meta = make_metadata()
    sec_page2 = StructuredSection(
        section_id="sec_notice_p2",
        title="Statutory Notice",
        clause_reference="Notice",
        node_type=NodeType.SECTION,
        category=ChunkCategory.GENERAL,
        content="The insurer reserves the right to modify terms with prior 90-day notification.",
        page_start=2,
        page_end=2,
    )
    sec_page10 = StructuredSection(
        section_id="sec_notice_p10",
        title="Statutory Notice",
        clause_reference="Notice",
        node_type=NodeType.SECTION,
        category=ChunkCategory.GENERAL,
        content="The insurer reserves the right to modify terms with prior 90-day notification.",
        page_start=10,
        page_end=10,
    )
    doc = StructuredDocument(
        doc_id="doc_boilerplate",
        metadata=meta,
        sections=[sec_page2, sec_page10],
        source_page_count=10,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    assert chunks[0].chunk_id != chunks[1].chunk_id
    assert chunks[0].page_number == 2
    assert chunks[1].page_number == 10


# =============================================================================
# Test 26: OCR Confidence & Extraction Provenance in Metadata
# =============================================================================
def test_ocr_confidence_provenance_metadata():
    """Scenario 26: Retains OCR confidence for scanned content, with None for digital text."""
    meta = make_metadata()
    sec_digital = StructuredSection(
        section_id="sec_dig",
        title="Digital Terms",
        clause_reference="1",
        node_type=NodeType.SECTION,
        category=ChunkCategory.COVERAGE,
        content="Native digital vector text without OCR.",
        page_start=1,
        page_end=1,
        source_block_ids=["block_dig_0"],
    )
    sec_ocr = StructuredSection(
        section_id="sec_scanned",
        title="Scanned Exclusion",
        clause_reference="2",
        node_type=NodeType.SECTION,
        category=ChunkCategory.EXCLUSIONS,
        content="Scanned text recognized by PaddleOCR engine.",
        page_start=2,
        page_end=2,
        source_block_ids=["ocr_b1", "doc_ocr_p2_b0"],
    )
    doc = StructuredDocument(
        doc_id="doc_provenance_test",
        metadata=meta,
        sections=[sec_digital, sec_ocr],
        source_page_count=2,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(doc, ocr_confidence_by_page={2: 0.94})

    assert len(chunks) == 2
    c_dig, c_ocr = chunks[0], chunks[1]

    # Digital chunk
    assert c_dig.metadata.get("extraction_method") == "DIGITAL"
    assert c_dig.metadata.get("ocr_confidence") is None

    # OCR chunk
    assert c_ocr.metadata.get("extraction_method") == "OCR"
    assert c_ocr.metadata.get("ocr_confidence") == 0.94


# =============================================================================
# Test 27: Overlap Configuration Behavior
# =============================================================================
def test_overlap_configuration_behavior():
    """Scenario 27: Safe sentence overlap is applied across parts only when overlap_tokens > 0."""
    meta = make_metadata()
    long_content = (
        "Sentence one establishes basic policy cover. "
        "Sentence two provides ICU bed limits. "
        "Sentence three defines road ambulance charges. "
        "Sentence four outlines pre-hospitalization allowances."
    )
    sec = StructuredSection(
        section_id="sec_ovl",
        title="1. Multi-part Clause",
        clause_reference="1",
        node_type=NodeType.SECTION,
        category=ChunkCategory.COVERAGE,
        content=long_content,
        page_start=1,
        page_end=1,
    )
    doc = StructuredDocument(doc_id="doc_ovl", metadata=meta, sections=[sec], source_page_count=1)

    # 1. Non-overlapping (default)
    chunker_no_ovl = SemanticChunker(target_chunk_size=20, max_chunk_size=30, overlap_tokens=0)
    chunks_no_ovl = chunker_no_ovl.chunk_document(doc)
    assert len(chunks_no_ovl) > 1
    assert "[Context:" not in chunks_no_ovl[1].content_text

    # 2. Safe overlap enabled
    chunker_with_ovl = SemanticChunker(target_chunk_size=20, max_chunk_size=35, overlap_tokens=15)
    chunks_with_ovl = chunker_with_ovl.chunk_document(doc)
    assert len(chunks_with_ovl) > 1
    assert "[Context:" in chunks_with_ovl[1].content_text

