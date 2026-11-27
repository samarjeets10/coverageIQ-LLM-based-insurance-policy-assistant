from ai_service.retrieval.query_analyzer import QueryAnalyzer

def test_query_analyzer_intent_and_entities():
    analyzer = QueryAnalyzer()
    
    # Coverage intent with entities
    q1 = "Is in-patient hospitalization covered after 24 hours?"
    res1 = analyzer.analyze(q1)
    assert res1.intent == "COVERAGE_CHECK"
    assert "hospital" in res1.entities
    assert any("in-patient" in term for term in res1.expanded_terms)

    # Exclusion intent
    q2 = "Is cosmetic surgery excluded under Section 4.1?"
    res2 = analyzer.analyze(q2)
    assert res2.intent == "EXCLUSION_INQUIRY"
    assert "cosmetic" in res2.entities
    assert any("section 4.1" in entity for entity in res2.entities)

    # Waiting period intent
    q3 = "What is the waiting period for maternity delivery?"
    res3 = analyzer.analyze(q3)
    assert res3.intent == "WAITING_PERIOD"
    assert "maternity" in res3.entities
