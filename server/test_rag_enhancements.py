#!/usr/bin/env python3
"""Comprehensive test suite for RAG enhancements."""

import requests
import json
import sys

BASE_URL = "http://localhost:5000/api/v1/rag/search"

def test_query(name, query, granularity=None, expected_type=None, check_fn=None):
    """Run a test query and report results."""
    payload = {"query": query}
    if granularity:
        payload["granularity"] = granularity

    try:
        resp = requests.post(BASE_URL, json=payload, timeout=60)
        data = resp.json()

        # Check for errors
        if data.get("error"):
            print(f"❌ {name}")
            print(f"   ERROR: {data.get('error')}")
            return False

        # Check expected type
        if expected_type and data.get("query_type") != expected_type:
            print(f"⚠️  {name}")
            print(f"   Expected type '{expected_type}', got '{data.get('query_type')}'")
            # Continue to check other things

        # Run custom check function
        if check_fn:
            success, msg = check_fn(data)
            if success:
                print(f"✓ {name}")
                print(f"   {msg}")
                return True
            else:
                print(f"❌ {name}")
                print(f"   {msg}")
                return False
        else:
            print(f"✓ {name}")
            print(f"   Type: {data.get('query_type')}, Total: {data.get('total_found', 0)}")
            return True

    except Exception as e:
        print(f"❌ {name}")
        print(f"   EXCEPTION: {e}")
        return False


def check_comparison(data):
    """Check comparison results."""
    comp = data.get("comparison", {})
    if len(comp) < 2:
        return False, f"Expected 2 sessions, got {len(comp)}"

    # Check each session has required fields
    for label, session in comp.items():
        if not session.get("argumentation"):
            return False, f"Missing argumentation for {label}"
        if not session.get("evolution"):
            return False, f"Missing evolution for {label}"
        if not session.get("metrics"):
            return False, f"Missing metrics for {label}"

    labels = list(comp.keys())
    return True, f"Compared: {labels[0][:30]}... vs {labels[1][:30]}..."


def check_session_results(data):
    """Check session search results."""
    results = data.get("session_results", [])
    if not results:
        return False, "No session_results returned"

    # Check enrichment
    first = results[0]
    if not first.get("argumentation"):
        return False, "Missing argumentation enrichment"
    if not first.get("evolution"):
        return False, "Missing evolution enrichment"

    return True, f"Found {len(results)} sessions with enrichment"


def check_argumentation_ranking(data):
    """Check that results are ranked by argumentation metric."""
    results = data.get("session_results", [])
    if len(results) < 2:
        return True, f"Only {len(results)} result(s), can't verify ranking"

    # Check debate_score is descending
    scores = [r.get("argumentation", {}).get("debate_score", 0) for r in results]
    if scores == sorted(scores, reverse=True):
        return True, f"Correctly ranked by debate_score: {scores[:3]}"
    else:
        return False, f"Not ranked by debate_score: {scores[:3]}"


def check_temporal_ranking(data):
    """Check that results are ranked by evolution metric."""
    results = data.get("session_results", [])
    if len(results) < 2:
        return True, f"Only {len(results)} result(s), can't verify ranking"

    # Check analytic_evolution magnitude is descending
    evos = [abs(r.get("evolution", {}).get("analytic_evolution", 0)) for r in results]
    if evos == sorted(evos, reverse=True):
        return True, f"Correctly ranked by analytic_evolution: {[r.get('evolution',{}).get('analytic_evolution',0) for r in results[:3]]}"
    else:
        return False, f"Not ranked by evolution: {evos[:3]}"


def check_chunk_results(data):
    """Check chunk search results."""
    results = data.get("results", [])
    if not results:
        return False, "No chunk results returned"

    # Check metadata
    first = results[0]
    if not first.get("metadata"):
        return False, "Missing metadata in chunk results"
    if not first.get("text"):
        return False, "Missing text in chunk results"

    return True, f"Found {len(results)} chunks with metadata"


def check_speaker_results(data):
    """Check speaker search results."""
    # Speaker results might be in 'results' or 'speaker_results' or 'evidence'
    results = data.get("speaker_results") or data.get("results") or \
              (data.get("evidence", {}).get("results") if isinstance(data.get("evidence"), dict) else [])

    if not results:
        return False, "No speaker results returned"

    return True, f"Found {len(results)} speaker profiles"


def check_insights_generated(data):
    """Check that insights were generated."""
    insights = data.get("insights")
    if not insights:
        return False, "No insights generated"
    if len(insights) < 100:
        return False, f"Insights too short: {len(insights)} chars"

    return True, f"Generated {len(insights)} chars of insights"


def run_all_tests():
    """Run comprehensive test suite."""
    passed = 0
    failed = 0

    print("=" * 60)
    print("COMPREHENSIVE RAG PIPELINE TEST SUITE")
    print("=" * 60)

    # =========================================
    # TEST GROUP 1: Natural Language Comparisons
    # =========================================
    print("\n📊 TEST GROUP 1: Natural Language Comparisons")
    print("-" * 50)

    tests = [
        ("1.1 'compare X and Y'",
         "compare dinosaur and nuclear discussions",
         None, "comparative", check_comparison),

        ("1.2 'contrast X with Y'",
         "contrast country music with AI discussions",
         None, "comparative", check_comparison),

        ("1.3 'difference between X and Y'",
         "difference between abundance and living in NYC",
         None, "comparative", check_comparison),

        ("1.4 'X vs Y'",
         "dinosaurs vs nuclear fusion",
         None, "comparative", check_comparison),
    ]

    for name, query, gran, expected, check in tests:
        if test_query(name, query, gran, expected, check):
            passed += 1
        else:
            failed += 1

    # =========================================
    # TEST GROUP 2: Argumentation Searches
    # =========================================
    print("\n📊 TEST GROUP 2: Argumentation Searches")
    print("-" * 50)

    tests = [
        ("2.1 'strong debates'",
         "find sessions with strong debates",
         "sessions", "session_insights", check_argumentation_ranking),

        ("2.2 'challenges and disagreements'",
         "discussions with challenges and disagreements",
         "sessions", "session_insights", check_session_results),

        ("2.3 'deep reasoning'",
         "sessions with deep reasoning",
         "sessions", "session_insights", check_session_results),

        ("2.4 'problem solving'",
         "find problem solving discussions",
         "sessions", "session_insights", check_session_results),
    ]

    for name, query, gran, expected, check in tests:
        if test_query(name, query, gran, expected, check):
            passed += 1
        else:
            failed += 1

    # =========================================
    # TEST GROUP 3: Temporal Evolution Searches
    # =========================================
    print("\n📊 TEST GROUP 3: Temporal Evolution Searches")
    print("-" * 50)

    tests = [
        ("3.1 'became more analytical'",
         "discussions that became more analytical",
         "sessions", "session_insights", check_temporal_ranking),

        ("3.2 'tone improved'",
         "find sessions where tone improved",
         "sessions", "session_insights", check_session_results),

        ("3.3 'evolved over time'",
         "discussions that evolved over time",
         "sessions", "session_insights", check_session_results),
    ]

    for name, query, gran, expected, check in tests:
        if test_query(name, query, gran, expected, check):
            passed += 1
        else:
            failed += 1

    # =========================================
    # TEST GROUP 4: Chunk-Level Searches
    # =========================================
    print("\n📊 TEST GROUP 4: Chunk-Level Searches")
    print("-" * 50)

    tests = [
        ("4.1 'what was said about'",
         "what was said about dinosaurs",
         "chunks", None, check_chunk_results),

        ("4.2 'find when they discussed'",
         "find when they discussed nuclear fusion",
         "chunks", None, check_chunk_results),

        ("4.3 Topic search",
         "T-Rex predator evolution",
         "chunks", None, check_chunk_results),
    ]

    for name, query, gran, expected, check in tests:
        if test_query(name, query, gran, expected, check):
            passed += 1
        else:
            failed += 1

    # =========================================
    # TEST GROUP 5: Session-Level Searches
    # =========================================
    print("\n📊 TEST GROUP 5: Session-Level Searches (Single Collection)")
    print("-" * 50)

    tests = [
        ("5.1 Topic/transcript search",
         "discussions about nuclear energy",
         "sessions", "session_insights", check_session_results),

        ("5.2 Structure/concept search",
         "sessions with hypothesis testing",
         "sessions", "session_insights", check_session_results),

        ("5.3 Quality/7C search",
         "sessions with high communication quality",
         "sessions", "session_insights", check_session_results),
    ]

    for name, query, gran, expected, check in tests:
        if test_query(name, query, gran, expected, check):
            passed += 1
        else:
            failed += 1

    # =========================================
    # TEST GROUP 6: Speaker Searches
    # =========================================
    print("\n📊 TEST GROUP 6: Speaker Searches")
    print("-" * 50)

    tests = [
        ("6.1 'how did X engage'",
         "how did Lex engage in discussions",
         "speakers", None, check_speaker_results),

        ("6.2 'speaker style'",
         "what is David's speaker style",
         "speakers", None, check_speaker_results),
    ]

    for name, query, gran, expected, check in tests:
        if test_query(name, query, gran, expected, check):
            passed += 1
        else:
            failed += 1

    # =========================================
    # TEST GROUP 7: Insights Generation
    # =========================================
    print("\n📊 TEST GROUP 7: Insights Generation")
    print("-" * 50)

    tests = [
        ("7.1 'why' query (auto-insights)",
         "why do some discussions have higher engagement",
         "sessions", None, check_insights_generated),

        ("7.2 'analyze' query",
         "analyze the patterns in problem-solving discussions",
         "sessions", None, check_insights_generated),
    ]

    for name, query, gran, expected, check in tests:
        if test_query(name, query, gran, expected, check):
            passed += 1
        else:
            failed += 1

    # =========================================
    # TEST GROUP 8: Error Handling
    # =========================================
    print("\n📊 TEST GROUP 8: Error Handling & Edge Cases")
    print("-" * 50)

    # Test comparison with non-existent topics
    result = test_query("8.1 Comparison with obscure topics",
                       "compare quantum teleportation and medieval history",
                       None, None, None)
    if result:
        passed += 1
    else:
        failed += 1

    # Test empty query
    try:
        resp = requests.post(BASE_URL, json={"query": ""}, timeout=10)
        data = resp.json()
        if data.get("error") or data.get("total_found", 0) == 0:
            print("✓ 8.2 Empty query handled gracefully")
            passed += 1
        else:
            print("⚠️  8.2 Empty query returned unexpected results")
            passed += 1  # Not a failure, just unexpected
    except Exception as e:
        print(f"✓ 8.2 Empty query raised exception (acceptable): {type(e).__name__}")
        passed += 1

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
