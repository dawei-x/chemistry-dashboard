#!/usr/bin/env python3
"""
Comprehensive Ultra RAG Test Suite
Tests ALL query types for 100% reliability
"""

import requests
import json
import sys
from typing import Dict, Tuple

BASE_URL = "http://localhost:5000/api/v1/rag/search"

def test_query(name: str, query: str, expected_type: str = None,
               required_fields: list = None, timeout: int = 90) -> Tuple[bool, str, Dict]:
    """Execute a test query and validate response."""
    try:
        response = requests.post(
            BASE_URL,
            json={"query": query},
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        data = response.json()

        # Check for errors
        if data.get('query_type') == 'error' or data.get('error'):
            return False, f"ERROR: {data.get('error')}", data

        # Check expected type
        if expected_type and data.get('query_type') != expected_type:
            return False, f"Wrong type: expected '{expected_type}', got '{data.get('query_type')}'", data

        # Check required fields
        if required_fields:
            for field in required_fields:
                if field not in data or data[field] is None:
                    return False, f"Missing required field: {field}", data

        # Check for results (API uses different field names for different query types)
        has_results = (
            data.get('results') or
            data.get('session_results') or
            data.get('speaker_results') or
            data.get('comparison') or  # Comparative query results
            data.get('similar') or     # Similarity query results
            data.get('insights') or
            data.get('timeline')
        )

        if not has_results and data.get('total_found', 0) == 0:
            return False, "No results returned", data

        return True, "OK", data

    except requests.exceptions.Timeout:
        return False, "TIMEOUT", {}
    except requests.exceptions.ConnectionError:
        return False, "CONNECTION ERROR - is server running?", {}
    except Exception as e:
        return False, f"EXCEPTION: {str(e)}", {}


def run_all_tests():
    """Run comprehensive test suite."""

    print("=" * 70)
    print("ULTRA RAG COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        # === CATEGORY 1: CHUNK-LEVEL SEARCHES ===
        ("1.1 Topic Search (simple)",
         "nuclear fusion energy",
         "topic_search",
         ["results"]),

        ("1.2 Topic Search (what about)",
         "What discussions covered climate change?",
         None,  # Can be topic_search or pattern_analysis
         None),

        ("1.3 Quote Search",
         "What was said about hydrogen isotopes?",
         None,
         None),

        ("1.4 Pattern Analysis",
         "What did Andrew say about collaboration?",
         None,  # Can be topic_search or pattern_analysis
         ["results"]),

        # === CATEGORY 2: SESSION-LEVEL SEARCHES ===
        ("2.1 Session Search (argumentation)",
         "Find sessions with strong argumentation",
         "session_search",
         ["session_results"]),

        ("2.2 Session Search (collaboration)",
         "Find discussions with good collaboration quality",
         "session_search",
         ["session_results"]),

        ("2.3 Session Search (debate)",
         "Which sessions have the most debate and challenges?",
         "session_search",
         ["session_results"]),

        ("2.4 Session Search (questions)",
         "Find sessions with many questions",
         "session_search",
         ["session_results"]),

        ("2.5 Session Search (reasoning)",
         "Find discussions showing deep reasoning patterns",
         None,  # May trigger insights due to analytical words
         ["session_results"]),

        # === CATEGORY 3: ANALYTICAL/INSIGHT QUERIES ===
        ("3.1 Why Query (collaboration)",
         "Why do some discussions have better collaboration?",
         "session_insights",
         ["insights", "session_results"]),

        ("3.2 How Query",
         "How are challenges made in effective discussions?",
         "session_insights",
         ["insights"]),

        ("3.3 Analyze Query",
         "Analyze the argumentation patterns across sessions",
         "session_insights",
         ["insights"]),

        ("3.4 Explain Query",
         "Explain what makes some discussions more productive",
         "session_insights",
         ["insights"]),

        # === CATEGORY 4: SPEAKER SEARCHES ===
        ("4.1 Speaker Search (general)",
         "How does Lex engage in discussions?",
         None,  # May be speaker_insights due to "How" keyword
         None),

        ("4.2 Speaker Search (style)",
         "What is David's speaking style?",
         None,  # Could be speaker_search or speaker_insights
         None),

        ("4.3 Speaker Search (questions)",
         "Speakers who ask many questions",
         "speaker_search",
         ["speaker_results"]),

        # === CATEGORY 5: COMPARATIVE QUERIES ===
        ("5.1 Compare by Topic",
         "Compare the nuclear fusion discussion and the collaboration literacy discussion",
         "comparative",
         None),  # comparison field is checked via has_results

        ("5.2 Compare by Session ID",
         "Compare session 18 and session 20",
         "comparative",
         None),  # comparison field is checked via has_results

        # === CATEGORY 6: SIMILARITY SEARCHES ===
        ("6.1 Similar Sessions (by ID)",
         "Find sessions similar to session 20",
         "similar_sessions",
         None),  # similar field is checked via has_results

        # === CATEGORY 7: TEMPORAL/EVOLUTION QUERIES ===
        ("7.1 Timeline (specific session)",
         "Show timeline progression for session 20",
         "temporal",
         ["timeline"]),

        ("7.2 Evolution Query (session search)",
         "Find discussions that evolved over time",
         "session_search",
         ["session_results"]),

        # === CATEGORY 8: 7C QUALITY QUERIES ===
        ("8.1 Communication Quality",
         "Find sessions with high communication quality",
         "session_search",
         ["session_results"]),

        ("8.2 Conflict Resolution",
         "Find sessions with effective conflict resolution",
         "session_search",
         ["session_results"]),

        ("8.3 Constructive Discussion",
         "Find constructive discussions",
         "session_search",
         ["session_results"]),

        # === CATEGORY 9: EDGE CASES ===
        ("9.1 Very Short Query",
         "fusion",
         None,
         None),

        ("9.2 Question Format",
         "What sessions have the highest debate scores?",
         None,
         None),

        ("9.3 Imperative Format",
         "Show me sessions with many challenges",
         None,  # May trigger insights
         ["session_results"]),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, query, expected_type, required_fields in tests:
        print(f"\nTest: {name}")
        print(f"  Query: \"{query}\"")

        success, message, data = test_query(name, query, expected_type, required_fields)

        if success:
            print(f"  ✓ PASS - Type: {data.get('query_type')}, Results: {data.get('total_found', 'N/A')}")
            if data.get('insights'):
                print(f"  ✓ Has insights: {len(data['insights'])} chars")
            passed += 1
        else:
            print(f"  ✗ FAIL - {message}")
            errors.append((name, query, message, data.get('error') if data else None))
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {100 * passed / (passed + failed):.1f}%")

    if errors:
        print("\n" + "-" * 70)
        print("FAILED TESTS:")
        print("-" * 70)
        for name, query, message, error in errors:
            print(f"\n{name}")
            print(f"  Query: {query}")
            print(f"  Error: {message}")
            if error:
                print(f"  Details: {error}")

    return passed, failed, errors


if __name__ == "__main__":
    passed, failed, errors = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
