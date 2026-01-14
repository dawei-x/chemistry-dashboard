"""
Comprehensive Test Suite for Agent V5

This test suite evaluates Agent V5 holistically:
1. Query Understanding accuracy
2. Context Assembly correctness
3. Response quality and grounding
4. Triangulation behavior
5. Tool usage patterns
6. Comparison with V4

Run with: python -m pytest agent_v5/tests/test_comprehensive.py -v
Or standalone: python agent_v5/tests/test_comprehensive.py
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_v5.query_understanding import understand_query, QueryIntent
from agent_v5.context_assembly import assemble_context
from agent_v5.agent import run_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# TEST CASES
# =============================================================================

QUERY_UNDERSTANDING_TESTS = [
    # (query, expected_intent_type, expected_retrieval_mode, expected_sessions, expected_speakers)

    # Describe queries - structured retrieval
    ("Tell me about the Nuclear Fusion session", "describe", "structured", [20], []),
    ("What happened in session 19?", "describe", "structured", [19], []),
    ("Summarize the AI Alive discussion", "describe", "structured", [19], []),

    # Speaker queries - structured retrieval
    ("What did Sam say in the AI Alive session?", "speaker", "structured", [19], ["Sam"]),
    ("What questions did Lex ask?", "speaker", "semantic", [], ["Lex"]),
    ("How did David contribute to the Nuclear Fusion discussion?", "speaker", "structured", [20], ["David"]),

    # Compare queries - structured retrieval
    ("Compare sessions 18 and 19", "compare", "structured", [], []),
    ("What's the difference between Nuclear Fusion and AI Alive?", "compare", "structured", [20, 19], []),

    # Explain/Why queries - contrastive retrieval
    ("Why do some discussions have better collaboration?", "explain", "contrastive", [], []),
    ("What makes some sessions more collaborative than others?", "explain", "contrastive", [], []),

    # Search queries - semantic retrieval
    ("Find discussions about climate change", "search", "semantic", [], []),
    ("Which sessions discuss artificial intelligence?", "search", "semantic", [], []),

    # Explore queries - agentic only
    ("What patterns do you see across all discussions?", "explore", "semantic", [], []),
    ("Tell me something interesting", "explore", "agentic_only", [], []),
]

CONTEXT_ASSEMBLY_TESTS = [
    # (query, expected_mode, should_have_sessions)
    ("Tell me about Nuclear Fusion", "structured", True),
    ("Why do some discussions have better collaboration?", "contrastive", True),
    ("What patterns exist across sessions?", "semantic", False),  # May or may not find
]

AGENT_RESPONSE_TESTS = [
    # (query, expectations)
    {
        "query": "Tell me about the Nuclear Fusion session",
        "expectations": {
            "should_mention": ["nuclear", "fusion"],
            "should_have_tools_used": False,  # Should use pre-loaded context
            "context_mode": "structured",
            "min_response_length": 100,
        }
    },
    {
        "query": "What did Sam say in the Is AI Alive discussion?",
        "expectations": {
            "should_mention": ["sam", "ai"],
            "context_mode": "structured",
            "should_have_speaker_quotes": True,
        }
    },
    {
        "query": "Why do some discussions have better collaboration than others?",
        "expectations": {
            "should_mention": ["collaboration", "score", "communication"],
            "context_mode": "contrastive",
            "should_compare_sessions": True,
        }
    },
    {
        "query": "Compare the collaboration quality of sessions 18 and 19",
        "expectations": {
            "should_mention": ["session", "collaboration"],
            "context_mode": "structured",
        }
    },
]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_query_understanding() -> Dict[str, Any]:
    """Test query understanding accuracy."""
    results = {
        "total": len(QUERY_UNDERSTANDING_TESTS),
        "passed": 0,
        "failed": 0,
        "details": []
    }

    for query, exp_intent, exp_mode, exp_sessions, exp_speakers in QUERY_UNDERSTANDING_TESTS:
        intent = understand_query(query)

        # Check each expectation
        issues = []

        if intent.intent_type != exp_intent:
            issues.append(f"Intent: expected '{exp_intent}', got '{intent.intent_type}'")

        if intent.retrieval_mode != exp_mode:
            issues.append(f"Retrieval mode: expected '{exp_mode}', got '{intent.retrieval_mode}'")

        if exp_sessions and set(intent.session_ids) != set(exp_sessions):
            issues.append(f"Sessions: expected {exp_sessions}, got {intent.session_ids}")

        if exp_speakers and set(s.lower() for s in intent.speaker_names) != set(s.lower() for s in exp_speakers):
            issues.append(f"Speakers: expected {exp_speakers}, got {intent.speaker_names}")

        passed = len(issues) == 0
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        results["details"].append({
            "query": query[:60] + "..." if len(query) > 60 else query,
            "passed": passed,
            "issues": issues,
            "actual": {
                "intent_type": intent.intent_type,
                "retrieval_mode": intent.retrieval_mode,
                "session_ids": intent.session_ids,
                "speaker_names": intent.speaker_names
            }
        })

    return results


def test_context_assembly() -> Dict[str, Any]:
    """Test context assembly for different query types."""
    results = {
        "total": len(CONTEXT_ASSEMBLY_TESTS),
        "passed": 0,
        "failed": 0,
        "details": []
    }

    for query, exp_mode, should_have_sessions in CONTEXT_ASSEMBLY_TESTS:
        intent = understand_query(query)
        context = assemble_context(intent, query, rag_service=None)  # Test without RAG

        issues = []

        actual_mode = context.get("retrieval_metadata", {}).get("mode", "unknown")

        # For contrastive without RAG, it falls back to manual
        if exp_mode == "contrastive" and actual_mode == "contrastive_manual":
            actual_mode = "contrastive"  # Accept manual fallback

        if actual_mode != exp_mode and actual_mode != "fallback":
            issues.append(f"Context mode: expected '{exp_mode}', got '{actual_mode}'")

        if should_have_sessions and not context.get("sessions_loaded"):
            if actual_mode != "fallback":  # Fallback is acceptable
                issues.append("Expected sessions to be loaded but none found")

        # Check context has content
        if not context.get("context_text") and actual_mode != "fallback":
            issues.append("Context text is empty")

        passed = len(issues) == 0
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        results["details"].append({
            "query": query[:60] + "..." if len(query) > 60 else query,
            "passed": passed,
            "issues": issues,
            "actual": {
                "mode": actual_mode,
                "sessions_loaded": context.get("sessions_loaded", []),
                "context_length": len(context.get("context_text", ""))
            }
        })

    return results


def test_agent_responses() -> Dict[str, Any]:
    """Test full agent response quality."""
    results = {
        "total": len(AGENT_RESPONSE_TESTS),
        "passed": 0,
        "failed": 0,
        "partial": 0,
        "details": []
    }

    for test_case in AGENT_RESPONSE_TESTS:
        query = test_case["query"]
        expectations = test_case["expectations"]

        logger.info(f"Testing: {query[:50]}...")

        try:
            response = run_agent(query, mode="enhanced")

            issues = []
            notes = []

            answer = response.get("answer", "").lower()

            # Check mentions
            if "should_mention" in expectations:
                for term in expectations["should_mention"]:
                    if term.lower() not in answer:
                        issues.append(f"Missing expected term: '{term}'")

            # Check response length
            if "min_response_length" in expectations:
                if len(answer) < expectations["min_response_length"]:
                    issues.append(f"Response too short: {len(answer)} chars")

            # Check context mode
            if "context_mode" in expectations:
                actual_mode = response.get("context_preloaded", {}).get("mode", "unknown")
                exp_mode = expectations["context_mode"]
                if actual_mode != exp_mode:
                    # Accept variants
                    if not (exp_mode == "contrastive" and actual_mode == "contrastive_manual"):
                        issues.append(f"Context mode: expected '{exp_mode}', got '{actual_mode}'")

            # Check tool usage
            tools_used = response.get("tools_used", [])
            if "should_have_tools_used" in expectations:
                if expectations["should_have_tools_used"] and not tools_used:
                    notes.append("Expected tool use but none occurred (may be OK if context was sufficient)")
                elif not expectations["should_have_tools_used"] and tools_used:
                    notes.append(f"Used tools when context should have been sufficient: {tools_used}")

            # Qualitative checks
            if "should_have_speaker_quotes" in expectations:
                if '"' not in answer and "'" not in answer and "said" not in answer:
                    issues.append("Expected speaker quotes but none found")

            if "should_compare_sessions" in expectations:
                comparison_words = ["higher", "lower", "better", "worse", "compared", "versus", "while", "whereas"]
                if not any(w in answer for w in comparison_words):
                    issues.append("Expected comparison language but none found")

            # Determine pass/fail
            if len(issues) == 0:
                results["passed"] += 1
                status = "passed"
            elif len(issues) <= 1 and len(notes) > 0:
                results["partial"] += 1
                status = "partial"
            else:
                results["failed"] += 1
                status = "failed"

            results["details"].append({
                "query": query[:60] + "..." if len(query) > 60 else query,
                "status": status,
                "issues": issues,
                "notes": notes,
                "response_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                "tools_used": tools_used,
                "context_mode": response.get("context_preloaded", {}).get("mode"),
                "turn_count": response.get("turn_count", 0)
            })

        except Exception as e:
            results["failed"] += 1
            results["details"].append({
                "query": query[:60] + "..." if len(query) > 60 else query,
                "status": "error",
                "error": str(e)
            })

    return results


def test_triangulation_behavior() -> Dict[str, Any]:
    """Test whether the agent naturally triangulates across sources."""

    test_queries = [
        "How well did the group collaborate in the Nuclear Fusion session?",
        "Analyze the discussion dynamics in session 19",
        "What made the Is AI Alive discussion effective or ineffective?",
    ]

    results = {
        "total": len(test_queries),
        "details": []
    }

    triangulation_indicators = [
        # Positive indicators (natural triangulation)
        ("mentions_score", ["score", "7c", "rating", "%", "/100"]),
        ("mentions_transcript", ["said", "stated", "mentioned", "asked", "explained"]),
        ("quotes_speakers", ['"', "'", "according to"]),
        ("connects_evidence", ["this is reflected", "consistent with", "suggests", "indicates", "shows"]),

        # Negative indicators (robotic enumeration)
        ("robotic_structure", ["the 7c shows", "the concept map shows", "the transcript shows", "firstly", "secondly"]),
    ]

    for query in test_queries:
        logger.info(f"Testing triangulation: {query[:50]}...")

        try:
            response = run_agent(query, mode="enhanced")
            answer = response.get("answer", "").lower()

            indicators_found = {}
            for indicator_name, patterns in triangulation_indicators:
                found = any(p.lower() in answer for p in patterns)
                indicators_found[indicator_name] = found

            # Evaluate triangulation quality
            has_quantitative = indicators_found.get("mentions_score", False)
            has_qualitative = indicators_found.get("mentions_transcript", False) or indicators_found.get("quotes_speakers", False)
            has_synthesis = indicators_found.get("connects_evidence", False)
            is_robotic = indicators_found.get("robotic_structure", False)

            if has_quantitative and has_qualitative and has_synthesis and not is_robotic:
                quality = "excellent"
            elif has_quantitative and has_qualitative and not is_robotic:
                quality = "good"
            elif (has_quantitative or has_qualitative) and not is_robotic:
                quality = "partial"
            elif is_robotic:
                quality = "robotic"
            else:
                quality = "minimal"

            results["details"].append({
                "query": query[:60] + "...",
                "triangulation_quality": quality,
                "indicators": indicators_found,
                "response_preview": answer[:300] + "..." if len(answer) > 300 else answer,
            })

        except Exception as e:
            results["details"].append({
                "query": query[:60] + "...",
                "error": str(e)
            })

    return results


def test_v4_v5_comparison() -> Dict[str, Any]:
    """Compare V4 and V5 on the same queries."""
    from agent_v4.agent import run_agent as run_agent_v4

    comparison_queries = [
        "Tell me about the Nuclear Fusion session",
        "What did Sam contribute to the Is AI Alive discussion?",
        "Why do some discussions have better collaboration?",
    ]

    results = {
        "queries": []
    }

    for query in comparison_queries:
        logger.info(f"Comparing V4 vs V5: {query[:40]}...")

        try:
            # Run V4
            v4_response = run_agent_v4(query, mode="enhanced")

            # Run V5
            v5_response = run_agent(query, mode="enhanced")

            results["queries"].append({
                "query": query,
                "v4": {
                    "turn_count": v4_response.get("turn_count", 0),
                    "tools_used": v4_response.get("tools_used", []),
                    "response_length": len(v4_response.get("answer", "")),
                    "preview": v4_response.get("answer", "")[:200] + "..."
                },
                "v5": {
                    "turn_count": v5_response.get("turn_count", 0),
                    "tools_used": v5_response.get("tools_used", []),
                    "response_length": len(v5_response.get("answer", "")),
                    "context_mode": v5_response.get("context_preloaded", {}).get("mode"),
                    "sessions_preloaded": v5_response.get("context_preloaded", {}).get("sessions_loaded", []),
                    "preview": v5_response.get("answer", "")[:200] + "..."
                },
                "comparison": {
                    "v5_fewer_turns": v5_response.get("turn_count", 0) < v4_response.get("turn_count", 0),
                    "v5_fewer_tools": len(v5_response.get("tools_used", [])) < len(v4_response.get("tools_used", [])),
                    "v5_used_preloading": v5_response.get("context_preloaded", {}).get("mode") != "none",
                }
            })

        except Exception as e:
            results["queries"].append({
                "query": query,
                "error": str(e)
            })

    return results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests(save_report: bool = True) -> Dict[str, Any]:
    """Run all tests and generate report."""

    report = {
        "timestamp": datetime.now().isoformat(),
        "agent_version": "v5",
        "tests": {}
    }

    print("\n" + "="*70)
    print("AGENT V5 COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")

    # Test 1: Query Understanding
    print("1. Testing Query Understanding...")
    report["tests"]["query_understanding"] = test_query_understanding()
    qu_results = report["tests"]["query_understanding"]
    print(f"   Passed: {qu_results['passed']}/{qu_results['total']}")
    if qu_results['failed'] > 0:
        print(f"   Failed cases:")
        for detail in qu_results['details']:
            if not detail['passed']:
                print(f"     - {detail['query']}")
                for issue in detail['issues']:
                    print(f"       Issue: {issue}")

    # Test 2: Context Assembly
    print("\n2. Testing Context Assembly...")
    report["tests"]["context_assembly"] = test_context_assembly()
    ca_results = report["tests"]["context_assembly"]
    print(f"   Passed: {ca_results['passed']}/{ca_results['total']}")
    if ca_results['failed'] > 0:
        print(f"   Failed cases:")
        for detail in ca_results['details']:
            if not detail['passed']:
                print(f"     - {detail['query']}")
                for issue in detail['issues']:
                    print(f"       Issue: {issue}")

    # Test 3: Agent Responses
    print("\n3. Testing Agent Responses (this may take a minute)...")
    report["tests"]["agent_responses"] = test_agent_responses()
    ar_results = report["tests"]["agent_responses"]
    print(f"   Passed: {ar_results['passed']}, Partial: {ar_results.get('partial', 0)}, Failed: {ar_results['failed']}")
    for detail in ar_results['details']:
        status_icon = "✓" if detail['status'] == 'passed' else ("~" if detail['status'] == 'partial' else "✗")
        print(f"   {status_icon} {detail['query']}")
        if detail.get('issues'):
            for issue in detail['issues']:
                print(f"      Issue: {issue}")
        if detail.get('notes'):
            for note in detail['notes']:
                print(f"      Note: {note}")

    # Test 4: Triangulation Behavior
    print("\n4. Testing Triangulation Behavior...")
    report["tests"]["triangulation"] = test_triangulation_behavior()
    for detail in report["tests"]["triangulation"]["details"]:
        quality = detail.get("triangulation_quality", "unknown")
        quality_icon = {"excellent": "★★★", "good": "★★☆", "partial": "★☆☆", "robotic": "⚠", "minimal": "○"}.get(quality, "?")
        print(f"   {quality_icon} {detail['query']} -> {quality}")

    # Test 5: V4 vs V5 Comparison
    print("\n5. Comparing V4 vs V5...")
    try:
        report["tests"]["v4_v5_comparison"] = test_v4_v5_comparison()
        for item in report["tests"]["v4_v5_comparison"]["queries"]:
            if "error" in item:
                print(f"   ✗ {item['query'][:40]}... - Error: {item['error']}")
            else:
                v4_turns = item["v4"]["turn_count"]
                v5_turns = item["v5"]["turn_count"]
                v5_preload = item["v5"]["context_mode"]
                efficiency = "more efficient" if v5_turns <= v4_turns else "less efficient"
                print(f"   - {item['query'][:40]}...")
                print(f"     V4: {v4_turns} turns, tools: {item['v4']['tools_used']}")
                print(f"     V5: {v5_turns} turns, preload: {v5_preload}, tools: {item['v5']['tools_used']}")
                print(f"     -> V5 is {efficiency}")
    except Exception as e:
        print(f"   Error running comparison: {e}")
        report["tests"]["v4_v5_comparison"] = {"error": str(e)}

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_passed = qu_results['passed'] + ca_results['passed'] + ar_results['passed']
    total_tests = qu_results['total'] + ca_results['total'] + ar_results['total']

    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Pass rate: {100 * total_passed / total_tests:.1f}%")

    # Save report
    if save_report:
        report_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nDetailed report saved to: {report_path}")

    return report


if __name__ == "__main__":
    run_all_tests()
