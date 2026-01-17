#!/usr/bin/env python3
"""
Comprehensive V7 Evaluation using V3's challenging test cases.

This evaluates the current V7 agent against challenging queries from V3's test suite,
focusing on detailed output analysis beyond simple metrics.
"""

import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_v7 import run_agent, get_memory, clear_memory

# Challenging test queries from V3's comprehensive evaluation
TEST_QUERIES = {
    # === Analytical (Complex Reasoning) ===
    "analytical": [
        {
            "id": "A1",
            "query": "Did Tucker demonstrate systems thinking in session 19?",
            "expectations": {
                "should_mention": ["Tucker"],
                "should_use_multiple_artifacts": True,
                "requires_operationalization": True,  # "systems thinking" is abstract
            },
            "challenge": "Must operationalize 'systems thinking' into observable indicators"
        },
        {
            "id": "A2",
            "query": "How well did participants collaborate in session 20?",
            "expectations": {
                "should_use_7c": True,
                "should_mention": ["collaboration"],
            },
            "challenge": "Should use 7C analysis with specific dimension scores"
        },
        {
            "id": "A3",
            "query": "What evidence shows critical thinking in the Dinosaurs session?",
            "expectations": {
                "should_resolve_session_name": True,  # Dinosaurs -> session 23
                "requires_operationalization": True,  # "critical thinking" is abstract
            },
            "challenge": "Must resolve session name AND operationalize 'critical thinking'"
        },
    ],

    # === Comparison (Cross-Session) ===
    "comparison": [
        {
            "id": "C1",
            "query": "Which session has the best collaboration quality?",
            "expectations": {
                "should_compare_all_sessions": True,
                "correct_answer": "Session 24",  # Country Music has highest 7C
            },
            "challenge": "Must compare ALL sessions, not just a subset"
        },
        {
            "id": "C2",
            "query": "Compare the AI Alive and Nuclear Fusion sessions",
            "expectations": {
                "should_resolve_session_names": True,
                "sessions": [19, 20],
            },
            "challenge": "Must resolve session names and compare systematically"
        },
        {
            "id": "C3",
            "query": "Which sessions discussed technology and its societal impact?",
            "expectations": {
                "should_search_all_sessions": True,
                "likely_sessions": [19, 20, 25],
            },
            "challenge": "Cross-session thematic search - should find multiple relevant sessions"
        },
    ],

    # === Graph/Path Queries ===
    "graph": [
        {
            "id": "G1",
            "query": "How are ideas about fusion connected to energy in session 20?",
            "expectations": {
                "should_use_concept_map": True,
                "should_trace_connections": True,
            },
            "challenge": "Should trace concept connections in the concept map"
        },
        {
            "id": "G2",
            "query": "What's the connection between AI consciousness and ethics in session 19?",
            "expectations": {
                "should_use_concept_map": True,
            },
            "challenge": "Should trace concept paths in the concept map"
        },
    ],

    # === Speaker-Focused ===
    "speaker": [
        {
            "id": "S1",
            "query": "What did David say about fusion in session 20?",
            "expectations": {
                "should_filter_by_speaker": True,
                "speaker": "David",
                "session": 20,
            },
            "challenge": "Should filter transcript by speaker"
        },
        {
            "id": "S2",
            "query": "Compare Tucker and David's contributions in session 19",
            "expectations": {
                "should_compare_speakers": True,
                "speakers": ["Tucker", "David"],
            },
            "challenge": "Should analyze both speakers' contributions"
        },
    ],

    # === Exploratory (Cross-Session Discovery) ===
    "exploratory": [
        {
            "id": "E1",
            "query": "What sessions show hypothesis testing?",
            "expectations": {
                "should_search_all_sessions": True,
            },
            "challenge": "Exploratory query - should NOT be constrained to single session"
        },
        {
            "id": "E2",
            "query": "Find all discussions about energy across sessions",
            "expectations": {
                "should_search_all_sessions": True,
            },
            "challenge": "Cross-session thematic search"
        },
    ],

    # === Edge Cases ===
    "edge_cases": [
        {
            "id": "EC1",
            "query": "What's the worst collaboration?",
            "expectations": {
                "should_compare_all_sessions": True,
            },
            "challenge": "Superlative query (worst) - needs global comparison"
        },
        {
            "id": "EC2",
            "query": "Session 99 overview",
            "expectations": {
                "should_handle_gracefully": True,
            },
            "challenge": "Invalid session ID - should handle error gracefully"
        },
    ],

    # === Multi-turn Context ===
    "multi_turn": [
        {
            "id": "M1",
            "turns": [
                {"query": "Tell me about the Nuclear Fusion session", "turn": 1},
                {"query": "Who were the speakers?", "turn": 2},
                {"query": "What did David specifically say about temperature?", "turn": 3},
            ],
            "challenge": "Context preservation across turns"
        },
    ],
}


def run_single_query(query: str, conversation_id: str) -> Dict[str, Any]:
    """Run a single query and return the response."""
    try:
        response = run_agent(conversation_id, query)
        return {
            "success": True,
            "answer": response.answer,
            "tools_used": response.tools_used,
            "confidence": response.confidence,
            "citations": response.citations,
            "follow_ups": response.follow_ups,
            "raw": response.__dict__ if hasattr(response, '__dict__') else str(response),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": "",
            "tools_used": [],
        }


def evaluate_query(query_info: Dict, response: Dict) -> Dict[str, Any]:
    """Evaluate a response against expectations."""
    result = {
        "query_id": query_info["id"],
        "query": query_info.get("query", query_info.get("turns", [{}])[0].get("query", "")),
        "challenge": query_info.get("challenge", ""),
        "response": response,
        "checks": [],
        "issues": [],
    }

    answer = response.get("answer", "").lower()
    tools = response.get("tools_used", [])
    expectations = query_info.get("expectations", {})

    # Check mentions
    if expectations.get("should_mention"):
        for term in expectations["should_mention"]:
            if term.lower() in answer:
                result["checks"].append({"check": f"mentions '{term}'", "passed": True})
            else:
                result["checks"].append({"check": f"mentions '{term}'", "passed": False})
                result["issues"].append(f"Missing expected term: '{term}'")

    # Check 7C usage
    if expectations.get("should_use_7c"):
        if "get_7c_analysis" in tools:
            result["checks"].append({"check": "uses 7C analysis", "passed": True})
        else:
            result["checks"].append({"check": "uses 7C analysis", "passed": False})
            result["issues"].append("Should use 7C analysis but didn't")

    # Check concept map usage
    if expectations.get("should_use_concept_map"):
        if "get_concept_map" in tools:
            result["checks"].append({"check": "uses concept map", "passed": True})
        else:
            result["checks"].append({"check": "uses concept map", "passed": False})
            result["issues"].append("Should use concept map but didn't")

    # Check speaker filter
    if expectations.get("should_filter_by_speaker"):
        speaker = expectations.get("speaker", "").lower()
        if speaker in answer:
            result["checks"].append({"check": f"mentions speaker '{speaker}'", "passed": True})
        else:
            result["checks"].append({"check": f"mentions speaker '{speaker}'", "passed": False})

    # Check correct answer
    if expectations.get("correct_answer"):
        correct = expectations["correct_answer"].lower()
        if correct in answer:
            result["checks"].append({"check": f"correct answer '{correct}'", "passed": True})
        else:
            result["checks"].append({"check": f"correct answer '{correct}'", "passed": False})
            result["issues"].append(f"Expected '{correct}' but not found in answer")

    # Check graceful error handling
    if expectations.get("should_handle_gracefully"):
        if response.get("success") or "not found" in answer or "doesn't exist" in answer or "no session" in answer:
            result["checks"].append({"check": "handles gracefully", "passed": True})
        else:
            result["checks"].append({"check": "handles gracefully", "passed": False})

    return result


def run_multi_turn(test_info: Dict) -> List[Dict]:
    """Run a multi-turn conversation test."""
    results = []
    conv_id = f"multi-{test_info['id']}-{int(time.time())}"

    for turn_info in test_info["turns"]:
        query = turn_info["query"]
        turn = turn_info["turn"]

        print(f"  Turn {turn}: {query}")
        response = run_single_query(query, conv_id)

        result = {
            "query_id": f"{test_info['id']}_T{turn}",
            "query": query,
            "turn": turn,
            "response": response,
            "checks": [],
            "issues": [],
        }

        # Check context preservation for turns > 1
        if turn > 1:
            # The answer should relate to previous context
            if response.get("success") and len(response.get("answer", "")) > 50:
                result["checks"].append({"check": "context preserved", "passed": True})
            else:
                result["issues"].append("May have lost context from previous turns")

        results.append(result)
        time.sleep(0.5)

    clear_memory(conv_id)
    return results


def run_evaluation():
    """Run full evaluation."""
    print("=" * 70)
    print("COMPREHENSIVE V7 EVALUATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    all_results = []

    for category, queries in TEST_QUERIES.items():
        print(f"\n{'='*50}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*50}")

        if category == "multi_turn":
            for test_info in queries:
                print(f"\n[{test_info['id']}] Multi-turn test")
                results = run_multi_turn(test_info)
                all_results.extend(results)
        else:
            for query_info in queries:
                query = query_info["query"]
                print(f"\n[{query_info['id']}] {query}")
                print(f"  Challenge: {query_info.get('challenge', 'N/A')}")

                conv_id = f"eval-{query_info['id']}-{int(time.time())}"
                response = run_single_query(query, conv_id)

                print(f"  Tools: {response.get('tools_used', [])}")
                print(f"  Answer preview: {response.get('answer', '')[:200]}...")

                result = evaluate_query(query_info, response)
                all_results.append(result)

                clear_memory(conv_id)
                time.sleep(0.5)

    # Generate summary
    total = len(all_results)
    with_issues = sum(1 for r in all_results if r.get("issues"))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total,
        "queries_with_issues": with_issues,
        "queries_clean": total - with_issues,
        "results": all_results,
    }

    # Save report
    report_path = "/home/ubuntu/chemistry-dashboard/server/agent_v7/tests/comprehensive_v7_eval_report.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total queries: {total}")
    print(f"Queries with issues: {with_issues}")
    print(f"Queries clean: {total - with_issues}")

    print("\n--- Issues Found ---")
    for result in all_results:
        if result.get("issues"):
            print(f"\n[{result['query_id']}] {result['query']}")
            print(f"  Challenge: {result.get('challenge', 'N/A')}")
            for issue in result["issues"]:
                print(f"  - {issue}")

    print(f"\nReport saved to: {report_path}")
    return summary


if __name__ == "__main__":
    run_evaluation()
