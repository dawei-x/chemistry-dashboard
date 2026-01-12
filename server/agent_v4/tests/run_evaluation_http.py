#!/usr/bin/env python3
"""
Comprehensive Agent V4 Evaluation via HTTP API

Tests the V4 agent through the Flask API endpoint.
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, Any, List

BASE_URL = "http://localhost:5000/api/v4/agent"

# Test queries organized by category
TEST_QUERIES = {
    "factual": [
        {
            "id": "F1",
            "query": "What was the Nuclear Fusion session about?",
            "mode": "enhanced",
            "alignment_check": "Should provide technical details about fusion physics",
            "gold_elements": ["fusion", "energy", "David", "Lex", "temperature", "plasma"],
        },
        {
            "id": "F2",
            "query": "List all available sessions",
            "mode": "enhanced",
            "alignment_check": "Should list all 8-9 sessions with names",
            "gold_elements": ["Living in NYC", "AI Alive", "Nuclear Fusion", "Country Music", "Dinosaurs"],
        },
        {
            "id": "F3",
            "query": "Who were the speakers in session 19?",
            "mode": "enhanced",
            "alignment_check": "Must identify Sam and Tucker",
            "gold_elements": ["Sam", "Tucker"],
        },
    ],

    "analytical": [
        {
            "id": "A1",
            "query": "Did Tucker demonstrate systems thinking in session 19?",
            "mode": "enhanced",
            "alignment_check": "Should operationalize 'systems thinking' with evidence",
            "gold_elements": ["Tucker", "systems", "thinking", "emergent"],
        },
        {
            "id": "A2",
            "query": "Analyze the quality of argumentation in the AI Alive discussion",
            "mode": "enhanced",
            "alignment_check": "Should identify claims, evidence, rebuttals",
            "gold_elements": ["AI", "argument", "Sam", "Tucker", "hallucination"],
        },
    ],

    "speaker_attribution": [
        {
            "id": "SA1",
            "query": "How did Sam contribute to the AI discussion?",
            "mode": "enhanced",
            "alignment_check": "CRITICAL: Sam is TECHNICAL EXPLAINER discussing hallucination as 'mathematical probability'",
            "gold_elements": ["Sam", "technical", "hallucination", "mathematical", "probability"],
            "must_not_contain": ["Are they alive", "spark of life"],  # Tucker's words
        },
        {
            "id": "SA2",
            "query": "What questions did Tucker ask in session 19?",
            "mode": "enhanced",
            "alignment_check": "Tucker asked: 'Are they alive?', 'spark of life', hallucinating vs lying",
            "gold_elements": ["Tucker", "alive", "spark", "hallucinating", "lying"],
        },
        {
            "id": "SA3",
            "query": "What did David explain about fusion in session 20?",
            "mode": "enhanced",
            "alignment_check": "David explained E=MC², temperature requirements, strong nuclear force",
            "gold_elements": ["David", "fusion", "temperature", "energy", "nuclear"],
        },
    ],

    "comparison": [
        {
            "id": "C1",
            "query": "Which session has the best collaboration quality?",
            "mode": "enhanced",
            "alignment_check": "Must identify Session 24 (Country Music) with score ~80",
            "gold_elements": ["Country Music", "24", "80", "collaboration"],
        },
        {
            "id": "C2",
            "query": "Which session has the worst collaboration?",
            "mode": "enhanced",
            "alignment_check": "Should identify Session 22 (Collaboration Literacy) ~50",
            "gold_elements": ["22", "Collaboration Literacy", "50", "single speaker"],
        },
        {
            "id": "C3",
            "query": "Compare Nuclear Fusion and Country Music sessions",
            "mode": "enhanced",
            "alignment_check": "Should compare 7C dimensions with specific scores",
            "gold_elements": ["Nuclear Fusion", "Country Music", "79", "80", "climate", "constructive"],
        },
    ],

    "cross_session": [
        {
            "id": "CS1",
            "query": "Which sessions discuss technology's impact on society?",
            "mode": "enhanced",
            "alignment_check": "Should find AI Alive, Nuclear Fusion, Abundance",
            "gold_elements": ["AI", "Nuclear", "Abundance", "technology"],
        },
        {
            "id": "CS2",
            "query": "What sessions show evidence of hypothesis testing?",
            "mode": "enhanced",
            "alignment_check": "Session 19 has strongest evidence",
            "gold_elements": ["19", "hypothesis", "questioning"],
        },
    ],

    "complex": [
        {
            "id": "CX1",
            "query": "Analyze how David's teaching style in the Nuclear Fusion session affected collaboration quality",
            "mode": "enhanced",
            "alignment_check": "Should connect David's expert role with 7C scores (high constructive 88, lower contribution 65)",
            "gold_elements": ["David", "teaching", "collaboration", "constructive", "contribution"],
        },
        {
            "id": "CX2",
            "query": "What makes the Country Music session the most collaborative?",
            "mode": "enhanced",
            "alignment_check": "Should explain WHY: climate 85, conflict 90, compatibility 80",
            "gold_elements": ["Country Music", "climate", "conflict", "compatibility"],
        },
    ],

    "baseline_comparison": [
        {
            "id": "B1",
            "query": "How well did participants collaborate in session 20?",
            "mode": "baseline",
            "alignment_check": "Baseline cannot access 7C - should infer from transcript or state limitation",
            "gold_elements": ["transcript", "speaker", "dialogue"],
            "should_not_use": ["7c", "collaboration score"],
        },
    ],

    "edge_cases": [
        {
            "id": "E1",
            "query": "Tell me about session 99",
            "mode": "enhanced",
            "alignment_check": "Should handle gracefully - not crash or hallucinate",
            "gold_elements": ["not found", "doesn't exist", "available"],
        },
        {
            "id": "E2",
            "query": "What did Alice say about quantum computing?",
            "mode": "enhanced",
            "alignment_check": "Alice is in NYC session, quantum computing not discussed - handle mismatch",
            "gold_elements": ["Alice", "NYC", "not discussed"],
        },
    ],

    "multi_turn": [
        {
            "id": "MT1",
            "turns": [
                {"query": "Tell me about the Nuclear Fusion session", "turn": 1},
                {"query": "Who were the speakers?", "turn": 2},
                {"query": "What did the first speaker say about temperature?", "turn": 3},
            ],
            "alignment_check": "Context must persist across turns",
            "gold_elements": ["David", "Lex", "100 million", "temperature"],
        },
    ],
}


def query_agent(query: str, mode: str = "enhanced", conversation_id: str = None) -> Dict:
    """Send query to V4 agent API."""
    data = {
        "query": query,
        "mode": mode,
    }
    if conversation_id:
        data["conversation_id"] = conversation_id

    try:
        response = requests.post(f"{BASE_URL}/query", json=data, timeout=120)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e), "answer": ""}


def evaluate_response(response: Dict, test_info: Dict) -> Dict:
    """Evaluate a response against expectations."""
    answer = response.get("answer", "").lower()
    tools_used = response.get("tools_used", [])

    result = {
        "passed": [],
        "failed": [],
        "issues": [],
    }

    # Check gold elements
    gold_elements = test_info.get("gold_elements", [])
    found = 0
    missing = []
    for element in gold_elements:
        if element.lower() in answer:
            found += 1
        else:
            missing.append(element)

    if gold_elements:
        coverage = found / len(gold_elements)
        if coverage >= 0.5:
            result["passed"].append(f"Gold coverage: {found}/{len(gold_elements)} ({coverage:.0%})")
        else:
            result["failed"].append(f"Low gold coverage: {found}/{len(gold_elements)}, missing: {missing[:3]}")

    # Check must_not_contain (attribution errors)
    for forbidden in test_info.get("must_not_contain", []):
        if forbidden.lower() in answer:
            result["failed"].append(f"ATTRIBUTION ERROR: Contains '{forbidden}'")

    # Check should_not_use (baseline tools)
    for term in test_info.get("should_not_use", []):
        if term.lower() in answer:
            result["issues"].append(f"Baseline used restricted info: '{term}'")

    return result


def run_single_test(test_info: Dict, category: str) -> Dict:
    """Run a single test."""
    query_id = test_info["id"]
    query = test_info["query"]
    mode = test_info.get("mode", "enhanced")

    print(f"\n[{query_id}] {query[:60]}...")

    start = time.time()
    response = query_agent(query, mode)
    elapsed = time.time() - start

    evaluation = evaluate_response(response, test_info)

    print(f"  Time: {elapsed:.1f}s | Tools: {response.get('tools_used', [])[:3]}")
    print(f"  Answer: {response.get('answer', '')[:150]}...")

    if evaluation["failed"]:
        for f in evaluation["failed"]:
            print(f"  ✗ {f}")
    if evaluation["passed"]:
        print(f"  ✓ {len(evaluation['passed'])} checks passed")

    return {
        "query_id": query_id,
        "query": query,
        "category": category,
        "mode": mode,
        "time": round(elapsed, 2),
        "tools_used": response.get("tools_used", []),
        "answer_preview": response.get("answer", "")[:500],
        "success": response.get("success", False),
        "passed": evaluation["passed"],
        "failed": evaluation["failed"],
        "issues": evaluation["issues"],
        "alignment_check": test_info.get("alignment_check", ""),
    }


def run_multi_turn_test(test_info: Dict) -> List[Dict]:
    """Run multi-turn test."""
    results = []
    conv_id = f"mt-{test_info['id']}-{int(time.time())}"

    print(f"\n=== MULTI-TURN: {test_info['id']} ===")

    for turn_info in test_info["turns"]:
        query = turn_info["query"]
        turn = turn_info["turn"]

        print(f"\n  Turn {turn}: {query}")

        start = time.time()
        response = query_agent(query, "enhanced", conv_id)
        elapsed = time.time() - start

        answer = response.get("answer", "").lower()

        result = {
            "query_id": f"{test_info['id']}_T{turn}",
            "query": query,
            "category": "multi_turn",
            "turn": turn,
            "time": round(elapsed, 2),
            "answer_preview": response.get("answer", "")[:300],
            "passed": [],
            "failed": [],
        }

        # Check context preservation
        if turn == 2:
            if any(x in answer for x in ["david", "lex", "session 20", "nuclear", "fusion"]):
                result["passed"].append("Turn 2 maintained session context")
            else:
                result["failed"].append("Turn 2 LOST session context")
        elif turn == 3:
            if any(x in answer for x in ["david", "temperature", "100", "million", "degrees"]):
                result["passed"].append("Turn 3 maintained context")
            else:
                result["failed"].append("Turn 3 LOST context about David/temperature")

        print(f"    Time: {elapsed:.1f}s")
        print(f"    Answer: {response.get('answer', '')[:100]}...")
        if result["failed"]:
            print(f"    ✗ {result['failed']}")

        results.append(result)

    return results


def main():
    """Run all tests."""
    print("=" * 70)
    print("COMPREHENSIVE AGENT V4 EVALUATION (HTTP)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    all_results = []

    # Run single-query tests
    for category, tests in TEST_QUERIES.items():
        if category == "multi_turn":
            continue

        print(f"\n{'='*50}")
        print(f"CATEGORY: {category.upper()}")
        print("=" * 50)

        for test_info in tests:
            result = run_single_test(test_info, category)
            all_results.append(result)
            time.sleep(0.3)

    # Run multi-turn tests
    for test_info in TEST_QUERIES.get("multi_turn", []):
        results = run_multi_turn_test(test_info)
        all_results.extend(results)

    # Summary
    print("\n\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total = len(all_results)
    passed_all = sum(1 for r in all_results if len(r.get("failed", [])) == 0)
    total_time = sum(r.get("time", 0) for r in all_results)

    print(f"Total queries: {total}")
    print(f"Queries with no failures: {passed_all}/{total}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg time: {total_time/total:.1f}s")

    # By category
    print("\n--- By Category ---")
    categories = {}
    for r in all_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "times": []}
        categories[cat]["total"] += 1
        if not r.get("failed"):
            categories[cat]["passed"] += 1
        categories[cat]["times"].append(r.get("time", 0))

    for cat, stats in categories.items():
        avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        print(f"  {cat}: {stats['passed']}/{stats['total']} passed, avg {avg_time:.1f}s")

    # All failures
    failures = [r for r in all_results if r.get("failed")]
    if failures:
        print("\n--- ALL FAILURES ---")
        for r in failures:
            print(f"\n[{r['query_id']}] {r['query'][:50]}...")
            print(f"  Alignment: {r.get('alignment_check', 'N/A')[:80]}")
            for f in r["failed"]:
                print(f"  ✗ {f}")

    # Critical issues (attribution)
    critical = [r for r in all_results if any("ATTRIBUTION" in f for f in r.get("failed", []))]
    if critical:
        print("\n--- CRITICAL ATTRIBUTION ISSUES ---")
        for r in critical:
            print(f"  🚨 [{r['query_id']}] {r['query']}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total,
        "passed_all": passed_all,
        "avg_time": round(total_time / total, 2),
        "by_category": {cat: {"passed": s["passed"], "total": s["total"]} for cat, s in categories.items()},
        "failures": [{"query_id": r["query_id"], "query": r["query"], "failures": r["failed"]} for r in failures],
        "detailed_results": all_results,
    }

    report_path = "/home/ubuntu/chemistry-dashboard/server/agent_v4/tests/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
