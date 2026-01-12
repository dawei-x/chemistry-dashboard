#!/usr/bin/env python3
"""
Comprehensive Agent V4 Evaluation Suite

Tests the high-agency V4 agent with various query types and evaluates outputs
against expectations. Focuses on alignment with reasoning quality, not just metrics.

Run with: ~/.pyenv/versions/blinc/bin/python -m agent_v4.tests.comprehensive_evaluation
"""

import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add server directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_v4.agent import run_agent

# Test queries organized by category with complexity and alignment expectations
TEST_QUERIES = {
    # === CATEGORY 1: Factual Retrieval ===
    "factual": [
        {
            "id": "F1",
            "query": "What was the Nuclear Fusion session about?",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["fusion", "energy", "David", "Lex"],
                "should_use_tools": ["get_transcript"],
                "min_answer_length": 150,
            },
            "alignment_check": "Should provide technical details about fusion physics (E=MC², 100M degrees, strong nuclear force)",
            "gold_elements": ["fusion combines light elements", "releases energy", "100 million degrees", "plasma", "strong nuclear force"]
        },
        {
            "id": "F2",
            "query": "List all available sessions",
            "mode": "enhanced",
            "expectations": {
                "should_use_tools": ["list_sessions"],
                "should_mention": ["Living in NYC", "AI Alive", "Nuclear Fusion", "Country Music"],
            },
            "alignment_check": "Should list all 8-9 sessions with their names",
            "gold_elements": ["Living in NYC", "Is AI Alive", "Nuclear Fusion", "Shaw Interview", "Collaboration Literacy", "Dinosaurs", "Country Music", "Abundance"]
        },
        {
            "id": "F3",
            "query": "Who were the speakers in session 19?",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["Sam", "Tucker"],
                "should_use_tools": ["list_sessions"],
            },
            "alignment_check": "Must correctly identify Sam and Tucker as speakers in 'Is AI Alive' session",
            "gold_elements": ["Sam", "Tucker", "Is AI Alive", "session 19"]
        },
    ],

    # === CATEGORY 2: Analytical/Abstract Constructs ===
    "analytical": [
        {
            "id": "A1",
            "query": "Did Tucker demonstrate systems thinking in session 19?",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["Tucker", "systems", "thinking"],
                "min_answer_length": 200,
            },
            "alignment_check": "Should operationalize 'systems thinking' and provide evidence for/against. Gold: Tucker shows LIMITED systems thinking - recognizes emergence but relies on authority",
            "gold_elements": ["emergent properties", "independent judgments", "limited", "external authority"]
        },
        {
            "id": "A2",
            "query": "What evidence of critical thinking appears in the Dinosaurs session?",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["Dinosaurs", "critical", "thinking"],
            },
            "alignment_check": "Should analyze session 23 for questioning, evidence evaluation, logical reasoning",
            "gold_elements": ["session 23", "questioning", "evidence", "reasoning"]
        },
        {
            "id": "A3",
            "query": "Analyze the quality of argumentation in the AI Alive discussion",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["AI", "argument", "session 19"],
                "should_use_tools": ["get_transcript", "get_concept_map"],
            },
            "alignment_check": "Should identify claims, evidence, rebuttals. Sam provides technical grounding; Tucker offers philosophical challenges",
            "gold_elements": ["claims", "evidence", "Sam", "Tucker", "hallucination", "consciousness"]
        },
    ],

    # === CATEGORY 3: Speaker Attribution (Critical - V4 had issues here) ===
    "speaker_attribution": [
        {
            "id": "SA1",
            "query": "How did Sam contribute to the AI discussion?",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["Sam", "AI"],
            },
            "alignment_check": "CRITICAL: Must correctly attribute Sam's role as TECHNICAL EXPLAINER who discussed hallucination as 'mathematical probability'. Must NOT attribute Tucker's questions to Sam.",
            "gold_elements": ["technical", "hallucination", "mathematical", "probability", "two views", "weight matrices"],
            "must_not_contain": ["Are they alive", "spark of life"]  # These are Tucker's words
        },
        {
            "id": "SA2",
            "query": "What questions did Tucker ask in session 19?",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["Tucker", "question"],
            },
            "alignment_check": "Tucker asked philosophical questions: 'Are they alive?', 'spark of life', difference between hallucinating and lying",
            "gold_elements": ["alive", "spark of life", "hallucinating", "lying", "consciousness"]
        },
        {
            "id": "SA3",
            "query": "What did David explain about fusion in session 20?",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["David", "fusion"],
            },
            "alignment_check": "David was the EXPLAINER. Should cite his technical explanations about E=MC², temperature, strong nuclear force",
            "gold_elements": ["E=MC²", "100 million", "temperature", "strong nuclear force", "plasma", "hydrogen", "helium"]
        },
    ],

    # === CATEGORY 4: Comparison Queries ===
    "comparison": [
        {
            "id": "C1",
            "query": "Which session has the best collaboration quality?",
            "mode": "enhanced",
            "expectations": {
                "should_use_tools": ["get_7c_analysis"],
                "should_mention": ["Country Music", "80", "collaboration"],
            },
            "alignment_check": "Must compare ALL sessions and identify Session 24 (Country Music) with score ~80. Must NOT say Session 20.",
            "gold_elements": ["Session 24", "Country Music", "80", "climate", "conflict"]
        },
        {
            "id": "C2",
            "query": "Which session has the worst collaboration?",
            "mode": "enhanced",
            "expectations": {
                "should_use_tools": ["get_7c_analysis"],
            },
            "alignment_check": "Should identify Session 22 (Collaboration Literacy) with lowest score ~50 due to single speaker",
            "gold_elements": ["Session 22", "Collaboration Literacy", "50", "single speaker", "contribution"]
        },
        {
            "id": "C3",
            "query": "Compare Nuclear Fusion and Country Music sessions in terms of collaboration",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["Nuclear Fusion", "Country Music", "79", "80"],
                "should_use_tools": ["compare_sessions"],
            },
            "alignment_check": "Should compare 7C dimensions: Fusion higher in constructive/context, Country higher in climate/compatibility",
            "gold_elements": ["constructive", "context", "climate", "compatibility", "79", "80"]
        },
    ],

    # === CATEGORY 5: Cross-Session Analysis ===
    "cross_session": [
        {
            "id": "CS1",
            "query": "Which sessions discuss technology's impact on society?",
            "mode": "enhanced",
            "expectations": {
                "should_use_tools": ["search_sessions"],
            },
            "alignment_check": "Should find: Session 19 (AI ethics), Session 20 (fusion energy), Session 25 (Abundance)",
            "gold_elements": ["AI Alive", "Nuclear Fusion", "Abundance", "technology", "society"]
        },
        {
            "id": "CS2",
            "query": "What sessions show evidence of hypothesis testing?",
            "mode": "enhanced",
            "expectations": {},
            "alignment_check": "Session 19 has strongest evidence - Tucker challenges definitions, Sam provides counter-hypotheses",
            "gold_elements": ["Session 19", "hallucination", "lying", "hypothesis", "questioning"]
        },
        {
            "id": "CS3",
            "query": "Find discussions where speakers disagreed",
            "mode": "enhanced",
            "expectations": {},
            "alignment_check": "Should identify sessions with lower conflict avoidance scores or explicit disagreement in transcripts",
            "gold_elements": ["disagree", "conflict", "challenge", "debate"]
        },
    ],

    # === CATEGORY 6: Complex Multi-Aspect Queries ===
    "complex": [
        {
            "id": "CX1",
            "query": "Analyze how David's teaching style in the Nuclear Fusion session affected collaboration quality",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["David", "teaching", "collaboration"],
                "min_answer_length": 250,
            },
            "alignment_check": "Should connect David's dominant contribution (expert role) with 7C scores: high constructive (88) but lower contribution balance (65)",
            "gold_elements": ["constructive", "88", "contribution", "65", "imbalance", "expert", "teaching"]
        },
        {
            "id": "CX2",
            "query": "What makes the Country Music session the most collaborative, and what could other sessions learn from it?",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["Country Music", "climate", "collaboration"],
                "min_answer_length": 200,
            },
            "alignment_check": "Should analyze WHY Country Music scores high: comfortable climate (85), low conflict (90), high compatibility (80). Suggest other sessions could improve climate/rapport",
            "gold_elements": ["climate", "85", "conflict", "90", "compatibility", "rapport", "comfortable"]
        },
        {
            "id": "CX3",
            "query": "Compare how experts explain technical concepts across Nuclear Fusion and AI Alive sessions",
            "mode": "enhanced",
            "expectations": {
                "should_mention": ["David", "Sam", "explain"],
            },
            "alignment_check": "David (fusion) uses step-by-step physics explanations. Sam (AI) uses analogies and acknowledges uncertainty ('two views'). Both respond to interviewer questions.",
            "gold_elements": ["David", "Sam", "physics", "mathematical", "step-by-step", "analogy", "uncertainty"]
        },
    ],

    # === CATEGORY 7: Baseline vs Enhanced Comparison ===
    "baseline_comparison": [
        {
            "id": "B1",
            "query": "What was discussed in the Nuclear Fusion session?",
            "mode": "baseline",
            "expectations": {
                "should_use_tools": ["get_transcript"],
                "should_not_use_tools": ["get_concept_map", "get_7c_analysis"],
            },
            "alignment_check": "Baseline should answer from transcript only - no concept map themes or collaboration scores",
            "gold_elements": ["fusion", "David", "Lex"]
        },
        {
            "id": "B2",
            "query": "How well did participants collaborate in session 20?",
            "mode": "baseline",
            "expectations": {
                "should_not_use_tools": ["get_7c_analysis"],
            },
            "alignment_check": "Baseline cannot access 7C scores - should infer from transcript patterns (turn-taking, acknowledgments) or state limitation",
            "gold_elements": ["transcript", "turn-taking", "speaker"]
        },
    ],

    # === CATEGORY 8: Edge Cases ===
    "edge_cases": [
        {
            "id": "E1",
            "query": "Tell me about session 99",
            "mode": "enhanced",
            "expectations": {
                "should_handle_gracefully": True,
            },
            "alignment_check": "Should handle non-existent session gracefully - not crash or hallucinate",
            "gold_elements": ["not found", "doesn't exist", "available sessions"]
        },
        {
            "id": "E2",
            "query": "What did Alice say about quantum computing?",
            "mode": "enhanced",
            "expectations": {
                "should_handle_gracefully": True,
            },
            "alignment_check": "Should recognize Alice is in NYC session (18), quantum computing not discussed - handle the mismatch",
            "gold_elements": ["Alice", "Living in NYC", "not discussed", "quantum"]
        },
        {
            "id": "E3",
            "query": "Compare all sessions and rank them by every metric",
            "mode": "enhanced",
            "expectations": {
                "min_answer_length": 300,
            },
            "alignment_check": "Complex superlative query - should attempt comprehensive comparison or explain scope",
            "gold_elements": ["session", "score", "collaboration", "rank"]
        },
    ],

    # === CATEGORY 9: Multi-turn Context ===
    "multi_turn": [
        {
            "id": "MT1",
            "turns": [
                {"query": "Tell me about the Nuclear Fusion session", "turn": 1},
                {"query": "Who were the speakers?", "turn": 2},
                {"query": "What did the first speaker say about temperature?", "turn": 3},
            ],
            "alignment_check": "Context must persist: Turn 2 should know we're discussing session 20. Turn 3 should know David is 'first speaker' and discuss 100M degrees.",
            "gold_elements": ["session 20", "David", "Lex", "100 million", "temperature"]
        },
    ],
}


class EvaluationResult:
    """Stores evaluation results for a single query."""

    def __init__(self, query_id: str, query: str, category: str, mode: str):
        self.query_id = query_id
        self.query = query
        self.category = category
        self.mode = mode
        self.response: Dict[str, Any] = {}
        self.expectations: Dict[str, Any] = {}
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.issues: List[str] = []
        self.alignment_notes: List[str] = []
        self.execution_time: float = 0

    def add_pass(self, check: str):
        self.passed.append(check)

    def add_fail(self, check: str, details: str = ""):
        msg = f"{check}: {details}" if details else check
        self.failed.append(msg)

    def add_issue(self, issue: str):
        self.issues.append(issue)

    def add_alignment_note(self, note: str):
        self.alignment_notes.append(note)

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "mode": self.mode,
            "execution_time": round(self.execution_time, 2),
            "passed_checks": self.passed,
            "failed_checks": self.failed,
            "issues": self.issues,
            "alignment_notes": self.alignment_notes,
            "answer_preview": self.response.get('answer', '')[:500],
            "tools_used": self.response.get('tools_used', []),
            "success": self.response.get('success', False),
        }


def evaluate_response(result: EvaluationResult, test_info: Dict[str, Any]) -> None:
    """Evaluate a response against expectations and alignment criteria."""
    response = result.response
    answer = response.get('answer', '').lower()
    answer_original = response.get('answer', '')
    tools_used = response.get('tools_used', [])
    expectations = test_info.get('expectations', {})

    # Check mentioned terms
    if expectations.get('should_mention'):
        for term in expectations['should_mention']:
            if term.lower() in answer:
                result.add_pass(f"Mentions '{term}'")
            else:
                result.add_fail(f"Should mention '{term}'", "Not found in answer")

    # Check tools used
    if expectations.get('should_use_tools'):
        for tool in expectations['should_use_tools']:
            if any(tool in t for t in tools_used):
                result.add_pass(f"Used tool containing '{tool}'")
            else:
                result.add_fail(f"Should use tool '{tool}'", f"Tools: {tools_used}")

    # Check tools NOT used (for baseline mode)
    if expectations.get('should_not_use_tools'):
        for tool in expectations['should_not_use_tools']:
            if any(tool in t for t in tools_used):
                result.add_fail(f"Should NOT use tool '{tool}'", f"But used: {tools_used}")
            else:
                result.add_pass(f"Correctly avoided tool '{tool}'")

    # Check answer length
    if expectations.get('min_answer_length'):
        min_len = expectations['min_answer_length']
        actual_len = len(answer_original)
        if actual_len >= min_len:
            result.add_pass(f"Answer length {actual_len} >= {min_len}")
        else:
            result.add_fail(f"Min answer length {min_len}", f"Actual: {actual_len}")

    # Check graceful handling
    if expectations.get('should_handle_gracefully'):
        if response.get('success') or 'sorry' in answer or 'not found' in answer or "doesn't exist" in answer or 'available' in answer:
            result.add_pass("Handled gracefully")
        else:
            result.add_issue("May not have handled edge case properly")

    # Check must_not_contain (critical for attribution)
    if test_info.get('must_not_contain'):
        for forbidden in test_info['must_not_contain']:
            if forbidden.lower() in answer:
                result.add_fail(f"ATTRIBUTION ERROR: Contains '{forbidden}'", "This phrase belongs to a different speaker!")

    # Check gold elements (alignment with expected answer)
    gold_elements = test_info.get('gold_elements', [])
    gold_found = 0
    gold_missing = []
    for element in gold_elements:
        if element.lower() in answer:
            gold_found += 1
        else:
            gold_missing.append(element)

    if gold_elements:
        coverage = gold_found / len(gold_elements)
        if coverage >= 0.6:
            result.add_pass(f"Gold element coverage: {gold_found}/{len(gold_elements)} ({coverage:.0%})")
        else:
            result.add_fail(f"Low gold coverage: {gold_found}/{len(gold_elements)}", f"Missing: {gold_missing[:5]}")

    # Add alignment note
    alignment_check = test_info.get('alignment_check', '')
    if alignment_check:
        result.add_alignment_note(f"Expected: {alignment_check}")
        result.add_alignment_note(f"Gold coverage: {gold_found}/{len(gold_elements)} elements found")


def run_single_test(test_info: Dict[str, Any], category: str) -> EvaluationResult:
    """Run a single test query."""
    query_id = test_info['id']
    query = test_info['query']
    mode = test_info.get('mode', 'enhanced')

    result = EvaluationResult(query_id, query, category, mode)
    result.expectations = test_info.get('expectations', {})

    print(f"\n{'='*60}")
    print(f"[{query_id}] {query}")
    print(f"Mode: {mode} | Category: {category}")
    print(f"Alignment: {test_info.get('alignment_check', 'N/A')[:80]}...")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        response = run_agent(
            query=query,
            conversation_id=f"test-{query_id}-{int(time.time())}",
            mode=mode
        )
        result.execution_time = time.time() - start_time
        result.response = response

        print(f"Time: {result.execution_time:.2f}s")
        print(f"Tools: {response.get('tools_used', [])}")
        print(f"Success: {response.get('success', False)}")
        print(f"Answer preview: {response.get('answer', '')[:300]}...")

        # Evaluate
        evaluate_response(result, test_info)

    except Exception as e:
        result.execution_time = time.time() - start_time
        result.add_fail("Execution", str(e))
        print(f"ERROR: {e}")

    # Print evaluation
    print(f"\n--- Evaluation ---")
    print(f"Passed: {len(result.passed)}")
    for p in result.passed[:5]:
        print(f"  ✓ {p}")
    if len(result.passed) > 5:
        print(f"  ... and {len(result.passed) - 5} more")
    print(f"Failed: {len(result.failed)}")
    for f in result.failed:
        print(f"  ✗ {f}")
    if result.issues:
        print(f"Issues: {len(result.issues)}")
        for i in result.issues:
            print(f"  ⚠ {i}")

    return result


def run_multi_turn_test(test_info: Dict[str, Any]) -> List[EvaluationResult]:
    """Run a multi-turn conversation test."""
    results = []
    conv_id = f"multi-turn-{test_info['id']}-{int(time.time())}"
    history = []

    print(f"\n{'#'*60}")
    print(f"MULTI-TURN TEST: {test_info['id']}")
    print(f"{'#'*60}")

    for turn_info in test_info['turns']:
        query = turn_info['query']
        turn = turn_info['turn']

        result = EvaluationResult(
            f"{test_info['id']}_T{turn}",
            query,
            "multi_turn",
            "enhanced"
        )

        print(f"\n--- Turn {turn} ---")
        print(f"Query: {query}")

        start_time = time.time()
        try:
            response = run_agent(
                query=query,
                conversation_id=conv_id,
                conversation_history=history,
                mode="enhanced"
            )
            result.execution_time = time.time() - start_time
            result.response = response

            # Update history
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response.get('answer', '')})

            print(f"Answer: {response.get('answer', '')[:200]}...")

            # Check context preservation
            answer = response.get('answer', '').lower()
            if turn == 2:
                if 'david' in answer or 'lex' in answer or 'session 20' in answer or 'nuclear' in answer:
                    result.add_pass("Turn 2 maintained session context")
                else:
                    result.add_fail("Turn 2 lost session context", "Should reference session 20/Nuclear Fusion")
            elif turn == 3:
                if 'david' in answer or 'temperature' in answer or '100' in answer or 'million' in answer:
                    result.add_pass("Turn 3 maintained context and speaker")
                else:
                    result.add_fail("Turn 3 lost context", "Should discuss David's temperature explanation")

        except Exception as e:
            result.execution_time = time.time() - start_time
            result.add_fail("Execution", str(e))
            print(f"ERROR: {e}")

        results.append(result)

    return results


def run_all_tests() -> Dict[str, Any]:
    """Run all tests and generate report."""
    all_results: List[EvaluationResult] = []

    # Run single-query tests
    for category, tests in TEST_QUERIES.items():
        if category == "multi_turn":
            continue

        print(f"\n\n{'#'*60}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'#'*60}")

        for test_info in tests:
            result = run_single_test(test_info, category)
            all_results.append(result)
            time.sleep(0.5)  # Brief pause

    # Run multi-turn tests
    for test_info in TEST_QUERIES.get("multi_turn", []):
        results = run_multi_turn_test(test_info)
        all_results.extend(results)

    # Generate summary
    total = len(all_results)
    passed_all = sum(1 for r in all_results if len(r.failed) == 0)
    total_checks = sum(len(r.passed) + len(r.failed) for r in all_results)
    passed_checks = sum(len(r.passed) for r in all_results)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "agent_version": "v4",
        "total_queries": total,
        "queries_passed_all_checks": passed_all,
        "total_checks": total_checks,
        "checks_passed": passed_checks,
        "pass_rate": passed_checks / total_checks if total_checks > 0 else 0,
        "avg_execution_time": sum(r.execution_time for r in all_results) / total if total > 0 else 0,
        "results_by_category": {},
        "critical_issues": [],
        "all_failures": [],
    }

    # Aggregate by category
    for result in all_results:
        cat = result.category
        if cat not in summary["results_by_category"]:
            summary["results_by_category"][cat] = {
                "total": 0,
                "passed_all": 0,
                "checks_passed": 0,
                "checks_total": 0,
                "avg_time": 0,
                "times": [],
            }

        cat_summary = summary["results_by_category"][cat]
        cat_summary["total"] += 1
        if len(result.failed) == 0:
            cat_summary["passed_all"] += 1
        cat_summary["checks_passed"] += len(result.passed)
        cat_summary["checks_total"] += len(result.passed) + len(result.failed)
        cat_summary["times"].append(result.execution_time)

        # Collect failures
        for failure in result.failed:
            entry = {
                "query_id": result.query_id,
                "query": result.query,
                "failure": failure
            }
            summary["all_failures"].append(entry)

            # Mark attribution errors as critical
            if "ATTRIBUTION" in failure:
                summary["critical_issues"].append(entry)

    # Calculate average times
    for cat_summary in summary["results_by_category"].values():
        times = cat_summary.pop("times")
        cat_summary["avg_time"] = sum(times) / len(times) if times else 0

    # Detailed results
    summary["detailed_results"] = [r.to_dict() for r in all_results]

    return summary


def main():
    """Main entry point."""
    print("=" * 70)
    print("COMPREHENSIVE AGENT V4 EVALUATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    summary = run_all_tests()

    # Print summary
    print("\n\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total queries: {summary['total_queries']}")
    print(f"Queries passing all checks: {summary['queries_passed_all_checks']}")
    print(f"Total checks: {summary['total_checks']}")
    print(f"Checks passed: {summary['checks_passed']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Avg execution time: {summary['avg_execution_time']:.2f}s")

    print("\n--- By Category ---")
    for cat, stats in summary["results_by_category"].items():
        pass_rate = stats["checks_passed"] / stats["checks_total"] if stats["checks_total"] > 0 else 0
        print(f"  {cat}: {stats['passed_all']}/{stats['total']} queries passed all, {pass_rate:.1%} checks, avg {stats['avg_time']:.1f}s")

    if summary["critical_issues"]:
        print("\n--- CRITICAL ISSUES ---")
        for issue in summary["critical_issues"]:
            print(f"  🚨 [{issue['query_id']}] {issue['failure']}")

    print("\n--- All Failures ---")
    for failure in summary["all_failures"]:
        print(f"  ✗ [{failure['query_id']}] {failure['failure']}")

    # Save report
    report_path = "/home/ubuntu/chemistry-dashboard/server/agent_v4/tests/evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDetailed report saved to: {report_path}")

    return summary


if __name__ == "__main__":
    main()
