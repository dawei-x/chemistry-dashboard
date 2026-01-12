#!/usr/bin/env python3
"""
Comprehensive Agent V3 Evaluation Suite

Tests the agent with various query types and evaluates outputs
against expectations. Documents issues found.

Run with: ~/.pyenv/versions/blinc/bin/python -m agent_v3.tests.comprehensive_evaluation
"""

import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add server directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_v3.graph import run_agent, reset_graph

# Test queries organized by category
TEST_QUERIES = {
    # === CATEGORY 1: Fast Path (Simple) ===
    "fast_path": [
        {
            "id": "F1",
            "query": "What was the Nuclear Fusion session about?",
            "expectations": {
                "should_use_fast_path": True,
                "should_mention": ["fusion", "energy", "temperature", "plasma"],
                "should_have_quotes": True,
                "min_answer_length": 100,
            },
            "tricky_aspect": "Should include specific technical details from transcript quotes"
        },
        {
            "id": "F2",
            "query": "List all sessions",
            "expectations": {
                "should_use_fast_path": True,
                "should_list_sessions": [18, 19, 20, 21, 22, 23, 24, 25],
                "min_answer_length": 50,
            },
            "tricky_aspect": "Should list all 8 sessions (18-25)"
        },
        {
            "id": "F3",
            "query": "Tell me about session 19",
            "expectations": {
                "should_use_fast_path": True,
                "should_mention": ["AI", "alive", "consciousness"],
                "should_have_quotes": True,
            },
            "tricky_aspect": "Should extract session name and key themes"
        },
    ],

    # === CATEGORY 2: Analytical (PRAS Path) ===
    "analytical": [
        {
            "id": "A1",
            "query": "Did Tucker demonstrate systems thinking in session 19?",
            "expectations": {
                "should_use_pras": True,
                "should_use_tools": ["search_transcripts", "get_concept_map"],
                "should_mention": ["Tucker", "systems", "thinking"],
                "should_have_citations": True,
                "min_tools_used": 3,
            },
            "tricky_aspect": "Requires operationalizing 'systems thinking' into indicators"
        },
        {
            "id": "A2",
            "query": "How well did participants collaborate in session 20?",
            "expectations": {
                "should_use_pras": True,
                "should_use_tools": ["get_collaboration_analysis"],
                "should_mention": ["7C", "collaboration", "score"],
                "should_have_citations": True,
            },
            "tricky_aspect": "Should use 7C analysis and cite specific dimension scores"
        },
        {
            "id": "A3",
            "query": "What evidence shows critical thinking in the Dinosaurs session?",
            "expectations": {
                "should_use_pras": True,
                "should_mention": ["Dinosaurs", "critical", "thinking"],
                "should_have_citations": True,
                "session_should_be": 23,
            },
            "tricky_aspect": "Should resolve 'Dinosaurs' to session 23 and operationalize 'critical thinking'"
        },
    ],

    # === CATEGORY 3: Comparison Queries ===
    "comparison": [
        {
            "id": "C1",
            "query": "Which session has the best collaboration quality?",
            "expectations": {
                "should_compare_all_sessions": True,
                "correct_answer_session": 24,  # Country Music has highest 7C (80)
                "should_mention": ["Session 24", "Country Music", "80"],
                "should_use_tools": ["compare_sessions"],
            },
            "tricky_aspect": "Must compare ALL sessions (18-25), not just a subset"
        },
        {
            "id": "C2",
            "query": "Compare the AI Alive and Nuclear Fusion sessions",
            "expectations": {
                "should_use_pras": True,
                "should_mention": ["Session 19", "Session 20", "AI", "fusion"],
                "sessions_compared": [19, 20],
            },
            "tricky_aspect": "Should resolve session names and compare specific aspects"
        },
        {
            "id": "C3",
            "query": "Which sessions discussed technology and its societal impact?",
            "expectations": {
                "should_search_across_sessions": True,
                "likely_sessions": [19, 20, 25],  # AI Alive, Nuclear Fusion, Abundance
            },
            "tricky_aspect": "Cross-session thematic search - should find multiple relevant sessions"
        },
    ],

    # === CATEGORY 4: Graph/Path Queries ===
    "graph": [
        {
            "id": "G1",
            "query": "How are ideas about fusion connected to energy in session 20?",
            "expectations": {
                "should_use_graph_tools": True,
                "should_use_tools": ["find_concept_path", "get_concept_map"],
                "should_mention": ["fusion", "energy", "connection"],
            },
            "tricky_aspect": "Should use find_concept_path for path tracing"
        },
        {
            "id": "G2",
            "query": "What's the connection between AI consciousness and ethics in session 19?",
            "expectations": {
                "should_use_graph_tools": True,
                "should_mention": ["AI", "consciousness", "ethics"],
            },
            "tricky_aspect": "Should trace concept paths in the concept map"
        },
    ],

    # === CATEGORY 5: Speaker-Focused ===
    "speaker": [
        {
            "id": "S1",
            "query": "What did David say about fusion in session 20?",
            "expectations": {
                "should_filter_by_speaker": True,
                "speaker": "David",
                "session": 20,
                "should_have_citations": True,
            },
            "tricky_aspect": "Should filter transcript search by speaker"
        },
        {
            "id": "S2",
            "query": "Compare Tucker and David's contributions in session 19",
            "expectations": {
                "should_compare_speakers": True,
                "speakers": ["Tucker", "David"],
                "session": 19,
            },
            "tricky_aspect": "Should analyze both speakers' contributions"
        },
    ],

    # === CATEGORY 6: Edge Cases / Tricky ===
    "edge_cases": [
        {
            "id": "E1",
            "query": "What sessions show hypothesis testing?",
            "expectations": {
                "should_search_across_sessions": True,
                "should_not_limit_to_one_session": True,
            },
            "tricky_aspect": "Exploratory query - should NOT be constrained to single session"
        },
        {
            "id": "E2",
            "query": "Tell me about it",  # Ambiguous without context
            "expectations": {
                "should_ask_for_clarification": True,
            },
            "tricky_aspect": "No context - should ask for clarification or handle gracefully",
            "conversation_id": "new_conversation"
        },
        {
            "id": "E3",
            "query": "What's the worst collaboration?",
            "expectations": {
                "should_compare_all_sessions": True,
                "should_find_lowest": True,
            },
            "tricky_aspect": "Superlative query (worst) - needs global comparison"
        },
        {
            "id": "E4",
            "query": "Session 99 overview",  # Non-existent session
            "expectations": {
                "should_handle_gracefully": True,
                "should_indicate_not_found": True,
            },
            "tricky_aspect": "Invalid session ID - should handle error gracefully"
        },
    ],

    # === CATEGORY 7: Multi-turn Context ===
    "multi_turn": [
        {
            "id": "M1",
            "queries": [
                {"query": "Tell me about the Nuclear Fusion session", "turn": 1},
                {"query": "Who were the speakers?", "turn": 2},
                {"query": "What did David specifically say about temperature?", "turn": 3},
            ],
            "expectations": {
                "turn_2_should_reference_session_20": True,
                "turn_3_should_reference_session_20_and_david": True,
            },
            "tricky_aspect": "Multi-turn context preservation - later turns should maintain session focus"
        },
    ],
}


class EvaluationResult:
    """Stores evaluation results for a single query."""

    def __init__(self, query_id: str, query: str, category: str):
        self.query_id = query_id
        self.query = query
        self.category = category
        self.response: Dict[str, Any] = {}
        self.expectations: Dict[str, Any] = {}
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.issues: List[str] = []
        self.execution_time: float = 0
        self.raw_response: Dict[str, Any] = {}

    def add_pass(self, check: str):
        self.passed.append(check)

    def add_fail(self, check: str, details: str = ""):
        msg = f"{check}: {details}" if details else check
        self.failed.append(msg)

    def add_issue(self, issue: str):
        self.issues.append(issue)

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "execution_time": self.execution_time,
            "passed_checks": self.passed,
            "failed_checks": self.failed,
            "issues": self.issues,
            "answer_preview": self.response.get('final_answer', '')[:500],
            "tools_used": self.response.get('tools_used', []),
            "confidence": self.response.get('confidence', 0),
            "citations_count": len(self.response.get('citations', [])),
        }


def evaluate_response(result: EvaluationResult, expectations: Dict[str, Any]) -> None:
    """Evaluate a response against expectations."""
    response = result.response
    answer = response.get('final_answer', '').lower()
    tools_used = response.get('tools_used', [])
    citations = response.get('citations', [])

    # Check fast path
    if expectations.get('should_use_fast_path'):
        route = response.get('route', '')
        if route == 'fast_path' or 'fast_path' in str(response.get('thought_history', [])):
            result.add_pass("Used fast path")
        else:
            result.add_fail("Should use fast path", f"Route was: {route}")

    # Check PRAS path
    if expectations.get('should_use_pras'):
        pras_stage = response.get('pras_stage', '')
        if pras_stage or 'pras' in str(response.get('thought_history', [])).lower():
            result.add_pass("Used PRAS path")
        else:
            result.add_fail("Should use PRAS path", f"pras_stage was: {pras_stage}")

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
            if tool in tools_used:
                result.add_pass(f"Used tool '{tool}'")
            else:
                result.add_fail(f"Should use tool '{tool}'", f"Tools used: {tools_used}")

    # Check minimum tools
    if expectations.get('min_tools_used'):
        min_tools = expectations['min_tools_used']
        if len(tools_used) >= min_tools:
            result.add_pass(f"Used {len(tools_used)} tools (min: {min_tools})")
        else:
            result.add_fail(f"Min tools {min_tools}", f"Only used {len(tools_used)}: {tools_used}")

    # Check citations
    if expectations.get('should_have_citations'):
        if len(citations) > 0:
            result.add_pass(f"Has {len(citations)} citations")
        else:
            result.add_fail("Should have citations", "No citations found")

    # Check quotes in answer
    if expectations.get('should_have_quotes'):
        if '"' in response.get('final_answer', '') or "said" in answer:
            result.add_pass("Contains quotes")
        else:
            result.add_issue("Expected quotes in answer but none found")

    # Check answer length
    if expectations.get('min_answer_length'):
        min_len = expectations['min_answer_length']
        actual_len = len(response.get('final_answer', ''))
        if actual_len >= min_len:
            result.add_pass(f"Answer length {actual_len} >= {min_len}")
        else:
            result.add_fail(f"Min answer length {min_len}", f"Actual: {actual_len}")

    # Check session list
    if expectations.get('should_list_sessions'):
        expected_sessions = expectations['should_list_sessions']
        found_all = all(str(s) in answer for s in expected_sessions)
        if found_all:
            result.add_pass(f"Listed all {len(expected_sessions)} sessions")
        else:
            missing = [s for s in expected_sessions if str(s) not in answer]
            result.add_fail("Should list all sessions", f"Missing: {missing}")

    # Check correct answer for comparison
    if expectations.get('correct_answer_session'):
        correct = expectations['correct_answer_session']
        if f"session {correct}" in answer or f"#{correct}" in answer:
            result.add_pass(f"Correct answer: Session {correct}")
        else:
            result.add_fail(f"Correct answer should be Session {correct}", f"Answer: {answer[:200]}")

    # Check graph tools
    if expectations.get('should_use_graph_tools'):
        graph_tools = ['find_concept_path', 'explore_concepts', 'get_concept_map', 'find_reasoning_path']
        used_graph = any(t in tools_used for t in graph_tools)
        if used_graph:
            result.add_pass(f"Used graph tools: {[t for t in tools_used if t in graph_tools]}")
        else:
            result.add_fail("Should use graph tools", f"Tools used: {tools_used}")

    # Check compare all sessions
    if expectations.get('should_compare_all_sessions'):
        if 'compare_sessions' in tools_used:
            result.add_pass("Used compare_sessions tool")
        else:
            result.add_fail("Should use compare_sessions", f"Tools: {tools_used}")

    # Check error handling
    if expectations.get('should_handle_gracefully'):
        if response.get('success', True) or 'sorry' in answer or 'not found' in answer or 'error' in answer:
            result.add_pass("Handled gracefully")
        else:
            result.add_issue("May not have handled error case properly")


def run_single_test(
    query_info: Dict[str, Any],
    category: str,
    conversation_id: str = None
) -> EvaluationResult:
    """Run a single test query and evaluate."""
    query_id = query_info['id']
    query = query_info['query']
    expectations = query_info.get('expectations', {})

    result = EvaluationResult(query_id, query, category)
    result.expectations = expectations

    conv_id = conversation_id or f"test-{query_id}-{int(time.time())}"

    print(f"\n{'='*60}")
    print(f"[{query_id}] {query}")
    print(f"Category: {category}")
    print(f"Tricky aspect: {query_info.get('tricky_aspect', 'N/A')}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        response = run_agent(
            query=query,
            conversation_id=conv_id,
            conversation_context=None
        )
        result.execution_time = time.time() - start_time
        result.response = response
        result.raw_response = response

        print(f"Time: {result.execution_time:.2f}s")
        print(f"Tools: {response.get('tools_used', [])}")
        print(f"Confidence: {response.get('confidence', 0):.2f}")
        print(f"Citations: {len(response.get('citations', []))}")
        print(f"Answer preview: {response.get('final_answer', '')[:300]}...")

        # Evaluate
        evaluate_response(result, expectations)

    except Exception as e:
        result.execution_time = time.time() - start_time
        result.add_fail("Execution", str(e))
        print(f"ERROR: {e}")

    # Print evaluation summary
    print(f"\n--- Evaluation ---")
    print(f"Passed: {len(result.passed)}")
    for p in result.passed:
        print(f"  ✓ {p}")
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
    context = None

    print(f"\n{'#'*60}")
    print(f"MULTI-TURN TEST: {test_info['id']}")
    print(f"{'#'*60}")

    for turn_info in test_info['queries']:
        query = turn_info['query']
        turn = turn_info['turn']

        result = EvaluationResult(
            f"{test_info['id']}_T{turn}",
            query,
            "multi_turn"
        )

        print(f"\n--- Turn {turn} ---")
        print(f"Query: {query}")

        start_time = time.time()
        try:
            response = run_agent(
                query=query,
                conversation_id=conv_id,
                conversation_context=context
            )
            result.execution_time = time.time() - start_time
            result.response = response

            # Update context for next turn
            context = {
                'current_session_focus': response.get('current_session_focus'),
                'session_history': response.get('session_history', []),
                'previous_query': query,
            }

            print(f"Session focus: {response.get('current_session_focus')}")
            print(f"Answer: {response.get('final_answer', '')[:200]}...")

            # Check context preservation
            if turn > 1:
                session_focus = response.get('current_session_focus')
                if session_focus:
                    result.add_pass(f"Maintained session focus: {session_focus}")
                else:
                    result.add_fail("Lost session focus", "current_session_focus is None")

        except Exception as e:
            result.execution_time = time.time() - start_time
            result.add_fail("Execution", str(e))
            print(f"ERROR: {e}")

        results.append(result)

    return results


def run_all_tests() -> Dict[str, Any]:
    """Run all tests and generate report."""
    all_results: List[EvaluationResult] = []

    # Reset graph to ensure clean state
    reset_graph()

    # Run single-query tests
    for category, queries in TEST_QUERIES.items():
        if category == "multi_turn":
            continue  # Handle separately

        print(f"\n\n{'#'*60}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'#'*60}")

        for query_info in queries:
            result = run_single_test(query_info, category)
            all_results.append(result)
            time.sleep(1)  # Brief pause between queries

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
        "total_queries": total,
        "queries_passed_all_checks": passed_all,
        "total_checks": total_checks,
        "checks_passed": passed_checks,
        "pass_rate": passed_checks / total_checks if total_checks > 0 else 0,
        "avg_execution_time": sum(r.execution_time for r in all_results) / total if total > 0 else 0,
        "results_by_category": {},
        "all_issues": [],
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
            }

        cat_summary = summary["results_by_category"][cat]
        cat_summary["total"] += 1
        if len(result.failed) == 0:
            cat_summary["passed_all"] += 1
        cat_summary["checks_passed"] += len(result.passed)
        cat_summary["checks_total"] += len(result.passed) + len(result.failed)

        # Collect issues and failures
        for issue in result.issues:
            summary["all_issues"].append({
                "query_id": result.query_id,
                "issue": issue
            })
        for failure in result.failed:
            summary["all_failures"].append({
                "query_id": result.query_id,
                "failure": failure
            })

    # Detailed results
    summary["detailed_results"] = [r.to_dict() for r in all_results]

    return summary


def main():
    """Main entry point."""
    print("=" * 70)
    print("COMPREHENSIVE AGENT V3 EVALUATION")
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
        print(f"  {cat}: {stats['passed_all']}/{stats['total']} queries passed all, {pass_rate:.1%} checks passed")

    print("\n--- All Failures ---")
    for failure in summary["all_failures"]:
        print(f"  [{failure['query_id']}] {failure['failure']}")

    print("\n--- All Issues ---")
    for issue in summary["all_issues"]:
        print(f"  [{issue['query_id']}] {issue['issue']}")

    # Save detailed report
    report_path = "/home/ubuntu/chemistry-dashboard/server/agent_v3/tests/evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDetailed report saved to: {report_path}")

    return summary


if __name__ == "__main__":
    main()
