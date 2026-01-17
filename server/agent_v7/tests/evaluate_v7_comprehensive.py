"""
Comprehensive Evaluation of V7 Agent

Tests holistic queries and evaluates:
1. Tool selection correctness
2. Multi-step reasoning (chaining)
3. Cross-session capability
4. Response completeness
5. Alignment with expected behavior

Run: python -m agent_v7.tests.evaluate_v7_comprehensive
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import V7 components
from agent_v7.tools_v2 import CORE_TOOLS, execute_tool
from agent_v7.react_agent import ScaffoldingAgent, run_agent

# =============================================================================
# TEST QUERIES WITH EXPECTATIONS
# =============================================================================

TEST_QUERIES = [
    # --- SINGLE SESSION QUERIES (should work well) ---
    {
        "id": "single_1",
        "query": "What was discussed in session 25?",
        "category": "single_session",
        "expected_tools": ["get_transcript"],
        "expected_behavior": "Retrieve transcript for session 25 and summarize",
        "success_criteria": [
            "Calls get_transcript with session_id=25",
            "Response includes specific quotes from session",
            "Mentions speakers by name"
        ]
    },
    {
        "id": "single_2",
        "query": "How did participants collaborate in session 22?",
        "category": "single_session",
        "expected_tools": ["get_7c_analysis"],
        "expected_behavior": "Retrieve 7C analysis and discuss collaboration patterns",
        "success_criteria": [
            "Calls get_7c_analysis with session_id=22",
            "Mentions specific 7C dimensions with scores",
            "Cites coded segments as evidence"
        ]
    },
    {
        "id": "single_3",
        "query": "What ideas emerged in session 25's concept map?",
        "category": "single_session",
        "expected_tools": ["get_concept_map"],
        "expected_behavior": "Retrieve concept map and describe idea structure",
        "success_criteria": [
            "Calls get_concept_map with session_id=25",
            "Mentions node types (ideas, questions, etc.)",
            "Describes relationships between concepts"
        ]
    },

    # --- CROSS-SESSION QUERIES (the critical test) ---
    {
        "id": "cross_1",
        "query": "What was said about AI across all sessions?",
        "category": "cross_session",
        "expected_tools": ["search_sessions", "get_transcript"],  # Multiple transcripts
        "expected_behavior": "Search for AI-related sessions, retrieve transcripts from EACH, synthesize",
        "success_criteria": [
            "Calls search_sessions with 'AI' query",
            "Retrieves transcripts from multiple sessions (not just one)",
            "Synthesizes findings across sessions",
            "Cites specific quotes from different sessions"
        ]
    },
    {
        "id": "cross_2",
        "query": "Which sessions discussed technology and media?",
        "category": "cross_session",
        "expected_tools": ["search_sessions"],
        "expected_behavior": "Search for relevant sessions and list them",
        "success_criteria": [
            "Calls search_sessions",
            "Returns multiple session IDs",
            "Provides session names and relevance"
        ]
    },
    {
        "id": "cross_3",
        "query": "Compare how different sessions handled disagreements",
        "category": "cross_session",
        "expected_tools": ["search_sessions", "get_7c_analysis"],  # Multiple 7C analyses
        "expected_behavior": "Get 7C conflict dimension from multiple sessions, compare",
        "success_criteria": [
            "Retrieves 7C analysis from multiple sessions",
            "Focuses on Conflict dimension",
            "Compares patterns across sessions"
        ]
    },

    # --- SPEAKER QUERIES ---
    {
        "id": "speaker_1",
        "query": "How did Lex engage in discussions?",
        "category": "speaker",
        "expected_tools": ["get_speaker_profile"],
        "expected_behavior": "Get speaker profile showing engagement patterns",
        "success_criteria": [
            "Calls get_speaker_profile with 'Lex'",
            "Shows sessions participated",
            "Shows metrics and sample quotes",
            "Mentions concept contributions"
        ]
    },
    {
        "id": "speaker_2",
        "query": "What questions did Ezra ask in session 25?",
        "category": "speaker",
        "expected_tools": ["get_transcript"],
        "expected_behavior": "Get transcript filtered by speaker and question content",
        "success_criteria": [
            "Calls get_transcript with session_id=25, speaker_filter='Ezra'",
            "OR calls get_transcript and filters in response",
            "Lists specific questions asked"
        ]
    },
    {
        "id": "speaker_3",
        "query": "How did Derek's contributions differ across sessions?",
        "category": "speaker",
        "expected_tools": ["get_speaker_profile", "get_transcript"],
        "expected_behavior": "Get speaker profile, potentially drill into specific sessions",
        "success_criteria": [
            "Calls get_speaker_profile for Derek",
            "Compares metrics across sessions",
            "May chain to get_transcript for details"
        ]
    },

    # --- AGGREGATION/COMPARISON QUERIES ---
    {
        "id": "agg_1",
        "query": "Which session had the best collaboration?",
        "category": "aggregation",
        "expected_tools": ["list_sessions", "get_7c_analysis"],  # Multiple
        "expected_behavior": "Get 7C scores from all sessions, compare, identify best",
        "success_criteria": [
            "Retrieves 7C analysis from multiple sessions",
            "Compares overall scores",
            "Identifies the best with justification"
        ]
    },
    {
        "id": "agg_2",
        "query": "Who asked the most questions across all discussions?",
        "category": "aggregation",
        "expected_tools": ["list_sessions", "get_speaker_profile"],
        "expected_behavior": "Get speaker profiles, compare question counts",
        "success_criteria": [
            "Identifies speakers across sessions",
            "Compares question counts",
            "Names the speaker with most questions"
        ]
    },

    # --- DISCOVERY QUERIES ---
    {
        "id": "disc_1",
        "query": "What sessions are available?",
        "category": "discovery",
        "expected_tools": ["list_sessions"],
        "expected_behavior": "List all sessions with metadata",
        "success_criteria": [
            "Calls list_sessions",
            "Shows session IDs and names",
            "Shows speakers for each"
        ]
    },
]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def run_single_query(conversation_id: str, query: str) -> Dict[str, Any]:
    """Run a single query through the agent and capture results."""

    result = {
        "query": query,
        "tools_called": [],
        "tool_outputs": [],
        "final_response": None,
        "turns": 0,
        "error": None
    }

    try:
        # Run the agent using convenience function
        response = run_agent(conversation_id, query)

        result["final_response"] = response.answer
        result["turns"] = len(response.tool_calls_made)
        # Convert ToolCall objects to dicts for evaluation
        result["tools_called"] = [
            {"tool": tc.name, "params": tc.params, "reason": tc.reason}
            for tc in response.tool_calls_made
        ]
        result["tool_outputs"] = response.evidence

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error running query: {e}", exc_info=True)

    return result


def evaluate_result(test_case: Dict, result: Dict) -> Dict[str, Any]:
    """Evaluate a result against expected behavior."""

    evaluation = {
        "test_id": test_case["id"],
        "query": test_case["query"],
        "category": test_case["category"],
        "passed_criteria": [],
        "failed_criteria": [],
        "issues": [],
        "overall_pass": False
    }

    tools_called = [t["tool"] if isinstance(t, dict) else t for t in result.get("tools_called", [])]
    response = result.get("final_response", "") or ""

    # Check each success criterion
    for criterion in test_case["success_criteria"]:
        passed = False

        # Tool-related criteria
        if "Calls" in criterion:
            tool_name = criterion.split("Calls ")[1].split(" ")[0]
            if tool_name in tools_called:
                passed = True

        # Response content criteria
        elif "mentions" in criterion.lower() or "shows" in criterion.lower() or "includes" in criterion.lower():
            # These need manual evaluation - mark as needs_review
            passed = "needs_review"

        # Multi-session criteria
        elif "multiple sessions" in criterion.lower():
            # Check if multiple session IDs appear in tool calls
            session_ids_in_calls = set()
            for t in result.get("tools_called", []):
                if isinstance(t, dict) and "params" in t:
                    sid = t["params"].get("session_id")
                    if sid:
                        session_ids_in_calls.add(sid)
            passed = len(session_ids_in_calls) > 1

        if passed == True:
            evaluation["passed_criteria"].append(criterion)
        elif passed == "needs_review":
            evaluation["passed_criteria"].append(f"[NEEDS_REVIEW] {criterion}")
        else:
            evaluation["failed_criteria"].append(criterion)

    # Identify issues
    if result.get("error"):
        evaluation["issues"].append(f"Error: {result['error']}")

    if not tools_called:
        evaluation["issues"].append("No tools were called")

    # For cross-session queries, check if only one session was retrieved
    if test_case["category"] == "cross_session":
        session_ids = set()
        for t in result.get("tools_called", []):
            if isinstance(t, dict) and "params" in t:
                sid = t["params"].get("session_id")
                if sid:
                    session_ids.add(sid)

        if len(session_ids) <= 1 and "get_transcript" in tools_called:
            evaluation["issues"].append(
                f"CROSS-SESSION FAILURE: Only retrieved from {len(session_ids)} session(s). "
                "Expected to retrieve from multiple sessions."
            )

    # Overall pass if no failed criteria and no critical issues
    critical_issues = [i for i in evaluation["issues"] if "FAILURE" in i or "Error" in i]
    evaluation["overall_pass"] = (
        len(evaluation["failed_criteria"]) == 0 and
        len(critical_issues) == 0
    )

    return evaluation


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_comprehensive_evaluation():
    """Run all test queries and generate evaluation report."""

    print("=" * 70)
    print("V7 COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print()

    # Verify imports work
    try:
        from agent_v7.react_agent import run_agent
        print("Agent imports successful")
    except Exception as e:
        print(f"Failed to import agent: {e}")
        return

    results = []
    evaluations = []

    for i, test_case in enumerate(TEST_QUERIES):
        print(f"\n[{i+1}/{len(TEST_QUERIES)}] Testing: {test_case['id']}")
        print(f"Query: {test_case['query']}")
        print(f"Category: {test_case['category']}")
        print("-" * 50)

        # Run query with unique conversation ID per test
        conversation_id = f"eval_{test_case['id']}_{datetime.now().timestamp()}"
        result = run_single_query(conversation_id, test_case["query"])
        results.append(result)

        # Evaluate
        evaluation = evaluate_result(test_case, result)
        evaluations.append(evaluation)

        # Print summary
        tool_names = [t["tool"] if isinstance(t, dict) else t for t in result.get('tools_called', [])]
        print(f"Tools called: {tool_names}")
        print(f"Turns: {result.get('turns', 0)}")

        if evaluation["issues"]:
            print(f"ISSUES: {evaluation['issues']}")

        if evaluation["failed_criteria"]:
            print(f"FAILED: {evaluation['failed_criteria']}")

        print(f"Overall: {'PASS' if evaluation['overall_pass'] else 'FAIL'}")

    # Generate summary report
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # By category
    categories = {}
    for ev in evaluations:
        cat = ev["category"]
        if cat not in categories:
            categories[cat] = {"passed": 0, "failed": 0, "issues": []}

        if ev["overall_pass"]:
            categories[cat]["passed"] += 1
        else:
            categories[cat]["failed"] += 1
            categories[cat]["issues"].extend(ev["issues"])

    print("\nResults by Category:")
    for cat, stats in categories.items():
        total = stats["passed"] + stats["failed"]
        print(f"  {cat}: {stats['passed']}/{total} passed")
        if stats["issues"]:
            for issue in stats["issues"][:3]:  # First 3 issues
                print(f"    - {issue}")

    # Critical issues
    all_issues = []
    for ev in evaluations:
        for issue in ev["issues"]:
            if "FAILURE" in issue:
                all_issues.append({
                    "test": ev["test_id"],
                    "query": ev["query"],
                    "issue": issue
                })

    if all_issues:
        print("\n" + "=" * 70)
        print("CRITICAL ISSUES FOUND")
        print("=" * 70)
        for issue in all_issues:
            print(f"\nTest: {issue['test']}")
            print(f"Query: {issue['query']}")
            print(f"Issue: {issue['issue']}")

    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(TEST_QUERIES),
        "passed": sum(1 for e in evaluations if e["overall_pass"]),
        "failed": sum(1 for e in evaluations if not e["overall_pass"]),
        "categories": categories,
        "critical_issues": all_issues,
        "detailed_results": [
            {
                "test_case": TEST_QUERIES[i],
                "result": results[i],
                "evaluation": evaluations[i]
            }
            for i in range(len(TEST_QUERIES))
        ]
    }

    report_path = os.path.join(os.path.dirname(__file__), "v7_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nDetailed report saved to: {report_path}")

    return report


if __name__ == "__main__":
    run_comprehensive_evaluation()
