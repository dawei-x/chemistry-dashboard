#!/usr/bin/env python3
"""
V7 Scaffolding Agent - Comprehensive Evaluation

Tests the new ReAct-based agent with various query types and evaluates 
outputs against expectations.

Run with:
    cd /home/ubuntu/chemistry-dashboard/server
    ~/.pyenv/versions/blinc/bin/python agent_v7/tests/evaluate_v7.py

Author: Claude (AI Assistant)
Date: 2026-01-15
"""

import sys
import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

# Add server directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# TEST QUERIES - Based on actual database content
# =============================================================================

# Sessions in database:
# 18: Living in NYC (Alice, Bob, Vanessa)
# 19: Is AI Alive (Tucker, Sam)
# 20: Nuclear Fusion (David, Lex)
# 21: Shaw Interview (Julia, Lex)
# 22: Collaboration Literacy (Speaker 17)
# 23: Dinosaurs (Dave, Lex)
# 24: Country Music (Oliver, Lex)
# 25: Abundance (Derek, Ezra, Lex)
# 26: CFAA Discussion (SPEAKER_00, SPEAKER_01, SPEAKER_02)

SINGLE_TURN_QUERIES = [
    # Basic Session Queries
    {
        "id": "basic_01",
        "query": "What sessions are available?",
        "category": "basic",
        "expected_tools": ["list_sessions"],
        "expected_in_answer": ["Living in NYC", "Is AI Alive", "Nuclear Fusion"],
        "description": "Should list all available sessions"
    },
    {
        "id": "basic_02", 
        "query": "Tell me about session 19",
        "category": "basic",
        "expected_tools": ["get_session_overview"],
        "expected_in_answer": ["AI", "Tucker", "Sam"],
        "description": "Should provide overview of Is AI Alive session"
    },
    {
        "id": "basic_03",
        "query": "What was the Nuclear Fusion discussion about?",
        "category": "basic",
        "expected_tools": ["get_session_overview", "get_transcript"],
        "expected_in_answer": ["fusion", "energy"],
        "description": "Should identify session by name and summarize"
    },
    
    # Speaker Queries
    {
        "id": "speaker_01",
        "query": "What did Tucker say in session 19?",
        "category": "speaker",
        "expected_tools": ["get_transcript"],
        "expected_in_answer": ["Tucker"],
        "description": "Should filter Tucker's contributions"
    },
    {
        "id": "speaker_02",
        "query": "Who were the main speakers in the Country Music discussion?",
        "category": "speaker",
        "expected_tools": ["get_session_overview"],
        "expected_in_answer": ["Oliver", "Lex"],
        "description": "Should identify speakers in session 24"
    },
    
    # 7C Analysis Queries
    {
        "id": "7c_01",
        "query": "How was the collaboration in the Living in NYC discussion?",
        "category": "7c_analysis",
        "expected_tools": ["get_7c_analysis"],
        "expected_in_answer": ["collaboration", "score"],
        "description": "Should retrieve 7C analysis for session 18"
    },
    {
        "id": "7c_02",
        "query": "What evidence shows the Nuclear Fusion discussion had good context awareness?",
        "category": "7c_analysis",
        "expected_tools": ["get_7c_analysis"],
        "expected_in_answer": ["context", "evidence"],
        "description": "Should cite specific 7C evidence (context score 90)"
    },
    
    # Concept Map Queries
    {
        "id": "concept_01",
        "query": "What concepts were discussed in session 20?",
        "category": "concept_map",
        "expected_tools": ["get_concept_map"],
        "expected_in_answer": ["concept"],
        "description": "Should list concepts from Nuclear Fusion"
    },
    
    # Comparison Queries
    {
        "id": "compare_01",
        "query": "Compare sessions 19 and 20",
        "category": "comparison",
        "expected_tools": ["compare_sessions", "get_session_overview"],
        "expected_in_answer": ["session"],
        "description": "Should compare AI Alive vs Nuclear Fusion"
    },
    
    # Hypothesis Verification
    {
        "id": "hypothesis_01",
        "query": "I think Nuclear Fusion had better collaboration than the AI discussion. Is that true?",
        "category": "hypothesis",
        "expected_tools": ["get_7c_analysis"],
        "expected_in_answer": ["score", "collaboration"],
        "description": "Should compare 7C scores and verify/refute"
    },
    
    # Scaffolding Test
    {
        "id": "scaffold_01",
        "query": "Give me specific quotes from the AI discussion about consciousness",
        "category": "scaffolding",
        "expected_tools": ["get_transcript"],
        "expected_in_answer": ['"'],  # Should have actual quotes
        "description": "Must include actual transcript quotes"
    },
    
    # Search Queries
    {
        "id": "search_01",
        "query": "Find discussions about energy",
        "category": "search",
        "expected_tools": ["search_sessions"],
        "expected_in_answer": ["Nuclear", "Fusion"],
        "description": "Should find Nuclear Fusion session"
    },
    
    # Steering Tests (LLM should understand preference)
    {
        "id": "steer_01",
        "query": "Focus on the transcript only - tell me about session 20",
        "category": "steering",
        "expected_tools": ["get_transcript"],
        "not_expected_tools": ["get_7c_analysis", "get_concept_map"],
        "description": "LLM should understand to use only transcript"
    },
    {
        "id": "steer_02",
        "query": "Don't use 7C analysis, just tell me what people said in session 19",
        "category": "steering",
        "expected_tools": ["get_transcript"],
        "not_expected_tools": ["get_7c_analysis"],
        "description": "LLM should understand to skip 7C"
    },
    
    # Edge Cases
    {
        "id": "edge_01",
        "query": "Tell me about session 999",
        "category": "edge_case",
        "expected_tools": ["get_session_overview"],
        "expected_error_handling": True,
        "description": "Should handle non-existent session gracefully"
    },
]

MULTI_TURN_TESTS = [
    {
        "id": "multi_01",
        "name": "Session Focus Persistence",
        "turns": [
            {
                "query": "Tell me about session 20",
                "expected_in_answer": ["Nuclear", "Fusion"]
            },
            {
                "query": "Who were the speakers?",
                "expected_in_answer": ["David", "Lex"]
            },
            {
                "query": "What was the collaboration quality?",
                "expected_in_answer": ["7c", "score"]
            }
        ],
        "description": "Follow-ups should maintain session 20 context"
    },
    {
        "id": "multi_02",
        "name": "Speaker Context Switch",
        "turns": [
            {
                "query": "What did Tucker say in session 19?",
                "expected_in_answer": ["Tucker"]
            },
            {
                "query": "What about Sam?",
                "expected_in_answer": ["Sam"]
            }
        ],
        "description": "Should maintain session 19 when asking about Sam"
    }
]


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    query_id: str
    query: str
    category: str
    answer: str = ""
    tools_used: List[str] = field(default_factory=list)
    iterations: int = 0
    elapsed_time: float = 0.0
    
    # Evaluation
    answered: bool = False
    scaffolded: bool = False
    accurate: bool = False
    expected_tools_ok: bool = False
    steering_compliant: bool = False
    
    issues: List[str] = field(default_factory=list)
    notes: str = ""


def check_content(text: str, expected: List[str], case_insensitive: bool = True) -> bool:
    """Check if text contains expected content."""
    if case_insensitive:
        text = text.lower()
        return all(e.lower() in text for e in expected)
    return all(e in text for e in expected)


def evaluate_single_query(spec: Dict, agent_func) -> QueryResult:
    """Evaluate a single query."""
    query_id = spec["id"]
    query = spec["query"]
    category = spec["category"]
    
    print(f"\n{'='*60}")
    print(f"[{query_id}] {query}")
    print(f"Category: {category}")
    print(f"Description: {spec.get('description', 'N/A')}")
    
    result = QueryResult(query_id=query_id, query=query, category=category)
    
    start = time.time()
    try:
        response = agent_func(query, conversation_id=f"eval_{query_id}")
        result.elapsed_time = time.time() - start
        
        result.answer = response.get("answer", "")
        result.tools_used = response.get("tools_used", [])
        result.iterations = response.get("iterations", 0)
        
        print(f"Answer: {result.answer[:200]}..." if len(result.answer) > 200 else f"Answer: {result.answer}")
        print(f"Tools: {result.tools_used}")
        print(f"Iterations: {result.iterations}, Time: {result.elapsed_time:.2f}s")
        
        # Evaluate
        result.answered = len(result.answer) > 20
        
        # Check scaffolding
        result.scaffolded = any(x in result.answer.lower() for x in ['"', 'score', 'evidence', 'said'])
        
        # Check expected content
        expected = spec.get("expected_in_answer", [])
        if expected:
            result.accurate = check_content(result.answer, expected)
            if not result.accurate:
                result.issues.append(f"Missing expected content: {expected}")
        else:
            result.accurate = result.answered
        
        # Check tools
        expected_tools = spec.get("expected_tools", [])
        not_expected = spec.get("not_expected_tools", [])
        
        if expected_tools:
            used_expected = any(t in result.tools_used for t in expected_tools)
            if not used_expected:
                result.issues.append(f"Expected tools {expected_tools}, used {result.tools_used}")
        else:
            used_expected = True
        
        # Check not-expected tools (steering compliance)
        if not_expected:
            used_not_expected = any(t in result.tools_used for t in not_expected)
            if used_not_expected:
                result.issues.append(f"Used excluded tools: {[t for t in not_expected if t in result.tools_used]}")
                result.steering_compliant = False
            else:
                result.steering_compliant = True
        else:
            result.steering_compliant = True
        
        result.expected_tools_ok = used_expected and result.steering_compliant
        
        # Check error handling for edge cases
        if spec.get("expected_error_handling"):
            if "not found" in result.answer.lower() or "doesn't exist" in result.answer.lower() or "no session" in result.answer.lower():
                result.accurate = True
            else:
                result.issues.append("Should handle missing session gracefully")
        
        result.notes = spec.get("description", "")
        
    except Exception as e:
        result.elapsed_time = time.time() - start
        result.issues.append(f"Exception: {str(e)}")
        print(f"ERROR: {e}")
    
    # Print evaluation
    status = "✓" if result.answered and result.accurate and result.expected_tools_ok else "✗"
    print(f"Result: {status} answered={result.answered}, accurate={result.accurate}, tools_ok={result.expected_tools_ok}")
    if result.issues:
        print(f"Issues: {result.issues}")
    
    return result


def evaluate_multi_turn(spec: Dict, agent_func, reset_func) -> List[QueryResult]:
    """Evaluate a multi-turn conversation."""
    test_id = spec["id"]
    test_name = spec["name"]
    
    print(f"\n{'='*60}")
    print(f"MULTI-TURN: [{test_id}] {test_name}")
    print(f"Description: {spec.get('description', 'N/A')}")
    print(f"{'='*60}")
    
    conv_id = f"eval_multi_{test_id}"
    results = []
    
    for i, turn in enumerate(spec["turns"]):
        query = turn["query"]
        expected = turn.get("expected_in_answer", [])
        
        print(f"\n--- Turn {i+1}: {query}")
        
        result = QueryResult(
            query_id=f"{test_id}_turn{i+1}",
            query=query,
            category="multi_turn"
        )
        
        start = time.time()
        try:
            response = agent_func(query, conversation_id=conv_id)
            result.elapsed_time = time.time() - start
            
            result.answer = response.get("answer", "")
            result.tools_used = response.get("tools_used", [])
            result.iterations = response.get("iterations", 0)
            
            print(f"Answer: {result.answer[:150]}...")
            
            result.answered = len(result.answer) > 20
            result.accurate = check_content(result.answer, expected) if expected else True
            
            if not result.accurate and expected:
                result.issues.append(f"Missing: {expected}")
            
        except Exception as e:
            result.elapsed_time = time.time() - start
            result.issues.append(f"Exception: {str(e)}")
            print(f"ERROR: {e}")
        
        results.append(result)
    
    # Reset for next test
    reset_func(conv_id)
    
    return results


def run_evaluation(categories: List[str] = None, skip_multi: bool = False):
    """Run the full evaluation."""
    from agent_v7.graph_v2 import invoke_agent, reset_conversation
    
    print("\n" + "="*80)
    print("V7 SCAFFOLDING AGENT - COMPREHENSIVE EVALUATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)
    
    all_results = []
    
    # Filter queries
    queries = SINGLE_TURN_QUERIES
    if categories:
        queries = [q for q in queries if q["category"] in categories]
    
    print(f"Running {len(queries)} single-turn queries")
    if not skip_multi:
        print(f"Running {len(MULTI_TURN_TESTS)} multi-turn tests")
    
    # Single-turn queries
    for spec in queries:
        result = evaluate_single_query(spec, invoke_agent)
        all_results.append(result)
        time.sleep(0.5)  # Brief pause
    
    # Multi-turn tests
    if not skip_multi:
        for spec in MULTI_TURN_TESTS:
            results = evaluate_multi_turn(spec, invoke_agent, reset_conversation)
            all_results.extend(results)
            time.sleep(0.5)
    
    # Summary
    total = len(all_results)
    answered = sum(1 for r in all_results if r.answered)
    accurate = sum(1 for r in all_results if r.accurate)
    scaffolded = sum(1 for r in all_results if r.scaffolded)
    tools_ok = sum(1 for r in all_results if r.expected_tools_ok)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total queries: {total}")
    print(f"Answered: {answered}/{total} ({answered/total*100:.1f}%)")
    print(f"Accurate: {accurate}/{total} ({accurate/total*100:.1f}%)")
    print(f"Scaffolded: {scaffolded}/{total} ({scaffolded/total*100:.1f}%)")
    print(f"Tools OK: {tools_ok}/{total} ({tools_ok/total*100:.1f}%)")
    print(f"Avg time: {sum(r.elapsed_time for r in all_results)/total:.2f}s")
    print(f"Avg iterations: {sum(r.iterations for r in all_results)/total:.1f}")
    
    # By category
    categories_found = set(r.category for r in all_results)
    print("\nBy Category:")
    for cat in sorted(categories_found):
        cat_results = [r for r in all_results if r.category == cat]
        cat_answered = sum(1 for r in cat_results if r.answered)
        cat_accurate = sum(1 for r in cat_results if r.accurate)
        print(f"  {cat}: {cat_accurate}/{len(cat_results)} accurate")
    
    # All issues
    all_issues = [(r.query_id, r.query, issue) for r in all_results for issue in r.issues]
    if all_issues:
        print("\n" + "="*80)
        print("ALL ISSUES FOUND")
        print("="*80)
        for qid, query, issue in all_issues:
            print(f"[{qid}] {query[:40]}...")
            print(f"  -> {issue}")
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "answered": answered,
            "accurate": accurate,
            "scaffolded": scaffolded,
            "tools_ok": tools_ok
        },
        "issues": [{"id": i[0], "issue": i[2]} for i in all_issues],
        "results": [
            {
                "id": r.query_id,
                "query": r.query,
                "answered": r.answered,
                "accurate": r.accurate,
                "tools_used": r.tools_used,
                "issues": r.issues
            }
            for r in all_results
        ]
    }
    
    report_path = os.path.join(os.path.dirname(__file__), "v7_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    return all_results


def quick_test():
    """Quick single-query test."""
    from agent_v7.graph_v2 import invoke_agent
    
    print("Quick test: What sessions are available?")
    result = invoke_agent("What sessions are available?", conversation_id="quick_test")
    print(f"Answer: {result.get('answer', 'No answer')}")
    print(f"Tools: {result.get('tools_used', [])}")
    print(f"Iterations: {result.get('iterations', 0)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test only")
    parser.add_argument("--category", type=str, help="Run specific category")
    parser.add_argument("--skip-multi", action="store_true", help="Skip multi-turn")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        cats = [args.category] if args.category else None
        run_evaluation(categories=cats, skip_multi=args.skip_multi)
