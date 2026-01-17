#!/usr/bin/env python3
"""
V7 Extended Evaluation - V3's Challenging Test Cases

Tests V7 agent on the more challenging queries from V3's test suite,
focusing on:
1. Abstract construct operationalization ("systems thinking", "critical thinking")
2. Cross-session thematic queries
3. Concept path tracing
4. Superlative queries ("best", "worst")

Run with:
    cd /home/ubuntu/chemistry-dashboard/server
    ~/.pyenv/versions/blinc/bin/python agent_v7/tests/v7_extended_eval.py
"""

import sys
import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Challenging queries from V3 test suite
CHALLENGING_QUERIES = [
    # === ANALYTICAL: Abstract Construct Operationalization ===
    {
        "id": "anal_01",
        "query": "Did Tucker demonstrate systems thinking in session 19?",
        "category": "analytical",
        "challenge": "Must operationalize 'systems thinking' into observable indicators",
        "expected_in_answer": ["Tucker"],
        "expected_artifact_types": ["transcript", "concept_map"],
        "notes": "Abstract construct - needs indicators like interconnections, feedback, holistic view"
    },
    {
        "id": "anal_02",
        "query": "What evidence shows critical thinking in the Dinosaurs session?",
        "category": "analytical",
        "challenge": "Must resolve session name AND operationalize 'critical thinking'",
        "expected_in_answer": ["session 23", "dinosaur"],
        "notes": "Needs to resolve 'Dinosaurs' -> session 23, then find critical thinking indicators"
    },
    {
        "id": "anal_03",
        "query": "Show me examples of hypothesis testing in the Nuclear Fusion discussion",
        "category": "analytical",
        "challenge": "Operationalize 'hypothesis testing' - proposing and evaluating ideas",
        "expected_in_answer": ["hypothesis", "test", "propos"],
        "notes": "Look for statements like 'what if', 'I think that', 'let's test'"
    },

    # === COMPARISON: Cross-Session Global Analysis ===
    {
        "id": "comp_01",
        "query": "Which session has the best collaboration quality?",
        "category": "comparison",
        "challenge": "Must compare ALL sessions, not just a subset",
        "expected_in_answer": ["session"],
        "expected_artifact_types": ["7c_analysis"],
        "notes": "Should analyze 7C scores for all sessions and identify highest"
    },
    {
        "id": "comp_02",
        "query": "What's the worst collaboration across all sessions?",
        "category": "comparison",
        "challenge": "Superlative query requiring global comparison",
        "expected_in_answer": ["session"],
        "notes": "Should find session with lowest 7C scores"
    },
    {
        "id": "comp_03",
        "query": "Compare the depth of discussion in AI Alive vs Nuclear Fusion",
        "category": "comparison",
        "challenge": "Must resolve session names AND compare depth",
        "expected_in_answer": ["session 19", "session 20"],
        "notes": "Depth could be measured by concept map complexity, 7C scores, etc."
    },

    # === THEMATIC: Cross-Session Topic Discovery ===
    {
        "id": "theme_01",
        "query": "What sessions discussed technology and its societal impact?",
        "category": "thematic",
        "challenge": "Cross-session thematic search",
        "expected_in_answer": ["session"],
        "notes": "Should find AI Alive (19), Nuclear Fusion (20), Abundance (25)"
    },
    {
        "id": "theme_02",
        "query": "Find all discussions about energy across sessions",
        "category": "thematic",
        "challenge": "Cross-session search for 'energy' theme",
        "expected_in_answer": ["Nuclear", "Fusion"],
        "notes": "Primary: Nuclear Fusion (20), possibly Abundance (25)"
    },
    {
        "id": "theme_03",
        "query": "Which sessions show hypothesis testing or scientific reasoning?",
        "category": "thematic",
        "challenge": "Exploratory query - should search multiple sessions",
        "expected_in_answer": ["session"],
        "notes": "Should NOT constrain to single session"
    },

    # === GRAPH: Concept Path Tracing ===
    {
        "id": "graph_01",
        "query": "How are ideas about fusion connected to energy in session 20?",
        "category": "graph",
        "challenge": "Trace concept connections in concept map",
        "expected_in_answer": ["fusion", "energy"],
        "expected_artifact_types": ["concept_map"],
        "notes": "Should use concept map to trace paths"
    },
    {
        "id": "graph_02",
        "query": "What's the connection between AI consciousness and ethics in session 19?",
        "category": "graph",
        "challenge": "Trace concept paths in concept map",
        "expected_in_answer": ["AI", "consciousness"],
        "notes": "Should find concept connections"
    },

    # === SPEAKER: Multi-Speaker Analysis ===
    {
        "id": "speak_01",
        "query": "Compare Tucker and Sam's contributions in session 19",
        "category": "speaker",
        "challenge": "Analyze and compare two speakers",
        "expected_in_answer": ["Tucker", "Sam"],
        "notes": "Should characterize each speaker's contributions"
    },
    {
        "id": "speak_02",
        "query": "Who contributed the most ideas in the Nuclear Fusion discussion?",
        "category": "speaker",
        "challenge": "Quantitative speaker analysis",
        "expected_in_answer": ["David", "Lex"],
        "notes": "Should analyze concept map or transcript for idea attribution"
    },
]


@dataclass
class QueryResult:
    """Result of a single query evaluation."""
    query_id: str
    query: str
    category: str
    challenge: str
    answer: str = ""
    tools_used: List[str] = field(default_factory=list)
    elapsed_time: float = 0.0

    # Evaluation
    answered: bool = False
    relevant: bool = False
    grounded: bool = False

    observations: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


def analyze_response(result: QueryResult, spec: Dict) -> None:
    """Analyze response quality beyond simple checks."""
    answer = result.answer.lower()
    tools = result.tools_used

    # Check if answered at all
    result.answered = len(result.answer) > 50

    # Check expected content
    expected = spec.get("expected_in_answer", [])
    if expected:
        found = [e for e in expected if e.lower() in answer]
        missing = [e for e in expected if e.lower() not in answer]
        if found:
            result.observations.append(f"Found expected: {found}")
        if missing:
            result.issues.append(f"Missing expected: {missing}")
        result.relevant = len(found) >= len(expected) / 2
    else:
        result.relevant = result.answered

    # Check artifact usage
    expected_artifacts = spec.get("expected_artifact_types", [])
    if expected_artifacts:
        artifact_to_tool = {
            "transcript": "get_transcript",
            "concept_map": "get_concept_map",
            "7c_analysis": "get_7c_analysis",
        }
        for artifact in expected_artifacts:
            expected_tool = artifact_to_tool.get(artifact)
            if expected_tool and expected_tool in tools:
                result.observations.append(f"Used {artifact}")
            elif expected_tool:
                result.issues.append(f"Should use {artifact} but didn't")

    # Check grounding (has specific evidence)
    grounding_indicators = ['"', 'said', 'stated', 'score', 'according to', 'in the transcript']
    result.grounded = any(ind in answer for ind in grounding_indicators)
    if result.grounded:
        result.observations.append("Response is grounded with evidence")

    # Check for abstract construct handling
    if spec["category"] == "analytical":
        operationalization_indicators = [
            'example', 'evidence', 'shows', 'demonstrates', 'indicates',
            'connection', 'relationship', 'pattern', 'because'
        ]
        has_operationalization = any(ind in answer for ind in operationalization_indicators)
        if has_operationalization:
            result.observations.append("Shows operationalization of abstract construct")
        else:
            result.issues.append("May not be operationalizing abstract construct")

    # Check for cross-session handling
    if spec["category"] in ["comparison", "thematic"]:
        session_mentions = answer.count("session")
        if session_mentions >= 2:
            result.observations.append(f"References multiple sessions ({session_mentions} mentions)")
        else:
            result.issues.append("May not be comparing across sessions")


def run_evaluation():
    """Run evaluation on challenging queries."""
    from agent_v7.graph_v2 import invoke_agent, reset_conversation

    print("\n" + "="*80)
    print("V7 EXTENDED EVALUATION - V3 CHALLENGING QUERIES")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)

    all_results = []

    for spec in CHALLENGING_QUERIES:
        query_id = spec["id"]
        query = spec["query"]
        category = spec["category"]
        challenge = spec["challenge"]

        print(f"\n{'='*70}")
        print(f"[{query_id}] {query}")
        print(f"Category: {category}")
        print(f"Challenge: {challenge}")
        print("-"*70)

        result = QueryResult(
            query_id=query_id,
            query=query,
            category=category,
            challenge=challenge
        )

        conv_id = f"eval_ext_{query_id}"

        start = time.time()
        try:
            response = invoke_agent(query, conversation_id=conv_id)
            result.elapsed_time = time.time() - start

            result.answer = response.get("answer", "")
            result.tools_used = response.get("tools_used", [])

            print(f"Tools: {result.tools_used}")
            print(f"Time: {result.elapsed_time:.2f}s")
            print(f"Answer preview: {result.answer[:300]}...")

            # Deep analysis
            analyze_response(result, spec)

        except Exception as e:
            result.elapsed_time = time.time() - start
            result.issues.append(f"Exception: {str(e)}")
            print(f"ERROR: {e}")

        # Print analysis
        print("\n--- Analysis ---")
        print(f"Answered: {result.answered}, Relevant: {result.relevant}, Grounded: {result.grounded}")
        if result.observations:
            print("Observations:")
            for obs in result.observations:
                print(f"  + {obs}")
        if result.issues:
            print("Issues:")
            for issue in result.issues:
                print(f"  - {issue}")

        all_results.append(result)
        reset_conversation(conv_id)
        time.sleep(0.5)

    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    total = len(all_results)
    answered = sum(1 for r in all_results if r.answered)
    relevant = sum(1 for r in all_results if r.relevant)
    grounded = sum(1 for r in all_results if r.grounded)

    print(f"Total queries: {total}")
    print(f"Answered: {answered}/{total} ({answered/total*100:.1f}%)")
    print(f"Relevant: {relevant}/{total} ({relevant/total*100:.1f}%)")
    print(f"Grounded: {grounded}/{total} ({grounded/total*100:.1f}%)")

    # By category
    categories = set(r.category for r in all_results)
    print("\nBy Category:")
    for cat in sorted(categories):
        cat_results = [r for r in all_results if r.category == cat]
        cat_relevant = sum(1 for r in cat_results if r.relevant)
        cat_grounded = sum(1 for r in cat_results if r.grounded)
        print(f"  {cat}: {cat_relevant}/{len(cat_results)} relevant, {cat_grounded}/{len(cat_results)} grounded")

    # All issues summary
    print("\n" + "="*80)
    print("ALL ISSUES BY CATEGORY")
    print("="*80)

    for cat in sorted(categories):
        cat_results = [r for r in all_results if r.category == cat]
        cat_issues = [(r.query_id, r.challenge, issue)
                      for r in cat_results for issue in r.issues]
        if cat_issues:
            print(f"\n{cat.upper()}:")
            for qid, challenge, issue in cat_issues:
                print(f"  [{qid}] {challenge}")
                print(f"      -> {issue}")

    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "answered": answered,
            "relevant": relevant,
            "grounded": grounded
        },
        "by_category": {},
        "results": []
    }

    for cat in sorted(categories):
        cat_results = [r for r in all_results if r.category == cat]
        report["by_category"][cat] = {
            "total": len(cat_results),
            "relevant": sum(1 for r in cat_results if r.relevant),
            "grounded": sum(1 for r in cat_results if r.grounded)
        }

    for r in all_results:
        report["results"].append({
            "id": r.query_id,
            "query": r.query,
            "category": r.category,
            "challenge": r.challenge,
            "tools_used": r.tools_used,
            "answered": r.answered,
            "relevant": r.relevant,
            "grounded": r.grounded,
            "observations": r.observations,
            "issues": r.issues,
            "answer_preview": r.answer[:500] if r.answer else ""
        })

    report_path = os.path.join(os.path.dirname(__file__), "v7_extended_eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    return all_results


if __name__ == "__main__":
    run_evaluation()
