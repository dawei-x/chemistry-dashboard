#!/usr/bin/env python3
"""
Comprehensive evaluation of Agent V6 with 48 test queries.
Handles rate limits and logs all results.
"""

import requests
import json
import time
import sys
from datetime import datetime

BASE_URL = "http://localhost:5000/api/v6/agent/query"

QUERIES = [
    # Session Discovery (Q1-3)
    "What sessions are available?",
    "List all the sessions",
    "What was discussed recently?",

    # Session Overview (Q4-8)
    "Tell me about the Nuclear Fusion session",
    "Give me an overview of the AI Alive session",
    "What topics were covered in the Abundance session?",
    "Who participated in the AI Alive session?",
    "What's the main theme of the Collaboration Literacy session?",

    # Collaboration Analysis (Q9-20)
    "Tell me about collaboration in the Nuclear Fusion session",
    "How was the collaboration quality in the Nuclear Fusion session?",
    "What are the 7C scores for the AI Alive session?",
    "Which session had the best collaboration?",
    "Why did some discussions have higher engagement than others?",
    "What were the strengths and weaknesses of collaboration in the Collaboration Literacy session?",
    "How did participants interact in the AI Alive session?",
    "Was there good participation balance in the Nuclear Fusion session?",
    "What was the communication quality like in the Abundance session?",
    "Did the AI Alive session have any conflict or disagreement?",
    "How constructive was the discussion in the Nuclear Fusion session?",
    "Which sessions had the most balanced contributions?",

    # Transcript/Quote queries (Q21-30)
    "What did David say about fusion in the Nuclear Fusion session?",
    "Find quotes about AI reasoning in the AI Alive session",
    "What specific statements did Tucker make about intelligence?",
    "Show me what was said about energy in the Nuclear Fusion session",
    "What questions were asked in the Collaboration Literacy session?",
    "Find quotes showing disagreement or constructive conflict",
    "What did speakers say about the future of AI?",
    "Show me key statements from the Collaboration Literacy session",
    "What was mentioned about learning analytics?",
    "Find quotes where speakers expressed uncertainty",

    # Concept Map queries (Q31-40)
    "What ideas were discussed in the Nuclear Fusion session?",
    "How do concepts connect in the Nuclear Fusion session?",
    "What problems were identified in the AI Alive session?",
    "Show me the solutions proposed in the AI Alive session",
    "What hypotheses were raised about AI?",
    "How did the discussion evolve from nuclear fusion basics to energy applications?",
    "What clusters of ideas emerged in the Collaboration Literacy session?",
    "Trace the reasoning path from fusion to energy in the Nuclear Fusion session",
    "What goals were identified in the Collaboration Literacy session?",
    "How are the ideas about AI connected to broader themes?",

    # Speaker Analysis (Q41-48)
    "Tell me about Tucker's contributions",
    "How did David contribute to the Nuclear Fusion discussion?",
    "What was Sam's communication style in the AI Alive session?",
    "Compare Tucker and Sam's participation patterns",
    "Who contributed the most ideas in the Nuclear Fusion session?",
    "What types of contributions did Lex make?",
    "How analytical was David's speaking style?",
    "Which speakers asked the most questions?",
]

CATEGORIES = [
    ("Session Discovery", 0, 3),
    ("Session Overview", 3, 8),
    ("Collaboration Analysis", 8, 20),
    ("Transcript/Quotes", 20, 30),
    ("Concept Map", 30, 40),
    ("Speaker Analysis", 40, 48),
]


def run_query(query, retry_count=3):
    """Run a single query with retry on rate limit."""
    for attempt in range(retry_count):
        try:
            response = requests.post(
                BASE_URL,
                json={"query": query},
                timeout=120
            )
            data = response.json()

            if "rate_limit" in str(data.get("error", "")).lower():
                wait_time = 5 * (attempt + 1)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            return data
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)

    return {"error": "Max retries exceeded", "success": False}


def evaluate_response(query, response, query_num):
    """Evaluate a single response and return issues."""
    issues = []

    success = response.get("success", False)
    answer = response.get("answer", "")
    tools = response.get("tools_used", [])
    error = response.get("error")

    # Check for errors
    if error:
        issues.append(f"Error: {error[:100]}")
        return issues

    if not success:
        issues.append("Query failed (success=False)")
        return issues

    # Check for empty answer
    if len(answer) < 50:
        issues.append(f"Very short answer ({len(answer)} chars)")

    # Check tool usage expectations
    query_lower = query.lower()

    # Session list queries should use list_sessions
    if any(w in query_lower for w in ["sessions available", "list all", "list the sessions"]):
        if "list_sessions" not in tools:
            issues.append("Expected list_sessions tool but not used")

    # 7C queries should use get_7c_analysis
    if "7c" in query_lower or "collaboration quality" in query_lower:
        if "get_7c_analysis" not in tools:
            issues.append("Expected get_7c_analysis tool but not used")

    # Speaker-specific queries should use speaker tools
    if any(name in query_lower for name in ["david", "tucker", "sam", "lex"]):
        speaker_tools = ["get_speaker_utterances", "get_speaker_profile", "get_transcript"]
        if not any(t in tools for t in speaker_tools):
            issues.append("Speaker query but no speaker/transcript tools used")

    # Concept map queries should use get_concept_map
    if any(w in query_lower for w in ["concepts connect", "ideas", "clusters", "reasoning path"]):
        if "get_concept_map" not in tools and "find_concept_path" not in tools:
            issues.append("Concept query but no concept_map tools used")

    # Quote queries should use transcript
    if any(w in query_lower for w in ["quotes", "what did", "said about", "statements"]):
        if "get_transcript" not in tools and "get_speaker_utterances" not in tools:
            issues.append("Quote query but no transcript tools used")

    return issues


def main():
    print(f"=" * 70)
    print(f"Agent V6 Comprehensive Evaluation")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Total queries: {len(QUERIES)}")
    print(f"=" * 70)

    results = []
    all_issues = []

    for cat_name, start, end in CATEGORIES:
        print(f"\n{'='*70}")
        print(f"Category: {cat_name} (Q{start+1}-Q{end})")
        print(f"{'='*70}")

        for i in range(start, end):
            query = QUERIES[i]
            query_num = i + 1

            print(f"\nQ{query_num}: {query[:60]}...")

            # Add delay between queries to avoid rate limits
            if i > start:
                time.sleep(2)

            response = run_query(query)
            issues = evaluate_response(query, response, query_num)

            result = {
                "query_num": query_num,
                "query": query,
                "category": cat_name,
                "success": response.get("success", False),
                "tools_used": response.get("tools_used", []),
                "answer_length": len(response.get("answer", "")),
                "issues": issues,
                "error": response.get("error"),
            }
            results.append(result)

            # Print result
            status = "✅" if not issues else "⚠️"
            print(f"  {status} Tools: {result['tools_used']}")
            print(f"  Answer length: {result['answer_length']} chars")
            if issues:
                for issue in issues:
                    print(f"  ⚠️ ISSUE: {issue}")
                all_issues.append((query_num, query, issues))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    success_count = sum(1 for r in results if r["success"] and not r["issues"])
    warning_count = sum(1 for r in results if r["success"] and r["issues"])
    error_count = sum(1 for r in results if not r["success"])

    print(f"✅ Successful (no issues): {success_count}/{len(QUERIES)}")
    print(f"⚠️ Successful (with issues): {warning_count}/{len(QUERIES)}")
    print(f"❌ Failed: {error_count}/{len(QUERIES)}")

    if all_issues:
        print(f"\n{'='*70}")
        print("ISSUES FOUND")
        print(f"{'='*70}")
        for query_num, query, issues in all_issues:
            print(f"\nQ{query_num}: {query[:50]}...")
            for issue in issues:
                print(f"  - {issue}")

    # Save results
    output_file = "/home/ubuntu/chemistry-dashboard/server/agent_v6/tests/evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
