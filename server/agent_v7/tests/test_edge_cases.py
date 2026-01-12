#!/usr/bin/env python3
"""
Deep Edge Case Tests for Agent V3

Probes corner cases, semantic alignment, and potential failure modes.
"""

import json
import time
import requests
from datetime import datetime

BASE_URL = "http://localhost:5000"
LOGIN_CREDS = {"email": "llmblinc", "password": "blinc25"}


class EdgeCaseTester:
    def __init__(self):
        self.session = requests.Session()
        self.results = []
        self.issues = []
        self.session.post(f"{BASE_URL}/api/v1/login", json=LOGIN_CREDS)
        print("✓ Logged in")

    def query(self, q: str, conv_id: str = None, timeout: int = 90) -> dict:
        if conv_id is None:
            conv_id = f"edge-{int(time.time())}"
        start = time.time()
        try:
            resp = self.session.post(
                f"{BASE_URL}/api/v3/agent/query",
                json={"query": q, "conversation_id": conv_id},
                timeout=timeout
            )
            data = resp.json()
        except Exception as e:
            data = {"error": str(e), "success": False}
        data['_ms'] = (time.time() - start) * 1000
        return data

    def test(self, name: str, query: str, expected: dict = None):
        expected = expected or {}
        print(f"\n{'='*50}\n{name}\n{'='*50}")
        print(f"Query: {query}")

        resp = self.query(query)
        issues = []

        answer = resp.get('answer', '')
        citations = resp.get('citations', [])
        success = resp.get('success', False)

        if not success:
            issues.append(f"Request failed: {resp.get('error', 'unknown')}")

        # Expected content checks
        if expected.get('must_contain'):
            for phrase in expected['must_contain']:
                if phrase.lower() not in answer.lower():
                    issues.append(f"Missing: '{phrase}'")

        if expected.get('must_not_contain'):
            for phrase in expected['must_not_contain']:
                if phrase.lower() in answer.lower():
                    issues.append(f"Should not contain: '{phrase}'")

        if expected.get('min_citations', 0) > len(citations):
            issues.append(f"Too few citations: {len(citations)}")

        if expected.get('expect_no_answer'):
            if len(answer) > 200 and 'no information' not in answer.lower() and 'sorry' not in answer.lower():
                issues.append("Expected no/limited answer but got substantial response")

        passed = len(issues) == 0
        status = "✓" if passed else "✗"

        print(f"{status} {resp.get('_ms', 0):.0f}ms | {len(citations)} citations")
        print(f"Answer: {answer[:120]}..." if len(answer) > 120 else f"Answer: {answer}")

        if issues:
            for i in issues:
                print(f"  ⚠ {i}")
                self.issues.append({"test": name, "issue": i})

        self.results.append({
            "name": name,
            "passed": passed,
            "ms": resp.get('_ms', 0),
            "issues": issues,
            "answer_len": len(answer),
            "citations": len(citations)
        })
        return resp

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('passed'))
        print(f"\n{'='*50}")
        print(f"EDGE CASES: {passed}/{total} passed ({100*passed/total:.0f}%)")
        print(f"{'='*50}")
        if self.issues:
            print(f"\nIssues ({len(self.issues)}):")
            for i in self.issues:
                print(f"  [{i['test']}] {i['issue']}")


def main():
    t = EdgeCaseTester()

    # === SEMANTIC ALIGNMENT TESTS ===
    print("\n" + "="*60 + "\nSEMANTIC ALIGNMENT TESTS\n" + "="*60)

    # Test: Answer should mention the right session
    t.test("Correct session reference",
           "What were the main points of the Abundance session?",
           {"must_contain": ["abundance", "session 25"]})

    # Test: Speaker attribution accuracy
    t.test("Speaker attribution",
           "What did Lex contribute to the dinosaurs discussion?",
           {"must_contain": ["lex"]})

    # Test: Should not hallucinate
    t.test("No hallucination on missing topic",
           "What did they say about cryptocurrency?",
           {"expect_no_answer": True})

    # === TRICKY QUERY PATTERNS ===
    print("\n" + "="*60 + "\nTRICKY QUERY TESTS\n" + "="*60)

    # Superlative queries
    t.test("Superlative - best collaboration",
           "Which session had the best collaboration?",
           {"min_citations": 1})

    t.test("Superlative - most active speaker",
           "Who was the most active speaker across all sessions?",
           {"min_citations": 0})

    # Negation queries
    t.test("Negation query",
           "What didn't they discuss in the AI session?",
           {"min_citations": 0})

    # Compound queries
    t.test("Compound query",
           "What did David and Lex discuss about energy, and how did they collaborate?",
           {"min_citations": 1})

    # Temporal queries
    t.test("Temporal - first topic",
           "What was the first thing discussed in session 20?",
           {"min_citations": 1})

    # === CONTEXT PRESERVATION ===
    print("\n" + "="*60 + "\nCONTEXT PRESERVATION TESTS\n" + "="*60)

    conv = f"ctx-{int(time.time())}"

    t.test("Context Turn 1 - Establish",
           "Tell me about the Country Music session")

    t.test("Context Turn 2 - Pronoun",
           "What topics did they cover in it?")

    t.test("Context Turn 3 - Follow-up",
           "Were there any interesting conclusions?")

    # === CITATION ACCURACY ===
    print("\n" + "="*60 + "\nCITATION ACCURACY TESTS\n" + "="*60)

    # Citation should match claim
    resp = t.test("Citation matches claim",
                  "Find quotes about nuclear reactions from session 20",
                  {"min_citations": 1})

    if resp.get('citations'):
        for c in resp['citations']:
            ref = c.get('artifactRef', {})
            if ref.get('sessionId') != 20:
                print(f"  ⚠ Citation from wrong session: {ref.get('sessionId')}")

    # === ROBUSTNESS TESTS ===
    print("\n" + "="*60 + "\nROBUSTNESS TESTS\n" + "="*60)

    # Empty/minimal queries
    t.test("Minimal - one word", "Fusion")
    t.test("Minimal - question mark only", "?")

    # Typos
    t.test("Typo in session name",
           "Tell me about the Nucular Fushion session")

    # Mixed case
    t.test("Mixed case",
           "WHAT WAS SESSION 20 ABOUT?")

    # === SPEAKER TOOL TESTS ===
    print("\n" + "="*60 + "\nSPEAKER PROFILE TESTS\n" + "="*60)

    t.test("Speaker LIWC metrics",
           "What is Tucker's level of analytic thinking?",
           {"must_contain": ["tucker"]})

    t.test("Speaker contribution types",
           "What types of contributions did David make?",
           {"must_contain": ["david"]})

    t.test("Speaker comparison",
           "How do Lex and David differ in their speaking styles?",
           {"must_contain": ["lex", "david"]})

    # Summary
    t.summary()

    # Save detailed report
    with open("/home/ubuntu/chemistry-dashboard/server/agent_v3/tests/EDGE_CASE_REPORT.json", 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": t.results,
            "issues": t.issues,
            "summary": {
                "total": len(t.results),
                "passed": sum(1 for r in t.results if r.get('passed')),
                "failed": sum(1 for r in t.results if not r.get('passed'))
            }
        }, f, indent=2)

    print("\nReport saved to EDGE_CASE_REPORT.json")


if __name__ == "__main__":
    main()
