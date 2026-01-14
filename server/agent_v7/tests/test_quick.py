#!/usr/bin/env python3
"""
Quick Agent V7 Tests - Focused subset for faster iteration.
"""

import json
import time
import requests
from typing import Dict, Any, List
from datetime import datetime

BASE_URL = "http://localhost:5000"
LOGIN_CREDS = {"email": "llmblinc", "password": "blinc25"}


class QuickTester:
    def __init__(self):
        self.session = requests.Session()
        self.results = []
        self.issues = []
        resp = self.session.post(f"{BASE_URL}/api/v1/login", json=LOGIN_CREDS)
        print("✓ Logged in")

    def query(self, q: str, conv_id: str = None) -> Dict:
        if conv_id is None:
            conv_id = f"test-{int(time.time())}"
        start = time.time()
        resp = self.session.post(
            f"{BASE_URL}/api/v3/agent/query",
            json={"query": q, "conversation_id": conv_id},
            timeout=90
        )
        elapsed = (time.time() - start) * 1000
        data = resp.json()
        data['_ms'] = elapsed
        return data

    def test(self, name: str, query: str, checks: Dict = None):
        """Run test with optional checks."""
        checks = checks or {}
        print(f"\n{'='*50}\nTEST: {name}\nQuery: {query}\n{'='*50}")

        try:
            resp = self.query(query)
        except Exception as e:
            print(f"✗ FAILED: {e}")
            self.results.append({"name": name, "passed": False, "error": str(e)})
            return

        issues = []
        answer = resp.get('answer', '')
        citations = resp.get('citations', [])
        cite_types = list(set(c.get('citationType', '?') for c in citations))

        # Checks
        if checks.get('min_citations', 0) > len(citations):
            issues.append(f"Expected {checks['min_citations']}+ citations, got {len(citations)}")

        if checks.get('expected_types'):
            for t in checks['expected_types']:
                if t not in cite_types:
                    issues.append(f"Missing citation type: {t}")

        if checks.get('must_contain'):
            for phrase in checks['must_contain']:
                if phrase.lower() not in answer.lower():
                    issues.append(f"Missing phrase: '{phrase}'")

        # Validate citation structure
        for i, c in enumerate(citations):
            if not c.get('artifactRef'):
                issues.append(f"Citation {i} missing artifactRef")
            if not c.get('preview', {}).get('content'):
                issues.append(f"Citation {i} has empty preview")

        passed = len(issues) == 0
        status = "✓ PASSED" if passed else "✗ FAILED"

        print(f"\nResult: {status} ({resp.get('_ms', 0):.0f}ms)")
        print(f"Citations: {len(citations)} ({', '.join(cite_types) or 'none'})")
        print(f"Answer: {answer[:150]}...")

        if issues:
            print(f"\nIssues:")
            for issue in issues:
                print(f"  - {issue}")
                self.issues.append({"test": name, "issue": issue})

        self.results.append({
            "name": name,
            "passed": passed,
            "ms": resp.get('_ms', 0),
            "citations": len(citations),
            "types": cite_types,
            "issues": issues
        })

        return resp

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('passed'))
        print(f"\n{'='*50}")
        print(f"SUMMARY: {passed}/{total} passed ({100*passed/total:.0f}%)")
        print(f"{'='*50}")
        if self.issues:
            print(f"\nAll issues ({len(self.issues)}):")
            for i in self.issues:
                print(f"  [{i['test']}] {i['issue']}")


def main():
    t = QuickTester()

    # === CITATION TESTS ===
    print("\n" + "="*60 + "\nCITATION TYPE TESTS\n" + "="*60)

    # 1. Session citations
    t.test("Session - List", "List sessions",
           {"min_citations": 1, "expected_types": ["session"]})

    # 2. Session overview
    t.test("Session - Overview", "What was session 20 about?",
           {"min_citations": 1, "expected_types": ["session"]})

    # 3. Transcript citations
    t.test("Transcript - Quote Search",
           "What did speakers say about fusion energy in session 20?",
           {"min_citations": 1, "expected_types": ["transcript"]})

    # 4. Speaker analysis
    t.test("Speaker - Analysis", "Tell me about Tucker's discussion style",
           {"min_citations": 1})

    # 5. 7C collaboration
    t.test("7C - Collaboration", "How was the collaboration quality in session 19?",
           {"min_citations": 1})

    # === SEMANTIC ALIGNMENT ===
    print("\n" + "="*60 + "\nSEMANTIC ALIGNMENT TESTS\n" + "="*60)

    # Check correct session attribution
    resp = t.test("Semantic - Session Attribution",
                  "What was discussed in the Dinosaurs session?",
                  {"min_citations": 1})

    if resp:
        answer = resp.get('answer', '').lower()
        citations = resp.get('citations', [])
        # Check if dinosaurs session (ID 23) is properly cited
        for c in citations:
            ref = c.get('artifactRef', {})
            sid = ref.get('sessionId')
            if sid and sid != 23:
                print(f"  ⚠ Citation references session {sid}, expected 23 (Dinosaurs)")

    # === EDGE CASES ===
    print("\n" + "="*60 + "\nEDGE CASE TESTS\n" + "="*60)

    # Short query
    t.test("Edge - Short Query", "Sessions", {"min_citations": 0})

    # Non-existent session
    t.test("Edge - Invalid Session", "Tell me about session 999", {"min_citations": 0})

    # Multi-turn context
    conv = f"multi-{int(time.time())}"
    t.test("Multi-Turn 1", "Tell me about the Nuclear Fusion session",
           {"min_citations": 1})

    # === SPEAKER PROFILE TOOLS ===
    print("\n" + "="*60 + "\nSPEAKER PROFILE TESTS\n" + "="*60)

    t.test("Speaker - Comprehensive Profile",
           "What is David's communication style?",
           {"min_citations": 1, "must_contain": ["David"]})

    # Summary
    t.summary()

    # Save report
    with open("/home/ubuntu/chemistry-dashboard/server/agent_v3/tests/QUICK_TEST_REPORT.json", 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": t.results,
            "issues": t.issues
        }, f, indent=2)

    print(f"\nReport saved to QUICK_TEST_REPORT.json")


if __name__ == "__main__":
    main()
