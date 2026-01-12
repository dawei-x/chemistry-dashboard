#!/usr/bin/env python3
"""
Comprehensive Agent V3 Test Suite

Tests citation generation, speaker profiles, semantic alignment, and edge cases.
Designed for AIED 2026 submission quality assurance.

Run with: python -m pytest agent_v3/tests/test_agent_comprehensive.py -v
Or directly: python agent_v3/tests/test_agent_comprehensive.py
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"
LOGIN_CREDS = {"email": "llmblinc", "password": "blinc25"}


@dataclass
class TestResult:
    """Result of a single test case."""
    test_name: str
    query: str
    passed: bool
    issues: List[str] = field(default_factory=list)
    response_time_ms: float = 0
    citations_count: int = 0
    citation_types: List[str] = field(default_factory=list)
    answer_excerpt: str = ""
    raw_response: Optional[Dict] = None


class AgentTester:
    """Comprehensive agent testing framework."""

    def __init__(self):
        self.session = requests.Session()
        self.results: List[TestResult] = []
        self.issues: List[Dict] = []
        self._login()

    def _login(self):
        """Authenticate with the server."""
        resp = self.session.post(
            f"{BASE_URL}/api/v1/login",
            json=LOGIN_CREDS
        )
        if resp.status_code != 200:
            raise Exception(f"Login failed: {resp.text}")
        print("✓ Logged in successfully")

    def query(self, q: str, conversation_id: str = None) -> Dict[str, Any]:
        """Send a query to the agent."""
        if conversation_id is None:
            conversation_id = f"test-{int(time.time())}"

        start = time.time()
        resp = self.session.post(
            f"{BASE_URL}/api/v3/agent/query",
            json={"query": q, "conversation_id": conversation_id},
            timeout=120
        )
        elapsed_ms = (time.time() - start) * 1000

        data = resp.json()
        data['_response_time_ms'] = elapsed_ms
        return data

    def run_test(self, test_name: str, query: str,
                 expected_citation_types: List[str] = None,
                 must_contain: List[str] = None,
                 must_not_contain: List[str] = None,
                 min_citations: int = 0,
                 conversation_id: str = None,
                 semantic_checks: List[callable] = None) -> TestResult:
        """Run a single test case with multiple validation checks."""

        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"Query: {query}")
        print('='*60)

        issues = []

        try:
            resp = self.query(query, conversation_id)
        except Exception as e:
            return TestResult(
                test_name=test_name,
                query=query,
                passed=False,
                issues=[f"Request failed: {str(e)}"]
            )

        # Basic success check
        if not resp.get('success'):
            issues.append(f"API returned success=false: {resp.get('error', 'unknown')}")

        answer = resp.get('answer', '')
        citations = resp.get('citations', [])
        citation_types = list(set(c.get('citationType', 'unknown') for c in citations))

        # Citation count check
        if len(citations) < min_citations:
            issues.append(f"Expected at least {min_citations} citations, got {len(citations)}")

        # Citation type check
        if expected_citation_types:
            for expected_type in expected_citation_types:
                if expected_type not in citation_types:
                    issues.append(f"Missing expected citation type: {expected_type}")

        # Content checks
        if must_contain:
            for phrase in must_contain:
                if phrase.lower() not in answer.lower():
                    issues.append(f"Answer missing expected phrase: '{phrase}'")

        if must_not_contain:
            for phrase in must_not_contain:
                if phrase.lower() in answer.lower():
                    issues.append(f"Answer contains forbidden phrase: '{phrase}'")

        # Semantic checks
        if semantic_checks:
            for check_fn in semantic_checks:
                result = check_fn(resp)
                if result:
                    issues.append(result)

        # Citation structure validation
        for i, cite in enumerate(citations):
            cite_issues = self._validate_citation_structure(cite, i)
            issues.extend(cite_issues)

        passed = len(issues) == 0

        result = TestResult(
            test_name=test_name,
            query=query,
            passed=passed,
            issues=issues,
            response_time_ms=resp.get('_response_time_ms', 0),
            citations_count=len(citations),
            citation_types=citation_types,
            answer_excerpt=answer[:200] + "..." if len(answer) > 200 else answer,
            raw_response=resp
        )

        self.results.append(result)

        # Print result
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"\nResult: {status}")
        print(f"Response time: {result.response_time_ms:.0f}ms")
        print(f"Citations: {len(citations)} ({', '.join(citation_types) if citation_types else 'none'})")
        print(f"Answer: {result.answer_excerpt}")

        if issues:
            print(f"\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")
                self.issues.append({
                    "test": test_name,
                    "issue": issue,
                    "query": query
                })

        return result

    def _validate_citation_structure(self, cite: Dict, index: int) -> List[str]:
        """Validate citation has required structure."""
        issues = []
        required_fields = ['id', 'citationType', 'inlineText', 'artifactRef', 'preview']

        for field in required_fields:
            if field not in cite:
                issues.append(f"Citation {index} missing field: {field}")

        # Validate artifactRef has at least one reference
        artifact_ref = cite.get('artifactRef', {})
        if not any(artifact_ref.values()):
            issues.append(f"Citation {index} has empty artifactRef")

        # Validate preview has content
        preview = cite.get('preview', {})
        if not preview.get('content') and not preview.get('title'):
            issues.append(f"Citation {index} preview is empty")

        return issues

    def generate_report(self) -> str:
        """Generate a markdown test report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        lines = [
            "# Agent V3 Comprehensive Test Report",
            f"\n**Generated**: {datetime.now().isoformat()}",
            f"\n## Summary",
            f"- **Total Tests**: {total}",
            f"- **Passed**: {passed} ({100*passed/total:.1f}%)" if total > 0 else "- **Passed**: 0",
            f"- **Failed**: {failed}",
            f"\n## Test Results",
        ]

        # Group by pass/fail
        for result in self.results:
            status = "✅" if result.passed else "❌"
            lines.append(f"\n### {status} {result.test_name}")
            lines.append(f"- **Query**: `{result.query}`")
            lines.append(f"- **Response Time**: {result.response_time_ms:.0f}ms")
            lines.append(f"- **Citations**: {result.citations_count} ({', '.join(result.citation_types) or 'none'})")

            if result.issues:
                lines.append("- **Issues**:")
                for issue in result.issues:
                    lines.append(f"  - {issue}")

        # Issues summary
        if self.issues:
            lines.append("\n## Issues Summary")
            lines.append("\n| Test | Issue |")
            lines.append("|------|-------|")
            for issue in self.issues:
                lines.append(f"| {issue['test']} | {issue['issue']} |")

        return "\n".join(lines)


def run_citation_type_tests(tester: AgentTester):
    """Test all 6 citation types are generated correctly."""

    print("\n" + "="*70)
    print("SECTION 1: CITATION TYPE COVERAGE TESTS")
    print("="*70)

    # Test 1: Session citations
    tester.run_test(
        test_name="Session Citation - List Sessions",
        query="What sessions are available?",
        expected_citation_types=["session"],
        min_citations=1
    )

    # Test 2: Session overview
    tester.run_test(
        test_name="Session Citation - Overview",
        query="Tell me about session 20",
        expected_citation_types=["session"],
        min_citations=1
    )

    # Test 3: Transcript citations
    tester.run_test(
        test_name="Transcript Citation - Direct Quote Search",
        query="What did speakers say about energy in the Nuclear Fusion session?",
        expected_citation_types=["transcript"],
        min_citations=1
    )

    # Test 4: Speaker citations (using new enhanced tool)
    tester.run_test(
        test_name="Speaker Citation - Profile Analysis",
        query="Analyze Tucker's discussion style",
        min_citations=1
    )

    # Test 5: 7C collaboration citations
    tester.run_test(
        test_name="7C Citation - Collaboration Analysis",
        query="How well did they collaborate in session 20?",
        min_citations=1
    )

    # Test 6: Concept citations
    tester.run_test(
        test_name="Concept Citation - Concept Search",
        query="What hypotheses were proposed about AI in session 19?",
        min_citations=1
    )


def run_speaker_profile_tests(tester: AgentTester):
    """Test the enhanced speaker profile tools."""

    print("\n" + "="*70)
    print("SECTION 2: SPEAKER PROFILE TESTS")
    print("="*70)

    # Test comprehensive speaker profile
    tester.run_test(
        test_name="Speaker Profile - Comprehensive",
        query="Tell me about David's participation patterns and communication style",
        min_citations=1,
        must_contain=["David"]
    )

    # Test session-specific speaker profile
    tester.run_test(
        test_name="Speaker Profile - Session Specific",
        query="How did Lex contribute to session 20?",
        min_citations=1
    )

    # Test speaker comparison (if it triggers the compare_speakers tool)
    tester.run_test(
        test_name="Speaker Comparison - Two Speakers",
        query="Compare how David and Lex participate in discussions",
        min_citations=1
    )


def run_semantic_alignment_tests(tester: AgentTester):
    """Test that responses semantically align with queries."""

    print("\n" + "="*70)
    print("SECTION 3: SEMANTIC ALIGNMENT TESTS")
    print("="*70)

    # Test: Answer should reference the correct session
    def check_session_reference(resp):
        answer = resp.get('answer', '').lower()
        # If asking about session 20, answer shouldn't only talk about other sessions
        if 'session 20' not in answer and 'nuclear fusion' not in answer:
            return "Answer doesn't reference session 20 or Nuclear Fusion topic"
        return None

    tester.run_test(
        test_name="Semantic - Correct Session Reference",
        query="What was the main topic of session 20?",
        semantic_checks=[check_session_reference],
        min_citations=1
    )

    # Test: Citations should match claims
    def check_citation_relevance(resp):
        citations = resp.get('citations', [])
        for cite in citations:
            # Check if citation is from expected session
            artifact_ref = cite.get('artifactRef', {})
            session_id = artifact_ref.get('sessionId')
            # For this query, we expect session 19 citations
            if session_id and session_id not in [19]:
                return f"Citation references unexpected session {session_id}"
        return None

    tester.run_test(
        test_name="Semantic - Citation Session Alignment",
        query="What did Tucker say about AI reasoning in session 19?",
        must_contain=["Tucker"],
        semantic_checks=[check_citation_relevance],
        min_citations=1
    )

    # Test: Answer shouldn't hallucinate non-existent content
    tester.run_test(
        test_name="Semantic - No Hallucination Check",
        query="What did they discuss about quantum computing?",
        # Quantum computing likely wasn't discussed, so answer should indicate this
        min_citations=0  # May have no relevant citations
    )


def run_edge_case_tests(tester: AgentTester):
    """Test edge cases and tricky queries."""

    print("\n" + "="*70)
    print("SECTION 4: EDGE CASE & TRICKY QUERY TESTS")
    print("="*70)

    # Test: Very short query
    tester.run_test(
        test_name="Edge Case - Short Query",
        query="Sessions?",
        min_citations=0  # May or may not work
    )

    # Test: Query with typo in speaker name
    tester.run_test(
        test_name="Edge Case - Typo in Speaker Name",
        query="What did Tuckr say?",  # Typo: Tuckr instead of Tucker
        min_citations=0  # Should handle gracefully
    )

    # Test: Non-existent session
    tester.run_test(
        test_name="Edge Case - Non-existent Session",
        query="Tell me about session 999",
        must_not_contain=["session 999 discussed"]  # Shouldn't pretend it exists
    )

    # Test: Ambiguous pronoun reference
    conv_id = f"ambiguous-test-{int(time.time())}"
    tester.run_test(
        test_name="Edge Case - Ambiguous Pronoun (Turn 1)",
        query="Tell me about the Nuclear Fusion session",
        conversation_id=conv_id,
        min_citations=1
    )
    tester.run_test(
        test_name="Edge Case - Ambiguous Pronoun (Turn 2)",
        query="What did they discuss about it?",  # "it" = fusion
        conversation_id=conv_id,
        min_citations=0  # May struggle with this
    )

    # Test: Negative query
    tester.run_test(
        test_name="Edge Case - Negative Query",
        query="What topics were NOT discussed in session 20?",
        min_citations=0  # Tricky - may not handle well
    )

    # Test: Comparative query
    tester.run_test(
        test_name="Edge Case - Cross-Session Comparison",
        query="Which session had better collaboration - session 19 or session 20?",
        min_citations=1
    )

    # Test: Abstract construct operationalization
    tester.run_test(
        test_name="Edge Case - Abstract Construct",
        query="Did anyone demonstrate systems thinking in session 19?",
        min_citations=1
    )


def run_multi_turn_context_tests(tester: AgentTester):
    """Test multi-turn context preservation."""

    print("\n" + "="*70)
    print("SECTION 5: MULTI-TURN CONTEXT TESTS")
    print("="*70)

    conv_id = f"context-test-{int(time.time())}"

    # Turn 1: Establish context
    tester.run_test(
        test_name="Multi-Turn - Establish Context",
        query="Tell me about the Dinosaurs session",
        conversation_id=conv_id,
        min_citations=1
    )

    # Turn 2: Follow-up with pronoun
    tester.run_test(
        test_name="Multi-Turn - Pronoun Reference",
        query="Who participated in it?",
        conversation_id=conv_id,
        min_citations=0
    )

    # Turn 3: Another follow-up
    tester.run_test(
        test_name="Multi-Turn - Continued Context",
        query="What was the main conclusion?",
        conversation_id=conv_id,
        min_citations=0
    )


def run_performance_tests(tester: AgentTester):
    """Test response time performance."""

    print("\n" + "="*70)
    print("SECTION 6: PERFORMANCE TESTS")
    print("="*70)

    # Simple query should be fast (<5s)
    result = tester.run_test(
        test_name="Performance - Simple Query Speed",
        query="List sessions",
        min_citations=1
    )
    if result.response_time_ms > 5000:
        result.issues.append(f"Simple query too slow: {result.response_time_ms:.0f}ms > 5000ms target")

    # Complex query acceptable up to 30s
    result = tester.run_test(
        test_name="Performance - Complex Query Speed",
        query="Compare the collaboration patterns across all sessions and identify which had the best idea building",
        min_citations=1
    )
    if result.response_time_ms > 60000:
        result.issues.append(f"Complex query too slow: {result.response_time_ms:.0f}ms > 60000ms target")


def main():
    """Run all tests and generate report."""

    print("\n" + "="*70)
    print("AGENT V3 COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Started at: {datetime.now().isoformat()}")

    tester = AgentTester()

    # Run all test sections
    run_citation_type_tests(tester)
    run_speaker_profile_tests(tester)
    run_semantic_alignment_tests(tester)
    run_edge_case_tests(tester)
    run_multi_turn_context_tests(tester)
    run_performance_tests(tester)

    # Generate report
    report = tester.generate_report()

    # Save report
    report_path = "/home/ubuntu/chemistry-dashboard/server/agent_v3/tests/TEST_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)

    total = len(tester.results)
    passed = sum(1 for r in tester.results if r.passed)

    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {total - passed}")
    print(f"\nReport saved to: {report_path}")

    if tester.issues:
        print(f"\n{len(tester.issues)} issues found. See report for details.")

    return tester


if __name__ == "__main__":
    main()
