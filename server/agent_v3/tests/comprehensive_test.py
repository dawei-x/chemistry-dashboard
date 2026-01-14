#!/usr/bin/env python3
"""
Comprehensive Test Suite for BLINC Agent V3

Tests cover:
1. Fast path routing accuracy
2. Complex reasoning capabilities
3. Session name resolution
4. Multi-turn conversation context
5. Edge cases and error handling
6. Tool selection accuracy
7. Response quality evaluation
"""

import json
import requests
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

BASE_URL = "http://localhost:5000"
COOKIES_FILE = "/tmp/cookies.txt"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    category: str
    passed: bool
    query: str
    expected: str
    actual: Dict[str, Any]
    issues: List[str] = field(default_factory=list)
    notes: str = ""


class AgentTester:
    """Comprehensive tester for Agent V3."""

    def __init__(self):
        self.session = requests.Session()
        self.results: List[TestResult] = []
        self.conversation_counter = 0

    def login(self) -> bool:
        """Login to get session cookie."""
        response = self.session.post(
            f"{BASE_URL}/api/v1/login",
            json={"email": "llmblinc", "password": "blinc25"}
        )
        return response.status_code == 200

    def query(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Send a query to Agent V3."""
        if not conversation_id:
            self.conversation_counter += 1
            conversation_id = f"test-{self.conversation_counter}-{int(time.time())}"

        response = self.session.post(
            f"{BASE_URL}/api/v3/agent/query",
            json={"query": query, "conversation_id": conversation_id},
            timeout=120
        )
        return response.json()

    def add_result(self, result: TestResult):
        """Add a test result."""
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.category}: {result.name}")
        if result.issues:
            for issue in result.issues:
                print(f"       Issue: {issue}")

    # =========================================================================
    # TEST CATEGORY 1: FAST PATH ROUTING
    # =========================================================================

    def test_fast_path_list_sessions(self):
        """Test: 'What sessions are available?' should use fast path."""
        query = "What sessions are available?"
        result = self.query(query)

        issues = []
        # Check if list_sessions was used
        tools_used = result.get('tools_used', [])
        if 'list_sessions' not in tools_used:
            issues.append(f"Expected list_sessions tool, got: {tools_used}")

        # Check if it returned all 8 sessions
        answer = result.get('answer', '')
        session_names = ['Living in NYC', 'AI Alive', 'Nuclear Fusion', 'Shaw Interview',
                        'Collaboration Literacy', 'Dinosaurs', 'Country Music', 'Abundance']
        found_sessions = sum(1 for name in session_names if name.lower() in answer.lower())
        if found_sessions < 5:
            issues.append(f"Only found {found_sessions}/8 sessions mentioned in answer")

        self.add_result(TestResult(
            name="List sessions query",
            category="Fast Path",
            passed=len(issues) == 0,
            query=query,
            expected="list_sessions tool, all 8 sessions mentioned",
            actual=result,
            issues=issues
        ))

    def test_fast_path_session_overview(self):
        """Test: 'Tell me about session 20' should use fast path."""
        query = "Tell me about session 20"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])
        if 'get_session_overview' not in tools_used:
            issues.append(f"Expected get_session_overview tool, got: {tools_used}")

        answer = result.get('answer', '')
        if 'nuclear fusion' not in answer.lower() and 'fusion' not in answer.lower():
            issues.append("Session 20 is Nuclear Fusion but topic not mentioned")

        self.add_result(TestResult(
            name="Session overview by ID",
            category="Fast Path",
            passed=len(issues) == 0,
            query=query,
            expected="get_session_overview tool, Nuclear Fusion topic",
            actual=result,
            issues=issues
        ))

    def test_fast_path_collaboration_score(self):
        """Test: 'What's the collaboration score for session 21?' should use fast path."""
        query = "What's the collaboration score for session 21?"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])
        if 'get_collaboration_analysis' not in tools_used:
            issues.append(f"Expected get_collaboration_analysis tool, got: {tools_used}")

        answer = result.get('answer', '')
        # Should contain a numeric score
        import re
        if not re.search(r'\d+\.?\d*', answer):
            issues.append("No numeric score found in answer")

        self.add_result(TestResult(
            name="Collaboration score query",
            category="Fast Path",
            passed=len(issues) == 0,
            query=query,
            expected="get_collaboration_analysis tool, numeric score",
            actual=result,
            issues=issues
        ))

    # =========================================================================
    # TEST CATEGORY 2: COMPLEX REASONING
    # =========================================================================

    def test_complex_best_collaboration(self):
        """Test: 'Which session had the best collaboration?' requires comparison."""
        query = "Which session had the best collaboration?"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])

        # Should use compare_sessions, not search
        if 'compare_sessions' not in tools_used:
            issues.append(f"Expected compare_sessions tool, got: {tools_used}")
        if 'search_transcripts' in tools_used:
            issues.append("Should not use search_transcripts for score comparison")

        answer = result.get('answer', '')
        # Session 20 (Nuclear Fusion) has best score (79.0)
        if '20' not in answer and 'nuclear fusion' not in answer.lower():
            issues.append("Did not identify Session 20/Nuclear Fusion as best")

        # Should mention a score
        import re
        if not re.search(r'7\d\.\d|8\d\.\d', answer):
            issues.append("No specific collaboration score mentioned")

        self.add_result(TestResult(
            name="Best collaboration query",
            category="Complex Reasoning",
            passed=len(issues) == 0,
            query=query,
            expected="compare_sessions, identify Session 20, mention score ~79",
            actual=result,
            issues=issues
        ))

    def test_complex_compare_two_sessions(self):
        """Test: Explicit comparison between two sessions."""
        query = "Compare the Nuclear Fusion and Shaw Interview sessions"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])

        if 'compare_sessions' not in tools_used and 'get_session_overview' not in tools_used:
            issues.append(f"Expected compare_sessions or get_session_overview, got: {tools_used}")

        answer = result.get('answer', '')
        if 'fusion' not in answer.lower():
            issues.append("Nuclear Fusion session not discussed")
        if 'shaw' not in answer.lower():
            issues.append("Shaw Interview session not discussed")

        self.add_result(TestResult(
            name="Compare two sessions",
            category="Complex Reasoning",
            passed=len(issues) == 0,
            query=query,
            expected="Both sessions discussed with comparison",
            actual=result,
            issues=issues
        ))

    def test_complex_speaker_across_sessions(self):
        """Test: Analysis requiring cross-session speaker lookup."""
        query = "What did Lex discuss across different sessions?"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])

        # Should use search or analyze_speaker
        search_tools = ['search_transcripts', 'analyze_speaker']
        if not any(t in tools_used for t in search_tools):
            issues.append(f"Expected search_transcripts or analyze_speaker, got: {tools_used}")

        answer = result.get('answer', '')
        if 'lex' not in answer.lower():
            issues.append("Lex not mentioned in answer")

        self.add_result(TestResult(
            name="Cross-session speaker analysis",
            category="Complex Reasoning",
            passed=len(issues) == 0,
            query=query,
            expected="Search or analyze Lex's contributions",
            actual=result,
            issues=issues
        ))

    # =========================================================================
    # TEST CATEGORY 3: SESSION NAME RESOLUTION
    # =========================================================================

    def test_resolution_exact_name(self):
        """Test: Exact session name should resolve correctly."""
        query = "Tell me about the Nuclear Fusion session"
        result = self.query(query)

        issues = []
        answer = result.get('answer', '')

        if 'session 20' not in answer.lower() and '20' not in str(result.get('citations', [])):
            issues.append("Session 20 not identified from 'Nuclear Fusion' name")

        if 'fusion' not in answer.lower():
            issues.append("Fusion topic not discussed")

        self.add_result(TestResult(
            name="Exact session name resolution",
            category="Session Resolution",
            passed=len(issues) == 0,
            query=query,
            expected="Resolve 'Nuclear Fusion' to session 20",
            actual=result,
            issues=issues
        ))

    def test_resolution_partial_name(self):
        """Test: Partial session name should resolve correctly."""
        query = "What happened in the Shaw session?"
        result = self.query(query)

        issues = []
        answer = result.get('answer', '')

        # Shaw Interview is session 21
        if 'shaw' not in answer.lower() and 'interview' not in answer.lower():
            issues.append("Shaw Interview session not recognized")

        self.add_result(TestResult(
            name="Partial session name resolution",
            category="Session Resolution",
            passed=len(issues) == 0,
            query=query,
            expected="Resolve 'Shaw' to Shaw Interview session",
            actual=result,
            issues=issues
        ))

    def test_resolution_topic_based(self):
        """Test: Topic-based session lookup."""
        query = "Which session discussed dinosaurs?"
        result = self.query(query)

        issues = []
        answer = result.get('answer', '')

        if 'dinosaur' not in answer.lower():
            issues.append("Dinosaurs session not found")

        self.add_result(TestResult(
            name="Topic-based session lookup",
            category="Session Resolution",
            passed=len(issues) == 0,
            query=query,
            expected="Find Dinosaurs session",
            actual=result,
            issues=issues
        ))

    # =========================================================================
    # TEST CATEGORY 4: MULTI-TURN CONVERSATION
    # =========================================================================

    def test_multiturn_context_retention(self):
        """Test: Context should be retained across turns."""
        conv_id = f"multiturn-{int(time.time())}"

        # Turn 1: Establish context
        result1 = self.query("Tell me about the Nuclear Fusion session", conv_id)

        # Turn 2: Reference "it"
        result2 = self.query("Who were the speakers in it?", conv_id)

        issues = []
        answer2 = result2.get('answer', '')

        # Session 20 speakers are David and Lex
        if 'david' not in answer2.lower() and 'lex' not in answer2.lower():
            issues.append("Did not identify speakers from context (David, Lex)")

        if result2.get('needs_clarification', False):
            issues.append("Asked for clarification instead of using context")

        self.add_result(TestResult(
            name="Context retention across turns",
            category="Multi-turn",
            passed=len(issues) == 0,
            query="Turn 1: Nuclear Fusion, Turn 2: 'Who were the speakers in it?'",
            expected="Retain session 20 context, answer about David/Lex",
            actual=result2,
            issues=issues
        ))

    def test_multiturn_topic_continuity(self):
        """Test: Topic continuity in follow-up questions."""
        conv_id = f"topic-{int(time.time())}"

        # Turn 1: Ask about collaboration
        result1 = self.query("How well did they collaborate in session 20?", conv_id)

        # Turn 2: Ask about specific dimension
        result2 = self.query("What about their communication specifically?", conv_id)

        issues = []
        answer2 = result2.get('answer', '')

        if 'communication' not in answer2.lower():
            issues.append("Did not address communication dimension")

        self.add_result(TestResult(
            name="Topic continuity",
            category="Multi-turn",
            passed=len(issues) == 0,
            query="Turn 1: Collaboration in 20, Turn 2: 'What about communication?'",
            expected="Continue discussing session 20's communication score",
            actual=result2,
            issues=issues
        ))

    # =========================================================================
    # TEST CATEGORY 5: EDGE CASES
    # =========================================================================

    def test_edge_nonexistent_session(self):
        """Test: Query about non-existent session."""
        query = "Tell me about session 999"
        result = self.query(query)

        issues = []
        answer = result.get('answer', '')

        # Should indicate session not found or provide helpful info
        if result.get('error'):
            pass  # Error is acceptable
        elif 'not found' not in answer.lower() and 'not exist' not in answer.lower() and 'available' not in answer.lower():
            issues.append("Did not indicate session 999 doesn't exist")

        self.add_result(TestResult(
            name="Non-existent session query",
            category="Edge Cases",
            passed=len(issues) == 0,
            query=query,
            expected="Indicate session not found or list available sessions",
            actual=result,
            issues=issues
        ))

    def test_edge_ambiguous_query(self):
        """Test: Highly ambiguous query handling."""
        query = "What about that?"
        result = self.query(query)  # No context

        issues = []

        # Should either clarify or attempt a general search
        if not result.get('needs_clarification') and not result.get('answer'):
            issues.append("Neither clarified nor attempted to answer")

        self.add_result(TestResult(
            name="Ambiguous query handling",
            category="Edge Cases",
            passed=len(issues) == 0,
            query=query,
            expected="Ask for clarification or make best attempt",
            actual=result,
            issues=issues
        ))

    def test_edge_typo_in_session_name(self):
        """Test: Typo tolerance in session names."""
        query = "Tell me about the Nucler Fuson session"  # Typos
        result = self.query(query)

        issues = []
        answer = result.get('answer', '')

        # Should still find Nuclear Fusion
        if 'fusion' not in answer.lower() and 'nuclear' not in answer.lower():
            issues.append("Could not resolve typo 'Nucler Fuson' to Nuclear Fusion")

        self.add_result(TestResult(
            name="Typo tolerance",
            category="Edge Cases",
            passed=len(issues) == 0,
            query=query,
            expected="Resolve to Nuclear Fusion despite typos",
            actual=result,
            issues=issues,
            notes="May fail - typo tolerance not explicitly implemented"
        ))

    def test_edge_mixed_case_query(self):
        """Test: Mixed case handling."""
        query = "WHAT IS THE COLLABORATION SCORE FOR SESSION 20?"
        result = self.query(query)

        issues = []
        answer = result.get('answer', '')

        import re
        if not re.search(r'\d+\.?\d*', answer):
            issues.append("No numeric score found despite valid query")

        self.add_result(TestResult(
            name="Mixed case handling",
            category="Edge Cases",
            passed=len(issues) == 0,
            query=query,
            expected="Handle all-caps query correctly",
            actual=result,
            issues=issues
        ))

    # =========================================================================
    # TEST CATEGORY 6: TOOL SELECTION ACCURACY
    # =========================================================================

    def test_tool_search_vs_measurement(self):
        """Test: Distinguish between search and measurement queries."""
        # Measurement query - should NOT search
        query = "What's the collaboration score?"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])

        # Should use analysis tools, not search
        if 'search_transcripts' in tools_used:
            issues.append("Used search_transcripts for measurement query")

        self.add_result(TestResult(
            name="Search vs measurement distinction",
            category="Tool Selection",
            passed=len(issues) == 0,
            query=query,
            expected="Use measurement tool, not search",
            actual=result,
            issues=issues
        ))

    def test_tool_graph_for_connections(self):
        """Test: Use graph tools for connection queries."""
        query = "How are ideas connected in session 20?"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])

        graph_tools = ['explore_concepts', 'get_concept_map', 'find_reasoning_path']
        if not any(t in tools_used for t in graph_tools):
            issues.append(f"Expected graph tools, got: {tools_used}")

        self.add_result(TestResult(
            name="Graph tools for connections",
            category="Tool Selection",
            passed=len(issues) == 0,
            query=query,
            expected="Use explore_concepts or get_concept_map",
            actual=result,
            issues=issues
        ))

    def test_tool_quote_retrieval(self):
        """Test: Use search_transcripts for quote retrieval."""
        query = "What did Tucker say about physics?"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])

        if 'search_transcripts' not in tools_used:
            issues.append(f"Expected search_transcripts for quote query, got: {tools_used}")

        self.add_result(TestResult(
            name="Transcript search for quotes",
            category="Tool Selection",
            passed=len(issues) == 0,
            query=query,
            expected="Use search_transcripts with speaker filter",
            actual=result,
            issues=issues
        ))

    # =========================================================================
    # TEST CATEGORY 7: TRICKY QUERIES
    # =========================================================================

    def test_tricky_negation(self):
        """Test: Query with negation."""
        query = "Which sessions did NOT have good collaboration?"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])

        # Should still compare sessions
        if 'compare_sessions' not in tools_used and 'get_collaboration_analysis' not in tools_used:
            issues.append(f"Should analyze collaboration for negation query, got: {tools_used}")

        answer = result.get('answer', '')
        # Should mention lower-scoring sessions
        if not any(word in answer.lower() for word in ['low', 'poor', 'worst', 'less', 'lacking']):
            issues.append("Did not identify sessions with poor collaboration")

        self.add_result(TestResult(
            name="Negation handling",
            category="Tricky Queries",
            passed=len(issues) == 0,
            query=query,
            expected="Identify sessions with low collaboration scores",
            actual=result,
            issues=issues
        ))

    def test_tricky_superlative_count(self):
        """Test: Query asking for top N."""
        query = "What are the top 3 most collaborative sessions?"
        result = self.query(query)

        issues = []
        answer = result.get('answer', '')

        # Should list multiple sessions
        import re
        session_refs = re.findall(r'session\s*\d+|session\s*(?:20|21|22|23|24|25|18|19)', answer.lower())
        if len(session_refs) < 3:
            issues.append(f"Expected 3 sessions, found references: {session_refs}")

        self.add_result(TestResult(
            name="Top N query",
            category="Tricky Queries",
            passed=len(issues) == 0,
            query=query,
            expected="List top 3 sessions by collaboration score",
            actual=result,
            issues=issues
        ))

    def test_tricky_temporal_reference(self):
        """Test: Query with temporal reference."""
        query = "What was discussed at the beginning of the Nuclear Fusion session?"
        result = self.query(query)

        issues = []
        tools_used = result.get('tools_used', [])

        # Should search transcripts
        if 'search_transcripts' not in tools_used and 'get_session_overview' not in tools_used:
            issues.append(f"Expected transcript search for temporal query, got: {tools_used}")

        self.add_result(TestResult(
            name="Temporal reference",
            category="Tricky Queries",
            passed=len(issues) == 0,
            query=query,
            expected="Search transcripts with temporal awareness",
            actual=result,
            issues=issues
        ))

    def test_tricky_compound_query(self):
        """Test: Compound query with multiple requirements."""
        query = "Which session had the best collaboration AND discussed AI?"
        result = self.query(query)

        issues = []
        answer = result.get('answer', '')

        # Should find intersection of high collaboration + AI topic
        # Session 19 "Is AI Alive" discusses AI
        if 'ai' not in answer.lower() and 'artificial intelligence' not in answer.lower():
            issues.append("Did not address AI topic requirement")

        if 'collaboration' not in answer.lower() and 'score' not in answer.lower():
            issues.append("Did not address collaboration requirement")

        self.add_result(TestResult(
            name="Compound query (AND)",
            category="Tricky Queries",
            passed=len(issues) == 0,
            query=query,
            expected="Find session matching both criteria",
            actual=result,
            issues=issues
        ))

    def test_tricky_hypothetical(self):
        """Test: Hypothetical/counterfactual query."""
        query = "If the Nuclear Fusion session had more speakers, would it be more collaborative?"
        result = self.query(query)

        issues = []
        answer = result.get('answer', '')

        # Should either answer thoughtfully or indicate speculation
        if not answer:
            issues.append("No answer provided for hypothetical query")

        # Accept if it acknowledges this is speculative or provides analysis
        self.add_result(TestResult(
            name="Hypothetical query",
            category="Tricky Queries",
            passed=len(issues) == 0,
            query=query,
            expected="Thoughtful response or acknowledgment of speculation",
            actual=result,
            issues=issues,
            notes="Subjective - may vary"
        ))

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================

    def run_all_tests(self):
        """Run all tests and generate report."""
        print("=" * 60)
        print("BLINC Agent V3 Comprehensive Test Suite")
        print("=" * 60)

        if not self.login():
            print("FATAL: Could not login")
            return

        print("\nRunning tests...\n")

        # Fast Path Tests
        print("\n--- FAST PATH TESTS ---")
        self.test_fast_path_list_sessions()
        self.test_fast_path_session_overview()
        self.test_fast_path_collaboration_score()

        # Complex Reasoning Tests
        print("\n--- COMPLEX REASONING TESTS ---")
        self.test_complex_best_collaboration()
        self.test_complex_compare_two_sessions()
        self.test_complex_speaker_across_sessions()

        # Session Resolution Tests
        print("\n--- SESSION RESOLUTION TESTS ---")
        self.test_resolution_exact_name()
        self.test_resolution_partial_name()
        self.test_resolution_topic_based()

        # Multi-turn Tests
        print("\n--- MULTI-TURN TESTS ---")
        self.test_multiturn_context_retention()
        self.test_multiturn_topic_continuity()

        # Edge Cases
        print("\n--- EDGE CASE TESTS ---")
        self.test_edge_nonexistent_session()
        self.test_edge_ambiguous_query()
        self.test_edge_typo_in_session_name()
        self.test_edge_mixed_case_query()

        # Tool Selection Tests
        print("\n--- TOOL SELECTION TESTS ---")
        self.test_tool_search_vs_measurement()
        self.test_tool_graph_for_connections()
        self.test_tool_quote_retrieval()

        # Tricky Queries
        print("\n--- TRICKY QUERY TESTS ---")
        self.test_tricky_negation()
        self.test_tricky_superlative_count()
        self.test_tricky_temporal_reference()
        self.test_tricky_compound_query()
        self.test_tricky_hypothetical()

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate final test report."""
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {"passed": 0, "failed": 0, "issues": []}
            if result.passed:
                categories[result.category]["passed"] += 1
            else:
                categories[result.category]["failed"] += 1
                categories[result.category]["issues"].append(result)

        total_passed = sum(c["passed"] for c in categories.values())
        total_failed = sum(c["failed"] for c in categories.values())
        total = total_passed + total_failed

        print(f"\nOverall: {total_passed}/{total} passed ({100*total_passed/total:.1f}%)\n")

        for cat, stats in categories.items():
            status = "PASS" if stats["failed"] == 0 else "ISSUES"
            print(f"  {cat}: {stats['passed']}/{stats['passed']+stats['failed']} [{status}]")

        # List all failures
        if total_failed > 0:
            print("\n" + "-" * 60)
            print("FAILED TESTS DETAIL")
            print("-" * 60)

            for result in self.results:
                if not result.passed:
                    print(f"\n[FAIL] {result.category}: {result.name}")
                    print(f"  Query: {result.query[:80]}...")
                    print(f"  Expected: {result.expected}")
                    for issue in result.issues:
                        print(f"  Issue: {issue}")
                    if result.notes:
                        print(f"  Note: {result.notes}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    tester = AgentTester()
    tester.run_all_tests()
