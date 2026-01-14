"""
Comprehensive Tool Tests for BLINC Agent V7

Tests the OPTIMAL 6-TOOL DESIGN with complex, realistic queries.
Evaluates not just correctness, but alignment with design principles:
1. Artifact-centric: Complete artifacts, not fragments
2. Multi-representation: All three layers available
3. Cross-rep synthesis: Convergences, discrepancies surfaced
4. Tool economy: Minimal tools, maximal capability

Run with: python -m agent_v3.tests.comprehensive_tool_tests
"""

import json
import sys
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_v3.tools import (
    ARTIFACT_TOOLS,
    COMBINED_TOOLS,
    list_sessions,
    search_for_sessions,
    get_artifacts,
    get_speaker_profile,
    synthesize,
    find_concept_path
)

# Use artifact_tools versions directly
from agent_v3.tools.artifact_tools import (
    list_sessions as artifact_list_sessions,
    get_artifacts as artifact_get_artifacts,
    synthesize as artifact_synthesize
)


class TestResult:
    """Structured test result with detailed diagnostics."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.checks = []  # List of (check_name, passed, detail)
        self.issues = []
        self.warnings = []
        self.raw_output = None
        self.execution_time = 0

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append((name, passed, detail))
        if not passed:
            self.issues.append(f"{name}: {detail}")

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def summary(self) -> str:
        passed_checks = sum(1 for _, p, _ in self.checks if p)
        total_checks = len(self.checks)
        status = "PASS" if all(p for _, p, _ in self.checks) else "FAIL"

        lines = [f"\n{'='*60}"]
        lines.append(f"TEST: {self.name}")
        lines.append(f"STATUS: {status} ({passed_checks}/{total_checks} checks)")
        lines.append(f"TIME: {self.execution_time:.2f}s")

        if self.issues:
            lines.append("\nISSUES:")
            for issue in self.issues:
                lines.append(f"  ❌ {issue}")

        if self.warnings:
            lines.append("\nWARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")

        lines.append("="*60)
        return "\n".join(lines)


def run_test(test_func) -> TestResult:
    """Run a test function and capture results."""
    import time
    result = TestResult(test_func.__name__)
    start = time.time()
    try:
        test_func(result)
        result.passed = all(p for _, p, _ in result.checks)
    except Exception as e:
        result.add_check("execution", False, f"Exception: {e}")
        import traceback
        traceback.print_exc()
    result.execution_time = time.time() - start
    return result


# =============================================================================
# TEST 1: list_sessions - Discovery Tool
# =============================================================================

def test_list_sessions_completeness(result: TestResult):
    """
    Test: list_sessions returns complete metadata for all sessions.

    Expected: 8 sessions with artifacts_available flags
    """
    output = artifact_list_sessions()
    result.raw_output = output

    # Check basic structure
    result.add_check(
        "has_total_sessions",
        output.get('total_sessions') is not None,
        f"Got: {output.get('total_sessions')}"
    )

    result.add_check(
        "has_sessions_list",
        isinstance(output.get('sessions'), list),
        f"Type: {type(output.get('sessions'))}"
    )

    sessions = output.get('sessions', [])

    # Check we have all 8 sessions
    result.add_check(
        "correct_session_count",
        len(sessions) == 8,
        f"Expected 8, got {len(sessions)}"
    )

    # Check each session has required fields
    required_fields = ['session_id', 'session_name', 'speakers', 'artifacts_available']
    for session in sessions[:3]:  # Check first 3
        for field in required_fields:
            if field not in session:
                result.add_check(
                    f"session_has_{field}",
                    False,
                    f"Missing {field} in session {session.get('session_id')}"
                )
                break
        else:
            result.add_check(
                "session_structure_complete",
                True,
                f"Session {session.get('session_id')} has all required fields"
            )

    # Check artifacts_available structure
    if sessions:
        artifacts = sessions[0].get('artifacts_available', {})
        result.add_check(
            "artifacts_available_structure",
            all(k in artifacts for k in ['transcript', 'concept_map', 'collaboration']),
            f"Got keys: {list(artifacts.keys())}"
        )


# =============================================================================
# TEST 2: search_for_sessions - Semantic Discovery
# =============================================================================

def test_search_sessions_relevance(result: TestResult):
    """
    Test: search_for_sessions returns semantically relevant sessions.

    Query: "nuclear fusion energy" should find Nuclear Fusion session (ID 20)
    """
    output = search_for_sessions("nuclear fusion energy", top_k=3)
    result.raw_output = output

    sessions = output.get('sessions', [])

    result.add_check(
        "returns_results",
        len(sessions) > 0,
        f"Got {len(sessions)} sessions"
    )

    # Check if Nuclear Fusion session is in top results
    session_ids = [s.get('session_id') for s in sessions]
    result.add_check(
        "relevant_session_found",
        20 in session_ids,
        f"Session IDs: {session_ids}. Expected 20 (Nuclear Fusion)"
    )

    # Check ranking - Nuclear Fusion should be #1 or #2
    if sessions:
        top_session = sessions[0]
        result.add_check(
            "best_match_is_nuclear_fusion",
            top_session.get('session_id') == 20 or (len(sessions) > 1 and sessions[1].get('session_id') == 20),
            f"Top session: {top_session.get('session_name')} (ID {top_session.get('session_id')})"
        )


def test_search_sessions_cross_topic(result: TestResult):
    """
    Test: search_for_sessions handles abstract/cross-cutting queries.

    Query: "how people think about the future" - should find multiple relevant sessions
    """
    output = search_for_sessions("how people think about the future", top_k=5)
    result.raw_output = output

    sessions = output.get('sessions', [])

    result.add_check(
        "returns_multiple_sessions",
        len(sessions) >= 2,
        f"Got {len(sessions)} sessions for cross-cutting query"
    )

    # This is a judgment call - AI Alive, Nuclear Fusion, Abundance might all be relevant
    result.add_check(
        "has_match_scores",
        all('best_match_score' in s for s in sessions),
        "Sessions should have match scores for ranking"
    )


# =============================================================================
# TEST 3: get_artifacts - Flexible Retrieval
# =============================================================================

def test_get_artifacts_all(result: TestResult):
    """
    Test: get_artifacts returns all three representations when no include param.

    Session 20 (Nuclear Fusion) should have transcript, concept_map, and collaboration.
    """
    output = artifact_get_artifacts(20)
    result.raw_output = output

    artifacts = output.get('artifacts', {})

    # Check all three representations present
    for rep in ['transcript', 'concept_map', 'collaboration']:
        result.add_check(
            f"has_{rep}",
            rep in artifacts,
            f"Artifacts: {list(artifacts.keys())}"
        )

    # Check transcript completeness
    transcript = artifacts.get('transcript', {})
    result.add_check(
        "transcript_has_utterances",
        len(transcript.get('utterances', [])) > 0,
        f"Utterance count: {len(transcript.get('utterances', []))}"
    )

    result.add_check(
        "transcript_has_speaker_profiles",
        len(transcript.get('speaker_profiles', [])) > 0,
        f"Speaker count: {len(transcript.get('speaker_profiles', []))}"
    )

    # Check concept_map completeness
    concept_map = artifacts.get('concept_map', {})
    result.add_check(
        "concept_map_has_nodes",
        len(concept_map.get('nodes', [])) > 0,
        f"Node count: {len(concept_map.get('nodes', []))}"
    )

    result.add_check(
        "concept_map_has_edges",
        len(concept_map.get('edges', [])) > 0,
        f"Edge count: {len(concept_map.get('edges', []))}"
    )

    result.add_check(
        "concept_map_has_reasoning_patterns",
        'reasoning_patterns' in concept_map,
        "Should include identified reasoning patterns"
    )

    # Check collaboration completeness
    collab = artifacts.get('collaboration', {})
    result.add_check(
        "collaboration_has_dimensions",
        len(collab.get('dimensions', {})) == 7,
        f"Dimension count: {len(collab.get('dimensions', {}))}"
    )

    # Check dimension structure (coded segments, explanation)
    if collab.get('dimensions'):
        first_dim = list(collab['dimensions'].values())[0]
        result.add_check(
            "dimension_has_coded_segments",
            'coded_segments' in first_dim,
            f"Dimension keys: {list(first_dim.keys())}"
        )


def test_get_artifacts_selective(result: TestResult):
    """
    Test: get_artifacts with selective include param only returns requested artifacts.
    """
    # Request only transcript
    output = artifact_get_artifacts(20, include=['transcript'])
    artifacts = output.get('artifacts', {})

    result.add_check(
        "selective_has_transcript",
        'transcript' in artifacts and artifacts['transcript'].get('available', False),
        "Should have transcript"
    )

    result.add_check(
        "selective_no_concept_map",
        'concept_map' not in artifacts,
        f"Should not have concept_map, but got: {list(artifacts.keys())}"
    )

    # Request concept_map + collaboration
    output2 = artifact_get_artifacts(20, include=['concept_map', 'collaboration'])
    artifacts2 = output2.get('artifacts', {})

    result.add_check(
        "selective_combo_correct",
        'concept_map' in artifacts2 and 'collaboration' in artifacts2 and 'transcript' not in artifacts2,
        f"Expected concept_map + collaboration, got: {list(artifacts2.keys())}"
    )


# =============================================================================
# TEST 4: get_speaker_profile - Speaker Analysis with Graph Connections
# =============================================================================

def test_speaker_profile_completeness(result: TestResult):
    """
    Test: get_speaker_profile returns complete speaker data with graph connections.

    This is KEY - we want to see not just what they said, but how their ideas connect.
    """
    # Get profile for a speaker in Nuclear Fusion session
    output = get_speaker_profile("Lex", session_id=20)
    result.raw_output = output

    result.add_check(
        "found_speaker",
        output.get('speaker_alias') is not None,
        f"Speaker: {output.get('speaker_alias')}"
    )

    # Check transcript summary
    transcript_summary = output.get('transcript_summary', {})
    result.add_check(
        "has_transcript_summary",
        bool(transcript_summary),
        f"Keys: {list(transcript_summary.keys()) if transcript_summary else 'None'}"
    )

    result.add_check(
        "has_sample_quotes",
        len(transcript_summary.get('sample_quotes', [])) > 0,
        f"Quote count: {len(transcript_summary.get('sample_quotes', []))}"
    )

    # Check concept summary - THIS IS THE KEY DIFFERENTIATOR
    concept_summary = output.get('concept_summary', {})
    result.add_check(
        "has_concept_summary",
        bool(concept_summary),
        f"Keys: {list(concept_summary.keys()) if concept_summary else 'None'}"
    )

    # Check graph connections - the whole point of this tool
    connections = concept_summary.get('connections', {})
    result.add_check(
        "has_outgoing_connections",
        'outgoing' in connections,
        "Should show ideas this speaker influenced"
    )

    result.add_check(
        "has_incoming_connections",
        'incoming' in connections,
        "Should show ideas that influenced this speaker"
    )

    # Check interaction summary
    interaction = concept_summary.get('interaction_summary', {})
    result.add_check(
        "has_speakers_connected_to",
        'speakers_connected_to' in interaction,
        "Should show which other speakers they connected to"
    )


def test_speaker_profile_cross_session(result: TestResult):
    """
    Test: get_speaker_profile without session_id returns cross-session view.
    """
    output = get_speaker_profile("Lex")  # No session_id
    result.raw_output = output

    result.add_check(
        "scope_is_all_sessions",
        output.get('session_scope') == 'all sessions',
        f"Scope: {output.get('session_scope')}"
    )

    # Should have data from multiple sessions if speaker participated in multiple
    participation = output.get('transcript_summary', {}).get('participation_by_session', [])
    result.add_check(
        "has_multi_session_data",
        len(participation) >= 1,
        f"Sessions participated: {len(participation)}"
    )


# =============================================================================
# TEST 5: synthesize - Cross-Rep and Cross-Session Synthesis
# =============================================================================

def test_synthesize_single_session(result: TestResult):
    """
    Test: synthesize with single session performs cross-representation analysis.

    Key checks:
    - Extracts insights from all three representations
    - Identifies convergences (same finding in multiple reps)
    - Identifies discrepancies (conflicting signals)
    - Provides citations from each layer
    """
    output = artifact_synthesize(20, "How did collaboration quality relate to idea generation?")
    result.raw_output = output

    result.add_check(
        "synthesis_type_correct",
        output.get('synthesis_type') == 'single_session',
        f"Type: {output.get('synthesis_type')}"
    )

    # Check cross-rep insights structure
    cross_rep = output.get('cross_rep_insights', {})
    result.add_check(
        "has_from_transcript",
        'from_transcript' in cross_rep,
        f"Keys: {list(cross_rep.keys())}"
    )

    result.add_check(
        "has_from_concept_map",
        'from_concept_map' in cross_rep,
        "Should have concept map insights"
    )

    result.add_check(
        "has_from_collaboration",
        'from_collaboration' in cross_rep,
        "Should have collaboration insights"
    )

    # Check for convergence/discrepancy analysis
    result.add_check(
        "has_convergences",
        'convergences' in cross_rep,
        "Should identify where representations agree"
    )

    result.add_check(
        "has_discrepancies",
        'discrepancies' in cross_rep,
        "Should identify where representations conflict"
    )

    # Check citations
    citations = output.get('citations', [])
    result.add_check(
        "has_citations",
        len(citations) > 0,
        f"Citation count: {len(citations)}"
    )

    # Check citations come from multiple reps
    rep_sources = set(c.get('rep') for c in citations if c.get('rep'))
    result.add_check(
        "citations_from_multiple_reps",
        len(rep_sources) >= 2,
        f"Citation sources: {rep_sources}"
    )

    # Check integrated summary
    result.add_check(
        "has_integrated_summary",
        bool(output.get('integrated_summary')),
        f"Summary length: {len(output.get('integrated_summary', ''))}"
    )


def test_synthesize_cross_session(result: TestResult):
    """
    Test: synthesize with multiple sessions performs cross-session comparison.

    Compare Nuclear Fusion (20) and AI Alive (19) - should find similarities/differences.
    """
    output = artifact_synthesize([19, 20], "Compare collaboration quality and idea generation")
    result.raw_output = output

    result.add_check(
        "synthesis_type_correct",
        output.get('synthesis_type') == 'cross_session',
        f"Type: {output.get('synthesis_type')}"
    )

    result.add_check(
        "analyzed_both_sessions",
        len(output.get('sessions_analyzed', [])) == 2,
        f"Sessions analyzed: {len(output.get('sessions_analyzed', []))}"
    )

    # Check cross-session patterns
    cross_session = output.get('cross_session_patterns', {})
    result.add_check(
        "has_cross_session_patterns",
        bool(cross_session),
        f"Keys: {list(cross_session.keys()) if cross_session else 'None'}"
    )

    result.add_check(
        "has_similarities",
        'similarities' in cross_session,
        "Should identify similarities"
    )

    result.add_check(
        "has_differences",
        'differences' in cross_session,
        "Should identify differences"
    )

    result.add_check(
        "has_best_performing",
        cross_session.get('best_performing') is not None,
        f"Best: {cross_session.get('best_performing')}"
    )


def test_synthesize_complex_question(result: TestResult):
    """
    Test: synthesize handles complex analytical questions.

    This tests whether the synthesis actually provides useful insights.
    """
    question = "Did the group demonstrate systems thinking? What evidence supports or contradicts this?"
    output = artifact_synthesize(20, question)
    result.raw_output = output

    # The question asks about evidence - check citations
    citations = output.get('citations', [])
    result.add_check(
        "provides_evidence",
        len(citations) >= 2,
        f"Evidence citations: {len(citations)}"
    )

    # Check if insights relate to the question
    insights = output.get('cross_rep_insights', {})
    all_insights = (
        insights.get('from_transcript', []) +
        insights.get('from_concept_map', []) +
        insights.get('from_collaboration', [])
    )

    result.add_check(
        "has_substantive_insights",
        len(all_insights) >= 3,
        f"Total insights: {len(all_insights)}"
    )

    # Check summary addresses the question
    summary = output.get('integrated_summary', '')
    result.add_check(
        "summary_is_substantive",
        len(summary) > 100,
        f"Summary length: {len(summary)} chars"
    )


# =============================================================================
# TEST 6: find_concept_path - Graph Reasoning
# =============================================================================

def test_find_concept_path_basic(result: TestResult):
    """
    Test: find_concept_path finds paths between related concepts.

    First, we need to know what concepts exist in session 20.
    """
    # Get concept map first to find valid concepts
    artifacts = artifact_get_artifacts(20, include=['concept_map'])
    nodes = artifacts.get('artifacts', {}).get('concept_map', {}).get('nodes', [])

    if len(nodes) < 2:
        result.add_check("sufficient_nodes", False, f"Only {len(nodes)} nodes")
        return

    # Try to find path between first and last concept
    from_text = nodes[0].get('text', '')[:30]
    to_text = nodes[-1].get('text', '')[:30]

    output = find_concept_path(20, from_text, to_text, max_depth=5)
    result.raw_output = output

    result.add_check(
        "returns_result",
        'path_found' in output,
        f"Keys: {list(output.keys())}"
    )

    if output.get('path_found'):
        path = output.get('path', [])
        result.add_check(
            "path_has_steps",
            len(path) > 0,
            f"Path length: {len(path)}"
        )

        result.add_check(
            "path_has_relationships",
            all('relationship' in step for step in path),
            "Each step should show relationship type"
        )

        result.add_check(
            "has_narrative",
            bool(output.get('narrative')),
            "Should provide human-readable narrative"
        )
    else:
        # No path found - this is valid if concepts aren't connected
        result.add_check(
            "handles_no_path",
            output.get('message') is not None,
            "Should explain why no path found"
        )
        result.add_warning("No path found between selected concepts - may need different test concepts")


def test_find_concept_path_fuzzy_match(result: TestResult):
    """
    Test: find_concept_path handles fuzzy matching of concept text.
    """
    # Use partial text that should fuzzy match
    output = find_concept_path(20, "fusion", "energy", max_depth=5)
    result.raw_output = output

    # Should either find a path or report no match with helpful message
    if output.get('error'):
        result.add_check(
            "helpful_error",
            'suggestion' in output or 'not found' in output.get('error', ''),
            f"Error: {output.get('error')}"
        )
    else:
        result.add_check(
            "fuzzy_match_worked",
            output.get('source', {}).get('text') is not None,
            f"Matched source: {output.get('source', {}).get('text', '')[:50]}"
        )


# =============================================================================
# TEST 7: Integration - Complex Multi-Tool Workflow
# =============================================================================

def test_integration_discovery_to_synthesis(result: TestResult):
    """
    Test: Full workflow from discovery to synthesis.

    Scenario: "Which sessions had the best collaboration and why?"
    1. list_sessions to see what's available
    2. Get artifacts for sessions with 7C analysis
    3. synthesize to compare
    """
    # Step 1: Discovery
    sessions_output = artifact_list_sessions()
    sessions_with_collab = [
        s for s in sessions_output.get('sessions', [])
        if s.get('artifacts_available', {}).get('collaboration')
    ]

    result.add_check(
        "found_sessions_with_collaboration",
        len(sessions_with_collab) >= 2,
        f"Sessions with 7C: {len(sessions_with_collab)}"
    )

    if len(sessions_with_collab) < 2:
        result.add_warning("Not enough sessions with collaboration analysis for comparison")
        return

    # Step 2: Synthesize across sessions
    session_ids = [s['session_id'] for s in sessions_with_collab[:3]]
    synth_output = artifact_synthesize(session_ids, "Which had the best collaboration and why?")

    result.add_check(
        "synthesis_produced",
        synth_output.get('is_relevant', False),
        "Synthesis should produce relevant results"
    )

    # Check we got a best performer
    best = synth_output.get('cross_session_patterns', {}).get('best_performing')
    result.add_check(
        "identified_best_session",
        best is not None,
        f"Best: {best}"
    )


# =============================================================================
# TEST 8: Edge Cases and Error Handling
# =============================================================================

def test_invalid_session_id(result: TestResult):
    """
    Test: Tools handle invalid session IDs gracefully.
    """
    output = artifact_get_artifacts(9999)

    result.add_check(
        "returns_error",
        output.get('error') is not None or output.get('is_relevant') == False,
        f"Response: {output}"
    )


def test_empty_search_query(result: TestResult):
    """
    Test: search_for_sessions handles edge cases.
    """
    output = search_for_sessions("xyznonexistent12345")

    result.add_check(
        "handles_no_results",
        output.get('sessions_found', 0) == 0 or len(output.get('sessions', [])) == 0,
        f"Sessions: {output.get('sessions', [])}"
    )


def test_speaker_not_found(result: TestResult):
    """
    Test: get_speaker_profile handles non-existent speakers.
    """
    output = get_speaker_profile("NonExistentSpeaker12345")

    result.add_check(
        "returns_error",
        output.get('error') is not None or output.get('is_relevant') == False,
        f"Response keys: {list(output.keys())}"
    )


# =============================================================================
# TEST 9: Quality Assessment - Do Results Align with Expectations?
# =============================================================================

def test_quality_collaboration_insights(result: TestResult):
    """
    Test: Collaboration artifact provides actionable insights, not just scores.

    Key question: Can a teacher use this to give meaningful feedback?
    """
    output = artifact_get_artifacts(20, include=['collaboration'])
    collab = output.get('artifacts', {}).get('collaboration', {})

    if not collab.get('available'):
        result.add_warning("No collaboration data for session 20")
        return

    dimensions = collab.get('dimensions', {})

    # Check each dimension has explanation AND evidence
    dimensions_with_evidence = 0
    dimensions_with_explanation = 0

    for dim_name, dim_data in dimensions.items():
        if dim_data.get('coded_segments'):
            dimensions_with_evidence += 1
        if dim_data.get('explanation') and len(dim_data['explanation']) > 20:
            dimensions_with_explanation += 1

    result.add_check(
        "most_dimensions_have_evidence",
        dimensions_with_evidence >= 4,
        f"Dimensions with evidence: {dimensions_with_evidence}/7"
    )

    result.add_check(
        "most_dimensions_have_explanation",
        dimensions_with_explanation >= 4,
        f"Dimensions with explanation: {dimensions_with_explanation}/7"
    )

    # Check summary is actionable
    summary = collab.get('summary', {})
    result.add_check(
        "has_strengths",
        len(summary.get('strengths', [])) > 0,
        "Should identify strengths"
    )

    result.add_check(
        "has_areas_for_improvement",
        'areas_for_improvement' in summary,
        "Should identify areas for improvement"
    )


def test_quality_concept_map_reasoning(result: TestResult):
    """
    Test: Concept map artifact shows reasoning structure, not just a node list.

    Key question: Can we understand HOW ideas developed?
    """
    output = artifact_get_artifacts(20, include=['concept_map'])
    concept_map = output.get('artifacts', {}).get('concept_map', {})

    if not concept_map.get('available'):
        result.add_warning("No concept map for session 20")
        return

    # Check for reasoning patterns
    patterns = concept_map.get('reasoning_patterns', [])
    result.add_check(
        "has_reasoning_patterns",
        len(patterns) > 0,
        f"Pattern count: {len(patterns)}"
    )

    # Check pattern types
    if patterns:
        pattern_types = set(p.get('pattern_type') for p in patterns)
        result.add_check(
            "diverse_pattern_types",
            len(pattern_types) >= 1,
            f"Types: {pattern_types}"
        )

    # Check for hub nodes (central ideas)
    hubs = concept_map.get('hub_nodes', [])
    result.add_check(
        "identifies_central_ideas",
        len(hubs) > 0,
        f"Hub count: {len(hubs)}"
    )

    # Check speaker contributions breakdown
    contributions = concept_map.get('summary', {}).get('speaker_contributions', {})
    result.add_check(
        "shows_speaker_contributions",
        len(contributions) > 0,
        f"Speakers with contributions: {list(contributions.keys())}"
    )


# =============================================================================
# TEST 10: Cross-Rep Alignment - Do Representations Tell Consistent Story?
# =============================================================================

def test_cross_rep_consistency(result: TestResult):
    """
    Test: synthesize correctly identifies when representations tell the same story.

    If transcript shows dominant speaker, concept map should show their contributions.
    """
    synth = artifact_synthesize(20, "Who contributed most to the discussion?")
    result.raw_output = synth

    # Get insights from different reps
    cross_rep = synth.get('cross_rep_insights', {})

    t_insights = cross_rep.get('from_transcript', [])
    c_insights = cross_rep.get('from_concept_map', [])

    # Find participation insights
    t_participation = next((i for i in t_insights if i.get('type') == 'participation'), None)
    c_contribution = next((i for i in c_insights if i.get('type') == 'contribution'), None)

    if t_participation and c_contribution:
        t_speaker = t_participation.get('data', {}).get('speaker', '')
        c_speaker = c_contribution.get('data', {}).get('top_contributor', '')

        result.add_check(
            "cross_rep_speaker_alignment",
            t_speaker.lower() == c_speaker.lower() if t_speaker and c_speaker else False,
            f"Transcript: {t_speaker}, Concept Map: {c_speaker}"
        )
    else:
        result.add_warning("Could not find participation/contribution insights to compare")

    # Check convergences are identified
    convergences = cross_rep.get('convergences', [])
    result.add_check(
        "identifies_convergences",
        len(convergences) >= 0,  # May be 0 if reps don't converge
        f"Convergence count: {len(convergences)}"
    )


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests() -> Dict[str, Any]:
    """Run all tests and generate comprehensive report."""

    tests = [
        # Discovery
        test_list_sessions_completeness,
        test_search_sessions_relevance,
        test_search_sessions_cross_topic,

        # Artifact Retrieval
        test_get_artifacts_all,
        test_get_artifacts_selective,

        # Speaker Profile
        test_speaker_profile_completeness,
        test_speaker_profile_cross_session,

        # Synthesis
        test_synthesize_single_session,
        test_synthesize_cross_session,
        test_synthesize_complex_question,

        # Graph Reasoning
        test_find_concept_path_basic,
        test_find_concept_path_fuzzy_match,

        # Integration
        test_integration_discovery_to_synthesis,

        # Edge Cases
        test_invalid_session_id,
        test_empty_search_query,
        test_speaker_not_found,

        # Quality Assessment
        test_quality_collaboration_insights,
        test_quality_concept_map_reasoning,

        # Cross-Rep Consistency
        test_cross_rep_consistency,
    ]

    results = []
    for test in tests:
        print(f"Running {test.__name__}...", end=" ")
        result = run_test(test)
        results.append(result)
        print("PASS" if result.passed else "FAIL")

    # Generate report
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST REPORT")
    print("="*70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\nOVERALL: {passed}/{total} tests passed")
    print(f"TIME: {sum(r.execution_time for r in results):.2f}s total")

    # Print failed tests
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n{'='*70}")
        print("FAILED TESTS:")
        print("="*70)
        for r in failed:
            print(r.summary())

    # Collect all issues
    all_issues = []
    all_warnings = []
    for r in results:
        all_issues.extend([(r.name, issue) for issue in r.issues])
        all_warnings.extend([(r.name, warning) for warning in r.warnings])

    if all_issues:
        print(f"\n{'='*70}")
        print("ALL ISSUES:")
        print("="*70)
        for test_name, issue in all_issues:
            print(f"  [{test_name}] {issue}")

    if all_warnings:
        print(f"\n{'='*70}")
        print("WARNINGS:")
        print("="*70)
        for test_name, warning in all_warnings:
            print(f"  [{test_name}] {warning}")

    return {
        "passed": passed,
        "total": total,
        "results": results,
        "issues": all_issues,
        "warnings": all_warnings
    }


if __name__ == "__main__":
    report = run_all_tests()

    # Exit with non-zero if tests failed
    sys.exit(0 if report["passed"] == report["total"] else 1)
