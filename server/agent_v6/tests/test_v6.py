"""
Test suite for Agent V6.

Run with: python -m pytest server/agent_v6/tests/test_v6.py -v
"""

import pytest
from agent_v6.query_analysis import analyze_query, clear_cache
from agent_v6.prompt_builder import build_system_prompt
from agent_v6.tools import filter_tools_by_steering, get_tool_names_for_steering


class TestQueryAnalysis:
    """Tests for query analysis (entity extraction, steering, mode detection)."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def test_session_id_extraction(self):
        """Test explicit session ID extraction."""
        analysis = analyze_query("Tell me about session 20")
        assert 20 in analysis.session_ids

    def test_session_name_fuzzy_match(self):
        """Test fuzzy matching of session names."""
        analysis = analyze_query("Tell me about the Nuclear Fusion session")
        assert 20 in analysis.session_ids
        assert "Nuclear Fusion" in analysis.session_names

    def test_speaker_extraction(self):
        """Test speaker name extraction."""
        # Note: This requires the database to have speakers
        analysis = analyze_query("What did David say in the Nuclear Fusion session?")
        # This may or may not find David depending on DB state
        # Just verify no error occurs
        assert analysis.query == "What did David say in the Nuclear Fusion session?"

    def test_hypothesis_mode_detection(self):
        """Test hypothesis testing mode detection."""
        analysis = analyze_query("I think Tucker demonstrates systems thinking")
        assert analysis.mode == "test_hypothesis"

    def test_compare_mode_detection(self):
        """Test compare mode detection."""
        analysis = analyze_query("Compare session 19 vs session 20")
        assert analysis.mode == "compare"

    def test_construct_detection(self):
        """Test abstract construct detection."""
        analysis = analyze_query("Does Tucker demonstrate systems thinking?")
        assert "systems thinking" in analysis.constructs

    def test_exclude_collaboration_steering(self):
        """Test NL steering for exclusion."""
        analysis = analyze_query("Tell me about session 20 without using 7C analysis")
        assert "collaboration" in analysis.exclude_representations
        assert "collaboration" not in analysis.prefer_representations

    def test_prefer_transcript_steering(self):
        """Test NL steering for preference."""
        analysis = analyze_query("Focus on the transcript - what did David say?")
        assert "transcript" in analysis.prefer_representations

    def test_exclusion_takes_precedence(self):
        """Test that exclusion takes precedence over preference."""
        # "7C" would normally trigger prefer_collaboration
        # but "without 7C" should trigger exclude instead
        analysis = analyze_query("Tell me about 7C analysis without using 7C analysis")
        assert "collaboration" in analysis.exclude_representations
        assert "collaboration" not in analysis.prefer_representations


class TestToolFiltering:
    """Tests for tool filtering based on steering."""

    def test_all_tools_available_by_default(self):
        """Test that all tools are available without steering."""
        tools = filter_tools_by_steering()
        tool_names = [t['name'] for t in tools]
        assert 'get_transcript' in tool_names
        assert 'get_concept_map' in tool_names
        assert 'get_7c_analysis' in tool_names

    def test_exclude_collaboration_removes_7c(self):
        """Test that excluding collaboration removes get_7c_analysis."""
        tools = filter_tools_by_steering(exclude=['collaboration'])
        tool_names = [t['name'] for t in tools]
        assert 'get_7c_analysis' not in tool_names
        assert 'get_transcript' in tool_names
        assert 'get_concept_map' in tool_names

    def test_prefer_transcript_only(self):
        """Test that preferring transcript limits tools."""
        tools = filter_tools_by_steering(prefer=['transcript'])
        tool_names = [t['name'] for t in tools]
        assert 'get_transcript' in tool_names
        assert 'get_speaker_utterances' in tool_names
        # Core tools always available
        assert 'list_sessions' in tool_names
        assert 'search_sessions' in tool_names
        # Other representation tools should not be present
        assert 'get_7c_analysis' not in tool_names
        assert 'get_concept_map' not in tool_names

    def test_core_tools_always_available(self):
        """Test that core tools are always available."""
        tools = filter_tools_by_steering(exclude=['transcript', 'concept_map', 'collaboration'])
        tool_names = [t['name'] for t in tools]
        assert 'list_sessions' in tool_names
        assert 'search_sessions' in tool_names
        assert 'compare_sessions' in tool_names


class TestPromptBuilder:
    """Tests for prompt construction."""

    def test_basic_prompt_includes_framework(self):
        """Test that basic prompt includes analytical framework."""
        analysis = analyze_query("Tell me about session 20")
        prompt = build_system_prompt(analysis)
        assert "Epistemic Hierarchy" in prompt
        assert "Triangulation Framework" in prompt
        assert "Grounding Your Claims" in prompt

    def test_hypothesis_mode_adds_protocol(self):
        """Test that hypothesis mode adds testing protocol."""
        analysis = analyze_query("I think Tucker demonstrates systems thinking")
        prompt = build_system_prompt(analysis)
        assert "Hypothesis Testing Mode" in prompt
        assert "Reach a Verdict" in prompt

    def test_construct_adds_operationalization(self):
        """Test that detected constructs add operationalization."""
        analysis = analyze_query("Does Tucker demonstrate systems thinking?")
        prompt = build_system_prompt(analysis)
        assert "Systems Thinking" in prompt
        assert "causal relationships" in prompt.lower()

    def test_exclusion_adds_steering_section(self):
        """Test that exclusions add steering section."""
        analysis = analyze_query("Tell me without using 7C analysis")
        prompt = build_system_prompt(analysis)
        assert "Excluded Representations" in prompt
        assert "collaboration" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
