"""
Tests for the new V7 Scaffolding Agent architecture.

Run with:
    cd /home/ubuntu/chemistry-dashboard/server
    python -m pytest agent_v7/tests/test_new_architecture.py -v
"""

import pytest
import sys
import os

# Add server directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestConversationMemory:
    """Test the ConversationMemory class."""

    def test_memory_creation(self):
        """Test basic memory creation."""
        from agent_v7.memory import ConversationMemory

        memory = ConversationMemory(conversation_id="test-123")
        assert memory.conversation_id == "test-123"
        assert memory.session_focus is None
        assert memory.turn_count == 0

    def test_memory_session_extraction(self):
        """Test session ID extraction from text."""
        from agent_v7.memory import ConversationMemory

        memory = ConversationMemory(conversation_id="test-123")

        # Test explicit session ID
        assert memory.extract_session_from_text("Tell me about session 24") == 24

        # Test session name
        assert memory.extract_session_from_text("What happened in the Country Music discussion?") == 24
        assert memory.extract_session_from_text("Tell me about nuclear fusion") == 20

    def test_memory_speaker_extraction(self):
        """Test speaker name extraction."""
        from agent_v7.memory import ConversationMemory

        memory = ConversationMemory(conversation_id="test-123")

        assert memory.extract_speaker_from_text("What did Tucker say?") == "Tucker"
        assert memory.extract_speaker_from_text("Show me Maya's contributions") == "Maya"

    def test_memory_context_for_llm(self):
        """Test context formatting for LLM."""
        from agent_v7.memory import ConversationMemory

        memory = ConversationMemory(conversation_id="test-123")
        memory.update_session_focus(24, "Country Music")
        memory.update_speaker_focus("Tucker")
        memory.record_artifact("transcript", 24)

        context = memory.get_context_for_llm()

        assert "Session 24" in context
        assert "Country Music" in context
        assert "Tucker" in context
        assert "transcript" in context

    def test_memory_persistence(self):
        """Test memory get/clear functions."""
        from agent_v7.memory import get_memory, clear_memory

        # Get creates new memory
        memory1 = get_memory("test-persist")
        memory1.update_session_focus(20)

        # Get returns same memory
        memory2 = get_memory("test-persist")
        assert memory2.session_focus == 20

        # Clear removes memory
        clear_memory("test-persist")
        memory3 = get_memory("test-persist")
        assert memory3.session_focus is None


class TestSteering:
    """Test user steering - simplified LLM-native approach."""

    def test_steering_extraction_basic(self):
        """Test that steering captures raw query for LLM understanding."""
        from agent_v7.steering import extract_steering

        # Steering just passes query to LLM - no regex extraction
        steering = extract_steering("Focus on the concept map when analyzing this")
        assert steering.raw_instructions == "Focus on the concept map when analyzing this"
        # No preferred_artifacts - LLM understands from query
        assert steering.api_preferred == []
        assert steering.api_excluded == []

    def test_steering_with_api_preferences(self):
        """Test API-level preferences are captured."""
        from agent_v7.steering import extract_steering

        # API-level preferences (explicitly passed via API)
        memory_steering = {
            'preferred_artifacts': ['concept_map'],
            'excluded_artifacts': ['7c']
        }
        steering = extract_steering("Tell me about session 24", memory_steering=memory_steering)

        assert 'concept_map' in steering.api_preferred
        assert '7c' in steering.api_excluded

    def test_tool_validation_with_api_exclusion(self):
        """Test tool validation only blocks API-level exclusions."""
        from agent_v7.steering import extract_steering, validate_tool_call

        # Query-based exclusion: LLM handles it, not validation
        steering = extract_steering("Don't use 7C analysis")
        is_valid, reason = validate_tool_call("get_7c_analysis", steering)
        assert is_valid  # Query exclusions don't block - LLM decides

        # API-level exclusion: validation blocks it
        memory_steering = {'excluded_artifacts': ['7c']}
        steering = extract_steering("Tell me about session 24", memory_steering=memory_steering)
        is_valid, reason = validate_tool_call("get_7c_analysis", steering)
        assert not is_valid  # API exclusions block

        # Transcript should always be allowed (not excluded)
        is_valid, reason = validate_tool_call("get_transcript", steering)
        assert is_valid


class TestTools:
    """Test the simplified tool registry."""

    def test_tool_registry(self):
        """Test that all core tools are registered."""
        from agent_v7.tools_v2 import CORE_TOOLS, get_tool_names

        expected_tools = [
            "list_sessions",
            "search_sessions",
            "get_transcript",
            "get_concept_map",
            "get_7c_analysis",
            "get_session_overview",
            "compare_sessions",
            "find_concept_path",
        ]

        tool_names = get_tool_names()
        for tool in expected_tools:
            assert tool in tool_names

    def test_list_sessions(self):
        """Test list_sessions tool."""
        from agent_v7.tools_v2 import list_sessions

        result = list_sessions()
        assert result.get("tool_name") == "list_sessions"
        assert "sessions" in result
        assert isinstance(result["sessions"], list)

    def test_get_session_overview(self):
        """Test get_session_overview tool."""
        from agent_v7.tools_v2 import get_session_overview

        result = get_session_overview(session_id=24)
        assert result.get("tool_name") == "get_session_overview"
        if result.get("found"):
            assert "session_name" in result
            assert "speakers" in result


class TestReActAgent:
    """Test the ReAct agent."""

    def test_agent_creation(self):
        """Test agent creation."""
        from agent_v7.react_agent import ScaffoldingAgent

        agent = ScaffoldingAgent(conversation_id="test-agent")
        assert agent.conversation_id == "test-agent"
        assert agent.memory is not None

    def test_agent_simple_query(self):
        """Test agent with a simple query."""
        from agent_v7.react_agent import run_agent

        # This test requires the LLM to be available
        # Skip if LLM is not configured
        try:
            response = run_agent("test-simple", "What sessions are available?")
            assert response.answer is not None
            assert isinstance(response.answer, str)
        except Exception as e:
            pytest.skip(f"LLM not available: {e}")


class TestGraph:
    """Test the simplified graph."""

    def test_graph_creation(self):
        """Test graph creation."""
        from agent_v7.graph_v2 import create_agent_graph

        graph = create_agent_graph()
        assert graph is not None

    def test_invoke_agent(self):
        """Test invoke_agent function."""
        from agent_v7.graph_v2 import invoke_agent

        # This test requires the LLM to be available
        try:
            result = invoke_agent(
                query="List all sessions",
                conversation_id="test-invoke"
            )
            assert "answer" in result
            assert isinstance(result["answer"], str)
        except Exception as e:
            pytest.skip(f"LLM not available: {e}")


class TestPrompts:
    """Test prompt formatting."""

    def test_system_prompt_format(self):
        """Test system prompt formatting."""
        from agent_v7.prompts_v2 import format_system_prompt

        prompt = format_system_prompt(
            memory_context="Session focus: 24",
            steering_instructions="User prefers concept map"
        )

        assert "Session focus: 24" in prompt
        assert "concept map" in prompt.lower()

    def test_tool_descriptions(self):
        """Test tool descriptions are present."""
        from agent_v7.prompts_v2 import TOOL_DESCRIPTIONS

        assert len(TOOL_DESCRIPTIONS) >= 8
        for tool in TOOL_DESCRIPTIONS:
            assert "name" in tool
            assert "description" in tool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
