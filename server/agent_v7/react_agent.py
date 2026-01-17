"""
ReAct Agent for BLINC Agent V7

A simple, flexible agent that:
1. Uses LLM to decide what tools to call (not hardcoded patterns)
2. Maintains conversation memory across turns
3. Produces scaffolded responses with specific evidence
4. Respects user steering preferences

V7 Enhancement: Query Classification + Exploratory Retrieval
- Classifies queries as exploratory (cross-session) or targeted (single-session)
- Exploratory queries get systematic multi-session retrieval
- Targeted queries use the flexible ReAct loop
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable

from .llm import get_reasoning_client, LLMResponse
from .memory import ConversationMemory, get_memory
from .steering import extract_steering, validate_tool_call, SteeringDirectives
from .tools_v2 import CORE_TOOLS, TOOL_SCHEMAS, execute_tool, get_tool_names
from .prompts_v2 import (
    format_system_prompt,
    format_tool_descriptions_for_llm,
    TOOL_DESCRIPTIONS
)
from .classifier import classify_query, QueryClassification, is_simple_discovery_query
from .exploratory import retrieve_exploratory, format_exploratory_evidence_for_synthesis, ExploratoryResult

logger = logging.getLogger(__name__)

# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 8

# Maximum evidence items to include in context
MAX_EVIDENCE_ITEMS = 20


@dataclass
class ToolCall:
    """Represents a tool call decision."""
    name: str
    params: Dict[str, Any]
    reason: str = ""


@dataclass
class AgentAction:
    """Represents an action decision by the agent."""
    action_type: str  # "tool_call" or "respond"
    tool_call: Optional[ToolCall] = None
    response: Optional[str] = None


@dataclass
class AgentResponse:
    """Final response from the agent."""
    answer: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_made: List[ToolCall] = field(default_factory=list)
    session_focus: Optional[int] = None
    speaker_focus: Optional[str] = None
    suggested_explorations: List[str] = field(default_factory=list)


class ScaffoldingAgent:
    """
    ReAct-based agent for scaffolded artifact exploration.

    Key features:
    - Query classification for exploratory vs targeted routing
    - Systematic multi-session retrieval for exploratory queries
    - LLM decides tool usage for targeted queries (not hardcoded patterns)
    - Conversation memory for context persistence
    - Steering compliance with validation
    - Scaffolded response generation
    """

    def __init__(self, conversation_id: str):
        """Initialize agent with conversation memory."""
        self.conversation_id = conversation_id
        self.memory = get_memory(conversation_id)
        self.llm = get_reasoning_client()
        self._tools_dict = self._create_tools_dict()

    def _create_tools_dict(self) -> Dict[str, Callable]:
        """
        Create a dictionary of callable tools for the exploratory retriever.

        This wraps execute_tool into individual callable functions.
        """
        def make_tool_fn(tool_name: str) -> Callable:
            def tool_fn(**kwargs):
                return execute_tool(tool_name, kwargs)
            return tool_fn

        return {
            'list_sessions': make_tool_fn('list_sessions'),
            'search_sessions': make_tool_fn('search_sessions'),
            'get_transcript': make_tool_fn('get_transcript'),
            'get_concept_map': make_tool_fn('get_concept_map'),
            'get_7c_analysis': make_tool_fn('get_7c_analysis'),
            'get_speaker_profile': make_tool_fn('get_speaker_profile'),
        }

    def respond(self, query: str) -> AgentResponse:
        """
        Process a user query and return a scaffolded response.

        This is the main entry point for the agent.

        Flow:
        1. Check for simple discovery queries (fast path)
        2. Classify query as exploratory or targeted
        3. Route exploratory queries to systematic multi-session retrieval
        4. Route targeted queries to ReAct loop

        Args:
            query: User's query

        Returns:
            AgentResponse with answer, evidence, and suggestions
        """
        logger.info(f"[Agent] Processing query: {query[:100]}...")

        # Start new turn
        self.memory.start_new_turn()
        self.memory.add_user_message(query)

        # Get steering preferences
        steering = extract_steering(
            query,
            self.memory.messages,
            self.memory.user_steering
        )

        # Extract session/speaker focus from query
        session_id = self.memory.extract_session_from_text(query)
        if session_id and session_id != self.memory.session_focus:
            self.memory.update_session_focus(session_id)

        speaker = self.memory.extract_speaker_from_text(query)
        if speaker and speaker != self.memory.speaker_focus:
            self.memory.update_speaker_focus(speaker)

        # =========================================================
        # FAST PATH: Simple discovery queries
        # =========================================================
        is_simple, tool_name, tool_args = is_simple_discovery_query(query)
        if is_simple and tool_name:
            logger.info(f"[Agent] Fast path: {tool_name}")
            result = execute_tool(tool_name, tool_args or {})
            evidence = [{"tool": tool_name, "params": tool_args or {}, "result": result}]
            answer = self._format_simple_discovery_response(query, result)
            self.memory.add_assistant_message(answer)
            return AgentResponse(
                answer=answer,
                evidence=evidence,
                tool_calls_made=[ToolCall(name=tool_name, params=tool_args or {})],
                session_focus=self.memory.session_focus,
                speaker_focus=self.memory.speaker_focus,
                suggested_explorations=["You can ask about specific sessions or search for topics."]
            )

        # =========================================================
        # CLASSIFY QUERY: Exploratory vs Targeted
        # =========================================================
        classification = classify_query(query, self.memory)
        logger.info(f"[Agent] Classification: is_exploratory={classification.is_exploratory}, reason={classification.reason}")

        if classification.is_exploratory:
            # =========================================================
            # EXPLORATORY PATH: Systematic multi-session retrieval
            # =========================================================
            return self._handle_exploratory_query(query, classification, steering)
        else:
            # =========================================================
            # TARGETED PATH: ReAct loop
            # =========================================================
            return self._handle_targeted_query(query, classification, steering)

    def _format_simple_discovery_response(self, query: str, result: Dict[str, Any]) -> str:
        """Format a simple response for discovery queries."""
        display = result.get('display', '')
        if display:
            return f"Here's what I found:\n\n{display}"
        return "I found some sessions. Would you like to explore any of them?"

    def _handle_exploratory_query(
        self,
        query: str,
        classification: QueryClassification,
        steering: SteeringDirectives
    ) -> AgentResponse:
        """
        Handle exploratory (cross-session) queries.

        Uses systematic multi-session retrieval instead of ReAct loop.
        This ensures we check ALL relevant sessions, not just what LLM decides.
        """
        logger.info(f"[Agent] Exploratory path: {classification.reason}")

        # Systematic retrieval across sessions
        # Note: retrieve_exploratory now handles superlative queries intelligently,
        # picking top candidates based on collaboration scores instead of fixed limit
        exploratory_result = retrieve_exploratory(
            query=query,
            classification=classification,
            tools=self._tools_dict
            # max_sessions now defaults to 20, and superlative queries use smart selection
        )

        logger.info(f"[Agent] Retrieved evidence from {len(exploratory_result.evidence)} sessions")

        # Convert to evidence format
        evidence = []
        tool_calls_made = []

        # Track that search_sessions was used to find relevant sessions
        if exploratory_result.sessions_searched:
            tool_calls_made.append(ToolCall(
                name='search_sessions',
                params={'query': query, 'sessions_found': exploratory_result.sessions_searched},
                reason='Find relevant sessions for exploratory query'
            ))

        for ev in exploratory_result.evidence:
            # Map artifact_type to actual tool name
            tool_name = f"get_{ev.artifact_type}"
            if ev.artifact_type == 'collaboration':
                tool_name = 'get_7c_analysis'

            evidence.append({
                "tool": tool_name,
                "params": {"session_id": ev.session_id},
                "result": ev.raw_result
            })
            tool_calls_made.append(ToolCall(
                name=tool_name,
                params={"session_id": ev.session_id},
                reason="Exploratory retrieval"
            ))

        # Synthesize cross-session response
        answer = self._synthesize_exploratory_response(query, exploratory_result, steering)

        # Generate suggestions
        suggestions = self._generate_exploratory_suggestions(query, exploratory_result)

        # Update memory
        self.memory.add_assistant_message(answer)

        return AgentResponse(
            answer=answer,
            evidence=evidence,
            tool_calls_made=tool_calls_made,
            session_focus=None,  # No single session focus for exploratory
            speaker_focus=self.memory.speaker_focus,
            suggested_explorations=suggestions
        )

    def _synthesize_exploratory_response(
        self,
        query: str,
        exploratory_result: ExploratoryResult,
        steering: SteeringDirectives
    ) -> str:
        """
        Synthesize a response from cross-session evidence.

        Uses a specialized prompt for cross-session synthesis.
        """
        if not exploratory_result.evidence:
            return "I searched across available sessions but couldn't find relevant information for your query. Could you try rephrasing or being more specific?"

        # Format evidence for synthesis
        evidence_str = format_exploratory_evidence_for_synthesis(exploratory_result)

        memory_context = self.memory.get_context_for_llm()
        system_prompt = format_system_prompt(
            memory_context=memory_context,
            steering_instructions=steering.raw_instructions
        )

        user_message = f"""Synthesize findings across multiple sessions to answer this query:

Query: {query}

{evidence_str}

Instructions for synthesis:
1. Compare and contrast findings across sessions
2. Cite specific evidence from each session (e.g., "In Session 19, speaker X said...")
3. Identify patterns or themes that appear across sessions
4. Note any differences or contradictions between sessions
5. Provide a comprehensive answer that draws from ALL sessions, not just one

When interpreting speaker participation patterns:
- Low participation % + high question rate often indicates a facilitator/interviewer role
- Compare actual participation to equal share to assess dominance vs deference
- Consistent patterns across sessions suggest a stable role (host, facilitator, etc.)

Write a clear, well-organized response that helps the user understand findings across all sessions."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.llm.complete(
                messages=messages,
                temperature=0.4,
                max_tokens=4000
            )
            return response.content
        except Exception as e:
            logger.error(f"[Agent] Exploratory synthesis error: {e}")
            # Fallback: return the summary
            return f"I found information across {len(exploratory_result.evidence)} sessions:\n\n{exploratory_result.summary}\n\nPlease ask a more specific question to explore these sessions further."

    def _generate_exploratory_suggestions(self, query: str, result: ExploratoryResult) -> List[str]:
        """Generate follow-up suggestions for exploratory queries."""
        suggestions = []

        if result.sessions_searched:
            # Suggest drilling into specific sessions
            session_id = result.sessions_searched[0]
            suggestions.append(f"Explore session {session_id} in more detail")

            if len(result.sessions_searched) > 1:
                suggestions.append(f"Compare collaboration quality between sessions {result.sessions_searched[0]} and {result.sessions_searched[1]}")

        return suggestions[:2]

    def _handle_targeted_query(
        self,
        query: str,
        classification: QueryClassification,
        steering: SteeringDirectives
    ) -> AgentResponse:
        """
        Handle targeted (single-session) queries using ReAct loop.

        This is the original ReAct implementation for queries about
        specific sessions, speakers, or topics within a known context.
        """
        logger.info(f"[Agent] Targeted path: {classification.reason}")

        # Build context
        memory_context = self.memory.get_context_for_llm()

        # ReAct loop
        evidence = []
        tool_calls_made = []
        tools_called_with_params = set()

        for iteration in range(MAX_ITERATIONS):
            logger.info(f"[Agent] Iteration {iteration + 1}/{MAX_ITERATIONS}")

            # Decide next action
            action = self._decide_action(
                query=query,
                memory_context=memory_context,
                evidence=evidence,
                steering=steering,
                suggested_tool=classification.suggested_tool
            )

            if action.action_type == "respond":
                logger.info("[Agent] Decided to respond")
                break

            elif action.action_type == "tool_call" and action.tool_call:
                tool_call = action.tool_call
                logger.info(f"[Agent] Calling tool: {tool_call.name}")

                # Create hash of tool call to detect duplicates
                params_str = json.dumps(tool_call.params, sort_keys=True)
                call_key = f"{tool_call.name}:{params_str}"

                if call_key in tools_called_with_params:
                    logger.info(f"[Agent] Skipping duplicate tool call: {tool_call.name}")
                    if len(evidence) > 0:
                        break
                    continue

                tools_called_with_params.add(call_key)

                # Validate against steering
                is_valid, reason = validate_tool_call(tool_call.name, steering)
                if not is_valid:
                    logger.warning(f"[Agent] Tool blocked by steering: {reason}")
                    evidence.append({
                        "type": "steering_block",
                        "tool": tool_call.name,
                        "reason": reason
                    })
                    continue

                # Execute tool
                result = execute_tool(tool_call.name, tool_call.params)
                evidence.append({
                    "tool": tool_call.name,
                    "params": tool_call.params,
                    "result": result
                })
                tool_calls_made.append(tool_call)

                # Record artifact retrieval in memory
                if tool_call.name in ['get_transcript', 'get_concept_map', 'get_7c_analysis']:
                    session_id = tool_call.params.get('session_id')
                    if session_id:
                        artifact_type = tool_call.name.replace('get_', '').replace('_analysis', '')
                        self.memory.record_artifact(artifact_type, session_id)
                        if not self.memory.session_focus:
                            self.memory.update_session_focus(session_id)

        # Synthesize response
        answer = self._synthesize_response(
            query=query,
            memory_context=memory_context,
            evidence=evidence,
            steering=steering
        )

        # Extract suggestions
        suggestions = self._extract_suggestions(answer, evidence)

        # Update memory
        self.memory.add_assistant_message(answer)

        return AgentResponse(
            answer=answer,
            evidence=evidence,
            tool_calls_made=tool_calls_made,
            session_focus=self.memory.session_focus,
            speaker_focus=self.memory.speaker_focus,
            suggested_explorations=suggestions
        )

    def _decide_action(
        self,
        query: str,
        memory_context: str,
        evidence: List[Dict],
        steering: SteeringDirectives,
        suggested_tool: Optional[str] = None
    ) -> AgentAction:
        """
        Use LLM to decide next action: call a tool or respond.

        Args:
            suggested_tool: Optional hint from classifier for recommended tool

        Returns:
            AgentAction with either tool_call or respond decision
        """
        # Build messages for LLM
        system_prompt = format_system_prompt(
            memory_context=memory_context,
            steering_instructions=steering.raw_instructions
        )

        # Format evidence for context
        evidence_str = self._format_evidence_for_context(evidence)

        # Add suggested tool hint if present
        tool_hint = ""
        if suggested_tool and not evidence:
            tool_hint = f"""
**RECOMMENDED TOOL**: {suggested_tool}
The query classification suggests using {suggested_tool} - this tool directly provides
the statistics being asked about (question counts, word counts, utterance counts, etc.)
"""

        user_message = f"""Query: {query}
{tool_hint}
Evidence gathered so far:
{evidence_str if evidence_str else "None yet"}

Decide your next action:
- If you have enough evidence to answer the query thoroughly with specific citations, respond now.
- If you need more information, call an appropriate tool.

Respond with either:
1. RESPOND: [your response]
2. TOOL: tool_name
   PARAMS: {{"param": "value"}}
   REASON: why this tool helps"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Call LLM with tools
        try:
            response = self.llm.complete_with_tools(
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=0.3,
                max_tokens=2000
            )

            # Check if LLM made a tool call
            if response.finish_reason == "tool_calls" and response.raw_response:
                tool_calls = response.raw_response.get("tool_calls", [])
                valid_tool_names = get_tool_names()

                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")

                    # Skip invalid tool names (like multi_tool_use, functions, etc.)
                    if tool_name not in valid_tool_names:
                        logger.debug(f"[Agent] Skipping invalid tool: {tool_name}")
                        continue

                    try:
                        params = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        params = {}

                    return AgentAction(
                        action_type="tool_call",
                        tool_call=ToolCall(
                            name=tool_name,
                            params=params,
                            reason="LLM tool call"
                        )
                    )

            # Otherwise, LLM wants to respond
            content = response.content

            # Parse text response for action
            if content.strip().upper().startswith("RESPOND:"):
                return AgentAction(
                    action_type="respond",
                    response=content[8:].strip()
                )
            elif "TOOL:" in content.upper():
                # Parse manual tool call format
                tool_call = self._parse_tool_call_from_text(content)
                if tool_call:
                    return AgentAction(
                        action_type="tool_call",
                        tool_call=tool_call
                    )

            # Default: if we have evidence, respond; otherwise try to get some
            if evidence:
                return AgentAction(action_type="respond")
            else:
                # Try to determine a sensible first tool call
                default_tool = self._get_default_tool_call(query)
                if default_tool:
                    return AgentAction(
                        action_type="tool_call",
                        tool_call=default_tool
                    )
                return AgentAction(action_type="respond")

        except Exception as e:
            logger.error(f"[Agent] Decision error: {e}")
            return AgentAction(action_type="respond")

    def _synthesize_response(
        self,
        query: str,
        memory_context: str,
        evidence: List[Dict],
        steering: SteeringDirectives
    ) -> str:
        """
        Synthesize a scaffolded response from gathered evidence.

        Returns:
            Natural language response with specific citations
        """
        system_prompt = format_system_prompt(
            memory_context=memory_context,
            steering_instructions=steering.raw_instructions
        )

        evidence_str = self._format_evidence_for_synthesis(evidence)

        user_message = f"""Based on the evidence gathered, provide a scaffolded response to this query:

Query: {query}

Evidence:
{evidence_str}

Instructions:
1. Point to SPECIFIC evidence (exact quotes, coded segments, concept nodes)
2. Explain WHY the evidence is relevant
3. Use natural language ("You can see this in...", "Notice how...")
4. If evidence is incomplete, acknowledge what couldn't be determined
5. Suggest related artifacts the user might want to explore

When interpreting speaker participation patterns:
- Low participation % + high question rate often indicates a facilitator/interviewer role
- Compare actual participation to equal share to assess dominance vs deference
- Consistent patterns across sessions suggest a stable role (host, facilitator, etc.)

Write a conversational response that guides the user through the evidence."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.llm.complete(
                messages=messages,
                temperature=0.4,
                max_tokens=3000
            )
            return response.content

        except Exception as e:
            logger.error(f"[Agent] Synthesis error: {e}")
            return self._fallback_response(query, evidence)

    def _format_evidence_for_context(self, evidence: List[Dict]) -> str:
        """Format evidence for decision-making context (concise summary).

        Uses the 'display' field from tool results but shows only first few lines
        for quick decision-making.
        """
        if not evidence:
            return ""

        lines = []
        for e in evidence[-MAX_EVIDENCE_ITEMS:]:
            if e.get("type") == "steering_block":
                lines.append(f"[BLOCKED] {e.get('tool')}: {e.get('reason')}")
            else:
                tool = e.get("tool", "unknown")
                result = e.get("result", {})

                if result.get("error"):
                    lines.append(f"[{tool}] Error: {result.get('error')}")
                else:
                    # Show first 3 lines of display as summary
                    display = result.get("display", "")
                    if display:
                        summary_lines = display.split("\n")[:3]
                        summary = " | ".join(line.strip() for line in summary_lines if line.strip())
                        lines.append(f"[{tool}] {summary}")
                    else:
                        lines.append(f"[{tool}] Completed")

        return "\n".join(lines)

    def _format_evidence_for_synthesis(self, evidence: List[Dict]) -> str:
        """Format evidence for synthesis (detailed).

        Now simply uses the 'display' field from tool results directly.
        Tools return LLM-ready text, so no formatting/transformation needed.
        This eliminates data loss from intermediate formatting.
        """
        if not evidence:
            return "No evidence gathered."

        sections = []

        for e in evidence:
            if e.get("type") == "steering_block":
                continue  # Skip blocked tools

            tool = e.get("tool", "unknown")
            result = e.get("result", {})

            if result.get("error"):
                sections.append(f"## {tool}\nError: {result.get('error')}")
                continue

            # Use display field directly - no transformation needed
            display = result.get("display", "")
            if display:
                sections.append(display)
            else:
                sections.append(f"## {tool}\n(No display content available)")

        return "\n\n".join(sections)

    def _parse_tool_call_from_text(self, text: str) -> Optional[ToolCall]:
        """Parse a tool call from text format."""
        import re

        # Look for TOOL: name pattern
        tool_match = re.search(r'TOOL:\s*(\w+)', text, re.IGNORECASE)
        if not tool_match:
            return None

        tool_name = tool_match.group(1)

        # Look for PARAMS: {json} pattern
        params = {}
        params_match = re.search(r'PARAMS:\s*(\{[^}]+\})', text, re.IGNORECASE)
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                pass

        # Look for REASON: text pattern
        reason = ""
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()

        return ToolCall(name=tool_name, params=params, reason=reason)

    def _get_default_tool_call(self, query: str) -> Optional[ToolCall]:
        """Get a sensible default tool call based on query."""
        query_lower = query.lower()

        # Check for session-specific queries
        import re
        session_match = re.search(r'session\s*(\d+)', query_lower)
        if session_match:
            session_id = int(session_match.group(1))
            # Default to transcript for most queries
            return ToolCall(
                name="get_transcript",
                params={"session_id": session_id},
                reason="Get transcript for specified session"
            )

        # Check for session listing
        if any(kw in query_lower for kw in ['sessions', 'available', 'list']):
            return ToolCall(
                name="list_sessions",
                params={},
                reason="List available sessions"
            )

        # Default to search
        return ToolCall(
            name="search_sessions",
            params={"query": query[:200], "top_k": 5},
            reason="Search for relevant sessions"
        )

    def _extract_suggestions(self, answer: str, evidence: List[Dict]) -> List[str]:
        """Extract or generate suggestions for further exploration."""
        suggestions = []

        # Check what artifacts were NOT retrieved
        retrieved_types = set()
        retrieved_sessions = set()

        for e in evidence:
            tool = e.get("tool", "")
            result = e.get("result", {})

            if tool == "get_transcript":
                retrieved_types.add("transcript")
                # session_id is still available in result metadata
                if result.get("session_id"):
                    retrieved_sessions.add(result.get("session_id"))
            elif tool == "get_concept_map":
                retrieved_types.add("concept_map")
                if result.get("session_id"):
                    retrieved_sessions.add(result.get("session_id"))
            elif tool == "get_7c_analysis":
                retrieved_types.add("7c")
                if result.get("session_id"):
                    retrieved_sessions.add(result.get("session_id"))

        # Suggest unexplored artifacts for retrieved sessions
        if retrieved_sessions:
            session_id = list(retrieved_sessions)[0]
            if "concept_map" not in retrieved_types:
                suggestions.append(f"You might want to explore the concept map for session {session_id} to see how ideas connect.")
            if "7c" not in retrieved_types:
                suggestions.append(f"The 7C collaboration analysis for session {session_id} could show interaction quality.")

        return suggestions[:2]  # Limit suggestions

    def _fallback_response(self, query: str, evidence: List[Dict]) -> str:
        """Generate a fallback response when synthesis fails."""
        if not evidence:
            return "I wasn't able to find relevant information for your query. Could you provide more details or specify a session?"

        # Try to summarize what was found
        tools_used = [e.get("tool") for e in evidence if e.get("tool")]
        return f"I gathered information using {', '.join(tools_used)}, but had trouble synthesizing a complete response. The evidence is available for your review."


# =============================================================================
# Convenience Functions
# =============================================================================

def run_agent(conversation_id: str, query: str) -> AgentResponse:
    """
    Run the agent for a single query.

    This is the main entry point for the routes.

    Args:
        conversation_id: Unique conversation identifier
        query: User's query

    Returns:
        AgentResponse with answer and metadata
    """
    agent = ScaffoldingAgent(conversation_id)
    return agent.respond(query)


def clear_conversation(conversation_id: str):
    """Clear conversation memory."""
    from .memory import clear_memory
    clear_memory(conversation_id)
