"""
ReAct Agent for BLINC Agent V7 (Pure ReAct Architecture)

A simple, flexible agent that:
1. Uses LLM to decide what tools to call (not hardcoded patterns)
2. Maintains conversation memory across turns
3. Produces scaffolded responses with specific evidence
4. Respects user steering preferences
5. Supports artifact steering (user controls which tools to use)

V7.2: Pure ReAct - All queries go through the ReAct loop.
The LLM decides what tools to call based on:
- Query understanding
- Tool guidance in system prompt
- User steering constraints
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable, Set

from .llm import get_reasoning_client, LLMResponse
from .memory import ConversationMemory, get_memory
from .steering import extract_steering, validate_tool_call, SteeringDirectives
from .tools_v2 import CORE_TOOLS, TOOL_SCHEMAS, execute_tool, get_tool_names
from .prompts_v2 import (
    format_system_prompt,
    format_tool_descriptions_for_llm,
    TOOL_DESCRIPTIONS
)
# Note: classifier.py and exploratory.py are deprecated in V7.2
# All queries now go through pure ReAct loop

logger = logging.getLogger(__name__)

# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 8

# Maximum evidence items to include in context
MAX_EVIDENCE_ITEMS = 20

# Session name to ID mapping (for query-aware response gating)
SESSION_NAME_MAPPING = {
    "living in nyc": 18,
    "living in new york": 18,
    "nyc": 18,
    "is ai alive": 19,
    "ai alive": 19,
    "nuclear fusion": 20,
    "fusion": 20,
    "shaw interview": 21,
    "shaw": 21,
    "collaboration literacy": 22,
    "collab literacy": 22,
    "dinosaurs": 23,
    "country music": 24,
    "abundance": 25,
    "cfaa discussion": 26,
    "cfaa": 26,
}

# =============================================================================
# Query Classification Patterns
# =============================================================================

# Patterns that indicate comparison/multi-session queries
COMPARISON_PATTERNS = [
    r'compare\s+(.+?)\s+(?:and|vs\.?|versus|with|to)\s+(.+)',
    r'(.+?)\s+vs\.?\s+(.+)',
    r'difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+)',
    r'how\s+(?:does|do|did)\s+(.+?)\s+(?:differ|compare)\s+(?:from|to|with)\s+(.+)',
]

# Patterns that indicate superlative/ranking queries needing multiple sessions
SUPERLATIVE_PATTERNS = [
    r'which\s+sessions?\s+(?:had|has|was|is|showed|demonstrated)\s+(?:the\s+)?(?:most|best|highest|greatest|lowest|worst|least)',
    r'(?:best|worst|highest|lowest|most|least)\s+(?:collaboration|engagement|participation|communication|constructive|conflict)',
    r'rank\s+(?:the\s+)?sessions',
    r'(?:more|most)\s+(?:constructive|balanced|engaging)',
    r'which\s+(?:session|discussion)\s+had\s+more',  # "Which session had more X" - comparative
    r'which\s+(?:session|discussion)\s+had\s+(?:better|worse)',  # "Which session had better/worse X"
    r'across\s+(?:all\s+)?sessions',  # "7C scores across sessions" - needs multiple sessions
]

# Patterns that indicate hypothesis testing queries
HYPOTHESIS_PATTERNS = [
    r'test\s+(?:whether|if|that)',
    r'verify\s+(?:whether|if|that)',
    r'is\s+it\s+true\s+that',
    r'(?:does|do|did)\s+.+\s+(?:have|show|demonstrate)\s+(?:more|less|better|worse|higher|lower)',
    r'(?:hypothesis|claim|theory|proposition)[\s:]+',
    r'evidence\s+(?:for|against|that)',
    # "X more/less than Y" patterns
    r'(.+?)\s+(?:more|less|higher|lower|better|worse)\s+(?:\w+\s+)?than\s+(.+)',
]

# Patterns that indicate thematic/topic-based queries (should use search_sessions)
THEMATIC_PATTERNS = [
    r'(?:what\s+was\s+)?(?:said|discussed|mentioned|talked)\s+about\s+(.+?)(?:\?|$|across|in\s+the)',
    r'(?:sessions?|discussions?)\s+(?:about|on|regarding|involving|related\s+to)\s+(.+?)(?:\?|$)',
    r'(?:find|search|look\s+for)\s+(?:sessions?|discussions?)\s+(?:about|on|regarding)\s+(.+?)(?:\?|$)',
    r'(?:where|when)\s+(?:was|were|did)\s+(.+?)\s+(?:discussed|mentioned|brought\s+up)',
    r'(?:any\s+)?(?:sessions?|discussions?)\s+(?:that\s+)?(?:mention|discuss|cover|address)\s+(.+?)(?:\?|$)',
    # Note: "across sessions" moved to SUPERLATIVE_PATTERNS - it indicates need for multi-session data
    r'(?:all|every|each)\s+session.*(?:about|discuss|mention)',
]

# Patterns that indicate structural queries (metadata only, use list_sessions)
STRUCTURAL_PATTERNS = [
    r'how\s+many\s+sessions',
    r'(?:list|show|display)\s+(?:all\s+)?sessions',
    r'sessions?\s+with\s+\d+\s+(?:speakers?|participants?)',
    r'what\s+sessions\s+(?:are\s+)?(?:available|exist)',
    r'(?:all|available)\s+sessions',
    r'session\s+(?:names?|ids?|list)',
]

# Patterns for speaker-focused queries
SPEAKER_PATTERNS = [
    r'how\s+(?:did|does)\s+(\w+)\s+(?:engage|participate|contribute)',
    r'(\w+)(?:\'s|s)\s+(?:style|pattern|behavior|contribution)',
    r'(?:most|least)\s+active\s+speaker',
    r'speaker\s+(?:comparison|analysis|profile)',
    r'who\s+(?:spoke|talked|contributed)\s+(?:the\s+)?(?:most|least)',
]


@dataclass
class QueryClassification:
    """Classification of query with data requirements."""
    query_type: str  # single_session, comparison, thematic, superlative, hypothesis, structural, speaker
    required_sessions: Set[int]  # Explicit sessions mentioned
    requires_search: bool  # Needs semantic search first
    requires_counter_evidence: bool  # For hypothesis testing
    topic: Optional[str]  # Extracted topic for thematic queries
    min_sessions_needed: int = 1  # Minimum sessions needed for complete answer


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
    Pure ReAct agent for scaffolded artifact exploration.

    Key features:
    - LLM decides what tools to call (no hardcoded routing)
    - Tool guidance in system prompt for different query types
    - Artifact steering support (user controls data sources)
    - Conversation memory for context persistence
    - Steering compliance with validation
    - Scaffolded response generation

    V7.2: Removed classifier and exploratory path. LLM reasoning replaces
    hardcoded query routing.
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

        V7.2 Flow (Pure ReAct):
        1. Extract context (session/speaker focus, steering)
        2. Run ReAct loop - LLM decides what tools to call
        3. Synthesize and return response

        The LLM uses tool guidance in system prompt to decide:
        - list_sessions for structural/superlative/hypothesis queries
        - search_sessions for topic-based discovery
        - Appropriate artifact tools based on query needs

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
        # PURE REACT: LLM decides what tools to call
        # =========================================================
        # No classifier routing - all queries go through ReAct
        # The system prompt guides the LLM on tool selection:
        # - Hypothesis testing → list_sessions first
        # - Structural queries → list_sessions for metadata
        # - Topic queries → search_sessions
        # - Artifact steering → respect user constraints
        return self._run_react_loop(query, steering)

    def _run_react_loop(
        self,
        query: str,
        steering: SteeringDirectives
    ) -> AgentResponse:
        """
        Run the ReAct loop to process a query.

        The LLM decides what tools to call based on:
        - Query understanding
        - Tool guidance in system prompt
        - User steering constraints

        This is the core of V7.2 - all queries go through this loop.
        """
        logger.info(f"[Agent] Running ReAct loop for query")

        # Build context
        memory_context = self.memory.get_context_for_llm()

        # ReAct loop
        evidence = []
        tool_calls_made = []
        tools_called_with_params = set()

        for iteration in range(MAX_ITERATIONS):
            logger.info(f"[Agent] Iteration {iteration + 1}/{MAX_ITERATIONS}")

            # Decide next action
            # Note: suggested_tool removed in V7.2 - LLM uses prompt guidance instead
            action = self._decide_action(
                query=query,
                memory_context=memory_context,
                evidence=evidence,
                steering=steering
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
                        # Before breaking, check if we need more data for comparison/superlative
                        analysis = self._analyze_query_completeness(query, evidence)
                        if not analysis['complete']:
                            logger.info(f"[Agent] Duplicate detected but data incomplete: "
                                       f"is_superlative={analysis.get('is_superlative')}, "
                                       f"is_comparison={analysis.get('is_comparison')}")
                            # Force retrieval for missing data
                            forced_action = self._create_retrieval_action_for_missing(analysis, query)
                            if forced_action.tool_call:
                                # Replace with forced tool call and fall through to execution
                                tool_call = forced_action.tool_call
                                logger.info(f"[Agent] Forcing retrieval: {tool_call.name}")
                                call_key = f"{tool_call.name}:{json.dumps(tool_call.params, sort_keys=True)}"
                                if call_key in tools_called_with_params:
                                    # Forced call is also a duplicate - give up
                                    break
                                # Fall through to execution below
                            else:
                                break
                        else:
                            break
                    else:
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
        steering: SteeringDirectives
    ) -> AgentAction:
        """
        Use LLM to decide next action: call a tool or respond.

        V7.2: The LLM uses guidance in the system prompt to decide what tools to call.
        No suggested_tool hint - the prompt explains when to use each tool.

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

        user_message = f"""Query: {query}

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
                # LLM wants to respond - but first check if we have enough data
                # for comparison/superlative queries
                if evidence:
                    analysis = self._analyze_query_completeness(query, evidence)
                    if not analysis['complete']:
                        logger.info(f"[Agent] LLM wants to respond but data incomplete: "
                                   f"is_comparison={analysis.get('is_comparison')}, "
                                   f"is_superlative={analysis.get('is_superlative')}, "
                                   f"missing={analysis.get('missing_sessions', set())}")
                        return self._create_retrieval_action_for_missing(analysis, query)
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

            # Fix: Handle "Please hold on while I retrieve..." pattern
            # LLM intended to make a tool call but didn't format it properly
            if self._mentions_retrieval_intent(content):
                logger.warning(f"[Agent] Detected retrieval intent without tool call: {content[:100]}...")
                # Force a default tool call instead of falling through to respond
                default_tool = self._get_default_tool_call(query)
                if default_tool:
                    return AgentAction(
                        action_type="tool_call",
                        tool_call=default_tool
                    )

            # Query-aware response gating:
            # For comparison/superlative queries, check if we have sufficient data
            if evidence:
                analysis = self._analyze_query_completeness(query, evidence)

                if analysis['complete']:
                    logger.info(f"[Agent] Query analysis: complete=True, responding")
                    return AgentAction(action_type="respond")
                else:
                    # Need more data - force retrieval for missing sessions
                    logger.info(f"[Agent] Query analysis: complete=False, "
                               f"missing={analysis.get('missing_sessions', set())}, "
                               f"is_comparison={analysis.get('is_comparison')}, "
                               f"is_superlative={analysis.get('is_superlative')}")
                    return self._create_retrieval_action_for_missing(analysis, query)
            else:
                # No evidence yet - try to determine a sensible first tool call
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

    def _extract_sessions_from_query(self, query: str) -> Set[int]:
        """
        Extract session IDs mentioned in query by name or number.

        Returns set of session IDs that should be retrieved for this query.
        """
        query_lower = query.lower()
        sessions = set()

        # Check for session names
        for name, session_id in SESSION_NAME_MAPPING.items():
            if name in query_lower:
                sessions.add(session_id)

        # Check for explicit session numbers
        session_num_matches = re.findall(r'session\s*(\d+)', query_lower)
        for match in session_num_matches:
            sessions.add(int(match))

        return sessions

    def _classify_query(self, query: str) -> QueryClassification:
        """
        Classify query and determine data requirements.

        This is the core of the principled query understanding system.
        Each query type has specific data requirements that must be satisfied
        before the agent can respond.

        Returns:
            QueryClassification with query_type and data requirements
        """
        query_lower = query.lower()
        required_sessions = self._extract_sessions_from_query(query)

        # 1. Check for THEMATIC patterns (should use search_sessions)
        # These are topic-based queries without explicit session references
        for pattern in THEMATIC_PATTERNS:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                # Try to extract topic from capture group
                topic = None
                try:
                    topic = match.group(1).strip() if match.lastindex else None
                except IndexError:
                    pass

                # If no topic extracted, try to find key terms
                if not topic:
                    topic = self._extract_likely_topic(query)

                return QueryClassification(
                    query_type='thematic',
                    required_sessions=required_sessions,
                    requires_search=True,
                    requires_counter_evidence=False,
                    topic=topic,
                    min_sessions_needed=1
                )

        # 2. Check for HYPOTHESIS patterns
        # These need evidence from all mentioned entities + counter-evidence
        for pattern in HYPOTHESIS_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryClassification(
                    query_type='hypothesis',
                    required_sessions=required_sessions,
                    requires_search=len(required_sessions) == 0,  # Search if no explicit sessions
                    requires_counter_evidence=True,
                    topic=self._extract_likely_topic(query) if len(required_sessions) == 0 else None,
                    min_sessions_needed=max(2, len(required_sessions))  # Need at least 2 for comparison
                )

        # 3. Check for COMPARISON patterns
        # These need data from ALL mentioned sessions
        for pattern in COMPARISON_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryClassification(
                    query_type='comparison',
                    required_sessions=required_sessions,
                    requires_search=False,
                    requires_counter_evidence=False,
                    topic=None,
                    min_sessions_needed=max(2, len(required_sessions))
                )

        # 4. Check for SUPERLATIVE patterns
        # These need list_sessions + detailed data for top N
        for pattern in SUPERLATIVE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryClassification(
                    query_type='superlative',
                    required_sessions=set(),  # Need to discover via list_sessions
                    requires_search=False,
                    requires_counter_evidence=False,
                    topic=None,
                    min_sessions_needed=2  # Need at least top 2 for comparison
                )

        # 5. Check for STRUCTURAL patterns (metadata only)
        for pattern in STRUCTURAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryClassification(
                    query_type='structural',
                    required_sessions=set(),
                    requires_search=False,
                    requires_counter_evidence=False,
                    topic=None,
                    min_sessions_needed=0  # list_sessions is sufficient
                )

        # 6. Check for SPEAKER patterns
        for pattern in SPEAKER_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return QueryClassification(
                    query_type='speaker',
                    required_sessions=required_sessions,
                    requires_search=len(required_sessions) == 0,  # Search if no explicit sessions
                    requires_counter_evidence=False,
                    topic=None,
                    min_sessions_needed=1
                )

        # 7. If explicit sessions mentioned, it's a SINGLE_SESSION query
        if required_sessions:
            return QueryClassification(
                query_type='single_session',
                required_sessions=required_sessions,
                requires_search=False,
                requires_counter_evidence=False,
                topic=None,
                min_sessions_needed=len(required_sessions)
            )

        # 8. Default: Treat as THEMATIC (use search to find relevant sessions)
        # This ensures we use semantic search for ambiguous queries
        return QueryClassification(
            query_type='unknown',
            required_sessions=set(),
            requires_search=True,
            requires_counter_evidence=False,
            topic=self._extract_likely_topic(query),
            min_sessions_needed=1
        )

    def _extract_likely_topic(self, query: str) -> str:
        """
        Extract the likely topic from a query for semantic search.

        Removes common question words and returns key content words.
        """
        # Remove common question starters
        query_lower = query.lower()
        remove_phrases = [
            r'^what\s+(was|were|is|are)\s+',
            r'^how\s+(did|does|do|was|were)\s+',
            r'^which\s+',
            r'^where\s+(did|does|was|were)\s+',
            r'^when\s+(did|does|was|were)\s+',
            r'^can\s+you\s+',
            r'^tell\s+me\s+about\s+',
            r'^show\s+me\s+',
            r'^find\s+',
            r'^search\s+for\s+',
        ]

        topic = query_lower
        for pattern in remove_phrases:
            topic = re.sub(pattern, '', topic, flags=re.IGNORECASE)

        # Remove trailing punctuation
        topic = re.sub(r'[?.!]+$', '', topic).strip()

        # If topic is too long, take first 100 chars
        if len(topic) > 100:
            topic = topic[:100]

        return topic if topic else query[:50]

    def _is_comparison_query(self, query: str) -> bool:
        """Check if query requires comparing multiple sessions."""
        query_lower = query.lower()

        # Check comparison patterns
        for pattern in COMPARISON_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True

        return False

    def _is_superlative_query(self, query: str) -> bool:
        """Check if query asks for best/worst/ranking across sessions."""
        query_lower = query.lower()

        for pattern in SUPERLATIVE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True

        return False

    def _get_sessions_retrieved(self, evidence: List[Dict]) -> Set[int]:
        """Extract session IDs from evidence that has been retrieved."""
        sessions = set()

        for e in evidence:
            if e.get("type") == "steering_block":
                continue

            tool = e.get("tool", "")
            result = e.get("result", {})
            params = e.get("params", {})

            # Get session_id from params or result
            session_id = params.get("session_id") or result.get("session_id")
            if session_id:
                sessions.add(session_id)

            # For list_sessions, check if detailed data was returned
            if tool == "list_sessions":
                # list_sessions returns overview, not detailed per-session data
                pass

        return sessions

    def _analyze_query_completeness(self, query: str, evidence: List[Dict]) -> dict:
        """
        Analyze whether we have sufficient evidence for the query.

        This is the key gating function that prevents premature responses.
        Uses query classification to determine data requirements.

        Returns:
            {
                'query_type': str,
                'classification': QueryClassification,
                'required_sessions': Set[int],
                'retrieved_sessions': Set[int],
                'missing_sessions': Set[int],
                'has_search_results': bool,
                'has_detailed_data': bool,
                'complete': bool,
                'reason': Optional[str]  # Why incomplete
            }
        """
        classification = self._classify_query(query)
        retrieved_sessions = self._get_sessions_retrieved(evidence)

        # Check what types of evidence we have
        has_search_results = any(
            e.get("tool") == "search_sessions"
            for e in evidence if e.get("type") != "steering_block"
        )
        has_list_overview = any(
            e.get("tool") == "list_sessions"
            for e in evidence if e.get("type") != "steering_block"
        )
        has_detailed_data = any(
            e.get("tool") in ["get_7c_analysis", "get_concept_map", "get_transcript"]
            for e in evidence if e.get("type") != "steering_block"
        )

        base_result = {
            'query_type': classification.query_type,
            'classification': classification,
            'required_sessions': classification.required_sessions,
            'retrieved_sessions': retrieved_sessions,
            'has_search_results': has_search_results,
            'has_detailed_data': has_detailed_data,
            # Legacy fields for backwards compatibility
            'is_comparison': classification.query_type in ['comparison', 'hypothesis'],
            'is_superlative': classification.query_type == 'superlative',
        }

        # THEMATIC: Need search + retrieval from at least one match
        if classification.query_type == 'thematic':
            if not has_search_results:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': 'Need to search for relevant sessions first',
                    'next_action': 'search_sessions'
                }
            if not has_detailed_data:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': 'Need to retrieve data from search results',
                    'next_action': 'get_artifact'
                }
            return {**base_result, 'missing_sessions': set(), 'complete': True, 'reason': None}

        # COMPARISON: Need data for ALL mentioned sessions
        if classification.query_type == 'comparison':
            missing = classification.required_sessions - retrieved_sessions
            if missing:
                return {
                    **base_result,
                    'missing_sessions': missing,
                    'complete': False,
                    'reason': f'Missing data for sessions: {missing}'
                }
            return {**base_result, 'missing_sessions': set(), 'complete': True, 'reason': None}

        # HYPOTHESIS: Need data for all entities + explicit handling
        if classification.query_type == 'hypothesis':
            missing = classification.required_sessions - retrieved_sessions
            # For hypothesis, we need data from at least 2 sessions for comparison
            needs_more = len(retrieved_sessions) < classification.min_sessions_needed

            if missing:
                return {
                    **base_result,
                    'missing_sessions': missing,
                    'complete': False,
                    'reason': f'Missing data for sessions: {missing}'
                }
            if needs_more and not has_search_results:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': f'Need data from at least {classification.min_sessions_needed} sessions for hypothesis testing',
                    'next_action': 'search_sessions' if not classification.required_sessions else 'get_artifact'
                }
            if classification.requires_search and not has_search_results:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': 'Need to search for relevant sessions for hypothesis',
                    'next_action': 'search_sessions'
                }
            return {**base_result, 'missing_sessions': set(), 'complete': True, 'reason': None}

        # SUPERLATIVE: Need list_sessions + detailed data for top candidates
        if classification.query_type == 'superlative':
            if not has_list_overview:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': 'Need list_sessions to see all scores',
                    'next_action': 'list_sessions'
                }
            if not has_detailed_data:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': 'Need detailed data for top candidates',
                    'next_action': 'get_7c_analysis'
                }
            if len(retrieved_sessions) < classification.min_sessions_needed:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': f'Need detailed data for at least {classification.min_sessions_needed} sessions',
                    'next_action': 'get_7c_analysis'
                }
            return {**base_result, 'missing_sessions': set(), 'complete': True, 'reason': None}

        # STRUCTURAL: list_sessions is sufficient
        if classification.query_type == 'structural':
            if not has_list_overview:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': 'Need list_sessions for structural query',
                    'next_action': 'list_sessions'
                }
            return {**base_result, 'missing_sessions': set(), 'complete': True, 'reason': None}

        # SPEAKER: Need speaker profile or search
        if classification.query_type == 'speaker':
            has_speaker_data = any(
                e.get("tool") == "get_speaker_profile"
                for e in evidence if e.get("type") != "steering_block"
            )
            if not has_speaker_data and not has_detailed_data:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': 'Need speaker profile or session data',
                    'next_action': 'get_speaker_profile' if not classification.requires_search else 'search_sessions'
                }
            return {**base_result, 'missing_sessions': set(), 'complete': True, 'reason': None}

        # SINGLE_SESSION: Need data for the specified session(s)
        if classification.query_type == 'single_session':
            missing = classification.required_sessions - retrieved_sessions
            if missing:
                return {
                    **base_result,
                    'missing_sessions': missing,
                    'complete': False,
                    'reason': f'Need data for session(s): {missing}'
                }
            return {**base_result, 'missing_sessions': set(), 'complete': True, 'reason': None}

        # UNKNOWN: Needs at least search or some evidence
        if classification.query_type == 'unknown':
            if not has_search_results and not has_detailed_data:
                return {
                    **base_result,
                    'missing_sessions': set(),
                    'complete': False,
                    'reason': 'Need to search for relevant sessions',
                    'next_action': 'search_sessions'
                }
            return {**base_result, 'missing_sessions': set(), 'complete': True, 'reason': None}

        # Default: Any evidence is sufficient
        return {
            **base_result,
            'missing_sessions': set(),
            'complete': len(evidence) > 0,
            'reason': 'No evidence gathered' if len(evidence) == 0 else None
        }

    def _create_retrieval_action_for_missing(self, analysis: dict, query: str) -> AgentAction:
        """
        Create a tool call to retrieve data based on what's missing.

        This is the smart retrieval function that uses query classification
        to determine the appropriate next tool call.

        Args:
            analysis: Result from _analyze_query_completeness (includes classification)
            query: Original query (to determine appropriate tool)

        Returns:
            AgentAction with tool_call to retrieve missing data
        """
        query_lower = query.lower()
        classification = analysis.get('classification')

        # Use suggested next_action if available
        next_action = analysis.get('next_action')
        if next_action:
            if next_action == 'search_sessions':
                topic = classification.topic if classification else self._extract_likely_topic(query)
                return AgentAction(
                    action_type="tool_call",
                    tool_call=ToolCall(
                        name="search_sessions",
                        params={"query": topic, "top_k": 5},
                        reason=f"Search for sessions about '{topic}'"
                    )
                )
            elif next_action == 'list_sessions':
                return AgentAction(
                    action_type="tool_call",
                    tool_call=ToolCall(
                        name="list_sessions",
                        params={},
                        reason="Get overview of all sessions"
                    )
                )
            elif next_action == 'get_speaker_profile':
                # Try to extract speaker name
                speaker_match = re.search(r"(\w+)'s?\s+(?:style|pattern|engagement|contribution)", query_lower)
                speaker_name = speaker_match.group(1) if speaker_match else "unknown"
                return AgentAction(
                    action_type="tool_call",
                    tool_call=ToolCall(
                        name="get_speaker_profile",
                        params={"speaker_name": speaker_name},
                        reason=f"Get profile for speaker '{speaker_name}'"
                    )
                )
            elif next_action in ['get_artifact', 'get_7c_analysis']:
                # Need to get artifact for a session - find one from search results or use top session
                session_id = self._get_session_to_retrieve(analysis, query)
                tool_name = self._select_artifact_tool(query)
                return AgentAction(
                    action_type="tool_call",
                    tool_call=ToolCall(
                        name=tool_name,
                        params={"session_id": session_id},
                        reason=f"Get {tool_name} for session {session_id}"
                    )
                )

        # Get first missing session if explicit sessions needed
        if analysis.get('missing_sessions'):
            session_id = list(analysis['missing_sessions'])[0]
            tool_name = self._select_artifact_tool(query)
            return AgentAction(
                action_type="tool_call",
                tool_call=ToolCall(
                    name=tool_name,
                    params={"session_id": session_id},
                    reason=f"Retrieve {tool_name} for missing session {session_id}"
                )
            )

        # For superlative/hypothesis queries needing more detailed data
        query_type = analysis.get('query_type', '')
        if query_type in ['superlative', 'hypothesis'] and not analysis['complete']:
            session_id = self._get_session_to_retrieve(analysis, query)
            tool_name = self._select_artifact_tool(query)
            return AgentAction(
                action_type="tool_call",
                tool_call=ToolCall(
                    name=tool_name,
                    params={"session_id": session_id},
                    reason=f"Get {tool_name} for detailed comparison"
                )
            )

        # For thematic queries that searched but didn't retrieve
        if query_type == 'thematic' and analysis.get('has_search_results') and not analysis.get('has_detailed_data'):
            session_id = self._get_session_to_retrieve(analysis, query)
            tool_name = self._select_artifact_tool(query)
            return AgentAction(
                action_type="tool_call",
                tool_call=ToolCall(
                    name=tool_name,
                    params={"session_id": session_id},
                    reason=f"Get {tool_name} for search result session {session_id}"
                )
            )

        # Fallback: respond anyway
        logger.info(f"[Agent] No retrieval action needed, falling back to respond")
        return AgentAction(action_type="respond")

    def _get_session_to_retrieve(self, analysis: dict, query: str) -> int:
        """
        Determine which session to retrieve data for next.

        Prioritizes:
        1. Sessions from search results
        2. Top sessions by collaboration score
        3. Default fallback
        """
        retrieved = analysis.get('retrieved_sessions', set())

        # For superlative queries, use sessions by score priority
        # These are ordered by collaboration score (highest first)
        top_sessions_by_score = [24, 21, 23, 20, 26, 19, 22, 18, 25]

        for session_id in top_sessions_by_score:
            if session_id not in retrieved:
                return session_id

        # Fallback
        return 24  # Country Music (highest collaboration score)

    def _mentions_retrieval_intent(self, content: str) -> bool:
        """
        Detect if LLM intended to make a tool call but didn't format it properly.

        This catches the "Please hold on while I retrieve..." pattern that
        indicates the LLM wanted to call a tool but generated text instead.
        """
        intent_phrases = [
            "i'll retrieve", "i'll get", "let me get", "let me retrieve",
            "i'll call", "please hold", "hold on while i", "i will retrieve",
            "i will get", "let me fetch", "i'll fetch",
            "proceed with retrieving", "let me proceed", "i'll now call",
            "i will now call", "let me now call", "let's start by retrieving"
        ]
        content_lower = content.lower()
        return any(phrase in content_lower for phrase in intent_phrases)

    def _parse_tool_call_from_text(self, text: str) -> Optional[ToolCall]:
        """Parse a tool call from text format."""

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
        """
        Get a sensible default tool call based on query classification.

        This is key for Problem 2 (rare search_sessions usage).
        The classification determines whether to use search_sessions vs list_sessions.
        """
        classification = self._classify_query(query)
        query_lower = query.lower()

        logger.info(f"[Agent] Query classification: type={classification.query_type}, "
                   f"requires_search={classification.requires_search}, "
                   f"topic={classification.topic}, "
                   f"required_sessions={classification.required_sessions}")

        # THEMATIC or UNKNOWN: Use search_sessions for semantic discovery
        if classification.requires_search and classification.topic:
            return ToolCall(
                name="search_sessions",
                params={"query": classification.topic, "top_k": 5},
                reason=f"Search for sessions about '{classification.topic}'"
            )

        # STRUCTURAL or SUPERLATIVE: Use list_sessions for overview
        if classification.query_type in ['structural', 'superlative']:
            return ToolCall(
                name="list_sessions",
                params={},
                reason="Get overview of all sessions for comparison"
            )

        # HYPOTHESIS without explicit sessions: Search first
        if classification.query_type == 'hypothesis' and classification.requires_search:
            topic = classification.topic or self._extract_likely_topic(query)
            return ToolCall(
                name="search_sessions",
                params={"query": topic, "top_k": 5},
                reason=f"Search for sessions relevant to hypothesis about '{topic}'"
            )

        # COMPARISON/HYPOTHESIS with explicit sessions: Get data for first session
        if classification.required_sessions:
            session_id = list(classification.required_sessions)[0]
            tool_name = self._select_artifact_tool(query)
            return ToolCall(
                name=tool_name,
                params={"session_id": session_id},
                reason=f"Get {tool_name} for session {session_id}"
            )

        # SPEAKER queries: Use get_speaker_profile or search
        if classification.query_type == 'speaker':
            # Try to extract speaker name
            speaker_match = re.search(r"(\w+)'s?\s+(?:style|pattern|engagement|contribution)", query_lower)
            if speaker_match:
                speaker_name = speaker_match.group(1)
                return ToolCall(
                    name="get_speaker_profile",
                    params={"speaker_name": speaker_name},
                    reason=f"Get profile for speaker '{speaker_name}'"
                )
            # Fallback to list_sessions
            return ToolCall(
                name="list_sessions",
                params={},
                reason="Get sessions to find speaker information"
            )

        # Default fallback: search_sessions with extracted topic
        topic = self._extract_likely_topic(query)
        return ToolCall(
            name="search_sessions",
            params={"query": topic, "top_k": 5},
            reason=f"Search for sessions about '{topic}'"
        )

    def _select_artifact_tool(self, query: str) -> str:
        """Select appropriate artifact tool based on query content."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ['collaborat', '7c', 'engagement', 'quality', 'score', 'constructive', 'conflict', 'climate']):
            return "get_7c_analysis"
        elif any(kw in query_lower for kw in ['concept', 'idea', 'problem', 'solution', 'goal', 'connect', 'map', 'node']):
            return "get_concept_map"
        else:
            return "get_transcript"

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
