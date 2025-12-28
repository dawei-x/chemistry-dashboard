"""
Agent State Definition for BLINC Agent V3

Clean, minimal state focused on reasoning rather than routing.
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from operator import add
from langchain_core.messages import BaseMessage


class RetrievalResult(TypedDict, total=False):
    """Result from a retrieval tool."""
    tool_name: str
    query_used: str
    results: List[Dict[str, Any]]
    relevance_scores: List[float]  # 0-1 scores from grading
    is_relevant: bool  # Overall relevance assessment
    result_count: int


class AgentState(TypedDict, total=False):
    """
    State for the Ultra Agent workflow.

    Designed for intelligent reasoning, not keyword routing.
    """

    # === Core Query ===
    original_query: str
    current_query: str  # May be rewritten

    # === Conversation Context ===
    conversation_id: str
    messages: Annotated[List[BaseMessage], add]

    # Session focus (for multi-turn)
    current_session_focus: Optional[int]
    previous_session_focus: Optional[int]
    session_history: List[int]
    compared_sessions: List[int]
    current_speaker_focus: Optional[str]

    # === Reasoning State ===
    # Current thought from the think tool
    current_thought: Optional[str]
    # Accumulated thoughts for transparency
    thought_history: List[str]

    # === Tool Execution ===
    # Current tool being called
    current_tool: Optional[str]
    current_tool_input: Optional[Dict[str, Any]]

    # All retrieval results (accumulated)
    retrieval_results: Annotated[List[RetrievalResult], add]

    # Tools used in this query
    tools_used: List[str]

    # === Self-Reflection ===
    # Number of query rewrites attempted
    rewrite_count: int
    # Maximum rewrites allowed
    max_rewrites: int

    # Document grading results
    grading_result: Optional[Dict[str, Any]]

    # === Control Flow ===
    iteration_count: int
    max_iterations: int

    # Next action: "continue", "rewrite", "synthesize", "clarify"
    next_action: str

    # === Output ===
    # Final synthesized answer
    final_answer: Optional[str]

    # Confidence in answer (0-1)
    confidence: float

    # Reflection on the answer
    reflection: Optional[str]

    # Citations for the answer
    citations: List[Dict[str, Any]]

    # Follow-up suggestions
    follow_ups: List[str]

    # Error if any
    error: Optional[str]


def create_initial_state(
    query: str,
    conversation_id: str,
    conversation_context: Optional[Dict] = None
) -> AgentState:
    """
    Create initial state for a new query.

    Args:
        query: User's query text
        conversation_id: Unique conversation identifier
        conversation_context: Optional context from previous turns
    """
    context = conversation_context or {}

    return AgentState(
        # Core
        original_query=query,
        current_query=query,

        # Conversation
        conversation_id=conversation_id,
        messages=[],

        # Session context (from previous turns)
        current_session_focus=context.get('current_session_focus'),
        previous_session_focus=context.get('previous_session_focus'),
        session_history=context.get('session_history', []),
        compared_sessions=context.get('compared_sessions', []),
        current_speaker_focus=context.get('current_speaker_focus'),

        # Reasoning
        current_thought=None,
        thought_history=[],

        # Tools
        current_tool=None,
        current_tool_input=None,
        retrieval_results=[],
        tools_used=[],

        # Self-reflection
        rewrite_count=0,
        max_rewrites=2,
        grading_result=None,

        # Control
        iteration_count=0,
        max_iterations=8,
        next_action="continue",

        # Output
        final_answer=None,
        confidence=0.0,
        reflection=None,
        citations=[],
        follow_ups=[],
        error=None
    )
