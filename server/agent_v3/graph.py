"""
LangGraph Workflow for BLINC Agent V3

A clean, intelligent agent workflow that relies on model reasoning
rather than keyword matching.

Workflow:
    START
      ↓
    process_input (resolve references)
      ↓
    ┌─────────────────────────────────────────┐
    │           REASONING LOOP                │
    │                                         │
    │  reason_and_act ←──────────────────┐   │
    │      ↓                              │   │
    │  ┌───┴────────────────────┐         │   │
    │  ↓           ↓            ↓         │   │
    │ execute  synthesize    clarify      │   │
    │  ↓                        ↓         │   │
    │ grade                  format       │   │
    │  ↓                        ↓         │   │
    │ ┌┴──────┐               END         │   │
    │ ↓       ↓                           │   │
    │ ok    rewrite ──────────────────────┘   │
    │ ↓                                       │
    │ └───────────────────────────────────┘   │
    │                                         │
    └─────────────────────────────────────────┘
      ↓
    synthesize
      ↓
    reflect
      ↓
    format_response
      ↓
    END
"""

import logging
from typing import Dict, Any, Literal

from langgraph.graph import StateGraph, END

from .state import AgentState, create_initial_state
from .nodes import (
    process_input,
    reason_and_act,
    execute_tool,
    grade_results,
    rewrite_query,
    synthesize,
    reflect,
    format_response
)

logger = logging.getLogger(__name__)


def create_agent_graph() -> StateGraph:
    """
    Create the Ultra Agent workflow graph.

    Returns:
        Configured StateGraph ready for compilation
    """
    # Create the graph with our state type
    graph = StateGraph(AgentState)

    # === Add nodes ===

    # Input processing
    graph.add_node("process_input", process_input)

    # Main reasoning loop
    graph.add_node("reason", reason_and_act)

    # Tool execution
    graph.add_node("execute", execute_tool)

    # Result grading (CRAG pattern)
    graph.add_node("grade", grade_results)

    # Query rewriting
    graph.add_node("rewrite", rewrite_query)

    # Synthesis and reflection
    graph.add_node("synthesize", synthesize)
    graph.add_node("reflect", reflect)

    # Final formatting
    graph.add_node("format", format_response)

    # === Add edges ===

    # Start -> process input
    graph.set_entry_point("process_input")

    # Process input -> reason
    graph.add_edge("process_input", "reason")

    # Reason -> conditional routing
    graph.add_conditional_edges(
        "reason",
        _route_after_reason,
        {
            "execute_tool": "execute",
            "synthesize": "synthesize",
            "clarify": "format",  # Clarification goes directly to format
            "continue": "reason"  # Continue reasoning (e.g., after think)
        }
    )

    # Execute -> conditional (grade or continue)
    graph.add_conditional_edges(
        "execute",
        _route_after_execute,
        {
            "grade": "grade",
            "continue": "reason",
            "clarify": "format"
        }
    )

    # Grade -> conditional (continue or rewrite)
    graph.add_conditional_edges(
        "grade",
        _route_after_grade,
        {
            "continue": "reason",
            "rewrite": "rewrite"
        }
    )

    # Rewrite -> reason (retry with new query)
    graph.add_edge("rewrite", "reason")

    # Synthesize -> reflect
    graph.add_edge("synthesize", "reflect")

    # Reflect -> format
    graph.add_edge("reflect", "format")

    # Format -> END
    graph.add_edge("format", END)

    return graph


def _route_after_reason(state: Dict[str, Any]) -> str:
    """Route after the reasoning node."""
    next_action = state.get('next_action', 'continue')

    if next_action == 'execute_tool':
        return 'execute_tool'
    elif next_action == 'synthesize':
        return 'synthesize'
    elif next_action == 'clarify':
        return 'clarify'
    else:
        return 'continue'


def _route_after_execute(state: Dict[str, Any]) -> str:
    """Route after tool execution."""
    next_action = state.get('next_action', 'continue')

    if next_action == 'grade':
        return 'grade'
    elif next_action == 'clarify':
        return 'clarify'
    else:
        return 'continue'


def _route_after_grade(state: Dict[str, Any]) -> str:
    """Route after grading results."""
    next_action = state.get('next_action', 'continue')

    if next_action == 'rewrite':
        return 'rewrite'
    else:
        return 'continue'


# Compiled graph (singleton)
_compiled_graph = None


def get_compiled_graph():
    """Get the compiled graph (singleton)."""
    global _compiled_graph

    if _compiled_graph is None:
        graph = create_agent_graph()
        _compiled_graph = graph.compile()
        logger.info("Agent V3 graph compiled successfully")

    return _compiled_graph


def run_agent(
    query: str,
    conversation_id: str,
    conversation_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run the agent on a query.

    Args:
        query: User's query
        conversation_id: Unique conversation identifier
        conversation_context: Optional context from previous turns

    Returns:
        Agent response with answer, citations, etc.
    """
    logger.info(f"Running agent V3 on query: '{query}'")

    # Create initial state
    initial_state = create_initial_state(
        query=query,
        conversation_id=conversation_id,
        conversation_context=conversation_context
    )

    # Get compiled graph
    graph = get_compiled_graph()

    # Run the graph
    try:
        final_state = graph.invoke(initial_state)

        logger.info(f"Agent completed: confidence={final_state.get('confidence', 0):.2f}, "
                   f"iterations={final_state.get('iteration_count', 0)}")

        return final_state

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)

        return {
            'final_answer': f"I encountered an error processing your request: {str(e)}",
            'confidence': 0.0,
            'citations': [],
            'tools_used': [],
            'follow_ups': ['Try rephrasing your question'],
            'success': False,
            'error': str(e),
            'current_session_focus': conversation_context.get('current_session_focus') if conversation_context else None,
            'session_history': conversation_context.get('session_history', []) if conversation_context else []
        }
