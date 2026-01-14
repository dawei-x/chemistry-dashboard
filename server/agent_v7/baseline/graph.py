"""
LangGraph Workflow for Baseline Agent (Transcript-Only)

AIED 2026 Comparison Baseline
=============================
Uses the SAME workflow structure as Agent V7 but with:
- Baseline tool registry (transcript-only)
- Baseline tool descriptions in prompts
- Tool execution validation to reject non-baseline tools

This ensures fair comparison - same reasoning capability, different data access.
"""

import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from ..state import AgentState, create_initial_state
from ..nodes import (
    process_input,
    synthesize,
    reflect,
    format_response,
    decompose_query,
    targeted_retrieve,
    reason_across_representations,
    synthesize_grounded_response
)
from ..nodes.query_router import execute_fast_path
from ..nodes.verify_claims import verify_claims

from .tools import BASELINE_TOOLS, BASELINE_TOOL_NAMES, EXCLUDED_TOOLS

logger = logging.getLogger(__name__)


# =============================================================================
# BASELINE-SPECIFIC NODE WRAPPERS
# =============================================================================

def execute_fast_path_baseline(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute fast path with baseline tool validation.

    Wraps the standard fast path but:
    1. Validates tools are in BASELINE_TOOLS
    2. Rejects concept map, 7C, LIWC tools
    """
    # Get the planned tool from state
    planned_tool = state.get('current_tool', '')

    # Check if tool is allowed in baseline
    if planned_tool and planned_tool not in BASELINE_TOOL_NAMES:
        logger.warning(f"[BASELINE] Blocked tool '{planned_tool}' - not in baseline registry")

        # Map to baseline equivalent or return error
        if planned_tool in ['get_concept_map', 'get_7c_analysis', 'get_liwc_metrics']:
            return {
                'final_answer': f"This baseline agent only has access to transcript data. "
                               f"The requested analysis ({planned_tool}) requires additional artifacts "
                               f"that are not available in this configuration.",
                'confidence': 0.3,
                'tools_used': [],
                'error': f"Tool '{planned_tool}' not available in baseline"
            }

        # For speaker profile, suggest the baseline alternative
        if planned_tool == 'get_speaker_profile':
            state = dict(state)
            state['current_tool'] = 'get_speaker_utterances'
            logger.info("[BASELINE] Replaced get_speaker_profile with get_speaker_utterances")

    # Execute with baseline tools
    return _execute_with_baseline_tools(state)


def _execute_with_baseline_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool using baseline registry."""
    tool_name = state.get('current_tool', '')
    tool_input = state.get('current_tool_input', {})

    if not tool_name:
        return state

    # Get tool from baseline registry
    tool_fn = BASELINE_TOOLS.get(tool_name)

    if not tool_fn:
        logger.error(f"[BASELINE] Tool '{tool_name}' not found in baseline registry")
        return {
            **state,
            'retrieval_results': [{
                'tool_name': tool_name,
                'error': f"Tool not available in baseline",
                'is_relevant': False
            }]
        }

    try:
        # Normalize input (convert session names to IDs if needed)
        normalized_input = _normalize_tool_input(tool_name, tool_input)

        # Execute tool
        result = tool_fn(**normalized_input)

        # Add to results
        current_results = state.get('retrieval_results', [])
        if not isinstance(current_results, list):
            current_results = []

        return {
            **state,
            'retrieval_results': current_results + [result],
            'tools_used': state.get('tools_used', []) + [tool_name]
        }

    except Exception as e:
        logger.error(f"[BASELINE] Tool execution error: {e}")
        return {
            **state,
            'retrieval_results': [{
                'tool_name': tool_name,
                'error': str(e),
                'is_relevant': False
            }]
        }


def _normalize_tool_input(tool_name: str, tool_input: dict) -> dict:
    """Normalize tool input parameters."""
    result = dict(tool_input)

    # Handle session_id normalization
    if 'session_id' in result:
        result['session_id'] = _normalize_session_id(result['session_id'])

    return result


def _normalize_session_id(value):
    """Convert session name to ID if needed."""
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            # Try name lookup
            from ..nodes.input_processor import get_session_patterns
            patterns = get_session_patterns()
            lower = value.lower()
            if lower in patterns:
                return patterns[lower]
    return value


def targeted_retrieve_baseline(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute targeted retrieval with baseline tool constraints.

    Wraps the standard targeted_retrieve but uses baseline tools.
    """
    from ..nodes.representation_planner import get_next_retrieval_step, plan_retrieval

    # === DETERMINISTIC PLANNING ===
    if not state.get('retrieval_plans'):
        logger.info("[BASELINE] Creating retrieval plans")
        plan_updates = plan_retrieval(state)
        state = {**state, **plan_updates}

    # Get next step to execute
    next_step = get_next_retrieval_step(state)

    if not next_step:
        logger.info("[BASELINE] All retrieval complete")
        return {
            'pras_stage': 'retrieve_complete',
            'next_action': 'reason'
        }

    subgoal = next_step['subgoal']
    step = next_step['step']
    subgoal_id = next_step['subgoal_id']

    tool_name = step.get('tool', 'list_sessions')
    params = step.get('parameters', {})

    # Map tool to baseline equivalent
    original_tool = tool_name
    if tool_name in EXCLUDED_TOOLS:
        if tool_name in ['get_concept_map', 'get_7c_analysis', 'get_liwc_metrics', 'search_concepts']:
            logger.info(f"[BASELINE] Skipping excluded tool: {tool_name}")
            return _mark_step_executed(state, subgoal_id, step, None)
        elif tool_name == 'get_speaker_profile':
            tool_name = 'get_speaker_utterances'
        else:
            tool_name = 'get_transcript'

    if tool_name != original_tool:
        logger.info(f"[BASELINE] Mapped {original_tool} -> {tool_name}")

    # Execute with baseline tools
    tool_fn = BASELINE_TOOLS.get(tool_name)
    if not tool_fn:
        logger.warning(f"[BASELINE] Tool {tool_name} not in baseline registry")
        return _mark_step_executed(state, subgoal_id, step, {'error': f'Tool {tool_name} not available'})

    try:
        normalized_params = _normalize_tool_input(tool_name, params)
        result = tool_fn(**normalized_params)
        logger.info(f"[BASELINE] Executed {tool_name}: {result.get('result_count', 0) if isinstance(result, dict) else 'n/a'} results")
        return _mark_step_executed(state, subgoal_id, step, result, tool_name)

    except Exception as e:
        logger.error(f"[BASELINE] Tool error: {e}")
        return _mark_step_executed(state, subgoal_id, step, {'error': str(e)})


def _mark_step_executed(state: Dict[str, Any], subgoal_id: str, step: dict,
                        result: Any, tool_name: str = None) -> Dict[str, Any]:
    """Mark a retrieval step as executed and update state."""
    subgoal_results = dict(state.get('subgoal_results', {}))

    if subgoal_id not in subgoal_results:
        subgoal_results[subgoal_id] = {
            'subgoal_id': subgoal_id,
            'steps_executed': [],
            'evidence': [],
            'representations_used': [],
            'satisfied': False,
            'reflection': None
        }

    # Use the structure expected by cross_rep_reasoner and grounded_synthesizer
    actual_tool = tool_name or step.get('tool')
    step_result = {
        'step': {
            'tool': actual_tool,
            'representation': step.get('representation', 'transcript'),
            'parameters': step.get('parameters', {}),
            'purpose': step.get('purpose', ''),
            'executed': True
        },
        'tool_result': result if result else {},
        'reflection': {}
    }
    subgoal_results[subgoal_id]['steps_executed'].append(step_result)

    # Track representation usage
    rep = step.get('representation', 'transcript')
    if rep and rep not in subgoal_results[subgoal_id].get('representations_used', []):
        subgoal_results[subgoal_id].setdefault('representations_used', []).append(rep)

    # Mark as satisfied if we got valid results
    if result and isinstance(result, dict) and not result.get('error'):
        subgoal_results[subgoal_id]['evidence'].append(result)
        subgoal_results[subgoal_id]['satisfied'] = True

    # Mark step as executed in retrieval_plans (dict keyed by subgoal_id)
    retrieval_plans = dict(state.get('retrieval_plans', {}))
    if subgoal_id in retrieval_plans:
        plan = retrieval_plans[subgoal_id]
        if isinstance(plan, dict):
            for s in plan.get('steps', []):
                if s.get('tool') == step.get('tool'):
                    s['executed'] = True
                    break

    tools_used = list(state.get('tools_used', []))
    if tool_name and tool_name not in tools_used:
        tools_used.append(tool_name)

    # Also populate retrieval_results for compatibility with synthesize node
    retrieval_results = list(state.get('retrieval_results', []))
    if result and isinstance(result, dict) and not result.get('error'):
        retrieval_results.append(result)

    return {
        'subgoal_results': subgoal_results,
        'retrieval_plans': retrieval_plans,
        'retrieval_results': retrieval_results,
        'tools_used': tools_used,
        'pras_iteration': state.get('pras_iteration', 0) + 1
    }


def decompose_query_baseline(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decompose query with baseline constraints.

    Modifies sub-goal generation to only request transcript representations.
    """
    # Call original decompose
    result = decompose_query(state)

    # Filter sub-goals to transcript only
    if 'sub_goals' in result:
        for sg in result.get('sub_goals', []):
            # Force primary representation to transcript
            if sg.get('primary_representation') not in ['transcript', 'discovery']:
                sg['primary_representation'] = 'transcript'

            # Remove non-transcript secondary representations
            sg['secondary_representations'] = [
                r for r in sg.get('secondary_representations', [])
                if r in ['transcript', 'discovery']
            ]

    return result


# =============================================================================
# GRAPH CREATION
# =============================================================================

def create_baseline_graph() -> StateGraph:
    """
    Create the baseline agent workflow graph.

    Same structure as Agent V7 but with baseline node wrappers.
    """
    graph = StateGraph(AgentState)

    # Input processing (shared)
    graph.add_node("process_input", process_input)

    # Query decomposition (baseline wrapper)
    graph.add_node("decompose", decompose_query_baseline)

    # Fast path (baseline wrapper)
    graph.add_node("fast_path", execute_fast_path_baseline)

    # PRAS Path with baseline retrieval
    graph.add_node("pras_retrieve", targeted_retrieve_baseline)
    graph.add_node("pras_reason", reason_across_representations)
    graph.add_node("pras_synthesize", synthesize_grounded_response)

    # Shared nodes (same as V3)
    graph.add_node("synthesize", synthesize)
    graph.add_node("verify", verify_claims)
    graph.add_node("reflect", reflect)
    graph.add_node("format", format_response)

    # === Define edges ===

    # Entry point
    graph.set_entry_point("process_input")
    graph.add_edge("process_input", "decompose")

    # Routing after decompose
    def route_after_decompose(state: Dict[str, Any]) -> str:
        path = state.get('query_path', 'pras')
        if path == 'fast':
            return 'fast_path'
        return 'pras_retrieve'

    graph.add_conditional_edges(
        "decompose",
        route_after_decompose,
        {
            'fast_path': 'fast_path',
            'pras_retrieve': 'pras_retrieve'
        }
    )

    # Fast path to synthesize
    graph.add_edge("fast_path", "synthesize")

    # PRAS path
    def route_pras_retrieval(state: Dict[str, Any]) -> str:
        # Check if we need more retrieval or can proceed to reasoning
        iteration = state.get('pras_iteration', 0)
        if iteration >= 2:  # Max 2 retrieval iterations
            return 'pras_reason'

        needs_more = state.get('needs_more_retrieval', False)
        if needs_more:
            return 'pras_retrieve'
        return 'pras_reason'

    graph.add_conditional_edges(
        "pras_retrieve",
        route_pras_retrieval,
        {
            'pras_retrieve': 'pras_retrieve',
            'pras_reason': 'pras_reason'
        }
    )

    graph.add_edge("pras_reason", "pras_synthesize")
    graph.add_edge("pras_synthesize", "synthesize")

    # Post-synthesis flow
    graph.add_edge("synthesize", "verify")
    graph.add_edge("verify", "reflect")
    graph.add_edge("reflect", "format")
    graph.add_edge("format", END)

    return graph


# Compiled graph singleton
_compiled_baseline_graph = None


def get_compiled_baseline_graph():
    """Get or create the compiled baseline graph."""
    global _compiled_baseline_graph
    if _compiled_baseline_graph is None:
        graph = create_baseline_graph()
        _compiled_baseline_graph = graph.compile()
        logger.info("Baseline agent graph compiled")
    return _compiled_baseline_graph


def run_baseline_agent(
    query: str,
    conversation_id: str,
    conversation_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run the baseline agent on a query.

    Args:
        query: User's query
        conversation_id: Unique conversation identifier
        conversation_context: Optional context from previous turns

    Returns:
        Agent response with answer, citations, etc.
    """
    logger.info(f"[BASELINE] Running baseline agent on query: '{query}'")

    # Create initial state
    initial_state = create_initial_state(
        query=query,
        conversation_id=conversation_id,
        conversation_context=conversation_context
    )

    # Mark as baseline mode (for logging/debugging)
    initial_state['agent_variant'] = 'baseline'

    # Force exclude non-transcript representations
    initial_state['exclude_representations'] = ['concept_map', '7c', 'liwc', 'speaker_profile']

    # Get compiled graph
    graph = get_compiled_baseline_graph()

    # Run the graph
    try:
        final_state = graph.invoke(initial_state)

        logger.info(f"[BASELINE] Completed: confidence={final_state.get('confidence', 0):.2f}")

        # Add baseline marker to response
        final_state['agent_variant'] = 'baseline'

        return final_state

    except Exception as e:
        logger.error(f"[BASELINE] Agent error: {e}", exc_info=True)

        return {
            'final_answer': f"I encountered an error processing your request: {str(e)}",
            'confidence': 0.0,
            'citations': [],
            'tools_used': [],
            'follow_ups': [],
            'success': False,
            'error': str(e),
            'agent_variant': 'baseline'
        }


__all__ = ['create_baseline_graph', 'run_baseline_agent', 'get_compiled_baseline_graph']
