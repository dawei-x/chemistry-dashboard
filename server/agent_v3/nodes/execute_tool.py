"""
Execute Tool Node for BLINC Agent V3

Executes the tool selected by reason_and_act.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Session name to ID mapping for input normalization
SESSION_NAME_TO_ID = {
    'living in nyc': 18, 'nyc': 18, 'new york': 18,
    'is ai alive': 19, 'ai alive': 19, 'ai': 19,
    'nuclear fusion': 20, 'fusion': 20,
    'shaw interview': 21, 'shaw': 21,
    'collaboration literacy': 22, 'literacy': 22,
    'dinosaurs': 23, 'dinosaur': 23,
    'country music': 24, 'country': 24, 'music': 24,
    'abundance': 25
}


def _normalize_session_id(value):
    """Convert session name to ID if needed."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # Try to parse as int first
        try:
            return int(value)
        except ValueError:
            pass
        # Look up by name
        normalized = value.lower().strip()
        if normalized in SESSION_NAME_TO_ID:
            return SESSION_NAME_TO_ID[normalized]
        # Try partial match
        for name, sid in SESSION_NAME_TO_ID.items():
            if name in normalized or normalized in name:
                return sid
    return value  # Return as-is if we can't normalize


def _normalize_tool_input(tool_name: str, tool_input: dict) -> dict:
    """Normalize tool input - convert session names to IDs, etc."""
    if not tool_input:
        return tool_input

    normalized = tool_input.copy()

    # Normalize session_id parameter
    if 'session_id' in normalized:
        normalized['session_id'] = _normalize_session_id(normalized['session_id'])

    # Normalize session_ids list
    if 'session_ids' in normalized and isinstance(normalized['session_ids'], list):
        normalized['session_ids'] = [
            _normalize_session_id(sid) for sid in normalized['session_ids']
        ]

    return normalized


def execute_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the tool selected by the reasoning node.

    Args:
        state: Current agent state with current_tool and current_tool_input

    Returns:
        Updated state with tool results
    """
    tool_name = state.get('current_tool')
    tool_input = state.get('current_tool_input', {})

    # Normalize input - convert session names to IDs, etc.
    tool_input = _normalize_tool_input(tool_name, tool_input)

    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

    if not tool_name:
        logger.warning("No tool specified for execution")
        return {
            'next_action': 'continue'
        }

    try:
        # Import tools
        from ..tools import ALL_TOOLS

        if tool_name not in ALL_TOOLS:
            logger.error(f"Unknown tool: {tool_name}")
            return {
                'retrieval_results': [{
                    'tool_name': tool_name,
                    'error': f"Unknown tool: {tool_name}",
                    'result_count': 0,
                    'results': [],
                    'is_relevant': False
                }],
                'next_action': 'continue'
            }

        # Get and execute the tool
        tool_fn = ALL_TOOLS[tool_name]
        result = tool_fn(**tool_input)

        # Track tools used
        tools_used = state.get('tools_used', []).copy()
        if tool_name not in tools_used:
            tools_used.append(tool_name)

        # Handle special cases
        if tool_name == 'think':
            # Thinking doesn't produce retrieval results
            return {
                'tools_used': tools_used,
                'current_thought': result.get('thought', ''),
                'next_action': 'continue'
            }

        if tool_name == 'clarify':
            # Clarification needs special handling
            return {
                'tools_used': tools_used,
                'next_action': 'clarify',
                'clarification_question': result.get('question'),
                'clarification_options': result.get('options', [])
            }

        # Normal tool result
        return {
            'retrieval_results': [result],
            'tools_used': tools_used,
            'next_action': 'grade' if result.get('result_count', 0) > 0 else 'continue'
        }

    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return {
            'retrieval_results': [{
                'tool_name': tool_name,
                'error': str(e),
                'result_count': 0,
                'results': [],
                'is_relevant': False
            }],
            'next_action': 'continue'
        }
