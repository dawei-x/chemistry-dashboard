"""
Format Response Node for BLINC Agent V3

Formats the final response for the API.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def format_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the final response for the API.

    This node:
    1. Packages the answer with metadata
    2. Includes citations and follow-ups
    3. Preserves session context for multi-turn
    4. Handles clarification requests

    Args:
        state: Current agent state

    Returns:
        Final state with formatted response
    """
    logger.info("Formatting final response")

    # Check if this is a clarification request
    if state.get('next_action') == 'clarify':
        return _format_clarification(state)

    # Format normal response
    return {
        # The answer
        'final_answer': state.get('final_answer', ''),

        # Metadata
        'confidence': state.get('confidence', 0.0),
        'reflection': state.get('reflection', ''),

        # Evidence
        'citations': state.get('citations', []),
        'tools_used': state.get('tools_used', []),

        # Follow-ups
        'follow_ups': state.get('follow_ups', []),

        # Context to preserve for multi-turn
        'current_session_focus': state.get('current_session_focus'),
        'previous_session_focus': state.get('previous_session_focus'),
        'session_history': state.get('session_history', []),
        'compared_sessions': state.get('compared_sessions', []),
        'current_speaker_focus': state.get('current_speaker_focus'),

        # Debug info
        'thought_history': state.get('thought_history', []),
        'iteration_count': state.get('iteration_count', 0),
        'rewrite_count': state.get('rewrite_count', 0),

        # Status
        'success': True,
        'needs_clarification': False,
        'error': state.get('error')
    }


def _format_clarification(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format a clarification request response."""
    question = state.get('clarification_question', 'Could you please clarify your question?')
    options = state.get('clarification_options', [])

    # Build clarification answer
    answer = question
    if options:
        answer += "\n\nOptions:\n"
        for i, opt in enumerate(options, 1):
            answer += f"{i}. {opt}\n"

    return {
        'final_answer': answer,
        'confidence': 0.0,
        'citations': [],
        'tools_used': state.get('tools_used', []),
        'follow_ups': options,  # Options as follow-ups for easy selection

        # Context preserved
        'current_session_focus': state.get('current_session_focus'),
        'previous_session_focus': state.get('previous_session_focus'),
        'session_history': state.get('session_history', []),
        'compared_sessions': state.get('compared_sessions', []),
        'current_speaker_focus': state.get('current_speaker_focus'),

        # Status
        'success': True,
        'needs_clarification': True,
        'error': None
    }
