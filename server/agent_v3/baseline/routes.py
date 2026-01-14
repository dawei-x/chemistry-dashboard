"""
API Routes for Baseline Agent (Transcript-Only)

AIED 2026 Comparison Baseline
=============================
Endpoint: /api/v3/agent/baseline/query

Same interface as the full Agent V3 but uses transcript-only tools.
Responses include 'agent_variant': 'baseline' for benchmarking.
"""

import logging
import uuid
from flask import Blueprint, request, jsonify, session

from .graph import run_baseline_agent
from ..nodes.input_processor import should_reset_context

logger = logging.getLogger(__name__)

# Create blueprint
baseline_bp = Blueprint('agent_v3_baseline', __name__, url_prefix='/api/v3/agent/baseline')

# In-memory conversation context storage (separate from full agent)
_baseline_contexts = {}

# Default user ID for unauthenticated requests
DEFAULT_USER_ID = 1


def _get_user_id():
    """Get user ID from session or use default."""
    user_dict = session.get('user')
    if user_dict and user_dict.get('id'):
        return user_dict['id']
    return DEFAULT_USER_ID


def _save_conversation_to_db(conversation_id: str, query: str, result: dict,
                              user_id: int, session_device_id: int = None):
    """Save baseline conversation to database."""
    try:
        import database as db_helper

        # Check if conversation exists
        existing = db_helper.get_agent_conversations(conversation_id=conversation_id)

        if not existing:
            from tables.agent_conversation import AgentConversation
            from app import db

            # Mark title as baseline conversation
            title = f"[Baseline] {query[:40]}..." if len(query) > 40 else f"[Baseline] {query}"
            conv = AgentConversation(
                user_id=user_id,
                session_device_id=session_device_id,
                title=title
            )
            conv.id = conversation_id
            db.session.add(conv)
            db.session.commit()
            logger.debug(f"[BASELINE] Created conversation {conversation_id[:8]}")

        # Save user message
        db_helper.add_agent_message(
            conversation_id=conversation_id,
            role='user',
            content=query
        )

        # Save assistant message
        db_helper.add_agent_message(
            conversation_id=conversation_id,
            role='assistant',
            content=result.get('final_answer', ''),
            citations=result.get('citations', []),
            tools_used=result.get('tools_used', []),
            reasoning_trace=result.get('thought_history', []),
            confidence=result.get('confidence', 0.0)
        )

        logger.debug(f"[BASELINE] Saved conversation {conversation_id[:8]} to database")

    except Exception as e:
        logger.error(f"[BASELINE] Error saving conversation: {e}")


@baseline_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for baseline agent."""
    return jsonify({
        'status': 'healthy',
        'agent_version': 'v3-baseline',
        'agent_variant': 'baseline',
        'features': [
            'transcript_only',
            'no_concept_map',
            'no_7c_analysis',
            'no_liwc_metrics'
        ],
        'purpose': 'AIED 2026 comparison baseline',
        'description': 'Same reasoning architecture as full agent, but restricted to transcript data only'
    })


@baseline_bp.route('/query', methods=['POST'])
def query():
    """
    Baseline query endpoint.

    Same interface as /api/v3/agent/query but uses transcript-only tools.

    Request body:
    {
        "query": "Your question here",
        "conversation_id": "optional-id",
        "session_device_id": optional_int
    }

    Response includes 'agent_variant': 'baseline' for benchmarking.
    """
    try:
        data = request.get_json()

        query_text = data.get('query', '').strip()
        if not query_text:
            return jsonify({'error': 'Query is required'}), 400

        # Get or create conversation ID
        conversation_id = data.get('conversation_id') or str(uuid.uuid4())

        # Get conversation context
        context = _baseline_contexts.get(conversation_id, {}).copy()

        # Check for context reset
        if should_reset_context(query_text):
            if context.get('current_session_focus'):
                context['previous_session_focus'] = context['current_session_focus']
            context['current_session_focus'] = None
            logger.info(f"[BASELINE] Context reset for {conversation_id[:8]}")

        # Add session context if provided
        if data.get('session_device_id'):
            context['current_session_focus'] = data['session_device_id']

        logger.info(f"[BASELINE] Query: '{query_text}' (conversation={conversation_id[:8]})")

        # Run the baseline agent
        result = run_baseline_agent(
            query=query_text,
            conversation_id=conversation_id,
            conversation_context=context
        )

        # Update conversation context
        _baseline_contexts[conversation_id] = {
            'current_session_focus': result.get('current_session_focus'),
            'previous_session_focus': result.get('previous_session_focus'),
            'session_history': result.get('session_history', []),
            'compared_sessions': result.get('compared_sessions', []),
            'current_speaker_focus': result.get('current_speaker_focus')
        }

        # Save to database
        user_id = _get_user_id()
        session_device_id = data.get('session_device_id')
        _save_conversation_to_db(
            conversation_id=conversation_id,
            query=query_text,
            result=result,
            user_id=user_id,
            session_device_id=session_device_id
        )

        # Format response (same structure as full agent + variant marker)
        response = {
            'answer': result.get('final_answer', ''),
            'confidence': result.get('confidence', 0.0),
            'citations': result.get('citations', []),
            'tools_used': result.get('tools_used', []),
            'follow_up_suggestions': result.get('follow_ups', []),
            'conversation_id': conversation_id,
            'success': result.get('success', True),
            'needs_clarification': result.get('needs_clarification', False),
            'error': result.get('error'),

            # Baseline-specific marker for benchmarking
            'agent_variant': 'baseline',

            # Debug info (optional)
            'debug': {
                'iterations': result.get('iteration_count', 0),
                'thoughts': result.get('thought_history', [])
            } if data.get('include_debug') else None
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"[BASELINE] Query error: {e}", exc_info=True)
        return jsonify({
            'answer': f"An error occurred: {str(e)}",
            'confidence': 0.0,
            'citations': [],
            'tools_used': [],
            'follow_up_suggestions': [],
            'success': False,
            'error': str(e),
            'agent_variant': 'baseline'
        }), 500


@baseline_bp.route('/context', methods=['GET'])
def get_context():
    """Get stored baseline conversation context."""
    conversation_id = request.args.get('conversation_id')

    if not conversation_id:
        return jsonify({'error': 'conversation_id required'}), 400

    context = _baseline_contexts.get(conversation_id, {})

    return jsonify({
        'conversation_id': conversation_id,
        'context': context,
        'agent_variant': 'baseline'
    })


@baseline_bp.route('/context', methods=['DELETE'])
def clear_context():
    """Clear baseline conversation context."""
    conversation_id = request.args.get('conversation_id')

    if conversation_id:
        _baseline_contexts.pop(conversation_id, None)
        return jsonify({
            'message': f'Context cleared for {conversation_id}',
            'agent_variant': 'baseline'
        })
    else:
        _baseline_contexts.clear()
        return jsonify({
            'message': 'All baseline contexts cleared',
            'agent_variant': 'baseline'
        })


@baseline_bp.route('/tools', methods=['GET'])
def list_tools():
    """List available baseline tools."""
    from .tools import BASELINE_TOOL_NAMES, EXCLUDED_TOOLS

    return jsonify({
        'agent_variant': 'baseline',
        'available_tools': list(BASELINE_TOOL_NAMES),
        'excluded_tools': list(EXCLUDED_TOOLS),
        'description': 'Transcript-only tools for AIED 2026 baseline comparison'
    })


__all__ = ['baseline_bp']
