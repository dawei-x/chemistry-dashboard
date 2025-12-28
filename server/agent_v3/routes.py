"""
API Routes for BLINC Agent V3

Clean API endpoints for the Ultra Agent.
"""

import logging
import uuid
from flask import Blueprint, request, jsonify

from .graph import run_agent

logger = logging.getLogger(__name__)

# Create blueprint
agent_v3_bp = Blueprint('agent_v3', __name__, url_prefix='/api/v3/agent')

# In-memory conversation context storage
# TODO: Move to Redis for production
_conversation_contexts = {}


@agent_v3_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'agent_version': 'v3',
        'features': [
            'intelligent_reasoning',
            'self_reflective_rag',
            'query_rewriting',
            'multi_turn_context'
        ]
    })


@agent_v3_bp.route('/query', methods=['POST'])
def query():
    """
    Main query endpoint for the Ultra Agent.

    Request body:
    {
        "query": "Your question here",
        "conversation_id": "optional-id",
        "session_device_id": optional_int
    }

    Response:
    {
        "answer": "The agent's response",
        "confidence": 0.0-1.0,
        "citations": [...],
        "tools_used": [...],
        "follow_up_suggestions": [...],
        "conversation_id": "uuid",
        "success": true/false,
        "needs_clarification": false,
        "error": null
    }
    """
    try:
        data = request.get_json()

        query_text = data.get('query', '').strip()
        if not query_text:
            return jsonify({'error': 'Query is required'}), 400

        # Get or create conversation ID
        conversation_id = data.get('conversation_id') or str(uuid.uuid4())

        # Get conversation context
        context = _conversation_contexts.get(conversation_id, {})

        # Add session context if provided
        if data.get('session_device_id'):
            context['current_session_focus'] = data['session_device_id']

        logger.info(f"Agent V3 query: '{query_text}' (conversation={conversation_id})")

        # Run the agent
        result = run_agent(
            query=query_text,
            conversation_id=conversation_id,
            conversation_context=context
        )

        # Update conversation context
        _conversation_contexts[conversation_id] = {
            'current_session_focus': result.get('current_session_focus'),
            'previous_session_focus': result.get('previous_session_focus'),
            'session_history': result.get('session_history', []),
            'compared_sessions': result.get('compared_sessions', []),
            'current_speaker_focus': result.get('current_speaker_focus')
        }

        # Format response
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

            # Debug info (optional)
            'debug': {
                'iterations': result.get('iteration_count', 0),
                'rewrites': result.get('rewrite_count', 0),
                'thoughts': result.get('thought_history', [])
            } if data.get('include_debug') else None
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return jsonify({
            'answer': f"An error occurred: {str(e)}",
            'confidence': 0.0,
            'citations': [],
            'tools_used': [],
            'follow_up_suggestions': [],
            'success': False,
            'error': str(e)
        }), 500


@agent_v3_bp.route('/context', methods=['GET'])
def get_context():
    """Get stored conversation context."""
    conversation_id = request.args.get('conversation_id')

    if not conversation_id:
        return jsonify({'error': 'conversation_id required'}), 400

    context = _conversation_contexts.get(conversation_id, {})

    return jsonify({
        'conversation_id': conversation_id,
        'context': context
    })


@agent_v3_bp.route('/context', methods=['DELETE'])
def clear_context():
    """Clear conversation context."""
    conversation_id = request.args.get('conversation_id')

    if conversation_id:
        _conversation_contexts.pop(conversation_id, None)
        return jsonify({'success': True, 'message': f'Cleared context for {conversation_id}'})
    else:
        _conversation_contexts.clear()
        return jsonify({'success': True, 'message': 'Cleared all contexts'})


@agent_v3_bp.route('/tools', methods=['GET'])
def list_tools():
    """List available tools."""
    from .prompts.tool_descriptions import TOOL_DESCRIPTIONS

    tools = []
    for name, info in TOOL_DESCRIPTIONS.items():
        tools.append({
            'name': name,
            'description': info['description'].strip()[:200],
            'parameters': list(info.get('parameters', {}).keys())
        })

    return jsonify({
        'tools': tools,
        'count': len(tools)
    })
