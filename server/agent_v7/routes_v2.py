"""
API Routes for BLINC Agent V7 (Simplified Architecture)

Uses the new ReAct-based agent with:
- Conversation memory
- Scaffolded responses
- User steering compliance
"""

import logging
import uuid
from flask import Blueprint, request, jsonify, session

from .graph_v2 import invoke_agent, reset_conversation
from .memory import get_memory, clear_memory

logger = logging.getLogger(__name__)

AGENT_VERSION = 'v7.2'  # Pure ReAct architecture - no classifier/exploratory path

# Create blueprint
agent_v7_bp = Blueprint('agent_v7', __name__, url_prefix='/api/v7/agent')

# Default user ID for unauthenticated requests (user id 1 = llmblinc)
DEFAULT_USER_ID = 1


def _get_user_id():
    """Get user ID from session or use default."""
    user_dict = session.get('user')
    if user_dict and user_dict.get('id'):
        return user_dict['id']
    return DEFAULT_USER_ID


def _save_conversation_to_db(conversation_id: str, query: str, result: dict,
                              user_id: int, session_device_id: int = None):
    """Save conversation and messages to database."""
    try:
        import database as db_helper

        # Check if conversation exists
        existing = db_helper.get_agent_conversations(conversation_id=conversation_id)

        if not existing:
            # Create new conversation with the specific ID
            from tables.agent_conversation import AgentConversation
            from app import db
            title = query[:50] + "..." if len(query) > 50 else query
            conv = AgentConversation(
                user_id=user_id,
                session_device_id=session_device_id,
                title=title,
                agent_version=AGENT_VERSION
            )
            conv.id = conversation_id
            db.session.add(conv)
            db.session.commit()
            logger.debug(f"Created new conversation {conversation_id[:8]}")
        else:
            # Update last_active timestamp
            existing.touch()
            from app import db
            db.session.commit()

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
            content=result.get('answer', ''),
            citations=[],  # Could extract from evidence
            tools_used=[tc.get('name') for tc in result.get('tool_calls', [])],
            reasoning_trace=[],
            confidence=0.8  # Default confidence
        )

        logger.debug(f"Saved conversation {conversation_id[:8]} to database")

    except Exception as e:
        logger.error(f"Error saving conversation to database: {e}")
        # Don't fail the request if DB save fails


@agent_v7_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'agent_version': AGENT_VERSION,
        'architecture': 'react_scaffolding',
        'features': [
            'react_agent_loop',
            'conversation_memory',
            'scaffolded_responses',
            'user_steering',
            'multi_turn_context'
        ]
    })


@agent_v7_bp.route('/query', methods=['POST'])
def query():
    """
    Main query endpoint for the Scaffolding Agent.

    Request body:
    {
        "query": "Your question here",
        "conversation_id": "optional-id",
        "session_device_id": optional_int,
        "preferred_representations": ["concept_map"],  // optional
        "exclude_representations": ["7c"],  // optional
        "include_debug": false  // optional
    }

    Response:
    {
        "answer": "The agent's scaffolded response",
        "confidence": 0.0-1.0,
        "citations": [...],
        "tools_used": [...],
        "follow_up_suggestions": [...],
        "conversation_id": "uuid",
        "success": true/false,
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

        # Extract session focus if provided
        session_focus = data.get('session_device_id')

        # Extract user steering preferences
        preferred_representations = data.get('preferred_representations', [])
        exclude_representations = data.get('exclude_representations', [])

        if preferred_representations or exclude_representations:
            logger.info(f"[{conversation_id[:8]}] User steering: prefer={preferred_representations}, exclude={exclude_representations}")

        logger.info(f"[Agent V7.1] Query: '{query_text[:50]}...' (conversation={conversation_id[:8]})")

        # Run the agent
        result = invoke_agent(
            query=query_text,
            conversation_id=conversation_id,
            session_focus=session_focus,
            preferred_representations=preferred_representations,
            exclude_representations=exclude_representations,
        )

        # Save to database for persistence
        user_id = _get_user_id()
        _save_conversation_to_db(
            conversation_id=conversation_id,
            query=query_text,
            result=result,
            user_id=user_id,
            session_device_id=session_focus
        )

        # Extract tools used from tool_calls AND auto-fetched tools from evidence
        tools_used = [tc.get('name') for tc in result.get('tool_calls', [])]
        # Also include auto-fetched tools from evidence
        for e in result.get('evidence', []):
            if e.get('auto_fetched') and e.get('tool'):
                tools_used.append(e.get('tool'))

        # Format response (maintain API compatibility)
        response = {
            'answer': result.get('answer', ''),
            'confidence': 0.8 if not result.get('error') else 0.0,
            'citations': _extract_citations(result.get('evidence', [])),
            'tools_used': tools_used,
            'follow_up_suggestions': result.get('suggestions', []),
            'conversation_id': conversation_id,
            'success': result.get('error') is None,
            'needs_clarification': False,
            'error': result.get('error'),

            # Context for multi-turn
            'session_focus': result.get('session_focus'),
            'speaker_focus': result.get('speaker_focus'),

            # Debug info (optional)
            'debug': {
                'tool_calls': result.get('tool_calls', []),
                'evidence_count': len(result.get('evidence', [])),
                'processing_time_ms': result.get('processing_time_ms', 0),
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


def _extract_citations(evidence: list) -> list:
    """Extract citations from evidence for response."""
    citations = []

    for e in evidence:
        tool = e.get('tool', '')
        result = e.get('result', {})

        if tool == 'get_transcript':
            discussion_id = result.get('discussion_id')
            session_name = result.get('session_name', '')
            utterances = result.get('utterances', [])
            if utterances:
                citations.append({
                    'type': 'transcript',
                    'discussion_id': discussion_id,
                    'session_name': session_name,
                    'count': len(utterances)
                })

        elif tool == 'get_concept_map':
            discussion_id = result.get('discussion_id')
            summary = result.get('summary', {})
            citations.append({
                'type': 'concept_map',
                'discussion_id': discussion_id,
                'nodes': summary.get('total_nodes', 0),
                'edges': summary.get('total_edges', 0)
            })

        elif tool == 'get_7c_analysis':
            discussion_id = result.get('discussion_id')
            summary = result.get('summary', {})
            citations.append({
                'type': 'collaboration',
                'discussion_id': discussion_id,
                'overall_score': summary.get('overall_score')
            })

    return citations


@agent_v7_bp.route('/context', methods=['GET'])
def get_context():
    """Get stored conversation context from memory."""
    conversation_id = request.args.get('conversation_id')

    if not conversation_id:
        return jsonify({'error': 'conversation_id required'}), 400

    memory = get_memory(conversation_id)

    return jsonify({
        'conversation_id': conversation_id,
        'context': {
            'session_focus': memory.session_focus,
            'session_name': memory.session_name,
            'speaker_focus': memory.speaker_focus,
            'turn_count': memory.turn_count,
            'artifacts_retrieved': [
                {
                    'type': a.artifact_type,
                    'discussion_id': a.session_id,  # session_id in memory maps to discussion_id in API
                    'turn': a.turn_number
                }
                for a in memory.artifacts_retrieved[-10:]
            ],
            'user_steering': memory.user_steering,
        }
    })


@agent_v7_bp.route('/context', methods=['DELETE'])
def clear_context():
    """Clear conversation context."""
    conversation_id = request.args.get('conversation_id')

    if conversation_id:
        clear_memory(conversation_id)
        return jsonify({'success': True, 'message': f'Cleared context for {conversation_id}'})
    else:
        from .memory import clear_all_memories
        clear_all_memories()
        return jsonify({'success': True, 'message': 'Cleared all contexts'})


@agent_v7_bp.route('/tools', methods=['GET'])
def list_tools():
    """List available tools."""
    from .tools_v2 import CORE_TOOLS

    tools = []
    for name, func in CORE_TOOLS.items():
        doc = func.__doc__ or ''
        description = doc.split('\n')[0].strip() if doc else name
        tools.append({
            'name': name,
            'description': description
        })

    return jsonify({
        'tools': tools,
        'count': len(tools)
    })


@agent_v7_bp.route('/memory', methods=['GET'])
def get_memory_state():
    """Get full memory state for debugging."""
    conversation_id = request.args.get('conversation_id')

    if not conversation_id:
        return jsonify({'error': 'conversation_id required'}), 400

    memory = get_memory(conversation_id)

    return jsonify({
        'conversation_id': conversation_id,
        'memory': memory.to_dict()
    })


@agent_v7_bp.route('/reset', methods=['POST'])
def reset():
    """Reset conversation and memory."""
    data = request.get_json() or {}
    conversation_id = data.get('conversation_id')

    if conversation_id:
        reset_conversation(conversation_id)
        return jsonify({'success': True, 'message': f'Reset conversation {conversation_id}'})
    else:
        return jsonify({'error': 'conversation_id required'}), 400


# =============================================================================
# CONVERSATION PERSISTENCE ENDPOINTS (for left panel)
# =============================================================================

@agent_v7_bp.route('/conversations', methods=['GET'])
def list_conversations():
    """
    List all conversations for the left panel.
    Uses database storage for persistence.
    """
    try:
        import database as db_helper
        user_id = _get_user_id()

        # Get conversations from database (filtered by agent version v7.x, no limit)
        conversations = db_helper.get_agent_conversations(user_id=user_id, agent_version=AGENT_VERSION, limit=None)

        # Format for frontend
        formatted = []
        for conv in conversations:
            formatted.append({
                'id': conv.id,
                'conversation_id': conv.id,
                'title': conv.title or 'Conversation',
                'created_at': conv.created_at.isoformat() if conv.created_at else None,
                'updated_at': conv.last_active.isoformat() if conv.last_active else None
            })

        return jsonify({
            'conversations': formatted,
            'count': len(formatted)
        })

    except Exception as e:
        logger.error(f"List conversations error: {e}")
        return jsonify({'conversations': [], 'count': 0, 'error': str(e)})


@agent_v7_bp.route('/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """
    Get a specific conversation with its messages.
    """
    try:
        import database as db_helper

        # Get conversation
        conv = db_helper.get_agent_conversations(conversation_id=conversation_id)
        if not conv:
            return jsonify({
                'conversation_id': conversation_id,
                'error': 'Conversation not found',
                'messages': []
            }), 404

        # Get messages
        messages = db_helper.get_agent_messages(conversation_id=conversation_id)

        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'id': msg.id,
                'role': msg.role,
                'content': msg.content,
                'citations': msg.citations,
                'tools_used': msg.tools_used,
                'created_at': msg.created_at.isoformat() if msg.created_at else None
            })

        return jsonify({
            'conversation_id': conversation_id,
            'title': conv.title or 'Conversation',
            'messages': formatted_messages,
            'created_at': conv.created_at.isoformat() if conv.created_at else None,
            'updated_at': conv.last_active.isoformat() if conv.last_active else None
        })

    except Exception as e:
        logger.error(f"Get conversation error: {e}")
        return jsonify({'error': str(e)}), 500


@agent_v7_bp.route('/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation and its messages."""
    try:
        import database as db_helper

        # Delete from database
        db_helper.delete_agent_conversation(conversation_id)

        # Clear in-memory context
        clear_memory(conversation_id)

        return jsonify({'success': True, 'message': f'Deleted conversation {conversation_id}'})

    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        return jsonify({'error': str(e)}), 500


@agent_v7_bp.route('/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation explicitly."""
    try:
        import database as db_helper

        data = request.get_json() or {}
        title = data.get('title', 'New Conversation')
        user_id = _get_user_id()

        # Create in database
        conv = db_helper.create_agent_conversation(
            user_id=user_id,
            title=title,
            agent_version=AGENT_VERSION
        )

        return jsonify({
            'conversation_id': conv.id,
            'title': title,
            'created': True
        })

    except Exception as e:
        logger.error(f"Create conversation error: {e}")
        return jsonify({'error': str(e)}), 500
