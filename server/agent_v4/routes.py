"""
API Routes for Agent V4.

Simple endpoints supporting:
- /query: Main query endpoint with mode switching
- /context: Get/clear conversation context
- /health: Health check
- /compare: A/B comparison for research
"""

import logging
import uuid
from flask import Blueprint, request, jsonify, session

from .agent import run_agent, conversation_manager
import database as db_helper

logger = logging.getLogger(__name__)

AGENT_VERSION = 'v4'


def _get_user_id():
    """Get user ID from session, default to 1 for testing."""
    return session.get('user_id', 1)

# Create blueprint
agent_v4_bp = Blueprint('agent_v4', __name__, url_prefix='/api/v4/agent')


def _save_conversation_to_db(conversation_id: str, query: str, result: dict, user_id: int):
    """Save conversation and messages to database for persistence."""
    try:
        # Check if conversation exists
        existing = db_helper.get_agent_conversations(conversation_id=conversation_id)

        if not existing:
            # Create new conversation with the specific ID
            from tables.agent_conversation import AgentConversation
            from app import db
            title = query[:50] + "..." if len(query) > 50 else query
            conv = AgentConversation(
                user_id=user_id,
                title=title,
                agent_version=AGENT_VERSION
            )
            conv.id = conversation_id  # Use the specific ID
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
            tools_used=result.get('tools_used', [])
        )

        logger.debug(f"Saved conversation {conversation_id[:8]} to database")

    except Exception as e:
        logger.error(f"Error saving conversation to database: {e}")
        # Don't fail the request if DB save fails


def _build_references(tool_results: list) -> list:
    """
    Build references from tool results for frontend display.

    Extracts key information from each tool call to show:
    - What data was retrieved (session, speaker, etc.)
    - A preview/summary
    - Link info for making it clickable

    This provides transparency without burdening the LLM.
    """
    references = []

    for tr in tool_results:
        tool = tr.get('tool', '')
        result = tr.get('result', {})

        ref = {
            'tool': tool,
            'type': 'data',
            'clickable': False
        }

        # Extract key info based on tool type
        if tool == 'list_sessions':
            sessions = result.get('sessions', [])
            ref['summary'] = f"Listed {len(sessions)} sessions"
            ref['type'] = 'session_list'

        elif tool == 'search_sessions':
            sessions = result.get('sessions', [])
            ref['summary'] = f"Found {len(sessions)} matching sessions for '{result.get('query', '')}'"
            ref['type'] = 'search'

        elif tool == 'get_transcript':
            session_name = result.get('session_name', f"Session {result.get('session_id')}")
            summary = result.get('summary', {})
            ref['summary'] = f"{session_name} transcript ({summary.get('total_utterances', 0)} utterances)"
            ref['session_id'] = result.get('session_id')
            ref['session_name'] = session_name
            ref['type'] = 'transcript'
            ref['clickable'] = True

        elif tool == 'get_concept_map':
            session_name = result.get('session_name', f"Session {result.get('session_id')}")
            summary = result.get('summary', {})
            ref['summary'] = f"{session_name} concept map ({summary.get('total_nodes', 0)} concepts)"
            ref['session_id'] = result.get('session_id')
            ref['session_name'] = session_name
            ref['type'] = 'concept_map'
            ref['clickable'] = True

        elif tool == 'get_7c_analysis':
            session_name = result.get('session_name', f"Session {result.get('session_id')}")
            summary = result.get('summary', {})
            score = summary.get('overall_score', 0)
            ref['summary'] = f"{session_name} collaboration analysis (score: {score}/100)"
            ref['session_id'] = result.get('session_id')
            ref['session_name'] = session_name
            ref['type'] = 'collaboration'
            ref['clickable'] = True

        elif tool == 'get_speaker_utterances':
            speaker = result.get('speaker', 'Unknown')
            summary = result.get('summary', {})
            ref['summary'] = f"{speaker}'s utterances ({summary.get('total_utterances', 0)} total, {summary.get('questions_asked', 0)} questions)"
            ref['speaker'] = speaker
            ref['type'] = 'speaker_utterances'

        elif tool == 'get_speaker_profile':
            speaker = result.get('speaker_alias', 'Unknown')
            ref['summary'] = f"{speaker}'s profile and concept contributions"
            ref['speaker'] = speaker
            ref['type'] = 'speaker_profile'

        elif tool == 'compare_sessions':
            ref['summary'] = f"Compared sessions: {result.get('session_ids', [])}"
            ref['type'] = 'comparison'

        elif tool == 'find_concept_path':
            if result.get('path_found'):
                ref['summary'] = f"Path found ({result.get('path_length', 0)} steps)"
            else:
                ref['summary'] = "No path found"
            ref['type'] = 'concept_path'

        else:
            # Generic fallback
            ref['summary'] = f"{tool} executed"

        references.append(ref)

    return references


@agent_v4_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'agent_version': 'v4',
        'design': 'high-agency',
        'default_model': 'gpt-4o',
        'features': [
            'react_style_loop',
            'natural_synthesis',
            'baseline_comparison',
            'multi_turn_context'
        ]
    })


@agent_v4_bp.route('/query', methods=['POST'])
def query():
    """
    Main query endpoint.

    Request body:
    {
        "query": "Your question here",
        "conversation_id": "optional-id",
        "mode": "enhanced" | "baseline",  // default: enhanced
        "model": "gpt-4o",  // optional model override
        "session_id": optional_int  // explicit session context
    }

    Response:
    {
        "answer": "The agent's response",
        "tools_used": [...],
        "conversation_id": "uuid",
        "mode": "enhanced" | "baseline",
        "model": "gpt-4o",
        "success": true/false
    }
    """
    try:
        data = request.get_json()

        query_text = data.get('query', '').strip()
        if not query_text:
            return jsonify({'error': 'Query is required'}), 400

        # Get or create conversation ID
        conversation_id = data.get('conversation_id') or str(uuid.uuid4())

        # Get mode (default: enhanced)
        mode = data.get('mode', 'enhanced')
        if mode not in ['enhanced', 'baseline']:
            mode = 'enhanced'

        # Get conversation context
        context = conversation_manager.get_context(conversation_id)

        # Override session focus if explicitly provided
        if data.get('session_id'):
            context['session_focus'] = data['session_id']

        # Get model (optional override)
        model = data.get('model')

        logger.info(f"[Agent V4] Query: '{query_text}' (mode={mode}, model={model or 'default'}, conversation={conversation_id[:8]})")

        # Run the agent
        result = run_agent(
            query=query_text,
            conversation_id=conversation_id,
            conversation_history=context.get('history'),
            mode=mode,
            session_context=context,
            model=model
        )

        # Update conversation context
        conversation_manager.update_context(conversation_id, result)
        conversation_manager.add_exchange(conversation_id, query_text, result.get('answer', ''))

        # Save to database for persistence
        user_id = _get_user_id()
        _save_conversation_to_db(conversation_id, query_text, result, user_id)

        # Build references from tool results (user requested transparency feature)
        references = _build_references(result.get('tool_results', []))

        # Build response (frontend-compatible format)
        response = {
            'answer': result.get('answer', ''),
            'tools_used': result.get('tools_used', []),
            'conversation_id': conversation_id,
            'mode': mode,
            'model': result.get('model', 'gpt-4o'),
            'success': result.get('success', True),
            'error': result.get('error'),

            # References - what data the agent looked at (clickable in frontend)
            'references': references,

            # Frontend compatibility fields
            'citations': [],  # V4 uses natural prose citations, not structured
            'confidence': None,  # V4 doesn't compute confidence scores
            'reasoning_trace': None,  # V4 doesn't expose reasoning trace
            'follow_up_suggestions': [],
            'needs_clarification': False,
            'is_direct_response': len(result.get('tools_used', [])) == 0,

            # Debug info (optional)
            'debug': {
                'turn_count': result.get('turn_count', 0),
                'tool_results': result.get('tool_results', [])
            } if data.get('include_debug') else None
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return jsonify({
            'answer': f"An error occurred: {str(e)}",
            'tools_used': [],
            'success': False,
            'error': str(e),
            'citations': [],
            'confidence': None,
            'reasoning_trace': None,
            'follow_up_suggestions': [],
            'needs_clarification': False,
            'is_direct_response': False
        }), 500


@agent_v4_bp.route('/context', methods=['GET'])
def get_context():
    """Get stored conversation context."""
    conversation_id = request.args.get('conversation_id')

    if not conversation_id:
        return jsonify({'error': 'conversation_id required'}), 400

    context = conversation_manager.get_context(conversation_id)

    return jsonify({
        'conversation_id': conversation_id,
        'context': {
            'session_focus': context.get('session_focus'),
            'speaker_focus': context.get('speaker_focus'),
            'history_length': len(context.get('history', []))
        }
    })


@agent_v4_bp.route('/context', methods=['DELETE'])
def clear_context():
    """Clear conversation context."""
    conversation_id = request.args.get('conversation_id')

    if conversation_id:
        conversation_manager.clear(conversation_id)
        return jsonify({'success': True, 'message': f'Cleared context for {conversation_id}'})
    else:
        conversation_manager.clear()
        return jsonify({'success': True, 'message': 'Cleared all contexts'})


@agent_v4_bp.route('/compare', methods=['POST'])
def compare_modes():
    """
    Run the same query in both modes for research comparison.

    Request body:
    {
        "query": "Your question here",
        "model": "gpt-4o"  // optional
    }

    Response:
    {
        "query": "...",
        "model": "...",
        "baseline": { answer, tools_used, turn_count },
        "enhanced": { answer, tools_used, turn_count }
    }
    """
    try:
        data = request.get_json()

        query_text = data.get('query', '').strip()
        if not query_text:
            return jsonify({'error': 'Query is required'}), 400

        model = data.get('model')
        logger.info(f"[Agent V4] Compare: '{query_text}' (model={model or 'default'})")

        # Run baseline (transcript only)
        baseline_result = run_agent(
            query=query_text,
            mode="baseline",
            model=model
        )

        # Run enhanced (all artifacts)
        enhanced_result = run_agent(
            query=query_text,
            mode="enhanced",
            model=model
        )

        return jsonify({
            'query': query_text,
            'model': enhanced_result.get('model', 'gpt-4o'),
            'baseline': {
                'answer': baseline_result.get('answer', ''),
                'tools_used': baseline_result.get('tools_used', []),
                'turn_count': baseline_result.get('turn_count', 0)
            },
            'enhanced': {
                'answer': enhanced_result.get('answer', ''),
                'tools_used': enhanced_result.get('tools_used', []),
                'turn_count': enhanced_result.get('turn_count', 0)
            }
        })

    except Exception as e:
        logger.error(f"Compare error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@agent_v4_bp.route('/tools', methods=['GET'])
def list_tools():
    """List available tools for each mode."""
    from .tools import TOOL_SCHEMAS, BASELINE_TOOL_SCHEMAS

    return jsonify({
        'enhanced': {
            'tools': [t['name'] for t in TOOL_SCHEMAS],
            'count': len(TOOL_SCHEMAS)
        },
        'baseline': {
            'tools': [t['name'] for t in BASELINE_TOOL_SCHEMAS],
            'count': len(BASELINE_TOOL_SCHEMAS)
        }
    })


# =============================================================================
# CONVERSATION PERSISTENCE ENDPOINTS (Database-backed)
# =============================================================================

@agent_v4_bp.route('/conversations', methods=['GET'])
def list_conversations():
    """
    List all conversations for the left panel.
    Uses database storage for persistence.
    """
    try:
        user_id = _get_user_id()

        # Get conversations from database (filtered by agent version)
        conversations = db_helper.get_agent_conversations(user_id=user_id, agent_version=AGENT_VERSION)

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


@agent_v4_bp.route('/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """
    Get a specific conversation with its messages.
    """
    try:
        # Get conversation from database
        conv = db_helper.get_agent_conversations(conversation_id=conversation_id)
        if not conv:
            return jsonify({
                'conversation_id': conversation_id,
                'error': 'Conversation not found',
                'messages': []
            }), 404

        # Get messages from database
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


@agent_v4_bp.route('/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation and its messages."""
    try:
        # Delete from database
        db_helper.delete_agent_conversation(conversation_id)

        # Clear in-memory context
        conversation_manager.clear(conversation_id)

        return jsonify({'success': True, 'message': f'Deleted conversation {conversation_id}'})

    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        return jsonify({'error': str(e)}), 500


@agent_v4_bp.route('/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation explicitly."""
    try:
        data = request.get_json() or {}
        title = data.get('title', 'New Conversation')
        user_id = _get_user_id()

        # Create in database (with agent version)
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
