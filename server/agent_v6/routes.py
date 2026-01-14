"""
API Routes for Agent V6: The Definitive Architecture.

Endpoints:
- /query: Main query endpoint with steering support
- /analyze: Debug endpoint for query analysis
- /context: Get/clear conversation context
- /conversations: Conversation management
- /health: Health check
- /tools: List available tools
"""

import logging
import uuid
from flask import Blueprint, request, jsonify, session

from .agent import run_agent, conversation_manager
from .query_analysis import analyze_query
from .tools import get_all_tool_schemas, filter_tools_by_steering
import database as db_helper

logger = logging.getLogger(__name__)

AGENT_VERSION = 'v6'


def _get_user_id():
    """Get user ID from session, default to 1 for testing."""
    return session.get('user_id', 1)


# Create blueprint
agent_v6_bp = Blueprint('agent_v6', __name__, url_prefix='/api/v6/agent')


# =============================================================================
# REFERENCE BUILDER
# =============================================================================

def _build_references(tool_results: list) -> list:
    """
    Build references from tool results for frontend display.
    Provides transparency about what data the agent used.
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
            ref['summary'] = f"Found {len(sessions)} matching sessions"
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
            ref['summary'] = f"{speaker}'s utterances ({summary.get('total_utterances', 0)} total)"
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
            ref['summary'] = f"{tool} executed"

        references.append(ref)

    return references


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


# =============================================================================
# MAIN ENDPOINTS
# =============================================================================

@agent_v6_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'agent_version': 'v6',
        'design': 'embedded-intelligence',
        'default_model': 'gpt-4o',
        'features': [
            'v3_analytical_intelligence',
            'v4_react_loop',
            'steering_first_class',
            'hypothesis_testing',
            'construct_operationalization',
            'triangulation_framework',
            'beyond_retrieval_reasoning'
        ]
    })


@agent_v6_bp.route('/query', methods=['POST'])
def query():
    """
    Main query endpoint with steering support.

    Request body:
    {
        "query": "Your question here",
        "conversation_id": "optional-id",
        "prefer_representations": ["transcript"],  // optional
        "exclude_representations": ["collaboration"],  // optional
        "mode": "test_hypothesis",  // optional: default, test_hypothesis, compare, trace
        "model": "gpt-4o",  // optional
        "session_id": 20  // optional explicit session context
    }

    Response:
    {
        "answer": "The agent's response",
        "tools_used": [...],
        "references": [...],
        "conversation_id": "uuid",
        "query_intent": {...},
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

        # Get conversation context
        context = conversation_manager.get_context(conversation_id)

        # Override session focus if explicitly provided
        if data.get('session_id'):
            context['session_focus'] = data['session_id']

        # Build API params for steering
        api_params = {
            'prefer_representations': data.get('prefer_representations', []),
            'exclude_representations': data.get('exclude_representations', []),
            'mode': data.get('mode'),
        }

        # Get model
        model = data.get('model')

        logger.info(f"[Agent V6] Query: '{query_text}' "
                    f"(prefer={api_params.get('prefer_representations')}, "
                    f"exclude={api_params.get('exclude_representations')}, "
                    f"mode={api_params.get('mode')}, "
                    f"model={model or 'default'})")

        # Run the agent
        result = run_agent(
            query=query_text,
            conversation_id=conversation_id,
            conversation_history=context.get('history'),
            session_context=context,
            model=model,
            api_params=api_params,
        )

        # Update conversation context
        conversation_manager.update_context(conversation_id, result)
        conversation_manager.add_exchange(conversation_id, query_text, result.get('answer', ''))

        # Save to database for persistence
        user_id = _get_user_id()
        _save_conversation_to_db(conversation_id, query_text, result, user_id)

        # Build references from tool results
        references = _build_references(result.get('tool_results', []))

        # Build response
        response = {
            'answer': result.get('answer', ''),
            'tools_used': result.get('tools_used', []),
            'references': references,
            'conversation_id': conversation_id,
            'model': result.get('model', 'gpt-4o'),
            'success': result.get('success', True),
            'error': result.get('error'),

            # V6-specific: Query understanding with steering info
            'query_intent': result.get('query_intent'),

            # Frontend compatibility
            'citations': [],
            'confidence': None,
            'reasoning_trace': None,
            'follow_up_suggestions': [],
            'needs_clarification': False,
            'is_direct_response': len(result.get('tools_used', [])) == 0,

            # Debug info (optional)
            'debug': {
                'turn_count': result.get('turn_count', 0),
                'elapsed_time': result.get('elapsed_time'),
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
            'references': [],
            'citations': [],
            'confidence': None,
            'reasoning_trace': None,
            'follow_up_suggestions': [],
            'needs_clarification': False,
            'is_direct_response': False
        }), 500


@agent_v6_bp.route('/analyze', methods=['POST'])
def analyze():
    """
    Debug endpoint to see how a query is analyzed.

    Request body:
    {
        "query": "Your question here",
        "prefer_representations": ["transcript"],  // optional
        "exclude_representations": ["collaboration"],  // optional
        "mode": "test_hypothesis",  // optional
        "conversation_id": "optional"
    }

    Response:
    {
        "query": "...",
        "analysis": {
            "session_ids": [...],
            "session_names": [...],
            "speaker_names": [...],
            "prefer_representations": [...],
            "exclude_representations": [...],
            "mode": "...",
            "constructs": [...]
        },
        "available_tools": [...]
    }
    """
    try:
        data = request.get_json()

        query_text = data.get('query', '').strip()
        if not query_text:
            return jsonify({'error': 'Query is required'}), 400

        # Build API params
        api_params = {
            'prefer_representations': data.get('prefer_representations', []),
            'exclude_representations': data.get('exclude_representations', []),
            'mode': data.get('mode'),
        }

        # Get conversation context if provided
        conv_context = {}
        if data.get('conversation_id'):
            ctx = conversation_manager.get_context(data['conversation_id'])
            conv_context = {
                'session_focus': ctx.get('session_focus'),
                'speaker_focus': ctx.get('speaker_focus'),
            }

        # Analyze query
        analysis = analyze_query(query_text, api_params, conv_context)

        # Get available tools with this steering
        tools = filter_tools_by_steering(
            prefer=analysis.prefer_representations,
            exclude=analysis.exclude_representations
        )

        return jsonify({
            'query': query_text,
            'analysis': {
                'session_ids': analysis.session_ids,
                'session_names': analysis.session_names,
                'speaker_names': analysis.speaker_names,
                'prefer_representations': analysis.prefer_representations,
                'exclude_representations': analysis.exclude_representations,
                'mode': analysis.mode,
                'constructs': analysis.constructs,
            },
            'available_tools': [t['name'] for t in tools],
            'conversation_context': conv_context
        })

    except Exception as e:
        logger.error(f"Analyze error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# =============================================================================
# CONTEXT ENDPOINTS
# =============================================================================

@agent_v6_bp.route('/context', methods=['GET'])
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
            'history_length': len(context.get('history', [])),
            'last_intent': context.get('last_intent')
        }
    })


@agent_v6_bp.route('/context', methods=['DELETE'])
def clear_context():
    """Clear conversation context."""
    conversation_id = request.args.get('conversation_id')

    if conversation_id:
        conversation_manager.clear(conversation_id)
        return jsonify({'success': True, 'message': f'Cleared context for {conversation_id}'})
    else:
        conversation_manager.clear()
        return jsonify({'success': True, 'message': 'Cleared all contexts'})


# =============================================================================
# CONVERSATION ENDPOINTS (Database-backed)
# =============================================================================

@agent_v6_bp.route('/conversations', methods=['GET'])
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


@agent_v6_bp.route('/conversations/<conversation_id>', methods=['GET'])
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


@agent_v6_bp.route('/conversations/<conversation_id>', methods=['DELETE'])
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


@agent_v6_bp.route('/conversations', methods=['POST'])
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


# =============================================================================
# TOOLS ENDPOINT
# =============================================================================

@agent_v6_bp.route('/tools', methods=['GET'])
def list_tools():
    """List available tools with optional steering preview."""
    prefer = request.args.getlist('prefer')
    exclude = request.args.getlist('exclude')

    if prefer or exclude:
        tools = filter_tools_by_steering(prefer, exclude)
        return jsonify({
            'tools': [{'name': t['name'], 'description': t['description'][:200] + '...'} for t in tools],
            'count': len(tools),
            'steering': {
                'prefer': prefer,
                'exclude': exclude
            }
        })
    else:
        tools = get_all_tool_schemas()
        return jsonify({
            'tools': [{'name': t['name'], 'description': t['description'][:200] + '...'} for t in tools],
            'count': len(tools)
        })
