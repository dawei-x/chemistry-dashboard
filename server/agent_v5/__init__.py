"""
Agent V5: Context-First Agentic Architecture

Design principles:
1. Always agentic - tools always available
2. Intelligent context pre-loading based on query intent
3. RAG integration for semantic and contrastive retrieval
4. Triangulation awareness - cross-source reasoning without forced structure
5. Multi-turn conversation support

Key insight: RAG is about retrieval, agentic is about control.
Pre-retrieve what you can predict, let agent search for what you can't.
"""

from .agent import run_agent, ConversationManager, conversation_manager
from .query_understanding import understand_query, QueryIntent
from .context_assembly import assemble_context
from .routes import agent_v5_bp

__all__ = [
    'run_agent',
    'ConversationManager',
    'conversation_manager',
    'understand_query',
    'QueryIntent',
    'assemble_context',
    'agent_v5_bp'
]
