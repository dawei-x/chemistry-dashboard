"""
BLINC Agent V7 - Scaffolding Agent

A simplified, flexible agent architecture based on ReAct pattern with:
- Query classification for exploratory vs targeted queries
- Systematic multi-session retrieval for cross-session queries
- Conversation memory for multi-turn context
- Scaffolded responses that point to specific evidence
- LLM-driven tool selection for targeted queries
- User steering compliance

Architecture:
- Query Classifier: Detects exploratory (cross-session) vs targeted (single-session) queries
- Exploratory Retriever: Systematic retrieval across multiple sessions
- ReAct loop: Flexible tool selection for targeted queries
- ConversationMemory: Context persistence across turns
- Scaffolding prompts: Evidence-rich responses

Key features:
1. Cross-session queries get systematic multi-session retrieval
2. Targeted queries use flexible ReAct loop
3. Points users to specific artifacts with explanations
4. Maintains context across conversation turns
"""

# Core agent components
from .react_agent import ScaffoldingAgent, run_agent, AgentResponse
from .memory import ConversationMemory, get_memory, clear_memory

# Query classification and exploratory retrieval (NEW)
from .classifier import classify_query, QueryClassification, is_simple_discovery_query
from .exploratory import (
    retrieve_exploratory,
    ExploratoryResult,
    ExploratoryEvidence,
    format_exploratory_evidence_for_synthesis
)

# Graph and routes
from .graph_v2 import create_agent_graph, get_graph, invoke_agent, reset_conversation
from .routes_v2 import agent_v7_bp

# Legacy architecture (for comparison/fallback)
from .graph import create_agent_graph as create_legacy_graph
from .routes import agent_v7_bp as legacy_agent_v7_bp

__all__ = [
    # Core agent
    'ScaffoldingAgent',
    'run_agent',
    'AgentResponse',
    'ConversationMemory',
    'get_memory',
    'clear_memory',
    # Query classification (NEW)
    'classify_query',
    'QueryClassification',
    'is_simple_discovery_query',
    # Exploratory retrieval (NEW)
    'retrieve_exploratory',
    'ExploratoryResult',
    'ExploratoryEvidence',
    'format_exploratory_evidence_for_synthesis',
    # Graph/routes
    'create_agent_graph',
    'get_graph',
    'invoke_agent',
    'reset_conversation',
    'agent_v7_bp',
    # Legacy (prefixed)
    'create_legacy_graph',
    'legacy_agent_v7_bp',
]
