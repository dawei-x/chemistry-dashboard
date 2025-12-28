"""
Tools for BLINC Agent V3

Clean tool implementations that leverage existing RAG infrastructure.
"""

from .search_tools import (
    search_transcripts,
    search_sessions,
    search_concepts,
    search_communities
)
from .analysis_tools import (
    get_session_overview,
    get_collaboration_analysis,
    compare_sessions,
    analyze_speaker
)
from .graph_tools import (
    explore_concepts,
    find_reasoning_path,
    get_concept_map
)
from .reasoning_tools import (
    think,
    clarify
)

# All tools available to the agent
ALL_TOOLS = {
    # Reasoning
    "think": think,
    "clarify": clarify,
    # Search
    "search_transcripts": search_transcripts,
    "search_sessions": search_sessions,
    "search_concepts": search_concepts,
    "search_communities": search_communities,
    # Analysis
    "get_session_overview": get_session_overview,
    "get_collaboration_analysis": get_collaboration_analysis,
    "compare_sessions": compare_sessions,
    "analyze_speaker": analyze_speaker,
    # Graph
    "explore_concepts": explore_concepts,
    "find_reasoning_path": find_reasoning_path,
    "get_concept_map": get_concept_map
}

__all__ = [
    'ALL_TOOLS',
    'think',
    'clarify',
    'search_transcripts',
    'search_sessions',
    'search_concepts',
    'search_communities',
    'get_session_overview',
    'get_collaboration_analysis',
    'compare_sessions',
    'analyze_speaker',
    'explore_concepts',
    'find_reasoning_path',
    'get_concept_map'
]
