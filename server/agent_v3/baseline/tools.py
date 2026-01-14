"""
Baseline Tool Registry - Transcript-Only Tools for AIED 2026 Comparison

This module defines the restricted tool set for the baseline agent.
Only transcript-based tools are available - no concept maps, 7C, or LIWC.

Tools included:
- list_sessions: Session discovery (shared)
- search_for_sessions: Semantic session search (shared)
- get_transcript: Full transcript WITHOUT LIWC scores (baseline version)
- search_transcripts: RAG transcript search (baseline version)
- get_speaker_utterances: Speaker's raw quotes only (baseline version)
- think: Explicit reasoning (shared)
- clarify: Ask for clarification (shared)

Tools EXCLUDED:
- get_concept_map
- get_7c_analysis
- get_liwc_metrics
- get_speaker_profile (replaced with get_speaker_utterances)
- find_concept_path
- synthesize
- compare_sessions
- test_hypothesis
- All cross-representation tools
"""

# Import baseline-specific tools (no LIWC)
from .baseline_tools import (
    get_transcript_baseline,
    get_speaker_utterances,
    search_transcripts_baseline
)

# Import shared tools from parent module
from ..tools.artifact_tools import (
    list_sessions,
    search_for_sessions
)
from ..tools.reasoning_tools import (
    think,
    clarify
)


# =============================================================================
# BASELINE TOOL REGISTRY
# =============================================================================

BASELINE_TOOLS = {
    # Discovery (shared with full agent)
    "list_sessions": list_sessions,
    "search_for_sessions": search_for_sessions,

    # Transcript retrieval (baseline versions - no LIWC)
    "get_transcript": get_transcript_baseline,  # Overrides full version
    "search_transcripts": search_transcripts_baseline,

    # Speaker (utterances only - no derived analysis)
    "get_speaker_utterances": get_speaker_utterances,

    # Reasoning (shared with full agent)
    "think": think,
    "clarify": clarify,
}

# Tool names for validation
BASELINE_TOOL_NAMES = set(BASELINE_TOOLS.keys())

# Explicitly excluded tools for documentation/logging
EXCLUDED_TOOLS = {
    "get_concept_map",
    "get_7c_analysis",
    "get_liwc_metrics",
    "get_speaker_profile",
    "find_concept_path",
    "synthesize",
    "compare_sessions",
    "test_hypothesis",
    "get_artifacts",
    "trace_to_transcript",
    "get_multi_rep_evidence",
    "get_speaker_unified_view",
    "check_evidence_convergence",
    "find_representation_gaps",
    "explore_concepts",
    "find_reasoning_path",
    "analyze_speaker",
    "compare_speakers",
    "search_concepts",
    "search_communities",
}


def get_baseline_tool(tool_name: str):
    """
    Get a baseline tool by name.

    Returns None if the tool is not in the baseline registry.
    """
    return BASELINE_TOOLS.get(tool_name)


def is_baseline_tool(tool_name: str) -> bool:
    """Check if a tool is available in the baseline."""
    return tool_name in BASELINE_TOOLS


def is_excluded_tool(tool_name: str) -> bool:
    """Check if a tool is explicitly excluded from baseline."""
    return tool_name in EXCLUDED_TOOLS


__all__ = [
    'BASELINE_TOOLS',
    'BASELINE_TOOL_NAMES',
    'EXCLUDED_TOOLS',
    'get_baseline_tool',
    'is_baseline_tool',
    'is_excluded_tool',
]
