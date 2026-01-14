"""
Tools Module for Agent V6.

Contains:
- Rich tool definitions with interpretive guidance (from V4)
- Tool filtering based on steering (V6 feature)
- Tool execution logic

Design: Tool descriptions embed domain knowledge about when/how to use each tool.
"""

import sys
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import implementations from v3/v4
from agent_v3.tools.artifact_tools import (
    list_sessions,
    search_for_sessions,
    get_transcript,
    get_concept_map,
    get_7c_analysis,
    get_speaker_profile,
    find_concept_path,
)
from agent_v3.tools.analysis_tools import compare_sessions

# Import enhanced speaker tool from v4
from agent_v4.tools import get_speaker_utterances


# =============================================================================
# REPRESENTATION TO TOOL MAPPING
# =============================================================================

# Maps representation types to their associated tools
REP_TO_TOOLS = {
    'transcript': ['get_transcript', 'get_speaker_utterances'],
    'concept_map': ['get_concept_map', 'find_concept_path'],
    'collaboration': ['get_7c_analysis'],
    'speakers': ['get_speaker_profile', 'get_speaker_utterances'],
}

# Tools that are always available regardless of steering
CORE_TOOLS = ['list_sessions', 'search_sessions', 'compare_sessions']


# =============================================================================
# TOOL SCHEMAS WITH RICH DESCRIPTIONS
# =============================================================================

ALL_TOOL_SCHEMAS = [
    {
        "name": "list_sessions",
        "description": """List all available discussion sessions.

WHEN TO USE:
- As a first step to discover what sessions exist
- When you don't know which sessions are available
- To find session IDs for specific session names

RETURNS:
- session_id: Unique identifier (use this in other tool calls)
- session_name: Human-readable name (e.g., "Nuclear Fusion", "Is AI Alive")
- speakers: List of participant names
- artifacts_available: Which data types exist

This tool helps you understand what data is available before diving into specifics.""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "search_sessions",
        "description": """Find sessions relevant to a topic using semantic search.

WHEN TO USE:
- When asked about a topic and you need to find relevant sessions
- When you don't know which sessions discuss a particular concept
- To discover sessions before detailed analysis

RETURNS:
- Ranked list of sessions with relevance scores
- Preview of matching content

NOTE: This searches across session content, not just names.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or keywords to search for"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_transcript",
        "description": """Get the complete transcript for a session.

WHEN TO USE:
- Questions about what participants SAID
- Need for direct quotes and evidence
- Analyzing discourse patterns
- Tracing argument development

EPISTEMIC STATUS: PRIMARY evidence - what was actually said.

RETURNS:
- utterances: Full text with speaker, timestamp, word count
- summary: Total utterances, words, questions
- speaker_profiles: Per-speaker participation stats
- linguistic_scores: Analytic thinking, certainty (0-100)

TRIANGULATION: Use transcript quotes to support/verify claims from other sources.
The transcript is ground truth - other representations are derived from it.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "Session ID to retrieve"
                }
            },
            "required": ["session_id"]
        }
    },
    {
        "name": "get_concept_map",
        "description": """Get the concept map showing how ideas connect.

WHEN TO USE:
- Questions about STRUCTURE of reasoning
- How ideas relate to each other
- Tracing conceptual evolution
- Identifying central ideas (hub nodes)

EPISTEMIC STATUS: DERIVED - extracted structure of ideas.

RETURNS:
- nodes: Concepts with type (claim, question, evidence) and speaker
- edges: Relationships (supports, challenges, causes, elaborates)
- clusters: Thematic groupings
- reasoning_patterns: Detected patterns (causal chains, Q&A pairs)
- hub_nodes: Most connected concepts

INTERPRETATION:
- Hub nodes = central ideas driving discussion
- Clusters = thematic areas of focus
- Edge types reveal reasoning quality (support vs challenge)

TRIANGULATION: Verify conceptual claims against transcript quotes.
The map shows WHAT ideas are present; the transcript shows HOW they were expressed.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "Session ID to retrieve"
                }
            },
            "required": ["session_id"]
        }
    },
    {
        "name": "get_7c_analysis",
        "description": """Get collaboration quality analysis with 7C dimension scores.

WHEN TO USE:
- Questions about collaboration quality
- Comparing sessions on interaction quality
- Understanding social dynamics
- Identifying strengths and improvement areas

EPISTEMIC STATUS: INTERPRETED - quantified assessment of interaction.

RETURNS 7 dimensions (0-100 scores):
- Climate: Psychological safety, supportive atmosphere
- Communication: Clarity, active listening, articulation
- Contribution: Balanced participation (high = all voices heard)
- Conflict: Disagreement handling (high = constructive)
- Context: Shared understanding, common ground
- Constructive: Building on others' ideas
- Compatibility: Working style alignment

Also returns:
- overall_score: Average across dimensions
- evidence: Coded segments supporting scores

INTERPRETATION GUIDANCE:
- High Climate (70+) = supportive environment
- Low Contribution (30-) = speaker imbalance - investigate WHO dominated
- High Conflict + High Constructive = healthy debate
- Low Communication + High Content = ideas present but poorly exchanged

TRIANGULATION: A score tells you WHAT; the transcript shows you HOW.
If contribution is low, find who spoke most in the transcript.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "Session ID to retrieve"
                }
            },
            "required": ["session_id"]
        }
    },
    {
        "name": "get_speaker_utterances",
        "description": """Get ALL utterances from a specific speaker.

WHEN TO USE:
- Questions about what a speaker SAID, ASKED, or EXPLAINED
- Analyzing a speaker's communication style
- Finding specific quotes from someone
- Understanding a speaker's role in discussion

CRITICAL: Specify session_id when possible for focused results.

RETURNS:
- role_summary: 'explainer', 'questioner', or 'balanced'
- utterances: ALL utterances with:
  - text: Full utterance
  - is_question: Whether marked as question
  - is_self_answered: TRUE = rhetorical (explaining via question)
  - intent: 'explaining', 'questioning', 'rhetorical_explaining', etc.
- genuine_questions: Actual information-seeking questions
- rhetorical_questions: Questions speaker answers themselves

INTERPRETATION:
- A speaker with many questions but high 'is_self_answered' is EXPLAINING
  through rhetorical questions, not genuinely asking
- Check 'role_summary.primary_role' for overall characterization""",
        "input_schema": {
            "type": "object",
            "properties": {
                "speaker_name": {
                    "type": "string",
                    "description": "Name of the speaker (fuzzy matched)"
                },
                "session_id": {
                    "type": "integer",
                    "description": "Session ID (strongly recommended)"
                }
            },
            "required": ["speaker_name"]
        }
    },
    {
        "name": "get_speaker_profile",
        "description": """Get a speaker's concept contributions and connections.

WHEN TO USE:
- Understanding a speaker's intellectual ROLE
- What CONCEPTS they contributed
- How their ideas connect to others
- Cross-session patterns (if no session_id specified)

For actual UTTERANCES (what they said), use get_speaker_utterances instead.

EPISTEMIC STATUS: AGGREGATED - patterns across contributions.

RETURNS:
- transcript_summary: Sessions participated, utterance counts
- concept_summary: Concepts contributed, types (claims, questions)
- connections: How this speaker's ideas link to others

INTERPRETATION:
- Compare with other speakers to understand relative contribution
- Look at concept types to understand their role (questioning vs claiming)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "speaker_name": {
                    "type": "string",
                    "description": "Name of the speaker (fuzzy matched)"
                },
                "session_id": {
                    "type": "integer",
                    "description": "Optional: limit to specific session"
                }
            },
            "required": ["speaker_name"]
        }
    },
    {
        "name": "compare_sessions",
        "description": """Compare multiple sessions on collaboration and participation.

WHEN TO USE:
- Comparing sessions (which had better collaboration?)
- Ranking sessions by quality
- Finding differences and similarities
- Contrastive analysis

RETURNS:
- Ranked comparison with scores
- Key differences and similarities
- Best/worst performers per metric

NOTE: This uses 7C scores for collaboration comparison.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Session IDs to compare (omit for all)"
                },
                "metric": {
                    "type": "string",
                    "enum": ["collaboration", "participation", "all"],
                    "description": "What to compare (default: all)",
                    "default": "all"
                }
            },
            "required": []
        }
    },
    {
        "name": "find_concept_path",
        "description": """Find the reasoning path between two concepts.

WHEN TO USE:
- Tracing how one idea led to another
- Understanding chains of reasoning
- Finding connections in concept map

RETURNS:
- path: Each step showing concept -> relationship -> concept
- narrative: Human-readable reasoning chain
- speakers_involved: Who contributed to this chain

INTERPRETATION:
- The path shows conceptual flow, not temporal sequence
- Multiple paths may exist - this returns the shortest""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "Session to search in"
                },
                "from_concept": {
                    "type": "string",
                    "description": "Starting concept (fuzzy matched)"
                },
                "to_concept": {
                    "type": "string",
                    "description": "Target concept (fuzzy matched)"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum path length (default: 5)",
                    "default": 5
                }
            },
            "required": ["session_id", "from_concept", "to_concept"]
        }
    }
]


# =============================================================================
# TOOL FILTERING
# =============================================================================

def filter_tools_by_steering(
    prefer: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> List[Dict]:
    """
    Filter tools based on steering preferences.

    Args:
        prefer: List of representation types to prefer (e.g., ['transcript', 'concept_map'])
        exclude: List of representation types to exclude (e.g., ['collaboration'])

    Returns:
        Filtered list of tool schemas
    """
    prefer = prefer or []
    exclude = exclude or []

    # Start with core tools (always available)
    included_tool_names = set(CORE_TOOLS)

    if prefer:
        # Only include tools for preferred representations
        for rep in prefer:
            if rep in REP_TO_TOOLS:
                included_tool_names.update(REP_TO_TOOLS[rep])
    else:
        # Include all tools by default
        for tools in REP_TO_TOOLS.values():
            included_tool_names.update(tools)

    # Remove tools for excluded representations
    for rep in exclude:
        if rep in REP_TO_TOOLS:
            for tool_name in REP_TO_TOOLS[rep]:
                # Don't remove if also in a preferred representation
                is_in_preferred = any(
                    tool_name in REP_TO_TOOLS.get(p, [])
                    for p in prefer
                )
                if not is_in_preferred:
                    included_tool_names.discard(tool_name)

    # Filter schemas
    return [
        schema for schema in ALL_TOOL_SCHEMAS
        if schema['name'] in included_tool_names
    ]


def get_tool_names_for_steering(
    prefer: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> List[str]:
    """Get list of tool names that will be available with given steering."""
    schemas = filter_tools_by_steering(prefer, exclude)
    return [s['name'] for s in schemas]


# =============================================================================
# TOOL EXECUTION
# =============================================================================

TOOL_FUNCTIONS = {
    "list_sessions": list_sessions,
    "search_sessions": search_for_sessions,
    "get_transcript": get_transcript,
    "get_concept_map": get_concept_map,
    "get_7c_analysis": get_7c_analysis,
    "get_speaker_utterances": get_speaker_utterances,
    "get_speaker_profile": get_speaker_profile,
    "compare_sessions": compare_sessions,
    "find_concept_path": find_concept_path,
}


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """
    Execute a tool and return the result.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool

    Returns:
        Tool execution result
    """
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {tool_name}", "is_relevant": False}

    try:
        logger.info(f"[V6] Executing tool: {tool_name} with input: {tool_input}")
        result = TOOL_FUNCTIONS[tool_name](**tool_input)
        result['tool_name'] = tool_name  # Ensure tool name is in result
        return result
    except Exception as e:
        logger.error(f"[V6] Tool execution error: {tool_name} - {e}")
        return {"error": str(e), "tool_name": tool_name, "is_relevant": False}


def get_all_tool_schemas() -> List[Dict]:
    """Get all tool schemas (no filtering)."""
    return ALL_TOOL_SCHEMAS


def describe_available_tools(
    prefer: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> str:
    """Generate a description of available tools for debugging/logging."""
    tool_names = get_tool_names_for_steering(prefer, exclude)
    return f"Available tools: {', '.join(tool_names)}"
