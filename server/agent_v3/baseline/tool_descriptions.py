"""
Tool Descriptions for Baseline Agent (Transcript-Only)

AIED 2026 Comparison Baseline
=============================
This agent has access ONLY to transcript data:
- No concept maps (graph structure)
- No 7C collaboration scores
- No LIWC linguistic metrics
- No cross-representation synthesis

Available tools:
1. list_sessions       - Discovery: what sessions exist
2. search_for_sessions - Discovery: find sessions by topic
3. get_transcript      - Get raw transcript (no LIWC scores)
4. search_transcripts  - Search within transcripts
5. get_speaker_utterances - Get speaker's raw quotes only
6. think              - Explicit reasoning
7. clarify            - Ask for clarification
"""

BASELINE_TOOL_DESCRIPTIONS = {

    # =========================================================================
    # DISCOVERY TOOLS
    # =========================================================================

    "list_sessions": {
        "description": """
List all available sessions with metadata.

USE THIS FIRST to understand what data exists.

WHEN TO USE:
- "What sessions are available?"
- "Show me all sessions"
- Starting point when you don't know which sessions exist
- Before retrieving transcripts to find valid session IDs

RETURNS: All sessions with:
- session_id, session_name
- speakers list
- transcript count
""",
        "parameters": {}
    },

    "search_for_sessions": {
        "description": """
Find sessions relevant to a query using semantic search.

WHEN TO USE:
- "Which sessions discussed [topic]?"
- Finding sessions about a specific theme
- Cross-session thematic search
- Don't know which sessions might be relevant

AFTER THIS: Use get_transcript to get the full discussion content.

RETURNS: Ranked list of relevant sessions with match scores
""",
        "parameters": {
            "query": "What to search for (topic, theme, keyword)",
            "top_k": "Number of sessions to return (default 3)"
        }
    },

    # =========================================================================
    # TRANSCRIPT TOOLS
    # =========================================================================

    "get_transcript": {
        "description": """
Get complete transcript for a session.

WHEN TO USE:
- "What was said in session X?"
- "Show me the discussion"
- Need to see what participants said
- Looking for specific quotes or statements

RETURNS:
- summary: total utterances, words, questions, speaker count
- speaker_profiles: per-speaker participation stats
- utterances: full transcript with timestamps

NOTE: This provides raw transcript text. Use search_transcripts for
semantic search if you need to find specific content.
""",
        "parameters": {
            "session_id": "The session ID to get transcript for"
        }
    },

    "search_transcripts": {
        "description": """
Search discussion transcripts for specific content.

WHEN TO USE:
- Finding what was said about a topic
- Looking for specific quotes or mentions
- Questions like "what did they say about X?"
- Semantic search within discussion content

RETURNS: Transcript chunks matching the query with:
- text: The matching utterance
- speaker: Who said it
- session_id, session_name: Which session
- timestamp: When it was said
- relevance_score: How well it matches the query
""",
        "parameters": {
            "query": "What to search for",
            "session_ids": "Optional list of session IDs to filter",
            "speaker": "Optional speaker name filter",
            "limit": "Maximum results (default 10)"
        }
    },

    "get_speaker_utterances": {
        "description": """
Get a speaker's transcript utterances.

WHEN TO USE:
- "What did [Name] say?"
- "Show me [Name]'s contributions"
- Finding a specific speaker's quotes
- Understanding what a participant discussed

RETURNS:
- speaker_alias, speaker_id
- participation: utterance count, word count, questions asked
- sample_quotes: Quick preview of key quotes
- utterances: Full list of speaker's statements with timestamps

NOTE: This provides raw quotes only. For understanding patterns
in what a speaker said, you'll need to analyze the utterances yourself.
""",
        "parameters": {
            "speaker_name": "Name or alias of the speaker",
            "session_id": "Optional session filter"
        }
    },

    # =========================================================================
    # REASONING TOOLS
    # =========================================================================

    "think": {
        "description": """
Explicitly record your reasoning process.

WHEN TO USE:
- Working through complex analysis
- Need to organize thoughts before responding
- Synthesizing information from multiple sources
- Making inferences from transcript data

This tool does not return data - it's for structured thinking.
""",
        "parameters": {
            "thought": "Your reasoning or analysis"
        }
    },

    "clarify": {
        "description": """
Ask the user for clarification when the query is ambiguous.

WHEN TO USE:
- Query is unclear or could have multiple interpretations
- Need to know which session the user is asking about
- Missing critical information to proceed

PREFER TO ANSWER when possible rather than over-clarifying.
""",
        "parameters": {
            "question": "What to ask the user",
            "options": "Optional list of choices to present"
        }
    },
}


def get_baseline_tools_prompt() -> str:
    """
    Generate a formatted tools prompt for the baseline reasoning model.

    This creates a clear, structured description of available transcript-only tools.
    """
    lines = [
        "# Available Tools",
        "",
        "NOTE: You have access to transcript data only. Concept maps, collaboration scores,",
        "and linguistic metrics are NOT available in this baseline configuration.",
        ""
    ]

    for tool_name, tool_info in BASELINE_TOOL_DESCRIPTIONS.items():
        lines.append(f"## {tool_name}")
        lines.append(tool_info["description"].strip())
        lines.append("")

        if tool_info.get("parameters"):
            lines.append("**Parameters:**")
            for param, desc in tool_info["parameters"].items():
                lines.append(f"- `{param}`: {desc}")
            lines.append("")

    return "\n".join(lines)


__all__ = ['BASELINE_TOOL_DESCRIPTIONS', 'get_baseline_tools_prompt']
