"""
Tool Descriptions for BLINC Agent V3

These descriptions follow OpenAI's best practices:
- Define WHEN the tool should be invoked
- Define HOW arguments should be constructed
- Provide clear examples
- Explain what NOT to use the tool for

The model will naturally select the right tool based on
understanding, not keyword matching.
"""

TOOL_DESCRIPTIONS = {

    # =========================================================================
    # REASONING TOOLS
    # =========================================================================

    "think": {
        "description": """
Use this tool to think through complex problems step by step.

WHEN TO USE:
- The query requires multi-step reasoning
- You need to analyze information from previous tool results
- You're deciding between multiple possible approaches
- The situation is ambiguous and needs careful consideration
- You need to synthesize information from multiple sources

WHEN NOT TO USE:
- Simple factual queries that can be answered directly
- When you already know exactly which tool to use

HOW TO USE:
- Write out your reasoning process clearly
- Consider multiple angles
- Identify what information you still need
- Plan your next steps

The thought will be recorded for transparency but not shown to the user.
""",
        "parameters": {
            "reasoning": "Your step-by-step reasoning process"
        }
    },

    "clarify": {
        "description": """
Ask the user for clarification when the query is genuinely ambiguous.

WHEN TO USE:
- The query references "the session" or "that discussion" with NO context
- Multiple very different interpretations are equally valid
- Critical information is missing that prevents any useful response

WHEN NOT TO USE (IMPORTANT - default to NOT clarifying):
- You can make a reasonable assumption and search
- The query mentions any specific topic, name, or identifier
- You have session context from the conversation
- A search would likely find relevant results anyway
- The query is general (e.g., "what topics were discussed?") - just search all

PRINCIPLE: When in doubt, SEARCH rather than ask. Users prefer results over questions.
""",
        "parameters": {
            "question": "A clear, specific question to ask the user",
            "options": "2-4 specific options the user can choose from"
        }
    },

    # =========================================================================
    # SEARCH TOOLS
    # =========================================================================

    "search_transcripts": {
        "description": """
Search discussion transcripts for specific content, quotes, or moments.

WHEN TO USE:
- Finding what was said about a specific topic
- Looking for exact quotes or statements
- Finding when something was mentioned
- Searching for specific moments in discussions
- Questions like "what did [Speaker] say about X?"

WHEN NOT TO USE:
- Finding session-level patterns (use search_sessions)
- Analyzing collaboration quality (use get_collaboration_analysis)
- Understanding how ideas connect (use explore_concepts)

HOW TO USE:
- query: The topic, concept, or phrase to search for
- session_ids: Optional list to limit search to specific sessions
- speaker: Optional speaker name to filter results (e.g., "Tucker", "David")
- limit: Number of results (default 10)

RETURNS: Transcript chunks with speaker, timestamp, and context.

EXAMPLE: To find what Tucker said about AI:
- First, check session table: Tucker is in Session 19
- search_transcripts(query="AI reasoning", session_ids=[19], speaker="Tucker")
""",
        "parameters": {
            "query": "Search query - topic, phrase, or concept to find",
            "session_ids": "Optional: List of session IDs to search within",
            "speaker": "Optional: Speaker name to filter results",
            "limit": "Number of results to return (default 10)"
        }
    },

    "search_sessions": {
        "description": """
Search for sessions by topic, pattern, or characteristics.

WHEN TO USE:
- Finding sessions about a specific topic
- Looking for sessions with certain patterns
- Questions like "which sessions discussed X?"
- Browsing what sessions are available
- Finding sessions with particular qualities

WHEN NOT TO USE:
- Finding specific quotes (use search_transcripts)
- Analyzing one session deeply (use get_session_overview)
- Comparing specific sessions (use compare_sessions)

HOW TO USE:
- query: What you're looking for in sessions
- limit: Number of sessions to return

RETURNS: Session summaries with topics, participants, and key metrics.
""",
        "parameters": {
            "query": "What to search for in sessions",
            "limit": "Number of sessions to return (default 5)"
        }
    },

    "search_concepts": {
        "description": """
Search the concept map for specific ideas, questions, or hypotheses.

WHEN TO USE:
- Finding specific concepts or ideas discussed
- Looking for questions that were asked
- Finding hypotheses or conclusions
- Understanding what ideas emerged in discussions

WHEN NOT TO USE:
- Finding exact quotes (use search_transcripts)
- Understanding how concepts connect (use explore_concepts)
- Getting full concept map (use get_concept_map)

RETURNS: Concept nodes with type, speaker, and theme context.
""",
        "parameters": {
            "query": "The concept, idea, or question to search for",
            "session_ids": "Optional: List of session IDs to search within",
            "concept_types": "Optional: Filter by type (question, idea, hypothesis, etc.)",
            "limit": "Number of results (default 10)"
        }
    },

    "search_communities": {
        "description": """
Search thematic communities (clusters of related concepts) across sessions.

WHEN TO USE:
- Understanding major themes across discussions
- Finding sessions that share similar topics
- Answering "what themes emerged?" questions
- Global understanding of discussion patterns

WHEN NOT TO USE:
- Finding specific quotes or moments
- Analyzing one session in detail

RETURNS: Community summaries with key concepts and participating sessions.
""",
        "parameters": {
            "query": "Theme or topic to search for",
            "limit": "Number of communities (default 5)"
        }
    },

    # =========================================================================
    # ANALYSIS TOOLS
    # =========================================================================

    "get_session_overview": {
        "description": """
Get a comprehensive overview of a specific session.

WHEN TO USE:
- "What happened in session X?"
- "Tell me about the [Name] session"
- Understanding a session before diving into details
- Getting context about participants, topics, and flow

WHEN NOT TO USE:
- Comparing multiple sessions (use compare_sessions)
- Finding specific content (use search_transcripts)
- Analyzing collaboration (use get_collaboration_analysis)

REQUIRES: session_id - the specific session to analyze

RETURNS: Session summary including:
- Main topics and themes
- Participants and their roles
- Key moments and insights
- Duration and structure
""",
        "parameters": {
            "session_id": "The session ID to get overview for"
        }
    },

    "get_collaboration_analysis": {
        "description": """
Get 7C collaboration quality analysis for a session.

WHEN TO USE:
- "How well did they collaborate?"
- "Was the discussion productive?"
- "Did everyone participate equally?"
- "Was there good communication?"
- Analyzing group dynamics and interaction quality

WHEN NOT TO USE:
- Finding what was discussed (use search_transcripts)
- Comparing session topics (use compare_sessions)
- Understanding concepts (use explore_concepts)

REQUIRES: session_id - the specific session to analyze

RETURNS: Seven collaboration dimensions (0-100 scores):
- Climate: Psychological safety, supportive atmosphere
- Communication: Clarity, active listening, articulation
- Contribution: Balanced participation, equal voice
- Conflict: Constructive disagreement, productive debate
- Context: Shared understanding, common ground
- Constructive: Building on others' ideas
- Compatibility: Working style alignment

Each dimension includes score, explanation, and evidence.
""",
        "parameters": {
            "session_id": "The session ID to analyze"
        }
    },

    "compare_sessions": {
        "description": """
Compare two or more sessions across multiple dimensions.

WHEN TO USE:
- "Compare session X and Y"
- "What's different between these sessions?"
- "Which session had better collaboration?"
- Analyzing patterns across sessions

WHEN NOT TO USE:
- Analyzing a single session (use get_session_overview)
- Finding content in sessions (use search_transcripts)

REQUIRES: session_ids - list of 2+ session IDs to compare

RETURNS: Comparison across:
- Topics and themes
- Collaboration metrics
- Participant dynamics
- Key differences and similarities
""",
        "parameters": {
            "session_ids": "List of session IDs to compare (minimum 2)"
        }
    },

    "analyze_speaker": {
        "description": """
Analyze a speaker's participation patterns across sessions.

WHEN TO USE:
- "How does [Name] participate?"
- "What's [Name]'s discussion style?"
- "How did [Name] contribute?"
- Understanding individual speaker patterns

WHEN NOT TO USE:
- Finding what a speaker said (use search_transcripts with speaker filter)
- Comparing speakers within one session (use get_speaker_comparison)

HOW TO USE:
- speaker_name: The speaker to analyze (e.g., "Lex", "Julia")
- session_ids: Optional - limit to specific sessions

RETURNS: Speaker profile including:
- Participation patterns
- Speaking style characteristics
- Contribution types (questions, ideas, etc.)
- Cross-session patterns
""",
        "parameters": {
            "speaker_name": "Name of the speaker to analyze",
            "session_ids": "Optional: Limit analysis to specific sessions"
        }
    },

    # =========================================================================
    # GRAPH NAVIGATION TOOLS
    # =========================================================================

    "explore_concepts": {
        "description": """
Explore how concepts connect in the discussion graph.

WHEN TO USE:
- "How does X relate to Y?"
- "What ideas are connected to X?"
- "What led to this conclusion?"
- Understanding reasoning chains and idea development
- Following the flow of discussion

WHEN NOT TO USE:
- Finding specific quotes (use search_transcripts)
- Getting all concepts (use get_concept_map)
- Session-level analysis (use get_session_overview)

HOW TO USE:
- concept_id: Starting concept to explore from
- direction: "outgoing" (what it leads to), "incoming" (what led to it), "both"
- depth: How many hops to explore (1-3)

RETURNS: Connected concepts with relationship types and paths.
""",
        "parameters": {
            "concept_id": "The concept node ID to explore from",
            "direction": "Exploration direction: 'outgoing', 'incoming', or 'both'",
            "depth": "How many hops to explore (1-3, default 2)"
        }
    },

    "find_reasoning_path": {
        "description": """
Find the reasoning path between two concepts.

WHEN TO USE:
- "How did they get from X to Y?"
- "What's the connection between these ideas?"
- "Trace the reasoning from X to Y"
- Understanding how conclusions were reached

REQUIRES: Both source and target concept IDs

RETURNS: The path of concepts and relationships connecting them.
""",
        "parameters": {
            "source_id": "Starting concept ID",
            "target_id": "Target concept ID",
            "max_depth": "Maximum path length to search (default 4)"
        }
    },

    "get_concept_map": {
        "description": """
Get the full concept map structure for a session.

WHEN TO USE:
- "Show me the concept map"
- "What ideas emerged in this session?"
- Understanding the complete structure of ideas
- Seeing how all concepts relate

WHEN NOT TO USE:
- Finding specific concepts (use search_concepts)
- Exploring connections (use explore_concepts)

REQUIRES: session_id

RETURNS: All concepts, relationships, and clusters for the session.
""",
        "parameters": {
            "session_id": "The session ID to get concept map for"
        }
    }
}


def get_tools_prompt() -> str:
    """
    Generate a formatted tools prompt for the reasoning model.

    This creates a clear, structured description of all available tools
    that the model can use to understand when and how to use each one.
    """
    lines = ["# Available Tools\n"]

    for tool_name, tool_info in TOOL_DESCRIPTIONS.items():
        lines.append(f"## {tool_name}")
        lines.append(tool_info["description"].strip())
        lines.append("")

        if tool_info.get("parameters"):
            lines.append("**Parameters:**")
            for param, desc in tool_info["parameters"].items():
                lines.append(f"- `{param}`: {desc}")
            lines.append("")

    return "\n".join(lines)
