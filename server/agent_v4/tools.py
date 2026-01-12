"""
Tool definitions for Agent V4.

Design principle: Clear, complete tool descriptions following Anthropic's ACI guidance.
Reuses implementations from agent_v3/tools/artifact_tools.py with critical fixes.
"""

import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import implementations from v3
from agent_v3.tools.artifact_tools import (
    list_sessions,
    search_for_sessions,
    get_transcript,
    get_concept_map,
    get_7c_analysis,
    get_speaker_profile as _get_speaker_profile_v3,
    find_concept_path,
    _get_db_connection,
)

# Also import compare functionality
from agent_v3.tools.analysis_tools import compare_sessions


# =============================================================================
# IMPROVED SPEAKER TOOL - Returns actual utterances with attribution context
# =============================================================================

import re

def _detect_self_answered(text: str) -> dict:
    """
    Detect if an utterance contains a self-answered question (rhetorical pattern).

    Rhetorical patterns: "Question? No/Yes/Well/Because/I think..."
    These indicate the speaker is explaining, not genuinely asking.
    """
    if not text:
        return {"is_self_answered": False, "pattern": None}

    # Pattern: question mark followed by answer indicators
    patterns = [
        (r'\?\s*(No,|Yes,|Well,|Because|I think|I mean|Actually|So|It\'s|They\'re|We)', 'question_then_answer'),
        (r'\?\s*[A-Z][^?]*\.$', 'question_then_statement'),  # Question followed by declarative sentence
        (r'(right|isn\'t it|don\'t you think|you know)\?', 'tag_question'),  # Tag questions are rhetorical
    ]

    for pattern, pattern_type in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {"is_self_answered": True, "pattern": pattern_type}

    return {"is_self_answered": False, "pattern": None}


def _classify_utterance_intent(text: str, is_question: bool, self_answered: dict) -> str:
    """
    Classify the communicative intent of an utterance.

    Returns: 'explaining', 'questioning', 'rhetorical', 'responding', 'facilitating'
    """
    if not text:
        return "unknown"

    text_lower = text.lower()

    # Self-answered questions are rhetorical/explaining
    if self_answered.get("is_self_answered"):
        return "rhetorical_explaining"

    # Tag questions are facilitative
    if self_answered.get("pattern") == "tag_question":
        return "facilitating"

    # Pure questions (marked as question and not self-answered)
    if is_question:
        # Check if it's a clarifying question
        if any(w in text_lower for w in ["what do you mean", "can you explain", "could you clarify"]):
            return "clarifying"
        # Check if it's a probing question
        if any(w in text_lower for w in ["why", "how come", "what if"]):
            return "probing"
        return "questioning"

    # Statements that build on others
    if any(w in text_lower for w in ["i agree", "building on", "to add to", "exactly"]):
        return "building"

    # Responses/reactions
    if any(w in text_lower for w in ["i think", "in my opinion", "i believe"]):
        return "opining"

    # Default to explaining for declarative statements
    return "explaining"


def get_speaker_utterances(
    speaker_name: str,
    session_id: int = None,
    include_context: bool = True
) -> dict:
    """
    Get ALL utterances from a specific speaker with their actual text.

    CRITICAL: Use this tool when asked about what a speaker SAID or ASKED.
    This returns the COMPLETE list of utterances with context for proper interpretation.

    Args:
        speaker_name: Name of the speaker (fuzzy matched)
        session_id: Limit to specific session (required for most queries)
        include_context: Include surrounding context (prev/next utterance)

    Returns:
        - speaker_info: speaker ID, alias, session participation
        - utterances: ALL utterances with full text, timestamps, is_question flag
        - role_summary: Classification of speaker's overall role (questioner/explainer/balanced)
        - questions_asked: List of questions (with rhetorical vs genuine classification)
        - summary: total utterances, words, questions count
    """
    logger.info(f"Getting utterances for speaker: {speaker_name}, session: {session_id}")

    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Find speaker
        cursor.execute("""
            SELECT id, alias FROM speaker WHERE alias LIKE %s LIMIT 1
        """, (f"%{speaker_name}%",))
        speaker = cursor.fetchone()

        if not speaker:
            cursor.close()
            connection.close()
            return {
                "tool_name": "get_speaker_utterances",
                "error": f"Speaker '{speaker_name}' not found",
                "suggestion": "Use list_sessions to see available speakers",
                "is_relevant": False
            }

        speaker_id = speaker['id']
        speaker_alias = speaker['alias']

        # Build query based on session filter
        session_filter = "AND t.session_device_id = %s" if session_id else ""
        params = [speaker_id]
        if session_id:
            params.append(session_id)

        # Get ALL utterances from this speaker with context
        cursor.execute(f"""
            SELECT
                t.id as utterance_id,
                t.session_device_id,
                COALESCE(s.name, sd.name) as session_name,
                t.transcript as text,
                t.start_time,
                t.word_count,
                t.question as is_question,
                t.analytic_thinking_value,
                t.certainty_value,
                t.speaker_id
            FROM transcript t
            JOIN session_device sd ON t.session_device_id = sd.id
            JOIN session s ON sd.session_id = s.id
            WHERE t.speaker_id = %s {session_filter}
            ORDER BY t.session_device_id, t.start_time
        """, params)

        utterances_raw = cursor.fetchall()

        if not utterances_raw:
            cursor.close()
            connection.close()
            return {
                "tool_name": "get_speaker_utterances",
                "speaker": speaker_alias,
                "session_id": session_id,
                "error": f"No utterances found for {speaker_alias}" + (f" in session {session_id}" if session_id else ""),
                "is_relevant": False
            }

        # Get conversation context (prev/next speakers) if requested
        context_map = {}
        if include_context and session_id:
            cursor.execute("""
                SELECT
                    t.id as utterance_id,
                    t.start_time,
                    sp.alias as speaker_name,
                    t.transcript as text
                FROM transcript t
                JOIN speaker sp ON t.speaker_id = sp.id
                WHERE t.session_device_id = %s
                ORDER BY t.start_time
            """, (session_id,))
            all_utterances = cursor.fetchall()

            # Build context map: for each utterance, who spoke before/after
            for i, u in enumerate(all_utterances):
                prev_speaker = all_utterances[i-1]['speaker_name'] if i > 0 else None
                next_speaker = all_utterances[i+1]['speaker_name'] if i < len(all_utterances)-1 else None
                prev_text = all_utterances[i-1]['text'][:100] if i > 0 else None
                next_text = all_utterances[i+1]['text'][:100] if i < len(all_utterances)-1 else None
                context_map[u['utterance_id']] = {
                    "prev_speaker": prev_speaker,
                    "next_speaker": next_speaker,
                    "prev_text_preview": prev_text,
                    "next_text_preview": next_text
                }

        cursor.close()
        connection.close()

        # Format utterances with enhanced classification
        utterances = []
        questions = []
        genuine_questions = []
        rhetorical_questions = []
        total_words = 0
        intent_counts = {}

        for u in utterances_raw:
            text = u['text'] or ""
            is_question = bool(u['is_question'])

            # Detect self-answered patterns
            self_answered = _detect_self_answered(text)

            # Classify intent
            intent = _classify_utterance_intent(text, is_question, self_answered)
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

            utterance = {
                "utterance_id": u['utterance_id'],
                "session_id": u['session_device_id'],
                "session_name": u['session_name'],
                "text": text,
                "start_time": u['start_time'],
                "word_count": u['word_count'] or 0,
                "is_question": is_question,
                "is_self_answered": self_answered.get("is_self_answered", False),
                "intent": intent,
                "analytic_thinking": u['analytic_thinking_value'],
                "certainty": u['certainty_value']
            }

            # Add context if available
            if u['utterance_id'] in context_map:
                utterance["context"] = context_map[u['utterance_id']]

            utterances.append(utterance)
            total_words += u['word_count'] or 0

            # Categorize questions
            if is_question:
                question_entry = {
                    "text": text,
                    "session_name": u['session_name'],
                    "start_time": u['start_time'],
                    "is_rhetorical": self_answered.get("is_self_answered", False),
                    "intent": intent
                }
                questions.append(question_entry)

                if self_answered.get("is_self_answered"):
                    rhetorical_questions.append(question_entry)
                else:
                    genuine_questions.append(question_entry)

        # Build role summary based on intent distribution
        total_utterances = len(utterances)
        explaining_count = sum(1 for u in utterances if u['intent'] in ['explaining', 'rhetorical_explaining', 'opining', 'building'])
        questioning_count = sum(1 for u in utterances if u['intent'] in ['questioning', 'probing', 'clarifying'])

        if explaining_count > questioning_count * 2:
            primary_role = "explainer"
            role_description = f"{speaker_alias} primarily explains and elaborates, with {explaining_count} explanatory utterances vs {questioning_count} questions."
        elif questioning_count > explaining_count * 2:
            primary_role = "questioner"
            role_description = f"{speaker_alias} primarily asks questions to probe and clarify, with {questioning_count} questions vs {explaining_count} explanatory utterances."
        else:
            primary_role = "balanced"
            role_description = f"{speaker_alias} balances questioning and explaining, with {questioning_count} questions and {explaining_count} explanatory utterances."

        return {
            "tool_name": "get_speaker_utterances",
            "speaker": speaker_alias,
            "speaker_id": speaker_id,
            "session_scope": session_id if session_id else "all sessions",
            "role_summary": {
                "primary_role": primary_role,
                "description": role_description,
                "intent_distribution": intent_counts,
                "genuine_questions": len(genuine_questions),
                "rhetorical_questions": len(rhetorical_questions),
                "interpretation_note": "Questions marked 'is_self_answered=True' are rhetorical - the speaker answers their own question, indicating they are explaining rather than genuinely asking."
            },
            "summary": {
                "total_utterances": total_utterances,
                "total_words": total_words,
                "questions_asked": len(questions),
                "genuine_questions": len(genuine_questions),
                "rhetorical_questions": len(rhetorical_questions),
                "sessions_participated": len(set(u['session_id'] for u in utterances))
            },
            "utterances": utterances,
            "questions_asked": questions,
            "genuine_questions": genuine_questions,
            "rhetorical_questions": rhetorical_questions,
            "is_relevant": True,
            "result_count": len(utterances)
        }

    except Exception as e:
        logger.error(f"Get speaker utterances error: {e}")
        return {"tool_name": "get_speaker_utterances", "error": str(e), "is_relevant": False}


def get_speaker_profile(speaker_name: str, session_id: int = None) -> dict:
    """
    Get speaker profile with concept map contributions and connections.

    Use this for understanding a speaker's ROLE and CONCEPT contributions.
    For their actual UTTERANCES, use get_speaker_utterances instead.

    Args:
        speaker_name: Name of the speaker (fuzzy matched)
        session_id: Optional - limit to specific session

    Returns:
        - transcript_summary: participation stats, sample quotes
        - concept_summary: concepts contributed, types, connections to other speakers
    """
    return _get_speaker_profile_v3(speaker_name, session_id)


# =============================================================================
# TOOL DEFINITIONS FOR CLAUDE API
# =============================================================================

# Tool schemas following Anthropic's tool_use format
TOOL_SCHEMAS = [
    {
        "name": "list_sessions",
        "description": """List all available discussion sessions.

Use this FIRST to discover what sessions exist and what data is available.

Returns for each session:
- session_id: Unique identifier
- session_name: Human-readable name (e.g., "Nuclear Fusion", "Is AI Alive")
- speakers: List of participant names
- artifacts_available: Which data types exist (transcript, concept_map, collaboration)

Example: To see all sessions, call list_sessions()""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "search_sessions",
        "description": """Find sessions relevant to a topic using semantic search.

Use when you need to find which sessions discuss a particular topic.

Args:
    query: What to search for (e.g., "nuclear energy", "collaboration challenges")
    top_k: How many results to return (default: 3)

Returns ranked sessions with relevance scores and preview of matching content.

Example: search_sessions(query="artificial intelligence consciousness")""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or keywords to search for"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_transcript",
        "description": """Get the complete transcript for a session.

Use this to see WHAT participants actually said.

Args:
    session_id: The session ID (integer)

Returns:
- summary: Total utterances, words, questions asked, avg linguistic scores
- speaker_profiles: Per-speaker statistics (utterance count, word count, questions asked)
- utterances: Full transcript with speaker, text, timestamp, word count, linguistic scores

The transcript includes LIWC linguistic scores per utterance:
- analytic_thinking: Logical, formal reasoning (0-100)
- certainty: Conviction and definitiveness (0-100)

Example: get_transcript(session_id=20) returns the Nuclear Fusion session transcript""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "The session ID to retrieve"
                }
            },
            "required": ["session_id"]
        }
    },
    {
        "name": "get_concept_map",
        "description": """Get the concept map for a session showing how ideas connect.

Use this to see the STRUCTURE of reasoning - how concepts relate to each other.

Args:
    session_id: The session ID (integer)

Returns:
- summary: Node counts by type, speaker contributions
- nodes: All concepts with type (claim, question, evidence, etc.), text, and speaker
- edges: Relationships between concepts (supports, challenges, causes, etc.)
- clusters: Thematic groupings of related concepts
- reasoning_patterns: Detected patterns (causal chains, hypothesis testing, Q&A pairs)
- hub_nodes: Most connected concepts (central ideas)

Example: get_concept_map(session_id=20) shows how ideas connect in the Nuclear Fusion discussion""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "The session ID to retrieve"
                }
            },
            "required": ["session_id"]
        }
    },
    {
        "name": "get_7c_analysis",
        "description": """Get the 7C collaboration quality analysis for a session.

Use this to see HOW WELL the group collaborated, with quantitative scores.

Args:
    session_id: The session ID (integer)

Returns scores (0-100) for 7 collaboration dimensions:
- climate: Psychological safety, supportive atmosphere
- communication: Clarity, active listening, articulation
- contribution: Balanced participation, equal voice
- conflict: Constructive disagreement handling
- context: Shared understanding, common ground
- constructive: Building on others' ideas
- compatibility: Working style alignment

Also returns:
- overall_score: Average across dimensions
- strengths: Top-scoring dimensions
- areas_for_improvement: Low-scoring dimensions
- evidence: Coded segments supporting each score

Example: get_7c_analysis(session_id=20) shows collaboration quality metrics""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "The session ID to retrieve"
                }
            },
            "required": ["session_id"]
        }
    },
    {
        "name": "get_speaker_utterances",
        "description": """Get ALL utterances from a specific speaker with enhanced attribution context.

CRITICAL: Use this tool when asked about what a speaker SAID, ASKED, or EXPLAINED.
This returns the COMPLETE list of utterances with intent classification.

Args:
    speaker_name: Name of the speaker (fuzzy matched)
    session_id: Session to search in (STRONGLY RECOMMENDED - specify this!)

Returns:
- role_summary: Whether speaker is primarily 'explainer', 'questioner', or 'balanced'
- summary: total utterances, words, genuine questions, rhetorical questions
- utterances: ALL utterances with:
  - text: Full utterance text
  - is_question: Whether marked as question
  - is_self_answered: TRUE if speaker answers their own question (rhetorical)
  - intent: 'explaining', 'questioning', 'rhetorical_explaining', 'probing', etc.
  - context: Who spoke before/after (when session_id specified)
- genuine_questions: Questions where speaker seeks information
- rhetorical_questions: Questions speaker answers themselves (actually explaining)

INTERPRETATION GUIDANCE:
- A speaker with many questions but high 'is_self_answered' is explaining via rhetorical questions
- Check 'role_summary.primary_role' for overall characterization
- Use 'intent' field to understand purpose of each utterance

Example: get_speaker_utterances(speaker_name="Sam", session_id=19)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "speaker_name": {
                    "type": "string",
                    "description": "Name of the speaker to look up"
                },
                "session_id": {
                    "type": "integer",
                    "description": "Session ID to search in (strongly recommended)"
                }
            },
            "required": ["speaker_name"]
        }
    },
    {
        "name": "get_speaker_profile",
        "description": """Get a speaker's CONCEPT MAP contributions and how their ideas connect.

Use this to understand a speaker's ROLE and CONCEPT contributions in the discussion.
For their actual UTTERANCES (what they said), use get_speaker_utterances instead.

Args:
    speaker_name: Name of the speaker (fuzzy matched)
    session_id: Optional - limit to specific session (None = all sessions)

Returns:
- transcript_summary: Sessions participated, utterance counts, questions asked
- concept_summary: Concepts contributed, types (claims, questions, etc.)
- connections: How this speaker's ideas connect to others' ideas

Example: get_speaker_profile(speaker_name="David", session_id=20)""",
        "input_schema": {
            "type": "object",
            "properties": {
                "speaker_name": {
                    "type": "string",
                    "description": "Name of the speaker to look up"
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
        "description": """Compare multiple sessions on collaboration quality and participation.

Use this when asked to compare, rank, or find differences between sessions.

Args:
    session_ids: List of session IDs to compare (or None for all sessions)
    metric: What to compare - "collaboration" (7C scores), "participation", or "all"

Returns:
- Ranked comparison across sessions
- Key differences and similarities
- Best/worst performers on each metric

Example: compare_sessions(session_ids=[18, 19, 20], metric="collaboration")
Example: compare_sessions(metric="all") compares all available sessions""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of session IDs to compare (omit for all sessions)"
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
        "description": """Find the reasoning path between two concepts in a concept map.

Use this to trace how one idea led to another - the chain of reasoning.

Args:
    session_id: The session to search in
    from_concept: Starting concept text (fuzzy matched)
    to_concept: Target concept text (fuzzy matched)
    max_depth: Maximum path length (default: 5)

Returns:
- path: Each step showing concept -> relationship -> concept
- narrative: Human-readable description of the reasoning chain
- speakers_involved: Who contributed to this chain of reasoning

Example: find_concept_path(session_id=20, from_concept="fusion reactor", to_concept="clean energy")""",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "integer",
                    "description": "The session to search in"
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

# Baseline tools (transcript only)
BASELINE_TOOL_SCHEMAS = [
    TOOL_SCHEMAS[0],  # list_sessions
    TOOL_SCHEMAS[1],  # search_sessions
    TOOL_SCHEMAS[2],  # get_transcript
]


# =============================================================================
# TOOL EXECUTION
# =============================================================================

# Map tool names to functions
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

BASELINE_TOOL_FUNCTIONS = {
    "list_sessions": list_sessions,
    "search_sessions": search_for_sessions,
    "get_transcript": get_transcript,
}


def execute_tool(tool_name: str, tool_input: dict, mode: str = "enhanced") -> dict:
    """Execute a tool and return the result.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool
        mode: "enhanced" or "baseline"

    Returns:
        Tool execution result
    """
    functions = TOOL_FUNCTIONS if mode == "enhanced" else BASELINE_TOOL_FUNCTIONS

    if tool_name not in functions:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = functions[tool_name](**tool_input)
        return result
    except Exception as e:
        return {"error": str(e)}


def get_tool_schemas(mode: str = "enhanced") -> list:
    """Get tool schemas for the specified mode.

    Args:
        mode: "enhanced" (all tools) or "baseline" (transcript only)

    Returns:
        List of tool schemas
    """
    if mode == "baseline":
        return BASELINE_TOOL_SCHEMAS
    return TOOL_SCHEMAS
