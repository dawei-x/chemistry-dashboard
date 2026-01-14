"""
Query Analysis Module for Agent V6.

Handles:
- Entity extraction (sessions, speakers) via database lookup
- Steering extraction (prefer/exclude representations) from API params and NL patterns
- Mode detection (hypothesis, compare, trace)
- Abstract construct detection for operationalization

This is NOT decomposition - just understanding what's being asked.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

import mysql.connector

from .domain_knowledge import detect_constructs_in_query

logger = logging.getLogger(__name__)

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QueryAnalysis:
    """Complete analysis of a user query."""
    # Entities
    session_ids: List[int] = field(default_factory=list)
    session_names: List[str] = field(default_factory=list)
    speaker_names: List[str] = field(default_factory=list)

    # Steering
    prefer_representations: List[str] = field(default_factory=list)
    exclude_representations: List[str] = field(default_factory=list)

    # Mode
    mode: str = "default"  # default, test_hypothesis, compare, trace

    # Abstract constructs needing operationalization
    constructs: List[str] = field(default_factory=list)

    # Original query
    query: str = ""


# =============================================================================
# STEERING PATTERNS (from V3)
# =============================================================================

STEERING_PATTERNS = {
    # Representation preferences
    'prefer_transcript': [
        r'focus on.*transcript',
        r'just.*transcript',
        r'only.*transcript',
        r'what.*(?:did|does).*(?:say|said)',
        r'exact.*(?:words|quotes)',
    ],
    'prefer_concept_map': [
        r'focus on.*concept.*map',
        r'how.*ideas.*connected',
        r'concept.*relationships?',
        r'intellectual.*structure',
    ],
    'prefer_collaboration': [
        r'focus on.*(?:7c|collaboration)',
        r'collaboration.*(?:scores?|metrics?|quality)',
        r'(?:7c|seven.*c).*(?:scores?|analysis)',
    ],

    # Exclusions
    'exclude_transcript': [
        r'without.*transcript',
        r'ignore.*transcript',
        r'don\'t.*use.*transcript',
    ],
    'exclude_concept_map': [
        r'without.*concept.*map',
        r'ignore.*concept.*map',
        r'don\'t.*use.*concept.*map',
    ],
    'exclude_collaboration': [
        r'without.*(?:7c|collaboration)',
        r'ignore.*(?:7c|collaboration)',
        r'don\'t.*use.*(?:7c|collaboration)',
    ],

    # Modes
    'hypothesis': [
        r'\bi\s+think\b',
        r'verify\s+(?:this|that|if)',
        r'test\s+(?:this|the)?\s*hypothesis',
        r'is\s+it\s+true\s+that',
        r'can\s+you\s+(?:verify|confirm)',
    ],
    'compare': [
        r'\bcompare\b',
        r'difference\s+between',
        r'\bvs\.?\b',
        r'versus',
        r'which\s+(?:one|session)\s+(?:is|has|was)',
        r'better\s+(?:than|or\s+worse)',
    ],
    'trace': [
        r'trace\s+(?:the\s+)?path',
        r'connection\s+between.*concept',
        r'how\s+(?:did|does).*(?:evolve|develop)',
        r'evolution\s+of',
    ],
}


# =============================================================================
# DATABASE ACCESS (from V5, principled)
# =============================================================================

def _get_db_connection():
    """Get database connection."""
    return mysql.connector.connect(
        host='127.0.0.1',
        user='vagrant',
        password='vagrant',
        database='discussion_capture'
    )


@lru_cache(maxsize=1)
def _get_all_sessions() -> Tuple[Tuple]:
    """Get all sessions from database (cached)."""
    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)
        cursor.execute("""
            SELECT sd.id as session_device_id, s.name
            FROM session_device sd
            JOIN session s ON sd.session_id = s.id
        """)
        sessions = cursor.fetchall()
        cursor.close()
        connection.close()
        # Convert to tuple for caching
        return tuple((s['session_device_id'], s['name']) for s in sessions)
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return ()


@lru_cache(maxsize=1)
def _get_all_speakers() -> Tuple[str]:
    """Get all speakers from database (cached)."""
    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)
        cursor.execute("""
            SELECT DISTINCT speaker_alias
            FROM utterance
            WHERE speaker_alias IS NOT NULL
            AND speaker_alias != ''
        """)
        speakers = cursor.fetchall()
        cursor.close()
        connection.close()
        return tuple(s['speaker_alias'] for s in speakers)
    except Exception as e:
        logger.error(f"Error getting speakers: {e}")
        return ()


def _fuzzy_match_session(query: str) -> List[Tuple[int, str, float]]:
    """
    Fuzzy match session names from query.
    Returns list of (session_id, name, quality_score).
    """
    sessions = _get_all_sessions()
    if not sessions:
        return []

    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))

    # Stop words to ignore
    stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
                  'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where',
                  'session', 'discussion', 'about', 'tell', 'me', 'can', 'you'}

    matches = []

    for session_id, name in sessions:
        name_lower = name.lower()
        name_words = set(re.findall(r'\b\w+\b', name_lower)) - stop_words

        if not name_words:
            continue

        # Calculate word overlap
        query_content_words = query_words - stop_words
        overlap = name_words & query_content_words

        if overlap:
            # Quality score: proportion of name words matched
            quality = len(overlap) / len(name_words)

            # Require at least 50% match or 1 word for short names
            if quality >= 0.5 or (len(name_words) <= 2 and len(overlap) >= 1):
                matches.append((session_id, name, quality))

    # Sort by quality descending
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


def _match_speakers(query: str) -> List[str]:
    """Match speaker names from query."""
    speakers = _get_all_speakers()
    if not speakers:
        return []

    query_lower = query.lower()
    matched = []

    for speaker in speakers:
        # Whole word match (case insensitive)
        pattern = r'\b' + re.escape(speaker.lower()) + r'\b'
        if re.search(pattern, query_lower):
            matched.append(speaker)

    return matched


def _extract_session_ids(query: str) -> List[int]:
    """Extract explicit session IDs from query."""
    # Match patterns like "session 20", "session ID 20", "#20"
    patterns = [
        r'session\s+(?:id\s+)?(\d+)',
        r'#(\d+)',
        r'\bsession\s*(\d+)\b',
    ]

    ids = []
    query_lower = query.lower()

    for pattern in patterns:
        for match in re.finditer(pattern, query_lower, re.IGNORECASE):
            try:
                ids.append(int(match.group(1)))
            except ValueError:
                pass

    return list(set(ids))


# =============================================================================
# STEERING EXTRACTION
# =============================================================================

def _extract_steering_from_patterns(query: str) -> Dict:
    """Extract steering from natural language patterns."""
    query_lower = query.lower()

    prefer = []
    exclude = []
    mode = "default"

    # Check exclusion patterns FIRST (they take precedence)
    for pattern in STEERING_PATTERNS['exclude_transcript']:
        if re.search(pattern, query_lower, re.IGNORECASE):
            exclude.append('transcript')
            break

    for pattern in STEERING_PATTERNS['exclude_concept_map']:
        if re.search(pattern, query_lower, re.IGNORECASE):
            exclude.append('concept_map')
            break

    for pattern in STEERING_PATTERNS['exclude_collaboration']:
        if re.search(pattern, query_lower, re.IGNORECASE):
            exclude.append('collaboration')
            break

    # Check preference patterns (but skip if already excluded)
    if 'transcript' not in exclude:
        for pattern in STEERING_PATTERNS['prefer_transcript']:
            if re.search(pattern, query_lower, re.IGNORECASE):
                prefer.append('transcript')
                break

    if 'concept_map' not in exclude:
        for pattern in STEERING_PATTERNS['prefer_concept_map']:
            if re.search(pattern, query_lower, re.IGNORECASE):
                prefer.append('concept_map')
                break

    if 'collaboration' not in exclude:
        for pattern in STEERING_PATTERNS['prefer_collaboration']:
            if re.search(pattern, query_lower, re.IGNORECASE):
                prefer.append('collaboration')
                break

    # Check mode patterns
    for pattern in STEERING_PATTERNS['hypothesis']:
        if re.search(pattern, query_lower, re.IGNORECASE):
            mode = "test_hypothesis"
            break

    if mode == "default":
        for pattern in STEERING_PATTERNS['compare']:
            if re.search(pattern, query_lower, re.IGNORECASE):
                mode = "compare"
                break

    if mode == "default":
        for pattern in STEERING_PATTERNS['trace']:
            if re.search(pattern, query_lower, re.IGNORECASE):
                mode = "trace"
                break

    return {
        'prefer': list(set(prefer)),
        'exclude': list(set(exclude)),
        'mode': mode
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_query(
    query: str,
    api_params: Optional[Dict] = None,
    conversation_context: Optional[Dict] = None
) -> QueryAnalysis:
    """
    Analyze a query to extract entities, steering, and mode.

    Args:
        query: The user's query
        api_params: Optional API parameters (prefer_representations, exclude_representations, mode)
        conversation_context: Optional conversation context (session_focus, speaker_focus)

    Returns:
        QueryAnalysis with all extracted information
    """
    api_params = api_params or {}
    conversation_context = conversation_context or {}

    analysis = QueryAnalysis(query=query)

    # 1. Extract explicit session IDs
    explicit_ids = _extract_session_ids(query)
    analysis.session_ids.extend(explicit_ids)

    # 2. Fuzzy match session names
    session_matches = _fuzzy_match_session(query)
    for session_id, name, quality in session_matches:
        if session_id not in analysis.session_ids:
            analysis.session_ids.append(session_id)
            analysis.session_names.append(name)

    # 3. Add session from conversation context if no session found
    if not analysis.session_ids and conversation_context.get('session_focus'):
        analysis.session_ids.append(conversation_context['session_focus'])

    # 4. Match speakers
    analysis.speaker_names = _match_speakers(query)

    # Add speaker from conversation context if relevant
    if not analysis.speaker_names and conversation_context.get('speaker_focus'):
        # Only add if query seems to reference previous context
        if any(word in query.lower() for word in ['they', 'their', 'them', 'he', 'she', 'same']):
            analysis.speaker_names.append(conversation_context['speaker_focus'])

    # 5. Extract steering from NL patterns
    nl_steering = _extract_steering_from_patterns(query)

    # 6. Merge API params with NL steering (API takes precedence)
    analysis.prefer_representations = api_params.get('prefer_representations', []) or nl_steering['prefer']
    analysis.exclude_representations = api_params.get('exclude_representations', []) or nl_steering['exclude']
    analysis.mode = api_params.get('mode') or nl_steering['mode']

    # 7. Detect abstract constructs needing operationalization
    analysis.constructs = detect_constructs_in_query(query)

    logger.info(f"[V6 Query Analysis] sessions={analysis.session_ids}, "
                f"speakers={analysis.speaker_names}, mode={analysis.mode}, "
                f"prefer={analysis.prefer_representations}, exclude={analysis.exclude_representations}, "
                f"constructs={analysis.constructs}")

    return analysis


def clear_cache():
    """Clear cached database lookups (useful for testing)."""
    _get_all_sessions.cache_clear()
    _get_all_speakers.cache_clear()
