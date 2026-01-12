"""
Query Understanding for Agent V5.

Principled approach:
- Session/speaker resolution via database lookup with fuzzy matching
- Intent classification via LLM (with heuristic fallback)
- No hard-coded entity lists - everything is dynamic
"""

import re
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
import mysql.connector

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Structured representation of query intent."""

    # Primary intent type
    intent_type: str  # describe, compare, explain, speaker, explore, search

    # Retrieval strategy
    retrieval_mode: str  # structured, semantic, contrastive, hybrid, agentic_only

    # Extracted entities
    session_ids: List[int] = field(default_factory=list)
    session_names: List[str] = field(default_factory=list)
    speaker_names: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    # For contrastive queries
    metric_focus: Optional[str] = None

    # Confidence in classification
    confidence: float = 0.8

    # Whether query requires data retrieval
    needs_retrieval: bool = True

    # Collections to search for semantic retrieval
    target_collections: List[str] = field(default_factory=lambda: ['transcripts', 'concepts', 'seven_c'])


def _get_db_connection():
    """Get database connection."""
    return mysql.connector.connect(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        user=os.getenv('MYSQL_USER', 'vagrant'),
        password=os.getenv('MYSQL_PASSWORD', 'vagrant'),
        database=os.getenv('MYSQL_DATABASE', 'discussion_capture')
    )


# =============================================================================
# DYNAMIC ENTITY EXTRACTION (from database)
# =============================================================================

@lru_cache(maxsize=1)
def _get_all_sessions() -> List[Dict]:
    """
    Get all session names from database.
    Cached to avoid repeated queries. Call clear_caches() when data changes.
    """
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

        return sessions

    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return []


@lru_cache(maxsize=1)
def _get_all_speakers() -> List[Dict]:
    """
    Get all speaker names from database.
    Cached to avoid repeated queries.
    """
    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)

        cursor.execute("SELECT id, alias FROM speaker WHERE alias IS NOT NULL AND alias != ''")
        speakers = cursor.fetchall()

        cursor.close()
        connection.close()

        return speakers

    except Exception as e:
        logger.error(f"Error getting speakers: {e}")
        return []


def _fuzzy_match_session(query: str) -> List[Tuple[int, str, float]]:
    """
    Find session mentions in query using fuzzy matching against database.
    Returns: List of (session_id, matched_name, match_quality)
    """
    sessions = _get_all_sessions()
    query_lower = query.lower()
    query_words = set(query_lower.split())
    matches = []

    # Check for explicit session ID pattern first
    id_match = re.search(r'\bsession\s*(\d+)\b', query_lower)
    if id_match:
        sid = int(id_match.group(1))
        # Verify this session exists
        for s in sessions:
            if s['session_device_id'] == sid:
                matches.append((sid, s['name'], 1.0))
                break

    # Check each session name
    for s in sessions:
        name = s['name'].lower().strip()
        sid = s['session_device_id']

        # Skip if already matched by ID
        if any(m[0] == sid for m in matches):
            continue

        # Exact substring match
        if name in query_lower:
            matches.append((sid, s['name'], 1.0))
            continue

        # Word overlap matching
        name_words = set(name.split())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'in', 'on', 'of', 'is', 'are', 'was', 'were'}
        significant_name_words = name_words - stop_words

        if significant_name_words:
            overlap = significant_name_words & query_words
            if overlap:
                quality = len(overlap) / len(significant_name_words)
                # Require at least 50% match or all significant words if only 1-2
                if quality >= 0.5 or (len(significant_name_words) <= 2 and len(overlap) >= 1):
                    matches.append((sid, s['name'], quality))

    # Sort by quality, deduplicate
    matches.sort(key=lambda x: x[2], reverse=True)
    seen = set()
    unique = []
    for m in matches:
        if m[0] not in seen:
            seen.add(m[0])
            unique.append(m)

    return unique


def _fuzzy_match_speaker(query: str) -> List[str]:
    """
    Find speaker mentions in query by matching against database speakers.
    Returns: List of speaker names (properly capitalized)
    """
    speakers = _get_all_speakers()
    query_lower = query.lower()
    matches = []

    for s in speakers:
        alias = s['alias']
        if not alias:
            continue

        # Check for whole word match (case-insensitive)
        pattern = rf'\b{re.escape(alias.lower())}\b'
        if re.search(pattern, query_lower):
            matches.append(alias)

    return matches


def _extract_topics(query: str) -> List[str]:
    """Extract potential topics/concepts from query using patterns."""
    topics = []

    topic_patterns = [
        r'about\s+([^?.,]+)',
        r'discuss(?:ing|ed|es)?\s+([^?.,]+)',
        r'mention(?:ing|ed|s)?\s+([^?.,]+)',
        r'talk(?:ing|ed|s)?\s+about\s+([^?.,]+)',
        r'related to\s+([^?.,]+)',
    ]

    for pattern in topic_patterns:
        matches = re.findall(pattern, query, re.I)
        for match in matches:
            topic = match.strip()
            if len(topic) > 2 and topic not in topics:
                topics.append(topic)

    return topics[:3]


# =============================================================================
# INTENT CLASSIFICATION (LLM-based with heuristic fallback)
# =============================================================================

def _classify_intent_with_llm(
    query: str,
    has_sessions: bool,
    has_speakers: bool
) -> Tuple[str, str, Optional[str]]:
    """
    Use LLM to classify query intent.
    Falls back to heuristics if LLM unavailable.
    """
    try:
        from openai import OpenAI
        client = OpenAI()

        prompt = f"""Classify this query about educational discussion analysis.

Query: "{query}"

Context:
- Has specific session mentioned: {has_sessions}
- Has specific speaker mentioned: {has_speakers}

Classify into exactly ONE intent type:
- "describe": User wants information about a specific session
- "speaker": User wants to know what a specific person said/asked/contributed
- "compare": User wants to compare multiple sessions or speakers
- "explain": User wants to understand WHY (e.g., why some discussions are better)
- "search": User wants to find sessions about a topic
- "explore": Open-ended exploration, no specific target

Determine retrieval mode:
- "structured": Direct database lookup (specific entities mentioned)
- "semantic": Similarity search (topic-based search)
- "contrastive": Compare high vs low performers (for "why" questions)
- "agentic_only": Let agent decide (very open queries)

If about collaboration quality, identify dimension (or null):
communication, climate, contribution, conflict, constructive, context, compatibility

Respond ONLY in this format:
INTENT: <type>
MODE: <mode>
METRIC: <dimension or null>"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0
        )

        result = response.choices[0].message.content

        intent_match = re.search(r'INTENT:\s*(\w+)', result)
        mode_match = re.search(r'MODE:\s*(\w+)', result)
        metric_match = re.search(r'METRIC:\s*(\w+|null)', result)

        intent = intent_match.group(1).lower() if intent_match else "explore"
        mode = mode_match.group(1).lower() if mode_match else "agentic_only"
        metric = metric_match.group(1).lower() if metric_match and metric_match.group(1).lower() != "null" else None

        # Validate
        if intent not in ['describe', 'speaker', 'compare', 'explain', 'search', 'explore']:
            intent = "explore"
        if mode not in ['structured', 'semantic', 'contrastive', 'agentic_only', 'hybrid']:
            mode = "agentic_only"

        logger.debug(f"LLM classification: intent={intent}, mode={mode}, metric={metric}")
        return (intent, mode, metric)

    except Exception as e:
        logger.info(f"LLM classification unavailable ({e}), using heuristics")
        return _classify_intent_heuristic(query, has_sessions, has_speakers)


def _classify_intent_heuristic(
    query: str,
    has_sessions: bool,
    has_speakers: bool
) -> Tuple[str, str, Optional[str]]:
    """
    Fallback heuristic classification when LLM is unavailable.
    Uses regex patterns - less accurate but works offline.
    """
    query_lower = query.lower()

    # Speaker-focused patterns
    if has_speakers:
        speaker_patterns = [
            r'\bwhat did\b', r'\bsaid\b', r'\basked\b', r'\bquestions?\b',
            r'\brole\b', r'\bcontribut', r'\bparticipat', r'\bhow did\b'
        ]
        if any(re.search(p, query_lower) for p in speaker_patterns):
            return ('speaker', 'structured' if has_sessions else 'semantic', None)

    # Comparison patterns
    if re.search(r'\b(compare|versus|vs\.?|difference between)\b', query_lower):
        return ('compare', 'structured', None)

    # Why/explain patterns (contrastive)
    if re.search(r'\bwhy\b.*\b(better|worse|more|less|higher|lower)\b', query_lower):
        return ('explain', 'contrastive', _detect_metric_heuristic(query_lower))

    if re.search(r'\b(why do some|what makes|why are some)\b', query_lower):
        return ('explain', 'contrastive', _detect_metric_heuristic(query_lower))

    # Describe patterns
    if has_sessions:
        if re.search(r'\b(tell me about|what happened|describe|summariz|overview|explain)\b', query_lower):
            return ('describe', 'structured', None)
        return ('describe', 'structured', None)

    # Search patterns
    if re.search(r'\b(find|search|which sessions?|what sessions?)\b', query_lower):
        return ('search', 'semantic', None)

    # Explore patterns
    if re.search(r'\b(patterns?|across|common|theme|all sessions)\b', query_lower):
        return ('explore', 'semantic', None)

    return ('explore', 'agentic_only', None)


def _detect_metric_heuristic(query: str) -> Optional[str]:
    """Detect which 7C dimension the query focuses on."""
    # Map keywords to dimensions
    dimension_keywords = {
        'communication': ['communication', 'communicate', 'exchange', 'dialogue', 'listening'],
        'climate': ['climate', 'atmosphere', 'supportive', 'friendly', 'safe', 'psychological'],
        'contribution': ['contribution', 'participate', 'participation', 'balance', 'equal'],
        'conflict': ['conflict', 'disagree', 'argument', 'tension', 'debate'],
        'constructive': ['constructive', 'build on', 'building', 'extend', 'elaborate'],
        'context': ['context', 'shared understanding', 'common ground'],
        'compatibility': ['compatibility', 'working style', 'alignment'],
    }

    query_lower = query.lower()
    for dimension, keywords in dimension_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return dimension

    # Default for general quality queries
    if any(w in query_lower for w in ['better', 'good', 'effective', 'quality']):
        return 'communication'

    return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def understand_query(
    query: str,
    conversation_context: Optional[Dict] = None
) -> QueryIntent:
    """
    Analyze query to determine intent and optimal retrieval strategy.

    This implementation is principled:
    1. Entity extraction via database lookup (no hard-coded lists)
    2. Intent classification via LLM (with heuristic fallback)
    3. Dynamic session/speaker resolution

    Args:
        query: User's natural language query
        conversation_context: Previous conversation state

    Returns:
        QueryIntent with classified intent and retrieval strategy
    """
    conversation_context = conversation_context or {}

    # Extract entities from database
    session_matches = _fuzzy_match_session(query)
    session_ids = [m[0] for m in session_matches]
    session_names = [m[1] for m in session_matches]

    speaker_names = _fuzzy_match_speaker(query)
    topics = _extract_topics(query)

    # Inherit session from conversation context if no explicit mention
    context_session = conversation_context.get('session_focus')
    if context_session and not session_ids:
        session_ids = [context_session]

    # Classify intent (LLM with fallback)
    intent_type, retrieval_mode, metric_focus = _classify_intent_with_llm(
        query,
        has_sessions=bool(session_ids),
        has_speakers=bool(speaker_names)
    )

    # Determine target collections
    target_collections = _determine_collections(intent_type, query)

    # Check if retrieval is needed
    needs_retrieval = _needs_retrieval(query)

    return QueryIntent(
        intent_type=intent_type,
        retrieval_mode=retrieval_mode,
        session_ids=session_ids,
        session_names=session_names,
        speaker_names=speaker_names,
        topics=topics,
        metric_focus=metric_focus,
        target_collections=target_collections,
        needs_retrieval=needs_retrieval
    )


def _determine_collections(intent_type: str, query: str) -> List[str]:
    """Determine which collections to search based on intent."""
    query_lower = query.lower()

    if intent_type == 'speaker':
        return ['transcripts']

    if intent_type == 'explain' or 'collaboration' in query_lower or '7c' in query_lower:
        return ['seven_c', 'transcripts']

    if any(w in query_lower for w in ['concept', 'idea', 'argumentation']):
        return ['concepts', 'transcripts']

    return ['transcripts', 'concepts', 'seven_c']


def _needs_retrieval(query: str) -> bool:
    """Check if query needs data retrieval."""
    query_lower = query.lower().strip()

    # Greetings don't need retrieval
    if query_lower in ['hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye']:
        return False

    # Meta questions
    if re.search(r'\b(what can you|how do i|help me|what are you)\b', query_lower):
        return False

    return True


def get_session_name(session_id: int) -> Optional[str]:
    """Get session name from database by ID."""
    sessions = _get_all_sessions()
    for s in sessions:
        if s['session_device_id'] == session_id:
            return s['name']
    return None


def clear_caches():
    """Clear cached data. Call when database changes."""
    _get_all_sessions.cache_clear()
    _get_all_speakers.cache_clear()
