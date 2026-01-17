"""
Query Classifier for BLINC Agent V7

Classifies queries as exploratory (cross-session) or targeted (single-session).
Uses heuristic patterns for speed and predictability.

Exploratory queries require systematic multi-session retrieval.
Targeted queries can use the simple ReAct loop.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .memory import ConversationMemory

logger = logging.getLogger(__name__)


# =============================================================================
# Exploratory Query Patterns
# =============================================================================

EXPLORATORY_PATTERNS = [
    # Explicit cross-session language
    r'\bacross\s+(?:all\s+)?sessions?\b',
    r'\ball\s+sessions?\b',
    r'\bevery\s+session\b',
    r'\bmultiple\s+sessions?\b',

    # Session discovery
    r'\bfind\s+sessions?\b',
    r'\bwhich\s+sessions?\b',
    r'\bwhat\s+sessions?\b',
    r'\blist\s+sessions?\b',
    r'\bany\s+sessions?\b',
    r'\bhow\s+many\s+sessions?\b',
    r'\bsessions?\s+(?:with|showing|that|where|discussed)\b',

    # Comparison/superlative patterns
    r'\bcompare\b.*\bsessions?\b',
    r'\bcomparison\b',
    r'\bbest\s+(?:collaboration|session|discussion)\b',
    r'\bworst\s+(?:collaboration|session|discussion)\b',
    r'\bmost\s+(?:collaborative|productive|engaged)\b',
    r'\bhighest\s+(?:score|collaboration|engagement)\b',
    r'\blowest\s+(?:score|collaboration|engagement)\b',
    r'\bwhich\s+(?:session|one)\s+(?:is\s+)?(?:better|best|worse|worst)\b',
    # Comparative patterns (cross-session by nature)
    r'\bhigher\s+(?:engagement|collaboration|score)\b',
    r'\blower\s+(?:engagement|collaboration|score)\b',
    r'\bwhy\s+(?:did|do|does)\s+some\b',  # "Why did some discussions..."

    # Aggregation patterns
    r'\bwho\s+(?:asked|contributed|said)\s+(?:the\s+)?most\b',
    r'\btotal\s+(?:across|from)\b',
    r'\boverall\b.*\bsessions?\b',

    # Topic search without session context
    r'\bwhat\s+was\s+said\s+about\b(?!.*\bsession\s+\d+)',
    r'\bwho\s+discussed\b(?!.*\bsession\s+\d+)',

    # Discovery/search patterns - imply searching across sessions
    # These catch "find evidence of X", "find quotes where Y", etc.
    r'\bfind\s+(?:evidence|quotes|examples|instances|mentions)\b',
    r'\bshow\s+(?:me\s+)?(?:evidence|quotes|examples)\s+(?:of|where|showing|about)\b',
]

# Patterns that indicate a targeted query (override exploratory)
TARGETED_PATTERNS = [
    r'\bsession\s+\d+\b',           # Explicit session ID
    r'\bin\s+session\s+\d+\b',      # "in session 25"
    r'\bsession\s+\d+\'?s?\b',      # "session 25's"
]

# Patterns for speaker-statistics queries (should use get_speaker_profile)
# These are NOT exploratory - they ask for aggregate stats about a speaker
SPEAKER_STATS_PATTERNS = [
    r'\bhow\s+many\s+(?:questions|utterances|words)\s+did\s+\w+\s+(?:ask|say|speak)\b',
    r'\b\w+\'?s?\s+(?:speaking|communication|analytical)\s+style\b',
    r'\bdescribe\s+\w+\'?s?\s+(?:speaking|style|contributions?)\b',
    r'\b(?:questions?|utterances?|words?)\s+(?:count|total|number)\s+for\s+\w+\b',
]

# Patterns for artifact type hints
COLLABORATION_PATTERNS = [
    r'\bcollaborat',
    r'\b7c\b',
    r'\bengagement\b',
    r'\bparticipation\b',
    r'\bcontribution\b',
    r'\binteraction\b',
    r'\bquality\b.*\bdiscussion\b',
    # FIX: Added disagreement/conflict patterns (maps to 7C Conflict dimension)
    r'\bdisagreement',
    r'\bconflict\b',
    r'\bhandl(?:e|ed|ing)\b.*\bdisagreement',
    r'\bresolution\b',
]

CONCEPT_MAP_PATTERNS = [
    r'\bideas?\b',
    r'\bconcepts?\b',
    r'\bconnect',
    r'\brelationship',
    r'\bstructure\b',
    r'\bthemes?\b',
]


@dataclass
class QueryClassification:
    """Result of query classification."""
    is_exploratory: bool
    session_ids: List[int] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    artifact_hint: Optional[str] = None  # transcript, concept_map, collaboration
    suggested_tool: Optional[str] = None  # Hint for ReAct: e.g., 'get_speaker_profile'
    reason: str = ""


def classify_query(
    query: str,
    memory: Optional[ConversationMemory] = None
) -> QueryClassification:
    """
    Classify a query as exploratory or targeted.

    Args:
        query: The user's query
        memory: Optional conversation memory for context

    Returns:
        QueryClassification with is_exploratory flag and extracted metadata
    """
    query_lower = query.lower()

    # Check for explicit targeted patterns first (these override exploratory)
    for pattern in TARGETED_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            # Extract session ID
            session_id_match = re.search(r'session\s+(\d+)', query_lower)
            session_ids = [int(session_id_match.group(1))] if session_id_match else []

            return QueryClassification(
                is_exploratory=False,
                session_ids=session_ids,
                speakers=_extract_speakers(query),
                topics=_extract_topics(query),
                artifact_hint=_determine_artifact_hint(query_lower),
                reason=f"Targeted: explicit session reference"
            )

    # Check for speaker-statistics queries (should use get_speaker_profile via ReAct)
    # These are NOT exploratory - they ask for aggregate stats about a specific speaker
    extracted_speakers = _extract_speakers(query)
    if extracted_speakers:
        for pattern in SPEAKER_STATS_PATTERNS:
            if re.search(pattern, query_lower):
                logger.info(f"[Classifier] Speaker-statistics query detected for: {extracted_speakers}")
                return QueryClassification(
                    is_exploratory=False,  # Route to ReAct, not exploratory
                    session_ids=[],
                    speakers=extracted_speakers,
                    topics=_extract_topics(query),
                    artifact_hint=None,
                    suggested_tool='get_speaker_profile',  # Hint for ReAct
                    reason=f"Targeted speaker-stats: use get_speaker_profile for {extracted_speakers[0]}"
                )

    # Check for exploratory patterns
    for pattern in EXPLORATORY_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryClassification(
                is_exploratory=True,
                session_ids=[],  # Will be discovered via search
                speakers=_extract_speakers(query),
                topics=_extract_topics(query),
                artifact_hint=_determine_artifact_hint(query_lower),
                reason=f"Exploratory: matched pattern '{pattern}'"
            )

    # Check memory for session context
    if memory and memory.session_focus:
        return QueryClassification(
            is_exploratory=False,
            session_ids=[memory.session_focus],
            speakers=_extract_speakers(query) or ([memory.speaker_focus] if memory.speaker_focus else []),
            topics=_extract_topics(query),
            artifact_hint=_determine_artifact_hint(query_lower),
            reason=f"Targeted: session {memory.session_focus} from conversation context"
        )

    # =======================================================================
    # PRINCIPLED FALLBACK: No session = must search
    # =======================================================================
    # If we reach here, we have:
    # - No explicit session ID (e.g., "session 20")
    # - No session from conversation memory
    # - No exploratory patterns matched
    #
    # A "targeted" query with no target is logically incoherent.
    # Without knowing which session to look at, we MUST use search_sessions
    # to discover relevant sessions. This is exploratory by definition.
    #
    # Examples that reach here:
    # - "What hypotheses were raised about AI?" → search for AI-related sessions
    # - "How are ideas connected to broader themes?" → search for relevant sessions
    # - "Tell me a joke" → search finds nothing → "no relevant sessions"
    #
    # This is more principled than pattern-matching content questions,
    # which is fragile and requires constant maintenance.
    # =======================================================================
    return QueryClassification(
        is_exploratory=True,
        session_ids=[],
        speakers=_extract_speakers(query),
        topics=_extract_topics(query),
        artifact_hint=_determine_artifact_hint(query_lower),
        reason="Exploratory: no session context - discovery required"
    )


def _extract_speakers(query: str) -> List[str]:
    """Extract speaker names mentioned in query.

    Uses database-loaded known speakers for reliable extraction.
    Matches any known speaker name that appears in the query with word boundaries.
    """
    from .memory import get_known_speakers

    # Get known speakers from database (cached)
    known_speakers = get_known_speakers()

    if not known_speakers:
        return []

    query_lower = query.lower()
    found_speakers = []

    # Check each known speaker with word boundary matching
    for speaker in known_speakers:
        speaker_lower = speaker.lower()
        # Use word boundary to avoid false matches
        pattern = r'\b' + re.escape(speaker_lower) + r'\b'
        if re.search(pattern, query_lower):
            found_speakers.append(speaker)

    return list(set(found_speakers))


def _extract_topics(query: str) -> List[str]:
    """Extract topic keywords from query.

    Handles both long topic words and short acronyms like "AI".
    """
    query_lower = query.lower()

    # First, extract specific known short acronyms/terms that might be topics
    short_topics = []
    known_short_topics = ['ai', 'ml', 'ar', 'vr', 'it', 'cs']
    for term in known_short_topics:
        # Match as whole word, case insensitive
        if re.search(rf'\b{term}\b', query_lower):
            short_topics.append(term)

    # Remove common question words and extract remaining significant words
    query_clean = re.sub(
        r'\b(what|who|how|why|when|where|which|did|does|do|was|were|is|are|the|a|an|in|on|about|across|all|sessions?|differ|different|compare|comparison)\b',
        '',
        query_lower
    )

    # Extract remaining words that might be topics (3+ chars)
    words = re.findall(r'\b([a-z]{3,})\b', query_clean)

    # Filter out very common non-topic words
    stop_words = {
        'said', 'discussed', 'asked', 'contributed', 'engaged', 'participated',
        'best', 'worst', 'most', 'least', 'handled', 'contributions'
    }
    long_topics = [w for w in words if w not in stop_words]

    # Combine short topics first (they're usually more specific), then long topics
    all_topics = short_topics + [t for t in long_topics if t not in short_topics]

    return all_topics[:5]  # Limit to top 5


def _determine_artifact_hint(query_lower: str) -> Optional[str]:
    """Determine which artifact type is most relevant for this query."""

    # Check collaboration patterns
    for pattern in COLLABORATION_PATTERNS:
        if re.search(pattern, query_lower):
            return 'collaboration'

    # Check concept map patterns
    for pattern in CONCEPT_MAP_PATTERNS:
        if re.search(pattern, query_lower):
            return 'concept_map'

    # Default to transcript for content queries
    if re.search(r'\b(?:said|say|quote|discuss|talk|mention|ask)\b', query_lower):
        return 'transcript'

    return None  # No strong hint


def is_simple_discovery_query(query: str) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Check if query is a simple discovery query that can be handled with a single tool.

    IMPORTANT: This should only match queries SPECIFICALLY asking about available sessions,
    not queries that happen to mention "session" while asking about content.

    Returns:
        (is_simple, tool_name, tool_args)
    """
    query_lower = query.lower().strip()

    # Very specific patterns for listing sessions - must NOT match content queries
    # Pattern must be about discovering what sessions exist, not about session content
    list_patterns = [
        # "What sessions are available?" / "What sessions do you have?"
        r'^what\s+sessions?\s+(?:are\s+)?(?:available|exist|do\s+(?:you|we)\s+have)',
        # "Which sessions are available?"
        r'^which\s+sessions?\s+(?:are\s+)?(?:available|exist)',
        # "List sessions" / "List all sessions" / "Show sessions"
        r'^(?:list|show)\s+(?:all\s+)?(?:the\s+)?sessions?$',
        # "Available sessions" / "Existing sessions"
        r'^(?:available|existing)\s+sessions?\s*\??$',
        # "What sessions are there?"
        r'^what\s+sessions?\s+are\s+there\s*\??$',
    ]

    for pattern in list_patterns:
        if re.search(pattern, query_lower):
            return (True, 'list_sessions', {})

    return (False, None, None)
