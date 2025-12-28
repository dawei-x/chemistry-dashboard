"""
Input Processor Node for BLINC Agent V3

Preprocesses the query and resolves references from conversation context.
"""

import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Session name patterns for resolution
SESSION_PATTERNS = {
    'nyc': 18,
    'living in nyc': 18,
    'new york': 18,
    'ai alive': 19,
    'is ai alive': 19,
    'artificial intelligence': 19,
    'nuclear fusion': 20,
    'fusion': 20,
    'shaw': 21,
    'shaw interview': 21,
    'collaboration literacy': 22,
    'literacy': 22,
    'dinosaurs': 23,
    'dinosaur': 23,
    'country music': 24,
    'country': 24,
    'music': 24,
    'abundance': 25
}


def process_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the input query and resolve references.

    This node:
    1. Normalizes the query
    2. Resolves session references ("Dinosaurs session" -> session 23)
    3. Resolves conversational references ("it", "that session", "go back")

    Args:
        state: Current agent state

    Returns:
        Updated state with resolved references
    """
    query = state.get('current_query', state.get('original_query', ''))
    query_lower = query.lower()

    logger.info(f"Processing input: '{query}'")

    updates = {
        'iteration_count': 0
    }

    # === Resolve conversational references ===

    # "Go back" or "previous" -> switch to previous session
    if any(phrase in query_lower for phrase in ['go back', 'previous session', 'earlier session']):
        if state.get('previous_session_focus'):
            updates['current_session_focus'] = state['previous_session_focus']
            updates['previous_session_focus'] = state.get('current_session_focus')
            logger.info(f"Switched to previous session: {updates['current_session_focus']}")

    # "This session" or "it" with no session mentioned -> use current focus
    if any(phrase in query_lower for phrase in ['this session', 'that session', 'the session']):
        if state.get('current_session_focus') and not _mentions_specific_session(query_lower):
            # Keep the current focus, query will use it
            pass

    # === Resolve session name references ===

    session_id = _resolve_session_name(query_lower)
    if session_id:
        # Update session focus
        if state.get('current_session_focus') != session_id:
            updates['previous_session_focus'] = state.get('current_session_focus')
            updates['current_session_focus'] = session_id

        # Add to session history
        history = state.get('session_history', []).copy()
        if session_id not in history:
            history.append(session_id)
        updates['session_history'] = history[-10:]  # Keep last 10

        logger.info(f"Resolved session reference: {session_id}")

    # === Resolve speaker references ===

    speaker = _resolve_speaker_name(query_lower)
    if speaker:
        updates['current_speaker_focus'] = speaker
        logger.info(f"Resolved speaker reference: {speaker}")

    # === Detect comparison queries ===

    if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference between']):
        # Try to find multiple session references
        sessions = _find_all_session_references(query_lower)
        if len(sessions) >= 2:
            updates['compared_sessions'] = sessions[:5]
            logger.info(f"Detected comparison: {sessions}")

    return updates


def _mentions_specific_session(query: str) -> bool:
    """Check if query mentions a specific session by name or ID."""
    # Check for session ID pattern
    if re.search(r'session\s*\d+', query):
        return True

    # Check for session names
    for pattern in SESSION_PATTERNS:
        if pattern in query:
            return True

    return False


def _resolve_session_name(query: str) -> int | None:
    """Resolve session name to session ID."""
    query = query.lower()

    # Check for explicit session ID
    match = re.search(r'session\s*(\d+)', query)
    if match:
        return int(match.group(1))

    # Check for session names
    for pattern, session_id in SESSION_PATTERNS.items():
        if pattern in query:
            return session_id

    return None


def _find_all_session_references(query: str) -> list:
    """Find all session references in a query (for comparisons)."""
    sessions = []
    query = query.lower()

    # Find explicit IDs
    for match in re.finditer(r'session\s*(\d+)', query):
        sessions.append(int(match.group(1)))

    # Find named sessions
    for pattern, session_id in SESSION_PATTERNS.items():
        if pattern in query and session_id not in sessions:
            sessions.append(session_id)

    return sessions


def _resolve_speaker_name(query: str) -> str | None:
    """Extract speaker name from query if mentioned."""
    query = query.lower()

    # Common patterns for speaker mentions
    patterns = [
        r"how did (\w+) ",
        r"what did (\w+) say",
        r"(\w+)'s (contribution|participation|style)",
        r"speaker (\w+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            name = match.group(1)
            # Filter out common words that aren't names
            if name not in ['the', 'a', 'this', 'that', 'each', 'every', 'some']:
                return name.title()

    return None
