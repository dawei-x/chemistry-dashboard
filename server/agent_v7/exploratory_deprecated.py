"""
Exploratory Retriever for BLINC Agent V7

Handles cross-session queries by systematically retrieving from multiple sessions.
This is the key capability that enables queries like "What was said about AI across all sessions?"

Design Principles:
1. Systematic retrieval - iterate over ALL relevant sessions
2. Artifact selection - choose appropriate artifact type based on query
3. Combine evidence - aggregate results for synthesis
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable

from .classifier import QueryClassification

logger = logging.getLogger(__name__)


@dataclass
class ExploratoryEvidence:
    """Evidence gathered from exploratory retrieval."""
    session_id: int
    session_name: str
    artifact_type: str  # transcript, concept_map, collaboration
    content: str  # The display field from tool output
    raw_result: Dict[str, Any]


@dataclass
class ExploratoryResult:
    """Result of exploratory retrieval across sessions."""
    query: str
    classification: QueryClassification
    sessions_searched: List[int]
    evidence: List[ExploratoryEvidence]
    summary: str  # Brief summary of what was found


def _extract_explicit_session_ids(query: str) -> List[int]:
    """
    Extract explicitly named sessions from query using DB-driven resolution.

    This is the PRINCIPLED approach: use database as source of truth.
    Returns list of session IDs that the user explicitly mentioned.
    """
    import re
    from .memory import get_session_name_mapping

    query_lower = query.lower()
    found_ids = set()

    # Check for explicit session IDs first (e.g., "session 19", "sessions 19 and 20")
    for match in re.finditer(r'sessions?\s*(\d+)', query_lower):
        found_ids.add(int(match.group(1)))
    # Also catch "19 and 20" pattern after "session(s)"
    for match in re.finditer(r'sessions?\s+(\d+)\s+and\s+(\d+)', query_lower):
        found_ids.add(int(match.group(1)))
        found_ids.add(int(match.group(2)))

    # Check for session names using DB-loaded mapping
    session_mapping = get_session_name_mapping()
    for name, sid in session_mapping:
        # Multi-word names - check both exact match and flexible match
        if ' ' in name:
            # Exact match
            if name in query_lower:
                found_ids.add(sid)
                logger.debug(f"[Exploratory] Matched session name '{name}' → {sid}")
            else:
                # Flexible match: handle cases like "AI Alive" matching "is ai alive"
                # Remove common prefixes/suffixes and try again
                name_words = name.split()
                # Try matching without first word if it's a common prefix
                if len(name_words) >= 2 and name_words[0] in ['is', 'the', 'a', 'an']:
                    short_name = ' '.join(name_words[1:])
                    if short_name in query_lower:
                        found_ids.add(sid)
                        logger.debug(f"[Exploratory] Matched session name (flexible) '{short_name}' → {sid}")

    return sorted(found_ids)


def _is_superlative_query(query: str) -> bool:
    """
    Check if query is a superlative/comparison query requiring ALL sessions.

    These queries need to compare across all sessions to find best/worst.

    IMPORTANT: Returns False if user explicitly names specific sessions.
    "Compare AI Alive and Nuclear Fusion" is NOT superlative - user wants those two.
    "Which session had the best collaboration?" IS superlative - needs ranking.
    """
    import re
    query_lower = query.lower()

    # If user explicitly names 2+ sessions, this is NOT a superlative query
    # They want those specific sessions compared, not a ranking of all sessions
    explicit_sessions = _extract_explicit_session_ids(query)
    if len(explicit_sessions) >= 2:
        logger.info(f"[Exploratory] User named {len(explicit_sessions)} sessions explicitly: {explicit_sessions} - not superlative")
        return False

    superlative_patterns = [
        r'\b(best|worst|highest|lowest|most|least)\b',
        r'\b(top|bottom)\s+\d+\b',
        r'\bwhich\s+session\s+has\b',
        # Fixed: only match "compare all sessions" or "compare sessions", not "compare the X"
        r'\bcompare\s+(?:all\s+)?sessions\b',
        r'\bcompare\s+the\s+sessions\b',
        r'\branking\b',
        r'\brank\s+the\b',
        # Comparative patterns (imply cross-session comparison)
        r'\bhigher\s+(?:engagement|collaboration|score)\b',
        r'\blower\s+(?:engagement|collaboration|score)\b',
        r'\bwhy\s+(?:did|do|does)\s+some\b',
    ]

    for pattern in superlative_patterns:
        if re.search(pattern, query_lower):
            return True
    return False


def retrieve_exploratory(
    query: str,
    classification: QueryClassification,
    tools: Dict[str, Callable],
    max_sessions: int = 20  # Increased from 5 to handle more sessions
) -> ExploratoryResult:
    """
    Retrieve evidence from multiple sessions for an exploratory query.

    This function implements the systematic multi-session retrieval that
    the simple ReAct loop cannot reliably achieve.

    INTELLIGENT SESSION SELECTION:
    - For superlative queries (best/worst): Get ALL sessions first, then use
      collaboration scores to pick top candidates for deep analysis
    - For thematic queries: Use search to find relevant sessions

    Args:
        query: The user's query
        classification: Query classification with hints
        tools: Dictionary of available tools
        max_sessions: Maximum sessions to retrieve from (default 20)

    Returns:
        ExploratoryResult with combined evidence from all sessions
    """
    logger.info(f"[Exploratory] Starting retrieval for: '{query}'")
    logger.info(f"[Exploratory] Classification: {classification.reason}")

    evidence_list: List[ExploratoryEvidence] = []
    sessions_searched: List[int] = []

    # Check if this is a superlative query
    is_superlative = _is_superlative_query(query)
    if is_superlative:
        logger.info("[Exploratory] Superlative query detected - will intelligently select top candidates")

    # Step 1: Find relevant sessions (intelligently)
    session_ids = _find_relevant_sessions(
        query, classification, tools, max_sessions,
        is_superlative=is_superlative
    )

    if not session_ids:
        logger.warning("[Exploratory] No sessions found, falling back to list_sessions")
        # Fallback: get all sessions
        list_result = tools['list_sessions']()
        if 'sessions' in list_result:
            sessions = list_result['sessions']
            # For superlative, sort by collaboration score and get top candidates
            if is_superlative and any(s.get('collaboration_score') for s in sessions):
                sessions_with_scores = sorted(
                    [s for s in sessions if s.get('collaboration_score') is not None],
                    key=lambda x: x.get('collaboration_score', 0),
                    reverse=True
                )
                # Get top 3 for deep analysis
                session_ids = [s['session_id'] for s in sessions_with_scores[:3]]
                logger.info(f"[Exploratory] Selected top 3 by score: {session_ids}")
            else:
                session_ids = [s['session_id'] for s in sessions[:max_sessions]]

    logger.info(f"[Exploratory] Found {len(session_ids)} relevant sessions: {session_ids}")

    # Step 2: Determine artifact types (plural) for triangulation
    artifact_types = _determine_artifact_types(query, classification)
    logger.info(f"[Exploratory] Using artifact types: {artifact_types}")

    # Step 3: Retrieve each artifact type from each session
    for session_id in session_ids:
        for artifact_type in artifact_types:
            try:
                result = _retrieve_from_session(
                    session_id=session_id,
                    artifact_type=artifact_type,
                    query=query,
                    classification=classification,
                    tools=tools
                )
                if result:
                    evidence_list.append(result)
                    if session_id not in sessions_searched:
                        sessions_searched.append(session_id)
                    logger.info(f"[Exploratory] Retrieved {artifact_type} from session {session_id}")
            except Exception as e:
                logger.error(f"[Exploratory] Error retrieving {artifact_type} from session {session_id}: {e}")

    # Step 3.5: For speaker queries, also get speaker profile (provides comparative context)
    # This gives the LLM data to reason about speaker's role, not hardcoded interpretations
    if classification.speakers and 'get_speaker_profile' in tools:
        for speaker in classification.speakers:
            try:
                profile_result = tools['get_speaker_profile'](speaker_name=speaker)
                if profile_result and profile_result.get('is_relevant', True):
                    # Add as evidence with special artifact type
                    evidence_list.append(ExploratoryEvidence(
                        session_id=0,  # Cross-session
                        session_name="Speaker Profile",
                        artifact_type="speaker_profile",
                        content=_format_speaker_profile_for_display(profile_result),
                        raw_result=profile_result
                    ))
                    logger.info(f"[Exploratory] Retrieved speaker profile for {speaker}")
            except Exception as e:
                logger.error(f"[Exploratory] Error getting speaker profile for {speaker}: {e}")

    # Step 4: Build summary
    summary = _build_evidence_summary(query, evidence_list)

    return ExploratoryResult(
        query=query,
        classification=classification,
        sessions_searched=sessions_searched,
        evidence=evidence_list,
        summary=summary
    )


def _is_all_sessions_query(query: str) -> bool:
    """
    Check if query explicitly requests ALL sessions.

    These queries bypass search filtering and use list_sessions directly.
    This ensures cross-session queries actually check all sessions.

    Patterns from committed V7's proven approach.
    """
    import re
    query_lower = query.lower()

    all_sessions_patterns = [
        r'\ball\s+sessions?\b',
        r'\bevery\s+session\b',
        r'\bacross\s+sessions?\b',
        r'\beach\s+session\b',
        r'\ball\s+the\s+sessions?\b',
        r'\bcompare\s+sessions?\b',
        r'\ball\s+discussions?\b',
        r'\bacross\s+all\b',
        r'\bmultiple\s+sessions?\b',
    ]

    for pattern in all_sessions_patterns:
        if re.search(pattern, query_lower):
            return True
    return False


def _find_relevant_sessions(
    query: str,
    classification: QueryClassification,
    tools: Dict[str, Callable],
    max_sessions: int,
    is_superlative: bool = False
) -> List[int]:
    """
    Find sessions relevant to the query.

    KEY INSIGHTS:
    - EXPLICIT SESSION NAMES: If user names sessions, use those (DB-driven resolution)
    - For "all sessions" or superlative queries, use list_sessions to get ALL sessions
    - For superlative, leverage collaboration_score to pick top candidates
    - For thematic queries, use search but with higher recall
    - Search is for DISCOVERY, not filtering - we fetch full artifacts later
    - SPEAKER QUERIES: Filter to sessions where that speaker actually appears

    Always deduplicates session IDs to avoid processing same session multiple times.
    """
    import re
    session_ids_set: set = set()  # Use set to avoid duplicates

    # If classification already has session IDs, use those
    if classification.session_ids:
        return classification.session_ids[:max_sessions]

    # =========================================================
    # STEP 0: Check for explicitly named sessions (PRINCIPLED)
    # If user says "Compare AI Alive and Nuclear Fusion", use those exact sessions.
    # This is more reliable than semantic search for named entities.
    # =========================================================
    explicit_ids = _extract_explicit_session_ids(query)
    if explicit_ids:
        logger.info(f"[Exploratory] Using explicitly named sessions: {explicit_ids}")
        return explicit_ids[:max_sessions]

    # SPEAKER-AWARE FILTERING (PRINCIPLED FIX)
    # If query mentions a specific speaker, filter to sessions where they appear
    # This prevents retrieving 8 sessions when the speaker only appears in 1
    if classification.speakers:
        speaker_name = classification.speakers[0].lower()
        logger.info(f"[Exploratory] Speaker mentioned: '{speaker_name}' - filtering to their sessions")
        try:
            list_result = tools['list_sessions']()
            if isinstance(list_result, dict) and 'sessions' in list_result:
                speaker_sessions = []
                for s in list_result['sessions']:
                    session_speakers = s.get('speakers', [])
                    # Check if speaker appears in this session (case-insensitive)
                    if any(speaker_name in sp.lower() for sp in session_speakers):
                        speaker_sessions.append(s['session_id'])

                if speaker_sessions:
                    logger.info(f"[Exploratory] Speaker '{speaker_name}' found in sessions: {speaker_sessions}")
                    return speaker_sessions[:max_sessions]
                else:
                    logger.warning(f"[Exploratory] Speaker '{speaker_name}' not found in any session")
                    # Continue to semantic search as fallback
        except Exception as e:
            logger.error(f"[Exploratory] Speaker lookup failed: {e}")

    # CRITICAL: Superlative and "all sessions" queries use list_sessions
    # This ensures we get complete data for comparison
    if is_superlative or _is_all_sessions_query(query):
        logger.info("[Exploratory] Using list_sessions for complete session data")
        try:
            list_result = tools['list_sessions']()
            if isinstance(list_result, dict) and 'sessions' in list_result:
                sessions = list_result['sessions']

                # For superlative queries about collaboration, sort by score
                # and pick top candidates for deep analysis
                if is_superlative and any(s.get('collaboration_score') for s in sessions):
                    sessions_with_scores = [
                        s for s in sessions
                        if s.get('collaboration_score') is not None
                    ]
                    # Sort by collaboration score (descending)
                    sessions_with_scores.sort(
                        key=lambda x: x.get('collaboration_score', 0),
                        reverse=True
                    )

                    # Intelligently select: top 2-3 for deep analysis
                    # This mirrors how RAG Discovery worked - not all sessions,
                    # but the most relevant ones
                    top_candidates = sessions_with_scores[:3]
                    session_ids = [s['session_id'] for s in top_candidates]
                    logger.info(
                        f"[Exploratory] Selected top {len(session_ids)} by collaboration score: "
                        f"{[(s['session_id'], s.get('collaboration_score')) for s in top_candidates]}"
                    )
                    return session_ids
                else:
                    # For non-superlative "all sessions" queries
                    for s in sessions:
                        if isinstance(s, dict) and 'session_id' in s:
                            session_ids_set.add(s['session_id'])

            logger.info(f"[Exploratory] list_sessions returned {len(session_ids_set)} sessions")
            session_ids = sorted(session_ids_set)
            return session_ids[:max_sessions]
        except Exception as e:
            logger.error(f"[Exploratory] list_sessions failed: {e}")

    # Try to search by topic (for targeted cross-session queries)
    # Use FULL query for search - it has better semantic context for embeddings
    # E.g., "What was said about AI?" scores much better than just "ai"
    if classification.topics or query:
        search_query = query  # Use full query, not just extracted topics
        logger.info(f"[Exploratory] Searching for sessions with query: '{search_query}'")

        try:
            # Increase top_k to improve recall
            search_result = tools['search_sessions'](query=search_query, top_k=max_sessions)

            # Handle different response formats
            if isinstance(search_result, dict):
                if 'sessions' in search_result:
                    for s in search_result['sessions']:
                        if isinstance(s, dict) and 'session_id' in s:
                            session_ids_set.add(s['session_id'])
                elif 'display' in search_result:
                    # Parse session IDs from display text
                    matches = re.findall(r'Session\s+(\d+)', search_result['display'])
                    for m in matches:
                        session_ids_set.add(int(m))

            logger.info(f"[Exploratory] Search returned {len(session_ids_set)} unique sessions")

        except Exception as e:
            logger.error(f"[Exploratory] Search failed: {e}")

    # Only supplement if search found NOTHING (don't dilute relevant results with all sessions)
    # If search found 1+ sessions, those are the relevant ones - trust the search
    if len(session_ids_set) == 0:
        logger.info("[Exploratory] No search results, falling back to list_sessions")
        try:
            list_result = tools['list_sessions']()

            if isinstance(list_result, dict):
                if 'sessions' in list_result:
                    for s in list_result['sessions']:
                        if isinstance(s, dict) and 'session_id' in s:
                            session_ids_set.add(s['session_id'])
                elif 'display' in list_result:
                    # Parse from display
                    matches = re.findall(r'Session\s+(\d+)', list_result['display'])
                    for m in matches:
                        session_ids_set.add(int(m))

        except Exception as e:
            logger.error(f"[Exploratory] list_sessions failed: {e}")
    else:
        logger.info(f"[Exploratory] Search found {len(session_ids_set)} relevant sessions, not supplementing")

    # Convert to sorted list (sorted for deterministic order)
    session_ids = sorted(session_ids_set)
    return session_ids[:max_sessions]


def _determine_artifact_types(query: str, classification: QueryClassification) -> List[str]:
    """
    Determine which artifact types to retrieve based on query.

    Returns a LIST of artifact types for triangulation across representations.
    For collaboration queries, we also need transcripts for quotes.
    For concept queries, we may need collaboration context.

    Returns: List of 'transcript', 'concept_map', or 'collaboration'
    """
    query_lower = query.lower()
    artifact_types = []

    # Primary artifact based on classification hint
    if classification.artifact_hint:
        artifact_types.append(classification.artifact_hint)

    # Collaboration keywords - includes 7C dimension-related terms
    collaboration_keywords = [
        'collaborat', '7c', 'engagement', 'quality', 'participation',
        'disagreement', 'conflict', 'handle', 'resolution',
        'contribution', 'climate', 'communication', 'constructive',
        'compatibility', 'context', 'balance'
    ]
    if any(kw in query_lower for kw in collaboration_keywords):
        if 'collaboration' not in artifact_types:
            artifact_types.append('collaboration')
        # Also get transcript for quotes/evidence
        if 'transcript' not in artifact_types:
            artifact_types.append('transcript')

    # Concept map keywords - also useful for "balanced participation" (idea contributions)
    concept_keywords = ['ideas', 'concepts', 'connect', 'themes', 'structure', 'debate', 'argument']
    if any(kw in query_lower for kw in concept_keywords):
        if 'concept_map' not in artifact_types:
            artifact_types.append('concept_map')

    # For participation/contribution queries, concept maps show idea contributions per speaker
    if any(kw in query_lower for kw in ['participation', 'contribut', 'balance']):
        if 'concept_map' not in artifact_types:
            artifact_types.append('concept_map')

    # Default to transcript if nothing else
    if not artifact_types:
        artifact_types.append('transcript')

    return artifact_types


def _retrieve_from_session(
    session_id: int,
    artifact_type: str,
    query: str,
    classification: QueryClassification,
    tools: Dict[str, Callable]
) -> Optional[ExploratoryEvidence]:
    """
    Retrieve a specific artifact from a session.

    CRITICAL FIX (learned from committed V7):
    For exploratory (cross-session) queries, do NOT apply keyword filters!

    The purpose of exploratory retrieval is to gather evidence from MULTIPLE
    sessions. If we apply keyword filters, sessions that don't heavily mention
    the topic will return empty transcripts, defeating the purpose.

    Instead:
    - Search is for DISCOVERY (which sessions are relevant)
    - Retrieval is for CONTENT (get full artifacts)
    - Synthesis is for FILTERING (LLM identifies relevant portions)
    """
    try:
        if artifact_type == 'collaboration':
            result = tools['get_7c_analysis'](session_id=session_id)
            tool_name = 'get_7c_analysis'
        elif artifact_type == 'concept_map':
            result = tools['get_concept_map'](session_id=session_id)
            tool_name = 'get_concept_map'
        else:  # transcript
            # CRITICAL: For exploratory queries, fetch FULL transcripts
            # Do NOT apply keyword filters - that defeats cross-session exploration
            # The synthesis LLM will identify relevant content
            #
            # Speaker filter is OK since it's about WHO spoke, not topic filtering
            if classification.speakers:
                result = tools['get_transcript'](
                    session_id=session_id,
                    speaker_filter=classification.speakers[0]
                )
            else:
                # Fetch full transcript - no keyword filter!
                result = tools['get_transcript'](session_id=session_id)
            tool_name = 'get_transcript'

        if not result:
            return None

        # Extract display content
        display_content = result.get('display', str(result))
        session_name = result.get('session_name', f'Session {session_id}')

        return ExploratoryEvidence(
            session_id=session_id,
            session_name=session_name,
            artifact_type=artifact_type,
            content=display_content,
            raw_result=result
        )

    except Exception as e:
        logger.error(f"[Exploratory] Failed to retrieve {artifact_type} from session {session_id}: {e}")
        return None


def _format_speaker_profile_for_display(profile: Dict[str, Any]) -> str:
    """
    Format speaker profile data for LLM consumption.

    Presents raw comparative data that the LLM can reason about.
    """
    lines = []
    alias = profile.get('speaker_alias', 'Unknown')
    summary = profile.get('transcript_summary', {})

    lines.append(f"## Speaker Profile: {alias}")
    lines.append(f"Sessions participated: {summary.get('sessions_participated', 0)}")
    lines.append("")
    lines.append("### Participation Patterns (comparative data)")
    lines.append("")

    for p in summary.get('participation_by_session', []):
        session = p.get('session_name', 'Unknown')
        lines.append(f"**{session}**:")
        lines.append(f"  - Utterances: {p.get('utterances', 0)}, Questions: {p.get('questions_asked', 0)}")
        lines.append(f"  - Question rate: {p.get('question_rate_pct', 0)}%")
        lines.append(f"  - Participation share: {p.get('participation_share_pct', 0)}% of session")
        lines.append(f"  - Session had {p.get('session_speaker_count', 1)} speakers (equal share = {p.get('expected_equal_share_pct', 100)}%)")
        lines.append("")

    # Add sample quotes
    quotes = summary.get('sample_quotes', [])
    if quotes:
        lines.append("### Sample Quotes")
        for q in quotes[:3]:
            text = q.get('text', '')[:200]
            is_q = "(question)" if q.get('is_question') else ""
            lines.append(f"  - \"{text}...\" {is_q}")
        lines.append("")

    return "\n".join(lines)


def _build_evidence_summary(query: str, evidence_list: List[ExploratoryEvidence]) -> str:
    """
    Build a brief summary of the evidence gathered.
    """
    if not evidence_list:
        return "No evidence found across sessions."

    # Unique sessions and artifact types
    session_names = list(set(e.session_name for e in evidence_list))
    artifact_types = list(set(e.artifact_type for e in evidence_list))

    artifact_str = ', '.join(artifact_types)
    return (
        f"Retrieved {len(evidence_list)} artifacts ({artifact_str}) from "
        f"{len(session_names)} sessions: {', '.join(session_names)}"
    )


def format_exploratory_evidence_for_synthesis(result: ExploratoryResult) -> str:
    """
    Format exploratory evidence for the synthesis prompt.

    This creates a comprehensive view of evidence across all sessions
    that the LLM can use to generate a cross-session synthesis.
    """
    sections = []

    sections.append(f"# Cross-Session Evidence for: {result.query}")
    sections.append(f"\nSessions analyzed: {len(result.evidence)}")
    sections.append(f"Session IDs: {result.sessions_searched}")
    sections.append("")

    for evidence in result.evidence:
        sections.append(f"## Session {evidence.session_id}: {evidence.session_name}")
        sections.append(f"Artifact: {evidence.artifact_type}")
        sections.append("")
        sections.append(evidence.content)
        sections.append("")
        sections.append("---")
        sections.append("")

    sections.append("# Instructions for Synthesis")
    sections.append(
        "Synthesize findings across ALL sessions above. "
        "Cite specific quotes and session numbers. "
        "Compare and contrast findings across sessions. "
        "Note any patterns or differences observed."
    )

    return '\n'.join(sections)
