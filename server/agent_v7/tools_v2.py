"""
Simplified Tool Registry for BLINC Agent V7

5 core tools that return LLM-ready text directly (no formatter needed):
1. list_sessions      - List all available sessions
2. search_sessions    - Find sessions by topic
3. get_transcript     - Get session transcript
4. get_concept_map    - Get concept map structure
5. get_7c_analysis    - Get collaboration metrics

Design principle: Tools return what the LLM should see directly.
No intermediate JSON that gets formatted later - this prevents data loss.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from functools import wraps

# Import existing tools
from .tools.artifact_tools import (
    list_sessions as _list_sessions,
    search_for_sessions as _search_sessions,
    get_artifacts as _get_artifacts,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 7C Framework Definitions (included in tool output for LLM context)
# =============================================================================

SEVEN_C_DEFINITIONS = {
    "climate": "The emotional and affective aspects of the collaboration",
    "communication": "The quantity and quality of information shared among group members",
    "compatibility": "How well group members' working and interaction styles complement each other",
    "conflict": "Approaches to handling disagreements and contentious situations that arise during group work",
    "context": "Environmental factors and situational awareness: the who, why, and where of the collaboration",
    "contribution": "Individual participation and effort balance: what individual participants are, and are not, bringing to the collaboration",
    "constructive": "Overall goals of the collaboration and the team's progress toward achieving them",
}


def tool_wrapper(tool_name: str):
    """Decorator to standardize tool output and logging."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"[Tool] {tool_name} called with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                result["tool_name"] = tool_name
                logger.info(f"[Tool] {tool_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"[Tool] {tool_name} error: {e}")
                return {
                    "tool_name": tool_name,
                    "display": f"Error: {str(e)}",
                    "error": str(e),
                }
        return wrapper
    return decorator


# =============================================================================
# Tool 1: list_sessions
# =============================================================================

@tool_wrapper("list_sessions")
def list_sessions() -> Dict[str, Any]:
    """
    List all available discussion sessions with collaboration scores.

    Returns:
        Dict with 'display' containing LLM-ready text of all sessions,
        including collaboration scores for intelligent session selection.
    """
    result = _list_sessions()

    sessions = result.get('sessions', [])

    # Sort by collaboration score for easier scanning (highest first)
    sessions_with_scores = [s for s in sessions if s.get('collaboration_score') is not None]
    sessions_without_scores = [s for s in sessions if s.get('collaboration_score') is None]
    sessions_with_scores.sort(key=lambda x: x.get('collaboration_score', 0), reverse=True)
    sorted_sessions = sessions_with_scores + sessions_without_scores

    # Build LLM-ready text
    lines = [f"=== Available Sessions ({len(sessions)} total) ==="]
    lines.append("(Sorted by collaboration score, highest first)\n")

    for s in sorted_sessions:
        sid = s.get('session_id', s.get('session_device_id', '?'))
        name = s.get('session_name', s.get('name', 'Unnamed'))
        speakers = s.get('speakers', [])
        speaker_count = s.get('speaker_count', len(speakers))
        speaker_str = ", ".join(speakers[:5]) if speakers else "Unknown"
        collab_score = s.get('collaboration_score')

        lines.append(f"Session {sid}: {name}")
        lines.append(f"  Speakers ({speaker_count}): {speaker_str}")

        # Show collaboration score prominently
        if collab_score is not None:
            lines.append(f"  Collaboration Score: {collab_score}/100")
        else:
            lines.append(f"  Collaboration Score: N/A")
        lines.append("")

    # Add guidance for LLM
    lines.append("---")
    lines.append("TIP: For detailed collaboration breakdown, call get_7c_analysis(session_id=N)")
    lines.append("TIP: For speaker contributions, call get_speaker_profile(speaker_name='Name')")

    return {
        "display": "\n".join(lines),
        "session_count": len(sessions),
        "sessions": sorted_sessions,  # Include structured data too
    }


# =============================================================================
# Tool 2: search_sessions
# =============================================================================

@tool_wrapper("search_sessions")
def search_sessions(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Find sessions relevant to a query using semantic search.

    Args:
        query: Topic or keyword to search for
        top_k: Maximum number of results

    Returns:
        Dict with 'display' containing LLM-ready text of matching sessions
    """
    result = _search_sessions(query=query, top_k=top_k)

    sessions = result.get('sessions', [])

    # Build LLM-ready text
    lines = [f"=== Search Results for \"{query}\" ({len(sessions)} found) ===\n"]

    if not sessions:
        lines.append("No matching sessions found.")
    else:
        for i, s in enumerate(sessions, 1):
            sid = s.get('session_id', s.get('session_device_id', '?'))
            name = s.get('session_name', s.get('name', 'Unnamed'))
            # Check multiple possible score field names
            score = s.get('relevance_score') or s.get('best_match_score') or s.get('score', 0)
            speakers = s.get('speakers', [])
            speaker_str = ", ".join(speakers[:5]) if speakers else "Unknown"
            preview = s.get('match_preview', '')

            lines.append(f"{i}. Session {sid}: {name}")
            if score:
                lines.append(f"   Relevance: {score:.2f}")
            lines.append(f"   Speakers: {speaker_str}")
            if preview:
                # Show first 150 chars of preview for context
                preview_short = preview[:150].strip()
                if len(preview) > 150:
                    preview_short += "..."
                lines.append(f"   Preview: {preview_short}")
            lines.append("")

    return {
        "display": "\n".join(lines),
        "session_count": len(sessions),
        "query": query,
        "sessions": sessions,  # Include for auto-fetch
    }


# =============================================================================
# Tool 3: get_transcript
# =============================================================================

@tool_wrapper("get_transcript")
def get_transcript(
    session_id: int,
    speaker_filter: str = None,
    keyword_filter: str = None
) -> Dict[str, Any]:
    """
    Get transcript for a session in human-readable format.

    Args:
        session_id: Session to get transcript for
        speaker_filter: Optional - only get utterances from this speaker
        keyword_filter: Optional - only get utterances containing this keyword

    Returns:
        Dict with 'display' containing LLM-ready formatted transcript
    """
    result = _get_artifacts(session_id, include=['transcript'])

    if result.get('error'):
        return {
            "display": f"Error getting transcript: {result.get('error')}",
            "error": result.get('error'),
        }

    artifacts = result.get('artifacts', {})
    transcript = artifacts.get('transcript', {})

    session_name = result.get('session_name', f'Session {session_id}')
    device_name = result.get('device_name', '')

    utterances = transcript.get('utterances', [])

    # Apply filters if provided
    if speaker_filter:
        speaker_lower = speaker_filter.lower()
        utterances = [
            u for u in utterances
            if speaker_lower in u.get('speaker', '').lower()
        ]

    if keyword_filter:
        keyword_lower = keyword_filter.lower()
        utterances = [
            u for u in utterances
            if keyword_lower in u.get('text', '').lower()
        ]

    # Build LLM-ready text
    lines = [
        f"=== Transcript: {session_name} ===",
        f"Session ID: {session_id}",
        f"Device: {device_name}",
    ]

    if speaker_filter:
        lines.append(f"Filtered by speaker: {speaker_filter}")
    if keyword_filter:
        lines.append(f"Filtered by keyword: {keyword_filter}")

    lines.append(f"Utterances: {len(utterances)}")
    lines.append("")
    lines.append("--- Begin Transcript ---")
    lines.append("")

    for u in utterances:
        speaker = u.get('speaker', 'Unknown') or 'Unknown'
        text = u.get('text', '').strip()

        # Format timestamp as [MM:SS]
        start_time = u.get('start_time', 0) or 0
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"

        lines.append(f"{timestamp} {speaker}: {text}")

    lines.append("")
    lines.append("--- End Transcript ---")

    return {
        "display": "\n".join(lines),
        "session_id": session_id,
        "session_name": session_name,
        "utterance_count": len(utterances),
        "utterances": utterances,  # Structured data for programmatic use
    }


# =============================================================================
# Tool 4: get_concept_map
# =============================================================================

@tool_wrapper("get_concept_map")
def get_concept_map(session_id: int) -> Dict[str, Any]:
    """
    Get concept map for a session showing ideas and their connections.

    Args:
        session_id: Session to get concept map for

    Returns:
        Dict with 'display' containing LLM-ready concept map text
    """
    result = _get_artifacts(session_id, include=['concept_map'])

    if result.get('error'):
        return {
            "display": f"Error getting concept map: {result.get('error')}",
            "error": result.get('error'),
        }

    artifacts = result.get('artifacts', {})
    concept_map = artifacts.get('concept_map', {})

    session_name = result.get('session_name', f'Session {session_id}')
    device_name = result.get('device_name', '')

    if not concept_map.get('available', False):
        return {
            "display": f"No concept map available for {session_name}",
            "session_id": session_id,
            "available": False,
        }

    nodes = concept_map.get('nodes', [])
    edges = concept_map.get('edges', [])
    summary = concept_map.get('summary', {})

    # Build node lookup by id
    node_lookup = {n['id']: n for n in nodes}

    # Build outgoing edges map
    outgoing = {}
    incoming_set = set()

    for edge in edges:
        source_id = edge.get('source')
        target_id = edge.get('target')
        relationship = edge.get('relationship', 'relates_to')

        if source_id not in outgoing:
            outgoing[source_id] = []
        outgoing[source_id].append((relationship, target_id))
        incoming_set.add(target_id)

    def format_node(node_id):
        node = node_lookup.get(node_id)
        if not node:
            return f"[unknown] {node_id}"
        node_type = node.get('type', 'concept')
        speaker = node.get('speaker', 'Unknown')
        text = node.get('text', '')
        return f"[{node_type}] {speaker}: \"{text}\""

    # Build LLM-ready text
    lines = [
        f"=== Concept Map: {session_name} ===",
        f"Session ID: {session_id}",
        f"Device: {device_name}",
        f"Total Nodes: {summary.get('total_nodes', len(nodes))}",
        f"Total Edges: {summary.get('total_edges', len(edges))}",
    ]

    # Add node types breakdown
    node_types = summary.get('node_types', {})
    if node_types:
        lines.append("")
        lines.append("Node Types:")
        for ntype, count in node_types.items():
            lines.append(f"  {ntype}: {count}")

    # Add speaker contributions with by_type breakdown
    speaker_contribs = summary.get('speaker_contributions', {})
    if speaker_contribs:
        lines.append("")
        lines.append("Speaker Contributions:")
        for speaker, data in speaker_contribs.items():
            if isinstance(data, dict):
                total = data.get('total', 0)
                by_type = data.get('by_type', {})
                if by_type:
                    type_str = ", ".join(f"{t}: {c}" for t, c in by_type.items())
                    lines.append(f"  {speaker}: {total} concepts ({type_str})")
                else:
                    lines.append(f"  {speaker}: {total} concepts")
            else:
                lines.append(f"  {speaker}: {data} concepts")

    lines.append("")
    lines.append("--- Concept Graph (Adjacency List) ---")
    lines.append("")

    # Build adjacency list
    for node in nodes:
        node_id = node['id']
        has_outgoing = node_id in outgoing
        has_incoming = node_id in incoming_set

        if has_outgoing or (not has_outgoing and not has_incoming):
            lines.append(format_node(node_id))

            if has_outgoing:
                for relationship, target_id in outgoing[node_id]:
                    target_str = format_node(target_id)
                    lines.append(f"   - {relationship} -> {target_str}")

            lines.append("")

    lines.append("--- End Concept Map ---")

    return {
        "display": "\n".join(lines),
        "session_id": session_id,
        "session_name": session_name,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,  # Structured data for programmatic use
        "edges": edges,  # Structured data for programmatic use
        "summary": summary,  # Structured data for programmatic use
    }


# =============================================================================
# Tool 5: get_7c_analysis
# =============================================================================

@tool_wrapper("get_7c_analysis")
def get_7c_analysis(session_id: int) -> Dict[str, Any]:
    """
    Get 7C collaboration analysis with scores and evidence.

    The 7C Framework measures collaboration quality across 7 dimensions.
    Each dimension includes a score (0-100), explanation, and coded segments
    (actual quotes demonstrating the dimension).

    Args:
        session_id: Session to get analysis for

    Returns:
        Dict with 'display' containing LLM-ready 7C analysis
    """
    result = _get_artifacts(session_id, include=['collaboration'])

    if result.get('error'):
        return {
            "display": f"Error getting 7C analysis: {result.get('error')}",
            "error": result.get('error'),
        }

    artifacts = result.get('artifacts', {})
    collaboration = artifacts.get('collaboration', {})

    session_name = result.get('session_name', f'Session {session_id}')
    device_name = result.get('device_name', '')

    if not collaboration.get('available', False):
        return {
            "display": f"No 7C analysis available for {session_name}",
            "session_id": session_id,
            "available": False,
        }

    raw_dimensions = collaboration.get('dimensions', {})

    # Calculate overall score
    scores = [d.get('score', 0) for d in raw_dimensions.values() if d.get('score')]
    overall_score = sum(scores) / len(scores) if scores else 0

    # Build LLM-ready text
    lines = [
        f"=== 7C Collaboration Analysis: {session_name} ===",
        f"Session ID: {session_id}",
        f"Device: {device_name}",
        f"Overall Score: {overall_score:.1f}/100",
        "",
        "The 7C Framework measures collaboration quality across 7 dimensions.",
        "",
    ]

    # Process each dimension
    for dim_name, dim_data in raw_dimensions.items():
        score = dim_data.get('score', 0)
        explanation = dim_data.get('explanation', 'No explanation available')
        definition = SEVEN_C_DEFINITIONS.get(dim_name, '')

        lines.append(f"--- {dim_name.upper()} ({score}/100) ---")
        lines.append(f"Definition: {definition}")
        lines.append(f"Explanation: {explanation}")

        # Add coded segments (evidence quotes)
        raw_segments = dim_data.get('coded_segments', [])
        quote_count = 0

        for seg in raw_segments:
            if isinstance(seg, dict) and seg.get('quote'):
                quote_count += 1
                speaker = seg.get('speaker', '')
                quote = seg.get('quote', '')

                # Format quote with speaker
                if speaker and speaker not in quote:
                    full_quote = f"{speaker}: {quote}"
                else:
                    full_quote = quote

                lines.append(f"  Evidence {quote_count}: \"{full_quote}\"")

        if quote_count == 0:
            lines.append("  (No specific quotes coded for this dimension)")

        lines.append("")

    lines.append("=== End 7C Analysis ===")

    return {
        "display": "\n".join(lines),
        "session_id": session_id,
        "session_name": session_name,
        "overall_score": overall_score,
        "dimensions": raw_dimensions,  # Structured data for programmatic use
    }


# =============================================================================
# Tool 6: get_speaker_profile
# =============================================================================

def _get_db_connection():
    """Get database connection."""
    import mysql.connector
    return mysql.connector.connect(
        host='localhost',
        user='vagrant',
        password='vagrant',
        database='discussion_capture'
    )


@tool_wrapper("get_speaker_profile")
def get_speaker_profile(speaker_name: str, session_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get a speaker's engagement profile across sessions.

    Returns:
    - Sessions participated (enables chaining to get_transcript)
    - Per-session metrics (utterances, words, questions, analytic/certainty scores)
    - Sample quotes (diverse selection: questions, high-certainty, high-analytic)
    - Concept contributions by type
    - Connections to other speakers via concepts

    For full transcript, use get_transcript(session_id, speaker_filter=name).

    Args:
        speaker_name: Name of the speaker (partial match supported)
        session_id: Optional - limit to specific session (None = all sessions)

    Returns:
        Dict with 'display' containing LLM-ready speaker profile
    """
    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Find ALL speaker IDs with this alias (same person has different ID per session)
        cursor.execute("""
            SELECT id, alias, session_device_id FROM speaker WHERE alias LIKE %s
        """, (f"%{speaker_name}%",))
        speakers = cursor.fetchall()

        if not speakers:
            cursor.close()
            connection.close()
            return {
                "display": f"Speaker '{speaker_name}' not found. Use list_sessions to see available speakers.",
                "found": False,
            }

        speaker_ids = [s['id'] for s in speakers]
        speaker_alias = speakers[0]['alias']
        speaker_id_list = ', '.join(str(sid) for sid in speaker_ids)

        # Session filters
        session_filter = f"AND t.session_device_id = {session_id}" if session_id else ""
        session_filter_unaliased = f"AND session_device_id = {session_id}" if session_id else ""

        # Get participation by session (across ALL speaker IDs)
        cursor.execute(f"""
            SELECT
                t.session_device_id,
                COALESCE(s.name, sd.name) as session_name,
                COUNT(*) as utterance_count,
                SUM(t.word_count) as word_count,
                SUM(CASE WHEN t.question = 1 THEN 1 ELSE 0 END) as questions,
                AVG(t.analytic_thinking_value) as avg_analytic,
                AVG(t.certainty_value) as avg_certainty,
                AVG(t.clout_value) as avg_clout,
                AVG(t.emotional_tone_value) as avg_emotional,
                (SELECT COUNT(*) FROM transcript t2 WHERE t2.session_device_id = t.session_device_id) as session_total_utterances,
                (SELECT COUNT(DISTINCT t2.speaker_id) FROM transcript t2 WHERE t2.session_device_id = t.session_device_id) as session_speaker_count
            FROM transcript t
            JOIN session_device sd ON t.session_device_id = sd.id
            JOIN session s ON sd.session_id = s.id
            WHERE t.speaker_id IN ({speaker_id_list}) {session_filter}
            GROUP BY t.session_device_id, s.name, sd.name
        """)
        session_data = cursor.fetchall()

        # Add comparative metrics for LLM to reason about
        for row in session_data:
            utterances = int(row.get('utterance_count') or 0)
            questions = int(row.get('questions') or 0)
            session_total = int(row.get('session_total_utterances') or 1)
            speaker_count = int(row.get('session_speaker_count') or 1)

            row['participation_share_pct'] = round(utterances * 100.0 / session_total, 1) if session_total > 0 else 0
            row['question_rate_pct'] = round(questions * 100.0 / utterances, 1) if utterances > 0 else 0
            row['expected_equal_share_pct'] = round(100.0 / speaker_count, 1) if speaker_count > 0 else 100

        # Get concept contributions (across ALL speaker IDs)
        session_concept_filter = f"AND cs.session_device_id = {session_id}" if session_id else ""
        cursor.execute(f"""
            SELECT
                cn.node_type,
                cn.text,
                cs.session_device_id
            FROM concept_node cn
            JOIN concept_session cs ON cn.concept_session_id = cs.id
            WHERE cn.speaker_id IN ({speaker_id_list}) {session_concept_filter}
        """)
        concept_nodes = cursor.fetchall()

        # Get sample quotes - diverse selection showing speaker style
        sample_quotes = []

        # Get questions (across ALL speaker IDs)
        cursor.execute(f"""
            SELECT transcript as text, session_device_id, certainty_value, analytic_thinking_value,
                   'question' as quote_type
            FROM transcript
            WHERE speaker_id IN ({speaker_id_list}) AND question = 1 AND word_count > 10 {session_filter_unaliased}
            ORDER BY word_count DESC LIMIT 2
        """)
        sample_quotes.extend(cursor.fetchall())

        # Get high-certainty statements (across ALL speaker IDs)
        cursor.execute(f"""
            SELECT transcript as text, session_device_id, certainty_value, analytic_thinking_value,
                   'high_certainty' as quote_type
            FROM transcript
            WHERE speaker_id IN ({speaker_id_list}) AND question = 0 AND certainty_value > 70 AND word_count > 15 {session_filter_unaliased}
            ORDER BY certainty_value DESC LIMIT 2
        """)
        sample_quotes.extend(cursor.fetchall())

        # Get high-analytic statements (across ALL speaker IDs)
        cursor.execute(f"""
            SELECT transcript as text, session_device_id, certainty_value, analytic_thinking_value,
                   'high_analytic' as quote_type
            FROM transcript
            WHERE speaker_id IN ({speaker_id_list}) AND question = 0 AND analytic_thinking_value > 70 AND word_count > 15 {session_filter_unaliased}
            ORDER BY analytic_thinking_value DESC LIMIT 2
        """)
        sample_quotes.extend(cursor.fetchall())

        # Get connections to other speakers (across ALL speaker IDs)
        cursor.execute(f"""
            SELECT cn.id FROM concept_node cn
            JOIN concept_session cs ON cn.concept_session_id = cs.id
            WHERE cn.speaker_id IN ({speaker_id_list}) {session_concept_filter}
        """)
        node_rows = cursor.fetchall()
        node_ids = [r['id'] for r in node_rows]

        speaker_connections = {}
        if node_ids:
            placeholders = ', '.join(['%s'] * len(node_ids))

            # Outgoing connections (this speaker → others)
            cursor.execute(f"""
                SELECT DISTINCT sp.alias as connected_speaker, ce.edge_type
                FROM concept_edge ce
                JOIN concept_node cn_tgt ON ce.target_node_id = cn_tgt.id
                JOIN speaker sp ON cn_tgt.speaker_id = sp.id
                WHERE ce.source_node_id IN ({placeholders})
                AND sp.alias != %s
            """, node_ids + [speaker_alias])
            outgoing = cursor.fetchall()

            # Incoming connections (others → this speaker)
            cursor.execute(f"""
                SELECT DISTINCT sp.alias as connected_speaker, ce.edge_type
                FROM concept_edge ce
                JOIN concept_node cn_src ON ce.source_node_id = cn_src.id
                JOIN speaker sp ON cn_src.speaker_id = sp.id
                WHERE ce.target_node_id IN ({placeholders})
                AND sp.alias != %s
            """, node_ids + [speaker_alias])
            incoming = cursor.fetchall()

            # Aggregate connections
            for conn in outgoing:
                other = conn['connected_speaker']
                if other not in speaker_connections:
                    speaker_connections[other] = {"outgoing": [], "incoming": []}
                if conn['edge_type'] not in speaker_connections[other]["outgoing"]:
                    speaker_connections[other]["outgoing"].append(conn['edge_type'])

            for conn in incoming:
                other = conn['connected_speaker']
                if other not in speaker_connections:
                    speaker_connections[other] = {"outgoing": [], "incoming": []}
                if conn['edge_type'] not in speaker_connections[other]["incoming"]:
                    speaker_connections[other]["incoming"].append(conn['edge_type'])

        cursor.close()
        connection.close()

        # Build LLM-ready display
        lines = [
            f"=== Speaker Profile: {speaker_alias} ===",
            f"Scope: {'Session ' + str(session_id) if session_id else 'All sessions'}",
            "",
        ]

        # Sessions participated
        total_utterances = sum(d['utterance_count'] for d in session_data)
        total_words = sum(d['word_count'] or 0 for d in session_data)
        total_questions = sum(d['questions'] or 0 for d in session_data)

        lines.append(f"--- Participation Summary ---")
        lines.append(f"Sessions: {len(session_data)}")
        lines.append(f"Total utterances: {total_utterances}")
        lines.append(f"Total words: {total_words}")
        lines.append(f"Questions asked: {total_questions}")
        lines.append("")

        lines.append(f"--- By Session (with comparative metrics) ---")
        for sd in session_data:
            lines.append(f"Session {sd['session_device_id']}: {sd['session_name']}")
            lines.append(f"  Utterances: {sd['utterance_count']}, Questions: {sd['questions'] or 0}")
            # Comparative metrics for LLM to interpret
            lines.append(f"  Participation: {sd.get('participation_share_pct', 0)}% of session (equal share would be {sd.get('expected_equal_share_pct', 0)}%)")
            lines.append(f"  Question rate: {sd.get('question_rate_pct', 0)}% of their utterances are questions")
            lines.append(f"  Avg metrics: analytic={(sd['avg_analytic'] or 0):.1f}, certainty={(sd['avg_certainty'] or 0):.1f}, clout={(sd['avg_clout'] or 0):.1f}")
        lines.append("")

        # Sample quotes (diverse selection)
        if sample_quotes:
            lines.append(f"--- Sample Quotes ({len(sample_quotes)}) ---")
            for q in sample_quotes:
                quote_type = q.get('quote_type', 'statement')
                text = q['text'][:200] if q['text'] else ''
                if len(q['text']) > 200:
                    text += "..."
                cert = q.get('certainty_value') or 0
                anal = q.get('analytic_thinking_value') or 0
                label = {'question': '[Question]', 'high_certainty': '[Certain]', 'high_analytic': '[Analytic]'}.get(quote_type, '')
                lines.append(f"{label} \"{text}\"")
                lines.append(f"  (certainty={cert:.0f}, analytic={anal:.0f})")
            lines.append("")

        # Concept contributions
        concept_by_type = {}
        for cn in concept_nodes:
            t = cn['node_type'] or 'concept'
            if t not in concept_by_type:
                concept_by_type[t] = []
            concept_by_type[t].append(cn['text'][:100] if cn['text'] else '')

        lines.append(f"--- Concept Contributions ({len(concept_nodes)} total) ---")
        for ctype, concepts in concept_by_type.items():
            lines.append(f"{ctype}: {len(concepts)}")
            for c in concepts[:3]:  # Show first 3 examples per type
                lines.append(f"  - {c}")
        lines.append("")

        # Speaker connections
        if speaker_connections:
            lines.append(f"--- Interactions with Other Speakers ---")
            for other_speaker, rels in speaker_connections.items():
                out_rels = ", ".join(rels["outgoing"]) if rels["outgoing"] else "none"
                in_rels = ", ".join(rels["incoming"]) if rels["incoming"] else "none"
                lines.append(f"{other_speaker}:")
                lines.append(f"  {speaker_alias} → {other_speaker}: {out_rels}")
                lines.append(f"  {other_speaker} → {speaker_alias}: {in_rels}")
        else:
            lines.append("(No concept-level interactions with other speakers found)")

        lines.append("")
        lines.append("--- Next Steps ---")
        lines.append(f"To see {speaker_alias}'s actual utterances in a session, use:")
        lines.append(f"  get_transcript(session_id=N, speaker_filter='{speaker_alias}')")
        lines.append("")
        lines.append("=== End Speaker Profile ===")

        return {
            "display": "\n".join(lines),
            "speaker_alias": speaker_alias,
            "speaker_ids": speaker_ids,  # List of all speaker IDs (one per session)
            "sessions": [{"session_id": d['session_device_id'], "session_name": d['session_name']} for d in session_data],
            "found": True,
        }

    except Exception as e:
        logger.error(f"Speaker profile error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "display": f"Error getting speaker profile: {str(e)}",
            "error": str(e),
        }


# =============================================================================
# Tool Registry
# =============================================================================

CORE_TOOLS = {
    "list_sessions": list_sessions,
    "search_sessions": search_sessions,
    "get_transcript": get_transcript,
    "get_concept_map": get_concept_map,
    "get_7c_analysis": get_7c_analysis,
    "get_speaker_profile": get_speaker_profile,
}


def get_tool(name: str) -> Optional[Callable]:
    """Get a tool by name."""
    return CORE_TOOLS.get(name)


def execute_tool(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with parameters."""
    tool = get_tool(name)
    if not tool:
        return {
            "tool_name": name,
            "display": f"Error: Unknown tool '{name}'",
            "error": f"Unknown tool: {name}",
        }
    return tool(**params)


def get_tool_names() -> List[str]:
    """Get list of all tool names."""
    return list(CORE_TOOLS.keys())


# =============================================================================
# Tool Schema for OpenAI Function Calling
# =============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_sessions",
            "description": "List all sessions with collaboration scores (0-100). USE FIRST for superlative/comparison queries (best/worst/compare). Shows scores to identify top candidates, then call get_7c_analysis for detailed breakdown.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_sessions",
            "description": "Search sessions by topic using semantic similarity. May miss related sessions - for exhaustive comparison, use list_sessions instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic or keyword to search"},
                    "top_k": {"type": "integer", "description": "Max results", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_transcript",
            "description": "Get session transcript with speaker names and timestamps. Use for quotes, content analysis, and verifying claims.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer", "description": "Session ID"},
                    "speaker_filter": {"type": "string", "description": "Filter by speaker"},
                    "keyword_filter": {"type": "string", "description": "Filter by keyword"}
                },
                "required": ["session_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_concept_map",
            "description": "Get concept map with ideas (nodes) and relationships (edges like builds_on, challenges, contrasts_with). Shows idea structure and speaker contributions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer", "description": "Session ID"}
                },
                "required": ["session_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_7c_analysis",
            "description": "Get detailed 7C collaboration analysis (scores 0-100 + evidence quotes). REQUIRED for collaboration assessment. Call after list_sessions identifies top candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer", "description": "Session ID"}
                },
                "required": ["session_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_speaker_profile",
            "description": "Get speaker's engagement profile: sessions, metrics, concept contributions, interactions. Chain with get_transcript for specific utterances.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speaker_name": {"type": "string", "description": "Speaker name (partial match supported)"},
                    "session_id": {"type": "integer", "description": "Optional: limit to specific session"}
                },
                "required": ["speaker_name"]
            }
        }
    },
]
