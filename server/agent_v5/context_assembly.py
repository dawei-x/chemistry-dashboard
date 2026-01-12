"""
Context Assembly for Agent V5.

Pre-fetches relevant context based on query intent.
Integrates structured database queries with RAG semantic search.
"""

import logging
from typing import Dict, List, Optional, Any
import mysql.connector
import os

from .query_understanding import QueryIntent, get_session_name

logger = logging.getLogger(__name__)


def _get_db_connection():
    """Get database connection."""
    return mysql.connector.connect(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        user=os.getenv('MYSQL_USER', 'vagrant'),
        password=os.getenv('MYSQL_PASSWORD', 'vagrant'),
        database=os.getenv('MYSQL_DATABASE', 'discussion_capture')
    )


def assemble_context(
    intent: QueryIntent,
    query: str,
    rag_service=None
) -> Dict[str, Any]:
    """
    Assemble context based on query intent.

    Args:
        intent: Classified query intent
        query: Original query string
        rag_service: Optional RAG service for semantic retrieval

    Returns:
        Dictionary with:
        - context_text: Formatted context for LLM
        - retrieval_metadata: What was retrieved and why
        - sessions_loaded: List of session IDs in context
    """
    if not intent.needs_retrieval:
        return {
            'context_text': '',
            'retrieval_metadata': {'mode': 'none', 'reason': 'No retrieval needed'},
            'sessions_loaded': []
        }

    # Route to appropriate assembly function
    if intent.retrieval_mode == 'structured':
        return _assemble_structured_context(intent)

    elif intent.retrieval_mode == 'semantic':
        return _assemble_semantic_context(intent, query, rag_service)

    elif intent.retrieval_mode == 'contrastive':
        return _assemble_contrastive_context(intent, query, rag_service)

    elif intent.retrieval_mode == 'hybrid':
        return _assemble_hybrid_context(intent, query, rag_service)

    else:  # agentic_only
        return {
            'context_text': _get_session_overview(),
            'retrieval_metadata': {
                'mode': 'agentic_only',
                'reason': 'Query requires exploration - providing session overview'
            },
            'sessions_loaded': []
        }


def _assemble_structured_context(intent: QueryIntent) -> Dict[str, Any]:
    """
    Assemble context using direct database queries.
    Used when entities (session/speaker) are explicitly specified.
    """
    context_parts = []
    sessions_loaded = []

    # Speaker-focused query
    if intent.intent_type == 'speaker' and intent.speaker_names:
        for speaker in intent.speaker_names:
            session_id = intent.session_ids[0] if intent.session_ids else None
            speaker_context = _get_speaker_context(speaker, session_id)
            if speaker_context:
                context_parts.append(speaker_context)
                if session_id:
                    sessions_loaded.append(session_id)

    # Session-focused query
    elif intent.session_ids:
        for session_id in intent.session_ids:
            session_context = _get_session_context(session_id)
            if session_context:
                context_parts.append(session_context)
                sessions_loaded.append(session_id)

    context_text = '\n\n---\n\n'.join(context_parts) if context_parts else ''

    return {
        'context_text': context_text,
        'retrieval_metadata': {
            'mode': 'structured',
            'sessions': sessions_loaded,
            'speakers': intent.speaker_names,
            'reason': f'Direct retrieval for {intent.intent_type} query'
        },
        'sessions_loaded': sessions_loaded
    }


def _assemble_semantic_context(
    intent: QueryIntent,
    query: str,
    rag_service
) -> Dict[str, Any]:
    """
    Assemble context using RAG semantic search.
    Used for topic-based and pattern queries.
    """
    if not rag_service:
        logger.warning("RAG service not available for semantic retrieval")
        return {
            'context_text': _get_session_overview(),
            'retrieval_metadata': {'mode': 'fallback', 'reason': 'RAG service unavailable'},
            'sessions_loaded': []
        }

    try:
        # Search across specified collections
        results = rag_service.search_sessions_multi(
            query=query,
            collections=intent.target_collections,
            n_results=5
        )

        # Get session IDs from fused results
        session_ids = [
            r['session_device_id']
            for r in results.get('fused_results', [])[:3]
            if r.get('session_device_id')
        ]

        # Fetch full context for top sessions
        context_parts = []
        for session_id in session_ids:
            session_context = _get_session_context(session_id, include_full_transcript=False)
            if session_context:
                context_parts.append(session_context)

        context_text = '\n\n---\n\n'.join(context_parts) if context_parts else ''

        return {
            'context_text': context_text,
            'retrieval_metadata': {
                'mode': 'semantic',
                'collections_searched': intent.target_collections,
                'sessions_found': session_ids,
                'reason': f'Semantic search for: {query[:50]}...'
            },
            'sessions_loaded': session_ids
        }

    except Exception as e:
        logger.error(f"Semantic retrieval error: {e}")
        return {
            'context_text': _get_session_overview(),
            'retrieval_metadata': {'mode': 'fallback', 'error': str(e)},
            'sessions_loaded': []
        }


def _assemble_contrastive_context(
    intent: QueryIntent,
    query: str,
    rag_service
) -> Dict[str, Any]:
    """
    Assemble context for "why" queries by comparing high vs low performers.
    """
    metric = intent.metric_focus or 'communication'

    if not rag_service:
        # Fallback to manual contrastive retrieval
        return _assemble_manual_contrastive(metric)

    try:
        # Get high and low performing sessions
        high_sessions, low_sessions = rag_service.get_contrastive_sessions(
            metric_name=f'{metric}_score',
            n_high=2,
            n_low=2
        )

        # Assemble context for both groups
        high_context = _format_session_group(high_sessions, f"HIGH {metric.upper()} SESSIONS")
        low_context = _format_session_group(low_sessions, f"LOW {metric.upper()} SESSIONS")

        context_text = f"""## CONTRASTIVE ANALYSIS CONTEXT

This query asks WHY some discussions perform better. I've retrieved sessions with HIGH and LOW {metric} scores for comparison.

{high_context}

{low_context}

## ANALYSIS GUIDANCE
Compare these groups to identify what distinguishes high-performing discussions from low-performing ones. Ground your analysis in specific quotes and metrics.
"""

        return {
            'context_text': context_text,
            'retrieval_metadata': {
                'mode': 'contrastive',
                'metric': metric,
                'high_sessions': high_sessions,
                'low_sessions': low_sessions,
                'reason': f'Contrastive retrieval for {metric} comparison'
            },
            'sessions_loaded': high_sessions + low_sessions
        }

    except Exception as e:
        logger.error(f"Contrastive retrieval error: {e}")
        return _assemble_manual_contrastive(metric)


def _assemble_manual_contrastive(metric: str) -> Dict[str, Any]:
    """Fallback contrastive retrieval using direct DB queries."""
    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)

        # Get sessions with 7C scores (use subquery to get one analysis per session)
        cursor.execute("""
            SELECT
                sd.id as session_device_id,
                s.name as session_name,
                (SELECT analysis_summary FROM seven_cs_analysis
                 WHERE session_device_id = sd.id
                 ORDER BY id DESC LIMIT 1) as analysis_summary
            FROM session_device sd
            JOIN session s ON sd.session_id = s.id
            WHERE EXISTS (SELECT 1 FROM seven_cs_analysis sca WHERE sca.session_device_id = sd.id)
        """)
        sessions = cursor.fetchall()
        cursor.close()
        connection.close()

        if not sessions:
            return {
                'context_text': _get_session_overview(),
                'retrieval_metadata': {'mode': 'fallback', 'reason': 'No 7C data available'},
                'sessions_loaded': []
            }

        # Sort by metric
        import json
        scored_sessions = []
        for s in sessions:
            try:
                summary = json.loads(s['analysis_summary']) if isinstance(s['analysis_summary'], str) else s['analysis_summary']
                score = summary.get(metric, {}).get('score', 0)
                scored_sessions.append((s, score))
            except:
                continue

        scored_sessions.sort(key=lambda x: x[1], reverse=True)

        # Get top 2 and bottom 2, avoiding duplicates
        high_sessions = [s[0]['session_device_id'] for s in scored_sessions[:2]]
        low_sessions = [s[0]['session_device_id'] for s in scored_sessions[-2:]
                        if s[0]['session_device_id'] not in high_sessions]

        # If we don't have enough low sessions, try to get more
        if len(low_sessions) < 2:
            for s in reversed(scored_sessions[:-2]):
                if s[0]['session_device_id'] not in high_sessions and s[0]['session_device_id'] not in low_sessions:
                    low_sessions.append(s[0]['session_device_id'])
                    if len(low_sessions) >= 2:
                        break

        high_context = _format_session_group(high_sessions, f"HIGH {metric.upper()} SESSIONS")
        low_context = _format_session_group(low_sessions, f"LOW {metric.upper()} SESSIONS")

        context_text = f"""## CONTRASTIVE ANALYSIS CONTEXT

{high_context}

{low_context}
"""

        return {
            'context_text': context_text,
            'retrieval_metadata': {
                'mode': 'contrastive_manual',
                'metric': metric,
                'high_sessions': high_sessions,
                'low_sessions': low_sessions
            },
            'sessions_loaded': high_sessions + low_sessions
        }

    except Exception as e:
        logger.error(f"Manual contrastive error: {e}")
        return {
            'context_text': _get_session_overview(),
            'retrieval_metadata': {'mode': 'fallback', 'error': str(e)},
            'sessions_loaded': []
        }


def _assemble_hybrid_context(
    intent: QueryIntent,
    query: str,
    rag_service
) -> Dict[str, Any]:
    """
    Hybrid retrieval: metric filtering + semantic search.
    """
    if not rag_service:
        return _assemble_structured_context(intent)

    try:
        # Use hybrid search from RAG service
        metric_filters = {}
        if intent.metric_focus:
            metric_filters[f'{intent.metric_focus}_score'] = ('>=', 60)

        results = rag_service.hybrid_session_search(
            query=query,
            metric_filters=metric_filters,
            n_results=4
        )

        session_ids = [r['session_device_id'] for r in results if r.get('session_device_id')]

        context_parts = []
        for session_id in session_ids[:3]:
            session_context = _get_session_context(session_id)
            if session_context:
                context_parts.append(session_context)

        context_text = '\n\n---\n\n'.join(context_parts)

        return {
            'context_text': context_text,
            'retrieval_metadata': {
                'mode': 'hybrid',
                'metric_filters': metric_filters,
                'sessions_found': session_ids,
                'reason': 'Hybrid metric + semantic search'
            },
            'sessions_loaded': session_ids[:3]
        }

    except Exception as e:
        logger.error(f"Hybrid retrieval error: {e}")
        return _assemble_semantic_context(intent, query, rag_service)


def _get_session_context(session_id: int, include_full_transcript: bool = True) -> Optional[str]:
    """Get complete context for a single session."""
    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)

        # Get session info
        cursor.execute("""
            SELECT s.name, sd.id as session_device_id
            FROM session_device sd
            JOIN session s ON sd.session_id = s.id
            WHERE sd.id = %s
        """, (session_id,))
        session = cursor.fetchone()

        if not session:
            cursor.close()
            connection.close()
            return None

        session_name = session['name']
        parts = [f"## Session: {session_name} (ID: {session_id})"]

        # Get 7C scores
        cursor.execute("""
            SELECT analysis_summary
            FROM seven_cs_analysis
            WHERE session_device_id = %s
        """, (session_id,))
        seven_c = cursor.fetchone()

        if seven_c and seven_c['analysis_summary']:
            import json
            try:
                summary = json.loads(seven_c['analysis_summary']) if isinstance(seven_c['analysis_summary'], str) else seven_c['analysis_summary']
                parts.append("\n### Collaboration Quality (7C Scores)")
                for dim in ['communication', 'climate', 'contribution', 'conflict', 'constructive']:
                    if dim in summary:
                        score = summary[dim].get('score', 'N/A')
                        parts.append(f"- {dim.capitalize()}: {score}")
            except:
                pass

        # Get concept map summary
        cursor.execute("""
            SELECT cs.id, COUNT(DISTINCT cn.id) as node_count, COUNT(DISTINCT ce.id) as edge_count
            FROM concept_session cs
            LEFT JOIN concept_node cn ON cs.id = cn.concept_session_id
            LEFT JOIN concept_edge ce ON cs.id = ce.concept_session_id
            WHERE cs.session_device_id = %s
            GROUP BY cs.id
        """, (session_id,))
        concept_map = cursor.fetchone()

        if concept_map:
            parts.append(f"\n### Concept Map")
            parts.append(f"- Nodes: {concept_map['node_count']}")
            parts.append(f"- Edges: {concept_map['edge_count']}")

            # Get edge type distribution
            cursor.execute("""
                SELECT ce.edge_type, COUNT(*) as count
                FROM concept_edge ce
                JOIN concept_session cs ON ce.concept_session_id = cs.id
                WHERE cs.session_device_id = %s
                GROUP BY ce.edge_type
            """, (session_id,))
            edge_types = cursor.fetchall()
            if edge_types:
                parts.append("- Edge types: " + ", ".join(f"{e['edge_type']}({e['count']})" for e in edge_types))

        # Get speakers
        cursor.execute("""
            SELECT DISTINCT sp.alias, COUNT(*) as utterance_count
            FROM transcript t
            JOIN speaker sp ON t.speaker_id = sp.id
            WHERE t.session_device_id = %s
            GROUP BY sp.id, sp.alias
            ORDER BY utterance_count DESC
        """, (session_id,))
        speakers = cursor.fetchall()

        if speakers:
            parts.append(f"\n### Participants")
            for sp in speakers:
                parts.append(f"- {sp['alias']}: {sp['utterance_count']} utterances")

        # Get transcript (optionally summarized)
        if include_full_transcript:
            cursor.execute("""
                SELECT t.transcript, t.start_time, sp.alias as speaker
                FROM transcript t
                LEFT JOIN speaker sp ON t.speaker_id = sp.id
                WHERE t.session_device_id = %s
                ORDER BY t.start_time
                LIMIT 50
            """, (session_id,))
            transcripts = cursor.fetchall()

            if transcripts:
                parts.append(f"\n### Transcript (first 50 utterances)")
                for t in transcripts:
                    speaker = t['speaker'] or 'Unknown'
                    time_min = int(t['start_time'] // 60)
                    time_sec = int(t['start_time'] % 60)
                    parts.append(f"[{speaker} at {time_min}:{time_sec:02d}]: {t['transcript']}")

        cursor.close()
        connection.close()

        return '\n'.join(parts)

    except Exception as e:
        logger.error(f"Error getting session context for {session_id}: {e}")
        return None


def _get_speaker_context(speaker_name: str, session_id: Optional[int] = None) -> Optional[str]:
    """Get context for a specific speaker."""
    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)

        # Find speaker
        cursor.execute("""
            SELECT id, alias FROM speaker
            WHERE LOWER(alias) LIKE %s
            LIMIT 1
        """, (f'%{speaker_name.lower()}%',))
        speaker = cursor.fetchone()

        if not speaker:
            cursor.close()
            connection.close()
            return None

        speaker_id = speaker['id']
        speaker_alias = speaker['alias']

        # Build query
        session_filter = "AND t.session_device_id = %s" if session_id else ""
        params = [speaker_id]
        if session_id:
            params.append(session_id)

        cursor.execute(f"""
            SELECT
                t.transcript as text,
                t.start_time,
                t.question as is_question,
                t.word_count,
                s.name as session_name
            FROM transcript t
            JOIN session_device sd ON t.session_device_id = sd.id
            JOIN session s ON sd.session_id = s.id
            WHERE t.speaker_id = %s {session_filter}
            ORDER BY t.start_time
        """, params)
        utterances = cursor.fetchall()

        cursor.close()
        connection.close()

        if not utterances:
            return None

        # Format speaker context
        parts = [f"## Speaker: {speaker_alias}"]

        if session_id:
            parts.append(f"Session: {utterances[0]['session_name']}")

        total_words = sum(u['word_count'] or 0 for u in utterances)
        questions = sum(1 for u in utterances if u['is_question'])

        parts.append(f"\n### Summary")
        parts.append(f"- Total utterances: {len(utterances)}")
        parts.append(f"- Total words: {total_words}")
        parts.append(f"- Questions asked: {questions}")

        parts.append(f"\n### Utterances")
        for u in utterances[:30]:  # Limit to 30
            time_min = int(u['start_time'] // 60)
            time_sec = int(u['start_time'] % 60)
            q_marker = " [Q]" if u['is_question'] else ""
            parts.append(f"[{time_min}:{time_sec:02d}]{q_marker}: {u['text']}")

        return '\n'.join(parts)

    except Exception as e:
        logger.error(f"Error getting speaker context: {e}")
        return None


def _format_session_group(session_ids: List[int], header: str) -> str:
    """Format a group of sessions for contrastive display."""
    parts = [f"### {header}"]

    for session_id in session_ids:
        context = _get_session_context(session_id, include_full_transcript=False)
        if context:
            parts.append(context)
            parts.append("")

    return '\n'.join(parts)


def _get_session_overview() -> str:
    """Get overview of all available sessions."""
    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True, buffered=True)

        cursor.execute("""
            SELECT
                sd.id as session_device_id,
                s.name,
                COUNT(DISTINCT t.id) as transcript_count
            FROM session_device sd
            JOIN session s ON sd.session_id = s.id
            LEFT JOIN transcript t ON sd.id = t.session_device_id
            GROUP BY sd.id, s.name
            ORDER BY s.name
        """)
        sessions = cursor.fetchall()

        cursor.close()
        connection.close()

        parts = ["## Available Sessions\n"]
        for s in sessions:
            parts.append(f"- **{s['name']}** (ID: {s['session_device_id']}, {s['transcript_count']} utterances)")

        parts.append("\n*Use tools to explore specific sessions or search for topics.*")

        return '\n'.join(parts)

    except Exception as e:
        logger.error(f"Error getting session overview: {e}")
        return "Error retrieving session overview."
