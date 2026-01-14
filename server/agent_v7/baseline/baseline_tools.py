"""
Baseline Tools for Transcript-Only Agent (AIED 2026)

These tools provide transcript access WITHOUT derived analysis:
- No LIWC metrics (analytic thinking, certainty scores)
- No concept map data
- No 7C collaboration scores
- Speaker data limited to raw utterances

This enables fair comparison to demonstrate how artifacts enhance LLM reasoning.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def _get_db_connection():
    """Get direct MySQL connection."""
    import mysql.connector
    return mysql.connector.connect(
        host='localhost',
        user='vagrant',
        password='vagrant',
        database='discussion_capture'
    )


def get_transcript_baseline(session_id: int) -> Dict[str, Any]:
    """
    Get complete transcript for a session WITHOUT LIWC scores.

    This is the baseline version that excludes linguistic analysis metrics.
    For fair comparison with the full agent that has LIWC access.

    Args:
        session_id: The session to retrieve transcript for

    Returns:
        - summary: total utterances, words, questions (NO avg LIWC scores)
        - speaker_profiles: per-speaker statistics (NO LIWC metrics)
        - utterances: full transcript with timestamps (NO per-utterance LIWC)
    """
    logger.info(f"[BASELINE] Getting transcript for session {session_id} (no LIWC)")

    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Get session metadata
        cursor.execute("""
            SELECT COALESCE(s.name, sd.name) as session_name
            FROM session_device sd
            JOIN session s ON s.id = sd.session_id
            WHERE sd.id = %s
        """, (session_id,))
        meta = cursor.fetchone()

        if not meta:
            cursor.close()
            connection.close()
            return {
                "tool_name": "get_transcript",
                "session_id": session_id,
                "available": False,
                "reason": f"Session {session_id} not found",
                "is_relevant": False
            }

        # Get speaker statistics (WITHOUT LIWC averages)
        cursor.execute("""
            SELECT
                sp.id as speaker_id,
                sp.alias,
                COUNT(t.id) as utterance_count,
                SUM(t.word_count) as word_count,
                SUM(CASE WHEN t.question = 1 THEN 1 ELSE 0 END) as questions_asked,
                MIN(t.start_time) as first_utterance,
                MAX(t.start_time) as last_utterance
            FROM transcript t
            JOIN speaker sp ON t.speaker_id = sp.id
            WHERE t.session_device_id = %s
            GROUP BY sp.id, sp.alias
        """, (session_id,))
        speakers = cursor.fetchall()

        # Get all transcript chunks (WITHOUT LIWC values)
        cursor.execute("""
            SELECT
                t.id as chunk_id,
                sp.alias as speaker,
                t.transcript as text,
                t.start_time,
                t.word_count,
                t.question as is_question
            FROM transcript t
            JOIN speaker sp ON t.speaker_id = sp.id
            WHERE t.session_device_id = %s
            ORDER BY t.start_time
        """, (session_id,))
        chunks_raw = cursor.fetchall()

        cursor.close()
        connection.close()

        if not chunks_raw:
            return {
                "tool_name": "get_transcript",
                "session_id": session_id,
                "session_name": meta['session_name'],
                "available": False,
                "reason": "No transcripts found",
                "is_relevant": False
            }

        # Format speaker profiles (NO LIWC metrics)
        speaker_profiles = [{
            "speaker_id": s['speaker_id'],
            "alias": s['alias'],
            "utterance_count": s['utterance_count'],
            "word_count": s['word_count'] or 0,
            "questions_asked": s['questions_asked'] or 0
            # NO avg_analytic_thinking
            # NO avg_certainty
        } for s in speakers]

        # Format utterances (NO LIWC values)
        utterances = [{
            "chunk_id": c['chunk_id'],
            "speaker": c['speaker'],
            "text": c['text'],
            "start_time": c['start_time'],
            "word_count": c['word_count'] or 0,
            "is_question": bool(c['is_question'])
            # NO analytic_thinking
            # NO certainty
        } for c in chunks_raw]

        # Calculate session-level statistics (NO LIWC averages)
        total_words = sum(c['word_count'] or 0 for c in chunks_raw)
        total_questions = sum(1 for c in chunks_raw if c['is_question'])

        return {
            "tool_name": "get_transcript",
            "session_id": session_id,
            "session_name": meta['session_name'],
            "available": True,
            "summary": {
                "total_utterances": len(chunks_raw),
                "total_words": total_words,
                "total_questions": total_questions,
                "speaker_count": len(speakers)
                # NO session_avg_analytic_thinking
                # NO session_avg_certainty
            },
            "speaker_profiles": speaker_profiles,
            "utterances": utterances,
            "is_relevant": True,
            "result_count": len(chunks_raw)
        }

    except Exception as e:
        logger.error(f"[BASELINE] Get transcript error: {e}")
        return {
            "tool_name": "get_transcript",
            "session_id": session_id,
            "available": False,
            "error": str(e),
            "is_relevant": False
        }


def get_speaker_utterances(speaker_name: str, session_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get a speaker's transcript utterances only - no derived analysis.

    This is the baseline version that returns ONLY raw transcript quotes.
    Unlike get_speaker_profile, this does NOT include:
    - communication_style (LIWC metrics)
    - contributions (concept map analysis)
    - interaction_patterns (derived metrics)
    - reasoning_hints (interpreted data)

    Args:
        speaker_name: Name or alias of the speaker
        session_id: Optional session filter

    Returns:
        - speaker_alias, speaker_id
        - participation: utterance count, word count, questions (raw stats)
        - utterances: list of raw transcript quotes with timestamps
    """
    logger.info(f"[BASELINE] Getting speaker utterances for '{speaker_name}' (session={session_id})")

    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Find speaker by name/alias
        cursor.execute("""
            SELECT id, alias FROM speaker
            WHERE alias LIKE %s OR alias LIKE %s
            LIMIT 1
        """, (f"%{speaker_name}%", speaker_name))
        speaker = cursor.fetchone()

        if not speaker:
            cursor.close()
            connection.close()
            return {
                "tool_name": "get_speaker_utterances",
                "speaker_name": speaker_name,
                "error": f"Speaker '{speaker_name}' not found",
                "is_relevant": False,
                "result_count": 0
            }

        # Build session filter
        session_filter = ""
        params = [speaker['id']]
        if session_id:
            session_filter = "AND t.session_device_id = %s"
            params.append(session_id)

        # Get utterances (NO LIWC fields)
        cursor.execute(f"""
            SELECT
                t.transcript as text,
                t.start_time as timestamp,
                t.session_device_id as session_id,
                COALESCE(s.name, sd.name) as session_name,
                t.word_count,
                t.question as is_question
            FROM transcript t
            JOIN session_device sd ON t.session_device_id = sd.id
            JOIN session s ON sd.session_id = s.id
            WHERE t.speaker_id = %s {session_filter}
            ORDER BY t.start_time
            LIMIT 100
        """, params)

        utterances_raw = cursor.fetchall()

        # Calculate basic stats
        total_words = sum(u['word_count'] or 0 for u in utterances_raw)
        total_questions = sum(1 for u in utterances_raw if u['is_question'])

        # Get unique sessions
        sessions_seen = {}
        for u in utterances_raw:
            sid = u['session_id']
            if sid not in sessions_seen:
                sessions_seen[sid] = u['session_name']

        # Format utterances
        utterances = [{
            "text": u['text'],
            "timestamp": u['timestamp'],
            "session_id": u['session_id'],
            "session_name": u['session_name'],
            "word_count": u['word_count'] or 0,
            "is_question": bool(u['is_question'])
        } for u in utterances_raw]

        # Sample quotes for quick reference
        sample_quotes = [u['text'][:200] + "..." if len(u['text']) > 200 else u['text']
                        for u in utterances_raw[:5]]

        cursor.close()
        connection.close()

        return {
            "tool_name": "get_speaker_utterances",
            "speaker_alias": speaker['alias'],
            "speaker_id": speaker['id'],
            "session_filter": session_id,
            "sessions_found": list(sessions_seen.keys()),
            "participation": {
                "total_utterances": len(utterances),
                "total_words": total_words,
                "total_questions": total_questions,
                "sessions_participated": len(sessions_seen)
            },
            "sample_quotes": sample_quotes,
            "utterances": utterances,
            "is_relevant": len(utterances) > 0,
            "result_count": len(utterances)
        }

    except Exception as e:
        logger.error(f"[BASELINE] Get speaker utterances error: {e}")
        return {
            "tool_name": "get_speaker_utterances",
            "speaker_name": speaker_name,
            "error": str(e),
            "is_relevant": False,
            "result_count": 0
        }


def search_transcripts_baseline(query: str, session_ids: List[int] = None,
                                 speaker: str = None, limit: int = 10) -> Dict[str, Any]:
    """
    Search transcripts using RAG - baseline version without LIWC in results.

    Args:
        query: Search query
        session_ids: Optional list of session IDs to filter
        speaker: Optional speaker name filter
        limit: Maximum results

    Returns:
        Transcript chunks matching the query (no LIWC scores)
    """
    logger.info(f"[BASELINE] Searching transcripts: '{query}' (sessions={session_ids}, speaker={speaker})")

    try:
        # Use RAG service for semantic search
        from rag_service import RAGService
        rag = RAGService()

        # Build filter
        where_filter = {}
        if session_ids:
            where_filter["session_device_id"] = {"$in": session_ids}
        if speaker:
            where_filter["speaker"] = {"$eq": speaker}

        # Search
        results = rag.search(
            query=query,
            collection_name="transcripts",
            n_results=limit,
            where=where_filter if where_filter else None
        )

        # Format results (strip any LIWC data that might be in metadata)
        formatted_results = []
        for doc, meta, score in zip(
            results.get('documents', [[]])[0],
            results.get('metadatas', [[]])[0],
            results.get('distances', [[]])[0]
        ):
            formatted_results.append({
                "text": doc,
                "speaker": meta.get('speaker', 'Unknown'),
                "session_id": meta.get('session_device_id'),
                "session_name": meta.get('session_name', 'Unknown'),
                "timestamp": meta.get('start_time'),
                "relevance_score": 1 - score  # Convert distance to similarity
                # NO analytic_thinking
                # NO certainty
            })

        return {
            "tool_name": "search_transcripts",
            "query": query,
            "session_filter": session_ids,
            "speaker_filter": speaker,
            "results": formatted_results,
            "is_relevant": len(formatted_results) > 0,
            "result_count": len(formatted_results)
        }

    except Exception as e:
        logger.error(f"[BASELINE] Search transcripts error: {e}")
        return {
            "tool_name": "search_transcripts",
            "query": query,
            "error": str(e),
            "results": [],
            "is_relevant": False,
            "result_count": 0
        }
