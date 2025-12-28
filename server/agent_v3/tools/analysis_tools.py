"""
Analysis Tools for BLINC Agent V3

Tools for deep analysis of sessions, collaboration, and speakers.
"""

import logging
import sys
import os
from typing import Dict, Any, List, Optional

# Add server directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


def _get_db_connection():
    """Get direct MySQL connection for agent tools."""
    import mysql.connector
    return mysql.connector.connect(
        host='localhost',
        user='vagrant',
        password='vagrant',
        database='discussion_capture'
    )


def get_session_overview(session_id: int) -> Dict[str, Any]:
    """
    Get comprehensive overview of a specific session.

    Args:
        session_id: The session device ID

    Returns:
        Session overview with topics, participants, and key info
    """
    logger.info(f"Getting session overview for: {session_id}")

    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Get session info
        cursor.execute("""
            SELECT
                sd.id as session_device_id,
                sd.name as device_name,
                s.name as session_name
            FROM session_device sd
            JOIN session s ON s.id = sd.session_id
            WHERE sd.id = %s
        """, (session_id,))
        session_info = cursor.fetchone()

        if not session_info:
            cursor.close()
            connection.close()
            return {
                "tool_name": "get_session_overview",
                "error": f"Session {session_id} not found",
                "result_count": 0,
                "results": [],
                "is_relevant": False
            }

        # Get speakers
        cursor.execute("""
            SELECT DISTINCT s.id, s.alias
            FROM speaker s
            JOIN transcript t ON t.speaker_id = s.id
            WHERE t.session_device_id = %s
        """, (session_id,))
        speakers = cursor.fetchall()

        # Get transcript count and duration
        cursor.execute("""
            SELECT
                COUNT(*) as transcript_count,
                MAX(start_time + length) as duration
            FROM transcript
            WHERE session_device_id = %s
        """, (session_id,))
        transcript_stats = cursor.fetchone()

        # Get concept session info
        cursor.execute("""
            SELECT
                id,
                discourse_type,
                generation_status
            FROM concept_session
            WHERE session_device_id = %s
        """, (session_id,))
        concept_session = cursor.fetchone()

        # Get cluster themes if available
        clusters = []
        if concept_session:
            cursor.execute("""
                SELECT cluster_name, node_count
                FROM concept_cluster
                WHERE concept_session_id = %s
                ORDER BY node_count DESC
                LIMIT 5
            """, (concept_session['id'],))
            clusters = cursor.fetchall()

        cursor.close()
        connection.close()

        # Build overview
        overview = {
            "session_device_id": session_id,
            "session_name": session_info.get('session_name') or session_info.get('device_name'),
            "duration_seconds": transcript_stats.get('duration') if transcript_stats else None,
            "transcript_count": transcript_stats.get('transcript_count', 0) if transcript_stats else 0,
            "speakers": [s['alias'] for s in speakers] if speakers else [],
            "speaker_count": len(speakers) if speakers else 0,
            "discourse_type": concept_session.get('discourse_type') if concept_session else None,
            "has_concept_map": concept_session is not None,
            "main_themes": [c['cluster_name'] for c in clusters] if clusters else []
        }

        return {
            "tool_name": "get_session_overview",
            "result_count": 1,
            "results": [overview],
            "is_relevant": True
        }

    except Exception as e:
        logger.error(f"Session overview error: {e}")
        return {
            "tool_name": "get_session_overview",
            "error": str(e),
            "result_count": 0,
            "results": [],
            "is_relevant": False
        }


def get_collaboration_analysis(session_id: int) -> Dict[str, Any]:
    """
    Get 7C collaboration quality analysis for a session.

    Args:
        session_id: The session device ID

    Returns:
        7C dimensions with scores and explanations
    """
    import json as json_lib
    logger.info(f"Getting collaboration analysis for: {session_id}")

    try:
        connection = _get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Get 7C analysis (stored as JSON in analysis_summary column)
        cursor.execute("""
            SELECT analysis_summary
            FROM seven_cs_analysis
            WHERE session_device_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (session_id,))
        row = cursor.fetchone()

        cursor.close()
        connection.close()

        if not row or not row.get('analysis_summary'):
            return {
                "tool_name": "get_collaboration_analysis",
                "error": f"No 7C analysis found for session {session_id}",
                "result_count": 0,
                "results": [],
                "is_relevant": False
            }

        # Parse the JSON analysis
        analysis = row['analysis_summary']
        if isinstance(analysis, str):
            analysis = json_lib.loads(analysis)

        # Format dimensions with descriptions
        dimension_descriptions = {
            "climate": "Psychological safety and supportive atmosphere",
            "communication": "Clarity, active listening, articulation",
            "contribution": "Balanced participation, equal voice",
            "conflict": "Constructive disagreement, productive debate",
            "context": "Shared understanding, common ground",
            "constructive": "Building on others' ideas",
            "compatibility": "Working style alignment"
        }

        dimensions = {}
        overall_score = 0
        for dim_name in ['climate', 'communication', 'contribution', 'conflict',
                         'context', 'constructive', 'compatibility']:
            dim_data = analysis.get(dim_name, {})
            score = dim_data.get('score', 0)
            overall_score += score
            dimensions[dim_name] = {
                "score": score,
                "explanation": dim_data.get('explanation', ''),
                "evidence": dim_data.get('evidence', []),
                "description": dimension_descriptions.get(dim_name, '')
            }

        overall_score = overall_score / 7 if dimensions else 0

        result = {
            "session_device_id": session_id,
            "overall_score": round(overall_score, 1),
            "dimensions": dimensions
        }

        return {
            "tool_name": "get_collaboration_analysis",
            "result_count": 1,
            "results": [result],
            "is_relevant": True
        }

    except Exception as e:
        logger.error(f"Collaboration analysis error: {e}")
        return {
            "tool_name": "get_collaboration_analysis",
            "error": str(e),
            "result_count": 0,
            "results": [],
            "is_relevant": False
        }


def compare_sessions(session_ids: List[int]) -> Dict[str, Any]:
    """
    Compare multiple sessions across dimensions.

    Args:
        session_ids: List of session IDs to compare (minimum 2)

    Returns:
        Comparison across topics, metrics, and participation
    """
    logger.info(f"Comparing sessions: {session_ids}")

    if len(session_ids) < 2:
        return {
            "tool_name": "compare_sessions",
            "error": "Need at least 2 sessions to compare",
            "result_count": 0,
            "results": [],
            "is_relevant": False
        }

    try:
        sessions = []
        for sid in session_ids[:5]:  # Max 5 sessions
            overview = get_session_overview(sid)
            if overview.get('results'):
                sessions.append(overview['results'][0])

            analysis = get_collaboration_analysis(sid)
            if analysis.get('results') and sessions:
                sessions[-1]['collaboration'] = analysis['results'][0]

        if len(sessions) < 2:
            return {
                "tool_name": "compare_sessions",
                "error": "Could not retrieve enough sessions for comparison",
                "result_count": 0,
                "results": [],
                "is_relevant": False
            }

        # Build comparison
        comparison = {
            "sessions_compared": [s['session_device_id'] for s in sessions],
            "session_details": sessions,
            "summary": {
                "themes": {s['session_device_id']: s.get('main_themes', []) for s in sessions},
                "speaker_counts": {s['session_device_id']: s.get('speaker_count', 0) for s in sessions},
                "collaboration_scores": {
                    s['session_device_id']: s.get('collaboration', {}).get('overall_score', 0)
                    for s in sessions if s.get('collaboration')
                }
            }
        }

        return {
            "tool_name": "compare_sessions",
            "result_count": len(sessions),
            "results": [comparison],
            "is_relevant": True
        }

    except Exception as e:
        logger.error(f"Session comparison error: {e}")
        return {
            "tool_name": "compare_sessions",
            "error": str(e),
            "result_count": 0,
            "results": [],
            "is_relevant": False
        }


def analyze_speaker(
    speaker_name: str,
    session_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Analyze a speaker's participation patterns.

    Args:
        speaker_name: Name of the speaker to analyze
        session_ids: Optional list of sessions to limit analysis

    Returns:
        Speaker profile with participation patterns
    """
    logger.info(f"Analyzing speaker: {speaker_name} (sessions={session_ids})")

    try:
        # Try to use the speaker collection for cross-session analysis
        from rag_service import RAGService
        rag = RAGService()

        # Search speaker profiles
        results = rag.speaker_collection.query(
            query_texts=[speaker_name],
            n_results=3
        )

        formatted = []
        if results and results.get('documents'):
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}

                formatted.append({
                    "speaker_alias": metadata.get('speaker_alias', speaker_name),
                    "profile": doc,
                    "session_count": metadata.get('session_count', 0),
                    "transcript_count": metadata.get('transcript_count', 0),
                    "concept_count": metadata.get('concept_count', 0),
                    "question_count": metadata.get('question_count', 0),
                    "avg_clout": metadata.get('avg_clout', 0),
                    "avg_analytic_thinking": metadata.get('avg_analytic_thinking', 0)
                })

        if not formatted:
            # Fallback to database query
            connection = _get_db_connection()
            cursor = connection.cursor(dictionary=True)

            cursor.execute("""
                SELECT
                    s.id, s.alias,
                    COUNT(DISTINCT t.session_device_id) as session_count,
                    COUNT(t.id) as transcript_count,
                    AVG(t.clout_value) as avg_clout
                FROM speaker s
                LEFT JOIN transcript t ON t.speaker_id = s.id
                WHERE s.alias LIKE %s
                GROUP BY s.id
            """, (f"%{speaker_name}%",))
            speaker_info = cursor.fetchone()

            cursor.close()
            connection.close()

            if speaker_info:
                formatted.append({
                    "speaker_alias": speaker_info['alias'],
                    "session_count": speaker_info['session_count'],
                    "transcript_count": speaker_info['transcript_count'],
                    "avg_clout": speaker_info['avg_clout'] or 0
                })

        return {
            "tool_name": "analyze_speaker",
            "result_count": len(formatted),
            "results": formatted,
            "is_relevant": len(formatted) > 0
        }

    except Exception as e:
        logger.error(f"Speaker analysis error: {e}")
        return {
            "tool_name": "analyze_speaker",
            "error": str(e),
            "result_count": 0,
            "results": [],
            "is_relevant": False
        }
