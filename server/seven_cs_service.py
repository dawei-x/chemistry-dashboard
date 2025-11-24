import logging
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from datetime import datetime
from app import db
from tables.seven_cs_analysis import SevenCsAnalysis
from tables.seven_cs_coded_segment import SevenCsCodedSegment
from tables.speaker import Speaker
import database as db_helper

load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# 7C Framework Definition
SEVEN_CS_FRAMEWORK = {
    "climate": {
        "description": "Emotional safety, respect, and comfort in group interactions",
        "indicators": ["respect", "comfort", "tone", "welcome", "safe", "listening", "being heard"],
        "scoring_criteria": "High scores indicate a respectful, comfortable environment where members feel safe to share ideas"
    },
    "communication": {
        "description": "Quality and effectiveness of information exchange",
        "indicators": ["verbal", "nonverbal", "discussion", "listening", "sharing", "goals", "expectations"],
        "scoring_criteria": "High scores indicate clear, active communication with good listening and information sharing"
    },
    "compatibility": {
        "description": "How well group members' working styles complement each other",
        "indicators": ["working style", "active", "equal distribution", "friends", "creative vision", "complementary skills"],
        "scoring_criteria": "High scores indicate compatible work styles and good team synergy"
    },
    "conflict": {
        "description": "Approaches to handling disagreements and contentious situations",
        "indicators": ["adapting", "differences", "confronting", "mediator", "resolution", "external validation"],
        "scoring_criteria": "High scores indicate effective conflict resolution and constructive handling of disagreements"
    },
    "context": {
        "description": "Environmental factors and situational awareness",
        "indicators": ["privacy", "out of school", "in/out of context", "interest", "group members", "setting"],
        "scoring_criteria": "High scores indicate appropriate context awareness and comfort with the environment"
    },
    "contribution": {
        "description": "Individual participation and effort balance",
        "indicators": ["accountable", "balance of work", "tracking", "engagement", "effort", "verbal contributions"],
        "scoring_criteria": "High scores indicate balanced participation and equitable contribution from all members"
    },
    "constructive": {
        "description": "Goal achievement and mutual benefit",
        "indicators": ["goal", "product", "efficiency", "learning", "mutual benefit", "insights"],
        "scoring_criteria": "High scores indicate productive collaboration toward shared goals with mutual learning"
    }
}

def analyze_session_seven_cs(session_device_id):
    """
    Perform full 7C analysis for a session device.
    This is called as a background job when a session ends or manually triggered.

    Args:
        session_device_id: ID of the session device to analyze

    Returns:
        SevenCsAnalysis object if successful, None if failed
    """
    if not client:
        logging.error(f"Cannot perform 7C analysis - OpenAI client not initialized")
        return None

    try:
        start_time = time.time()

        # Create analysis entry
        analysis = SevenCsAnalysis(
            session_device_id=session_device_id,
            analysis_status='processing'
        )
        db.session.add(analysis)
        db.session.commit()

        # Get all transcripts for the session
        transcripts = db_helper.get_transcripts(session_device_id=session_device_id)

        if not transcripts:
            logging.info(f"No transcripts found for session_device {session_device_id}")
            analysis.analysis_status = 'failed'
            db.session.commit()
            return None

        # Process transcripts in sliding windows for coding
        coded_segments = code_transcripts_with_seven_cs(analysis.id, transcripts)

        # Generate overall summary and scores
        full_transcript_text = prepare_full_transcript(transcripts)
        summary_result = generate_overall_seven_cs_summary(full_transcript_text, coded_segments)

        # Calculate processing time and token usage
        processing_time = time.time() - start_time
        total_tokens = estimate_tokens(full_transcript_text) + len(coded_segments) * 50

        # Update analysis with results
        analysis.update_summary(
            summary_data=summary_result,
            segments_analyzed=len(coded_segments),
            processing_time=processing_time,
            tokens_used=total_tokens
        )
        db.session.commit()

        logging.info(f"Successfully completed 7C analysis for session_device {session_device_id}")
        return analysis

    except Exception as e:
        logging.error(f"Error in 7C analysis for session_device {session_device_id}: {str(e)}")
        if 'analysis' in locals():
            analysis.analysis_status = 'failed'
            db.session.commit()
        return None

def code_transcripts_with_seven_cs(analysis_id, transcripts, window_size=30, overlap=10):
    """
    Process transcripts in sliding windows and code them with 7C dimensions.

    Args:
        analysis_id: ID of the analysis record
        transcripts: List of transcript objects
        window_size: Size of sliding window in seconds
        overlap: Overlap between windows in seconds

    Returns:
        List of SevenCsCodedSegment objects
    """
    coded_segments = []

    # Sort transcripts by start time
    sorted_transcripts = sorted(transcripts, key=lambda t: t.start_time)

    if not sorted_transcripts:
        logging.info(f"No transcripts to process for analysis {analysis_id}")
        return coded_segments

    # Create sliding windows
    start_time = sorted_transcripts[0].start_time
    end_time = sorted_transcripts[-1].start_time + sorted_transcripts[-1].length

    logging.info(f"Processing {len(sorted_transcripts)} transcripts from {start_time} to {end_time} seconds")

    current_window_start = start_time

    while current_window_start < end_time:
        window_end = min(current_window_start + window_size, end_time)

        # Get transcripts in this window
        window_transcripts = [
            t for t in sorted_transcripts
            if t.start_time < window_end and t.start_time + t.length > current_window_start
        ]

        if window_transcripts:
            # Prepare window text
            window_text = prepare_window_text(window_transcripts)
            logging.info(f"Processing window {current_window_start}-{window_end}s with {len(window_transcripts)} transcripts")

            # Code this window
            window_codings = code_window_with_seven_cs(
                window_text,
                analysis_id,
                window_transcripts,
                current_window_start,
                window_end
            )

            logging.info(f"Window produced {len(window_codings)} codings")
            coded_segments.extend(window_codings)

        # Move to next window
        current_window_start += (window_size - overlap)

    return coded_segments

def get_speaker_aliases(transcripts):
    """
    Get a mapping of speaker_id to alias for all speakers in transcripts.

    Args:
        transcripts: List of transcript objects

    Returns:
        Dictionary mapping speaker_id to speaker alias
    """
    # Collect unique speaker IDs
    speaker_ids = set()
    for t in transcripts:
        if t.speaker_id:
            speaker_ids.add(t.speaker_id)

    # Fetch speakers from database
    speaker_map = {}
    for speaker_id in speaker_ids:
        speaker = db_helper.get_speakers(id=speaker_id)
        if speaker:
            speaker_map[speaker_id] = speaker.get_alias()
        else:
            # Fallback if speaker not found
            speaker_map[speaker_id] = f"Speaker {speaker_id}"

    return speaker_map

def prepare_window_text(transcripts):
    """
    Prepare transcript text for a window of transcripts.

    Args:
        transcripts: List of transcript objects in the window

    Returns:
        Formatted string of transcripts
    """
    # Get speaker aliases
    speaker_aliases = get_speaker_aliases(transcripts)

    transcript_lines = []
    for t in transcripts:
        # Determine speaker name
        if t.speaker_id and t.speaker_id in speaker_aliases:
            speaker = speaker_aliases[t.speaker_id]
        elif t.speaker_tag:
            speaker = f"Speaker {t.speaker_tag}"
        else:
            speaker = "Unknown"

        time_min, time_sec = divmod(t.start_time, 60)
        transcript_lines.append(f"[{speaker} at {time_min}:{time_sec:02d}]: {t.transcript}")

    return "\n".join(transcript_lines)

def code_window_with_seven_cs(window_text, analysis_id, window_transcripts, window_start, window_end):
    """
    Use LLM to code a window of transcripts with 7C dimensions.

    Args:
        window_text: Formatted text of the window
        analysis_id: ID of the analysis record
        window_transcripts: List of transcript objects in the window
        window_start: Start time of window in seconds
        window_end: End time of window in seconds

    Returns:
        List of SevenCsCodedSegment objects
    """
    coded_segments = []

    # Build the prompt
    prompt = f"""Analyze this discussion segment and identify which of the 7 dimensions of collaboration are present.

For EACH dimension that is clearly present in this segment, provide:
1. Whether it's strongly present (yes/no)
2. A specific quote from the transcript showing this dimension
3. Brief explanation of why this represents the dimension
4. Confidence level (0.0 to 1.0)

The 7 dimensions are:
- Climate: {SEVEN_CS_FRAMEWORK['climate']['description']}
  Indicators: {', '.join(SEVEN_CS_FRAMEWORK['climate']['indicators'])}

- Communication: {SEVEN_CS_FRAMEWORK['communication']['description']}
  Indicators: {', '.join(SEVEN_CS_FRAMEWORK['communication']['indicators'])}

- Compatibility: {SEVEN_CS_FRAMEWORK['compatibility']['description']}
  Indicators: {', '.join(SEVEN_CS_FRAMEWORK['compatibility']['indicators'])}

- Conflict: {SEVEN_CS_FRAMEWORK['conflict']['description']}
  Indicators: {', '.join(SEVEN_CS_FRAMEWORK['conflict']['indicators'])}

- Context: {SEVEN_CS_FRAMEWORK['context']['description']}
  Indicators: {', '.join(SEVEN_CS_FRAMEWORK['context']['indicators'])}

- Contribution: {SEVEN_CS_FRAMEWORK['contribution']['description']}
  Indicators: {', '.join(SEVEN_CS_FRAMEWORK['contribution']['indicators'])}

- Constructive: {SEVEN_CS_FRAMEWORK['constructive']['description']}
  Indicators: {', '.join(SEVEN_CS_FRAMEWORK['constructive']['indicators'])}

Discussion Segment:
{window_text}

Return a JSON object with a "segments" array containing ONLY the dimensions that are clearly present:
{{
    "segments": [
        {{
            "dimension": "climate|communication|compatibility|conflict|context|contribution|constructive",
            "quote": "exact quote from transcript",
            "explanation": "why this shows the dimension",
            "confidence": 0.0-1.0,
            "speaker": "speaker identifier if available"
        }}
    ]
}}

If no clear evidence of any dimension is found, return {{"segments": []}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in qualitative analysis of collaborative learning, specializing in the 7-dimension framework. Identify clear evidence of each dimension when present."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000
        )

        # Parse response
        result = json.loads(response.choices[0].message.content)
        logging.info(f"LLM response for coding window: {json.dumps(result)[:500]}")

        # Handle both array and object responses
        if isinstance(result, dict) and 'codings' in result:
            codings = result['codings']
        elif isinstance(result, dict) and 'segments' in result:
            codings = result['segments']
        elif isinstance(result, dict) and 'results' in result:
            codings = result['results']
        elif isinstance(result, dict) and 'dimensions' in result:
            codings = result['dimensions']
        elif isinstance(result, list):
            codings = result
        else:
            # If result is a dict with dimension names as keys
            codings = []
            for key, value in result.items():
                if key in ['climate', 'communication', 'compatibility', 'conflict', 'context', 'contribution', 'constructive']:
                    if isinstance(value, dict) and value.get('quote'):
                        value['dimension'] = key
                        codings.append(value)

        logging.info(f"Extracted {len(codings)} codings from LLM response")

        # Create coded segment objects
        for coding in codings:
            if not isinstance(coding, dict):
                continue

            # Find the transcript that contains this quote (if possible)
            transcript_id = window_transcripts[0].id if window_transcripts else None

            segment = SevenCsCodedSegment(
                analysis_id=analysis_id,
                transcript_id=transcript_id,
                dimension=coding.get('dimension', '').lower(),
                start_time=window_start,
                end_time=window_end,
                text_snippet=coding.get('quote', '')[:500],  # Limit to 500 chars
                speaker_tag=coding.get('speaker'),
                coding_reason=coding.get('explanation', ''),
                confidence=float(coding.get('confidence', 0.7))
            )
            db.session.add(segment)
            coded_segments.append(segment)

        db.session.commit()

    except Exception as e:
        logging.error(f"Error coding window with 7 Cs: {e}")

    return coded_segments

def generate_overall_seven_cs_summary(full_transcript_text, coded_segments):
    """
    Generate overall scores and explanations for each of the 7 Cs.

    Args:
        full_transcript_text: Complete transcript text
        coded_segments: List of coded segments

    Returns:
        Dict with scores and explanations for each dimension
    """
    # Count segments per dimension
    dimension_counts = {}
    for segment in coded_segments:
        if segment.dimension not in dimension_counts:
            dimension_counts[segment.dimension] = 0
        dimension_counts[segment.dimension] += 1

    # Build prompt with context
    prompt = f"""Based on this full discussion transcript and the coded segments analysis, provide a comprehensive assessment of collaboration quality across the 7-dimension framework.

Coded segments found:
{json.dumps(dimension_counts, indent=2)}

Full Discussion Transcript:
{full_transcript_text[:8000]}  # Limit to avoid token limits

For EACH of the 7 dimensions, provide:
1. A score from 0-100
2. A detailed explanation (3-4 sentences) of the score
3. 2-3 key evidence points from the transcript

The 7 dimensions ("7 Cs") to analyze:
- Climate: {SEVEN_CS_FRAMEWORK['climate']['scoring_criteria']}
- Communication: {SEVEN_CS_FRAMEWORK['communication']['scoring_criteria']}
- Compatibility: {SEVEN_CS_FRAMEWORK['compatibility']['scoring_criteria']}
- Conflict: {SEVEN_CS_FRAMEWORK['conflict']['scoring_criteria']}
- Context: {SEVEN_CS_FRAMEWORK['context']['scoring_criteria']}
- Contribution: {SEVEN_CS_FRAMEWORK['contribution']['scoring_criteria']}
- Constructive: {SEVEN_CS_FRAMEWORK['constructive']['scoring_criteria']}

Return a JSON object with this structure:
{{
    "climate": {{
        "score": 0-100,
        "explanation": "detailed explanation",
        "evidence": ["evidence point 1", "evidence point 2", "evidence point 3"],
        "keywords_found": ["list", "of", "relevant", "keywords"]
    }},
    // ... repeat for all 7 dimensions
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in collaborative learning assessment using the 7-dimension framework. Provide nuanced, evidence-based evaluations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2000
        )

        result = json.loads(response.choices[0].message.content)

        # Ensure all 7 dimensions are present
        for dimension in SEVEN_CS_FRAMEWORK.keys():
            if dimension not in result:
                result[dimension] = {
                    "score": 50,
                    "explanation": "Insufficient data to assess this dimension",
                    "evidence": [],
                    "keywords_found": []
                }

        return result

    except Exception as e:
        logging.error(f"Error generating 7 Cs summary: {e}")
        # Return default scores if error
        return {
            dimension: {
                "score": 50,
                "explanation": "Analysis could not be completed",
                "evidence": [],
                "keywords_found": []
            }
            for dimension in SEVEN_CS_FRAMEWORK.keys()
        }

def prepare_full_transcript(transcripts):
    """
    Prepare the full transcript text for overall analysis.

    Args:
        transcripts: List of all transcript objects

    Returns:
        Formatted string of all transcripts
    """
    sorted_transcripts = sorted(transcripts, key=lambda t: t.start_time)

    # Get speaker aliases
    speaker_aliases = get_speaker_aliases(sorted_transcripts)

    transcript_lines = []

    for t in sorted_transcripts:
        # Determine speaker name
        if t.speaker_id and t.speaker_id in speaker_aliases:
            speaker = speaker_aliases[t.speaker_id]
        elif t.speaker_tag:
            speaker = f"Speaker {t.speaker_tag}"
        else:
            speaker = "Unknown"

        time_min, time_sec = divmod(t.start_time, 60)
        transcript_lines.append(f"[{speaker} at {time_min}:{time_sec:02d}]: {t.transcript}")

    return "\n".join(transcript_lines)

def estimate_tokens(text):
    """
    Rough estimation of tokens in text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Rough estimate: ~4 characters per token
    return len(text) // 4

def update_seven_cs_analysis(session_device_id):
    """
    Update/re-run 7 Cs analysis for a session device (manual trigger).

    Args:
        session_device_id: ID of the session device to analyze

    Returns:
        SevenCsAnalysis object if successful, None if failed
    """
    # Check if there's an existing analysis
    existing = db.session.query(SevenCsAnalysis).filter_by(
        session_device_id=session_device_id
    ).order_by(SevenCsAnalysis.created_at.desc()).first()

    if existing and existing.analysis_status == 'processing':
        logging.info(f"Analysis already in progress for session_device {session_device_id}")
        return existing

    # Run new analysis
    return analyze_session_seven_cs(session_device_id)