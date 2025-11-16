import logging
import json
import database
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def generate_llm_scores_for_session_device(session_device_id):
    """
    Generate and store LLM scores for a session device.
    This is called as a background job when a session ends.
    
    Args:
        session_device_id: ID of the session device to score
    
    Returns:
        LLMMetrics object if successful, None if failed
    """
    if not client:
        logging.error(f"Cannot generate LLM scores - OpenAI client not initialized")
        return None
    
    try:
        # Get transcripts for this session device (last 100)
        # Get all transcripts, then slice to last 100
        transcripts = database.get_transcripts(session_device_id=session_device_id)
        transcripts = transcripts[-100:] if len(transcripts) > 100 else transcripts
        
        if not transcripts:
            logging.info(f"No transcripts found for session_device {session_device_id}")
            return None
        
        # Prepare transcript text for LLM
        transcript_text = prepare_transcript_text(transcripts)
        
        # Get LLM scores
        scores_result = get_llm_scores(transcript_text)
        
        if not scores_result:
            logging.error(f"Failed to get LLM scores for session_device {session_device_id}")
            return None
        
        # Extract scores and explanations
        scores_dict = {
            'emotional_tone': scores_result.get('emotional_tone', {}).get('score'),
            'analytic_thinking': scores_result.get('analytic_thinking', {}).get('score'),
            'clout': scores_result.get('clout', {}).get('score'),
            'authenticity': scores_result.get('authenticity', {}).get('score'),
            'certainty': scores_result.get('certainty', {}).get('score')
        }
        
        explanations_dict = {
            'emotional_tone': scores_result.get('emotional_tone', {}).get('explanation', ''),
            'analytic_thinking': scores_result.get('analytic_thinking', {}).get('explanation', ''),
            'clout': scores_result.get('clout', {}).get('explanation', ''),
            'authenticity': scores_result.get('authenticity', {}).get('explanation', ''),
            'certainty': scores_result.get('certainty', {}).get('explanation', '')
        }
        
        # Store in database
        llm_metrics = database.add_llm_metrics(
            session_device_id=session_device_id,
            scores_dict=scores_dict,
            explanations_dict=explanations_dict,
            transcript_count=len(transcripts),
            llm_model='gpt-4o'
        )
        
        logging.info(f"Successfully generated LLM scores for session_device {session_device_id}")
        return llm_metrics
        
    except Exception as e:
        logging.error(f"Error generating LLM scores for session_device {session_device_id}: {str(e)}")
        return None

def prepare_transcript_text(transcripts):
    """
    Prepare transcript text for LLM analysis.
    
    Args:
        transcripts: List of transcript objects
    
    Returns:
        Formatted string of transcripts
    """
    transcript_lines = []
    for t in transcripts:
        speaker = f"Speaker {t.speaker_id}" if t.speaker_id else f"Speaker {t.speaker_tag}"
        time_min, time_sec = divmod(t.start_time, 60)
        transcript_lines.append(f"[{speaker} at {time_min}:{time_sec:02d}]: {t.transcript}")
    
    return "\n".join(transcript_lines)

def get_llm_scores(transcript_text):
    """
    Call OpenAI API to get scores for the five metrics.
    
    Args:
        transcript_text: Formatted transcript text
    
    Returns:
        Dict with scores and explanations for each metric
    """
    prompt = f"""Analyze this discussion transcript and score it on five key communication metrics. 

Discussion Transcript:
{transcript_text}

Score each metric from 0-100 and provide a 2-3 sentence explanation for each score.
Consider the context, nuance, and overall patterns in the conversation.

IMPORTANT: Return ONLY a valid JSON object with this exact structure:
{{
    "emotional_tone": {{
        "score": [0-100],
        "explanation": "Brief explanation of why this score was given, with specific examples from the transcript"
    }},
    "analytic_thinking": {{
        "score": [0-100],
        "explanation": "Brief explanation focusing on logical structure, evidence usage, and systematic reasoning"
    }},
    "clout": {{
        "score": [0-100],
        "explanation": "Brief explanation about confidence, leadership, and authority displayed in discussion"
    }},
    "authenticity": {{
        "score": [0-100],
        "explanation": "Brief explanation about genuineness, honesty, and personal disclosure"
    }},
    "certainty": {{
        "score": [0-100],
        "explanation": "Brief explanation about confidence in statements and decisiveness"
    }}
}}

Scoring Guidelines:
- Emotional Tone: 0=very negative, 50=neutral, 100=very positive
- Analytic Thinking: 0=narrative/informal, 50=balanced, 100=highly analytical/formal
- Clout: 0=tentative/submissive, 50=balanced, 100=authoritative/dominant
- Authenticity: 0=guarded/formal, 50=balanced, 100=very personal/revealing
- Certainty: 0=uncertain/hesitant, 50=balanced, 100=very certain/decisive"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert communication analyst specializing in discourse analysis and collaboration assessment. You provide precise, contextual evaluations of communication patterns."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=800
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None

def update_llm_scores_for_session_device(session_device_id, time_range=None):
    """
    Update LLM scores for a session device (called when user requests refresh).
    Can optionally use a specific time range.
    
    Args:
        session_device_id: ID of the session device to score
        time_range: Optional dict with 'start' and 'end' times in seconds
    
    Returns:
        LLMMetrics object if successful, None if failed
    """
    # For now, just regenerate the full session scores
    return generate_llm_scores_for_session_device(session_device_id)