"""
Synthesize Node for BLINC Agent V3

Generates the final answer from retrieved information.
"""

import json
import logging
from typing import Dict, Any, List

from openai import OpenAI

logger = logging.getLogger(__name__)


def synthesize(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize the final answer from retrieved information.

    This node:
    1. Gathers all relevant retrieval results
    2. Builds context for synthesis
    3. Generates a comprehensive answer with citations

    Args:
        state: Current agent state with retrieval_results

    Returns:
        Updated state with final_answer and citations
    """
    query = state.get('original_query', '')
    results = state.get('retrieval_results', [])

    logger.info(f"Synthesizing answer for: '{query}' with {len(results)} result sets")

    # Build context
    context = {
        'current_session_focus': state.get('current_session_focus'),
        'compared_sessions': state.get('compared_sessions', []),
        'current_speaker_focus': state.get('current_speaker_focus')
    }

    try:
        # Generate answer using GPT-4o
        client = OpenAI()

        system_prompt = _get_synthesis_system_prompt()
        user_prompt = _format_synthesis_prompt(query, results, context)

        response = client.chat.completions.create(
            model="gpt-4o",  # Use powerful model for synthesis
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        answer = response.choices[0].message.content

        # Extract citations from results
        citations = _extract_citations(results)

        logger.info(f"Generated answer with {len(citations)} citations")

        return {
            'final_answer': answer,
            'citations': citations,
            'next_action': 'reflect'
        }

    except Exception as e:
        import traceback
        logger.error(f"Synthesis error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Generate a basic answer on error
        basic_answer = _generate_fallback_answer(query, results)

        return {
            'final_answer': basic_answer,
            'citations': [],
            'next_action': 'reflect',
            'error': str(e)
        }


def _format_comparison_data(comparison: dict) -> str:
    """Format compare_sessions output into readable text."""
    lines = []

    sessions = comparison.get('sessions_compared', [])
    summary = comparison.get('summary', {})

    lines.append(f"**Sessions Compared:** {sessions}")
    lines.append("")

    # Format collaboration scores as a ranked list
    scores = summary.get('collaboration_scores', {})
    if scores:
        lines.append("**Collaboration Scores (7C Overall):**")
        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (session_id, score) in enumerate(sorted_scores, 1):
            lines.append(f"  {rank}. Session {session_id}: {score}/100")
        lines.append("")

    # Format speaker counts
    speaker_counts = summary.get('speaker_counts', {})
    if speaker_counts:
        lines.append("**Speaker Counts:**")
        for session_id, count in speaker_counts.items():
            lines.append(f"  - Session {session_id}: {count} speakers")
        lines.append("")

    # Format themes
    themes = summary.get('themes', {})
    if themes:
        lines.append("**Main Themes:**")
        for session_id, theme_list in themes.items():
            if theme_list:
                lines.append(f"  - Session {session_id}: {', '.join(theme_list[:3])}")

    return "\n".join(lines)


def _format_collaboration_data(analysis: dict) -> str:
    """Format 7C collaboration analysis into readable text."""
    lines = []

    session_id = analysis.get('session_device_id', '')
    overall = analysis.get('overall_score', 0)

    lines.append(f"**Session {session_id} Collaboration Analysis**")
    lines.append(f"Overall Score: {overall}/100")
    lines.append("")

    dimensions = analysis.get('dimensions', {})
    if dimensions:
        lines.append("**7C Dimension Scores:**")
        for dim_name, dim_data in dimensions.items():
            score = dim_data.get('score', 0)
            explanation = dim_data.get('explanation', '')[:200]
            lines.append(f"  - {dim_name.title()}: {score}/100")
            if explanation:
                lines.append(f"    {explanation}")

    return "\n".join(lines)


def _get_synthesis_system_prompt() -> str:
    """Get the system prompt for synthesis."""
    return """You are synthesizing an answer about collaborative discussions.

Your response should:
1. Directly address the user's question
2. Reference specific evidence from the retrieved information
3. Be clear and well-organized
4. Acknowledge any limitations in the available data

Format guidelines:
- Use markdown for formatting when helpful
- Keep responses focused (2-4 paragraphs for most queries)
- Use bullet points for lists
- Cite sessions and speakers when relevant

Do NOT:
- Make up information not in the retrieved results
- Include raw JSON or technical details
- Be unnecessarily verbose
"""


def _format_synthesis_prompt(query: str, results: list, context: dict) -> str:
    """Format the synthesis prompt with all information."""

    # Format retrieved information
    info_sections = []

    for result in results:
        if not result.get('is_relevant', True):
            continue  # Skip irrelevant results

        tool_name = result.get('tool_name', 'Search')
        items = result.get('results', [])

        if not items:
            continue

        lines = [f"### From {tool_name}"]

        for item in items[:5]:  # Top 5 per source
            if isinstance(item, dict):
                # Handle compare_sessions structured output
                if 'sessions_compared' in item and 'summary' in item:
                    lines.append(_format_comparison_data(item))
                    continue

                # Handle 7C collaboration analysis
                if 'dimensions' in item and 'overall_score' in item:
                    lines.append(_format_collaboration_data(item))
                    continue

                # Standard item handling
                session = item.get('session_device_id', item.get('session_id', ''))
                speaker = item.get('speaker', item.get('speaker_alias', ''))
                text = item.get('text', item.get('content', item.get('summary', '')))

                if session:
                    header = f"**Session {session}**"
                    if speaker:
                        header += f" ({speaker})"
                    lines.append(header)

                if text and isinstance(text, str):
                    lines.append(text[:600])
                elif text and isinstance(text, dict):
                    # Handle structured data (like comparison summaries)
                    for key, value in text.items():
                        if isinstance(value, (str, int, float)):
                            lines.append(f"- {key}: {value}")
                        elif isinstance(value, dict):
                            lines.append(f"- {key}: {list(value.keys())}")
                else:
                    # Handle structured data (like 7C analysis)
                    for key, value in item.items():
                        if key not in ['session_device_id', 'session_id', 'distance', 'relevance']:
                            if isinstance(value, str):
                                lines.append(f"- {key}: {value[:300]}")
                            elif isinstance(value, (int, float)):
                                lines.append(f"- {key}: {value}")
                            elif isinstance(value, list):
                                lines.append(f"- {key}: {value[:5]}")

                lines.append("")
            else:
                lines.append(str(item)[:600])
                lines.append("")

        info_sections.append("\n".join(lines))

    info_str = "\n\n".join(info_sections) if info_sections else "No specific information was retrieved."

    # Format context
    context_lines = []
    if context.get('current_session_focus'):
        context_lines.append(f"User is focused on Session {context['current_session_focus']}")
    if context.get('compared_sessions'):
        context_lines.append(f"Comparing Sessions {context['compared_sessions']}")
    if context.get('current_speaker_focus'):
        context_lines.append(f"Focusing on speaker: {context['current_speaker_focus']}")

    context_str = "\n".join(context_lines) if context_lines else "General query"

    return f"""## User Query
{query}

## Context
{context_str}

## Retrieved Information
{info_str}

## Task
Generate a clear, helpful answer based on the retrieved information.
Cite specific sessions, speakers, or timestamps when available."""


def _extract_citations(results: list) -> List[Dict[str, Any]]:
    """Extract citations from results."""
    citations = []

    for result in results:
        tool_name = result.get('tool_name', '')
        items = result.get('results', [])

        for item in items[:3]:  # Top 3 per source
            if isinstance(item, dict):
                citation = {
                    'source': tool_name,
                    'session_id': item.get('session_device_id', item.get('session_id')),
                    'speaker': item.get('speaker', item.get('speaker_alias', '')),
                    'text': (item.get('text', item.get('content', ''))[:200]
                             if item.get('text') or item.get('content') else None),
                    'timestamp': item.get('start_time')
                }
                # Only add if we have useful info
                if citation['session_id'] or citation['text']:
                    citations.append(citation)

    return citations[:10]  # Max 10 citations


def _generate_fallback_answer(query: str, results: list) -> str:
    """Generate a basic answer when LLM fails."""
    if not results:
        return f"I wasn't able to find specific information about '{query}' in the discussion database. Could you try rephrasing your question or asking about a specific session?"

    # Try to extract some info
    all_items = []
    for result in results:
        all_items.extend(result.get('results', []))

    if not all_items:
        return f"I found some results related to '{query}' but couldn't extract specific information. Please try a more specific query."

    # Build a simple summary
    sessions_mentioned = set()
    for item in all_items:
        sid = item.get('session_device_id', item.get('session_id'))
        if sid:
            sessions_mentioned.add(sid)

    return f"Based on my search for '{query}', I found information in sessions {list(sessions_mentioned)[:5]}. Please ask a more specific question to get detailed insights."
