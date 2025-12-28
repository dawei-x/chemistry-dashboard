"""
Reason and Act Node for BLINC Agent V3

The core reasoning loop that decides what action to take.
Uses GPT-4o with tool descriptions to naturally select tools.
NO keyword matching - trusts the model's understanding.
"""

import json
import logging
from typing import Dict, Any

from openai import OpenAI

logger = logging.getLogger(__name__)


def reason_and_act(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main reasoning node that decides what to do next.

    This node:
    1. Analyzes the query and context
    2. Considers previous retrieval results
    3. Decides whether to use a tool, synthesize, or clarify
    4. Returns the decision with tool parameters

    The key insight: We use GPT-4o (not mini) and trust its reasoning
    with well-written tool descriptions. No keyword matching.

    Args:
        state: Current agent state

    Returns:
        Updated state with action decision
    """
    query = state.get('current_query', state.get('original_query', ''))
    iteration = state.get('iteration_count', 0) + 1

    logger.info(f"Reasoning iteration {iteration}: '{query}'")

    # Check iteration limit
    if iteration > state.get('max_iterations', 8):
        logger.warning("Max iterations reached, forcing synthesis")
        return {
            'iteration_count': iteration,
            'next_action': 'synthesize'
        }

    # Build context for the model
    context = _build_context(state)
    previous_results = state.get('retrieval_results', [])

    # Check if we have enough information to synthesize
    if _should_synthesize(previous_results, iteration):
        logger.info("Have enough relevant results, proceeding to synthesis")
        return {
            'iteration_count': iteration,
            'next_action': 'synthesize'
        }

    # Call GPT-4o for reasoning
    try:
        client = OpenAI()

        system_prompt = _get_system_prompt()
        user_prompt = _get_user_prompt(query, context, previous_results)

        response = client.chat.completions.create(
            model="gpt-4o",  # Use the powerful model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent reasoning
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        decision = json.loads(response.choices[0].message.content)
        logger.info(f"Reasoning decision: {decision.get('action')} - {decision.get('thought', '')[:100]}")

        # Record thought if present
        thought_history = state.get('thought_history', []).copy()
        if decision.get('thought'):
            thought_history.append(decision['thought'])

        # Handle the decision
        action = decision.get('action', 'think')

        if action == 'synthesize':
            return {
                'iteration_count': iteration,
                'next_action': 'synthesize',
                'thought_history': thought_history,
                'current_thought': decision.get('thought')
            }

        elif action == 'clarify':
            return {
                'iteration_count': iteration,
                'next_action': 'clarify',
                'current_tool': 'clarify',
                'current_tool_input': decision.get('action_input', {}),
                'thought_history': thought_history,
                'current_thought': decision.get('thought')
            }

        elif action == 'think':
            # Just thinking, continue the loop
            return {
                'iteration_count': iteration,
                'next_action': 'continue',
                'thought_history': thought_history,
                'current_thought': decision.get('thought') or decision.get('action_input', {}).get('reasoning', '')
            }

        else:
            # Tool call
            return {
                'iteration_count': iteration,
                'next_action': 'execute_tool',
                'current_tool': action,
                'current_tool_input': decision.get('action_input', {}),
                'thought_history': thought_history,
                'current_thought': decision.get('thought')
            }

    except Exception as e:
        logger.error(f"Reasoning error: {e}")

        # On error, try a simple search if we haven't tried anything
        if not previous_results:
            return {
                'iteration_count': iteration,
                'next_action': 'execute_tool',
                'current_tool': 'search_transcripts',
                'current_tool_input': {'query': query, 'limit': 10},
                'error': str(e)
            }
        else:
            # If we have some results, synthesize
            return {
                'iteration_count': iteration,
                'next_action': 'synthesize',
                'error': str(e)
            }


def _should_synthesize(results: list, iteration: int) -> bool:
    """Determine if we have enough information to synthesize."""
    if not results:
        return False

    # Count relevant results
    relevant_count = sum(1 for r in results if r.get('is_relevant', False))

    # Synthesize if we have good results or we've tried multiple times
    if relevant_count >= 2:
        return True
    if relevant_count >= 1 and iteration >= 3:
        return True

    return False


def _build_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build context object for the prompt."""
    return {
        'current_session_focus': state.get('current_session_focus'),
        'previous_session_focus': state.get('previous_session_focus'),
        'session_history': state.get('session_history', []),
        'compared_sessions': state.get('compared_sessions', []),
        'current_speaker_focus': state.get('current_speaker_focus')
    }


def _get_system_prompt() -> str:
    """Get the system prompt for reasoning."""
    from ..prompts.tool_descriptions import get_tools_prompt
    from ..prompts.reasoning import REASONING_SYSTEM_PROMPT

    tools_prompt = get_tools_prompt()

    return f"""{REASONING_SYSTEM_PROMPT}

{tools_prompt}

## Decision Making

Based on the query and context, decide your next action:

1. **Use a tool** - If you need information to answer the query
2. **synthesize** - If you have enough relevant information from previous results
3. **think** - If you need to reason through a complex problem
4. **clarify** - ONLY if the query is genuinely ambiguous (prefer searching)

## Response Format

Always respond with a JSON object:
{{
    "thought": "Brief reasoning about what to do",
    "action": "tool_name OR synthesize OR think OR clarify",
    "action_input": {{...parameters...}} OR null
}}

## Key Principles

- Trust the tool descriptions - they explain WHEN to use each tool
- For "best" or "compare" queries, use compare_sessions with session IDs [18,19,20,21,22,23,24,25]
- Consider the conversation context for references like "it" or "that session"
- Be efficient - don't call unnecessary tools
"""


def _get_user_prompt(query: str, context: dict, previous_results: list) -> str:
    """Build the user prompt with current state."""

    # Format context
    context_lines = []
    if context.get('current_session_focus'):
        context_lines.append(f"- Currently focused on: Session {context['current_session_focus']}")
    if context.get('previous_session_focus'):
        context_lines.append(f"- Previous session: Session {context['previous_session_focus']}")
    if context.get('compared_sessions'):
        context_lines.append(f"- Comparing: Sessions {context['compared_sessions']}")
    if context.get('current_speaker_focus'):
        context_lines.append(f"- Speaker focus: {context['current_speaker_focus']}")

    context_str = "\n".join(context_lines) if context_lines else "No prior context"

    # Format previous results
    results_lines = []
    for result in previous_results[-3:]:  # Last 3
        tool = result.get('tool_name', 'unknown')
        count = result.get('result_count', 0)
        relevant = "relevant" if result.get('is_relevant', False) else "not relevant"
        query_used = result.get('query_used', '')[:50]
        results_lines.append(f"- {tool}('{query_used}'): {count} results ({relevant})")

    results_str = "\n".join(results_lines) if results_lines else "No results yet"

    return f"""## User Query
{query}

## Conversation Context
{context_str}

## Previous Results This Turn
{results_str}

## Your Task
Decide what to do next. If you have enough relevant information, synthesize an answer.
Otherwise, call the appropriate tool to get more information.

Remember: Tool descriptions explain WHEN to use each tool. Trust them."""
