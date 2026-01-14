"""
Agent Core Module for Agent V6.

Implements the ReAct loop (from V4) with V6's embedded intelligence.
Simple, fast, and effective.

Architecture:
1. Analyze query (extract entities, steering, mode, constructs)
2. Build system prompt (embed V3's intelligence)
3. Run ReAct loop (LLM decides when to call tools)
4. Return result with metadata
"""

import json
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime
from openai import OpenAI

from .query_analysis import analyze_query, QueryAnalysis
from .prompt_builder import build_system_prompt
from .tools import (
    filter_tools_by_steering,
    execute_tool,
    get_all_tool_schemas,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "gpt-4o"
MAX_TURNS = 10  # Safety limit on ReAct loop iterations


# =============================================================================
# CONVERSATION MANAGER
# =============================================================================

class ConversationManager:
    """Manages conversation state across turns."""

    def __init__(self):
        self._contexts: Dict[str, Dict] = {}

    def get_context(self, conversation_id: str) -> Dict:
        """Get context for a conversation."""
        if conversation_id not in self._contexts:
            self._contexts[conversation_id] = {
                'history': [],
                'session_focus': None,
                'speaker_focus': None,
                'last_intent': None,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'title': 'New Conversation',
            }
        return self._contexts[conversation_id]

    def update_context(self, conversation_id: str, result: Dict):
        """Update context based on agent result."""
        ctx = self.get_context(conversation_id)

        # Extract session/speaker focus from tool results
        for tool_result in result.get('tool_results', []):
            tr = tool_result.get('result', {})
            if 'session_id' in tr and tr['session_id']:
                ctx['session_focus'] = tr['session_id']
            if 'speaker' in tr and tr['speaker']:
                ctx['speaker_focus'] = tr['speaker']

        # Store last intent
        ctx['last_intent'] = result.get('query_intent')
        ctx['updated_at'] = datetime.now().isoformat()

    def add_exchange(self, conversation_id: str, query: str, response: str):
        """Add a query/response exchange to history."""
        ctx = self.get_context(conversation_id)
        ctx['history'].append({'role': 'user', 'content': query})
        ctx['history'].append({'role': 'assistant', 'content': response})

        # Generate title from first query if not set
        if ctx.get('title') == 'New Conversation' and query:
            ctx['title'] = query[:50] + ('...' if len(query) > 50 else '')

        ctx['updated_at'] = datetime.now().isoformat()

    def create_conversation(self, conversation_id: str, title: str = 'New Conversation'):
        """Explicitly create a new conversation."""
        self._contexts[conversation_id] = {
            'history': [],
            'session_focus': None,
            'speaker_focus': None,
            'last_intent': None,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'title': title,
        }

    def list_conversations(self) -> List[Dict]:
        """List all conversations sorted by last activity."""
        conversations = []
        for conv_id, ctx in self._contexts.items():
            if ctx.get('history'):  # Only include conversations with history
                conversations.append({
                    'conversation_id': conv_id,
                    'title': ctx.get('title', 'Conversation'),
                    'created_at': ctx.get('created_at'),
                    'updated_at': ctx.get('updated_at'),
                    'message_count': len(ctx.get('history', [])),
                    'session_focus': ctx.get('session_focus'),
                    'speaker_focus': ctx.get('speaker_focus'),
                })

        # Sort by updated_at descending
        conversations.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return conversations

    def clear(self, conversation_id: str = None):
        """Clear conversation context."""
        if conversation_id:
            self._contexts.pop(conversation_id, None)
        else:
            self._contexts.clear()


# Global conversation manager
conversation_manager = ConversationManager()


# =============================================================================
# AGENT CORE
# =============================================================================

def run_agent(
    query: str,
    conversation_id: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None,
    mode: str = "enhanced",
    session_context: Optional[Dict] = None,
    model: Optional[str] = None,
    api_params: Optional[Dict] = None,
) -> Dict:
    """
    Run the V6 agent on a query.

    Args:
        query: The user's question
        conversation_id: Optional conversation ID for context
        conversation_history: Optional list of previous messages
        mode: "enhanced" (all tools) or "baseline" (transcript only)
        session_context: Optional context from conversation manager
        model: Optional model override
        api_params: Optional API parameters (prefer_representations, exclude_representations, mode)

    Returns:
        {
            "answer": str,
            "tools_used": List[str],
            "tool_results": List[Dict],
            "turn_count": int,
            "query_intent": Dict,
            "model": str,
            "success": bool,
            "error": Optional[str],
        }
    """
    start_time = time.time()

    try:
        # Initialize OpenAI client
        client = OpenAI()
        model = model or DEFAULT_MODEL

        # Phase 1: Query Analysis
        logger.info(f"[V6] Phase 1: Analyzing query")
        api_params = api_params or {}

        # Build conversation context for analysis
        conv_context = session_context or {}
        if conversation_id:
            stored_ctx = conversation_manager.get_context(conversation_id)
            conv_context = {
                'session_focus': stored_ctx.get('session_focus'),
                'speaker_focus': stored_ctx.get('speaker_focus'),
            }

        analysis = analyze_query(query, api_params, conv_context)

        # For baseline mode, limit steering
        if mode == "baseline":
            analysis.prefer_representations = ['transcript']
            analysis.exclude_representations = ['concept_map', 'collaboration']

        # Phase 2: Prompt Construction
        logger.info(f"[V6] Phase 2: Building system prompt")
        system_prompt = build_system_prompt(analysis)

        # Phase 3: Tool Selection
        logger.info(f"[V6] Phase 3: Selecting tools")
        tools = filter_tools_by_steering(
            prefer=analysis.prefer_representations,
            exclude=analysis.exclude_representations
        )
        tool_names = [t['name'] for t in tools]
        logger.info(f"[V6] Available tools: {tool_names}")

        # Build messages
        messages = []

        # Add conversation history if available
        if conversation_history:
            messages.extend(conversation_history)

        # Add current query
        messages.append({"role": "user", "content": query})

        # Phase 4: ReAct Loop
        logger.info(f"[V6] Phase 4: Starting ReAct loop")
        tools_used = []
        tool_results = []
        turn_count = 0

        while turn_count < MAX_TURNS:
            turn_count += 1
            logger.info(f"[V6] Turn {turn_count}")

            # Call LLM
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                tools=[{"type": "function", "function": t} for t in tools] if tools else None,
                tool_choice="auto" if tools else None,
            )

            message = response.choices[0].message

            # Check for tool calls
            if message.tool_calls:
                # Execute each tool call
                messages.append(message)

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)

                    logger.info(f"[V6] Tool call: {tool_name}({tool_input})")
                    tools_used.append(tool_name)

                    # Execute tool
                    result = execute_tool(tool_name, tool_input)
                    tool_results.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result": result
                    })

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str)
                    })
            else:
                # No tool calls - we have the final answer
                answer = message.content or ""
                break
        else:
            # Hit max turns
            answer = message.content if message else "I was unable to complete the analysis within the allowed iterations."

        elapsed = time.time() - start_time
        logger.info(f"[V6] Completed in {elapsed:.2f}s with {turn_count} turns, {len(tools_used)} tool calls")

        return {
            "answer": answer,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "turn_count": turn_count,
            "query_intent": {
                "session_ids": analysis.session_ids,
                "session_names": analysis.session_names,
                "speaker_names": analysis.speaker_names,
                "mode": analysis.mode,
                "prefer_representations": analysis.prefer_representations,
                "exclude_representations": analysis.exclude_representations,
                "constructs": analysis.constructs,
            },
            "model": model,
            "success": True,
            "error": None,
            "elapsed_time": elapsed,
        }

    except Exception as e:
        logger.error(f"[V6] Agent error: {e}", exc_info=True)
        return {
            "answer": f"An error occurred: {str(e)}",
            "tools_used": [],
            "tool_results": [],
            "turn_count": 0,
            "query_intent": None,
            "model": model or DEFAULT_MODEL,
            "success": False,
            "error": str(e),
        }
