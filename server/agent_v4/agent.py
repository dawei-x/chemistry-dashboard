"""
Core Agent for V4: High-Agency ReAct-Style Agent.

Design principles:
1. Let the LLM decide what tools to use
2. Let the LLM synthesize naturally
3. Simple loop: think -> act -> observe -> repeat
4. No pattern matching, no forced pipelines

Direct implementation using OpenAI SDK - simple, correct, no unnecessary abstractions.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional

from openai import OpenAI

from .prompts import get_system_prompt
from .tools import get_tool_schemas, TOOL_FUNCTIONS, BASELINE_TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

# Configuration
MAX_TURNS = 10  # Maximum tool-use turns before forcing a response
DEFAULT_MODEL = os.getenv("LLM_REASONING_MODEL", "gpt-4o")

# Initialize OpenAI client
_client = None

def _get_client():
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI()  # Uses OPENAI_API_KEY from environment
    return _client


def _convert_tools_to_openai_format(tool_schemas: List[Dict]) -> List[Dict]:
    """Convert our tool schemas to OpenAI function calling format."""
    openai_tools = []
    for tool in tool_schemas:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
        })
    return openai_tools


def run_agent(
    query: str,
    conversation_id: str = None,
    conversation_history: List[Dict] = None,
    mode: str = "enhanced",
    session_context: Dict = None,
    model: str = None
) -> Dict[str, Any]:
    """
    Run the agent to answer a query.

    This is a simple ReAct-style agent:
    1. Send query + tools to LLM
    2. If LLM calls tools, execute them and continue
    3. When LLM responds with text, return it

    Args:
        query: User's question
        conversation_id: Optional conversation ID for multi-turn
        conversation_history: Previous messages in conversation
        mode: "enhanced" (all tools) or "baseline" (transcript only)
        session_context: Optional context (e.g., current session focus)
        model: Optional model override (default: gpt-4o from environment)

    Returns:
        Agent response with answer, tools used, and metadata
    """
    model = model or DEFAULT_MODEL
    client = _get_client()

    logger.info(f"[Agent V4] Query: '{query}' (mode={mode}, model={model})")

    # Build messages with system prompt
    messages = [
        {"role": "system", "content": get_system_prompt(mode)}
    ]

    # Add conversation history if provided (for multi-turn)
    if conversation_history:
        messages.extend(conversation_history)

    # Add context hint if we have a session focus
    user_message = query
    if session_context and session_context.get("session_focus"):
        focus = session_context["session_focus"]
        user_message = f"[Context: Currently discussing session {focus}]\n\n{query}"

    messages.append({"role": "user", "content": user_message})

    # Get tools for this mode
    tool_schemas = get_tool_schemas(mode)
    tools = _convert_tools_to_openai_format(tool_schemas)
    tool_functions = TOOL_FUNCTIONS if mode == "enhanced" else BASELINE_TOOL_FUNCTIONS

    # Track what happens
    tools_used = []
    tool_results = []
    turn_count = 0

    # Simple ReAct loop
    while turn_count < MAX_TURNS:
        turn_count += 1
        logger.info(f"[Agent V4] Turn {turn_count}")

        # Call OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            max_tokens=4096
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        logger.info(f"[Agent V4] Finish reason: {finish_reason}")

        # No tool calls = we're done
        if not message.tool_calls:
            text_content = message.content or ""
            logger.info(f"[Agent V4] Final response ({len(text_content)} chars)")

            return {
                "success": True,
                "answer": text_content,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "turn_count": turn_count,
                "mode": mode,
                "model": model,
                "conversation_id": conversation_id,
                "session_focus": _extract_session_focus(text_content, session_context)
            }

        # LLM wants to use tools
        # Add the assistant message (with tool calls) to history - use native format
        messages.append(message)

        # Execute each tool call and add results
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments

            try:
                tool_input = json.loads(tool_args_str)
            except json.JSONDecodeError:
                tool_input = {}

            logger.info(f"[Agent V4] Tool call: {tool_name}({json.dumps(tool_input)[:100]}...)")

            # Execute the tool
            if tool_name in tool_functions:
                try:
                    result = tool_functions[tool_name](**tool_input)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            # Track usage
            tools_used.append(tool_name)
            tool_results.append({
                "tool": tool_name,
                "input": tool_input,
                "result": result,  # Full result for reference building
                "result_preview": _preview_result(result)
            })

            # Add tool result to messages (OpenAI format)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

    # Hit max turns
    logger.warning(f"[Agent V4] Hit max turns ({MAX_TURNS})")

    return {
        "success": True,
        "answer": "I wasn't able to complete the analysis within the expected steps. Please try rephrasing your question.",
        "tools_used": tools_used,
        "tool_results": tool_results,
        "turn_count": turn_count,
        "mode": mode,
        "model": model,
        "conversation_id": conversation_id,
        "warning": "max_turns_exceeded"
    }


def _preview_result(result: dict) -> str:
    """Create a short preview of a tool result for logging."""
    if not isinstance(result, dict):
        return str(result)[:100]

    if "error" in result:
        return f"ERROR: {result['error']}"

    preview_parts = []

    for key in ["session_name", "total_sessions", "sessions_found", "result_count", "available"]:
        if key in result:
            preview_parts.append(f"{key}={result[key]}")

    if "summary" in result and isinstance(result["summary"], dict):
        summary = result["summary"]
        for key in ["overall_score", "total_utterances", "total_nodes"]:
            if key in summary:
                preview_parts.append(f"{key}={summary[key]}")

    return ", ".join(preview_parts) if preview_parts else "ok"


def _extract_session_focus(response_text: str, current_context: Dict = None) -> Optional[int]:
    """Extract session focus from response for multi-turn context."""
    import re

    if not response_text:
        if current_context and current_context.get("session_focus"):
            return current_context["session_focus"]
        return None

    # Check for "Session X" pattern
    match = re.search(r'\bSession\s+(\d+)\b', response_text)
    if match:
        return int(match.group(1))

    # Check for session names
    session_name_to_id = {
        'living in nyc': 18, 'nyc': 18,
        'is ai alive': 19, 'ai alive': 19,
        'nuclear fusion': 20, 'fusion': 20,
        'shaw interview': 21, 'shaw': 21,
        'collaboration literacy': 22,
        'dinosaurs': 23,
        'country music': 24,
        'abundance': 25
    }

    response_lower = response_text.lower()
    for name, sid in session_name_to_id.items():
        if name in response_lower:
            return sid

    # Keep current context if no new session mentioned
    if current_context and current_context.get("session_focus"):
        return current_context["session_focus"]

    return None


# =============================================================================
# CONVERSATION MANAGEMENT
# =============================================================================

class ConversationManager:
    """Manages multi-turn conversation state."""

    def __init__(self):
        self.conversations: Dict[str, Dict] = {}

    def get_context(self, conversation_id: str) -> Dict:
        """Get context for a conversation."""
        return self.conversations.get(conversation_id, {
            "history": [],
            "session_focus": None,
            "speaker_focus": None
        })

    def update_context(self, conversation_id: str, response: Dict) -> None:
        """Update context after a response."""
        from datetime import datetime

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "history": [],
                "session_focus": None,
                "speaker_focus": None,
                "title": "New Conversation",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

        ctx = self.conversations[conversation_id]
        ctx["updated_at"] = datetime.now().isoformat()

        if response.get("session_focus"):
            ctx["session_focus"] = response["session_focus"]

    def add_exchange(self, conversation_id: str, query: str, response: str) -> None:
        """Add a query-response exchange to history."""
        from datetime import datetime

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "history": [],
                "session_focus": None,
                "speaker_focus": None,
                "title": query[:50] + "..." if len(query) > 50 else query,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

        ctx = self.conversations[conversation_id]
        ctx["updated_at"] = datetime.now().isoformat()

        # Set title from first query if not set
        if ctx.get("title") == "New Conversation" and query:
            ctx["title"] = query[:50] + "..." if len(query) > 50 else query

        history = ctx["history"]
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})

        # Keep last 10 messages (5 exchanges)
        if len(history) > 10:
            ctx["history"] = history[-10:]

    def clear(self, conversation_id: str = None) -> None:
        """Clear conversation context."""
        if conversation_id:
            self.conversations.pop(conversation_id, None)
        else:
            self.conversations.clear()

    def list_conversations(self) -> List[Dict]:
        """List all conversations sorted by last update."""
        conversations = []
        for conv_id, ctx in self.conversations.items():
            if ctx.get("history"):  # Only include conversations with messages
                conversations.append({
                    "conversation_id": conv_id,
                    "title": ctx.get("title", "Conversation"),
                    "created_at": ctx.get("created_at"),
                    "updated_at": ctx.get("updated_at"),
                    "message_count": len(ctx.get("history", [])),
                    "session_focus": ctx.get("session_focus")
                })

        # Sort by updated_at, most recent first
        conversations.sort(
            key=lambda c: c.get("updated_at") or "",
            reverse=True
        )

        return conversations

    def create_conversation(self, conversation_id: str, title: str = "New Conversation") -> Dict:
        """Explicitly create a new conversation."""
        from datetime import datetime

        self.conversations[conversation_id] = {
            "history": [],
            "session_focus": None,
            "speaker_focus": None,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        return self.conversations[conversation_id]


# Global conversation manager
conversation_manager = ConversationManager()
