"""
Core Agent for V5: Context-First Agentic Architecture.

Design principles:
1. Pre-load relevant context BEFORE the agentic loop
2. Tools are ALWAYS available (truly agentic)
3. Query understanding determines retrieval strategy
4. Triangulation-aware prompts guide cross-source reasoning
5. Simple ReAct loop after context injection

This combines the best of RAG (intelligent retrieval) with agentic control (tool autonomy).
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional

from openai import OpenAI

from .query_understanding import understand_query, QueryIntent, get_session_name
from .context_assembly import assemble_context
from .prompts import (
    get_system_prompt,
    format_context_injection,
    TOOL_AVAILABILITY_NOTE,
    TOOL_AVAILABILITY_NOTE_BASELINE
)

# Import tools from V4 (reuse the same tool implementations)
from agent_v4.tools import (
    get_tool_schemas,
    TOOL_FUNCTIONS,
    BASELINE_TOOL_FUNCTIONS
)

logger = logging.getLogger(__name__)

# Configuration
MAX_TURNS = 10  # Maximum tool-use turns
DEFAULT_MODEL = os.getenv("LLM_REASONING_MODEL", "gpt-4o")

# Initialize OpenAI client
_client = None
_rag_service = None


def _get_client():
    """Get or create OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _get_rag_service():
    """Get RAG service for semantic retrieval."""
    global _rag_service
    if _rag_service is None:
        try:
            from rag_service import RAGService
            _rag_service = RAGService()
            logger.info("[Agent V5] RAG service initialized")
        except Exception as e:
            logger.warning(f"[Agent V5] Could not initialize RAG service: {e}")
            _rag_service = None
    return _rag_service


def _convert_tools_to_openai_format(tool_schemas: List[Dict]) -> List[Dict]:
    """Convert tool schemas to OpenAI function calling format."""
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
    Run the V5 agent: context-first, always agentic.

    Flow:
    1. Understand query intent
    2. Pre-load relevant context based on intent
    3. Inject context into system prompt
    4. Run agentic loop with tools ALWAYS available
    5. Return response with metadata

    Args:
        query: User's question
        conversation_id: Optional conversation ID for multi-turn
        conversation_history: Previous messages in conversation
        mode: "enhanced" (all tools + artifacts) or "baseline" (transcript only)
        session_context: Current context (session focus, etc.)
        model: Optional model override

    Returns:
        Agent response with answer, tools used, context used, and metadata
    """
    model = model or DEFAULT_MODEL
    client = _get_client()
    rag_service = _get_rag_service() if mode == "enhanced" else None

    session_context = session_context or {}

    logger.info(f"[Agent V5] Query: '{query}' (mode={mode}, model={model})")

    # ==========================================================================
    # PHASE 1: Query Understanding
    # ==========================================================================

    # Build conversation context for query understanding
    conversation_ctx = {
        "session_focus": session_context.get("session_focus"),
        "speaker_focus": session_context.get("speaker_focus"),
        "has_history": bool(conversation_history)
    }

    intent = understand_query(query, conversation_ctx)

    logger.info(f"[Agent V5] Intent: {intent.intent_type}, Mode: {intent.retrieval_mode}, "
                f"Sessions: {intent.session_ids}, Speakers: {intent.speaker_names}")

    # ==========================================================================
    # PHASE 2: Context Pre-loading
    # ==========================================================================

    context_result = {"context_text": "", "retrieval_metadata": {}, "sessions_loaded": []}

    if intent.needs_retrieval:
        context_result = assemble_context(intent, query, rag_service)
        logger.info(f"[Agent V5] Context assembled: mode={context_result['retrieval_metadata'].get('mode')}, "
                   f"sessions={context_result.get('sessions_loaded', [])}")

    # ==========================================================================
    # PHASE 3: Build Messages with Injected Context
    # ==========================================================================

    # Get base system prompt
    system_prompt = get_system_prompt(mode)

    # Add tool availability note
    tool_note = TOOL_AVAILABILITY_NOTE if mode == "enhanced" else TOOL_AVAILABILITY_NOTE_BASELINE
    system_prompt = f"{system_prompt}\n\n{tool_note}"

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)

    # Build user message with optional context injection
    if context_result["context_text"]:
        # Inject pre-loaded context as a system message before user query
        context_injection = format_context_injection(
            context_result["context_text"],
            context_result["retrieval_metadata"]
        )
        messages.append({"role": "system", "content": context_injection})

    # Add session context hint if relevant
    user_message = query
    if session_context.get("session_focus") and not intent.session_ids:
        # Only add context hint if no explicit session in query
        focus = session_context["session_focus"]
        focus_name = get_session_name(focus) or f"Session {focus}"
        user_message = f"[Continuing discussion about {focus_name}]\n\n{query}"

    messages.append({"role": "user", "content": user_message})

    # ==========================================================================
    # PHASE 4: Agentic Loop (tools ALWAYS available)
    # ==========================================================================

    tool_schemas = get_tool_schemas(mode)
    tools = _convert_tools_to_openai_format(tool_schemas)
    tool_functions = TOOL_FUNCTIONS if mode == "enhanced" else BASELINE_TOOL_FUNCTIONS

    # Track execution
    tools_used = []
    tool_results = []
    turn_count = 0

    # ReAct loop
    while turn_count < MAX_TURNS:
        turn_count += 1
        logger.info(f"[Agent V5] Turn {turn_count}")

        # Call OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            max_tokens=4096
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        logger.info(f"[Agent V5] Finish reason: {finish_reason}")

        # No tool calls = we're done
        if not message.tool_calls:
            text_content = message.content or ""
            logger.info(f"[Agent V5] Final response ({len(text_content)} chars)")

            # Extract session focus for multi-turn
            new_session_focus = _extract_session_focus(
                text_content,
                intent.session_ids,
                session_context
            )

            return {
                "success": True,
                "answer": text_content,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "turn_count": turn_count,
                "mode": mode,
                "model": model,
                "conversation_id": conversation_id,
                "session_focus": new_session_focus,
                # V5-specific metadata
                "query_intent": {
                    "type": intent.intent_type,
                    "retrieval_mode": intent.retrieval_mode,
                    "entities": {
                        "sessions": intent.session_ids,
                        "speakers": intent.speaker_names,
                        "topics": intent.topics
                    }
                },
                "context_preloaded": {
                    "mode": context_result["retrieval_metadata"].get("mode", "none"),
                    "sessions_loaded": context_result.get("sessions_loaded", []),
                    "metadata": context_result.get("retrieval_metadata", {})
                }
            }

        # LLM wants to use tools
        messages.append(message)

        # Execute each tool call
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments

            try:
                tool_input = json.loads(tool_args_str)
            except json.JSONDecodeError:
                tool_input = {}

            logger.info(f"[Agent V5] Tool call: {tool_name}({json.dumps(tool_input)[:100]}...)")

            # Execute
            if tool_name in tool_functions:
                try:
                    result = tool_functions[tool_name](**tool_input)
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            # Track
            tools_used.append(tool_name)
            tool_results.append({
                "tool": tool_name,
                "input": tool_input,
                "result": result,
                "result_preview": _preview_result(result)
            })

            # Add result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

    # Hit max turns
    logger.warning(f"[Agent V5] Hit max turns ({MAX_TURNS})")

    return {
        "success": True,
        "answer": "I wasn't able to complete the analysis within the expected steps. Please try rephrasing your question.",
        "tools_used": tools_used,
        "tool_results": tool_results,
        "turn_count": turn_count,
        "mode": mode,
        "model": model,
        "conversation_id": conversation_id,
        "warning": "max_turns_exceeded",
        "query_intent": {
            "type": intent.intent_type,
            "retrieval_mode": intent.retrieval_mode
        },
        "context_preloaded": {
            "mode": context_result["retrieval_metadata"].get("mode", "none"),
            "sessions_loaded": context_result.get("sessions_loaded", [])
        }
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


def _extract_session_focus(
    response_text: str,
    intent_sessions: List[int],
    current_context: Dict = None
) -> Optional[int]:
    """Extract session focus for multi-turn context."""
    import re

    # If intent had explicit session, use that
    if intent_sessions:
        return intent_sessions[0]

    if not response_text:
        if current_context and current_context.get("session_focus"):
            return current_context["session_focus"]
        return None

    # Check for "Session X" pattern in response
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
# CONVERSATION MANAGEMENT (mirrors V4)
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

        # Store query intent for potential future use
        if response.get("query_intent"):
            ctx["last_intent"] = response["query_intent"]

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
            if ctx.get("history"):
                conversations.append({
                    "conversation_id": conv_id,
                    "title": ctx.get("title", "Conversation"),
                    "created_at": ctx.get("created_at"),
                    "updated_at": ctx.get("updated_at"),
                    "message_count": len(ctx.get("history", [])),
                    "session_focus": ctx.get("session_focus")
                })

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

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get full conversation history for a conversation."""
        ctx = self.conversations.get(conversation_id, {})
        return ctx.get("history", [])


# Global conversation manager
conversation_manager = ConversationManager()
