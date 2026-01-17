"""
Full Data Flow Trace for V7 Agent

This script traces EXACTLY what the agent receives at each phase of execution.
NO summarization, NO truncation - captures the complete data flow.

Purpose: Debug data loss issues by seeing the actual data passed between components.
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_v7.memory import clear_caches, get_memory, clear_memory, ConversationMemory
from agent_v7.classifier import classify_query, QueryClassification
from agent_v7.tools_v2 import (
    list_sessions, search_sessions, get_transcript,
    get_concept_map, get_7c_analysis, get_speaker_profile
)

# Output file
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "full_data_flow_trace.md")


def trace_tool_call(tool_name: str, tool_args: dict) -> Dict[str, Any]:
    """Execute a tool and return the FULL result."""
    tools = {
        'list_sessions': list_sessions,
        'search_sessions': search_sessions,
        'get_transcript': get_transcript,
        'get_concept_map': get_concept_map,
        'get_7c_analysis': get_7c_analysis,
        'get_speaker_profile': get_speaker_profile,
    }

    if tool_name not in tools:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = tools[tool_name](**tool_args)
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def format_result_for_markdown(result: Any, max_depth: int = 10) -> str:
    """Format result as markdown code block, preserving full content."""
    if isinstance(result, dict):
        try:
            return json.dumps(result, indent=2, ensure_ascii=False, default=str)
        except:
            return str(result)
    elif isinstance(result, (list, tuple)):
        try:
            return json.dumps(list(result), indent=2, ensure_ascii=False, default=str)
        except:
            return str(result)
    else:
        return str(result)


def trace_exploratory_query(query: str, conversation_id: str) -> Dict[str, Any]:
    """
    Trace an exploratory (cross-session) query.

    This simulates what the exploratory retrieval path does.
    """
    trace = {
        "query": query,
        "query_type": "EXPLORATORY",
        "conversation_id": conversation_id,
        "timestamp": datetime.now().isoformat(),
        "phases": []
    }

    print(f"\n{'='*60}")
    print(f"TRACING EXPLORATORY: {query}")
    print(f"{'='*60}")

    # Phase 1: Query Classification
    classification = classify_query(query, memory=None)

    phase1 = {
        "phase": "1. Query Classification",
        "input": query,
        "output": {
            "is_exploratory": classification.is_exploratory,
            "session_ids": classification.session_ids,
            "speakers": classification.speakers,
            "topics": classification.topics,
            "artifact_hint": classification.artifact_hint,
            "reason": classification.reason
        }
    }
    trace["phases"].append(phase1)
    print(f"\nPhase 1 - Classification: {classification.reason}")
    print(f"  is_exploratory: {classification.is_exploratory}")
    print(f"  artifact_hint: {classification.artifact_hint}")

    # Phase 2: Tool Execution
    tool_calls = []

    # Step 2a: list_sessions to get all sessions with scores
    print("\nPhase 2a - Calling list_sessions")
    list_result = trace_tool_call('list_sessions', {})
    tool_calls.append({
        "tool": "list_sessions",
        "args": {},
        "result": list_result
    })

    # Step 2b: Select top 3 sessions by collaboration score
    if 'sessions' in list_result:
        sessions = list_result['sessions']
        top_sessions = sorted(
            [s for s in sessions if s.get('collaboration_score') is not None],
            key=lambda x: x.get('collaboration_score', 0),
            reverse=True
        )[:3]

        print(f"\nPhase 2b - Selected top 3 sessions: {[(s['session_id'], s.get('collaboration_score')) for s in top_sessions]}")

        # Step 2c: Get 7C analysis for each top session
        for session in top_sessions:
            sid = session.get('session_id')
            print(f"\nPhase 2c - Calling get_7c_analysis for session {sid}")
            seven_c_result = trace_tool_call('get_7c_analysis', {'session_id': sid})
            tool_calls.append({
                "tool": "get_7c_analysis",
                "args": {"session_id": sid},
                "result": seven_c_result
            })

    phase2 = {
        "phase": "2. Tool Execution (Exploratory Path)",
        "tool_calls": tool_calls
    }
    trace["phases"].append(phase2)

    # Phase 3: Combined evidence for LLM
    evidence_parts = []
    for tc in tool_calls:
        tool_name = tc['tool']
        result = tc['result']
        if isinstance(result, dict) and 'display' in result:
            evidence_parts.append(f"=== {tool_name} ===\n{result['display']}")
        else:
            evidence_parts.append(f"=== {tool_name} ===\n{format_result_for_markdown(result)}")

    combined_evidence = "\n\n".join(evidence_parts)

    phase3 = {
        "phase": "3. Evidence Passed to LLM",
        "combined_evidence": combined_evidence,
        "evidence_length_chars": len(combined_evidence),
        "evidence_length_tokens_approx": len(combined_evidence) // 4
    }
    trace["phases"].append(phase3)

    print(f"\nPhase 3 - Combined evidence: {len(combined_evidence)} chars (~{len(combined_evidence)//4} tokens)")

    return trace


def trace_targeted_query(query: str, session_id: int, conversation_id: str) -> Dict[str, Any]:
    """
    Trace a targeted (single-session) query with all artifact types.

    Gets transcript, concept map, AND 7C analysis for the specified session.
    """
    trace = {
        "query": query,
        "query_type": "TARGETED",
        "target_session": session_id,
        "conversation_id": conversation_id,
        "timestamp": datetime.now().isoformat(),
        "phases": []
    }

    print(f"\n{'='*60}")
    print(f"TRACING TARGETED: {query}")
    print(f"Target Session: {session_id}")
    print(f"{'='*60}")

    # Phase 1: Query Classification (with memory that has session focus)
    memory = ConversationMemory(conversation_id)
    memory.update_session_focus(session_id)

    classification = classify_query(query, memory=memory)

    phase1 = {
        "phase": "1. Query Classification",
        "input": query,
        "memory_context": {
            "session_focus": memory.session_focus,
            "session_name": memory.session_name
        },
        "output": {
            "is_exploratory": classification.is_exploratory,
            "session_ids": classification.session_ids,
            "speakers": classification.speakers,
            "topics": classification.topics,
            "artifact_hint": classification.artifact_hint,
            "reason": classification.reason
        }
    }
    trace["phases"].append(phase1)
    print(f"\nPhase 1 - Classification: {classification.reason}")
    print(f"  is_exploratory: {classification.is_exploratory}")
    print(f"  session_ids: {classification.session_ids}")

    # Phase 2: Tool Execution - Get ALL three artifact types
    tool_calls = []

    # 2a: Get transcript
    print(f"\nPhase 2a - Calling get_transcript for session {session_id}")
    transcript_result = trace_tool_call('get_transcript', {'session_id': session_id})
    tool_calls.append({
        "tool": "get_transcript",
        "args": {"session_id": session_id},
        "result": transcript_result
    })

    # 2b: Get concept map
    print(f"\nPhase 2b - Calling get_concept_map for session {session_id}")
    concept_map_result = trace_tool_call('get_concept_map', {'session_id': session_id})
    tool_calls.append({
        "tool": "get_concept_map",
        "args": {"session_id": session_id},
        "result": concept_map_result
    })

    # 2c: Get 7C analysis
    print(f"\nPhase 2c - Calling get_7c_analysis for session {session_id}")
    seven_c_result = trace_tool_call('get_7c_analysis', {'session_id': session_id})
    tool_calls.append({
        "tool": "get_7c_analysis",
        "args": {"session_id": session_id},
        "result": seven_c_result
    })

    phase2 = {
        "phase": "2. Tool Execution (Targeted Path - All Artifacts)",
        "tool_calls": tool_calls
    }
    trace["phases"].append(phase2)

    # Phase 3: Combined evidence for LLM
    evidence_parts = []
    for tc in tool_calls:
        tool_name = tc['tool']
        result = tc['result']
        if isinstance(result, dict) and 'display' in result:
            evidence_parts.append(f"=== {tool_name} ===\n{result['display']}")
        else:
            evidence_parts.append(f"=== {tool_name} ===\n{format_result_for_markdown(result)}")

    combined_evidence = "\n\n".join(evidence_parts)

    phase3 = {
        "phase": "3. Evidence Passed to LLM",
        "combined_evidence": combined_evidence,
        "evidence_length_chars": len(combined_evidence),
        "evidence_length_tokens_approx": len(combined_evidence) // 4
    }
    trace["phases"].append(phase3)

    print(f"\nPhase 3 - Combined evidence: {len(combined_evidence)} chars (~{len(combined_evidence)//4} tokens)")

    return trace


def write_trace_document(traces: List[Dict[str, Any]]):
    """Write the full trace to a markdown document."""

    lines = [
        "# V7 Agent Full Data Flow Trace",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        "",
        "## Purpose",
        "",
        "This document shows the COMPLETE, UNTRUNCATED data that flows through the V7 agent.",
        "Use this to debug data loss issues by seeing exactly what each component receives.",
        "",
        "## Test Queries",
        "",
        "1. **Exploratory Query**: \"Which session has the best collaboration?\"",
        "   - Triggers: list_sessions -> get_7c_analysis (top 3 sessions)",
        "",
        "2. **Targeted Query**: \"What did they discuss and how did their ideas connect?\" (Session 24)",
        "   - Triggers: get_transcript, get_concept_map, get_7c_analysis",
        "",
        "---",
        ""
    ]

    for i, trace in enumerate(traces, 1):
        lines.append(f"# Query {i}: {trace['query']}")
        lines.append("")
        lines.append(f"**Query Type**: {trace.get('query_type', 'Unknown')}")
        if 'target_session' in trace:
            lines.append(f"**Target Session**: {trace['target_session']}")
        lines.append(f"**Conversation ID**: {trace['conversation_id']}")
        lines.append(f"**Timestamp**: {trace['timestamp']}")
        lines.append("")

        for phase in trace['phases']:
            lines.append(f"## {phase['phase']}")
            lines.append("")

            if phase['phase'] == "1. Query Classification":
                lines.append("### Input Query")
                lines.append("```")
                lines.append(phase['input'])
                lines.append("```")
                lines.append("")

                if 'memory_context' in phase:
                    lines.append("### Memory Context")
                    lines.append("```json")
                    lines.append(json.dumps(phase['memory_context'], indent=2))
                    lines.append("```")
                    lines.append("")

                lines.append("### Classification Result")
                lines.append("```json")
                lines.append(json.dumps(phase['output'], indent=2))
                lines.append("```")
                lines.append("")

            elif phase['phase'].startswith("2. Tool Execution"):
                for j, tc in enumerate(phase['tool_calls'], 1):
                    lines.append(f"### Tool Call {j}: `{tc['tool']}`")
                    lines.append("")
                    lines.append("**Arguments:**")
                    lines.append("```json")
                    lines.append(json.dumps(tc['args'], indent=2))
                    lines.append("```")
                    lines.append("")

                    # Show full result
                    result = tc['result']
                    result_str = format_result_for_markdown(result)

                    lines.append(f"**FULL Result** ({len(result_str)} chars):")
                    lines.append("")
                    lines.append("```json")
                    lines.append(result_str)
                    lines.append("```")
                    lines.append("")

                    # If there's a 'display' field, show it separately for clarity
                    if isinstance(result, dict) and 'display' in result:
                        lines.append("**Display Field (what LLM sees for this tool):**")
                        lines.append("")
                        lines.append("```")
                        lines.append(result['display'])
                        lines.append("```")
                        lines.append("")

            elif phase['phase'] == "3. Evidence Passed to LLM":
                lines.append(f"**Total Evidence Size**: {phase['evidence_length_chars']:,} characters (~{phase['evidence_length_tokens_approx']:,} tokens)")
                lines.append("")
                lines.append("### Combined Evidence String")
                lines.append("")
                lines.append("This is EXACTLY what gets passed to the LLM for synthesis:")
                lines.append("")
                lines.append("```")
                lines.append(phase['combined_evidence'])
                lines.append("```")
                lines.append("")

        lines.append("---")
        lines.append("")

    # Write to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n{'='*60}")
    print(f"Trace written to: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE):,} bytes")
    print(f"{'='*60}")


def main():
    """Run traces for selected queries."""

    # Clear caches to ensure fresh data
    clear_caches()

    traces = []

    # Query 1: Exploratory - "Which session has the best collaboration?"
    trace1 = trace_exploratory_query(
        "Which session has the best collaboration?",
        "trace_exploratory_1"
    )
    traces.append(trace1)

    # Query 2: Targeted - Session 24 (Country Music) with all artifacts
    trace2 = trace_targeted_query(
        "What did they discuss and how did their ideas connect?",
        session_id=24,  # Country Music - has good data
        conversation_id="trace_targeted_1"
    )
    traces.append(trace2)

    # Write the document
    write_trace_document(traces)

    return traces


if __name__ == "__main__":
    main()
