#!/usr/bin/env python3
"""
Data Flow Tracer for V7 Agent

Traces what data the agent ACTUALLY sees at each phase.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_FLOW_LOG = []

def log_phase(phase: str, data: dict):
    """Log data at each phase."""
    DATA_FLOW_LOG.append({
        "phase": phase,
        "data": data
    })
    print(f"\n{'='*60}")
    print(f"PHASE: {phase}")
    print(f"{'='*60}")
    for key, value in data.items():
        if isinstance(value, str) and len(value) > 500:
            print(f"{key}: [{len(value)} chars] {value[:300]}...")
        elif isinstance(value, list) and len(value) > 10:
            print(f"{key}: [{len(value)} items] {value[:3]}...")
        else:
            print(f"{key}: {value}")


def trace_query(query: str):
    """Trace a single query through the system."""
    from agent_v7.classifier import classify_query
    from agent_v7.memory import get_memory, clear_memory
    from agent_v7.exploratory import retrieve_exploratory, format_exploratory_evidence_for_synthesis
    from agent_v7.tools_v2 import execute_tool

    conv_id = f"trace_{hash(query) % 10000}"
    memory = get_memory(conv_id)

    print(f"\n{'#'*70}")
    print(f"TRACING: {query}")
    print(f"{'#'*70}")

    # Phase 1: Classification
    classification = classify_query(query, memory)
    log_phase("1. CLASSIFICATION", {
        "query": query,
        "is_exploratory": classification.is_exploratory,
        "session_ids": classification.session_ids,
        "speakers": classification.speakers,
        "topics": classification.topics,
        "artifact_hint": classification.artifact_hint,
    })

    if classification.is_exploratory:
        # Phase 2: Exploratory Retrieval with tool tracing
        tool_calls = []
        
        def make_tool_fn(tool_name):
            def fn(**kwargs):
                result = execute_tool(tool_name, kwargs)
                display = result.get('display', str(result)) if isinstance(result, dict) else str(result)
                tool_calls.append({
                    "tool": tool_name,
                    "kwargs": kwargs,
                    "result_length": len(display),
                })
                log_phase(f"2a. TOOL: {tool_name}", {
                    "kwargs": kwargs,
                    "result_length": len(display),
                    "result_preview": display[:800] if len(display) > 800 else display,
                })
                return result
            return fn

        tools_dict = {
            'list_sessions': make_tool_fn('list_sessions'),
            'search_sessions': make_tool_fn('search_sessions'),
            'get_transcript': make_tool_fn('get_transcript'),
            'get_7c_analysis': make_tool_fn('get_7c_analysis'),
            'get_concept_map': make_tool_fn('get_concept_map'),
            'get_speaker_profile': make_tool_fn('get_speaker_profile'),
        }

        result = retrieve_exploratory(query, classification, tools_dict)

        log_phase("2b. EXPLORATORY SUMMARY", {
            "sessions_searched": result.sessions_searched,
            "evidence_count": len(result.evidence),
            "evidence_sessions": [e.session_id for e in result.evidence],
            "evidence_tools": [e.tool_used for e in result.evidence],
            "total_content_chars": sum(len(e.content) for e in result.evidence),
        })

        # Phase 3: Synthesis context
        synthesis_context = format_exploratory_evidence_for_synthesis(result)
        log_phase("3. SYNTHESIS CONTEXT", {
            "total_length": len(synthesis_context),
            "first_2000_chars": synthesis_context[:2000],
        })

    else:
        log_phase("2. TARGETED PATH", {
            "note": "Using ReAct loop, not exploratory",
        })

    clear_memory(conv_id)
    return DATA_FLOW_LOG.copy()


def main():
    global DATA_FLOW_LOG
    
    queries = [
        "Did Tucker demonstrate systems thinking in session 19?",
        "Which session has the best collaboration quality?",
        "What sessions discussed technology and its societal impact?",
    ]

    all_traces = {}

    for query in queries:
        DATA_FLOW_LOG = []
        try:
            trace = trace_query(query)
            all_traces[query] = trace
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "data_flow_traces.json")
    with open(output_path, 'w') as f:
        json.dump(all_traces, f, indent=2, default=str)
    print(f"\n\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
