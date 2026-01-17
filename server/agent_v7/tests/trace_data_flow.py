#!/usr/bin/env python3
"""
Data Flow Tracer for V7 Agent

Traces data through each phase to identify information loss:
1. Input Processing - What context is extracted
2. Tool Selection - What the LLM decides to call
3. Tool Execution - Raw data returned by tools
4. Evidence Accumulation - How evidence is formatted
5. Synthesis - Final answer generation

Run: cd /home/ubuntu/chemistry-dashboard/server && \
     OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d'=' -f2) \
     ~/.pyenv/versions/blinc/bin/python agent_v7/tests/trace_data_flow.py
"""

import sys
import os
import json
from typing import Dict, Any, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Patch tools to capture raw output
_tool_raw_outputs = []

def trace_tool_call(tool_name: str, params: Dict, result: Any):
    """Capture tool call with raw output."""
    _tool_raw_outputs.append({
        "tool": tool_name,
        "params": params,
        "raw_result": result,
        "result_type": type(result).__name__,
        "result_length": len(str(result)) if result else 0
    })

# Complex queries that should trigger multi-step reasoning
TRACE_QUERIES = [
    {
        "id": "trace_01",
        "query": "Compare the collaboration quality between the AI discussion and Nuclear Fusion session. Which one had better teamwork?",
        "expected_flow": [
            "Should identify sessions 19 and 20",
            "Should call get_7c_analysis for both",
            "Should compare specific dimensions",
            "Synthesis should cite scores and evidence"
        ],
        "key_data_points": {
            "session_19_7c": "Should have 7C scores for Is AI Alive",
            "session_20_7c": "Should have 7C scores for Nuclear Fusion",
            "comparison": "Should compare specific C dimensions"
        }
    },
    {
        "id": "trace_02",
        "query": "What did David say about temperature in the Nuclear Fusion discussion, and how does that connect to the main concepts?",
        "expected_flow": [
            "Should identify session 20",
            "Should call get_transcript with speaker filter",
            "Should call get_concept_map",
            "Should connect transcript to concepts"
        ],
        "key_data_points": {
            "transcript": "David's quotes about temperature",
            "concepts": "Related concept nodes",
            "connection": "How transcript relates to concepts"
        }
    },
    {
        "id": "trace_03",
        "query": "I think Tucker dominated the AI discussion. Can you verify this with specific evidence from the transcript and collaboration scores?",
        "expected_flow": [
            "Should identify session 19",
            "Should get transcript for Tucker vs Sam",
            "Should get 7C analysis (contribution dimension)",
            "Should verify/refute with evidence"
        ],
        "key_data_points": {
            "tucker_utterances": "Count and content",
            "sam_utterances": "Count and content",
            "contribution_score": "7C contribution dimension"
        }
    }
]


def trace_single_query(query_spec: Dict) -> Dict:
    """Trace a single query through all phases."""
    global _tool_raw_outputs
    _tool_raw_outputs = []  # Reset

    query_id = query_spec["id"]
    query = query_spec["query"]

    print(f"\n{'='*80}")
    print(f"TRACING: [{query_id}]")
    print(f"Query: {query}")
    print(f"{'='*80}")

    trace_result = {
        "query_id": query_id,
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "phases": {}
    }

    # =========================================================================
    # PHASE 1: Input Processing
    # =========================================================================
    print("\n--- PHASE 1: Input Processing ---")

    from agent_v7.memory import get_memory, clear_memory

    conv_id = f"trace_{query_id}"
    clear_memory(conv_id)  # Fresh start
    memory = get_memory(conv_id)

    # Extract context
    session_id = memory.extract_session_from_text(query)
    speaker = memory.extract_speaker_from_text(query)

    phase1 = {
        "extracted_session": session_id,
        "extracted_speaker": speaker,
        "memory_session_focus": memory.session_focus,
        "memory_speaker_focus": memory.speaker_focus,
    }
    trace_result["phases"]["1_input_processing"] = phase1

    print(f"  Extracted session: {session_id}")
    print(f"  Extracted speaker: {speaker}")

    # =========================================================================
    # PHASE 2: Tool Selection (First LLM Call)
    # =========================================================================
    print("\n--- PHASE 2: Tool Selection ---")

    # IMPORTANT: Patch execute_tool BEFORE importing ScaffoldingAgent
    # to ensure the patched version is used
    from agent_v7 import tools_v2
    original_execute = tools_v2.execute_tool

    def patched_execute(tool_name: str, params: Dict) -> str:
        result = original_execute(tool_name, params)
        trace_tool_call(tool_name, params, result)
        return result

    tools_v2.execute_tool = patched_execute

    # Now import and use the agent (it will use patched execute_tool)
    # Force reimport to pick up patched version
    import importlib
    from agent_v7 import react_agent as ra
    ra.execute_tool = patched_execute  # Patch the already-imported reference

    from agent_v7.react_agent import ScaffoldingAgent
    from agent_v7.steering import extract_steering

    agent = ScaffoldingAgent(conversation_id=conv_id)
    steering = extract_steering(query)

    # Get memory context
    memory_context = memory.get_context_for_llm()

    phase2 = {
        "memory_context_for_llm": memory_context,
        "steering_raw": steering.raw_instructions,
        "steering_api_preferred": steering.api_preferred,
        "steering_api_excluded": steering.api_excluded,
    }
    trace_result["phases"]["2_tool_selection"] = phase2

    print(f"  Memory context length: {len(memory_context)} chars")
    print(f"  Steering: preferred={steering.api_preferred}, excluded={steering.api_excluded}")

    # =========================================================================
    # PHASE 3: Tool Execution (Intercept actual tool calls)
    # =========================================================================
    print("\n--- PHASE 3: Tool Execution ---")

    try:
        # Run the agent (patched execute_tool will capture outputs)
        response = agent.respond(query)
    finally:
        # Restore original
        tools_v2.execute_tool = original_execute
        ra.execute_tool = original_execute

    # Collect tool execution data
    phase3 = {
        "tool_calls": [],
        "total_raw_data_chars": 0
    }

    for i, tool_output in enumerate(_tool_raw_outputs):
        tool_info = {
            "order": i + 1,
            "tool": tool_output["tool"],
            "params": tool_output["params"],
            "raw_result_preview": str(tool_output["raw_result"])[:500],
            "raw_result_length": tool_output["result_length"],
        }
        phase3["tool_calls"].append(tool_info)
        phase3["total_raw_data_chars"] += tool_output["result_length"]

        print(f"  Tool {i+1}: {tool_output['tool']}")
        print(f"    Params: {tool_output['params']}")
        print(f"    Result length: {tool_output['result_length']} chars")
        print(f"    Preview: {str(tool_output['raw_result'])[:200]}...")

    trace_result["phases"]["3_tool_execution"] = phase3

    # =========================================================================
    # PHASE 4: Evidence Accumulation
    # =========================================================================
    print("\n--- PHASE 4: Evidence Accumulation ---")

    phase4 = {
        "evidence_count": len(response.evidence),
        "evidence_items": []
    }

    for i, ev in enumerate(response.evidence):
        ev_info = {
            "order": i + 1,
            "type": ev.get("type", "unknown"),
            "source": ev.get("source", "unknown"),
            "content_preview": str(ev.get("content", ""))[:300],
            "content_length": len(str(ev.get("content", "")))
        }
        phase4["evidence_items"].append(ev_info)

        print(f"  Evidence {i+1}: type={ev.get('type')}, source={ev.get('source')}")
        print(f"    Content length: {len(str(ev.get('content', '')))} chars")

    trace_result["phases"]["4_evidence_accumulation"] = phase4

    # =========================================================================
    # PHASE 5: Synthesis
    # =========================================================================
    print("\n--- PHASE 5: Synthesis ---")

    phase5 = {
        "final_answer": response.answer,
        "answer_length": len(response.answer),
        "tool_calls_made": [
            {"name": tc.name, "params": tc.params}
            for tc in response.tool_calls_made
        ],
        "suggested_explorations": response.suggested_explorations,
        "session_focus": response.session_focus,
        "speaker_focus": response.speaker_focus,
    }
    trace_result["phases"]["5_synthesis"] = phase5

    print(f"  Answer length: {len(response.answer)} chars")
    print(f"  Tools used: {[tc.name for tc in response.tool_calls_made]}")
    print(f"  Session focus: {response.session_focus}")

    # =========================================================================
    # ANALYSIS: Check for information loss
    # =========================================================================
    print("\n--- ANALYSIS: Information Flow ---")

    analysis = {
        "raw_data_chars": phase3["total_raw_data_chars"],
        "evidence_chars": sum(e["content_length"] for e in phase4["evidence_items"]),
        "answer_chars": phase5["answer_length"],
        "compression_ratio": None,
        "potential_issues": []
    }

    if phase3["total_raw_data_chars"] > 0:
        analysis["compression_ratio"] = phase5["answer_length"] / phase3["total_raw_data_chars"]

    # Check for issues
    if len(phase3["tool_calls"]) == 0:
        analysis["potential_issues"].append("No tools called - may be answering from memory only")

    if phase4["evidence_count"] == 0 and len(phase3["tool_calls"]) > 0:
        analysis["potential_issues"].append("Tools called but no evidence accumulated")

    if analysis["compression_ratio"] and analysis["compression_ratio"] < 0.05:
        analysis["potential_issues"].append(f"Very high compression ({analysis['compression_ratio']:.2%}) - may lose detail")

    trace_result["analysis"] = analysis

    print(f"  Raw data: {analysis['raw_data_chars']} chars")
    print(f"  Evidence: {analysis['evidence_chars']} chars")
    print(f"  Answer: {analysis['answer_chars']} chars")
    if analysis["compression_ratio"]:
        print(f"  Compression ratio: {analysis['compression_ratio']:.2%}")

    if analysis["potential_issues"]:
        print(f"  ⚠️ Potential issues:")
        for issue in analysis["potential_issues"]:
            print(f"    - {issue}")

    # =========================================================================
    # Print full answer for manual inspection
    # =========================================================================
    print("\n--- FULL ANSWER ---")
    print(response.answer)

    return trace_result


def run_all_traces():
    """Run traces for all complex queries."""
    print("="*80)
    print("V7 AGENT DATA FLOW TRACER")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*80)

    all_traces = []

    for query_spec in TRACE_QUERIES:
        trace = trace_single_query(query_spec)
        all_traces.append(trace)
        print("\n" + "="*80)

    # Save detailed trace
    trace_file = os.path.join(os.path.dirname(__file__), "data_flow_trace.json")
    with open(trace_file, "w") as f:
        json.dump(all_traces, f, indent=2, default=str)
    print(f"\nDetailed trace saved to: {trace_file}")

    # Summary
    print("\n" + "="*80)
    print("TRACE SUMMARY")
    print("="*80)

    for trace in all_traces:
        qid = trace["query_id"]
        phases = trace["phases"]
        analysis = trace["analysis"]

        print(f"\n[{qid}]")
        print(f"  Query: {trace['query'][:60]}...")
        print(f"  Tools called: {len(phases['3_tool_execution']['tool_calls'])}")
        print(f"  Evidence items: {phases['4_evidence_accumulation']['evidence_count']}")
        print(f"  Raw data: {analysis['raw_data_chars']} chars -> Answer: {analysis['answer_chars']} chars")

        if analysis["potential_issues"]:
            for issue in analysis["potential_issues"]:
                print(f"  ⚠️ {issue}")


if __name__ == "__main__":
    run_all_traces()
