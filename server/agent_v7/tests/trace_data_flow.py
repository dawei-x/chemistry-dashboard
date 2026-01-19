#!/usr/bin/env python3
"""
Data Flow Tracing Script for V7.2 Agent
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_v7.tools_v2 import execute_tool
from agent_v7.react_agent import ScaffoldingAgent


def trace_query(query: str, output_file: str = None):
    """Trace full data flow for a query."""

    print(f"\n{'='*80}")
    print(f"TRACING DATA FLOW FOR QUERY:")
    print(f"  \"{query}\"")
    print(f"{'='*80}\n")

    conversation_id = f"trace-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    agent = ScaffoldingAgent(conversation_id)

    tool_results = []
    original_execute = execute_tool

    def traced_execute(tool_name, params):
        result = original_execute(tool_name, params)
        tool_results.append({
            'tool': tool_name,
            'params': params,
            'result_keys': list(result.keys()) if isinstance(result, dict) else type(result).__name__,
            'display_length': len(result.get('display', '')) if isinstance(result, dict) else 0,
            'full_display': result.get('display', '') if isinstance(result, dict) else str(result),
        })
        return result

    import agent_v7.tools_v2 as tools_module
    import agent_v7.react_agent as react_module
    original_exec = tools_module.execute_tool
    tools_module.execute_tool = traced_execute
    react_module.execute_tool = traced_execute

    try:
        response = agent.respond(query)
    finally:
        tools_module.execute_tool = original_exec
        react_module.execute_tool = original_exec

    # Phase 1: Tool Results
    print(f"\n{'='*80}")
    print("PHASE 1: TOOL EXECUTION RESULTS (RAW)")
    print(f"{'='*80}")
    print(f"Total tool calls: {len(tool_results)}")

    for i, tr in enumerate(tool_results):
        print(f"\n--- Tool Call {i+1}: {tr['tool']} ---")
        print(f"Params: {json.dumps(tr['params'], indent=2)}")
        print(f"Display length: {tr['display_length']} chars")
        print(f"\nFULL DISPLAY CONTENT:")
        print("-" * 40)
        print(tr['full_display'])
        print("-" * 40)

    # Phase 2: Context Evidence
    print(f"\n{'='*80}")
    print("PHASE 2: EVIDENCE FOR CONTEXT (Decision-making)")
    print(f"{'='*80}")

    context_evidence = agent._format_evidence_for_context(response.evidence)
    print(f"Formatted length: {len(context_evidence)} chars")
    print(f"\nFULL CONTEXT EVIDENCE:")
    print("-" * 40)
    print(context_evidence)
    print("-" * 40)

    # Phase 3: Synthesis Evidence
    print(f"\n{'='*80}")
    print("PHASE 3: EVIDENCE FOR SYNTHESIS (Response generation)")
    print(f"{'='*80}")

    synthesis_evidence = agent._format_evidence_for_synthesis(response.evidence)
    print(f"Formatted length: {len(synthesis_evidence)} chars")
    print(f"\nFULL SYNTHESIS EVIDENCE:")
    print("-" * 40)
    print(synthesis_evidence)
    print("-" * 40)

    # Phase 4: Final Answer
    print(f"\n{'='*80}")
    print("PHASE 4: FINAL ANSWER")
    print(f"{'='*80}")
    print(f"Answer length: {len(response.answer)} chars")
    print(f"\nFULL ANSWER:")
    print("-" * 40)
    print(response.answer)
    print("-" * 40)

    # Summary
    print(f"\n{'='*80}")
    print("DATA FLOW SUMMARY")
    print(f"{'='*80}")
    print(f"Tools called: {[tr['tool'] for tr in tool_results]}")
    total_raw = sum(tr['display_length'] for tr in tool_results)
    print(f"Total raw display data: {total_raw} chars")
    print(f"Context evidence: {len(context_evidence)} chars")
    print(f"Synthesis evidence: {len(synthesis_evidence)} chars")
    print(f"Final answer: {len(response.answer)} chars")

    if total_raw > 0:
        print(f"\n--- Data Preservation Analysis ---")
        print(f"Synthesis/Raw ratio: {len(synthesis_evidence)/total_raw*100:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trace_data_flow.py 'query text'")
        sys.exit(1)
    trace_query(sys.argv[1])
