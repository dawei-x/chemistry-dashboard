"""
Pipeline Data Flow Tracer for V7 Agent

Traces exactly what data flows through each phase:
1. Tool execution -> raw tool output
2. Evidence formatting for context (decision phase)
3. Evidence formatting for synthesis
4. Final LLM input

Run from server directory:
    python -m agent_v7.tests.trace_pipeline
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_v7.tools_v2 import execute_tool, CORE_TOOLS


# Import the formatting functions directly from react_agent module
# We'll recreate them here to avoid LLM initialization
MAX_EVIDENCE_ITEMS = 20


def format_evidence_for_context(evidence: list) -> str:
    """Format evidence for decision-making context (concise).
    Copy of the function from react_agent.py for tracing.
    """
    if not evidence:
        return ""

    lines = []
    for e in evidence[-MAX_EVIDENCE_ITEMS:]:
        if e.get("type") == "steering_block":
            lines.append(f"[BLOCKED] {e.get('tool')}: {e.get('reason')}")
        else:
            tool = e.get("tool", "unknown")
            result = e.get("result", {})

            if result.get("error"):
                lines.append(f"[{tool}] Error: {result.get('error')}")
            elif tool == "get_transcript":
                transcript = result.get("transcript", "")
                count = transcript.count("\n") + 1 if transcript else 0
                session_name = result.get("session_name", "")
                lines.append(f"[{tool}] Session '{session_name}': {count} utterances")
            elif tool == "get_concept_map":
                summary = result.get("summary", {})
                session_name = result.get("session_name", "")
                lines.append(f"[{tool}] Session '{session_name}': {summary.get('total_nodes', 0)} nodes, {summary.get('total_edges', 0)} edges")
            elif tool == "get_7c_analysis":
                dimensions = result.get("dimensions", {})
                scores = [d.get("score", 0) for d in dimensions.values() if d.get("score")]
                overall = sum(scores) / len(scores) if scores else 0
                session_name = result.get("session_name", "")
                lines.append(f"[{tool}] Session '{session_name}': Average {overall:.1f}/100")
            elif tool == "search_sessions":
                sessions = result.get("sessions", [])
                lines.append(f"[{tool}] Found {len(sessions)} sessions")
            elif tool == "list_sessions":
                sessions = result.get("sessions", [])
                lines.append(f"[{tool}] {len(sessions)} sessions available")
            else:
                lines.append(f"[{tool}] Completed")

    return "\n".join(lines)


def format_evidence_for_synthesis(evidence: list) -> str:
    """Format evidence for synthesis (detailed).
    Copy of the function from react_agent.py for tracing.
    """
    if not evidence:
        return "No evidence gathered."

    sections = []

    for e in evidence:
        if e.get("type") == "steering_block":
            continue

        tool = e.get("tool", "unknown")
        result = e.get("result", {})

        if result.get("error"):
            sections.append(f"## {tool}\nError: {result.get('error')}")
            continue

        section = [f"## {tool}"]

        if tool == "get_transcript":
            session_name = result.get("session_name", "")
            section.append(f"Session: {session_name}")
            transcript = result.get("transcript", "")
            if transcript:
                lines = transcript.split("\n")[:30]
                section.append("Transcript:")
                section.extend(lines)
            else:
                section.append("No transcript content available.")

        elif tool == "get_concept_map":
            session_name = result.get("session_name", "")
            section.append(f"Session: {session_name}")
            summary = result.get("summary", {})
            section.append(f"Nodes: {summary.get('total_nodes', 0)}, Edges: {summary.get('total_edges', 0)}")

            if summary.get("speaker_contributions"):
                section.append("Speaker contributions:")
                for speaker, data in summary["speaker_contributions"].items():
                    section.append(f"  {speaker}: {data.get('total', 0)} contributions")

            graph = result.get("graph", "")
            if graph:
                section.append("Concept graph:")
                graph_lines = graph.split("\n")[:40]
                section.extend(graph_lines)

        elif tool == "get_7c_analysis":
            session_name = result.get("session_name", "")
            section.append(f"Session: {session_name}")
            dimensions = result.get("dimensions", {})

            scores = [d.get("score", 0) for d in dimensions.values() if d.get("score")]
            if scores:
                overall = sum(scores) / len(scores)
                section.append(f"Average Score: {overall:.1f}/100")

            for dim_name, dim_data in dimensions.items():
                score = dim_data.get("score", 0)
                explanation = dim_data.get("explanation", "")[:200]
                section.append(f"  {dim_name}: {score}/100 - {explanation}")

                coded = dim_data.get("coded_segments", [])[:3]
                for seg in coded:
                    if isinstance(seg, dict):
                        quote = seg.get("quote", "")[:150]
                        reason = seg.get("reason", "")[:100]
                        section.append(f"    Quote: \"{quote}\"")
                        if reason:
                            section.append(f"    Reason: {reason}")

        elif tool == "list_sessions":
            sessions = result.get("sessions", [])
            section.append(f"Total: {len(sessions)} sessions available")
            for s in sessions:
                speakers = ", ".join(s.get("speakers", [])[:5]) or "Unknown"
                section.append(f"  - Session {s.get('session_id')}: {s.get('session_name', '')} (Speakers: {speakers})")

        elif tool == "search_sessions":
            sessions = result.get("sessions", [])
            section.append(f"Found {len(sessions)} relevant sessions:")
            for s in sessions[:5]:
                session_name = s.get('session_name', s.get('name', ''))
                section.append(f"  - Session {s.get('session_id')}: {session_name}")

        sections.append("\n".join(section))

    return "\n\n".join(sections)


def trace_tool_output(tool_name: str, params: dict) -> dict:
    """Execute a tool and capture its raw output."""
    print(f"\n{'='*80}")
    print(f"PHASE 1: RAW TOOL OUTPUT")
    print(f"Tool: {tool_name}")
    print(f"Params: {json.dumps(params, indent=2)}")
    print(f"{'='*80}")

    result = execute_tool(tool_name, params)

    print(f"\nRaw result type: {type(result)}")
    print(f"Raw result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    print(f"\nFull raw output:")
    output_str = json.dumps(result, indent=2, default=str)
    print(output_str[:5000])

    if len(output_str) > 5000:
        print(f"\n... (truncated, full length: {len(output_str)} chars)")

    return result


def trace_evidence_formatting(evidence: list):
    """Trace how evidence is formatted for context and synthesis."""

    print(f"\n{'='*80}")
    print(f"PHASE 2: EVIDENCE FORMATTED FOR CONTEXT (Decision Phase)")
    print(f"{'='*80}")

    context_formatted = format_evidence_for_context(evidence)
    print(f"\nContext format (what LLM sees during tool decision):")
    print("-" * 40)
    print(context_formatted)
    print("-" * 40)

    print(f"\n{'='*80}")
    print(f"PHASE 3: EVIDENCE FORMATTED FOR SYNTHESIS")
    print(f"{'='*80}")

    synthesis_formatted = format_evidence_for_synthesis(evidence)
    print(f"\nSynthesis format (what LLM sees when generating response):")
    print("-" * 40)
    print(synthesis_formatted[:4000])
    if len(synthesis_formatted) > 4000:
        print(f"\n... (truncated, full length: {len(synthesis_formatted)} chars)")
    print("-" * 40)

    return context_formatted, synthesis_formatted


def trace_full_pipeline(query: str, tool_name: str, params: dict):
    """Trace the full pipeline for a query."""

    print(f"\n{'#'*80}")
    print(f"FULL PIPELINE TRACE")
    print(f"Query: {query}")
    print(f"Tool: {tool_name}")
    print(f"{'#'*80}")

    # Phase 1: Raw tool output
    raw_output = trace_tool_output(tool_name, params)

    # Build evidence structure (as react_agent does)
    evidence = [{
        "tool": tool_name,
        "params": params,
        "result": raw_output
    }]

    # Phase 2 & 3: Format evidence
    context_fmt, synthesis_fmt = trace_evidence_formatting(evidence)

    # Phase 4: Show what the synthesis prompt would look like
    print(f"\n{'='*80}")
    print(f"PHASE 4: FULL SYNTHESIS PROMPT STRUCTURE")
    print(f"{'='*80}")

    user_message = f"""Based on the evidence gathered, provide a scaffolded response to this query:

Query: {query}

Evidence:
{synthesis_fmt}

Instructions:
1. Point to SPECIFIC evidence (exact quotes, coded segments, concept nodes)
2. Explain WHY the evidence is relevant
3. Use natural language ("You can see this in...", "Notice how...")
4. If evidence is incomplete, acknowledge what couldn't be determined
5. Suggest related artifacts the user might want to explore

Write a conversational response that guides the user through the evidence."""

    print(f"\n--- USER MESSAGE THAT WOULD BE SENT TO LLM ---")
    print(user_message[:3000])
    if len(user_message) > 3000:
        print(f"\n... (total length: {len(user_message)} chars)")

    return {
        "raw_output": raw_output,
        "context_formatted": context_fmt,
        "synthesis_formatted": synthesis_fmt,
        "user_message_length": len(user_message)
    }


def verify_key_data_presence(trace_result: dict, tool_name: str):
    """Verify that key data is present in the formatted output."""

    print(f"\n{'='*80}")
    print(f"VERIFICATION: Key Data Presence Check")
    print(f"{'='*80}")

    raw = trace_result["raw_output"]
    synthesis = trace_result["synthesis_formatted"]

    issues = []

    if tool_name == "get_transcript":
        transcript = raw.get("transcript", "")
        if transcript:
            lines = transcript.split("\n")
            if lines:
                first_line = lines[0]
                print(f"\nFirst line of raw transcript: '{first_line}'")

                # Extract speaker name from format: [MM:SS] Speaker: text
                if "]" in first_line:
                    speaker_part = first_line.split("]")[1].strip()
                    if ":" in speaker_part:
                        speaker_name = speaker_part.split(":")[0].strip()
                        print(f"✓ First speaker in raw data: '{speaker_name}'")

                        if speaker_name in synthesis:
                            print(f"✓ Speaker '{speaker_name}' IS in synthesis format")
                        else:
                            print(f"✗ Speaker '{speaker_name}' NOT FOUND in synthesis format")
                            issues.append(f"Speaker '{speaker_name}' missing from synthesis")

            # Check all unique speakers
            speakers = set()
            for line in lines:
                if "]" in line and ":" in line:
                    parts = line.split("]")
                    if len(parts) > 1:
                        speaker_part = parts[1].strip()
                        if ":" in speaker_part:
                            speakers.add(speaker_part.split(":")[0].strip())

            print(f"\nAll speakers in transcript: {speakers}")
            for speaker in speakers:
                if speaker in synthesis:
                    print(f"  ✓ '{speaker}' found in synthesis")
                else:
                    print(f"  ✗ '{speaker}' NOT in synthesis")
                    issues.append(f"Speaker '{speaker}' missing")

        session_name = raw.get("session_name", "")
        if session_name:
            print(f"\nSession name in raw: '{session_name}'")
            if session_name in synthesis:
                print(f"✓ Session name IS in synthesis format")
            else:
                print(f"✗ Session name NOT in synthesis format")
                issues.append("Session name missing")

    elif tool_name == "get_7c_analysis":
        dimensions = raw.get("dimensions", {})
        print(f"\nDimensions in raw output: {list(dimensions.keys())}")

        for dim_name, dim_data in dimensions.items():
            score = dim_data.get("score")
            coded = dim_data.get("coded_segments", [])

            print(f"\n  {dim_name}:")
            print(f"    Score: {score}")
            print(f"    Coded segments: {len(coded)}")

            if dim_name in synthesis:
                print(f"    ✓ Dimension name in synthesis")
            else:
                print(f"    ✗ Dimension name NOT in synthesis")
                issues.append(f"{dim_name} missing")

            if str(score) in synthesis:
                print(f"    ✓ Score '{score}' in synthesis")
            else:
                print(f"    ✗ Score NOT in synthesis")

            if coded:
                first_seg = coded[0]
                if isinstance(first_seg, dict):
                    quote = first_seg.get("quote", "")
                    print(f"    First quote: '{quote[:80]}...'")

                    # Check if quote content appears in synthesis
                    if quote[:50] in synthesis:
                        print(f"    ✓ Quote content in synthesis")
                    else:
                        print(f"    ✗ Quote content NOT in synthesis")

    elif tool_name == "get_concept_map":
        graph = raw.get("graph", "")
        summary = raw.get("summary", {})

        print(f"\nGraph length: {len(graph)} chars")
        print(f"Total nodes: {summary.get('total_nodes', 0)}")
        print(f"Total edges: {summary.get('total_edges', 0)}")

        speaker_contribs = summary.get("speaker_contributions", {})
        print(f"Speaker contributions: {list(speaker_contribs.keys())}")

        for speaker in speaker_contribs.keys():
            if speaker in synthesis:
                print(f"  ✓ '{speaker}' in synthesis")
            else:
                print(f"  ✗ '{speaker}' NOT in synthesis")
                issues.append(f"Speaker '{speaker}' missing from concept map")

        # Check first few graph lines
        if graph:
            graph_lines = graph.split("\n")[:5]
            print(f"\nFirst 5 graph lines:")
            for line in graph_lines:
                print(f"  {line[:80]}")

    print(f"\n{'='*80}")
    if issues:
        print(f"ISSUES FOUND: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ ALL KEY DATA PRESENT IN FORMATTED OUTPUT")
    print(f"{'='*80}")

    return issues


if __name__ == "__main__":
    # Test 1: Speaker query (transcript)
    print("\n" + "#"*80)
    print("TEST 1: TRANSCRIPT TOOL - Speaker Query")
    print("#"*80)

    result1 = trace_full_pipeline(
        "In the Abundance session, who are the speakers?",
        "get_transcript",
        {"session_id": 25}
    )
    issues1 = verify_key_data_presence(result1, "get_transcript")

    # Test 2: 7C Analysis
    print("\n" + "#"*80)
    print("TEST 2: 7C ANALYSIS TOOL - Collaboration Query")
    print("#"*80)

    result2 = trace_full_pipeline(
        "What was the collaboration quality in session 25?",
        "get_7c_analysis",
        {"session_id": 25}
    )
    issues2 = verify_key_data_presence(result2, "get_7c_analysis")

    # Test 3: Concept map
    print("\n" + "#"*80)
    print("TEST 3: CONCEPT MAP TOOL - Ideas Query")
    print("#"*80)

    result3 = trace_full_pipeline(
        "What ideas were discussed in the Abundance session?",
        "get_concept_map",
        {"session_id": 25}
    )
    issues3 = verify_key_data_presence(result3, "get_concept_map")

    # Summary
    print("\n" + "#"*80)
    print("FINAL SUMMARY")
    print("#"*80)
    print(f"Test 1 (transcript): {'PASS' if not issues1 else 'FAIL - ' + str(len(issues1)) + ' issues'}")
    print(f"Test 2 (7C analysis): {'PASS' if not issues2 else 'FAIL - ' + str(len(issues2)) + ' issues'}")
    print(f"Test 3 (concept map): {'PASS' if not issues3 else 'FAIL - ' + str(len(issues3)) + ' issues'}")

    total_issues = len(issues1) + len(issues2) + len(issues3)
    if total_issues == 0:
        print("\n✓ PIPELINE DATA FLOW VERIFIED - All data correctly passed through")
    else:
        print(f"\n✗ PIPELINE HAS {total_issues} ISSUES - Data may be lost in formatting")
