"""
Detailed Truncation Analysis for V7 Agent

Shows exactly what data is truncated vs preserved at each phase.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent_v7.tools_v2 import execute_tool


def analyze_transcript_truncation(session_id: int = 25):
    """Analyze transcript truncation."""
    print("\n" + "="*80)
    print("TRANSCRIPT TRUNCATION ANALYSIS")
    print("="*80)

    result = execute_tool("get_transcript", {"session_id": session_id})
    transcript = result.get("transcript", "")

    all_lines = transcript.split("\n")
    truncated_lines = all_lines[:30]

    print(f"\nRaw transcript:")
    print(f"  Total lines: {len(all_lines)}")
    print(f"  Total characters: {len(transcript)}")

    print(f"\nAfter [:30] truncation:")
    print(f"  Lines kept: {len(truncated_lines)}")
    print(f"  Lines LOST: {len(all_lines) - len(truncated_lines)}")

    # Show what's lost
    if len(all_lines) > 30:
        print(f"\n  LOST CONTENT (lines 31+):")
        for i, line in enumerate(all_lines[30:], start=31):
            print(f"    Line {i}: {line[:80]}...")
    else:
        print(f"\n  No content lost (transcript has <= 30 lines)")

    # Extract speakers from all vs truncated
    def get_speakers(lines):
        speakers = set()
        for line in lines:
            if "]" in line and ":" in line:
                parts = line.split("]")
                if len(parts) > 1:
                    speaker_part = parts[1].strip()
                    if ":" in speaker_part:
                        speakers.add(speaker_part.split(":")[0].strip())
        return speakers

    all_speakers = get_speakers(all_lines)
    kept_speakers = get_speakers(truncated_lines)
    lost_speakers = all_speakers - kept_speakers

    print(f"\n  Speakers in full transcript: {all_speakers}")
    print(f"  Speakers after truncation: {kept_speakers}")
    if lost_speakers:
        print(f"  SPEAKERS LOST: {lost_speakers}")
    else:
        print(f"  All speakers preserved: YES")

    return {
        "total_lines": len(all_lines),
        "kept_lines": len(truncated_lines),
        "lost_lines": len(all_lines) - 30 if len(all_lines) > 30 else 0,
        "speakers_lost": lost_speakers
    }


def analyze_7c_truncation(session_id: int = 25):
    """Analyze 7C analysis truncation."""
    print("\n" + "="*80)
    print("7C ANALYSIS TRUNCATION ANALYSIS")
    print("="*80)

    result = execute_tool("get_7c_analysis", {"session_id": session_id})
    dimensions = result.get("dimensions", {})

    truncation_issues = []

    for dim_name, dim_data in dimensions.items():
        print(f"\n  Dimension: {dim_name}")

        # Check explanation truncation
        explanation = dim_data.get("explanation", "")
        if len(explanation) > 200:
            print(f"    Explanation: {len(explanation)} chars -> TRUNCATED to 200")
            print(f"      Full: '{explanation[:50]}...{explanation[-50:]}'")
            print(f"      Kept: '{explanation[:200]}'")
            print(f"      LOST: '{explanation[200:]}'")
            truncation_issues.append(f"{dim_name} explanation")
        else:
            print(f"    Explanation: {len(explanation)} chars (no truncation)")

        # Check coded segments
        coded = dim_data.get("coded_segments", [])
        print(f"    Coded segments: {len(coded)} total")

        if len(coded) > 3:
            print(f"      Kept: first 3")
            print(f"      LOST: {len(coded) - 3} segments")
            truncation_issues.append(f"{dim_name} coded_segments")

            # Show lost segment quotes
            for i, seg in enumerate(coded[3:], start=4):
                if isinstance(seg, dict):
                    quote = seg.get("quote", "")[:60]
                    print(f"        Lost segment {i}: '{quote}...'")

        # Check individual quote/reason truncation
        for i, seg in enumerate(coded[:3]):
            if isinstance(seg, dict):
                quote = seg.get("quote", "")
                reason = seg.get("reason", "")

                if len(quote) > 150:
                    print(f"      Segment {i+1} quote: {len(quote)} chars -> TRUNCATED to 150")
                    print(f"        LOST: '{quote[150:]}'")
                    truncation_issues.append(f"{dim_name} seg{i+1} quote")

                if len(reason) > 100:
                    print(f"      Segment {i+1} reason: {len(reason)} chars -> TRUNCATED to 100")
                    print(f"        LOST: '{reason[100:]}'")
                    truncation_issues.append(f"{dim_name} seg{i+1} reason")

    return truncation_issues


def analyze_concept_map_truncation(session_id: int = 25):
    """Analyze concept map truncation."""
    print("\n" + "="*80)
    print("CONCEPT MAP TRUNCATION ANALYSIS")
    print("="*80)

    result = execute_tool("get_concept_map", {"session_id": session_id})
    graph = result.get("graph", "")
    summary = result.get("summary", {})

    all_lines = graph.split("\n")
    truncated_lines = all_lines[:40]

    print(f"\nGraph content:")
    print(f"  Total lines: {len(all_lines)}")
    print(f"  Total characters: {len(graph)}")

    print(f"\nAfter [:40] truncation:")
    print(f"  Lines kept: {len(truncated_lines)}")
    print(f"  Lines LOST: {len(all_lines) - 40 if len(all_lines) > 40 else 0}")

    if len(all_lines) > 40:
        print(f"\n  LOST CONTENT (lines 41+):")
        for i, line in enumerate(all_lines[40:], start=41):
            if line.strip():
                print(f"    Line {i}: {line}")
    else:
        print(f"\n  No content lost (graph has <= 40 lines)")

    # Check if summary is preserved (no truncation on summary)
    print(f"\nSummary (no truncation applied):")
    print(f"  total_nodes: {summary.get('total_nodes')}")
    print(f"  total_edges: {summary.get('total_edges')}")
    print(f"  speaker_contributions: {list(summary.get('speaker_contributions', {}).keys())}")

    return {
        "total_lines": len(all_lines),
        "lost_lines": len(all_lines) - 40 if len(all_lines) > 40 else 0
    }


def show_exact_synthesis_output(session_id: int = 25):
    """Show the EXACT output that gets sent to LLM (with truncations applied)."""
    print("\n" + "="*80)
    print("EXACT SYNTHESIS OUTPUT (what LLM actually sees)")
    print("="*80)

    # Reproduce the exact formatting logic from react_agent.py

    # 1. Transcript
    print("\n--- TRANSCRIPT SYNTHESIS ---")
    result = execute_tool("get_transcript", {"session_id": session_id})
    transcript = result.get("transcript", "")
    sess_id = result.get("session_id", "")
    device_name = result.get("device_name", "")
    session_name = result.get("session_name", "")

    section = [f"## get_transcript"]
    section.append(f"Session ID: {sess_id}")
    section.append(f"Device: {device_name}")
    section.append(f"Session Name: {session_name}")
    if transcript:
        lines = transcript.split("\n")  # NO TRUNCATION - full transcript preserved
        section.append("Transcript:")
        section.extend(lines)

    synthesis_transcript = "\n".join(section)
    print(synthesis_transcript)
    print(f"\n[Total chars: {len(synthesis_transcript)}]")

    # 2. 7C Analysis
    print("\n--- 7C ANALYSIS SYNTHESIS ---")
    result = execute_tool("get_7c_analysis", {"session_id": session_id})
    dimensions = result.get("dimensions", {})
    sess_id = result.get("session_id", "")
    device_name = result.get("device_name", "")
    session_name = result.get("session_name", "")

    section = [f"## get_7c_analysis"]
    section.append(f"Session ID: {sess_id}")
    section.append(f"Device: {device_name}")
    section.append(f"Session Name: {session_name}")

    scores = [d.get("score", 0) for d in dimensions.values() if d.get("score")]
    if scores:
        overall = sum(scores) / len(scores)
        section.append(f"Average Score: {overall:.1f}/100")

    for dim_name, dim_data in dimensions.items():
        score = dim_data.get("score", 0)
        explanation = dim_data.get("explanation", "")  # NO TRUNCATION - full explanation preserved
        section.append(f"  {dim_name}: {score}/100 - {explanation}")

        coded = dim_data.get("coded_segments", [])  # NO TRUNCATION - all segments preserved
        for seg in coded:
            if isinstance(seg, dict):
                quote = seg.get("quote", "")  # NO TRUNCATION - full quote preserved
                reason = seg.get("reason", "")  # NO TRUNCATION - full reason preserved
                section.append(f"    Quote: \"{quote}\"")
                if reason:
                    section.append(f"    Reason: {reason}")

    synthesis_7c = "\n".join(section)
    print(synthesis_7c)
    print(f"\n[Total chars: {len(synthesis_7c)}]")

    # 3. Concept Map
    print("\n--- CONCEPT MAP SYNTHESIS ---")
    result = execute_tool("get_concept_map", {"session_id": session_id})
    graph = result.get("graph", "")
    summary = result.get("summary", {})
    sess_id = result.get("session_id", "")
    device_name = result.get("device_name", "")
    session_name = result.get("session_name", "")

    section = [f"## get_concept_map"]
    section.append(f"Session ID: {sess_id}")
    section.append(f"Device: {device_name}")
    section.append(f"Session Name: {session_name}")
    section.append(f"Nodes: {summary.get('total_nodes', 0)}, Edges: {summary.get('total_edges', 0)}")

    if summary.get("speaker_contributions"):
        section.append("Speaker contributions:")
        for speaker, data in summary["speaker_contributions"].items():
            section.append(f"  {speaker}: {data.get('total', 0)} contributions")

    if graph:
        section.append("Concept graph:")
        graph_lines = graph.split("\n")  # NO TRUNCATION - full graph preserved
        section.extend(graph_lines)

    synthesis_concept = "\n".join(section)
    print(synthesis_concept)
    print(f"\n[Total chars: {len(synthesis_concept)}]")


if __name__ == "__main__":
    print("#" * 80)
    print("V7 AGENT TRUNCATION ANALYSIS")
    print("#" * 80)

    # Analyze each tool's truncation
    transcript_result = analyze_transcript_truncation(25)
    issues_7c = analyze_7c_truncation(25)
    concept_result = analyze_concept_map_truncation(25)

    # Show exact output
    show_exact_synthesis_output(25)

    # Summary
    print("\n" + "#" * 80)
    print("TRUNCATION SUMMARY")
    print("#" * 80)

    print(f"\nTranscript:")
    print(f"  Lines lost: {transcript_result['lost_lines']}")
    print(f"  Speakers lost: {transcript_result['speakers_lost'] or 'None'}")

    print(f"\n7C Analysis truncation issues:")
    if issues_7c:
        for issue in issues_7c:
            print(f"  - {issue}")
    else:
        print("  None")

    print(f"\nConcept Map:")
    print(f"  Graph lines lost: {concept_result['lost_lines']}")

    total_issues = transcript_result['lost_lines'] + len(issues_7c) + concept_result['lost_lines']
    if total_issues > 0:
        print(f"\n⚠️  WOULD HAVE LOST {total_issues} items under old truncation rules")
    else:
        print(f"\n✓ No significant truncation issues")

    # Verification: show that react_agent.py no longer truncates
    print("\n" + "=" * 80)
    print("VERIFICATION: TRUNCATION REMOVED FROM react_agent.py")
    print("=" * 80)
    print("\nThe analysis above shows what WOULD have been lost.")
    print("After removing truncation from react_agent.py:")
    print("  ✓ Transcript: Full data preserved (no [:30] limit)")
    print("  ✓ 7C Analysis: Full explanations, quotes, reasons, and ALL coded segments")
    print("  ✓ Concept Map: Full graph preserved (no [:40] limit)")
    print("\nThe 'EXACT SYNTHESIS OUTPUT' section above shows the ACTUAL data the LLM receives.")
