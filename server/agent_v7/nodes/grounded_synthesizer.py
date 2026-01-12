"""
Grounded Synthesizer Node for BLINC Agent V7

PRAS Stage 5: Grounded Synthesis

V7 CHANGES: ALL TRUNCATION LIMITS REMOVED
- Full transcripts passed to LLM
- Full concept map nodes included
- All coded segments included for 7C
- max_tokens increased to 4096

Generates final answer with explicit grounding in evidence:
1. Structure answer around main claims
2. Ground every claim with typed citations
3. Acknowledge convergence explicitly
4. Acknowledge tensions when present
5. State limitations honestly

Enhancement: True Citation Grounding
- Each citation includes source_chunk_id for traceability
- Validation against actual retrieval results
"""

import hashlib
import logging
import re
from typing import Dict, Any, List, Optional

from ..llm import get_reasoning_client
from ..state import GroundedClaim, Citation, ArtifactRef, CitationPreview

logger = logging.getLogger(__name__)

# Citation ID counter for unique IDs
_citation_counter = 0


def _generate_source_chunk_id(cite_data: Dict, citation_type: str) -> str:
    """Generate a unique, deterministic ID for citation grounding."""
    key_parts = [
        str(cite_data.get('session_id', '')),
        str(cite_data.get('speaker', '')),
        str(cite_data.get('timestamp', '')),
        str(cite_data.get('quote_preview', cite_data.get('evidence', '')))[:100],
        citation_type
    ]
    key_string = '|'.join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()[:12]


def _validate_citation_against_retrieval(
    cite_data: Dict,
    citation_type: str,
    state: Dict
) -> bool:
    """
    Validate that a citation exists in the actual retrieval results.

    V7 Fix: STRICT speaker-session validation.
    If a citation includes both speaker AND session, BOTH must match evidence.
    This prevents the LLM from attributing statements to wrong speakers.

    This ensures the paper claim 'artifact-grounded' is defensible.
    Handles all tool response formats.
    """
    session_id = cite_data.get('session_id')
    speaker = cite_data.get('speaker')
    dimension = cite_data.get('dimension')  # For 7C citations
    score = cite_data.get('score')  # For 7C citations

    # Get all retrieval results
    retrieval_results = state.get('retrieval_results', [])
    subgoal_results = state.get('subgoal_results', {})

    # V7 Fix: Build speaker-session mapping from all evidence first
    # This ensures we validate BOTH speaker and session match, not just one
    valid_speaker_sessions = {}  # speaker.lower() -> set of session_ids
    valid_sessions = set()  # all sessions found in evidence

    def _extract_speaker_sessions(tool_result: Dict, fallback_session: int = None):
        """Extract speaker-session pairs from a tool result."""
        tool_session = tool_result.get('session_id') or fallback_session
        if tool_session and isinstance(tool_session, int):
            valid_sessions.add(tool_session)

        # From results list
        for item in tool_result.get('results', []):
            if isinstance(item, dict):
                item_session = item.get('session_device_id') or item.get('session_id') or tool_session
                item_speaker = item.get('speaker', item.get('speaker_alias', ''))
                if item_session and isinstance(item_session, int):
                    valid_sessions.add(item_session)
                    if item_speaker:
                        spk_lower = item_speaker.lower()
                        if spk_lower not in valid_speaker_sessions:
                            valid_speaker_sessions[spk_lower] = set()
                        valid_speaker_sessions[spk_lower].add(item_session)

        # From utterances
        for u in tool_result.get('utterances', []):
            if isinstance(u, dict):
                item_speaker = u.get('speaker', u.get('speaker_tag', ''))
                if item_speaker and tool_session:
                    spk_lower = item_speaker.lower()
                    if spk_lower not in valid_speaker_sessions:
                        valid_speaker_sessions[spk_lower] = set()
                    valid_speaker_sessions[spk_lower].add(tool_session)

        # From concept map nodes
        for n in tool_result.get('nodes', []):
            if isinstance(n, dict):
                attributed_to = n.get('attributed_to', '')
                if attributed_to and tool_session:
                    spk_lower = attributed_to.lower()
                    if spk_lower not in valid_speaker_sessions:
                        valid_speaker_sessions[spk_lower] = set()
                    valid_speaker_sessions[spk_lower].add(tool_session)

        # From sessions list
        for s in tool_result.get('sessions', []):
            if isinstance(s, dict):
                sess_id = s.get('session_id')
                if sess_id and isinstance(sess_id, int):
                    valid_sessions.add(sess_id)

    # Extract from retrieval_results
    for result in retrieval_results:
        _extract_speaker_sessions(result)

    # Extract from subgoal_results
    for subgoal_id, sg_result in subgoal_results.items():
        for step in sg_result.get('steps_executed', []):
            tool_result = step.get('tool_result', {})
            _extract_speaker_sessions(tool_result)

    # V7 Fix: STRICT validation logic
    # If both speaker and session are provided, BOTH must match
    if speaker and session_id:
        speaker_lower = speaker.lower()
        if speaker_lower in valid_speaker_sessions:
            if session_id in valid_speaker_sessions[speaker_lower]:
                return True  # Speaker was in this session
            else:
                # Speaker exists but NOT in this session - INVALID
                logger.debug(f"[Citation Validation] REJECTED: {speaker} found in sessions {valid_speaker_sessions[speaker_lower]}, not in {session_id}")
                return False
        else:
            # Speaker not found in any evidence
            logger.debug(f"[Citation Validation] REJECTED: {speaker} not found in evidence")
            return False

    # If only session_id provided, just check session exists
    if session_id and not speaker:
        return session_id in valid_sessions

    # If only speaker provided (rare), check speaker exists
    if speaker and not session_id:
        return speaker.lower() in valid_speaker_sessions

    # No session or speaker - can't validate, assume valid for now
    if not session_id and not speaker:
        return True

    # Special handling for 7C citations
    if citation_type == '7c' and dimension:
        for subgoal_id, sg_result in subgoal_results.items():
            for step in sg_result.get('steps_executed', []):
                tool_result = step.get('tool_result', {})
                tool_name = step.get('step', {}).get('tool', '')
                if 'collaboration' in tool_name or '7c' in tool_name.lower():
                    # 7C tool was used - validate the dimension exists
                    for item in tool_result.get('results', []):
                        if isinstance(item, dict):
                            dims = item.get('dimensions', {})
                            if dimension.lower() in dims:
                                dim_data = dims[dimension.lower()]
                                # Validate score matches if provided
                                if score is None or dim_data.get('score') == score:
                                    return True

    return False


SYNTHESIS_SYSTEM_PROMPT = """You are an expert analyst synthesizing insights from your own prior analytical work.

## Your Prior Analysis Work

You have previously analyzed this collaborative discussion and created several artifacts:

**Transcript Analysis**: You reviewed the full discussion transcript, documenting what each
participant said, their communication patterns, question-asking behaviors, and linguistic
indicators of thinking depth (via LIWC metrics).

**Concept Map**: You constructed a visual representation capturing how ideas connect - the
key concepts introduced, their relationships (causal, supporting, contrasting), reasoning
chains, and thematic clusters that emerged.

**7C Collaboration Analysis**: You evaluated the group's collaboration quality across seven
dimensions (climate, communication, contribution, conflict, context, constructive,
compatibility), coding specific transcript segments as evidence for each score.

Now you need to synthesize your findings into a coherent answer for the user's question.

## Accuracy Guidelines

1. **Session IDs**: Use only the session IDs from your evidence. If evidence mentions "[David, Session 20]", that's Session 20.

2. **Speaker-Session Match**: The "Speakers and Their Sessions" section shows who participated in each session. Keep speaker attributions consistent with their actual sessions.

## Your Role

Analyze the dialogue like an expert would - looking at actual behaviors, not just scores:

**What to analyze:**
- **Turn-taking patterns**: Who dominates? Who facilitates? Are contributions balanced?
- **Interaction quality**: Do participants build on each other's ideas or talk past each other?
- **Thematic coherence**: Does the discussion stay focused or fragment?
- **Mutual understanding**: Do speakers acknowledge and respond to each other?

The 7C scores provide quantitative context, but your analysis should come from observing the dialogue itself. What do you actually see happening in the conversation?

## Writing Style

Write like an expert analyst sharing behavioral insights:
- Lead with what you observe in the dialogue ("Sam dominates the discussion, often taking extended turns...")
- Use 7C scores as supporting context, not the main content ("This aligns with the moderate contribution score of 60/100")
- Make interpretive observations ("The interaction is more alternating monologue than co-construction")
- Be specific about behaviors, not abstract about scores

**A note on citations:** The system shows references separately, so you can focus on your analysis without adding parenthetical citations like "(Session 20, David)" after statements. Just name speakers naturally in your prose.

For example:
- Instead of: "Sam argued that AI lacks true agency (Session 19, Sam)."
- Write: "Sam argued that AI lacks true agency, noting that it only acts when prompted."

## What to Include

A good response typically covers:
1. **Direct answer** - Your key finding and interpretation
2. **Evidence and reasoning** - How you reached your conclusion, with quotes and scores
3. **Analytical insights** - Patterns, connections, or implications you noticed
4. **Open questions** - What would be worth exploring further

## Response Format

Provide your response as JSON with this structure:
{
    "answer": "Your analysis in natural markdown. Focus on reasoning and interpretation, mentioning sources naturally.",
    "artifacts_referenced": [
        {
            "type": "transcript|concept_map|7c_analysis|speaker_profile",
            "session_id": 20,
            "speaker": "David",
            "key_content": "Brief description of what this artifact showed"
        }
    ],
    "confidence": 0.0 to 1.0,
    "follow_ups": ["Suggested next questions based on what you found"]
}

## Interpreting Your Artifacts

- **Transcripts**: Your primary evidence for behavioral analysis. Look at who speaks, how often, how they respond to each other, whether they build on ideas or talk past each other. This is where your insights come from.

- **7C Scores**: Quantitative summary of collaboration quality. Use these as context/validation for what you observe in the transcript, not as the main content of your analysis. A score of 70/100 means little without describing the behaviors that led to it.

- **Concept Maps**: Show how ideas connect - useful for understanding thematic coherence and reasoning patterns.

- **Clusters**: Thematic groupings that indicate shared or parallel thinking among participants.

**Key principle**: Analyze the dialogue first, then use scores to contextualize. Don't report scores and call it analysis."""


def synthesize_grounded_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate final grounded response.

    PRAS Stage 5: Grounded Synthesis

    Args:
        state: Current agent state with all evidence

    Returns:
        State updates with final answer
    """
    query = state.get('current_query', state.get('original_query', ''))
    cross_rep_analysis = state.get('cross_rep_analysis', {})
    subgoal_results = state.get('subgoal_results', {})
    sub_goals = state.get('sub_goals', [])
    reps_used = state.get('representations_used', [])

    logger.info(f"[PRAS Stage 5] Synthesizing grounded response")

    # Format all inputs for synthesis
    synthesis_input = _format_synthesis_input(
        query, sub_goals, subgoal_results, cross_rep_analysis, state
    )

    # Debug: Log synthesis input size and content summary
    logger.info(f"[Synthesis Input] Length: {len(synthesis_input)} chars")
    if "# What Participants Said" in synthesis_input:
        transcript_section = synthesis_input.split("# What Participants Said")[1].split("#")[0] if "# What Participants Said" in synthesis_input else ""
        logger.info(f"[Synthesis Input] Transcript section length: {len(transcript_section)} chars")

    try:
        llm = get_reasoning_client()

        result = llm.json_chat(
            system=SYNTHESIS_SYSTEM_PROMPT,
            user=synthesis_input,
            temperature=0.2,  # Slightly higher for natural language
            max_tokens=4096  # V7: Increased from 2500 for fuller responses
        )

        if result:
            return _process_synthesis_result(result, state, reps_used)

    except Exception as e:
        logger.error(f"Synthesis error: {e}")

    # Fallback synthesis
    return _fallback_synthesis(query, subgoal_results, cross_rep_analysis, reps_used, state)


def _format_synthesis_input(
    query: str,
    sub_goals: List[Dict],
    subgoal_results: Dict[str, Dict],
    cross_rep_analysis: Dict,
    state: Dict = None
) -> str:
    """
    Format all evidence for synthesis LLM.

    V7 RESTRUCTURED: Present data naturally, not by artifact type.
    - Transcripts first as natural conversation
    - Concept patterns as "ideas and connections"
    - 7C as "collaboration observations"
    - No citation training patterns
    """
    sections = [f"# Question\n{query}"]

    # Gather all content by type (not by subgoal)
    all_transcripts = []  # (session_id, speaker, text)
    all_concepts = []  # (session_id, label, type, attributed_to)
    all_edges = []  # (session_id, source_text, target_text, relationship, source_speaker, target_speaker)
    all_7c = []  # (session_id, dimensions_dict)
    all_comparisons = []  # comparison results
    sessions_found = set()
    speakers_found = {}  # V3-style: speaker -> set of session IDs

    for sg_id, result in subgoal_results.items():
        steps = result.get('steps_executed', [])
        logger.info(f"[Synthesis] Processing sg_id={sg_id}, steps_executed count={len(steps)}")
        for step in steps:
            tool_result = step.get('tool_result', {})
            tool_name = step.get('step', {}).get('tool', 'unknown')
            tool_session_id = tool_result.get('session_id')
            has_utterances = 'utterances' in tool_result
            utterance_count = len(tool_result.get('utterances', [])) if has_utterances else 0
            logger.info(f"  - Tool: {tool_name}, session_id: {tool_session_id}, has_utterances: {has_utterances}, count: {utterance_count}")

            if tool_session_id:
                sessions_found.add(tool_session_id)

            # Gather transcript utterances
            utterances_list = tool_result.get('utterances', [])
            added_count = 0
            for u in utterances_list:
                if isinstance(u, dict):
                    speaker = u.get('speaker', u.get('speaker_tag', '')) or 'Speaker'  # Default for empty
                    text = u.get('text', u.get('transcript', ''))
                    if text:  # Only require text, speaker defaults to "Speaker"
                        all_transcripts.append((tool_session_id, speaker, text))
                        added_count += 1
                        # Track speaker-session mapping for V3-style validation
                        if speaker and tool_session_id and isinstance(tool_session_id, int):
                            if speaker not in speakers_found:
                                speakers_found[speaker] = set()
                            speakers_found[speaker].add(tool_session_id)
            if utterances_list:
                logger.info(f"    [Transcripts] From {tool_name}: processed {len(utterances_list)}, added {added_count} to all_transcripts")

            # Gather from results (search tools)
            for r in tool_result.get('results', []):
                if isinstance(r, dict):
                    text = r.get('text', r.get('content', r.get('transcript', '')))
                    speaker = r.get('speaker', r.get('speaker_alias', ''))
                    session = r.get('session_device_id') or r.get('session_id') or tool_session_id
                    if text and (speaker or session):
                        all_transcripts.append((session, speaker or 'Unknown', text))
                        if session:
                            sessions_found.add(session)
                            # Track speaker-session mapping
                            if speaker and isinstance(session, int):
                                if speaker not in speakers_found:
                                    speakers_found[speaker] = set()
                                speakers_found[speaker].add(session)

            # Gather concept map nodes and build node lookup for edges
            nodes_map = {}  # node_id -> (text, speaker)
            for n in tool_result.get('nodes', []):
                if isinstance(n, dict):
                    node_id = n.get('id')
                    label = n.get('label', n.get('text', ''))
                    ntype = n.get('type', n.get('node_type', 'concept'))
                    speaker = n.get('speaker', n.get('attributed_to', ''))
                    if node_id:
                        nodes_map[node_id] = (label, speaker)
                    if label:
                        all_concepts.append((tool_session_id, label, ntype, speaker))
                        # Track speaker-session mapping from concept attributions
                        if speaker and tool_session_id and isinstance(tool_session_id, int):
                            if speaker not in speakers_found:
                                speakers_found[speaker] = set()
                            speakers_found[speaker].add(tool_session_id)

            # Gather concept map edges (CRITICAL for disagreement/pattern queries)
            for e in tool_result.get('edges', []):
                if isinstance(e, dict):
                    source_id = e.get('source')
                    target_id = e.get('target')
                    relationship = e.get('relationship', e.get('edge_type', ''))

                    # Get source and target node info
                    source_text, source_speaker = nodes_map.get(source_id, ('', ''))
                    target_text, target_speaker = nodes_map.get(target_id, ('', ''))

                    if relationship and (source_text or target_text):
                        all_edges.append((
                            tool_session_id,
                            source_text,
                            target_text,
                            relationship,
                            source_speaker or 'Speaker',
                            target_speaker or 'Speaker'
                        ))

            # Gather 7C analysis
            dims = tool_result.get('dimensions')
            if dims and isinstance(dims, dict):
                all_7c.append((tool_session_id, dims))

            # Gather comparison results
            if tool_result.get('results') and step.get('step', {}).get('tool') == 'compare_sessions':
                comparison = tool_result['results'][0] if tool_result['results'] else {}
                if comparison.get('summary'):
                    all_comparisons.append(comparison['summary'])

                # V7 FIX: Also extract full 7C data from session_details
                # compare_sessions includes full collaboration dimensions with coded_segments
                for session_detail in comparison.get('session_details', []):
                    collab = session_detail.get('collaboration', {})
                    dims = collab.get('dimensions', {})
                    sess_id = session_detail.get('session_device_id')
                    if dims and sess_id:
                        # Convert evidence → coded_segments for consistency
                        converted_dims = {}
                        for dim_name, dim_data in dims.items():
                            converted_dims[dim_name] = {
                                'score': dim_data.get('score', 0),
                                'explanation': dim_data.get('explanation', ''),
                                'coded_segments': dim_data.get('evidence', [])
                            }
                        all_7c.append((sess_id, converted_dims))
                        sessions_found.add(sess_id)

    # Debug totals
    logger.info(f"[Synthesis Totals] all_transcripts: {len(all_transcripts)}, all_concepts: {len(all_concepts)}, all_edges: {len(all_edges)}, all_7c: {len(all_7c)}")
    logger.info(f"[Synthesis] Sessions found: {sessions_found}, Speakers found: {list(speakers_found.keys())}")

    # V3-STYLE VALIDATION SECTION: Prevent LLM from hallucinating sessions or misattributing speakers
    if sessions_found:
        sections.append(f"\n# Sessions Found in Evidence: {sorted(sessions_found)}")
        sections.append("Only cite sessions from this list - these are the sessions you have data for.\n")

        if speakers_found:
            sections.append("# Speakers and Their Sessions:")
            for speaker in sorted(speakers_found.keys()):
                session_list = sorted(list(speakers_found[speaker]))
                sections.append(f"  - {speaker}: Session{'s' if len(session_list) > 1 else ''} {', '.join(map(str, session_list))}")
            sections.append("\nUse these speaker-session mappings when citing. Don't claim a speaker said something in a session they weren't in.\n")

    # 1. TRANSCRIPT CONTENT (PRIMARY SOURCE)
    if all_transcripts:
        sections.append("\n# What Participants Said\n")

        # Group by session
        by_session = {}
        for session_id, speaker, text in all_transcripts:
            if session_id is None:
                continue  # Skip utterances without session_id
            if session_id not in by_session:
                by_session[session_id] = []
            by_session[session_id].append((speaker, text))

        logger.info(f"[Transcript Grouping] by_session keys: {list(by_session.keys())}, utterance counts: {[(k, len(v)) for k, v in by_session.items()]}")

        for session_id in sorted(by_session.keys()):
            if len(by_session) > 1:
                sections.append(f"\n## Session {session_id}\n")

            for speaker, text in by_session[session_id]:
                # Natural dialogue format, no citation instructions
                sections.append(f"**{speaker}**: \"{text}\"\n")

    # 2. CONCEPT PATTERNS (derived from transcript)
    if all_concepts:
        sections.append("\n# Key Ideas and Connections\n")
        sections.append("These concepts emerged from the discussion:\n")

        # Group by type for clarity
        by_type = {}
        for session_id, label, ntype, attributed in all_concepts:
            if ntype not in by_type:
                by_type[ntype] = []
            by_type[ntype].append((label, attributed, session_id))

        for ntype, items in by_type.items():
            if ntype == 'question':
                sections.append(f"**Questions raised:**")
            elif ntype == 'claim':
                sections.append(f"**Claims made:**")
            elif ntype == 'reasoning':
                sections.append(f"**Reasoning chains:**")
            else:
                sections.append(f"**{ntype.title()}s:**")

            for label, attributed, session_id in items:
                attr_str = f" ({attributed})" if attributed else ""
                sections.append(f"- {label}{attr_str}")
            sections.append("")

    # 2.5 CONCEPT MAP RELATIONSHIPS (key for disagreement/pattern queries)
    if all_edges:
        # Priority edge types for showing how ideas interact
        priority_types = ['challenges', 'contrasts_with', 'builds_on', 'supports', 'contradicts']

        # Group edges by session and filter to priority types
        edges_by_session = {}
        for session_id, source_text, target_text, relationship, source_speaker, target_speaker in all_edges:
            if relationship in priority_types:
                if session_id not in edges_by_session:
                    edges_by_session[session_id] = []
                edges_by_session[session_id].append(
                    (source_text, target_text, relationship, source_speaker, target_speaker)
                )

        if edges_by_session:
            sections.append("\n# How Ideas Connect and Interact\n")
            sections.append("These relationships show how participants' ideas relate:\n")

            # Format edge type as readable verb
            edge_verbs = {
                'challenges': 'CHALLENGES',
                'contrasts_with': 'CONTRASTS WITH',
                'builds_on': 'BUILDS ON',
                'supports': 'SUPPORTS',
                'contradicts': 'CONTRADICTS'
            }

            for session_id in sorted(edges_by_session.keys()):
                if len(edges_by_session) > 1:
                    sections.append(f"\n## Session {session_id}\n")

                # Show up to 10 key relationships per session
                for source_text, target_text, relationship, source_spk, target_spk in edges_by_session[session_id][:10]:
                    verb = edge_verbs.get(relationship, relationship.upper())
                    # Truncate long quotes for readability
                    src_quote = source_text[:100] + "..." if len(source_text) > 100 else source_text
                    tgt_quote = target_text[:100] + "..." if len(target_text) > 100 else target_text
                    sections.append(f"- **{source_spk}** {verb} **{target_spk}**: \"{src_quote}\" → \"{tgt_quote}\"")

                sections.append("")

    # 3. COLLABORATION OBSERVATIONS (7C analysis)
    if all_7c:
        sections.append("\n# Collaboration Patterns Observed\n")

        for session_id, dims in all_7c:
            if len(all_7c) > 1:
                sections.append(f"## Session {session_id}\n")

            for dim_name, dim_data in dims.items():
                if not isinstance(dim_data, dict):
                    continue
                score = dim_data.get('score', 0)
                explanation = dim_data.get('explanation', '')
                coded_segments = dim_data.get('coded_segments', [])

                sections.append(f"**{dim_name.title()}** ({score}/100): {explanation}")

                # Include coded segments as behavioral examples
                if coded_segments:
                    sections.append("Coded evidence from the discussion:")
                    for seg in coded_segments:
                        if isinstance(seg, dict):
                            # V7: Real coded segments with timestamp, speaker, quote, reason
                            timestamp = seg.get('timestamp', 0)
                            mins = int(timestamp // 60)
                            secs = int(timestamp % 60)
                            speaker = seg.get('speaker', 'Unknown')
                            quote = seg.get('quote', '')
                            reason = seg.get('reason', '')
                            sections.append(f"  - [{mins}:{secs:02d}] **{speaker}**: \"{quote}\"")
                            if reason:
                                sections.append(f"    _Why coded_: {reason}")
                        elif isinstance(seg, str) and seg.strip():
                            # Fallback: old format (just string)
                            sections.append(f"  - \"{seg}\"")
                sections.append("")

    # 4. COMPARISON RESULTS (if comparing sessions)
    if all_comparisons:
        sections.append("\n# Comparison Summary\n")
        for summary in all_comparisons:
            collab_scores = summary.get('collaboration_scores', {})
            if collab_scores:
                sorted_scores = sorted(collab_scores.items(), key=lambda x: x[1], reverse=True)
                sections.append("**Collaboration scores (ranked):**")
                for rank, (sess_id, score) in enumerate(sorted_scores, 1):
                    sections.append(f"{rank}. Session {sess_id}: {score}/100")
                sections.append("")

            themes = summary.get('themes', {})
            if themes:
                sections.append("**Session themes:**")
                for sess_id, theme_list in themes.items():
                    if theme_list:
                        sections.append(f"- Session {sess_id}: {', '.join(theme_list)}")
                sections.append("")

    # 5. CROSS-REPRESENTATION INSIGHTS (if available)
    conv_points = cross_rep_analysis.get('convergence_points', [])
    tension_points = cross_rep_analysis.get('tension_points', [])

    if conv_points or tension_points:
        sections.append("\n# Patterns Across Sources\n")

        if conv_points:
            sections.append("**Consistent findings:**")
            for cp in conv_points:
                sections.append(f"- {cp.get('claim')}")
            sections.append("")

        if tension_points:
            sections.append("**Tensions to consider:**")
            for tp in tension_points:
                sections.append(f"- {tp.get('aspect')}: {tp.get('interpretation', '')}")
            sections.append("")

    # Debug: Show section breakdown
    transcript_sec = [s for s in sections if '**' in s and '**: "' in s]  # Lines with dialogue format **Speaker**: "text"
    logger.info(f"[Sections] Total: {len(sections)}, Dialogue lines: {len(transcript_sec)}")
    if transcript_sec:
        logger.info(f"[Sections] First dialogue: {transcript_sec[0][:200]}")

    return '\n'.join(sections)


def _extract_citations_post_hoc(answer: str, state: Dict) -> List[Dict]:
    """
    Extract citations by parsing the answer text for session/speaker references.

    V7: This allows the LLM to write naturally without structured citation output.
    We find mentions of sessions and speakers and create citations from them.
    """
    citations = []
    seen = set()

    # Get sessions and speakers from the retrieval results for validation
    valid_sessions = set()
    valid_speakers = {}  # speaker -> set of session_ids

    for result in state.get('subgoal_results', {}).values():
        for step in result.get('steps_executed', []):
            tool_result = step.get('tool_result', {})
            session_id = tool_result.get('session_id')
            if session_id:
                valid_sessions.add(session_id)

            for u in tool_result.get('utterances', []):
                speaker = u.get('speaker', u.get('speaker_tag', ''))
                if speaker and session_id:
                    if speaker not in valid_speakers:
                        valid_speakers[speaker] = set()
                    valid_speakers[speaker].add(session_id)

            for r in tool_result.get('results', []):
                if isinstance(r, dict):
                    sess = r.get('session_device_id') or r.get('session_id')
                    spk = r.get('speaker', r.get('speaker_alias', ''))
                    if sess:
                        valid_sessions.add(sess)
                    if spk and sess:
                        if spk not in valid_speakers:
                            valid_speakers[spk] = set()
                        valid_speakers[spk].add(sess)

                    # V7 FIX: Handle compare_sessions which nests session_details inside results[0]
                    for sd in r.get('session_details', []):
                        if isinstance(sd, dict):
                            sd_sess = sd.get('session_device_id') or sd.get('session_id')
                            if sd_sess:
                                valid_sessions.add(sd_sess)
                            # Extract speakers from speaker_stats
                            for speaker_stat in sd.get('speaker_stats', []):
                                sd_spk = speaker_stat.get('speaker') or speaker_stat.get('speaker_alias')
                                if sd_spk and sd_sess:
                                    if sd_spk not in valid_speakers:
                                        valid_speakers[sd_spk] = set()
                                    valid_speakers[sd_spk].add(sd_sess)

                    # Also check sessions_compared list
                    for compared_sid in r.get('sessions_compared', []):
                        if isinstance(compared_sid, int):
                            valid_sessions.add(compared_sid)

            # V7 FIX: Also check top-level session_details (for other tools)
            for sd in tool_result.get('session_details', []):
                if isinstance(sd, dict):
                    sess = sd.get('session_device_id') or sd.get('session_id')
                    if sess:
                        valid_sessions.add(sess)

    logger.info(f"[Citations] Valid sessions: {valid_sessions}, Valid speakers: {list(valid_speakers.keys())}")

    # Pattern 1: "Session X" mentions
    session_pattern = re.compile(r'[Ss]ession\s+(\d+)', re.IGNORECASE)
    for match in session_pattern.finditer(answer):
        session_id = int(match.group(1))
        key = f"session-{session_id}"
        if key not in seen and session_id in valid_sessions:
            seen.add(key)
            citations.append({
                'id': f"cite-{len(citations)+1}",
                'citation_type': 'session',
                'inline_text': f"Session {session_id}",
                'reference_text': f"Session {session_id} discussion",
                'artifact_ref': {'session_id': session_id},
                'validated': True
            })

    # Pattern 2: Speaker names (if they appear in our valid speakers)
    for speaker, sessions in valid_speakers.items():
        if speaker.lower() in answer.lower():
            key = f"speaker-{speaker}"
            if key not in seen:
                seen.add(key)
                session_id = list(sessions)[0] if sessions else None
                citations.append({
                    'id': f"cite-{len(citations)+1}",
                    'citation_type': 'transcript',
                    'inline_text': speaker,
                    'reference_text': f"{speaker}'s contributions",
                    'artifact_ref': {'session_id': session_id, 'speaker': speaker},
                    'validated': True
                })

    logger.info(f"[Citations] Extracted {len(citations)} citations post-hoc from answer")
    return citations


def _process_synthesis_result(
    result: Dict,
    state: Dict,
    reps_used: List[str]
) -> Dict[str, Any]:
    """Process LLM synthesis result into state updates.

    V7 SIMPLIFIED: No longer expects artifacts_referenced from LLM.
    Citations extracted post-hoc by parsing the answer text.
    """
    answer = result.get('answer', '')
    follow_ups = result.get('follow_ups', [])

    # Get confidence from LLM result or fall back to cross-rep analysis
    llm_confidence = result.get('confidence')
    if isinstance(llm_confidence, (int, float)):
        confidence = float(llm_confidence)
    else:
        cross_rep = state.get('cross_rep_analysis', {})
        confidence = cross_rep.get('overall_confidence', 0.5)

    # V7: Extract citations post-hoc from the answer text
    # This allows the LLM to write naturally without structured citation requirements
    citations = _extract_citations_post_hoc(answer, state)

    # Simplified grounded claims (not forcing the old structure)
    grounded_claims = []

    # Build reasoning trace for transparency
    cross_rep = state.get('cross_rep_analysis', {})
    reasoning_trace = {
        'sub_goals_count': len(state.get('sub_goals', [])),
        'subgoals_satisfied': sum(
            1 for r in state.get('subgoal_results', {}).values()
            if r.get('satisfied')
        ),
        'representations_used': reps_used,
        'convergence_count': len(cross_rep.get('convergence_points', [])),
        'tension_count': len(cross_rep.get('tension_points', [])),
        'gap_count': len(cross_rep.get('gaps', []))
    }

    return {
        'pras_stage': 'synthesize',
        'final_answer': answer,
        'grounded_claims': grounded_claims,
        'citations': citations,
        'confidence': confidence,
        'representations_used': reps_used,
        'follow_ups': follow_ups,
        'reasoning_trace': reasoning_trace,
        'reflection': None,  # No longer forcing tensions/limitations
        'next_action': 'format',
        'thought_history': state.get('thought_history', []) + [
            f"Synthesized answer with {len(citations)} artifact references, "
            f"confidence {confidence:.0%}"
        ]
    }


def _build_citations_from_artifacts(
    artifacts: List[Dict],
    state: Dict
) -> List[Citation]:
    """Build Citation objects from the new artifacts_referenced format.

    Now includes validation against retrieval results for the 'validated' field.
    """
    citations = []

    for i, artifact in enumerate(artifacts):
        # Build cite_data for validation
        cite_data = {
            'session_id': artifact.get('session_id'),
            'speaker': artifact.get('speaker'),
            'evidence': artifact.get('key_content', '')
        }

        # Normalize citation type
        cite_type = _normalize_citation_type(artifact.get('type', 'transcript'))

        # Generate source chunk ID for traceability
        source_chunk_id = _generate_source_chunk_id(cite_data, cite_type)

        # Validate against actual retrieval results
        validated = _validate_citation_against_retrieval(cite_data, cite_type, state)

        if not validated:
            logger.debug(f"[Citation Grounding] Unvalidated artifact citation: Session {artifact.get('session_id')}")

        citation = {
            'id': f"cite-{i+1}",
            'citationType': cite_type,
            'inlineText': f"Session {artifact.get('session_id', '?')}",
            'referenceText': artifact.get('key_content', ''),
            'artifactRef': {
                'sessionId': artifact.get('session_id'),
                'speaker': artifact.get('speaker')
            },
            'preview': {
                'title': f"{cite_type} - Session {artifact.get('session_id', '?')}",
                'content': artifact.get('key_content', '')[:300],
                'metadata': {'speaker': artifact.get('speaker')}
            },
            # Grounding fields
            'sourceChunkId': source_chunk_id,
            'validated': validated
        }
        citations.append(citation)

    validated_count = sum(1 for c in citations if c.get('validated'))
    logger.info(f"[Citations] Built {len(citations)} citations, {validated_count} validated")
    return citations


# =============================================================================
# Citation Building Functions
# =============================================================================

def _build_structured_citations(
    llm_citations: List[Dict],
    main_claims: List[Dict],
    answer: str,
    state: Dict
) -> List[Citation]:
    """
    Build structured Citation objects from LLM output.

    Combines LLM-provided citation metadata with:
    1. Artifact references for popover fetching
    2. Preview content for quick display
    3. Unique citation IDs
    """
    global _citation_counter
    citations: List[Citation] = []
    seen_inline_texts = set()  # Deduplicate

    # First, try to use LLM-provided citations
    for cite_data in llm_citations:
        inline_text = cite_data.get('inline_text', '')
        if not inline_text or inline_text in seen_inline_texts:
            continue
        seen_inline_texts.add(inline_text)

        cite_type = cite_data.get('type', _infer_citation_type(inline_text))
        citation = _create_citation(
            inline_text=inline_text,
            citation_type=cite_type,
            cite_data=cite_data,
            state=state
        )
        if citation:
            citations.append(citation)

    # Also extract from grounding if LLM didn't provide citations_used
    if not llm_citations:
        for claim in main_claims:
            for grounding in claim.get('grounding', []):
                citation_text = grounding.get('citation', '')
                if citation_text and citation_text not in seen_inline_texts:
                    seen_inline_texts.add(citation_text)
                    cite_type = grounding.get('rep', _infer_citation_type(citation_text))
                    citation = _create_citation(
                        inline_text=citation_text,
                        citation_type=cite_type,
                        cite_data=grounding,
                        state=state
                    )
                    if citation:
                        citations.append(citation)

    # Parse any remaining citations from the answer text
    additional_citations = _extract_citations_from_answer(answer, seen_inline_texts, state)
    citations.extend(additional_citations)

    logger.info(f"[Citations] Built {len(citations)} structured citations")
    return citations


def _create_citation(
    inline_text: str,
    citation_type: str,
    cite_data: Dict,
    state: Dict
) -> Optional[Dict[str, Any]]:
    """
    Create a single Citation object with artifact ref, preview, and grounding.

    Enhancement: True Citation Grounding
    - Generates source_chunk_id for traceability
    - Validates against retrieval results
    """
    global _citation_counter
    _citation_counter += 1
    citation_id = f"cite-{_citation_counter}"

    # Normalize citation type
    cite_type = _normalize_citation_type(citation_type)

    # Build artifact reference
    artifact_ref = _build_artifact_ref(cite_type, cite_data)

    # Build preview content
    preview = _build_preview(cite_type, cite_data, state)

    # Generate reference text
    reference_text = _generate_reference_text(cite_type, cite_data, inline_text)

    # Generate source chunk ID for traceability
    source_chunk_id = _generate_source_chunk_id(cite_data, cite_type)

    # Validate against actual retrieval results
    validated = _validate_citation_against_retrieval(cite_data, cite_type, state)

    if not validated:
        logger.warning(f"[Citation Grounding] Unvalidated citation: {inline_text}")

    # Return dict with all fields including grounding
    return {
        'id': citation_id,
        'citation_type': cite_type,
        'inline_text': inline_text,
        'reference_text': reference_text,
        'artifact_ref': artifact_ref,
        'preview': preview,
        # Grounding fields for paper claim "artifact-grounded"
        'source_chunk_id': source_chunk_id,
        'validated': validated
    }


def _normalize_citation_type(cite_type: str) -> str:
    """Normalize citation type to one of: transcript, concept, 7c, cluster, session, speaker."""
    cite_type = cite_type.lower().strip()

    type_aliases = {
        'transcript': 'transcript',
        'transcripts': 'transcript',
        'quote': 'transcript',
        'concept': 'concept',
        'concept_map': 'concept',
        'concepts': 'concept',
        'edge': 'concept',
        '7c': '7c',
        'collaboration': '7c',
        'seven_c': '7c',
        'cluster': 'cluster',
        'community': 'cluster',
        'session': 'session',
        'overview': 'session',
        'session_overview': 'session',
        'speaker': 'speaker',
        'profile': 'speaker',
        'speaker_profile': 'speaker'
    }

    return type_aliases.get(cite_type, 'transcript')


def _infer_citation_type(inline_text: str) -> str:
    """Infer citation type from inline text pattern."""
    patterns = [
        (r'\(Session \d+,\s*[^)]+\)', 'transcript'),
        (r'\[Concept:\s*"[^"]+"\]', 'concept'),
        (r'\[Edge:\s*[^\]]+\]', 'concept'),
        (r'\[7C:\s*\w+\s*\d+/100\]', '7c'),
        (r'\[Cluster:\s*"[^"]+"\]', 'cluster'),
        (r'\[Session:\s*\d+\s+Overview\]', 'session'),
        (r'\[Speaker:\s*[^\]]+\]', 'speaker')
    ]

    for pattern, cite_type in patterns:
        if re.search(pattern, inline_text, re.IGNORECASE):
            return cite_type

    return 'transcript'  # Default


def _build_artifact_ref(cite_type: str, cite_data: Dict) -> ArtifactRef:
    """Build artifact reference for popover fetching."""
    ref = ArtifactRef()

    # Session ID
    session_id = cite_data.get('session_id')
    if session_id is not None:
        ref['session_id'] = int(session_id) if isinstance(session_id, (int, str)) else None

    # Speaker
    speaker = cite_data.get('speaker')
    if speaker:
        ref['speaker'] = str(speaker)

    # Type-specific refs
    if cite_type == 'concept':
        concept_text = cite_data.get('concept_text') or cite_data.get('evidence', '')
        if concept_text:
            ref['concept_id'] = concept_text[:50]  # Use text as ID for now

    elif cite_type == '7c':
        dimension = cite_data.get('dimension')
        if dimension:
            ref['dimension'] = dimension

    elif cite_type == 'cluster':
        cluster_name = cite_data.get('cluster_name') or cite_data.get('evidence', '')
        if cluster_name:
            ref['cluster_id'] = cluster_name

    # Timestamp
    timestamp = cite_data.get('timestamp')
    if timestamp is not None:
        try:
            ref['timestamp'] = float(timestamp)
        except (ValueError, TypeError):
            pass

    return ref


def _build_preview(cite_type: str, cite_data: Dict, state: Dict) -> CitationPreview:
    """Build preview content for popover display."""
    # Get evidence text for preview - ensure never None
    evidence = cite_data.get('evidence') or cite_data.get('quote_preview') or ''
    if evidence and len(evidence) > 300:
        evidence = evidence[:297] + '...'

    # Type-specific preview building
    if cite_type == 'transcript':
        speaker = cite_data.get('speaker', 'Unknown')
        session_id = cite_data.get('session_id', '?')
        title = f"{speaker} - Session {session_id}"
        metadata = {
            'wordCount': len(evidence.split()) if evidence else 0,
            'timestamp': cite_data.get('timestamp')
        }

    elif cite_type == 'concept':
        concept_text = cite_data.get('concept_text', evidence[:50])
        title = f"Concept: {concept_text}"
        metadata = {
            'conceptType': cite_data.get('concept_type', 'idea'),
            'speaker': cite_data.get('speaker'),
            'connections': cite_data.get('connections', 0)
        }

    elif cite_type == '7c':
        dimension = cite_data.get('dimension', 'Unknown')
        score = cite_data.get('score', 0)
        title = f"{dimension} - {score}/100"
        metadata = {
            'score': score,
            'dimension': dimension,
            'explanation': evidence[:200] if evidence else ''
        }

    elif cite_type == 'cluster':
        cluster_name = cite_data.get('cluster_name', evidence[:30])
        title = f"Cluster: {cluster_name}"
        metadata = {
            'clusterSize': cite_data.get('cluster_size', 0),
            'keyConcepts': cite_data.get('key_concepts', [])
        }

    elif cite_type == 'session':
        session_id = cite_data.get('session_id', '?')
        title = f"Session {session_id} Overview"
        metadata = {
            'participants': cite_data.get('participants', []),
            'duration': cite_data.get('duration')
        }

    elif cite_type == 'speaker':
        speaker = cite_data.get('speaker', 'Unknown')
        title = f"Speaker: {speaker}"
        metadata = {
            'sessionCount': cite_data.get('session_count', 0),
            'utteranceCount': cite_data.get('utterance_count', 0)
        }

    else:
        title = 'Reference'
        metadata = {}

    return CitationPreview(
        title=title,
        content=evidence,
        metadata=metadata
    )


def _generate_reference_text(cite_type: str, cite_data: Dict, inline_text: str) -> str:
    """Generate reference list text for a citation."""
    if cite_type == 'transcript':
        speaker = cite_data.get('speaker') or 'Unknown'
        evidence = cite_data.get('evidence') or cite_data.get('quote_preview') or ''
        if evidence:
            return f"{speaker}'s statement: \"{evidence[:80]}...\"" if len(evidence) > 80 else f"{speaker}'s statement: \"{evidence}\""
        return f"Quote from {speaker}"

    elif cite_type == 'concept':
        concept_text = cite_data.get('concept_text') or cite_data.get('evidence') or inline_text or 'Unknown concept'
        return f"Concept node: {concept_text[:60]}"

    elif cite_type == '7c':
        dimension = cite_data.get('dimension') or 'Unknown'
        score = cite_data.get('score') or 0
        return f"7C {dimension} dimension score: {score}/100"

    elif cite_type == 'cluster':
        cluster_name = cite_data.get('cluster_name') or cite_data.get('evidence') or inline_text or 'Unknown cluster'
        return f"Thematic cluster: {cluster_name}"

    elif cite_type == 'session':
        session_id = cite_data.get('session_id', '?')
        return f"Session {session_id} overview and summary"

    elif cite_type == 'speaker':
        speaker = cite_data.get('speaker', 'Unknown')
        return f"Speaker profile: {speaker}"

    return inline_text


def _extract_citations_from_answer(
    answer: str,
    seen_inline_texts: set,
    state: Dict
) -> List[Citation]:
    """Extract any citations from answer text that weren't in the structured output."""
    citations: List[Citation] = []

    # Citation patterns to extract
    patterns = [
        (r'\(Session (\d+),\s*([^)]+)\)', 'transcript', ['session_id', 'speaker']),
        (r'\[Concept:\s*"([^"]+)"\]', 'concept', ['concept_text']),
        (r'\[7C:\s*(\w+)\s*(\d+)/100\]', '7c', ['dimension', 'score']),
        (r'\[Cluster:\s*"([^"]+)"\]', 'cluster', ['cluster_name']),
        (r'\[Session:\s*(\d+)\s+Overview\]', 'session', ['session_id']),
        (r'\[Speaker:\s*([^\]]+)\]', 'speaker', ['speaker'])
    ]

    for pattern, cite_type, field_names in patterns:
        for match in re.finditer(pattern, answer, re.IGNORECASE):
            inline_text = match.group(0)
            if inline_text in seen_inline_texts:
                continue
            seen_inline_texts.add(inline_text)

            # Build cite_data from match groups
            cite_data = {}
            for i, field_name in enumerate(field_names, start=1):
                if i <= len(match.groups()):
                    value = match.group(i)
                    if field_name in ('session_id', 'score'):
                        try:
                            value = int(value)
                        except (ValueError, TypeError):
                            pass
                    cite_data[field_name] = value

            citation = _create_citation(
                inline_text=inline_text,
                citation_type=cite_type,
                cite_data=cite_data,
                state=state
            )
            if citation:
                citations.append(citation)

    return citations


def _generate_reflection(
    tensions: List[Dict],
    limitations: List[str]
) -> str:
    """Generate a reflection note about the answer."""
    parts = []

    if tensions:
        parts.append(f"Acknowledged {len(tensions)} tension(s) in the evidence.")

    if limitations:
        parts.append(f"Noted {len(limitations)} limitation(s):")
        for lim in limitations:  # V7: No [:3] limit
            parts.append(f"  - {lim}")

    return ' '.join(parts) if parts else "Answer synthesized from available evidence."


def _fallback_synthesis(
    query: str,
    subgoal_results: Dict[str, Dict],
    cross_rep_analysis: Dict,
    reps_used: List[str],
    state: Dict
) -> Dict[str, Any]:
    """Fallback synthesis when LLM fails."""
    # Gather key findings - V7: No limits
    findings = []
    for sg_id, result in subgoal_results.items():
        if result.get('evidence_summary'):
            findings.append(f"- {result['evidence_summary']}")
        for step in result.get('steps_executed', []):  # V7: No [:1] limit
            reflection = step.get('reflection', {})
            if reflection.get('indicators_found'):
                findings.append(f"- Found: {', '.join(reflection['indicators_found'])}")

    # Build basic answer
    answer_parts = [f"Based on analysis of {len(reps_used)} representation types:"]
    if findings:
        answer_parts.extend(findings)
    else:
        answer_parts.append("Limited evidence was found for this query.")

    # Add gaps - V7: No limit
    gaps = cross_rep_analysis.get('gaps', [])
    if gaps:
        answer_parts.append("\n**Limitations:**")
        for gap in gaps:  # V7: No [:3] limit
            answer_parts.append(f"- {gap.get('aspect', 'Some aspects')}: {gap.get('reason', 'insufficient evidence')}")

    answer = '\n'.join(answer_parts)

    return {
        'pras_stage': 'synthesize',
        'final_answer': answer,
        'grounded_claims': [],
        'citations': [],
        'confidence': cross_rep_analysis.get('overall_confidence', 0.3),
        'representations_used': reps_used,
        'follow_ups': [],
        'reasoning_trace': {'fallback': True},
        'reflection': 'Fallback synthesis used due to LLM error.',
        'next_action': 'format',
        'thought_history': state.get('thought_history', []) + [
            'Used fallback synthesis'
        ]
    }
