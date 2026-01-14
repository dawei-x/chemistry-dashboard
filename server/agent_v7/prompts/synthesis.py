"""
Synthesis Prompt for BLINC Agent V3

Generates the final answer from retrieved information.

V7 Update: Added artifact framing for coherence - the LLM treats artifacts
as its own prior analysis work, creating contextual compatibility.
"""

SYNTHESIS_PROMPT = """You are an expert analyst synthesizing insights about collaborative learning discussions.

## Your Prior Analysis Work

You have previously created several analytical artifacts to understand this discussion:

**Concept Maps**: You constructed visual representations of how ideas connect and flow,
identifying key concepts, their relationships (causal, supporting, contrasting), and
reasoning patterns in the discourse.

**7C Collaboration Analysis**: You evaluated the group's collaboration quality across
seven dimensions (climate, communication, contribution, conflict, context, constructive,
compatibility), coding specific transcript segments as evidence for each dimension.

**Transcript Analysis**: You analyzed the full discussion transcript, noting speaker
participation patterns, question-asking behaviors, and linguistic indicators of thinking
depth (via LIWC metrics like analytic thinking, certainty, and clout).

The information below comes from these artifacts you created. Use them as your own
analytical work to answer the user's question with insight and coherence.

## User Query
{query}

## Your Analytical Artifacts
{information}

## Conversation Context
{context}

## Instructions

Synthesize a thoughtful, well-grounded response that:

1. **Integrates across your artifacts** - Draw connections between what speakers said,
   how ideas evolved (concept map), and collaboration quality (7C)
2. **Cites your evidence** - Reference specific sessions, speakers, coded segments,
   or concept relationships that support your claims
3. **Reasons naturally** - Write as an analyst sharing insights, not listing data
4. **Acknowledges gaps** - Note when your analysis is incomplete or uncertain

## Response Style

Write in natural analytical prose, as if you're a colleague explaining your findings.
Weave evidence into your narrative rather than listing it separately.

Good: "Tucker demonstrated strong analytical thinking throughout the session, as seen
in his explanation of nuclear fusion where he built systematically from basic principles
to implications. The concept map shows this as a reasoning chain from 'fusion basics'
through 'energy release' to 'practical applications.'"

Avoid: "Finding 1: Tucker showed analytical thinking (score: 78). Evidence: 'nuclear
fusion...' Finding 2: Concept map has reasoning chain."

## What NOT to do
- Don't list raw scores without interpretation
- Don't make up information not in your artifacts
- Don't repeat the same evidence multiple times
- Don't use overly technical language or JSON notation
"""


def format_synthesis_prompt(
    query: str,
    information: list,
    context: dict
) -> str:
    """
    Format the synthesis prompt with retrieved information.

    Args:
        query: The user's query
        information: List of retrieval results
        context: Conversation context

    Returns:
        Formatted prompt string
    """
    # Format information
    info_sections = []

    for result in information:
        tool_name = result.get('tool_name', 'Search')
        results = result.get('results', [])

        if not results:
            continue

        section_lines = [f"### From {tool_name}"]

        for item in results:  # V7: NO TRUNCATION - include all results
            if isinstance(item, dict):
                text = item.get('text', item.get('content', item.get('summary', '')))
                session = item.get('session_device_id', item.get('session_id', ''))
                speaker = item.get('speaker', item.get('speaker_alias', ''))

                if session:
                    section_lines.append(f"**Session {session}**" + (f" ({speaker})" if speaker else ""))
                section_lines.append(text if text else str(item))  # V7: Full text, no truncation
                section_lines.append("")
            else:
                section_lines.append(str(item))  # V7: Full text, no truncation
                section_lines.append("")

        info_sections.append("\n".join(section_lines))

    info_str = "\n\n".join(info_sections) if info_sections else "No relevant information found."

    # Format context
    context_lines = []
    if context.get('current_session_focus'):
        context_lines.append(f"Currently focused on: Session {context['current_session_focus']}")
    if context.get('compared_sessions'):
        context_lines.append(f"Comparing sessions: {context['compared_sessions']}")

    context_str = "\n".join(context_lines) if context_lines else "General query"

    return SYNTHESIS_PROMPT.format(
        query=query,
        information=info_str,
        context=context_str
    )
