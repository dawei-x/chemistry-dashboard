"""
Prompt Builder Module for Agent V6.

Constructs the system prompt that embeds V3's analytical intelligence:
- Epistemic hierarchy
- Construct operationalizations (when relevant)
- Triangulation framework
- Steering section (when provided)
- Hypothesis testing protocol (when in hypothesis mode)
- Grounding requirements
- Permission to reason beyond retrieval

This is where V3's intelligence lives - in the prompt, not the pipeline.
"""

from typing import List, Optional
from .query_analysis import QueryAnalysis
from .domain_knowledge import (
    EPISTEMIC_HIERARCHY,
    TRIANGULATION_FRAMEWORK,
    GROUNDING_REQUIREMENTS,
    BEYOND_RETRIEVAL,
    format_operationalization_for_prompt,
)

# =============================================================================
# BASE SYSTEM PROMPT
# =============================================================================

BASE_SYSTEM_PROMPT = """You are an expert learning analytics researcher analyzing educational discussions.

You have access to a database of discussion sessions with multiple data representations:
- **Transcripts**: What participants actually said (primary evidence)
- **Concept Maps**: Extracted structure of ideas and relationships
- **7C Collaboration Analysis**: Quantified interaction quality across 7 dimensions
- **Speaker Profiles**: Individual participation patterns

Your role is to help users understand these discussions through rigorous, evidence-based analysis.
"""

# =============================================================================
# HYPOTHESIS TESTING PROTOCOL
# =============================================================================

HYPOTHESIS_PROTOCOL = """
## Hypothesis Testing Mode

The user has proposed a hypothesis to test. Follow this protocol:

### 1. Understand the Claim
What specifically is being claimed? Restate it clearly.

### 2. Operationalize
- What would count as evidence FOR the claim?
- What would count as evidence AGAINST?
- What data sources would be most relevant?

### 3. Gather Evidence
Search for BOTH supporting and contradicting evidence.
Don't just confirm - actively look for counter-evidence.

### 4. Weigh the Evidence
- How much evidence supports? How strong is it?
- How much evidence contradicts? How strong is it?
- Are there gaps that matter?

### 5. Reach a Verdict
Based on the evidence balance:
- **Supported**: Clear evidence in favor, little against
- **Partially supported**: Mixed evidence, some qualifications needed
- **Not supported**: Little evidence in favor, or contradicting evidence
- **Insufficient evidence**: Cannot determine either way

### 6. Explain Your Reasoning
Why did you reach this verdict? What evidence was decisive?
What caveats or qualifications apply?

**Important**: Intellectual honesty matters more than agreement with the user.
If the evidence doesn't support their hypothesis, say so clearly.
"""

# =============================================================================
# STEERING SECTION
# =============================================================================

def _build_steering_section(analysis: QueryAnalysis) -> str:
    """Build the steering section of the prompt."""
    sections = []

    if analysis.prefer_representations:
        reps = ', '.join(analysis.prefer_representations)
        sections.append(f"""## Representation Focus
The user has asked you to focus on: **{reps}**

Prioritize these data sources in your analysis. You may still reference other sources
if they provide important context, but anchor your main claims in the preferred sources.
""")

    if analysis.exclude_representations:
        reps = ', '.join(analysis.exclude_representations)
        sections.append(f"""## Excluded Representations
The user has asked you to NOT use: **{reps}**

Do not call tools related to these representations. Do not reference these sources
in your analysis. Use only the remaining available representations.
""")

    return '\n'.join(sections)


# =============================================================================
# CONTEXT SECTION
# =============================================================================

def _build_context_section(analysis: QueryAnalysis) -> str:
    """Build context hints for the agent."""
    hints = []

    if analysis.session_ids:
        session_info = ', '.join(str(s) for s in analysis.session_ids)
        hints.append(f"- Session(s) identified: {session_info}")

    if analysis.session_names:
        names = ', '.join(analysis.session_names)
        hints.append(f"- Session name(s): {names}")

    if analysis.speaker_names:
        speakers = ', '.join(analysis.speaker_names)
        hints.append(f"- Speaker(s) mentioned: {speakers}")

    if not hints:
        return ""

    return """## Query Context
Based on the user's query, the following entities were identified:
""" + '\n'.join(hints) + """

Use this information to guide your initial tool calls.
"""


# =============================================================================
# MAIN PROMPT BUILDER
# =============================================================================

def build_system_prompt(analysis: QueryAnalysis) -> str:
    """
    Build the complete system prompt based on query analysis.

    The prompt embeds V3's analytical intelligence:
    - Always includes epistemic hierarchy and triangulation framework
    - Adds construct operationalizations when abstract concepts detected
    - Adds steering section when preferences/exclusions specified
    - Adds hypothesis protocol when in hypothesis mode
    - Always includes grounding requirements and beyond-retrieval permission
    """
    sections = [BASE_SYSTEM_PROMPT]

    # Always include core analytical framework
    sections.append(EPISTEMIC_HIERARCHY)
    sections.append(TRIANGULATION_FRAMEWORK)
    sections.append(GROUNDING_REQUIREMENTS)

    # Add operationalizations for detected constructs
    if analysis.constructs:
        operationalization_text = format_operationalization_for_prompt(analysis.constructs)
        if operationalization_text:
            sections.append(operationalization_text)

    # Add steering section if applicable
    steering_section = _build_steering_section(analysis)
    if steering_section:
        sections.append(steering_section)

    # Add mode-specific sections
    if analysis.mode == "test_hypothesis":
        sections.append(HYPOTHESIS_PROTOCOL)
    elif analysis.mode == "compare":
        sections.append("""
## Comparison Mode
The user wants to compare sessions or speakers. Structure your analysis to:
1. Establish clear dimensions of comparison
2. Gather equivalent evidence for each subject
3. Present similarities and differences systematically
4. Provide a summary assessment
""")
    elif analysis.mode == "trace":
        sections.append("""
## Trace Mode
The user wants to trace the evolution of ideas or concepts. Structure your analysis to:
1. Identify the starting point and endpoint
2. Follow the progression through the discussion
3. Note key transitions and developments
4. Show how the concept evolved or connected
""")

    # Add context hints
    context_section = _build_context_section(analysis)
    if context_section:
        sections.append(context_section)

    # Always allow reasoning beyond retrieval
    sections.append(BEYOND_RETRIEVAL)

    # Final instructions
    sections.append("""
## Response Guidelines
- Be concise but thorough
- Ground claims in evidence
- Distinguish data from interpretation
- Acknowledge limitations and gaps
- Provide actionable insights when possible
""")

    return '\n\n'.join(sections)


def build_user_message(query: str, conversation_history: Optional[List] = None) -> str:
    """
    Build the user message, incorporating conversation context if available.

    For multi-turn conversations, we might add context about what was discussed before.
    """
    # For now, just return the query
    # Multi-turn context is handled by including history in messages
    return query
