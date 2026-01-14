"""
Domain Knowledge Module for Agent V6.

Contains V3's analytical intelligence:
- Construct operationalizations (what to look for when analyzing abstract concepts)
- Epistemic hierarchy (how to weigh different evidence sources)
- Representation capabilities (what each data source provides)

This is the core intelligence that V3 embedded in pipeline stages,
now extracted for embedding in prompts.
"""

# =============================================================================
# CONSTRUCT OPERATIONALIZATIONS
# =============================================================================
# When users ask about abstract concepts, these define what to look for

CONSTRUCT_OPERATIONALIZATIONS = {
    "systems thinking": [
        "identifying causal relationships between concepts",
        "seeing interconnections across ideas",
        "understanding feedback loops",
        "considering multiple perspectives",
        "recognizing emergent patterns",
        "acknowledging complexity and uncertainty"
    ],
    "critical thinking": [
        "questioning assumptions",
        "evaluating evidence quality",
        "considering alternative viewpoints",
        "identifying logical flaws or fallacies",
        "distinguishing fact from opinion",
        "synthesizing information from multiple sources"
    ],
    "collaboration": [
        "building on others' ideas",
        "active listening indicators (referencing what others said)",
        "balanced turn-taking",
        "constructive disagreement",
        "shared problem-solving",
        "mutual respect in discourse"
    ],
    "engagement": [
        "question asking frequency",
        "contribution length and depth",
        "response to others' points",
        "sustained participation over time",
        "emotional investment in topics",
        "initiative in steering discussion"
    ],
    "creativity": [
        "novel idea generation",
        "making unexpected connections",
        "divergent thinking patterns",
        "challenging conventional assumptions",
        "hypothetical reasoning ('what if...')",
        "metaphorical or analogical thinking"
    ],
    "argumentation": [
        "claim-evidence-reasoning structure",
        "use of warrants and backing",
        "acknowledgment of counterarguments",
        "logical coherence",
        "appropriate hedging and qualification",
        "distinction between assertion and support"
    ],
    "knowledge building": [
        "progressive refinement of ideas",
        "integration of new information",
        "explicit revision of prior understanding",
        "synthesis across contributions",
        "collective advancement of understanding",
        "metacognitive awareness of learning"
    ],
    "productive discourse": [
        "topic coherence and development",
        "appropriate depth vs breadth",
        "balance of exploration and convergence",
        "effective use of questions",
        "constructive responses to disagreement",
        "progress toward shared understanding"
    ]
}

# =============================================================================
# EPISTEMIC HIERARCHY
# =============================================================================
# How to weigh evidence from different sources

EPISTEMIC_HIERARCHY = """
## Epistemic Hierarchy (weight evidence accordingly)

When weighing evidence, respect this hierarchy of source reliability:

1. **TRANSCRIPT** (primary) - What was actually said. Ground truth.
   - Direct quotes with speaker attribution
   - Temporal sequence of contributions
   - Exact wording and phrasing
   - Use for: specific claims about what someone said or did

2. **CONCEPT_MAP** (derived) - Extracted structure of ideas. Shows reasoning patterns.
   - Concepts and their relationships
   - Idea clusters and themes
   - Reasoning chains and connections
   - Use for: understanding intellectual structure, idea evolution

3. **COLLABORATION/7C** (interpreted) - Quantified interaction quality.
   - Dimension scores (Climate, Communication, Contribution, etc.)
   - Overall collaboration quality assessment
   - Use for: comparing sessions, identifying patterns

4. **SPEAKER_PROFILE** (aggregated) - Patterns across sessions.
   - Cross-session participation patterns
   - Typical contribution style
   - Use for: characterizing individuals, not specific events

5. **SESSION_OVERVIEW** (summary) - High-level context.
   - Topic, duration, participant count
   - General characterization
   - Use for: orientation, not specific claims

**Principle**: Higher-ranked sources should anchor substantive claims.
Lower-ranked sources support, contextualize, or suggest patterns to investigate.
"""

# =============================================================================
# REPRESENTATION CAPABILITIES
# =============================================================================
# What each data representation can tell you

REPRESENTATION_CAPABILITIES = {
    "transcript": {
        "provides": [
            "exact quotes with speaker attribution",
            "temporal sequence of discussion",
            "discourse markers and linguistic patterns",
            "question-answer exchanges",
            "turn-taking dynamics"
        ],
        "use_for": [
            "grounding claims in specific evidence",
            "analyzing discourse patterns",
            "identifying who said what",
            "tracing argument development"
        ],
        "limitations": [
            "requires interpretation of meaning",
            "may miss non-verbal communication",
            "volume can be overwhelming"
        ]
    },
    "concept_map": {
        "provides": [
            "key concepts extracted from discussion",
            "relationships between concepts (supports, contradicts, elaborates)",
            "thematic clusters",
            "conceptual structure of discussion"
        ],
        "use_for": [
            "understanding intellectual content",
            "tracing idea evolution",
            "identifying reasoning patterns",
            "comparing conceptual coverage"
        ],
        "limitations": [
            "derived, not primary",
            "may miss nuance",
            "relationship types are interpretive"
        ]
    },
    "collaboration": {
        "provides": [
            "7C dimension scores (0-100)",
            "quantified interaction quality",
            "comparison baseline",
            "flagging of potential issues"
        ],
        "use_for": [
            "quick quality assessment",
            "session comparison",
            "identifying dimensions to investigate",
            "pattern detection across sessions"
        ],
        "limitations": [
            "scores are interpretive",
            "may not capture context",
            "need transcript to understand WHY"
        ]
    },
    "speaker_profile": {
        "provides": [
            "individual participation patterns",
            "cross-session contribution style",
            "question/statement ratios",
            "concept contributions"
        ],
        "use_for": [
            "understanding individual roles",
            "comparing speaker styles",
            "identifying dominant/quiet participants"
        ],
        "limitations": [
            "aggregated data",
            "may not reflect specific session dynamics"
        ]
    }
}

# =============================================================================
# TRIANGULATION FRAMEWORK
# =============================================================================
# How to reason across multiple sources

TRIANGULATION_FRAMEWORK = """
## Triangulation Framework

When making claims, consider how sources relate:

### Convergence
Multiple sources support the same conclusion. This strengthens confidence.

Example: "The high communication score (85/100) is evident in the transcript where
participants frequently build on each other's ideas. Sarah says 'Building on what
Mike mentioned...' and the concept map shows multiple 'elaborates' relationships."

### Tension
Sources suggest different things. This is signal, not error - interpret what it means.

Example: "The concept map shows deep causal reasoning with sophisticated connections,
but the contribution score is low (20/100). This suggests one person drove the
intellectual work while others were less engaged."

### Gaps
What you couldn't find. Be honest about limitations.

Example: "The transcript doesn't contain explicit statements about fusion energy
applications, though the concept map suggests this theme was present. The gap
may indicate this was discussed non-verbally or in portions not captured."

**Principle**: Don't just list what each source says. Interpret how they relate
and what the combination tells you.
"""

# =============================================================================
# GROUNDING REQUIREMENTS
# =============================================================================

GROUNDING_REQUIREMENTS = """
## Grounding Your Claims

Every substantive claim should be traceable to evidence:

1. **Quote specific statements** with speaker attribution
   - "David explained: 'Fusion is what happens in stars...'"

2. **Reference specific scores** with dimension names
   - "The communication score (80/100) suggests active exchange"

3. **Note which representation** supports each point
   - "According to the concept map..." or "The transcript shows..."

4. **Distinguish data from interpretation**
   - "The data shows X. This suggests Y."
   - "One interpretation is..." or "This might indicate..."

**Principle**: The reader should be able to verify your claims by checking the sources.
"""

# =============================================================================
# BEYOND RETRIEVAL
# =============================================================================

BEYOND_RETRIEVAL = """
## Beyond Retrieval

You are not just a search engine. You are an expert analyst who can:

- **Apply domain knowledge**: Use your understanding of education, learning science,
  and discourse analysis to interpret patterns

- **Identify unstated patterns**: Notice things the data alone doesn't show explicitly

- **Suggest explanations**: Propose reasons for observed phenomena based on theory

- **Connect to research**: Relate observations to broader educational research

- **Offer insights**: Provide analytical interpretations, not just data summaries

**Principle**: When reasoning beyond the data, be clear about it.
- "The data shows..." (factual)
- "This suggests..." (interpretation)
- "Research indicates..." (connecting to broader knowledge)
- "One explanation might be..." (hypothesizing)
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_operationalization(construct: str) -> list:
    """Get operationalization for a construct (case-insensitive, partial match)."""
    construct_lower = construct.lower()

    # Exact match first
    if construct_lower in CONSTRUCT_OPERATIONALIZATIONS:
        return CONSTRUCT_OPERATIONALIZATIONS[construct_lower]

    # Partial match
    for key, indicators in CONSTRUCT_OPERATIONALIZATIONS.items():
        if construct_lower in key or key in construct_lower:
            return indicators

    return []


def detect_constructs_in_query(query: str) -> list:
    """Detect which abstract constructs are mentioned in a query."""
    query_lower = query.lower()
    detected = []

    for construct in CONSTRUCT_OPERATIONALIZATIONS.keys():
        # Check for construct mention (with word boundaries)
        if construct in query_lower:
            detected.append(construct)
        # Also check key words
        elif construct == "systems thinking" and ("system" in query_lower and "think" in query_lower):
            detected.append(construct)
        elif construct == "critical thinking" and ("critical" in query_lower and "think" in query_lower):
            detected.append(construct)

    return detected


def format_operationalization_for_prompt(constructs: list) -> str:
    """Format operationalizations for detected constructs into prompt text."""
    if not constructs:
        return ""

    lines = ["## Operationalizing Abstract Constructs",
             "When analyzing the following concepts, look for these observable indicators:", ""]

    for construct in constructs:
        indicators = get_operationalization(construct)
        if indicators:
            lines.append(f"**{construct.title()}** → Look for:")
            for indicator in indicators:
                lines.append(f"- {indicator}")
            lines.append("")

    return "\n".join(lines)
