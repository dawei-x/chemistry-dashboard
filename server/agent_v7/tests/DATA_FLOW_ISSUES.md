# V7 Agent Data Flow Issues Analysis

**Date**: 2026-01-17
**Purpose**: Document actual data flow through the V7 agent pipeline for diagnosis
**Status**: DOCUMENTATION ONLY - No fixes applied

---

## Executive Summary

Tracing 3 representative queries through the V7 agent pipeline revealed **critical data flow issues** that cause the agent to miss relevant information. These issues are **not related to missing SubGoal Decomposition** but are fundamental data pipeline bugs.

| Issue | Severity | Root Cause |
|-------|----------|------------|
| Session Truncation | **Critical** | `max_sessions=5` in exploratory.py truncates results |
| Speaker Extraction | **Medium** | classifier.py only extracts speakers from specific patterns |
| Search Exclusion | **High** | Relevant sessions excluded due to low semantic similarity |

---

## Query 1: "Did Tucker demonstrate systems thinking in session 19?"

### Classification Phase
```
is_exploratory: False (targeted - has session ID)
session_ids: [19]
speakers: []         ← ISSUE: Tucker NOT extracted as speaker
topics: ['tucker', 'demonstrate', 'systems', 'thinking']  ← Tucker in topics!
artifact_hint: None
```

### Data Flow Issue: Speaker Extraction Failure

**What happened**: The classifier's `_extract_speakers()` function only extracts speakers from patterns like:
- "what did X say"
- "how did X contribute"
- "X's contributions"

**Root cause**: `classifier.py:185-215`
```python
patterns = [
    r'\b(?:how\s+did\s+)(\w+)\s+(?:engage|contribute|participate)',
    r'\b(?:what\s+did\s+)(\w+)\s+(?:say|ask|discuss)',
    r'\b(\w+)\'s\s+(?:contributions?|questions?|statements?)',
    r'\bspeaker\s+(\w+)\b',
]
```

The query "Did **Tucker** demonstrate systems thinking" doesn't match any of these patterns, so Tucker is classified as a **topic** instead of a **speaker**.

**Impact**:
- Agent cannot apply speaker-specific filters
- May search for "tucker" as a general topic rather than focusing on a specific person
- Less targeted retrieval

---

## Query 2: "Which session has the best collaboration quality?"

### Classification Phase
```
is_exploratory: True
session_ids: []
speakers: []
topics: ['session', 'quality']
artifact_hint: 'collaboration'
```

### Data Flow Issue: Session Truncation

**What happened**: The agent only retrieved 7C analysis for 5 sessions:
- Session 18: 67.9/100
- Session 19: 55.0/100
- Session 20: 79.0/100 ← Agent concluded this is "best"
- Session 21: 69.3/100
- Session 22: 50.0/100

**MISSING sessions**:
- Session 23: NOT RETRIEVED
- Session 24: NOT RETRIEVED ← Has score ~80/100 (actual best)
- Session 25: NOT RETRIEVED
- Session 26: NOT RETRIEVED

**Root cause**: `exploratory.py:46, 183, 236`
```python
def retrieve_exploratory(
    query: str,
    classification: QueryClassification,
    tools: Dict[str, Callable],
    max_sessions: int = 5   # ← HARDCODED LIMIT
) -> ExploratoryResult:
    ...
    session_ids = sorted(session_ids_set)
    return session_ids[:max_sessions]  # ← TRUNCATES TO 5
```

**Impact**:
- Agent answered "Session 20 has the best collaboration (79.0/100)"
- **WRONG**: Session 24 actually has ~80/100 but was NEVER SEEN
- Complete answer failure for superlative queries

---

## Query 3: "What sessions discussed technology and its societal impact?"

### Classification Phase
```
is_exploratory: True
session_ids: []
speakers: []
topics: ['technology', 'societal', 'impact']
artifact_hint: None
```

### Data Flow Issue: Search Exclusion

**What happened**: Only sessions 19 and 22 were retrieved.

**MISSING sessions**:
- Session 20 (Nuclear Fusion) - Score 0.13, below threshold 0.17
- Session 25 (Abundance) - Discusses technology and politics

**Root cause**: Search uses semantic similarity scoring. Session 20 ("Nuclear Fusion") doesn't have high semantic similarity to "technology societal impact" even though fusion IS technology with societal impact.

**Process trace**:
1. Search query: "technology societal impact"
2. Session 20 relevance score: 0.13
3. Threshold: 0.17
4. Result: Session 20 EXCLUDED

**Impact**:
- Nuclear Fusion session completely missed from answer
- Incomplete coverage of cross-session thematic queries
- User gets partial information

---

## Data Flow Trace: What the Agent Actually Saw

### For "Compare collaboration between AI discussion and Nuclear Fusion"

**Phase 1: Raw Tool Output**
```json
{
  "tool": "search_sessions",
  "query": "Nuclear Fusion",
  "result": {
    "sessions_found": 1,
    "sessions": [{"session_id": 20, "session_name": "Session 20"}]
  }
}
```

**Phase 2: Evidence Formatted for Context (Decision Phase)**
```
[get_7c_analysis] Session 'Is AI Alive': Average 55.0/100
```

**Phase 3: Synthesis Input**
- Only received 7C for session 19
- Did NOT receive 7C for session 20 (Nuclear Fusion)

**Result**: Agent apologized for not having Nuclear Fusion data when it should have retrieved it.

---

### For "What did David say about temperature in Nuclear Fusion?"

**Phase 1: Classification**
```
extracted_session: 19  ← WRONG! Should resolve "Nuclear Fusion" to 20
extracted_speaker: "David"
```

**Phase 2: Tool Calls**
```json
{
  "tool": "get_transcript",
  "params": {
    "session_id": 19,  // ← WRONG SESSION
    "speaker_filter": "David",
    "keyword_filter": "temperature"
  },
  "result": "utterance_count": 0  // ← Empty because wrong session
}
```

**Phase 3: Agent Response**
Agent correctly identified the mismatch and noted it couldn't find David's comments on temperature, then searched for Nuclear Fusion. But the initial session resolution was wrong.

---

## Root Cause Analysis: Is SubGoal Decomposition the Fix?

### Examination of Issues

| Issue | Would SubGoal Decomposition Fix? |
|-------|----------------------------------|
| Session truncation at max_sessions=5 | **NO** - This is a hardcoded parameter issue |
| Speaker not extracted from query | **NO** - This is a regex pattern matching issue |
| Search excludes low-score sessions | **NO** - This is a threshold/scoring issue |

### Conclusion

The issues found are **data pipeline bugs**, not architectural gaps:

1. **max_sessions=5** - Arbitrary limit that should be configurable or removed for "all sessions" queries
2. **Speaker extraction** - Regex patterns don't cover all natural language forms
3. **Search thresholds** - Semantic similarity may not capture topical relevance

**SubGoal Decomposition** would help with:
- Multi-part queries ("Find X AND Y")
- Tracking whether all aspects of a query were addressed
- Verifying completion criteria

But it would **NOT** fix the current issues where the agent **never sees** the relevant data in the first place.

---

## Evidence: Committed V7's Approach

The committed V7 has `CONSTRUCT_OPERATIONALIZATIONS` for abstract constructs:

```python
CONSTRUCT_OPERATIONALIZATIONS = {
    "systems thinking": [
        "identifying causal relationships between concepts",
        "seeing interconnections across ideas",
        "understanding feedback loops",
        "considering multiple perspectives",
        "recognizing emergent patterns"
    ],
    ...
}
```

This helps **operationalize** abstract queries, but it doesn't address the data pipeline issues documented here.

---

## Recommendations (NOT YET IMPLEMENTED)

1. **Session Truncation**: Remove max_sessions limit for "all sessions" queries, or set it much higher (e.g., 50)

2. **Speaker Extraction**: Add more patterns to recognize speakers in various query forms:
   - "Did Tucker..."
   - "Tucker's contribution..."
   - "How does Tucker..."

3. **Search Thresholds**: For exploratory queries, either lower the threshold or bypass search entirely and use list_sessions

4. **Session Name Resolution**: Add mapping for session names (e.g., "Nuclear Fusion" → session 20) before tool calls

---

## Files Involved

| File | Issue |
|------|-------|
| `classifier.py:185-215` | Speaker extraction patterns |
| `exploratory.py:46, 183, 236` | max_sessions=5 truncation |
| `tools_v2.py` | Search threshold logic |
| `graph_v2.py` | Session name → ID resolution |
