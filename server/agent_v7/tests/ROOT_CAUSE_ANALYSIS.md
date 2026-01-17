# V7 Data Flow Issues: Root Cause Analysis

**Date**: 2026-01-17
**Status**: Analysis complete - Plan needed

---

## Executive Summary

The issues in our V7 are **NOT** from missing SubGoal Decomposition. They are **architectural regressions** from V3's proven design. V3 worked better because:

1. **V3 loaded session patterns from DATABASE** - Our V7 uses a hardcoded static list
2. **V3 had NO `max_sessions` limit** - Our V7 hardcodes `max_sessions=5`
3. **V3 let GPT-4o reason about ALL queries** - Our V7 bypasses LLM with heuristic routing

---

## Detailed Comparison

### 1. Session Name Resolution

#### V3 (`input_processor.py:66-120`)
```python
def _load_session_patterns() -> Dict[str, int]:
    """Load session name patterns from database dynamically."""
    conn = _get_db_connection()
    cursor.execute("""
        SELECT s.id, s.name, sd.id as session_device_id
        FROM session s
        JOIN session_device sd ON sd.session_id = s.id
    """)
    for row in cursor.fetchall():
        # Add full name: "nuclear fusion" -> 20
        patterns[name] = session_id
        # Add individual words: "fusion" -> 20
        for word in name.split():
            if word not in patterns:
                patterns[word] = session_id
```

**V3 dynamically discovers session names from the database**, including:
- Full session names ("nuclear fusion" → 20)
- Individual significant words ("fusion" → 20)
- Automatic refresh every 5 minutes

#### Our V7 (`memory.py:23-48`)
```python
SESSION_NAME_TO_ID = [
    ('nuclear fusion', 20),
    ('ai alive', 19),
    ('dinosaurs', 23),
    # ... hardcoded static list
]
```

**Our V7 uses a hardcoded list** that:
- Must be manually maintained
- May be incomplete or outdated
- Doesn't auto-discover new sessions

**Impact**: Works for known sessions but won't adapt to new data.

---

### 2. Session Truncation (CRITICAL)

#### V3 (`query_router.py`)
```python
# V3's approach: Let GPT-4o decide
logger.info("Using flexible reasoning path (GPT-4o will select tools)")
return {'route': 'reasoning'}
```

V3 routes most queries to the **reasoning path** where GPT-4o decides:
- What tools to call
- How many sessions to check
- When to stop iterating

**There is NO hardcoded session limit** - the LLM controls iteration.

#### Our V7 (`exploratory.py:46`, `react_agent.py:211`)
```python
def retrieve_exploratory(
    query: str,
    classification: QueryClassification,
    tools: Dict[str, Callable],
    max_sessions: int = 5   # ← HARDCODED LIMIT
) -> ExploratoryResult:
    ...
    return session_ids[:max_sessions]  # ← TRUNCATES TO 5
```

And in react_agent.py:
```python
exploratory_result = retrieve_exploratory(
    query=query,
    classification=classification,
    tools=self._tools_dict,
    max_sessions=5  # ← HARDCODED
)
```

**Impact**:
- For "Which session has the best collaboration?", only 5 sessions are checked
- Session 24 (actual best at ~80/100) is NEVER SEEN
- Agent incorrectly concludes Session 20 is best (79.0/100)

---

### 3. Speaker Extraction Split

#### Our V7 has TWO systems that don't agree:

**System 1: `memory.py:294-311` (lenient)**
```python
def extract_speaker_from_text(self, text: str) -> Optional[str]:
    known_speakers = ['Tucker', 'Maya', 'Oliver', ...]
    for speaker in known_speakers:
        if speaker.lower() in text_lower:  # Simple substring match
            return speaker
```

**System 2: `classifier.py:185-215` (strict)**
```python
def _extract_speakers(query: str) -> List[str]:
    patterns = [
        r'\b(?:how\s+did\s+)(\w+)\s+(?:engage|contribute)',
        r'\b(?:what\s+did\s+)(\w+)\s+(?:say|ask)',
        r'\b(\w+)\'s\s+(?:contributions?)',
    ]
```

**The classifier uses the STRICT system** for query classification.

**Impact**:
- Query: "Did Tucker demonstrate systems thinking?"
- classifier.py: No match for any pattern → `speakers: []`
- Tucker ends up in `topics: ['tucker', ...]` instead

---

### 4. Exploratory Path Bypasses LLM

#### V3's Approach
```
Query → Query Router → Reasoning Path (GPT-4o)
                              ↓
                    GPT-4o decides tools
                              ↓
                    LLM iterates as needed
```

V3 lets GPT-4o reason about what tools to call. The LLM can:
- Call `list_sessions` to see all sessions
- Iterate through as many as needed
- Stop when it has enough evidence

#### Our V7's Approach
```
Query → Classifier → Exploratory/Targeted Split
              ↓
    Hardcoded retrieval logic
              ↓
    max_sessions=5 truncation
              ↓
    LLM only sees synthesis
```

Our V7 uses **heuristic classification** that:
- Bypasses LLM for retrieval decisions
- Applies hardcoded limits
- Only involves LLM for final synthesis

---

## Root Cause Summary

| Issue | V3 Approach | Our V7 Approach | Why V3 Worked |
|-------|-------------|-----------------|---------------|
| Session names | Database-driven | Hardcoded list | Auto-discovery, always current |
| Session limit | None (LLM decides) | `max_sessions=5` | LLM could iterate as needed |
| Speaker extraction | LLM reasoning | Strict regex patterns | LLM understood context |
| Tool selection | LLM decides | Heuristic + hardcoded | LLM adapted to query needs |

---

## Why SubGoal Decomposition Is NOT the Fix

SubGoal Decomposition (from committed V7) would help with:
- Tracking multi-part queries
- Verifying all aspects addressed
- Explicit satisfaction checking

But it **cannot fix** these issues because:

1. **Session truncation** happens BEFORE any reasoning
2. **Speaker extraction** fails at classification time
3. **The agent never sees the missing data** to reason about

The data pipeline must be fixed FIRST. SubGoal Decomposition is an enhancement for reasoning, not data access.

---

## Recommended Fixes (Priority Order)

### P0: Critical - Must Fix

1. **Remove session truncation for superlative queries**
   - File: `exploratory.py:46, 183, 236`
   - Change: Detect "all/best/worst" queries → no limit
   - Or: Increase default to 50+ sessions

2. **Fix speaker extraction in classifier**
   - File: `classifier.py:185-215`
   - Change: Add more patterns OR use memory.py's lenient approach

### P1: Important - Should Fix

3. **Consider database-driven session names**
   - File: `memory.py:23-48`
   - Change: Either sync with database OR ensure list is complete

4. **Review exploratory path limits**
   - For comparison/superlative queries, must check ALL sessions
   - The heuristic should not truncate

### P2: Nice to Have

5. **Unify speaker extraction**
   - Have one source of truth for speaker detection
   - Use in both classifier and memory

6. **Consider hybrid approach**
   - Use heuristic classification for routing
   - But let LLM decide iteration depth

---

## Evidence From Traces

### Query: "Which session has the best collaboration?"

**V3 would have**:
- Called `list_sessions` to get all 9 sessions
- GPT-4o would iterate through all
- Found Session 24 with highest score

**Our V7 actually did**:
- Classified as exploratory
- Retrieved only 5 sessions (18-22)
- Missed Session 24 (highest score ~80)
- Incorrectly concluded Session 20 (79.0) was best

### Query: "Did Tucker demonstrate systems thinking in session 19?"

**V3 would have**:
- Let GPT-4o reason about the query
- LLM would understand "Tucker" is a speaker

**Our V7 actually did**:
- classifier.py didn't match any speaker pattern
- Put "tucker" in topics list
- Lost speaker context for targeted retrieval

---

## Conclusion

The issues are not about missing SubGoal Decomposition or LLM reasoning enhancements. They are about **data pipeline bugs** that prevent the agent from seeing all relevant data:

1. `max_sessions=5` is an arbitrary limit that breaks superlative queries
2. Speaker extraction patterns are too strict
3. The heuristic-driven exploratory path bypasses LLM flexibility

V3 worked better because it **trusted the LLM more** and had **fewer hardcoded limits**.
