# V7 Scaffolding Agent - Known Issues

This document captures issues identified during critical evaluation of the new architecture.

---

## Comprehensive Evaluation Results (2026-01-15)

### Summary Metrics
| Metric | Score |
|--------|-------|
| **Answered** | 20/20 (100%) |
| **Accurate** | 18/20 (90%) |
| **Scaffolded** | 14/20 (70%) |
| **Tools OK** | 14/20 (70%) |
| **Avg Time** | 12.53s |
| **Avg Iterations** | 1.6 |

### By Category Performance
| Category | Accuracy |
|----------|----------|
| 7C Analysis | 2/2 ✓ |
| Basic Queries | 3/3 ✓ |
| Comparison | 1/1 ✓ |
| Concept Map | 1/1 ✓ |
| Edge Case | 1/1 ✓ |
| Hypothesis | 1/1 ✓ |
| Multi-turn | 5/5 ✓ |
| Scaffolding | 1/1 ✓ |
| Steering | 2/2 ✓ |
| Search | 0/1 ✗ |
| Speaker | 1/2 ✗ |

### Issues Found in Evaluation

1. **Tool Selection Not Optimal** (basic_02)
   - Query: "Tell me about session 19"
   - Expected: `get_session_overview`
   - Got: `get_transcript`, `get_concept_map`
   - Impact: Works but inefficient

2. **Speaker Info Missing** (speaker_02)
   - Query: "Who were the main speakers in the Country Music discussion?"
   - Expected: Oliver, Lex
   - Answer didn't include speaker names clearly

3. **Search Relevance** (search_01)
   - Query: "Find discussions about energy"
   - Expected: Nuclear Fusion session
   - Search didn't return expected results

---

## Fixed Issues

### ~~1. Steering Extraction Bug~~ - FIXED

**Original problem**: Regex-based steering extraction was fundamentally flawed.

**Solution**: Removed regex parsing entirely. The LLM understands "focus on concept map, don't use 7C" natively - that's what LLMs are for. No pattern matching needed.

### ~~4. Session Name Resolution Bug~~ - FIXED (2026-01-15)

**Original problem**: Dict iteration checked 'ai' before 'nuclear fusion', so "Compare AI discussion and Nuclear Fusion" matched session 19 instead of 20.

**Root cause**: `SESSION_NAME_TO_ID` was a dict with arbitrary iteration order. Short generic terms like 'ai' matched before longer specific terms. Additionally, substring matching caused "said" to match "ai".

**Solution**:
1. Changed to ordered list of tuples with longer/more specific names first
2. Added word boundary matching for single-word terms (prevents "said" matching "ai", "confusion" matching "fusion")

```python
# Single words use word boundaries
pattern = r'\b' + re.escape(name) + r'\b'
if re.search(pattern, text_lower):
    return sid
```

**File**: `memory.py` lines 20-48, 269-280

### ~~2. Preference and Exclusion Conflict~~ - FIXED

**Original problem**: Regex could put artifacts in both preferred and excluded lists.

**Solution**: No more regex extraction. LLM reads the query and understands user intent.

---

## Remaining Issues

### 3. Memory Persistence - In-Memory Only

**Severity**: Medium
**File**: `memory.py` line 280

**Problem**: Conversation memory is stored in a Python dict `_memory_store`. This is lost on server restart.

**Current**:
```python
_memory_store: Dict[str, ConversationMemory] = {}
```

**Fix needed**: Implement Redis-backed storage for production:
```python
# Option 1: Redis
memory_json = redis_client.get(f"memory:{conversation_id}")

# Option 2: Database
memory = AgentMemory.query.filter_by(conversation_id=conversation_id).first()
```

---

### 4. Hardcoded Session Name Mapping

**Severity**: Medium
**File**: `memory.py` lines 19-30

**Problem**: `SESSION_NAME_TO_ID` is hardcoded. New sessions won't be recognized by name.

```python
SESSION_NAME_TO_ID = {
    'living in nyc': 18,
    'is ai alive': 19,
    # ... hardcoded
}
```

**Fix needed**: Query database for session name → ID mapping, or use semantic search.

---

### 5. API Response Compatibility

**Severity**: Medium
**Files**: `routes_v2.py` vs `routes.py`

**Problem**: The new routes return slightly different response structure:

| Field | Old | New |
|-------|-----|-----|
| `final_answer` | ✓ | `answer` |
| `confidence` | Float from agent | Hardcoded 0.8 |
| `reasoning_trace` | ✓ | Missing |
| `verification` | ✓ | Missing |

**Impact**: Frontend code depending on old field names may break.

---

### 6. compare_sessions Makes Multiple Tool Calls Internally

**Severity**: Medium
**File**: `tools_v2.py` lines 280-310

**Problem**: `compare_sessions` internally calls `get_session_overview`, `get_7c_analysis`, and `get_concept_map` for each session. This:
- Bypasses steering validation (user said "skip 7C" but compare_sessions uses it)
- Makes many database queries
- Could be slow for many sessions

**Fix needed**: Pass steering context to compare_sessions or use cached data.

---

## Low Issues

### 7. No Rate Limiting on LLM Calls

**Severity**: Low
**File**: `react_agent.py`

**Problem**: MAX_ITERATIONS=8 allows up to 8 LLM calls per request. No rate limiting.

---

### 8. Evidence Truncation May Lose Context

**Severity**: Low
**File**: `react_agent.py` lines 450-460

**Problem**: Quotes are truncated to 150 chars, reasons to 100 chars. Complex evidence may be cut off.

```python
quote = seg.get("quote", "")[:150]  # Truncation
reason = seg.get("reason", "")[:100]  # Truncation
```

---

### 9. Tool Schemas Use Basic Types Only

**Severity**: Low
**File**: `tools_v2.py` TOOL_SCHEMAS

**Problem**: No enum constraints on parameters. LLM could pass invalid values.

Example: `speaker_filter` accepts any string, but valid speakers are limited.

---

### 10. find_concept_path Untested

**Severity**: Low
**File**: `tools_v2.py`

**Problem**: The `find_concept_path` tool wraps the legacy implementation but wasn't tested in the new architecture.

---

## Architectural Observations

### What's Working Well (Confirmed by Evaluation)

1. **ReAct loop** - Successfully decides when to call tools vs respond
2. **Deduplication** - Prevents repeated identical tool calls
3. **Invalid tool filtering** - Skips `multi_tool_use`, `functions` etc.
4. **Memory context injection** - Session focus persists across turns (5/5 multi-turn tests passed)
5. **Scaffolded responses** - Cites specific quotes and segments (70% scaffolded)
6. **LLM-native steering** - User preferences respected without regex (2/2 steering tests passed)
7. **7C Analysis queries** - Works perfectly (2/2)
8. **Hypothesis verification** - LLM flexibility maintained (1/1)

### What Needs Improvement

1. **Search relevance** - Semantic search not finding obvious matches (0/1)
2. **Tool selection** - Sometimes uses transcript when overview would suffice
3. **Parallel execution** - Tools execute sequentially, could parallelize
4. **Error recovery** - No retry logic if LLM call fails
5. **Confidence scoring** - Currently hardcoded to 0.8
6. **Claim tracking** - `claims_made` in memory is never populated

---

## Testing Gaps

1. **find_concept_path** - Not tested
2. **compare_sessions with steering** - Steering may be bypassed
3. **Very long conversations** - Memory growth not bounded
4. **Concurrent requests** - Thread safety of `_memory_store`
5. **Edge case queries** - Empty queries, very long queries, non-English

---

## Recommended Priority (Updated)

1. **Improve search relevance** - "energy" should find Nuclear Fusion
2. **Optimize tool selection** - Add prompt guidance for tool choice
3. **Add Redis memory** - Production requirement
4. **Verify API compatibility** - Check frontend still works
5. **Add confidence scoring** - Replace hardcoded 0.8

---

## Change Log

- **2026-01-15**: Comprehensive evaluation completed. 90% accuracy, 70% scaffolded.
- **2026-01-15**: Removed regex-based steering - LLM handles naturally.
- **2026-01-14**: Initial V7 architecture with ReAct loop.
