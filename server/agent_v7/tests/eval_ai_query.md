# Evaluation: "What was said about AI?"

**Date**: 2026-01-17
**Query**: "What was said about AI?"
**Expected Behavior**: Search across sessions for quotes mentioning AI, return relevant excerpts.

## Classification Result

| Field | Value | Assessment |
|-------|-------|------------|
| is_exploratory | True | Correct |
| session_ids | [] | Correct (no specific session) |
| topics | ['ai'] | Correct |
| artifact_hint | transcript | Correct |
| reason | Matched `\bwhat\s+was\s+said\s+about\b` | Correct |

**Classification: PASS**

---

## Sessions Retrieved

| Session | Name | Has AI Mentions? | Relevance |
|---------|------|------------------|-----------|
| 18 | Living in NYC | NO | False positive |
| 19 | Is AI Alive | YES (7 utterances) | Highly relevant |
| 21 | Shaw Interview | NO (false match on "traits", "said") | False positive |
| 22 | Collaboration Literacy | NO | False positive |

**Session Discovery: PARTIAL FAIL** - 3 of 4 sessions are false positives

---

## Issues Found

### Issue 1: Keyword Filter Uses Substring Match (BUG)

**Location**: `tools_v2.py:221-225`

```python
if keyword_filter:
    keyword_lower = keyword_filter.lower()
    utterances = [
        u for u in utterances
        if keyword_lower in u.get('text', '').lower()  # <- substring match
    ]
```

**Problem**: "ai" matches "tr**ai**ts", "s**ai**d", "expl**ai**n", "cont**ai**ns", etc.

**Evidence**: Session 21 (Shaw Interview) returns 12 "AI" matches, but none are about AI - they're words containing "ai".

**Fix**: Use word boundary regex:
```python
import re
pattern = r'\b' + re.escape(keyword_filter.lower()) + r'\b'
utterances = [u for u in utterances if re.search(pattern, u.get('text', '').lower())]
```

---

### Issue 2: Exploratory Retrieval Doesn't Use Keyword Filter

**Location**: `exploratory.py:_retrieve_from_session()`

**Problem**: For a query like "What was said about AI?", the agent retrieves FULL transcripts (2000-4000 chars each) instead of filtered excerpts. This:
- Wastes context tokens
- Makes synthesis harder for LLM
- Returns irrelevant content

**Current behavior**:
```python
get_transcript(session_id=X)  # Full transcript
```

**Expected behavior**:
```python
get_transcript(session_id=X, keyword_filter='AI')  # Only AI mentions
```

**Fix**: In exploratory retrieval, pass the topic as keyword_filter when artifact_type is 'transcript'.

---

### Issue 3: Semantic Search Returns False Positives

**Problem**: Sessions 18 and 22 have zero AI mentions but are returned by search.

| Session | Relevance Score | Actual AI Mentions |
|---------|-----------------|-------------------|
| 19 | 0.29 | 7 utterances |
| 22 | 0.16 | 0 utterances |
| 18 | ~0.10 | 0 utterances |
| 21 | ~0.10 | 0 utterances |

**Cause**: Semantic similarity doesn't require exact keyword match. Sessions about technology topics (like collaboration analytics) have some embedding similarity to "AI" even without mentioning AI.

**Current threshold**: `min_relative_score: 0.35` (scores must be 35% of best match)

**Potential fixes**:
1. Raise `min_score` threshold
2. Add keyword verification step after search
3. Rank by actual keyword occurrence count

---

### Issue 4: No Relevance Ranking in Evidence

**Problem**: All sessions are treated equally. Session 19 (most relevant, explicitly about AI) is not prioritized over false positives.

**Fix**: Order evidence by:
1. Keyword occurrence count in transcript
2. Semantic search score
3. Explicitly requested sessions first

---

## Impact Assessment

| Issue | Severity | Impact |
|-------|----------|--------|
| Keyword substring bug | HIGH | Wrong utterances returned, misleading LLM |
| No keyword filter in exploratory | MEDIUM | Wastes tokens, dilutes relevance |
| Search false positives | MEDIUM | Retrieves irrelevant sessions |
| No relevance ranking | LOW | LLM sees all evidence equally |

---

## Recommended Fixes (Priority Order)

1. **Fix keyword filter** to use word boundaries (HIGH - straightforward fix)
2. **Pass keyword_filter in exploratory retrieval** when topic is extracted (MEDIUM)
3. **Add keyword verification** after search to filter sessions with zero matches (MEDIUM)
4. **Rank evidence by relevance** in exploratory result (LOW)

---

## Test Cases After Fix

```python
# Should return only Session 19 (Is AI Alive)
get_transcript(session_id=19, keyword_filter='AI')
# -> 7 utterances about AI, not 0 or 12

# Should NOT return Sessions 18, 21, 22
# (they have no actual AI mentions)
```
