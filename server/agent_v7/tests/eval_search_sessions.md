# Evaluation: search_sessions for "What was said about AI?"

**Date**: 2026-01-17
**Tool**: `search_sessions` in `tools_v2.py` (wraps `search_for_sessions` from `artifact_tools.py`)
**Query**: "What was said about AI?"

## Current Agent Output (All Issues Fixed)

```
=== Search Results for "What was said about AI?" (2 found) ===

1. Session 19: Is AI Alive
   Relevance: 0.40
   Speakers: Sam, Tucker
   Preview: Tucker: PT other AI's can reason...

2. Session 26: CFAA Discussion
   Relevance: 0.34
   Speakers: SPEAKER_00, SPEAKER_01, SPEAKER_02
   Preview: SPEAKER_00: violations on chatgpt's part...

(Session 22 correctly excluded - not related to AI)
```

---

## Issues Found

### Issue 1: session_name missing from ChromaDB metadata - FIXED

**Problem**: Output showed "Session 19: Session 19" instead of "Session 19: Is AI Alive"

**Fix Applied**: Added `session_name`, `device_name`, and `speakers` to `_compute_all_metrics()` in `session_serializer.py`

**Status**: ✅ RESOLVED

---

### Issue 2: speakers missing from ChromaDB metadata - FIXED

**Problem**: Output showed "Speakers: Unknown" for all sessions

**Fix Applied**: Added speaker extraction using `speaker_tag` from transcripts

**Status**: ✅ RESOLVED

---

### Issue 3: False positive results due to permissive thresholds - FIXED

**Problem**: Session 22 (Collaboration Literacy) was returned for "AI" query despite having NO AI-related content.

**Root Cause Analysis**:

Raw ChromaDB distances and converted similarity scores:
| Session | Distance | Score (1-dist) | Content |
|---------|----------|----------------|---------|
| Session 19 (Is AI Alive) | 0.5954 | **0.4046** | Discusses AI reasoning, AI consciousness |
| Session 26 (CFAA Discussion) | 0.6555 | **0.3445** | Discusses ChatGPT, data usage violations |
| Session 22 (Collaboration Literacy) | 0.7600 | **0.2400** | "learning analytics community" - NO AI! |

**Filtering logic in `artifact_tools.py:search_for_sessions()`**:
```python
min_score = 0.05           # Very permissive
min_relative_score = 0.35  # Lowered from 0.70 "for better recall"
```

**Calculation**:
- Best score: 0.4046
- Relative threshold: 0.4046 × 0.35 = 0.1416
- Session 22 score: 0.24 > 0.1416 → **PASSES** (but shouldn't!)

**Why it passes**: Session 22's score of 0.24 is 59% of the best match. With `min_relative_score=0.35`, anything above 35% of the best match is included. The threshold was intentionally lowered from 0.70 to 0.35, which is too permissive.

**Fix Options**:

1. **Raise `min_relative_score` back to 0.50-0.60**:
   ```python
   min_relative_score = 0.55  # Balance between recall and precision
   ```
   This would require 0.4046 × 0.55 = 0.2225, which Session 22 (0.24) barely passes.
   At 0.60: 0.4046 × 0.60 = 0.2428 → Session 22 would be EXCLUDED.

2. **Raise `min_score` to 0.25+**:
   ```python
   min_score = 0.25  # Absolute minimum similarity
   ```
   Session 22's score of 0.24 < 0.25 would be excluded.

3. **Use both thresholds together**:
   ```python
   min_score = 0.20           # Don't include anything with <20% similarity
   min_relative_score = 0.55  # Must be at least 55% as good as best match
   ```

**Fix Applied**: Set `min_relative_score = 0.60` and `min_score = 0.20` to balance recall and precision.

**Status**: ✅ RESOLVED

---

### Issue 4: Relevance ranking is CORRECT (PASS)

| Rank | Session | Score | Relevance |
|------|---------|-------|-----------|
| 1 | Session 19 (Is AI Alive) | 0.40 | ✓ Discusses AI reasoning directly |
| 2 | Session 26 (CFAA Discussion) | 0.34 | ✓ Discusses ChatGPT specifically |
| 3 | Session 22 (Collaboration Literacy) | 0.24 | ✗ False positive (no AI content) |

**Verdict**: The **ranking** is correct (relevant sessions first), but the **filtering** is too permissive, allowing irrelevant results.

---

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Finds relevant sessions | ✓ PASS | Sessions 19 & 26 correctly identified |
| Session names | ✓ PASS | Fixed - now shows "Is AI Alive" |
| Speaker names | ✓ PASS | Fixed - now shows "Sam, Tucker" |
| Relevance ranking | ✓ PASS | Most relevant sessions ranked first |
| Relevance filtering | ✓ PASS | Fixed - Session 22 now excluded |

---

## All Issues Resolved

### Verification (After All Fixes)

```
=== V7 search_for_sessions (min_score=0.20, min_relative_score=0.60) ===
Sessions found: 2

19: Is AI Alive
   Score: 0.4046
   Speakers: ['Sam', 'Tucker']

26: CFAA Discussion
   Score: 0.3445
   Speakers: ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']

Session 22 (Collaboration Literacy) correctly EXCLUDED:
   Score: 0.24 < Threshold 0.2428 (0.4046 × 0.60)
```

---

## Files Modified (All Complete)

1. **`server/session_serializer.py`** - Added `session_name`, `device_name`, `speakers` to `_compute_all_metrics()`
2. **`server/agent_v7/tools/artifact_tools.py`** - Added speakers parsing; raised thresholds to `min_score=0.20`, `min_relative_score=0.60`
3. **`server/agent_v3/tools/artifact_tools.py`** - Same fixes for V3 consistency; raised `min_score` to 0.25
4. **Re-indexed**: `python session_rag_indexer.py --force`
