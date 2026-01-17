# V7 Agent Design Simplification Proposal

**Date**: 2026-01-17
**Status**: Proposal for discussion
**Context**: Conversation about over-constraining the agent

---

## The Problem: Over-Constraining a Powerful LLM

The current V7 architecture adds multiple layers of structure that may unnecessarily restrict GPT-4o's natural reasoning capabilities:

### Current Architecture Layers

| Layer | Implementation | Restriction Imposed |
|-------|----------------|---------------------|
| **Classifier** | `classifier.py` → `is_exploratory` | Forces query into fixed path before LLM sees it |
| **Exploratory Path** | `exploratory.py` | Prescribes systematic multi-artifact retrieval pattern |
| **ReAct Path** | `react_agent.py` | Only used when classifier says "not exploratory" |
| **Synthesis Instructions** | Numbered steps in prompts | Tells LLM *how* to synthesize |
| **Domain Knowledge Rules** | "Low participation + high questions = facilitator" | Tells LLM *what* to conclude |

### The Core Insight

GPT-4o is already capable of:
- Understanding what information it needs for a query
- Deciding which tools to call
- Recognizing patterns (like facilitator behavior) from data
- Synthesizing coherent responses
- Citing evidence appropriately

**We don't need to tell it how to think. We need to give it good data to think with.**

---

## The Philosophy: Artifacts as Grounding Scaffolds

### Current Mental Model (Wrong)
```
Query → Classification Rules → Forced Retrieval Pattern → Prescriptive Synthesis → Response
         ↑                      ↑                         ↑
         Restriction            Restriction               Restriction
```

### Proposed Mental Model (Better)
```
┌─────────────────────────────────────────────────────────┐
│                   GPT-4o Reasoning                       │
│  (understands queries, recognizes patterns, synthesizes) │
└─────────────────────────┬───────────────────────────────┘
                          │
                          │ grounded by (not restricted by)
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Artifacts (Evidence)                    │
│  - Transcripts: what was actually said                   │
│  - Concept maps: how ideas connect                       │
│  - 7C analysis: collaboration quality metrics            │
│  - Speaker profiles: participation patterns              │
└─────────────────────────────────────────────────────────┘
```

**Key principle**: Artifacts ENRICH reasoning with real data. They don't RESTRICT how the LLM thinks.

---

## Proposed Changes

### 1. Remove or Bypass the Classifier

**Current**: `classifier.py` decides `is_exploratory` and forces different code paths.

**Proposed**: Let the LLM decide what information it needs through the ReAct loop.

**Rationale**: The LLM can read a query like "How did Lex engage across sessions?" and decide to call `list_sessions` then `get_speaker_profile` then `get_transcript` for relevant sessions. It doesn't need a classifier to tell it this is "exploratory."

### 2. Remove the Forced Exploratory Path

**Current**: `exploratory.py` systematically retrieves transcript + concept_map + 7C for each relevant session.

**Proposed**: Let the LLM call the tools it needs. If it wants all three artifact types, it will call all three.

**Rationale**: The exploratory path assumes we know what artifacts are needed. But the LLM might only need transcripts for some queries, or only 7C for others. Let it decide.

### 3. Simplify to Pure ReAct

**Current Architecture**:
```
Query → Classifier
            ├── exploratory=True  → exploratory.py (forced pattern)
            └── exploratory=False → ReAct loop (LLM decides)
```

**Proposed Architecture**:
```
Query → ReAct Loop (LLM always decides what tools to call)
```

### 4. Simplify Synthesis Prompts

**Current** (react_agent.py lines 294-306):
```
Instructions for synthesis:
1. Compare and contrast findings across sessions
2. Cite specific evidence from each session
3. Identify patterns or themes
4. Note any differences or contradictions
5. Provide a comprehensive answer

When interpreting speaker participation patterns:
- Low participation % + high question rate = facilitator
- Compare actual to equal share
- Consistent patterns = stable role
```

**Proposed**:
```
Synthesize a response to the user's query based on the evidence above.
Ground your claims in specific evidence from the artifacts.
```

**Rationale**: GPT-4o knows how to synthesize. It knows how to cite evidence. It can recognize a facilitator pattern from seeing "5% participation, 80% questions" without being told the rule.

### 5. Remove Explicit Domain Knowledge Rules

**Current**: We embed rules like "low participation + high questions = facilitator" in prompts.

**Proposed**: Remove these. The LLM can infer this from the data.

**Rationale**: If the speaker profile shows:
```
Participation: 8% of session (equal share would be 33%)
Question rate: 75% of their utterances are questions
```

GPT-4o can conclude "this looks like a facilitator/interviewer" without being told the rule. In fact, it might notice nuances we didn't encode (e.g., the *type* of questions asked).

---

## The Simpler Target Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Query                             │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                     ReAct Loop                           │
│                                                          │
│  LLM decides:                                            │
│  - What tools to call                                    │
│  - In what order                                         │
│  - When it has enough information                        │
│                                                          │
│  Available tools:                                        │
│  - list_sessions (discover what exists)                  │
│  - search_sessions (find by topic)                       │
│  - get_transcript (what was said)                        │
│  - get_concept_map (how ideas connect)                   │
│  - get_7c_analysis (collaboration quality)               │
│  - get_speaker_profile (participation patterns)          │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      Synthesis                           │
│                                                          │
│  Simple prompt:                                          │
│  "Based on the evidence, answer the user's query.        │
│   Ground your claims in the specific data retrieved."    │
└─────────────────────────────────────────────────────────┘
```

---

## What We Keep

1. **The 6 tools** - These are well-designed and provide the right data
2. **Tool output format** - LLM-ready text with clear structure
3. **Comparative metrics in speaker profile** - Raw data (not interpretations) for LLM to reason about
4. **Conversation memory** - Context across turns

## What We Remove

1. **Classifier** - No forced path decisions
2. **Exploratory path** - No prescriptive retrieval patterns
3. **Numbered synthesis instructions** - No telling LLM how to synthesize
4. **Domain knowledge rules** - No telling LLM what patterns mean

---

## Risk Assessment

### Risks of This Change

| Risk | Mitigation |
|------|------------|
| LLM might not retrieve enough data | Good tool descriptions guide it; can add "consider multiple sessions" hint if needed |
| LLM might hallucinate without grounding | Keep emphasis on "ground in evidence" in system prompt |
| Less predictable behavior | Trade-off for more flexible, intelligent responses |
| Harder to debug | Log tool calls and LLM reasoning |

### Benefits

| Benefit | Impact |
|---------|--------|
| Simpler architecture | Easier to maintain, fewer code paths |
| More flexible responses | Can handle novel queries we didn't anticipate |
| Leverages LLM intelligence | Uses GPT-4o's full reasoning capability |
| Faster iteration | Change prompts, not code, to adjust behavior |

---

## Implementation Steps (When Ready)

1. **Backup current working state** - Commit all current changes
2. **Create feature branch** - `git checkout -b simplify-agent-architecture`
3. **Modify react_agent.py** - Remove classifier check, always use ReAct
4. **Simplify synthesis prompts** - Remove numbered instructions and domain rules
5. **Test extensively** - Run all evaluation queries
6. **Compare results** - Before/after quality assessment
7. **Iterate** - Adjust based on findings

---

## Open Questions

1. Should we keep *any* classification (e.g., to set max_turns differently)?
2. Should we keep a minimal "grounding reminder" in synthesis prompts?
3. How do we ensure the LLM retrieves enough context for cross-session queries?
4. Should we add a "reflect" step where LLM assesses if it has enough evidence?

---

## Related Files

- `server/agent_v7/react_agent.py` - Main agent logic
- `server/agent_v7/classifier.py` - Current classifier (to be removed/simplified)
- `server/agent_v7/exploratory.py` - Current exploratory path (to be removed)
- `server/agent_v7/tools_v2.py` - Tool definitions (keep as-is)
- `server/agent_v7/graph_v2.py` - Graph structure (simplify)

---

*This document captures the design discussion for future reference. No changes have been made yet.*
