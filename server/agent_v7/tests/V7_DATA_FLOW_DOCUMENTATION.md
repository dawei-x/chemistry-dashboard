# V7 Agent Data Flow Documentation

**Generated**: 2026-01-16
**Purpose**: Document exact data flow and truncations through V7 pipeline phases

---

## Pipeline Architecture

```
Phase 1: Raw Tool Output
    ↓
Phase 2: Context Format (SUMMARY ONLY - for tool decision)
    ↓
Phase 3: Synthesis Format (DETAILED - sent to LLM for response)
    ↓
Phase 4: LLM generates response
```

**Important**: Phase 2 is intentionally a one-line summary. It's NOT meant to carry full data - it's only used by the agent to decide if more tools are needed. The actual data flows from Phase 1 → Phase 3.

---

## Truncation Rules Applied in Phase 3

| Data Type | Limit | Code Location |
|-----------|-------|---------------|
| Transcript lines | First 30 | `[:30]` |
| Graph lines | First 40 | `[:40]` |
| Explanation text | 200 chars | `[:200]` |
| Quote text | 150 chars | `[:150]` |
| Reason text | 100 chars | `[:100]` |
| Coded segments | First 3 per dimension | `[:3]` |

---

## Test Results: Session 25 (Abundance)

### Transcript Tool

**Raw Data**:
- Total lines: 18
- Total characters: 4,524
- Speakers: Lex, Ezra, Derek

**After Truncation**:
- Lines kept: 18 (all)
- Lines lost: 0
- Speakers preserved: All ✓

**Verdict**: ✓ NO DATA LOSS

---

### Concept Map Tool

**Raw Data**:
- Total lines: 32
- Total characters: 1,456
- Nodes: 15, Edges: 15

**After Truncation**:
- Lines kept: 32 (all)
- Lines lost: 0

**Verdict**: ✓ NO DATA LOSS

---

### 7C Analysis Tool

**Raw Data**:
- 7 dimensions with scores, explanations, coded segments

**After Truncation**: ⚠️ **24 TRUNCATION ISSUES**

#### Detailed Truncation Analysis:

| Dimension | Explanation | Coded Segments | Quote/Reason Truncations |
|-----------|-------------|----------------|-------------------------|
| climate | 340→200 chars | 1 kept | 2 truncations |
| communication | 340→200 chars | 3 of 9 kept (6 LOST) | 3 truncations |
| contribution | 250→200 chars | 0 (none in raw) | 0 |
| conflict | 290→200 chars | 0 (none in raw) | 0 |
| context | 311→200 chars | 3 of 4 kept (1 LOST) | 4 truncations |
| constructive | 267→200 chars | 3 of 6 kept (3 LOST) | 5 truncations |
| compatibility | 269→200 chars | 0 (none in raw) | 0 |

**Lost Data Examples**:

1. **Communication dimension lost 6 coded segments**:
   - `"Ezra at 1:59: the left is tends to be more worried about..."`
   - `"Derek at 2:29: a thermostatic public opinion in American politics..."`
   - `"Derek: clicks into focus, a new skill that is suddenly in critical demand..."`
   - (and 3 more)

2. **Explanation truncations** (all 7 dimensions):
   - Climate: Lost `"...ong participants. The interaction is primarily intellectual, with a focus on exchanging ideas rather than fostering a supportive atmosphere."`
   - Communication: Lost `"...espond thoughtfully to each other. However, the discussion is somewhat one-sided, with Ezra and Derek providing more extended contributions."`

3. **Quote/Reason mid-sentence cuts**:
   - Quote cut: `"...as one of the most intellectually rigorous"` (missing: `" voices on the left."`)
   - Reason cut: `"...creating an emotionally safe and respectful envir"` (missing: `"onment for the discussion."`)

---

## Exact Output Sent to LLM

### Phase 2: Context Format (Summary Only)

```
[get_transcript] Session 'Abundance': 18 utterances
```

```
[get_7c_analysis] Session 'Abundance': Average 69.3/100
```

```
[get_concept_map] Session 'Abundance': 15 nodes, 15 edges
```

**Note**: This is intentionally minimal - used only for deciding if more tools are needed.

---

### Phase 3: Synthesis Format (What LLM Sees)

#### Transcript (4,573 chars - COMPLETE)

```
## get_transcript
Session: Abundance
Transcript:
[00:13] Lex: spectrum. As there have been a fan of yours for a long time, uh you're often referred to at least I think of you as one of the most intellectually rigorous voices on the left. Uh
[00:23] Lex: Can you try to define? Can you define the ideals and the vision of the American left? Oh good sir. Start small here. And maybe contrast them
[00:30] Ezra: with the American right. Sure. Um, so the thing I should say here is that you can define the left in different ways...
[01:00] Ezra: entire lives
[01:02] Ezra: And as such, the people for whom it works out well, we don't deserve all of that...
[01:32] Ezra: do a kind of universal dignity, right? So people can have lives of flourishing...
[01:37] Ezra: is fundamentally more skeptical of capitalism...
[01:59] Ezra: the left is tends to be more worried about the fact that you can get rich uh building coal fire power plants...
[02:15] Ezra: And so, there's a set of goals about regulating the the unchecked potential of capitalism...
[02:29] Derek: a thermostatic public opinion in American politics...
[02:55] Derek: I also explains why
[03:00] Derek: Democrats and Republicans have struggled to hold on to power for 6-year, 8-year, 12-year terms...
[03:16] Derek: We were just talking off-camera about how every age of communications technology revolution
[03:23] Derek: clicks into focus, a new skill that is suddenly in critical demand for the electron, right?...
[03:53] Derek: Because he's so damn handsome, and also just electrically compelling on a screen.
[03:59] Derek: We've a new screen technology right now, which is not just television and steroids...
[04:11] Derek: individuals, influencers, and even celebrities and politicians who were good at something like live wire authenticity...
[04:24] Derek: Trump is an absolute marvel at performing authenticity, even when the audience somehow acknowledges that he might be bullshit.
```

#### 7C Analysis (4,167 chars - TRUNCATED)

```
## get_7c_analysis
Session: Abundance
Average Score: 69.3/100
  climate: 60/100 - The discussion environment appears respectful and comfortable, allowing participants to express their ideas freely. However, there is limited evidence of explicit encouragement or emotional support am  ← TRUNCATED
    Quote: "Lex: As there have been a fan of yours for a long time, uh you're often referred to at least I think of you as one of the most intellectually rigorous"  ← TRUNCATED
    Reason: Lex expresses admiration and respect towards Ezra, creating an emotionally safe and respectful envir  ← TRUNCATED
  communication: 85/100 - The communication is clear and active, with participants articulating their points well and building on each other's ideas. There is a strong flow of information, and participants seem to listen and r  ← TRUNCATED
    Quote: "Lex: Can you try to define? Can you define the ideals and the vision of the American left?"
    Reason: Lex is clearly articulating a question, facilitating effective information exchange by asking Ezra t  ← TRUNCATED
    Quote: "Ezra: Um, so the thing I should say here is that you can define the left in different ways."
    Reason: Ezra is sharing information and setting the stage for a discussion by acknowledging different perspe  ← TRUNCATED
    Quote: "Ezra: I always think this is hard to talk about because what we call unchecked capitalism is nevertheless very much supported by government."
    Reason: Ezra is effectively sharing his thoughts and engaging in a discussion about the complexities of capi  ← TRUNCATED
    [6 MORE CODED SEGMENTS NOT SHOWN - LOST]
  contribution: 65/100 - While the main contributors, Ezra and Derek, provide substantial input, the participation is not entirely balanced. Lex facilitates the discussion but contributes less to the content, indicating an im  ← TRUNCATED
  conflict: 50/100 - There is no evidence of conflict or disagreement in the transcript, which suggests a lack of opportunity to evaluate conflict resolution skills. The discussion is harmonious, but the absence of differ  ← TRUNCATED
  context: 75/100 - Participants demonstrate a good awareness of the context, discussing political ideologies and media dynamics with depth and relevance. The conversation is well-situated within the broader socio-politi  ← TRUNCATED
    [3 quotes shown, 1 LOST]
  constructive: 80/100 - The discussion is productive, with participants collaboratively building on each other's ideas and contributing to a deeper understanding of the topics. There is evidence of mutual learning, as partic  ← TRUNCATED
    [3 quotes shown, 3 LOST]
  compatibility: 70/100 - The participants demonstrate a compatible work style, with a shared focus on intellectual discussion and analysis. There is a synergy in their approach to exploring complex topics, though the conversa  ← TRUNCATED
```

#### Concept Map (1,625 chars - COMPLETE)

```
## get_concept_map
Session: Abundance
Nodes: 15, Edges: 15
Speaker contributions:
  Lex: 2 contributions
  Derek: 6 contributions
  Ezra: 7 contributions
Concept graph:
[idea] Lex: "intellectually rigorous voices on the left"
   - elaborates -> [question] Lex: "define the ideals and vision of the American left"

[idea] Derek: "Donald Trump as a media figure"
   - contrasts_with -> [idea] Derek: "new screen technology"
   - relates_to -> [idea] Derek: "communications technology revolution"

[idea] Derek: "communications technology revolution"
   - elaborates -> [idea] Derek: "new screen technology"
   - relates_to -> [idea] Derek: "performing authenticity"

[idea] Derek: "performing authenticity"
   - relates_to -> [idea] Derek: "new screen technology"

[idea] Ezra: "life is unfair"
   - relates_to -> [goal] Ezra: "universal dignity for flourishing lives"
   - relates_to -> [solution] Ezra: "rectify unfairness, not perfect equality"

[goal] Ezra: "universal dignity for flourishing lives"
   - supports -> [solution] Ezra: "rectify unfairness, not perfect equality"

[idea] Ezra: "skepticism of unchecked capitalism"
   - contrasts_with -> [problem] Derek: "parties overreach and lose power"
   - relates_to -> [idea] Ezra: "markets supported by government"
   - supports -> [goal] Ezra: "regulating unchecked capitalism"

[goal] Ezra: "regulating unchecked capitalism"
   - relates_to -> [problem] Ezra: "exploitation of workers"

[idea] Derek: "thermostatic public opinion"
   - relates_to -> [idea] Derek: "Donald Trump as a media figure"
   - relates_to -> [problem] Derek: "parties overreach and lose power"
```

---

## Summary

| Tool | Data Loss | Impact |
|------|-----------|--------|
| `get_transcript` | None | ✓ All speakers and quotes preserved |
| `get_concept_map` | None | ✓ All nodes and edges preserved |
| `get_7c_analysis` | **24 truncations** | ⚠️ Lost coded segments, incomplete explanations |

### Recommendations

1. **Consider increasing limits for 7C analysis**:
   - `[:200]` → `[:400]` for explanations
   - `[:150]` → `[:250]` for quotes
   - `[:3]` → `[:5]` for coded segments per dimension

2. **Phase 2 is working as designed** - it's a summary, not data carrier

3. **For larger sessions**, transcript and concept map limits may need adjustment

---

## Code Locations

- Tool definitions: `tools_v2.py:56-324`
- Evidence formatting: `react_agent.py:422-507`
- Truncation trace script: `tests/trace_truncations.py`
