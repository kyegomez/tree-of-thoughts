# Pattern Detection Prompt

## Purpose
Identify hidden patterns, correlations, and causal relationships across daily captures to reveal system dynamics, trigger points, and leverage opportunities.

## Input
Collection of daily capture data (typically 7-30 days) containing:
- Metrics arrays (8 dimensions per day)
- Project status and progress data
- Journal entries and reflections
- Blockers and obstacles noted
- Internal/emotional states (from journal)

## Task
Discover and articulate patterns across multiple analytical dimensions:

### 1. **Correlation Detection** (What moves together?)

**Positive Correlations:**
- Which metrics rise together?
  - Example: "On days when sleep quality > 7, focus score averages +2 higher"
- Which external factors predict metric improvements?
  - Example: "Deep work sessions before noon correlate with +1.5 productivity boost"
- What project progress patterns hold across similar work?

**Negative Correlations:**
- What metric pairs work against each other?
  - Example: "High social time correlates with reduced deep work hours"
- What conditions trigger declines in key metrics?
- Where is there hidden tradeoff?

**Neutral/Surprising Findings:**
- What correlations are weaker than expected?
- What doesn't correlate despite assumption it would?
- Unexpected data patterns

### 2. **Causal Chain Analysis** (Why do patterns form?)

**Primary Drivers:**
- What's the root cause of observed patterns?
- Is it external (environment, schedule) or internal (capacity, state)?
- Trace causality: Action → Effect → Secondary Effect

**Trigger Events:**
- What consistently precedes metric dips?
  - Low-quality interactions?
  - Schedule disruptions?
  - Insufficient recovery?
  - Decision fatigue?

**Amplification Loops:**
- Where do positive feedbacks exist? (virtuous cycles)
  - Example: "Good sleep → higher energy → better work → more satisfaction → sleep confidence"
- Where do negative feedbacks exist? (vicious cycles)
  - Example: "Missed deadline → stress → poor sleep → low energy → lower output"

**Leverage Points:**
- Where can one small change create outsized effects?
- What's the highest-ROI intervention point in observed chains?

### 3. **Time-Domain Patterns**

**Daily Rhythms:**
- Are there consistent time-of-day patterns?
- When is energy/focus highest and lowest?
- When are social/emotional needs highest?
- Optimal work windows vs. recovery windows

**Weekly Cycles:**
- Do specific days show consistent characteristics?
  - Example: "Mondays show 15% lower focus, peak on Tuesdays/Wednesdays"
- When in the week are projects most likely to hit blockers?
- Recovery patterns within the week

**Frequency Analysis:**
- How often do high-performance days occur?
- How often do energy crashes occur?
- Spacing and distribution of events

### 4. **State-Space Patterns** (What constellation of conditions enables peak performance?)

**Peak Performance Conditions:**
- What combination of factors produces optimal days?
- Is it reproducible? (Same factors, same outcome?)
- Minimum viable conditions for good day

**Struggle Conditions:**
- What factor combinations predict difficult days?
- How reliably can difficult periods be predicted?
- Early warning signs

**Transition Patterns:**
- How does system move from struggle to peak?
- Speed and predictability of transitions
- Intervention points during transitions

### 5. **Project-Specific Patterns**

**By Project Type:**
- Do different project types have different success profiles?
- Which projects enable energy vs. drain energy?
- Project-specific blockers that repeat

**Progress Velocity Patterns:**
- Why do some projects accelerate while others decelerate?
- Workload distribution effects
- Momentum factors (what keeps projects moving?)

**Context-Switching Effects:**
- Impact of juggling N projects simultaneously
- Optimal project portfolio size
- Cost of context switches

### 6. **Internal State Patterns** (From journal analysis)

**Emotional Weather:**
- Baseline emotional state throughout period
- Emotional volatility vs. stability
- What triggers shifts?

**Confidence/Capability Perception:**
- Days of high self-efficacy vs. self-doubt
- Impact on actual performance
- Belief-reality misalignment

**Narrative Patterns:**
- Common themes in journal reflections
- Stories told repeatedly
- Hidden assumptions or beliefs

**IFS-Related Patterns:**
- Which parts activate in response to what triggers?
- Parts' protective strategies effectiveness
- Parts coalition vs. conflict patterns

### 7. **System Resilience Patterns**

**Recovery Capability:**
- How quickly does system bounce back from disruption?
- Recovery time vs. disruption magnitude
- Recovery capacity over time (improving/declining?)

**Stress Response Patterns:**
- Under pressure, what dimensions suffer first?
- Which metrics are most vulnerable?
- Stress-cascade sequences

**Adaptation Patterns:**
- How does system adapt to new demands?
- Learning speed
- Habit formation velocity

### 8. **Anomaly Detection** (What breaks the pattern?)

**Positive Outliers:**
- Best days: What made them different?
- High-achievement periods: Setup conditions
- Breakthrough moments: Preceding patterns

**Negative Outliers:**
- Worst days: Root cause analysis
- Energy crashes: Preceding conditions
- Derailment events: What went wrong?

**Unexplained Variance:**
- Data points that don't fit patterns
- Possible hidden variables
- Measurement noise vs. real signal

## Output Format
```
# Pattern Detection Report - [PERIOD]

## 📊 Correlation Matrix
### Strong Positive Correlations
- [Metric A] ↔ [Metric B]: r=0.8X
  - Interpretation: ...
  - Examples: ...

### Strong Negative Correlations
- [Metric A] ↔ [Metric B]: r=-0.7X
  - Interpretation: ...

### Surprising Findings
- ...

## 🔗 Causal Chains
### Primary Drivers
- ...

### Trigger Events
- ...

### Amplification Loops
#### Virtuous Cycles
- ...
#### Vicious Cycles
- ...

### Leverage Points
1. [Point 1]: [Expected Impact]
2. [Point 2]: [Expected Impact]

## ⏰ Time-Domain Patterns
### Daily Rhythms
- Best time: ...
- Worst time: ...

### Weekly Cycles
- ...

### Frequency Analysis
- ...

## 🎯 Peak Performance Profile
**Conditions:** ...
**Reproducibility:** ...
**Minimum Viable:** ...

## 📈 Project Patterns
- ...

## 💭 Internal State Patterns
- ...

## 💪 Resilience Patterns
- ...

## 🚨 Anomalies
### Positive Outliers
- ...
### Negative Outliers
- ...

## 🎯 Key Insights
1. ...
2. ...
3. ...
```

## Tone
- Analytical and precise
- Pattern-focused (concrete examples)
- Curiosity-driven
- Humble about causality (suggest, don't declare)
- Systems-aware
