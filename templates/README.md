# Daily Intelligence System

A comprehensive personal performance optimization system combining daily data capture, Claude AI-powered analysis, and automated reporting through Zapier integration.

## 📋 System Components

### 1. **Daily Intelligence Capture Template** (`TPL___Daily_Intelligence_Capture.md`)

The core input template that captures daily data across 8 key metrics, 4 active projects, free-write reflections, and optional Internal Family Systems (IFS) work.

**Metrics Tracked (8 dimensions):**
- Health & Energy: Sleep quality, Physical energy
- Productivity & Flow: Focus score, Task completion rate
- Learning & Growth: Knowledge intake, Skill practice
- Relationships & Social: Connection quality, Emotional balance

**Components:**
- Structured YAML metrics (1-10 scales)
- Project status tracking (4 active projects)
- Free-write journal (morning, afternoon, evening reflections)
- Optional IFS parts work
- Tomorrow's intention

**Input Time:** ~15-20 minutes daily

**Best Practice:** Complete in evening or morning depending on your preference

---

### 2. **Claude AI Prompt Suite** (5 Specialized Prompts)

Use these prompts with Claude API (via Zapier or direct integration) to generate insights at different time scales.

#### **Daily Intelligence Generator** (`PROMPT___Daily_Intelligence_Generator.md`)
- **Input:** Single day's capture data
- **Output:** Daily report with metrics analysis, project velocity, patterns, insights, and recommendations
- **Frequency:** Daily (automated or manual)
- **Output Length:** ~1,500-2,000 tokens

#### **Weekly Synthesis Generator** (`PROMPT___Weekly_Synthesis_Generator.md`)
- **Input:** 7 daily reports + consolidated data
- **Output:** Weekly narrative with trajectory, metrics trends, system-level patterns, and next-week orientation
- **Frequency:** Weekly (Sundays recommended)
- **Output Length:** ~2,000-2,500 tokens

#### **Monthly Deep Dive Generator** (`PROMPT___Monthly_Deep_Dive_Generator.md`)
- **Input:** 4 weekly syntheses + cumulative month data
- **Output:** Deep analysis of domain health, project portfolio, pattern archaeology, IFS work, and monthly recommendations
- **Frequency:** Monthly (end of month/month start)
- **Output Length:** ~3,000-4,000 tokens

#### **Pattern Detection** (`PROMPT___Pattern_Detection.md`)
- **Input:** 7-30 days of capture data
- **Output:** Correlation analysis, causal chains, time-domain patterns, peak performance profile, anomalies
- **Frequency:** Every 2 weeks or monthly
- **Output Length:** ~2,500-3,500 tokens

#### **Recommendation Engine** (`PROMPT___Recommendation_Engine.md`)
- **Input:** Intelligence report(s) + analysis
- **Output:** Prioritized recommendations across immediate actions, weekly optimizations, monthly initiatives, skill development, and experiments
- **Frequency:** Weekly or after major analysis
- **Output Length:** ~2,000-3,000 tokens

---

## 🔄 Workflow: Three Implementation Paths

### Path 1: Manual (Most Flexible)
```
Create Capture → Store in Document/Notes →
Copy to Claude.com → Generate Report → Review & Act
```
- ✅ No setup required
- ✅ Full control over prompts
- ❌ More manual steps
- ⏱️ Time per report: 10-15 min

### Path 2: Claude API Direct Integration (Most Developer-Friendly)
```
Create Capture → API Call with Prompt →
Store Response → Review & Act
```
- ✅ Programmatic control
- ✅ Easy to customize
- ⚙️ Requires coding
- ⏱️ Time per report: Instant

### Path 3: Zapier Automation (Most Turnkey) ⭐ **Recommended**
```
Create Capture → Zapier Trigger → Claude API Call →
Auto-store in Notion/Docs → Email Notification → Review & Act
```
- ✅ Fully automated
- ✅ No coding required
- ✅ Report auto-stored
- ⏱️ Time per report: Instant + review time

**See:** `../zapier/ZAPIER_AUTOMATION_SETUP.md`

---

## 🚀 Getting Started

### Quick Setup (Manual Mode)

1. **Create daily captures:**
   - Copy `TPL___Daily_Intelligence_Capture.md` to your notes app
   - Fill out daily at end of day or morning

2. **Generate insights:**
   - Open one daily prompt from `/prompts/` folder
   - Go to Claude.com and paste prompt + capture data
   - Save generated report

3. **Review weekly:**
   - Collect 7 daily reports
   - Use `PROMPT___Weekly_Synthesis_Generator.md`
   - Generate weekly synthesis

### Complete Setup (Zapier Automation)

See full guide: `../zapier/ZAPIER_AUTOMATION_SETUP.md`

1. Get Claude API key (5 min)
2. Set up Zapier automation (20-30 min)
3. Create first capture (15 min)
4. Receive automated report (30-60 sec)
5. Review and iterate (10 min)

---

## 📊 Data Architecture

```
Daily Capture
  ├── 8 Metrics (scored 1-10 + context)
  ├── 4 Projects (status + progress)
  ├── Journal (free-write 3 sessions)
  └── Optional IFS work

        ↓ Claude Processing ↓

Daily Report
  ├── Metrics analysis
  ├── Project velocity
  ├── Pattern recognition
  └── Recommendations

        ↓ After 7 days ↓

Weekly Synthesis
  ├── Metrics trajectory
  ├── Project portfolio health
  ├── System-level patterns
  └── Next week orientation

        ↓ After 4 weeks ↓

Monthly Deep Dive
  ├── Domain analysis (4 areas)
  ├── Project lifecycle
  ├── Pattern archaeology
  ├── IFS quarterly report
  └── Month +1 blueprint
```

---

## 🎯 Use Cases

### Personal Development Tracking
Track growth across health, skills, and relationships. Identify patterns in what enables peak performance.

### Project Management
Track 4 active projects simultaneously. Get automated velocity reports and blocker alerts.

### Wellness & Recovery
Monitor energy levels and emotional balance. Get early warnings before burnout.

### Learning Optimization
Track knowledge intake and skill development. Identify optimal learning conditions.

### Internal Work (IFS)
Track parts' activities and progress toward Self-leadership integration.

### Habit Formation
Identify which habits/practices drive improvements. Get recommendations for new habits.

### Goal Achievement
Quarterly alignment of daily actions with monthly and longer-term goals.

---

## 💡 Key Features

### Holistic Metrics
- Not just productivity: includes health, learning, relationships, emotional balance
- Quantified (1-10 scales) and qualitative (context and notes)
- 8 dimensions covering major life domains

### Project-Centric
- 4 concurrent projects (recommended optimal number)
- Status tracking: Not Started | In Progress | Blocked | Complete
- Progress percentage and milestone tracking
- Blocker identification and support needed

### Free-Write Integration
- Unstructured reflection alongside metrics
- Captures nuance that numbers miss
- Morning intentions, afternoon insights, evening reflections
- Reveals patterns in thinking and emotional state

### Automated Analysis
- Claude AI reviews data daily/weekly/monthly
- Identifies correlations and causal chains
- Detects patterns humans miss
- Provides specific, actionable recommendations

### Internal Family Systems Integration (Optional)
- For those doing IFS therapy or work
- Track parts' protective strategies
- Monitor Self-leadership development
- Understand internal system dynamics

### Intelligent Reporting
- Auto-generates reports at multiple time scales
- Smart summaries and deep analysis available
- Pattern recognition and anomaly detection
- Personalized recommendations

---

## 📈 Success Metrics

Over time, this system helps you track:

- ✅ Metric trends (improving, stable, declining)
- ✅ Project completion velocity
- ✅ Energy and sustainability patterns
- ✅ High-performance conditions (how to replicate)
- ✅ Recovery speed (resilience)
- ✅ Capability development
- ✅ Goal progress alignment
- ✅ Internal system integration (IFS)

---

## 🔧 Technical Requirements

### For Manual Mode
- Text editor or notes app
- Access to Claude.com
- ~20 minutes per day

### For Zapier Automation
- Claude API key ($0.01-1.00/month estimated)
- Zapier account (free plan sufficient)
- Document storage (Notion, Google Docs, etc.)
- ~45 minutes setup time

### Recommended Stack
- **Capture Storage:** Notion database or Google Forms
- **Reporting:** Notion or Google Docs
- **Automation:** Zapier
- **API:** Claude API (Anthropic)

---

## 📚 Files Included

### Templates
- `TPL___Daily_Intelligence_Capture.md` - Daily capture template

### Prompts
- `PROMPT___Daily_Intelligence_Generator.md` - Daily analysis
- `PROMPT___Weekly_Synthesis_Generator.md` - Weekly synthesis
- `PROMPT___Monthly_Deep_Dive_Generator.md` - Monthly analysis
- `PROMPT___Pattern_Detection.md` - Pattern recognition
- `PROMPT___Recommendation_Engine.md` - Personalized recommendations

### Integration
- `../zapier/ZAPIER_AUTOMATION_SETUP.md` - Full Zapier setup guide

---

## 🚀 Next Steps

1. **Start with Daily Captures**
   - Copy template to your preferred notes app
   - Complete first capture today
   - Get familiar with the metrics and structure

2. **Generate Your First Report**
   - Collect 3-7 days of captures
   - Open `PROMPT___Daily_Intelligence_Generator.md`
   - Use with Claude.com to generate reports
   - Notice what insights emerge

3. **Optimize Based on Patterns**
   - After 2-3 weeks, run `PROMPT___Pattern_Detection.md`
   - Identify what enables good days
   - Use `PROMPT___Recommendation_Engine.md` for next steps

4. **Set Up Automation (Optional)**
   - Follow `../zapier/ZAPIER_AUTOMATION_SETUP.md`
   - Save time on report generation
   - Receive insights automatically

5. **Iterate & Refine**
   - Adjust metrics if some don't resonate
   - Modify prompts for your specific context
   - Build templates for recurring insights

---

## 🤔 FAQ

**Q: Is this just another productivity system?**
A: No. It's holistic, including health, relationships, learning, and internal development alongside productivity.

**Q: Do I need to use all 8 metrics?**
A: Start with all 8 to identify what matters most, then focus. The system is customizable.

**Q: Can I use this without Claude API/Zapier?**
A: Yes! You can manually use prompts with Claude.com or even without AI entirely (captures alone are valuable).

**Q: How long does this actually take?**
A: ~20 min daily for capture, 5-10 min to review auto-generated report. Weekly synthesis is 30 min. Monthly is 1-2 hours. Total: 2.5-3 hours per week.

**Q: What if I miss days?**
A: The system is forgiving. Missing a day or two doesn't break the pattern. The longer patterns matter more than daily consistency.

**Q: Can I adjust the metrics?**
A: Absolutely. These are suggested starting points. Customize to your life, values, and goals.

**Q: What about privacy?**
A: Your captures are stored where you choose (Notion, Google Docs, etc.). If using Zapier + Claude, consider what data you're comfortable sending.

---

## 📞 Support

- Questions about prompts? Review the prompt file headers
- Technical issues with Zapier? See troubleshooting section in setup guide
- Claude API questions? Visit Anthropic docs: https://docs.anthropic.com
- Want to modify the system? All files are editable markdown/text

---

## 📄 License

These templates and prompts are provided as-is for personal use.

---

**System Version:** 1.0
**Last Updated:** 2025-03-05
**Created for:** Tree of Thoughts Project
