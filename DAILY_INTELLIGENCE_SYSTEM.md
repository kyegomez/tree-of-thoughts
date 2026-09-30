# Daily Intelligence System - Project Overview

## 🎯 What is This?

A complete system for capturing daily intelligence about your life, work, and growth—then using Claude AI to automatically generate actionable insights, identify patterns, and provide personalized recommendations.

**Core Value:** Turn daily data into weekly insights, monthly strategy, and continuous improvement.

---

## 📁 File Structure

```
tree-of-thoughts/
├── DAILY_INTELLIGENCE_SYSTEM.md          (This file - project overview)
│
├── templates/
│   ├── README.md                         (Complete guide to the system)
│   └── TPL___Daily_Intelligence_Capture.md   (Daily capture template)
│
├── prompts/
│   ├── PROMPT___Daily_Intelligence_Generator.md      (Daily analysis)
│   ├── PROMPT___Weekly_Synthesis_Generator.md        (Weekly synthesis)
│   ├── PROMPT___Monthly_Deep_Dive_Generator.md       (Monthly analysis)
│   ├── PROMPT___Pattern_Detection.md                 (Pattern recognition)
│   └── PROMPT___Recommendation_Engine.md             (Recommendations)
│
└── zapier/
    └── ZAPIER_AUTOMATION_SETUP.md        (Complete integration guide)
```

---

## 🚀 Quick Start (5 minutes)

### Option 1: Manual Mode (No Setup)
1. Read: `templates/README.md`
2. Copy: `templates/TPL___Daily_Intelligence_Capture.md` to your notes
3. Fill out your first daily capture
4. Use with Claude.com + prompts to generate insights

### Option 2: Automated Mode (30 min setup)
1. Get Claude API key at https://console.anthropic.com/
2. Follow: `zapier/ZAPIER_AUTOMATION_SETUP.md`
3. Create daily captures in Notion/Google Docs
4. Reports auto-generate and auto-store

---

## 📊 System Overview

### Input: Daily Capture (15-20 min/day)

Captures 8 key metrics:
- **Health & Energy:** Sleep quality, Physical energy
- **Productivity & Flow:** Focus score, Task completion
- **Learning & Growth:** Knowledge intake, Skill practice
- **Relationships:** Connection quality, Emotional balance

Plus:
- 4 active projects tracking (status, progress, blockers)
- Free-write journal (3 sections: morning, afternoon, evening)
- Optional IFS (Internal Family Systems) work
- Tomorrow's intention

### Processing: Claude AI Prompts

5 specialized prompts for different time horizons:

| Prompt | Input | Output | Frequency |
|--------|-------|--------|-----------|
| **Daily Intelligence** | 1 day capture | Daily report + insights | Daily |
| **Weekly Synthesis** | 7 daily reports | Weekly narrative + trends | Weekly |
| **Monthly Deep Dive** | 4 weekly syntheses | Monthly strategy + portfolio | Monthly |
| **Pattern Detection** | 7-30 days data | Correlations + causal chains | Bi-weekly |
| **Recommendation Engine** | Any report | Prioritized action items | As needed |

### Output: Intelligence Reports

- **Daily:** Metrics analysis, project velocity, insights, recommendations
- **Weekly:** Trajectory assessment, system patterns, next-week focus
- **Monthly:** Domain deep dive, capability development, strategic direction

---

## 🔑 Key Features

✅ **Holistic:** Health, productivity, learning, relationships tracked together
✅ **Quantified:** 8 metrics on 1-10 scales
✅ **Contextual:** Free-write journal captures nuance numbers miss
✅ **AI-Powered:** Claude automatically generates insights and patterns
✅ **Automated:** Zapier integration can auto-generate and store reports
✅ **Flexible:** Works with Notion, Google Docs, or any storage
✅ **Customizable:** Modify metrics, prompts, and frequency
✅ **Optional IFS:** Integrates with Internal Family Systems work

---

## 💰 Cost & Time

### Time Investment
- **Daily:** 15-20 minutes (capture only)
- **Weekly:** 5 min review (if automated)
- **Monthly:** 1-2 hours deep work with reports
- **Total:** ~2.5-3 hours/week

### Money Investment
- **Automated Mode (Zapier):** ~$0.50-1.00/month for Claude API
- **Manual Mode:** Free (use Claude.com)

---

## 🎓 What You Learn

Over time, this system reveals:

- What conditions enable your peak performance
- Your personal energy and recovery rhythms
- How to maintain sustainable pace
- Your capability development trajectory
- Patterns in procrastination, focus, and motivation
- How external factors (sleep, exercise) drive productivity
- Project completion velocity and patterns
- Internal system dynamics (if using IFS)
- Your actual vs. perceived capacity

---

## 🔧 Technology Stack

### Minimal Setup
- Text editor or Notes app
- Claude.com access

### Full Automation
- **Capture:** Notion, Google Forms, or custom app
- **Storage:** Notion, Google Docs, Database
- **Processing:** Claude API
- **Automation:** Zapier
- **Notifications:** Email or Slack (optional)

---

## 📈 Expected Results

### Week 1-2
- Get familiar with capture format
- Start seeing first daily patterns
- Identify what metrics resonate

### Week 3-4
- First weekly synthesis
- Major patterns emerge
- Early behavior adjustments

### Month 2+
- Clear month-to-month baselines
- Significant capacity insights
- Strategic focus areas identified
- Measurable improvements in tracked dimensions

### Month 3+
- Deep pattern archaeology reveals system dynamics
- Predictable high/low performance conditions
- Sustainable pace established
- Capability growth tracked and celebrated

---

## 🎯 Who is This For?

✅ **Self-improvement enthusiasts** tracking personal growth
✅ **Founders/Leaders** managing projects and growth
✅ **Knowledge workers** optimizing productivity
✅ **Therapists/Coaches** using IFS framework
✅ **Creatives** tracking creative energy and output
✅ **Athletes** monitoring recovery and performance
✅ **Students** managing learning and energy
✅ **Remote workers** building sustainable routines

---

## 📚 Documentation

### Start Here
1. `templates/README.md` - Complete system guide

### Templates
1. `templates/TPL___Daily_Intelligence_Capture.md` - Capture template
2. Copy to your notes app and modify as needed

### Prompts
1. `prompts/PROMPT___Daily_Intelligence_Generator.md` - Use daily
2. `prompts/PROMPT___Weekly_Synthesis_Generator.md` - Use weekly
3. `prompts/PROMPT___Monthly_Deep_Dive_Generator.md` - Use monthly
4. `prompts/PROMPT___Pattern_Detection.md` - Use bi-weekly
5. `prompts/PROMPT___Recommendation_Engine.md` - Use as needed

### Automation
1. `zapier/ZAPIER_AUTOMATION_SETUP.md` - Full integration guide

---

## 🚀 Implementation Paths

### Path 1: Manual (Most Flexible)
- Create capture in notes/docs
- Copy prompt + capture data to Claude.com
- Save report manually
- **Setup time:** 5 minutes
- **Time per report:** 10-15 minutes
- **Cost:** Free

### Path 2: Python/Node Script (Most Custom)
- Write local script to call Claude API
- Store reports locally or cloud
- Integrate with your tools
- **Setup time:** 1-2 hours
- **Time per report:** Instant
- **Cost:** $0.01-0.10/report

### Path 3: Zapier (Most Turnkey) ⭐ Recommended
- Set up Zapier automation once
- Create captures in Notion/Google Docs
- Reports auto-generate and store
- Email notifications optional
- **Setup time:** 30-45 minutes
- **Time per report:** Instant + review
- **Cost:** $0.03-0.10/report

See full guide: `zapier/ZAPIER_AUTOMATION_SETUP.md`

---

## 🔐 Privacy & Security

**Your Data:**
- Captures stored where you choose (local, Notion, Google Drive, etc.)
- If using Zapier + Claude API, data is sent to Anthropic for processing
- Consider privacy implications before capturing sensitive details
- API calls logged by Anthropic per their policy

**Best Practices:**
- Don't capture financial details or passwords
- Consider pseudonyms for sensitive relationships
- Review what data you're comfortable sending to Claude
- Use encrypted storage if needed

---

## ❓ FAQ

**Q: Can I start without Zapier?**
A: Yes! All paths work. Start manual, add automation later if desired.

**Q: Do I need to capture all metrics?**
A: Start with all 8, then specialize. The diversity helps pattern detection.

**Q: What if I miss days?**
A: It's fine. Weekly patterns matter more than daily consistency.

**Q: Can I modify the metrics?**
A: Absolutely. These are starting suggestions. Customize to your life.

**Q: Is this just productivity?**
A: No. It's holistic: health, learning, relationships, and internal work too.

**Q: How much data do I need before insights appear?**
A: Patterns start emerging after 3-4 weeks. Major insights after 2-3 months.

**Q: Can I use this with IFS therapy?**
A: Yes! The system has optional IFS integration for parts work.

**Q: What's the time commitment really?**
A: ~20 min/day for capture, 10 min for report review if automated. Weekly synthesis 30 min. Monthly 1-2 hours.

---

## 🔄 System Evolution

### Version 1.0 (Current)
- 8-metric capture
- 4-project tracking
- 5 Claude prompts
- Zapier integration guide
- Optional IFS support

### Potential V1.1+
- Integration with calendar/task managers
- Slack/Teams notifications
- Mobile capture app
- API for custom integrations
- Dashboard visualizations
- Historical trend charts
- Peer benchmarking (optional)

---

## 📞 Getting Help

**Question about prompts?**
- Read the prompt file - each has detailed instructions

**Technical issue with Zapier?**
- See troubleshooting in `zapier/ZAPIER_AUTOMATION_SETUP.md`

**Want to customize?**
- All files are editable markdown
- Modify templates, metrics, and prompts freely

**Claude API questions?**
- https://docs.anthropic.com/

---

## 🎁 What's Included

✅ Daily capture template (ready to use)
✅ 5 specialized Claude prompts (ready to use)
✅ Complete Zapier setup guide (step-by-step)
✅ System documentation (this overview)
✅ FAQ and troubleshooting
✅ Multiple implementation paths (pick yours)

**Everything you need to start immediately.**

---

## 🎯 Next Steps

### Right Now (5 min)
1. Read `templates/README.md`
2. Copy `templates/TPL___Daily_Intelligence_Capture.md`

### Today (20 min)
1. Create your first daily capture
2. Complete the template fully

### This Week (varies)
1. Decide: Manual or Automated?
2. If manual: generate first report using Claude.com
3. If automated: Follow `zapier/ZAPIER_AUTOMATION_SETUP.md`

### Next 2-4 Weeks
1. Keep capturing daily
2. Generate weekly synthesis
3. Notice emerging patterns
4. Adjust metrics if needed

### Month 2+
1. Monthly deep dives
2. Pattern detection runs
3. Experiment based on recommendations
4. Measure improvements

---

## 📄 Version Info

**System Version:** 1.0
**Created:** 2025-03-05
**Branch:** `claude/add-daily-intelligence-template-Y5Pom`
**Project:** Tree of Thoughts
**Status:** Ready to use

---

**Start your journey to data-driven self-improvement today! 🚀**

Choose your path: Manual → Automated → Custom Integration
