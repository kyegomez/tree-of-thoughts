# Daily Intelligence System - Zapier Automation Setup

## Overview
Automate the Daily Intelligence reporting pipeline by triggering Claude API calls when new Daily Intelligence Captures are created.

## System Architecture

```
[Create Daily Capture]
        ↓
   [Zapier Trigger]
        ↓
   [Claude API Call]
        ↓
[Generate Report] → [Store Result] → [Notify User]
```

## Prerequisites

- **Zapier Account:** Free or paid plan
- **Claude API Key:** From Anthropic console
- **Data Storage:**
  - Document platform (Notion, Google Docs, etc.) OR
  - JSON file storage (if self-hosted) OR
  - Database (PostgreSQL, etc.)

---

## Zapier Integration Steps

### Step 1: Create the Trigger

#### Option A: Trigger from Document Creation (Recommended)

**If using Notion:**
1. In Zapier, select "Notion" as the trigger app
2. Choose trigger: "Database item created" or "Database item updated"
3. Connect your Notion workspace
4. Select the database where Daily Captures are created
5. Filter conditions:
   - Trigger when: New page created
   - Optional: Filter to only pages with specific property values

**If using Google Docs:**
1. Select "Google Docs" as trigger app
2. Choose: "New document created"
3. Or use Google Drive + conditional logic for file naming pattern

**If using Email:**
1. Select "Email" trigger
2. Send formatted email to dedicated address when capture complete
3. Zapier reads email content

**If using Webhook:**
1. Select "Webhooks by Zapier"
2. Choose "Catch Raw Hook"
3. Use this webhook URL in your capture app/script
4. POST request with capture data as JSON

### Step 2: Extract & Format Capture Data

**Add a "Formatter" step:**

1. Choose "Formatter by Zapier" → "Parse JSON"
2. Map trigger output fields to structured JSON:

```json
{
  "date": "YYYY-MM-DD",
  "metrics": {
    "sleep_quality": 8,
    "physical_energy": 7,
    "focus_score": 8,
    "task_completion": "7/8",
    "knowledge_intake": 2.5,
    "skill_practice": 1.5,
    "connection_quality": 7,
    "emotional_balance": 8
  },
  "projects": [
    {
      "name": "Project 1",
      "status": "In Progress",
      "progress": 65,
      "blockers": "Resource constraint",
      "win": "Completed module A"
    }
    // ... 4 projects total
  ],
  "journal_morning": "...",
  "journal_afternoon": "...",
  "journal_evening": "...",
  "ifs_notes": "...",
  "tomorrow_intention": "..."
}
```

### Step 3: Call Claude API

**Add "Webhooks by Zapier" → "POST" action:**

**URL:**
```
https://api.anthropic.com/v1/messages
```

**Method:** POST

**Headers:**
```
x-api-key: [YOUR_CLAUDE_API_KEY]
anthropic-version: 2023-06-01
content-type: application/json
```

**Body:**
```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 2000,
  "messages": [
    {
      "role": "user",
      "content": "{{YOUR_PROMPT_CONTENT}}"
    }
  ],
  "system": "{{SYSTEM_PROMPT}}"
}
```

**Parameter Details:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model` | `claude-opus-4-6` | Latest Claude model |
| `max_tokens` | 1500-2000 | For daily reports |
| `temperature` | 1.0 | Default (can adjust 0-2) |

### Step 4A: Store Report in Notion

**Add "Notion" action → "Create database item":**

1. Choose your Reports database
2. Map Claude response to:
   - **Title:** Daily Report [Date]
   - **Date:** [From trigger]
   - **Report Content:** [Claude response]
   - **Status:** Complete
   - **Metrics Data:** [Numeric values]
   - **Relation:** [Link back to original Capture]

### Step 4B: Store Report in Google Docs

**Add "Google Docs" action → "Create document":**

1. **Title:** `Daily Report - {{date}}`
2. **Content:** Format Claude response in readable structure
3. **Folder:** Your reports folder
4. **Sharing:** (Optional) Set permissions

### Step 4C: Store Report in Email

**Add "Gmail" action → "Send email":**

1. **To:** Your email
2. **Subject:** `Daily Intelligence Report - {{date}}`
3. **Body:** Formatted Claude response
4. **Attachments:** (Optional) PDF export

---

## System Prompts & Parameters

### System Prompt for Claude

Place this in the "system" field or as part of user message:

```
You are the Daily Intelligence Generator for a personal performance optimization system.

Context:
- User is tracking 8 key metrics across Health, Productivity, Learning, and Relationships
- User has 4 active projects being managed
- This system uses Internal Family Systems (IFS) framework optionally
- Reports should be insightful, actionable, and encouraging

Your role:
- Analyze the daily capture data provided
- Generate structured insights using the Daily Intelligence Generator prompt format
- Highlight patterns, progress, and opportunities
- Provide specific, actionable recommendations
- Maintain a supportive but honest tone

Output your report in clear sections with emojis for visual scanning.
```

### Claude API Call Template

**For Daily Intelligence Generation:**

```javascript
const body = {
  "model": "claude-opus-4-6",
  "max_tokens": 2000,
  "messages": [
    {
      "role": "user",
      "content": `Daily Intelligence Capture Data:\n${JSON.stringify(captureData, null, 2)}\n\nPlease generate a comprehensive daily intelligence report analyzing this data. Use the structure: Summary, Metrics Analysis, Project Velocity, Pattern Recognition, Key Insights, Recommendations, and IFS Integration if applicable.`
    }
  ],
  "system": "[SYSTEM_PROMPT_ABOVE]"
}
```

---

## Automation Workflows

### Workflow 1: Daily Report Generation

**Trigger:** New daily capture created
**Steps:**
1. Extract capture data
2. Call Claude API (Daily Intelligence Generator prompt)
3. Store report in Notion/Google Docs
4. Send email notification
5. (Optional) Update metrics dashboard

**Frequency:** Daily, triggered manually or on schedule

---

### Workflow 2: Weekly Synthesis

**Trigger:** Every Sunday at 6 PM (or manually)
**Steps:**
1. Fetch last 7 daily reports
2. Compile metrics array for all 7 days
3. Call Claude API (Weekly Synthesis Generator prompt)
4. Store weekly report
5. Send digest email with weekly insights

**Note:** May require custom Zapier multi-step retrieval

---

### Workflow 3: Pattern Detection Run

**Trigger:** End of month (or every 14 days)
**Steps:**
1. Fetch all captures for period
2. Extract all metrics data
3. Call Claude API (Pattern Detection prompt)
4. Store pattern report
5. Notify user of discoveries

**Note:** For large datasets, may need to handle token limits

---

### Workflow 4: Monthly Recommendations

**Trigger:** First of month (or manually)
**Steps:**
1. Fetch previous month's data (captures, reports, patterns)
2. Summarize month achievements/challenges
3. Call Claude API (Recommendation Engine prompt)
4. Store recommendations document
5. Schedule calendar reminders for top 3 actions

---

## API Rate Limiting & Costs

### Claude API Pricing (as of 2025)
- **Input:** $0.003 per 1K tokens
- **Output:** $0.015 per 1K tokens

### Estimated Costs

| Scenario | Tokens | Cost |
|----------|--------|------|
| Daily Report | ~1,500 in + 1,500 out | ~$0.03 |
| Weekly Synthesis | ~3,000 in + 2,000 out | ~$0.039 |
| Pattern Detection | ~5,000 in + 2,000 out | ~$0.045 |
| Monthly Deep Dive | ~8,000 in + 3,000 out | ~0.069 |

**Monthly estimate (daily reports only):** ~$0.90/month

### Rate Limiting

Claude API rate limits (Hobby tier):
- 20 requests per minute
- 40,000 input tokens per minute
- 40,000 output tokens per minute

Zapier will queue requests if limits hit.

---

## Error Handling & Fallbacks

### Common Issues

**1. API Key Invalid**
- Add error handler in Zapier
- Action: Send email alert to user
- Check: Verify API key in Anthropic console

**2. Token Limit Exceeded**
- Reduce data payload
- Summarize before sending to Claude
- Use two-step process for large datasets

**3. Zapier Execution Fails**
- Automated retry: Built into Zapier
- Webhook status: Check Zapier logs
- Manual recovery: Re-trigger manually

### Setup Error Handling Step

**Add "Email" action with condition:**
- Condition: "If previous step error occurred"
- Email alert to user with error details
- Include Zapier execution log

---

## Testing the Automation

### Test 1: Trigger Test
1. Create a test daily capture
2. Verify Zapier detects trigger
3. Check execution logs

### Test 2: Data Extraction
1. Verify formatter correctly parses capture
2. Check JSON structure is valid
3. Review mapped fields

### Test 3: Claude API Call
1. Verify headers and auth correct
2. Check token count reasonable
3. Review Claude response

### Test 4: Storage
1. Verify report saves to correct location
2. Check formatting is readable
3. Ensure all fields populated

### Test 5: End-to-End
1. Create capture → Receive report (should take 30-60 seconds)
2. Verify report quality and accuracy
3. Check data relationships preserved

---

## Advanced Setup (Optional)

### Multi-Language Reports

Add language parameter to API call:
```json
{
  "messages": [{
    "role": "user",
    "content": "Generate report in [LANGUAGE]..."
  }]
}
```

### Custom Report Formats

Add formatter step to convert Claude response:
- HTML for email
- Markdown for documents
- JSON for API consumption

### Dashboard Integration

After storing reports, trigger:
- Update metrics spreadsheet
- Update Notion dashboard
- Send Slack notification

### Conditional Branching

Zapier "Filter" step can trigger different prompts based on:
- Current streak status
- Energy level indicators
- Project urgency flags
- Monthly/weekly/daily context

---

## Template Files Required

To make this work, ensure these exist:
- `/prompts/PROMPT___Daily_Intelligence_Generator.md`
- `/prompts/PROMPT___Weekly_Synthesis_Generator.md`
- `/prompts/PROMPT___Monthly_Deep_Dive_Generator.md`
- `/prompts/PROMPT___Pattern_Detection.md`
- `/prompts/PROMPT___Recommendation_Engine.md`

Each prompt should be loaded into Zapier as custom user message or system prompt.

---

## Security Considerations

1. **API Key Safety:**
   - Store in Zapier with locked value
   - Never commit to Git
   - Rotate regularly

2. **Data Privacy:**
   - Personal data stored in capture
   - Ensure Notion/Google Docs have proper access controls
   - Consider encryption for sensitive data

3. **Zapier Permissions:**
   - Grant minimum necessary permissions
   - Review connected apps regularly
   - Test with limited data first

---

## Monitoring & Optimization

### Metrics to Track

- **Success Rate:** % of triggers that complete successfully
- **Execution Time:** How long to generate reports
- **API Costs:** Actual spend vs. budget
- **Report Quality:** User satisfaction with generated insights

### Optimization Tips

1. Cache repeated data (metrics definitions, project names)
2. Batch operations where possible (weekly vs. daily)
3. Use smaller models for simple tasks
4. Implement request queuing for peak times

---

## Troubleshooting

### Report not generating?
- [ ] Check Zapier is connected to trigger source
- [ ] Verify Claude API key is valid
- [ ] Check API call body format (JSON valid?)
- [ ] Review execution logs for error details

### Report quality low?
- [ ] Review system prompt clarity
- [ ] Check capture data completeness
- [ ] Increase max_tokens if truncated
- [ ] Add more context to prompt

### Automation slow?
- [ ] Check Zapier plan tier
- [ ] Reduce data payload size
- [ ] Use task queue for heavy lifting
- [ ] Schedule non-urgent reports off-peak

---

## Next Steps

1. **Get Claude API Key:** https://console.anthropic.com/
2. **Create Zapier Account:** https://zapier.com/
3. **Connect Data Source:** Notion, Google Docs, etc.
4. **Configure First Workflow:** Start with Daily Reports
5. **Test End-to-End:** Create sample capture
6. **Monitor & Iterate:** Refine based on results

**Estimated Setup Time:** 30-45 minutes for first automation
