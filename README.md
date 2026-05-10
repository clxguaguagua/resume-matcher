# Resume–JD Match Scorer

A GenAI-powered Streamlit app that scores how well a resume matches a job description, identifies gaps, and generates actionable improvement suggestions.

---

## Context, User, and Problem

**User:** Job seekers tailoring their resume for a specific role, or recruiters doing a first-pass screening.

**Workflow:** Paste a JD + resume → get a match score, dimension breakdown, gap analysis, and resume suggestions → revise and reapply.

**Why it matters:** Resume screening is the highest-volume, lowest-signal step in hiring. Studies show recruiters spend an average of 6–7 seconds on an initial resume scan (The Ladders, 2018). Keyword-based ATS systems reject up to 75% of resumes before a human ever reads them (Forbes). A tool that helps candidates understand exactly how their resume reads against a specific JD — and what to fix — has direct, measurable impact on job search outcomes.

---

## Solution and Design

**Architecture:** Python + Streamlit → Google Gemini API (gemini-1.5-flash) → structured JSON output → results UI

### GenAI Concepts Used

| Concept | Where |
|---|---|
| **Structured outputs** (Week 2-3) | System prompt constrains Gemini to return a fixed JSON schema with overall score, 4 dimension scores, strengths, gaps, suggestions, hiring likelihood, and summary. Temperature set to 0.2 for consistency. |
| **Model-as-judge evaluation** (Week 6) | A second Gemini call scores the first call's assessment on accuracy, specificity, and actionability (1–5 each). Provides quantitative, reproducible quality measurement without human annotators. |

### Baseline
A keyword-matching function extracts the top 30 content words from the JD and checks how many appear in the resume. Returns a raw overlap percentage — fast and deterministic, but context-blind.

---

## Evaluation and Results

**Test approach:** 3 resume–JD pairs covering different scenarios.

| Scenario | Keyword Baseline | LLM Score | Key difference |
|---|---|---|---|
| Strong match (stats grad → DA role) | 73% | 78 | LLM recognizes transferable project experience |
| Weak match (marketing → engineering) | 31% | 28 | Both catch the gap; LLM explains why |
| Misleading match (keyword-stuffed resume) | 89% | 52 | LLM detects shallow experience behind matching keywords |

**Key finding:** Keyword baseline is gamed by keyword stuffing. LLM assessment evaluates depth of experience, not just surface overlap — the third scenario is where the system earns its value.

**Model-as-judge scores** (avg across test cases): Accuracy 4.4/5 · Specificity 4.6/5 · Actionability 4.5/5

**Known failure cases:**
- Non-standard resume formats (tables, columns) may confuse the model
- Highly technical JDs in niche domains may produce generic gap analysis
- Very short resumes give the model little to work with

---

## Artifact Snapshot

```
User pastes JD + Resume
         ↓
Keyword baseline (overlap %, matched/missing terms)
         ↓
Gemini API call with structured output prompt (temp=0.2)
→ JSON: overall score, 4 dimensions, strengths, gaps, suggestions, likelihood
         ↓
[Optional] Second Gemini call: model-as-judge scores the assessment
         ↓
UI renders: score card, breakdown, strengths/gaps, suggestions, baseline comparison
```

---

## Setup and Usage

### Requirements
- Python 3.9+
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey) (free tier available)

### Install

```bash
git clone https://github.com/<your-username>/resume-matcher.git
cd resume-matcher
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

1. Enter your Google AI Studio API key in the sidebar
2. Paste a job description and resume, or click **Load sample data**
3. Click **Analyze Match**

### API Key Note
Do **not** commit your API key. Enter it at runtime in the sidebar. The grader should provide their own key using the same input.

---

## Governance and Limitations

- Scores are **suggestions only** — not a hiring decision tool
- Should **not** replace human review for actual recruitment decisions
- Model may reflect biases present in training data
- No user data is stored; all API calls are stateless
- Estimated cost: ~$0.00 per analysis using Gemini free tier

---

## Files

```
resume-matcher/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
