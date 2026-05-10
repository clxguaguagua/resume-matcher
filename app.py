import streamlit as st
import google.generativeai as genai
import json
import re
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume–JD Match Scorer",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
.score-box   { text-align:center; padding:20px; border-radius:10px; margin:8px 0; }
.score-high  { background:#dcfce7; border:2px solid #16a34a; }
.score-mid   { background:#fef9c3; border:2px solid #ca8a04; }
.score-low   { background:#fee2e2; border:2px solid #dc2626; }
.gap-item    { background:#f1f5f9; border-left:4px solid #6366f1; padding:10px 14px; border-radius:4px; margin:6px 0; }
.tip-item    { background:#f0fdf4; border-left:4px solid #16a34a; padding:10px 14px; border-radius:4px; margin:6px 0; }
.judge-box   { background:#f5f3ff; border:1px solid #a78bfa; padding:12px 16px; border-radius:6px; margin:8px 0; }
.keyword-hit { background:#bbf7d0; padding:2px 6px; border-radius:3px; margin:2px; display:inline-block; font-size:0.85em; }
.keyword-miss{ background:#fecaca; padding:2px 6px; border-radius:3px; margin:2px; display:inline-block; font-size:0.85em; }
</style>
""", unsafe_allow_html=True)

# ── Prompts ───────────────────────────────────────────────────────────────────
SCORER_SYSTEM = """You are an expert recruiter and resume coach. Given a job description (JD) and a candidate's resume, evaluate how well the resume matches the JD.

Output ONLY a valid JSON object with this exact structure:
{
  "overall_score": <integer 0-100>,
  "dimension_scores": {
    "skills_match": <integer 0-100>,
    "experience_relevance": <integer 0-100>,
    "education_fit": <integer 0-100>,
    "keywords_coverage": <integer 0-100>
  },
  "matched_strengths": [<3-5 specific strengths as strings>],
  "gaps": [<3-5 specific gaps or missing requirements as strings>],
  "resume_suggestions": [<3-5 actionable suggestions to improve the resume for this JD as strings>],
  "hiring_likelihood": "low" | "medium" | "high",
  "summary": "<2-3 sentence overall assessment>"
}

Be specific. Reference actual content from the JD and resume. Do not be overly generous — average match = 50-65, strong match = 75-85, exceptional = 90+.
Output ONLY the JSON object, no markdown, no preamble."""

JUDGE_SYSTEM = """You are evaluating the quality of an AI-generated resume-JD match assessment.

Score on three dimensions (1-5 each):
- accuracy: Does the assessment correctly identify real matches and gaps? (5 = highly accurate)
- specificity: Does it reference concrete details rather than generic statements? (5 = very specific)
- actionability: Are the suggestions concrete and useful? (5 = highly actionable)

Output ONLY a JSON object:
{"accuracy": <int>, "specificity": <int>, "actionability": <int>, "comment": "<one sentence>"}
No markdown, no preamble."""

STOPWORDS = {
    "and","or","the","a","an","in","of","to","for","with","is","are","be","will",
    "have","has","we","you","our","your","this","that","on","at","by","from",
    "as","it","its","not","but","if","they","their","can","all","any","both",
    "each","more","also","about","into","than","then","when","where","which",
    "who","how","what","must","should","may","would","could","been","was","were",
    "use","using","used","including","such","other","new","well","strong",
    "ability","experience","work","working","team","role","position","candidate",
    "required","preferred","plus","bonus","across","within","per","via"
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def keyword_baseline(jd: str, resume: str):
    def extract(text):
        words = re.findall(r'\b[a-zA-Z][a-zA-Z+#.]{2,}\b', text.lower())
        return Counter(w for w in words if w not in STOPWORDS)
    jd_kw = extract(jd)
    resume_kw = extract(resume)
    jd_top = set(w for w, _ in jd_kw.most_common(30))
    hits = jd_top & set(resume_kw.keys())
    misses = jd_top - hits
    score = int(len(hits) / max(len(jd_top), 1) * 100)
    return score, sorted(hits), sorted(misses)


def call_gemini(model, system: str, user: str) -> str:
    response = model.generate_content(f"{system}\n\n{user}")
    return response.text.strip()


def score_with_llm(jd: str, resume: str, model) -> dict:
    raw = call_gemini(model, SCORER_SYSTEM, f"JOB DESCRIPTION:\n{jd}\n\n---\n\nRESUME:\n{resume}")
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


def judge_assessment(jd: str, resume: str, assessment: dict, model) -> dict:
    user = (
        f"JD (first 500 chars): {jd[:500]}\n\n"
        f"Resume (first 500 chars): {resume[:500]}\n\n"
        f"Assessment:\n{json.dumps(assessment, indent=2)}"
    )
    raw = call_gemini(model, JUDGE_SYSTEM, user)
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


def score_css(score):
    if score >= 75: return "score-high"
    if score >= 50: return "score-mid"
    return "score-low"


def likelihood_emoji(l):
    return {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}.get(l, l)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📄 Resume–JD Match Scorer")
st.caption("Paste a job description and resume to get an AI-powered match score, gap analysis, and improvement suggestions.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Google AI Studio API Key", type="password", placeholder="AIza...")
    run_judge = st.checkbox("Run model-as-judge scoring", value=True)
    show_baseline = st.checkbox("Show keyword baseline comparison", value=True)
    st.divider()
    st.caption("**Model:** gemini-1.5-flash\n\n**Course concepts:**\n- Structured outputs (Week 2-3)\n- Model-as-judge evaluation (Week 6)")

# ── Sample data ───────────────────────────────────────────────────────────────
SAMPLE_JD = """Data Analyst – Business Intelligence
We are looking for a Data Analyst to join our BI team. You will transform raw data into actionable insights that drive business decisions.

Responsibilities:
- Build and maintain dashboards in Tableau or Power BI
- Write complex SQL queries to extract and analyze data
- Collaborate with business stakeholders to define KPIs
- Conduct A/B test analysis and present findings to leadership
- Automate reporting pipelines using Python

Requirements:
- 2+ years of experience in data analysis or BI role
- Proficiency in SQL (required) and Python (preferred)
- Experience with Tableau, Power BI, or similar BI tools
- Strong communication skills; able to present to non-technical audiences
- Bachelor's degree in Statistics, Computer Science, or related field"""

SAMPLE_RESUME = """Jane Smith | jane@email.com

EDUCATION
B.S. Statistics – University of Hong Kong, 2023

EXPERIENCE
Data Intern – ABC Retail (Jun 2022 – Aug 2022)
- Wrote SQL queries to analyze sales data across 50+ stores
- Built weekly Excel dashboards for the operations team
- Assisted with customer segmentation using Python (pandas, matplotlib)

Research Assistant – HKU Statistics Dept (Sep 2021 – May 2022)
- Cleaned and analyzed survey datasets using R and Python
- Co-authored a report on consumer behavior trends

SKILLS
SQL, Python (pandas, numpy, matplotlib), Excel, basic Power BI, R

PROJECTS
Customer Segmentation – K-Means and DBSCAN on e-commerce dataset
Sales Analysis – End-to-end analysis using SQL + Python + Power BI"""

if st.button("Load sample data"):
    st.session_state["jd"] = SAMPLE_JD
    st.session_state["resume"] = SAMPLE_RESUME
    st.rerun()

col_jd, col_res = st.columns(2)
with col_jd:
    jd_input = st.text_area(
        "Job Description",
        value=st.session_state.get("jd", ""),
        height=300,
        placeholder="Paste the full job description here…"
    )
with col_res:
    resume_input = st.text_area(
        "Resume",
        value=st.session_state.get("resume", ""),
        height=300,
        placeholder="Paste the candidate's resume here…"
    )

analyze_btn = st.button(
    "🔍 Analyze Match",
    type="primary",
    disabled=not (jd_input.strip() and resume_input.strip())
)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if not api_key:
        st.error("Please enter your Google AI Studio API key in the sidebar.")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    kw_score, kw_hits, kw_misses = keyword_baseline(jd_input, resume_input)

    with st.spinner("Analyzing with Gemini…"):
        try:
            result = score_with_llm(jd_input, resume_input, model)
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    st.divider()

    # Overall score
    overall = result.get("overall_score", 0)
    likelihood = result.get("hiring_likelihood", "medium")

    oc1, oc2, oc3 = st.columns([1, 2, 1])
    with oc2:
        st.markdown(
            f'<div class="score-box {score_css(overall)}">'
            f'<div style="font-size:3rem;font-weight:800">{overall}/100</div>'
            f'<div style="font-size:1.1rem;margin-top:4px">Overall Match Score</div>'
            f'<div style="margin-top:8px">Hiring Likelihood: <b>{likelihood_emoji(likelihood)}</b></div></div>',
            unsafe_allow_html=True
        )

    st.markdown(f"> {result.get('summary', '')}")

    # Dimension scores
    st.subheader("📊 Score Breakdown")
    dims = result.get("dimension_scores", {})
    d1, d2, d3, d4 = st.columns(4)
    for col, label, key in zip(
        [d1, d2, d3, d4],
        ["Skills Match", "Experience", "Education", "Keywords"],
        ["skills_match", "experience_relevance", "education_fit", "keywords_coverage"]
    ):
        s = dims.get(key, 0)
        col.markdown(
            f'<div class="score-box {score_css(s)}">'
            f'<div style="font-size:1.8rem;font-weight:700">{s}</div>'
            f'<div style="font-size:0.85rem">{label}</div></div>',
            unsafe_allow_html=True
        )

    # Strengths & Gaps
    sg1, sg2 = st.columns(2)
    with sg1:
        st.subheader("✅ Matched Strengths")
        for s in result.get("matched_strengths", []):
            st.markdown(f'<div class="tip-item">✓ {s}</div>', unsafe_allow_html=True)
    with sg2:
        st.subheader("⚠️ Gaps & Missing Requirements")
        for g in result.get("gaps", []):
            st.markdown(f'<div class="gap-item">✗ {g}</div>', unsafe_allow_html=True)

    # Suggestions
    st.subheader("💡 Resume Improvement Suggestions")
    for tip in result.get("resume_suggestions", []):
        st.markdown(f'<div class="tip-item">→ {tip}</div>', unsafe_allow_html=True)

    # Model-as-judge
    if run_judge:
        with st.spinner("Running model-as-judge…"):
            try:
                judge = judge_assessment(jd_input, resume_input, result, model)
                avg = round((judge["accuracy"] + judge["specificity"] + judge["actionability"]) / 3, 1)
                st.markdown(
                    f'<div class="judge-box">🤖 <b>Model-as-judge:</b> '
                    f'Accuracy {judge["accuracy"]}/5 · Specificity {judge["specificity"]}/5 · '
                    f'Actionability {judge["actionability"]}/5 · <b>Avg {avg}/5</b><br>'
                    f'<i>{judge["comment"]}</i></div>',
                    unsafe_allow_html=True
                )
            except Exception:
                pass

    # Baseline comparison
    if show_baseline:
        st.divider()
        st.subheader("📊 Baseline Comparison")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**🔤 Keyword Baseline**")
            st.caption(f"Score: {kw_score}/100 — {len(kw_hits)} of top-30 JD keywords found")
            st.markdown("**Matched:** " + " ".join(
                f'<span class="keyword-hit">{k}</span>' for k in kw_hits[:15]
            ), unsafe_allow_html=True)
            st.markdown("**Missing:** " + " ".join(
                f'<span class="keyword-miss">{k}</span>' for k in kw_misses[:15]
            ), unsafe_allow_html=True)
        with bc2:
            st.markdown("**🧠 Gemini LLM Assessment**")
            st.caption(f"Score: {overall}/100 — context-aware analysis")
            st.markdown(f"- Hiring likelihood: **{likelihood_emoji(likelihood)}**")
            st.markdown("- Identifies nuanced gaps keyword matching can't catch")
            st.markdown("- Generates actionable resume improvement suggestions")

        diff = overall - kw_score
        if abs(diff) > 10:
            direction = "higher" if diff > 0 else "lower"
            reason = "recognizing transferable skills keyword overlap missed" if diff > 0 else "catching requirement gaps that surface-level keyword matching masked"
            st.info(f"💡 LLM score is **{abs(diff)} points {direction}** than keyword baseline — {reason}.")
