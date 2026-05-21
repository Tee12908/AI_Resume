"""
app.py
AI Resume Checker – Streamlit frontend.

Run with: streamlit run app.py
"""

import io
import time
from datetime import datetime
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ats_analyzer import ATSAnalyzer
from image_checker import ImageChecker
from resume_parser import ResumeParser

# ── Page config – MUST be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="AI Resume Checker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        #MainMenu, footer { visibility: hidden; }

        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .main-header h1 { margin: 0; font-size: 2.4rem; font-weight: 700; }
        .main-header p  { margin: 0.5rem 0 0; opacity: .9; font-size: 1.05rem; }

        /* Badge utilities */
        .badge { padding: 3px 11px; border-radius: 20px; font-size: .78rem;
                 display: inline-block; margin: 2px; font-weight: 500; }
        .badge-success { background: #d4edda; color: #155724; }
        .badge-danger  { background: #f8d7da; color: #721c24; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-skill   { background: linear-gradient(135deg,#667eea,#764ba2);
                         color: white; }
        .badge-recommend { background: #ff6b6b; color: white; }

        /* Improvement cards */
        .imp-card {
            border-left: 4px solid #667eea;
            background: #f8f9ff;
            padding: 10px 15px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 8px;
            font-size: .9rem;
        }
        .imp-high   { border-left-color: #ff4757 !important; }
        .imp-medium { border-left-color: #ffa502 !important; }
        .imp-low    { border-left-color: #2ed573 !important; }

        /* Upload drag-drop hint */
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #667eea !important;
            border-radius: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────────

def _header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>📄 AI Resume Checker</h1>
            <p>ATS scoring · Skills detection · Job-role prediction · Professional feedback</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Go to",
            ["📤 Upload & Analyse", "📊 Dashboard", "💡 Tips & Guides"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### 📋 Supported Formats")
        for fmt in ["PDF", "DOCX / DOC", "PNG / JPG / JPEG"]:
            st.markdown(f"- {fmt}")

        st.markdown("---")
        st.markdown("### 📏 Score Guide")
        st.markdown(
            "| Score | Grade |\n|-------|-------|\n"
            "| 90-100 | A+ – Excellent |\n"
            "| 80-89 | A – Very Good |\n"
            "| 70-79 | B – Good |\n"
            "| 60-69 | C – Average |\n"
            "| 50-59 | D – Below Average |\n"
            "| < 50 | F – Needs Work |"
        )

        st.markdown("---")
        st.caption("Built with Streamlit · Powered by NLP")
    return page


# ──────────────────────────────────────────────────────────────────────────────
# Display blocks
# ──────────────────────────────────────────────────────────────────────────────

def _show_ats_score(ats_data: Dict) -> None:
    score = ats_data["total"]
    grade = ats_data["grade"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"ATS Score  |  Grade: <b>{grade}</b>", "font": {"size": 15}},
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#667eea"},
                "steps": [
                    {"range": [0, 50], "color": "#ffe0e0"},
                    {"range": [50, 70], "color": "#fff3cc"},
                    {"range": [70, 85], "color": "#d4f1d4"},
                    {"range": [85, 100], "color": "#b2dfdb"},
                ],
                "threshold": {
                    "line": {"color": "crimson", "width": 3},
                    "thickness": 0.75,
                    "value": 60,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown progress bars
    st.markdown("**Score Breakdown**")
    for cat, data in ats_data.get("breakdown", {}).items():
        s = data.get("score", 0)
        m = data.get("max", 10)
        pct = s / m if m else 0
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.progress(pct, text=cat)
        with col_b:
            st.caption(f"{s}/{m}")


def _show_skills(detected: Dict, recommended: list) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Detected Skills")
        if detected:
            for cat, skills in detected.items():
                st.markdown(f"**{cat.replace('_',' ').title()}**")
                html = " ".join(
                    f'<span class="badge badge-skill">{s}</span>' for s in skills
                )
                st.markdown(html, unsafe_allow_html=True)
                st.write("")
        else:
            st.info("No specific skills detected. List skills explicitly in your resume.")

    with col2:
        st.markdown("#### 💡 Recommended Skills to Add")
        if recommended:
            html = " ".join(
                f'<span class="badge badge-recommend">{s}</span>' for s in recommended
            )
            st.markdown(html, unsafe_allow_html=True)
            st.info("Add these to your resume if applicable.")
        else:
            st.success("Your skill set looks comprehensive!")


def _show_sections(present: list, missing: Dict) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Detected Sections")
        for s in present:
            st.markdown(f'<span class="badge badge-success">✓ {s}</span>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### ❌ Missing Sections")
        critical = missing.get("critical", [])
        optional = missing.get("optional", [])

        if critical:
            st.markdown("**Critical:**")
            for s in critical:
                st.markdown(f'<span class="badge badge-danger">✗ {s}</span>', unsafe_allow_html=True)
        if optional:
            st.markdown("**Recommended:**")
            for s in optional:
                st.markdown(f'<span class="badge badge-warning">△ {s}</span>', unsafe_allow_html=True)
        if not critical and not optional:
            st.success("All important sections are present!")


def _show_job_prediction(job: Dict) -> None:
    primary = job.get("primary", "Unknown")
    confidence = job.get("confidence", 0)
    top = job.get("top_matches", [])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                        padding:24px;border-radius:14px;color:white;text-align:center;">
                <div style="font-size:.85rem;opacity:.8;">Predicted Role</div>
                <div style="font-size:1.25rem;font-weight:700;margin:8px 0;">{primary}</div>
                <div style="font-size:.85rem;">Confidence: {confidence:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(confidence / 100)

    with col2:
        if top:
            roles = [m["role"] for m in top]
            pcts = [m["match_percentage"] for m in top]
            fig = px.bar(
                x=pcts, y=roles, orientation="h",
                labels={"x": "Match %", "y": ""},
                color=pcts, color_continuous_scale="Viridis",
                title="Top Role Matches",
            )
            fig.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


def _show_improvements(improvements: list) -> None:
    colour_map = {"High": "#ff4757", "Medium": "#ffa502", "Low": "#2ed573"}
    class_map = {"High": "imp-high", "Medium": "imp-medium", "Low": "imp-low"}

    for imp in improvements:
        pri = imp.get("priority", "Medium")
        cat = imp.get("category", "")
        sug = imp.get("suggestion", "")
        col = colour_map.get(pri, "#667eea")
        cls = class_map.get(pri, "")
        st.markdown(
            f'<div class="imp-card {cls}">'
            f'<strong style="color:{col};">[{pri}]</strong> '
            f'<strong>{cat}:</strong> {sug}</div>',
            unsafe_allow_html=True,
        )


def _show_strengths_weaknesses(strengths: list, weaknesses: list) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💪 Strengths")
        for s in strengths:
            st.markdown(f"✅ {s}")
    with col2:
        st.markdown("#### ⚠️ Weaknesses")
        for w in weaknesses:
            st.markdown(f"❌ {w}")


def _show_contact(contact: Dict) -> None:
    labels = {
        "name": "👤 Name", "email": "📧 Email", "phone": "📱 Phone",
        "linkedin": "💼 LinkedIn", "github": "💻 GitHub", "website": "🌐 Website",
    }
    for key, label in labels.items():
        val = contact.get(key)
        if val:
            st.markdown(f"**{label}:** {val}")
        else:
            st.markdown(
                f"**{label}:** <span style='color:#bbb'>Not found</span>",
                unsafe_allow_html=True,
            )


def _show_image_analysis(img_data: Dict) -> None:
    if not img_data.get("has_image"):
        st.info("ℹ️ No profile image detected in this resume.")
        st.caption(
            "Adding a professional headshot can help in some regions/industries."
        )
        return

    st.success(f"📸 {img_data['image_count']} image(s) detected in resume.")
    q = img_data.get("quality", {})

    if q:
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolution", q.get("resolution", "–"))
        c2.metric("Blur", "Blurry ⚠️" if q.get("is_blurry") else "Sharp ✓")
        c3.metric("Quality", q.get("overall_quality", "–"))

    if img_data.get("has_face"):
        st.success(f"✅ Face detected ({img_data['face_count']} face(s))")
    else:
        st.warning("⚠️ No face detected – ensure the photo is a clear headshot.")

    recs = img_data.get("recommendations", [])
    if recs:
        st.markdown("**Image Recommendations:**")
        for r in recs:
            st.markdown(f"• {r}")


# ──────────────────────────────────────────────────────────────────────────────
# Export helpers
# ──────────────────────────────────────────────────────────────────────────────

def _generate_pdf(analysis: Dict, contact: Dict) -> bytes:
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title banner
        pdf.set_fill_color(102, 126, 234)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 14, "AI Resume Analysis Report", ln=True, align="C", fill=True)
        pdf.set_text_color(100, 100, 100)
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", ln=True, align="R")
        pdf.ln(4)

        def section(title: str) -> None:
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 9, title, ln=True)
            pdf.set_draw_color(102, 126, 234)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)

        def row(label: str, value: str) -> None:
            pdf.set_font("Arial", "", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, f"  {label}: {value}")

        ats = analysis.get("ats_score", {})
        section(f"ATS Score: {ats.get('total', 0)}/100  (Grade {ats.get('grade','?')})")

        section("Contact Information")
        for k, v in contact.items():
            if v:
                row(k.title(), str(v))

        section("Strengths")
        for s in analysis.get("strengths", []):
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 6, f"  + {s}")

        section("Areas for Improvement")
        for w in analysis.get("weaknesses", []):
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 6, f"  - {w}")

        section("Recommendations")
        for imp in analysis.get("improvements", []):
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 6, f"  [{imp['priority']}] {imp['category']}: {imp['suggestion']}")

        section("Detected Skills")
        for cat, skills in analysis.get("detected_skills", {}).items():
            if skills:
                pdf.set_font("Arial", "", 10)
                pdf.multi_cell(0, 6, f"  {cat.replace('_',' ').title()}: {', '.join(skills)}")

        return bytes(pdf.output())
    except ImportError:
        return b""


def _generate_csv(analysis: Dict, contact: Dict) -> str:
    rows = []
    ats = analysis.get("ats_score", {})
    rows += [
        {"Category": "ATS Score", "Item": "Total", "Value": str(ats.get("total", 0))},
        {"Category": "ATS Score", "Item": "Grade", "Value": ats.get("grade", "?")},
    ]
    for k, v in contact.items():
        rows.append({"Category": "Contact", "Item": k, "Value": str(v or "—")})
    for cat, skills in analysis.get("detected_skills", {}).items():
        rows.append({"Category": "Skills", "Item": cat, "Value": ", ".join(skills)})
    for s in analysis.get("strengths", []):
        rows.append({"Category": "Strength", "Item": s, "Value": "Yes"})
    for w in analysis.get("weaknesses", []):
        rows.append({"Category": "Weakness", "Item": w, "Value": "Yes"})
    job = analysis.get("job_role", {})
    rows.append({"Category": "Job Role", "Item": "Primary", "Value": job.get("primary", "—")})
    return pd.DataFrame(rows).to_csv(index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────

def _page_tips() -> None:
    st.title("💡 Resume Tips & ATS Guide")

    tab_ats, tab_write, tab_mistakes = st.tabs(
        ["ATS Optimisation", "Writing Tips", "Common Mistakes"]
    )

    with tab_ats:
        st.markdown(
            """
### How ATS Works
**Applicant Tracking Systems** filter resumes before a human reads them.

#### Key ATS factors
| Factor | Impact |
|--------|--------|
| Keywords matching job description | Very High |
| Standard section headers | High |
| Clean, parseable formatting | High |
| Correct contact details | Medium |
| Quantified achievements | Medium |

#### Score Breakdown
| Category | Max Points |
|----------|-----------|
| Contact Information | 15 |
| Resume Sections | 20 |
| Skills Detection | 15 |
| Action Verbs | 15 |
| Quantified Results | 10 |
| Resume Length | 10 |
| Formatting | 10 |
| Grammar | 5 |
"""
        )

    with tab_write:
        st.markdown(
            """
### Writing Tips

#### ✅ DO
- Start bullets with **strong action verbs**
- **Quantify everything** – numbers catch the eye
- **Tailor** for every application
- Keep to **1–2 pages**
- Use **consistent formatting**

#### ❌ DON'T
- Use personal pronouns (I, me, my)
- Write generic phrases like "responsible for"
- Submit without proofreading
- Use complex graphics that confuse parsers

#### Transformation examples
| Weak | Strong |
|------|--------|
| Responsible for sales | Grew sales 35 % in Q1 2024 |
| Worked on projects | Led 5 cross-functional projects delivered on schedule |
| Good at Python | Built 3 production Python APIs serving 50 K+ requests/day |
"""
        )

    with tab_mistakes:
        st.markdown(
            """
### Top 10 Resume Mistakes

1. **Missing contact details** – always include email, phone, LinkedIn
2. **No skills section** – list skills explicitly so ATS can find them
3. **No professional summary** – add 2–3 lines at the top
4. **Generic language** – replace weak words with specific, impactful ones
5. **No numbers** – quantify every achievement possible
6. **Wrong file format** – use PDF unless the employer asks for DOCX
7. **Spelling / grammar errors** – proofread at least three times
8. **One-size-fits-all** – customise for each role
9. **Outdated information** – keep technology stacks and dates current
10. **No online presence** – include LinkedIn and GitHub when relevant
"""
        )


def _page_upload_and_analyse() -> None:
    st.markdown("## 📤 Upload Your Resume")

    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["pdf", "docx", "doc", "png", "jpg", "jpeg"],
            help="PDF, DOCX, PNG or JPG — max 10 MB",
        )
        if uploaded:
            st.success(f"✅ **{uploaded.name}** – {uploaded.size / 1024:.1f} KB")

    with col_info:
        st.markdown(
            """
            **Requirements**
            - English language
            - Max 10 MB
            - 1–2 pages recommended
            """
        )

    if uploaded and st.button("🔍 Analyse Resume", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            raw = uploaded.read()
            ext = uploaded.name.rsplit(".", 1)[-1]

            status.text("📖 Extracting text…")
            progress.progress(15)
            parser = ResumeParser()
            parse_result = parser.extract_text(raw, ext)

            if parse_result.get("error"):
                st.error(f"Extraction error: {parse_result['error']}")
                return

            text = parse_result.get("text", "").strip()
            if not text:
                st.error("Could not extract text. Try a different file or format.")
                return

            status.text("🔍 Analysing structure…")
            progress.progress(35)
            contact = parser.extract_contact_info(text)
            sections = parser.extract_sections(text)

            status.text("🖼️ Checking for profile image…")
            progress.progress(55)
            image_checker = ImageChecker()
            image_data = image_checker.check_resume_for_image(raw, ext)

            status.text("🤖 Running AI analysis…")
            progress.progress(75)
            analyzer = ATSAnalyzer()
            analysis = analyzer.analyze(text, sections, contact)

            progress.progress(100)
            status.text("✅ Complete!")
            time.sleep(0.4)
            status.empty()
            progress.empty()

            # Persist results
            st.session_state.results = {
                "analysis": analysis,
                "contact": contact,
                "sections": sections,
                "image": image_data,
                "text": text,
                "parse_info": parse_result,
                "filename": uploaded.name,
            }
            st.session_state.analysis_done = True
            st.balloons()
            st.success("Analysis complete! See results below ↓")

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


def _page_dashboard() -> None:
    if not st.session_state.get("analysis_done"):
        st.warning("No analysis data yet – please upload a resume first.")
        return

    r = st.session_state.results
    analysis = r["analysis"]
    contact = r["contact"]
    sections = r["sections"]
    image_data = r["image"]
    text = r["text"]
    parse_info = r["parse_info"]

    st.markdown("## 📊 Resume Analysis Dashboard")

    # ── Top KPI strip ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("ATS Score", f"{analysis['ats_score']['total']}/100",
              delta=f"Grade {analysis['ats_score']['grade']}")
    total_skills = sum(len(v) for v in analysis["detected_skills"].values())
    k2.metric("Skills Found", total_skills)
    detected_count = sum(1 for v in sections.values() if v)
    k3.metric("Sections", f"{detected_count}/{len(sections)}")
    k4.metric("Words", analysis["word_count"])
    k5.metric("Action Verbs", analysis["action_verb_count"]["count"])

    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    t_score, t_contact, t_skills, t_job, t_improve, t_export = st.tabs([
        "🎯 ATS Score",
        "👤 Contact & Sections",
        "🛠️ Skills",
        "📈 Job Prediction",
        "💡 Improvements",
        "📤 Export",
    ])

    # ── Tab 1: ATS Score ───────────────────────────────────────────────────────
    with t_score:
        col_g, col_s = st.columns([1, 1])

        with col_g:
            st.markdown("### ATS Compatibility Score")
            _show_ats_score(analysis["ats_score"])

        with col_s:
            st.markdown("### Resume Stats")
            read = analysis.get("readability_score", {})
            st.metric("Readability", read.get("rating", "–"))
            st.metric("Avg Sentence Length", f"{read.get('avg_sentence_length', 0)} words")

            verbs = analysis.get("action_verb_count", {})
            st.metric("Action Verbs", f"{verbs.get('count',0)}  ({verbs.get('rating','–')})")

            quant = analysis.get("quantified_achievements", {})
            st.metric("Quantified Achievements", f"{quant.get('count',0)}  ({quant.get('rating','–')})")

            gi = len(analysis.get("grammar_issues", []))
            st.metric("Grammar Issues", gi,
                      delta="Clean ✓" if gi == 0 else f"{gi} found",
                      delta_color="normal" if gi == 0 else "inverse")

        st.markdown("### 💪 Strengths & Weaknesses")
        _show_strengths_weaknesses(analysis.get("strengths", []), analysis.get("weaknesses", []))

        grammar_issues = analysis.get("grammar_issues", [])
        if grammar_issues:
            with st.expander(f"⚠️ {len(grammar_issues)} Grammar Issue(s)"):
                for i, issue in enumerate(grammar_issues, 1):
                    st.markdown(f"**{i}.** {issue.get('message','')}")
                    if issue.get("context"):
                        st.caption(f"Context: …{issue['context'][:80]}…")

    # ── Tab 2: Contact & Sections ──────────────────────────────────────────────
    with t_contact:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 📋 Contact Information")
            _show_contact(contact)

            st.markdown("### 📸 Profile Image")
            _show_image_analysis(image_data)

        with c2:
            st.markdown("### 📑 Resume Sections")
            _show_sections(
                analysis.get("present_sections", []),
                analysis.get("missing_sections", {}),
            )

        st.markdown("### 📄 Document Info")
        d1, d2, d3 = st.columns(3)
        d1.metric("File", r["filename"])
        d2.metric("Pages", parse_info.get("page_count", 1))
        d3.metric("Parser", parse_info.get("extraction_method", "–"))

    # ── Tab 3: Skills ──────────────────────────────────────────────────────────
    with t_skills:
        st.markdown("### 🛠️ Skills Analysis")
        detected = analysis.get("detected_skills", {})

        if detected:
            skill_counts = {
                cat.replace("_", " ").title(): len(skills)
                for cat, skills in detected.items()
                if skills
            }
            fig = px.pie(
                values=list(skill_counts.values()),
                names=list(skill_counts.keys()),
                title="Skills Distribution by Category",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        _show_skills(detected, analysis.get("recommended_skills", []))

        kw_data = analysis.get("keyword_density", {}).get("top_keywords", [])
        if kw_data:
            st.markdown("### 🔑 Top Resume Keywords")
            kw_df = pd.DataFrame(kw_data)
            fig2 = px.bar(
                kw_df, x="count", y="word", orientation="h",
                labels={"count": "Frequency", "word": ""},
                color="count", color_continuous_scale="Viridis",
                title="Keyword Frequency",
            )
            fig2.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 4: Job Prediction ──────────────────────────────────────────────────
    with t_job:
        st.markdown("### 📈 Job Role Prediction")
        _show_job_prediction(analysis.get("job_role", {}))

        top_matches = analysis.get("job_role", {}).get("top_matches", [])
        if top_matches:
            primary_kws = top_matches[0].get("keywords_found", [])
            if primary_kws:
                st.markdown("### 🔑 Matched Keywords for Primary Role")
                html = " ".join(
                    f'<span class="badge badge-skill">{kw}</span>' for kw in primary_kws
                )
                st.markdown(html, unsafe_allow_html=True)

    # ── Tab 5: Improvements ────────────────────────────────────────────────────
    with t_improve:
        st.markdown("### 💡 Improvement Recommendations")
        st.info("Recommendations are ranked by priority to maximise your ATS score.")

        improvements = analysis.get("improvements", [])
        for level in ["High", "Medium", "Low"]:
            subset = [i for i in improvements if i.get("priority") == level]
            if subset:
                colours = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                st.markdown(f"#### {colours[level]} {level} Priority")
                _show_improvements(subset)

        # Suggested action verbs not yet in the resume
        found_verbs = set(analysis.get("action_verb_count", {}).get("verbs", []))
        missing_verbs = [v for v in ATSAnalyzer.ACTION_VERBS if v not in found_verbs][:16]
        if missing_verbs:
            st.markdown("#### ✍️ Action Verbs to Consider")
            html = " ".join(
                f'<span class="badge badge-skill">{v}</span>' for v in missing_verbs
            )
            st.markdown(html, unsafe_allow_html=True)

    # ── Tab 6: Export ──────────────────────────────────────────────────────────
    with t_export:
        st.markdown("### 📤 Download Your Analysis Report")

        ex1, ex2 = st.columns(2)

        with ex1:
            st.markdown("#### 📄 PDF Report")
            if st.button("Generate PDF", use_container_width=True):
                with st.spinner("Generating PDF…"):
                    pdf_bytes = _generate_pdf(analysis, contact)
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.error("PDF generation requires fpdf2. Run: pip install fpdf2")

        with ex2:
            st.markdown("#### 📊 CSV Export")
            csv_data = _generate_csv(analysis, contact)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_data,
                file_name=f"resume_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("#### 📝 Extracted Resume Text")
        with st.expander("View raw text"):
            st.text_area("", text, height=300, disabled=True)

    st.markdown("---")
    if st.button("🔄 Analyse Another Resume", use_container_width=True):
        st.session_state.analysis_done = False
        st.session_state.results = {}
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()
    _header()
    page = _sidebar()

    # Initialise session state
    st.session_state.setdefault("analysis_done", False)
    st.session_state.setdefault("results", {})

    if "Tips" in page:
        _page_tips()

    elif "Dashboard" in page:
        _page_dashboard()

    else:  # Upload & Analyse
        _page_upload_and_analyse()
        # Show dashboard inline after analysis completes
        if st.session_state.get("analysis_done"):
            _page_dashboard()


if __name__ == "__main__":
    main()
