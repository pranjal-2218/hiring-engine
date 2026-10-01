"""
frontend/app.py  —  Intelligent Hiring Engine UI
"""
from __future__ import annotations
import time
import os
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Hiring Engine", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.header{background:linear-gradient(135deg,#1a1a2e,#0f3460);padding:2rem;
        border-radius:12px;color:white;text-align:center;margin-bottom:1.5rem;}
.header h1{font-size:2.2rem;font-weight:700;margin:0;}
.skill-tag{display:inline-block;background:#eaf4fb;color:#1a5276;
           padding:3px 10px;border-radius:15px;font-size:.8rem;margin:2px;}
.skill-miss{display:inline-block;background:#fdf2f8;color:#922b21;
            padding:3px 10px;border-radius:15px;font-size:.8rem;margin:2px;}
</style>
""", unsafe_allow_html=True)


def get(path, timeout=10):
    try:
        r = httpx.get(f"{API}{path}", timeout=timeout)
        r.raise_for_status(); return r.json()
    except httpx.ConnectError:
        st.error("❌ Backend offline. Run: uvicorn backend.main:app --reload --port 8000")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def post(path, json_data=None, files=None, timeout=30):
    try:
        if files:
            r = httpx.post(f"{API}{path}", files=files, timeout=timeout)
        else:
            r = httpx.post(f"{API}{path}", json=json_data, timeout=timeout)
        r.raise_for_status(); return r.json()
    except Exception as e:
        st.error(f"API error: {e}"); return None


st.markdown('<div class="header"><h1>🧠 Intelligent Hiring Engine</h1>'
            '<p>Explainable AI Resume Screening + Candidate Fit Prediction</p></div>',
            unsafe_allow_html=True)

page = st.sidebar.selectbox("Navigate",
    ["📤 Upload Resumes","📋 Post Job","🏆 Rank Candidates","📊 Analytics"])

st.sidebar.markdown("---")
health = get("/health") or {}
icon = "🟢" if health.get("status") == "healthy" else "🔴"
st.sidebar.markdown(f"{icon} API: `{health.get('status','offline')}`")
st.sidebar.markdown("---\n**Ranking Weights**")
st.sidebar.markdown("- Semantic: **40%**\n- ML Score: **35%**\n- Skills: **15%**\n- Experience: **10%**")


# ── Page 1: Upload Resumes ────────────────────────────────────────────────────
if page == "📤 Upload Resumes":
    st.header("📤 Upload Resumes")
    files = st.file_uploader("Drag & drop PDFs / DOCX", accept_multiple_files=True, type=["pdf","docx","doc"])

    col1, col2 = st.columns([2,1])
    with col2:
        st.markdown("**Extracts:** Name · Email · Skills · Experience · Education · Projects")
        st.markdown("**Detects:** Duplicates · Keyword Stuffing · Hidden Text")

    if files and st.button("🚀 Parse All", type="primary", use_container_width=True):
        bar = st.progress(0)
        for i, f in enumerate(files):
            with st.spinner(f"Parsing {f.name}…"):
                resp = post("/resumes/upload", files={"file":(f.name, f.read(), f.type)})
            if resp:
                warns = resp.get("warnings",[])
                msg = f"✅ **{f.name}** — ID: `{str(resp.get('resume_id',''))[:8]}…`"
                if warns: msg += " " + " | ".join(warns)
                st.markdown(msg)
            bar.progress((i+1)/len(files))
        st.success("Done!")

    st.markdown("---")
    st.subheader("📁 All Resumes")
    resumes = get("/resumes/") or []
    if resumes:
        df = pd.DataFrame([{
            "Name": r.get("candidate_name","?"), "Email": r.get("email","—"),
            "Exp (yr)": round(r.get("total_experience_months",0)/12,1),
            "Skills": len(r.get("skills",[])), "Education": r.get("highest_education_level","—"),
            "Dup": "⚠️" if r.get("is_duplicate") else "✅",
            "Stuffing": "⚠️" if r.get("has_keyword_stuffing") else "✅",
        } for r in resumes])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"{len(resumes)} resumes")
    else:
        st.info("No resumes yet.")


# ── Page 2: Post Job ──────────────────────────────────────────────────────────
elif page == "📋 Post Job":
    st.header("📋 Post a Job Description")
    SAMPLE = """Senior Machine Learning Engineer

We need a Senior ML Engineer to build production AI systems.

Requirements:
• 4+ years ML experience
• Expert Python, TensorFlow or PyTorch
• NLP, BERT/Transformer models
• MLOps: MLflow, Kubeflow or Airflow
• SQL, Apache Spark for large-scale data
• Docker, Kubernetes for deployment
• AWS SageMaker or GCP Vertex AI

Preferred: LLM/RAG experience, XGBoost, AWS certification

Education: B.Tech / M.Tech in CS or AI"""

    jd_text = st.text_area("Paste JD", value=SAMPLE, height=280)
    if st.button("🔍 Parse with AI", type="primary") and jd_text.strip():
        with st.spinner("LLM parsing…"):
            r = post("/jobs/", json_data={"raw_text": jd_text})
        if r:
            st.success(f"JD created — ID: `{r.get('jd_id')}`")
            st.session_state["jd_id"] = r.get("jd_id")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Title:** {r.get('title')}")
                st.markdown(f"**Min Exp:** {r.get('min_experience_years',0)}+ yr")
                st.markdown(f"**Education:** {r.get('education_requirement','—')}")
            with col_b:
                st.markdown("**Required Skills:**")
                for s in r.get("required_skills",[])[:10]:
                    tag = "🔴" if s.get("is_mandatory") else "🟡"
                    st.markdown(f'<span class="skill-tag">{tag} {s["name"]}</span>', unsafe_allow_html=True)

    st.markdown("---")
    jds = get("/jobs/") or []
    if jds:
        st.subheader("Posted Jobs")
        for jd in jds:
            with st.expander(f"📌 {jd.get('title')} — `{str(jd.get('jd_id',''))[:8]}…`"):
                st.markdown(f"**ID:** `{jd.get('jd_id')}`  |  **Min Exp:** {jd.get('min_experience_years',0)}+ yr")
                skills = ", ".join(s["name"] for s in jd.get("required_skills",[])[:8])
                st.markdown(f"**Required:** {skills}")
                if st.button("Select →", key=f"sel_{jd['jd_id']}"):
                    st.session_state["jd_id"] = jd["jd_id"]
                    st.success("Selected! Go to 🏆 Rank Candidates")


# ── Page 3: Rank Candidates ───────────────────────────────────────────────────
elif page == "🏆 Rank Candidates":
    st.header("🏆 Rank Candidates")
    jds = get("/jobs/") or []
    resumes = get("/resumes/") or []

    if not jds: st.warning("Post a job first."); st.stop()
    if not resumes: st.warning("Upload resumes first."); st.stop()

    c1, c2, c3 = st.columns([3,1,1])
    jd_map = {f"{j['title']} ({str(j['jd_id'])[:8]}…)": j["jd_id"] for j in jds}
    with c1: sel = st.selectbox("Job", list(jd_map.keys()))
    with c2: top_k = st.number_input("Top K", 5, 50, 10)
    with c3: show_rej = st.checkbox("Show Rejected", False)

    if st.button("🚀 Run Ranking Engine", type="primary", use_container_width=True):
        with st.spinner("Computing embeddings + ML ranking + SHAP explanations…"):
            t0 = time.time()
            result = post("/ranking/rank", json_data={
                "jd_id": jd_map[sel], "top_k": top_k, "include_rejected": show_rej
            })
            elapsed = time.time() - t0

        if not result: st.stop()
        cands = result.get("ranked_candidates", [])
        st.success(f"Ranked {result.get('total_candidates')} candidates in {elapsed:.1f}s | Showing {len(cands)}")

        # KPIs
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Total", result.get("total_candidates"))
        m2.metric("Strong Hire", sum(1 for c in cands if c.get("recommendation")=="Strong Hire"))
        m3.metric("Consider", sum(1 for c in cands if c.get("recommendation")=="Consider"))
        m4.metric("Avg Score", f"{sum(c['final_score'] for c in cands)/max(len(cands),1):.1f}")

        if cands:
            df_chart = pd.DataFrame([{
                "Name": c.get("candidate_name","?")[:15],
                "Score": c.get("final_score",0),
                "Rec": c.get("recommendation",""),
            } for c in cands])
            fig = px.bar(df_chart, x="Name", y="Score", color="Rec",
                color_discrete_map={"Strong Hire":"#2ecc71","Consider":"#f39c12","Reject":"#e74c3c"},
                title="Candidate Scores", text="Score")
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        for c in cands:
            score = c.get("final_score",0)
            rec = c.get("recommendation","")
            icon = {"Strong Hire":"🟢","Consider":"🟡","Reject":"🔴"}.get(rec,"⚪")
            with st.expander(f"{icon} #{c.get('rank')} {c.get('candidate_name','?')} — {score:.1f}/100 — {rec}",
                             expanded=c.get("rank",99)<=2):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**Email:** {c.get('email') or '—'}")
                    for label, val in [
                        ("Semantic",c.get("semantic_score",0)),
                        ("ML Score",c.get("ml_score",0)),
                        ("Skill Match",c.get("skill_match_score",0)),
                        ("Experience",c.get("experience_score",0))
                    ]:
                        st.markdown(f"*{label}:* {val:.1f}%")
                        st.progress(val/100)
                with cb:
                    sg = c.get("skill_gap",{})
                    if sg.get("matched_skills"):
                        st.markdown("**✅ Matched:**")
                        st.markdown(" ".join(f'<span class="skill-tag">{s}</span>' for s in sg["matched_skills"][:7]), unsafe_allow_html=True)
                    if sg.get("missing_mandatory"):
                        st.markdown("**❌ Missing mandatory:**")
                        st.markdown(" ".join(f'<span class="skill-miss">{s}</span>' for s in sg["missing_mandatory"]), unsafe_allow_html=True)
                    if sg.get("missing_preferred"):
                        st.markdown("**⚠️ Missing preferred:**")
                        st.markdown(" ".join(f'<span class="skill-tag">{s}</span>' for s in sg["missing_preferred"][:5]), unsafe_allow_html=True)

                pos = c.get("top_positive_factors",[])
                neg = c.get("top_negative_factors",[])
                if pos or neg:
                    pa, pb = st.columns(2)
                    with pa:
                        st.markdown("**Why selected:**")
                        for p in pos: st.markdown(f"- {p}")
                    with pb:
                        st.markdown("**Concerns:**")
                        for n in neg: st.markdown(f"- {n}")

                flags = []
                if c.get("is_duplicate"): flags.append("⚠️ Duplicate")
                if c.get("has_keyword_stuffing"): flags.append("⚠️ Keyword Stuffing")
                if flags: st.warning("  |  ".join(flags))

        if cands:
            csv = pd.DataFrame([{
                "Rank":c["rank"],"Name":c.get("candidate_name"),"Email":c.get("email"),
                "Score":c.get("final_score"),"Recommendation":c.get("recommendation"),
                "SemanticPct":c.get("semantic_score"),"SkillMatchPct":c.get("skill_match_score"),
            } for c in cands]).to_csv(index=False)
            st.download_button("⬇️ Export CSV", csv, "ranking_results.csv", "text/csv", use_container_width=True)


# ── Page 4: Analytics ─────────────────────────────────────────────────────────
elif page == "📊 Analytics":
    st.header("📊 Analytics Dashboard")
    stats = get("/analytics/dashboard")
    if not stats: st.info("Upload resumes to see analytics."); st.stop()

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Resumes", stats.get("total_resumes",0))
    k2.metric("Duplicates", stats.get("duplicate_count",0))
    k3.metric("Keyword Stuffed", stats.get("keyword_stuffing_count",0))
    k4.metric("Avg Experience", f"{stats.get('avg_experience_years',0):.1f} yr")

    c1, c2 = st.columns(2)
    with c1:
        top = stats.get("top_skills",[])
        if top:
            fig = px.bar(pd.DataFrame(top[:15]), x="count", y="skill", orientation="h",
                title="Top 15 Skills", color="count", color_continuous_scale="Blues")
            fig.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        edu = stats.get("education_distribution",{})
        if edu:
            fig = px.pie(values=list(edu.values()), names=list(edu.keys()),
                title="Education Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)

    total = max(stats.get("total_resumes",1),1)
    dup_p = stats.get("duplicate_count",0)/total*100
    stf_p = stats.get("keyword_stuffing_count",0)/total*100
    fig = go.Figure(go.Bar(
        x=["Clean","Duplicates","Keyword Stuffed"], y=[100-dup_p-stf_p,dup_p,stf_p],
        marker_color=["#2ecc71","#f39c12","#e74c3c"],
        text=[f"{v:.1f}%" for v in [100-dup_p-stf_p,dup_p,stf_p]], textposition="outside"))
    fig.update_layout(title="Resume Quality Distribution", height=300)
    st.plotly_chart(fig, use_container_width=True)
