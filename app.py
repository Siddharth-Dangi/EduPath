"""
app.py  —  EduPulse: AI-Powered Academic Intelligence Platform
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from data_engine import PerformanceAnalyzer
from tutor_agent import chat as tutor_chat

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPulse — AI Academic Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (premium dark theme) ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B0F1A;
    color: #E2E8F0;
}
.main { background-color: #0B0F1A; }
section[data-testid="stSidebar"] { background: #0F1624 !important; border-right: 1px solid #1E293B; }

/* ── Header ── */
.edu-header {
    display: flex; align-items: center; gap: 16px;
    padding: 32px 8px 8px 8px;
}
.edu-logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; box-shadow: 0 0 24px rgba(99,102,241,0.45);
}
.edu-brand { display: flex; flex-direction: column; }
.edu-brand h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #818CF8, #C084FC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.edu-brand p { margin: 0; font-size: 0.82rem; color: #64748B; font-weight: 400; }

/* ── Stat cards ── */
.stat-row { display: flex; gap: 16px; margin: 20px 0; }
.stat-card {
    flex: 1; background: #111827;
    border: 1px solid #1E293B; border-radius: 16px;
    padding: 20px 22px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(99,102,241,0.18); }
.stat-card .label { font-size: 0.73rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; }
.stat-card .value { font-size: 1.65rem; font-weight: 700; color: #E2E8F0; margin-top: 4px; }
.stat-card .badge { font-size: 0.72rem; padding: 2px 10px; border-radius: 99px; margin-top: 6px; display: inline-block; }
.badge-green  { background: rgba(16,185,129,0.15); color: #34D399; }
.badge-yellow { background: rgba(245,158,11,0.15); color: #FCD34D; }
.badge-red    { background: rgba(239,68,68,0.15);  color: #FCA5A5; }
.badge-purple { background: rgba(139,92,246,0.15); color: #C4B5FD; }

/* ── Result panel ── */
.result-panel {
    background: linear-gradient(135deg, #1E1B4B 0%, #111827 100%);
    border: 1px solid #3730A3; border-radius: 16px;
    padding: 24px; margin-top: 16px;
}
.result-panel h4 {
    font-family: 'Space Grotesk', sans-serif;
    color: #A5B4FC; margin: 0 0 14px 0; font-size: 1rem;
}
.result-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1E293B; }
.result-row:last-child { border-bottom: none; }
.result-key   { color: #94A3B8; font-size: 0.87rem; }
.result-val   { color: #E2E8F0; font-size: 0.87rem; font-weight: 600; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important; font-weight: 600 !important;
    color: #64748B !important; border-radius: 10px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #818CF8 !important;
    background: rgba(99,102,241,0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 10px 22px !important; font-size: 0.88rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.55) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div { background: #6366F1 !important; }

/* ── Inputs ── */
.stTextInput input, .stSelectbox select {
    background: #111827 !important; color: #E2E8F0 !important;
    border: 1px solid #1E293B !important; border-radius: 10px !important;
}

/* ── Info / warning / success banners ── */
.stInfo, .stSuccess, .stWarning, .stError {
    border-radius: 12px !important; border: none !important;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] > div {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}
div[data-testid="stChatInputTextArea"] { border-radius: 16px !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 14px !important;
    padding: 14px !important; margin-bottom: 10px !important;
}

/* ── Sidebar labels ── */
.sidebar-label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.06em; color: #64748B;
    font-weight: 600; margin-bottom: 6px;
}
.sidebar-metric {
    background: #111827; border: 1px solid #1E293B;
    border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;
}
.sidebar-metric .sm-val { font-size: 1.2rem; font-weight: 700; color: #A5B4FC; }
.sidebar-metric .sm-lbl { font-size: 0.78rem; color: #64748B; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #1E293B; }

/* ── Matplotlib dark ── */
.stPlot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state bootstrap ────────────────────────────────────────────────────
if "analyzer" not in st.session_state:
    st.session_state.analyzer = PerformanceAnalyzer()
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "learner_tier" not in st.session_state:
    st.session_state.learner_tier = None
if "pass_prob" not in st.session_state:
    st.session_state.pass_prob = None


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-header">
  <div class="edu-logo">⚡</div>
  <div class="edu-brand">
    <h1>EduPulse</h1>
    <p>AI-Powered Academic Intelligence · Real-time Coaching · Predictive Analytics</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.markdown("---")

    if st.session_state.learner_tier:
        tier = st.session_state.learner_tier
        prob = st.session_state.pass_prob
        tier_color = {"Excelling": "badge-green", "Developing": "badge-yellow", "Struggling": "badge-red"}.get(tier, "badge-purple")

        st.markdown(f"""
        <div class='sidebar-metric'>
          <div class='sm-lbl'>Active Learner Tier</div>
          <div class='sm-val'>{tier}</div>
        </div>
        <div class='sidebar-metric'>
          <div class='sm-lbl'>Pass Probability</div>
          <div class='sm-val'>{prob:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No learner profile loaded yet. Use the **Analytics** tab to generate and analyse data.")

    st.markdown("---")
    st.markdown("#### About EduPulse")
    st.caption(
        "EduPulse combines ML-driven learner analytics with a LangGraph "
        "agentic tutor to give every student a personalised academic coach."
    )
    st.markdown("---")
    if st.session_state.analyzer.is_ready:
        acc = st.session_state.analyzer.model_accuracy
        st.markdown(f"""
        <div class='sidebar-metric'>
          <div class='sm-lbl'>Model Accuracy</div>
          <div class='sm-val'>{acc:.1%}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_analytics, tab_coach = st.tabs(["📊  Analytics & Predictions", "💬  AI Tutor"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### Learning Analytics Dashboard")

    left, right = st.columns([1, 2], gap="large")

    with left:
        # ── Step 1: Generate data ──
        st.markdown("#### Step 1 — Load Dataset")
        if st.button("⚙️  Generate & Train Models", use_container_width=True):
            with st.spinner("Building dataset and training models…"):
                df_raw = st.session_state.analyzer.build_sample_dataset()
                df_trained = st.session_state.analyzer.fit(df_raw)
                st.session_state.dataset = df_trained
            st.success(f"✅  Models ready! Accuracy: {st.session_state.analyzer.model_accuracy:.1%}")

        st.markdown("---")

        # ── Step 2: Simulate a student ──
        st.markdown("#### Step 2 — Simulate a Learner")
        if st.session_state.analyzer.is_ready:
            quiz_avg    = st.slider("📝  Average Quiz Score",   0,   100, 72)
            study_hours = st.slider("⏱️  Weekly Study Hours",   0,    60, 18)
            tasks_done  = st.slider("✅  Tasks Completed",      0,    12,  6)
            attend_pct  = st.slider("🏫  Attendance %",        50,   100, 80)

            if st.button("🔍  Analyse Learner", use_container_width=True):
                result = st.session_state.analyzer.evaluate_student(
                    quiz_avg, study_hours, tasks_done, attend_pct
                )
                st.session_state.learner_tier = result["tier"]
                st.session_state.pass_prob    = result["pass_probability"]

                tier   = result["tier"]
                prob   = result["pass_probability"]
                passed = result["will_pass"]

                tier_emoji = {"Excelling": "🌟", "Developing": "📈", "Struggling": "🆘"}.get(tier, "📊")
                outcome_badge = "badge-green" if passed else "badge-red"

                st.markdown(f"""
                <div class="result-panel">
                  <h4>{tier_emoji} Analysis Complete</h4>
                  <div class="result-row">
                    <span class="result-key">Learner Tier</span>
                    <span class="result-val">{tier}</span>
                  </div>
                  <div class="result-row">
                    <span class="result-key">Pass Probability</span>
                    <span class="result-val">{prob:.1%}</span>
                  </div>
                  <div class="result-row">
                    <span class="result-key">Predicted Outcome</span>
                    <span class="result-val">{'✅ Pass' if passed else '❌ At Risk'}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Generate and train models first.")

    with right:
        if st.session_state.dataset is not None:
            df = st.session_state.dataset

            # ── Stat cards ──
            total   = len(df)
            pct_exc = (df["learner_tier"] == "Excelling").mean()
            pct_str = (df["learner_tier"] == "Struggling").mean()
            pass_rt = df["passed"].mean()

            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-card">
                <div class="label">Total Students</div>
                <div class="value">{total}</div>
                <div class="badge badge-purple">Synthetic dataset</div>
              </div>
              <div class="stat-card">
                <div class="label">Excelling</div>
                <div class="value">{pct_exc:.0%}</div>
                <div class="badge badge-green">High performers</div>
              </div>
              <div class="stat-card">
                <div class="label">Struggling</div>
                <div class="value">{pct_str:.0%}</div>
                <div class="badge badge-red">Need intervention</div>
              </div>
              <div class="stat-card">
                <div class="label">Pass Rate</div>
                <div class="value">{pass_rt:.0%}</div>
                <div class="badge badge-yellow">Predicted</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Dataset preview ──
            st.markdown("#### 📋 Dataset Preview")
            display_cols = ["student_id", "quiz_avg", "study_hours", "tasks_done", "attendance_pct", "learner_tier", "passed"]
            st.dataframe(
                df[display_cols].head(8).rename(columns={
                    "student_id": "ID", "quiz_avg": "Quiz Avg",
                    "study_hours": "Study Hrs", "tasks_done": "Tasks Done",
                    "attendance_pct": "Attendance %", "learner_tier": "Tier", "passed": "Passed"
                }),
                use_container_width=True, hide_index=True
            )

            # ── Scatter plot ──
            st.markdown("#### 🔵 Learner Cluster Map")
            plt.style.use("dark_background")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor("#111827")

            palette = {"Excelling": "#34D399", "Developing": "#FCD34D", "Struggling": "#F87171"}

            sns.scatterplot(
                data=df, x="study_hours", y="quiz_avg",
                hue="learner_tier", palette=palette,
                alpha=0.75, s=50, ax=axes[0]
            )
            axes[0].set_facecolor("#0B0F1A")
            axes[0].set_title("Study Hours vs Quiz Score", color="#E2E8F0", fontsize=11)
            axes[0].set_xlabel("Study Hours / Week", color="#94A3B8")
            axes[0].set_ylabel("Quiz Average", color="#94A3B8")
            axes[0].tick_params(colors="#64748B")
            axes[0].spines[:].set_color("#1E293B")
            axes[0].legend(title="Tier", title_fontsize=8, fontsize=8, labelcolor="#E2E8F0",
                           facecolor="#111827", edgecolor="#1E293B")

            tier_counts = df["learner_tier"].value_counts()
            colors_bar  = [palette.get(t, "#94A3B8") for t in tier_counts.index]
            axes[1].bar(tier_counts.index, tier_counts.values, color=colors_bar, edgecolor="#0B0F1A", linewidth=1.5, width=0.55)
            axes[1].set_facecolor("#0B0F1A")
            axes[1].set_title("Learner Tier Distribution", color="#E2E8F0", fontsize=11)
            axes[1].set_xlabel("Learner Tier", color="#94A3B8")
            axes[1].set_ylabel("Number of Students", color="#94A3B8")
            axes[1].tick_params(colors="#64748B")
            axes[1].spines[:].set_color("#1E293B")

            plt.tight_layout(pad=2)
            st.pyplot(fig)
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#334155;">
              <div style="font-size:3rem;">📊</div>
              <p style="font-size:1rem;margin-top:12px;">
                Click <b>Generate &amp; Train Models</b> on the left to load the analytics dashboard.
              </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AI Tutor
# ══════════════════════════════════════════════════════════════════════════════
with tab_coach:
    st.markdown("### 💬 Personalised AI Tutor")

    if st.session_state.learner_tier is None:
        st.warning(
            "⚠️  **No learner profile detected.** "
            "Please go to the **Analytics** tab, generate data, and click **Analyse Learner** first."
        )
    else:
        tier = st.session_state.learner_tier
        prob = st.session_state.pass_prob
        tier_color = {"Excelling": "#34D399", "Developing": "#FCD34D", "Struggling": "#F87171"}.get(tier, "#A5B4FC")

        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1E293B;border-radius:14px;padding:16px 20px;margin-bottom:20px;display:flex;align-items:center;gap:16px;">
          <div style="font-size:2rem;">🤖</div>
          <div>
            <div style="font-size:0.78rem;color:#64748B;text-transform:uppercase;letter-spacing:0.05em;">Active Session</div>
            <div style="color:#E2E8F0;font-weight:600;font-size:0.95rem;">
              Tier: <span style="color:{tier_color}">{tier}</span> &nbsp;|&nbsp;
              Pass Probability: <span style="color:#A5B4FC">{prob:.0%}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Render history
        for msg in st.session_state.chat_history:
            role = getattr(msg, "type", "")
            if role == "human":
                st.chat_message("user", avatar="🧑‍🎓").write(msg.content)
            elif role == "ai" and msg.content.strip():
                st.chat_message("assistant", avatar="⚡").write(msg.content)

        # Chat input
        if prompt := st.chat_input("Ask EduPulse anything — concepts, study tips, problems…"):
            st.chat_message("user", avatar="🧑‍🎓").write(prompt)

            with st.spinner("EduPulse is thinking…"):
                try:
                    updated_history = tutor_chat(
                        user_message=prompt,
                        learner_tier=st.session_state.learner_tier,
                        pass_probability=st.session_state.pass_prob,
                        history=st.session_state.chat_history,
                    )
                    st.session_state.chat_history = updated_history
                    ai_msgs = [m for m in updated_history if getattr(m, "type", "") == "ai" and m.content.strip()]
                    if ai_msgs:
                        st.chat_message("assistant", avatar="⚡").write(ai_msgs[-1].content)
                except Exception as exc:
                    st.error(f"⚠️ Could not reach the AI tutor: {exc}\n\nMake sure `GROQ_API_KEY` is set in your `.env` file.")

        # Clear button
        if st.session_state.chat_history:
            st.markdown("---")
            if st.button("🗑️  Clear Conversation"):
                st.session_state.chat_history = []
                st.rerun()
