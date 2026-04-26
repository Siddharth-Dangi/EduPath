# ⚡ EduPulse — AI-Powered Academic Intelligence Platform

> A full-stack AI application that fuses **predictive machine learning** with a **LangGraph agentic tutor** to deliver truly personalised academic coaching.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-blueviolet)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Engine-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI_Framework-red)

---

## 📖 What Is EduPulse?

EduPulse analyses a learner's academic behaviour (quiz scores, study hours, task completion, attendance) and classifies them into performance tiers — *Struggling*, *Developing*, or *Excelling* — using an unsupervised **K-Means** model.

A supervised **Gradient Boosting** classifier then estimates their probability of passing.

These ML signals feed directly into an autonomous **LangGraph** agent that adjusts its tutoring tone, depth, and resource recommendations in real time, powered by **Groq's ultra-fast LLM API**.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔵 Learner Clustering | K-Means tiers learners into Struggling / Developing / Excelling |
| 📈 Pass Prediction | Gradient Boosting classifier with per-student probability score |
| 🤖 Agentic Tutor | LangGraph multi-step reasoning with tool use |
| 📚 RAG Knowledge Base | ChromaDB + HuggingFace embeddings for resource retrieval |
| 💬 Session Memory | Continuous, stateful tutoring conversations |
| 🎨 Premium Dark UI | Glassmorphism-inspired Streamlit dashboard |

---

## 🛠️ Tech Stack

- **UI**: Streamlit with custom CSS dark theme
- **Agentic Layer**: LangGraph + LangChain Core
- **ML**: Scikit-Learn (KMeans, GradientBoostingClassifier, MinMaxScaler)
- **LLM**: Groq API (`llama-3.3-70b-versatile`)
- **Vector Store**: ChromaDB + `all-MiniLM-L6-v2` embeddings

---

## 🚀 Quick Start

### 1. Clone & enter the project
```bash
git clone https://github.com/yourusername/EduPulse.git
cd EduPulse
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
```bash
cp .env.example .env
# Open .env and set GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com/).

### 5. Run
```bash
streamlit run app.py
```

---

## 🧠 Architecture

```
User Input (Sliders)
       │
       ▼
 data_engine.py          ← MinMaxScaler + KMeans + GradientBoosting
       │  tier + pass_prob
       ▼
 tutor_agent.py          ← LangGraph StateGraph
       │  system prompt injected with tier context
       ▼
 knowledge_base.py       ← ChromaDB RAG (fetch_study_material tool)
       │
       ▼
 Groq LLM (llama-3.3-70b-versatile)
```

---

## 📝 License
MIT
