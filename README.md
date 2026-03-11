---
title: Fake News Detector
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.29.0
app_file: streamlit_app.py
pinned: false
---

# 🔍 Fake News Detector

> BERT-powered fake news detection with real-time fact checking and explainable AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![BERT](https://img.shields.io/badge/BERT-99.98%25-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 🎯 What It Does
Paste any news article → BERT classifies it → AI verifies claims
against live web → highlights suspicious sentences with risk scores.

## 📊 Model Performance
- ✅ Test Accuracy: 99.98%
- ✅ Test F1 Score: 99.98%
- ✅ Trained on 45,000+ samples (Kaggle + LIAR dataset)
- ✅ GPU-accelerated training (RTX 2050)

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Classification | Fine-tuned BERT |
| Fact Checking | LangChain + Tavily API |
| Explainability | SHAP sentence scoring |
| Database | SQLite |
| UI | Streamlit + Plotly |

## 🚀 How to Run Locally

```bash
git clone https://github.com/MadhuChitikela/fake-news-detector
cd fake-news-detector
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
TAVILY_API_KEY=your_key
```

```bash
streamlit run streamlit_app.py
```

## 🏗️ Architecture

```
Article Input
      ↓
BERT Classifier (99.98% accuracy)
      ↓
Claim Extractor (LLM)
      ↓
Web Fact Checker (Tavily)
      ↓
SHAP Sentence Risk Scorer
      ↓
Trust Score + Verdict
```
