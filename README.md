# OpenMath AI

A friendly, dashboard-style Streamlit tutor for Grade 9 math focused on
linear equations in the form **y = mx + b** — slope, y-intercept, graphing,
and word problems. It generates practice problems, offers scaffolded hints,
shows step-by-step solutions, and tracks mastery using a scikit-learn
Random Forest classifier.

## Features
- 📈 Slope, 🎯 y-intercept, 📐 graphing, 📝 word problems
- Practice problems at easy / medium / hard difficulty
- Scaffolded hints and step-by-step solutions
- Mastery tracking (scikit-learn RandomForestClassifier)
- Rule-based adaptive "next action" suggestions
- Student dashboard and teacher session view (with CSV export)
- Works fully offline; Perplexity API optional for free-response replies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Fork or clone this repo to your GitHub account.
2. Go to https://share.streamlit.io and click **New app**.
3. Pick this repo and set the entry point to `app.py`.
4. (Optional) In the app's **Secrets** settings, add:
   ```
   PERPLEXITY_API_KEY = "your_key_here"
   ```
   The app works without an API key — it falls back to built-in local
   explanations, hints, and practice problems.

## Files
- `app.py` — Streamlit UI, session state, routing
- `content.py` — topics, explanations, problem generator
- `ml_model.py` — Random Forest mastery classifier + rule-based policy
- `utils.py` — answer checking, graphing, optional Perplexity API helper
- `requirements.txt` — Python dependencies
