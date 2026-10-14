# Customer Churn Prediction & Intelligence Lab (ChurnShield AI)

An end-to-end Machine Learning project comparing **Gradient Boosting (Champion, ROC-AUC 0.872)**, **Random Forest (0.860)**, and **Logistic Regression (0.777)** on bank customer churn with a live interactive web dashboard and real-time simulator.

---

## ⚡ Live Demo & Web App

A production-grade, interactive dashboard is included right in the repository:

- 🔮 **Real-Time Customer Risk Simulator**: Live risk scoring via embedded 150-tree Gradient Boosting and Logistic Regression engines with zero backend latency.
- ⚡ **One-Click Personas**: Instant simulation of high-risk seniors in Germany, safe multi-product savers, inactive accounts, and zero-balance digital users.
- 📊 **Model Benchmarks & ROC Curves**: High-resolution ROC canvas with interactive decision threshold tuning (&tau; = 0.05 to 0.95) and live updating confusion matrices.
- 🔍 **Feature Drivers & Non-Linear Insights**: Interactive permutation importance and deep dive into the **NumOfProducts Paradox** (why holding 2 products is optimal at 7.6% churn, while 3+ products spikes to 83%+).
- 📁 **Batch Scoring & Test Set Cohort**: Search and filter 30 held-out test customers, drag-and-drop custom CSVs to score immediately, and export results with one click.
- 📘 **Executive Strategy Briefing**: Complete strategic roadmap and retention playbooks based on `RESULTS.md`.

### Run the Live Demo Locally:
```bash
# Start local HTTP server
python3 -m http.server 8080

# Open in browser:
# http://localhost:8080/index.html
```

### Deploy to Netlify / GitHub Pages:
This application is fully standalone and static-ready with client-side inference. You can deploy it directly to Netlify:
- **Build command:** *(leave blank)*
- **Publish directory:** `.` (root directory containing `index.html`, `styles.css`, `app.js`, and `demo_data.json`)

---

## 🔬 Model Training & Evaluation Pipeline

To retrain the models and regenerate evaluation artifacts:

```bash
pip install -r requirements.txt
python churn_model.py data/Churn_Modelling.csv output
```

To re-export live demo assets:
```bash
python export_data.py
```

### Held-out Test Set Performance (n = 2,000)

| Model Architecture | CV ROC-AUC | Test ROC-AUC | PR-AUC | Precision @ Tuned | Recall @ Tuned | F1 Score | Tuned Cutoff |
|---|---|---|---|---|---|---|---|
| **Gradient Boosting** (Champion) | **0.863 ± 0.009** | **0.872** | **0.725** | 62.5% | **65.1%** | **0.638** | **0.30** |
| Random Forest (500 trees) | 0.858 ± 0.012 | 0.860 | 0.690 | 65.6% | 61.4% | 0.635 | 0.52 |
| Logistic Regression (Weighted) | 0.766 ± 0.021 | 0.777 | 0.468 | 41.3% | 64.4% | 0.503 | 0.54 |

For full experimental analysis and driver findings, see [RESULTS.md](file:///Users/shivam356/Downloads/Chrome_download/unlox_project/churn_prediction/RESULTS.md).
