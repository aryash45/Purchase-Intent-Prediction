# Purchase Intent Prediction — E-Commerce Conversion ML

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Can we predict which website visitors will buy — before they decide?**  
> This project builds a binary classifier on 12,330 real e-commerce sessions to flag high-intent users, enabling smarter retargeting, personalized discounts, and reduced cart abandonment.

---

## Results at a Glance

| Model | Accuracy | AUC-ROC | Recall (Purchase) |
|---|---|---|---|
| Logistic Regression | ~87% | ~0.87 | baseline |
| K-Nearest Neighbors (tuned) | ~88% | ~0.89 | improved |
| Decision Tree | ~87% | ~0.86 | interpretable |
| **Random Forest** | **~90%** | **~0.93** | **best** |

*Class imbalance (84/16 split) corrected with SMOTE before training.*

---

## Business Problem

~85% of e-commerce visitors leave without purchasing. Identifying **which sessions are likely to convert** lets platforms act in real time — trigger a discount, send a push notification, or prioritize a customer support chat. A 1% lift in conversion rate translates to millions in revenue for mid-size retailers.

---

## What This Project Covers

| Stage | What I Did |
|---|---|
| **Data Understanding** | Explored 18 behavioral features across 12,330 sessions |
| **Preprocessing** | Encoded categoricals, scaled numerics with StandardScaler |
| **Class Imbalance** | Applied SMOTE to oversample the minority purchase class |
| **Dimensionality Reduction** | PCA retaining 95% variance — reduced noise and training time |
| **Modelling** | Trained and compared 4 classifiers with cross-validation |
| **Tuning** | Hyperparameter search for optimal K in KNN |
| **Evaluation** | Accuracy, AUC-ROC, confusion matrix, classification report |

---

## Dataset

**UCI Online Shoppers Purchasing Intention** — 12,330 web sessions  
18 features: `PageValues`, `BounceRates`, `ExitRates`, `SpecialDay`, `VisitorType`, session duration attributes, and more.  
Target: `Revenue` (True if session ended in purchase)

→ [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

---

## Key Findings

- **`PageValues`** is the single strongest predictor of purchase intent — users who viewed high-value pages are disproportionately likely to buy
- **Bounce rate and exit rate** together form a reliable "disengagement signal" — high on both = very unlikely to convert
- **SMOTE** improved purchase-class recall substantially — models without it were biased toward always predicting "no purchase"
- **Random Forest** generalised best, outperforming all single classifiers on both accuracy and AUC-ROC

---

## Tech Stack

```
Python 3.10+   pandas   numpy   scikit-learn   imbalanced-learn   matplotlib   seaborn
```

---

## Run Locally

```bash
git clone https://github.com/aryash45/Purchase-Intent-Prediction.git
cd purchase-intent-prediction
pip install -r requirements.txt
jupyter notebook Purchase_Intent_Prediction.ipynb
```

---

## Project Structure

```
purchase-intent-prediction/
├── Purchase_Intent_Prediction.ipynb    # Full analysis: EDA → modelling → evaluation
├── requirements.txt                    # All dependencies pinned
├── .gitignore
└── README.md
```

---

## Roadmap (Planned Enhancements)

- [ ] Replace KNN with XGBoost / LightGBM for benchmark comparison
- [ ] SHAP values for feature-level explainability per prediction
- [ ] Streamlit app — live session input → purchase probability output
- [ ] Deploy to Streamlit Cloud with public URL

---

## Author

**Aryash Gupta** — B.Tech, CS  
[GitHub](https://github.com/aryash45) · [LinkedIn](https://linkedin.com/in/aryashgupta)
