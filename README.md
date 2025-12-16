# loan-default-loss-prediction
From binary default flags to continuous loss prediction using Random Forest regression

# 📉 Loan Default Loss Prediction  
**Applied Data Mining – Financial Risk Modeling**
---

##  Project Overview

This project builds a supervised machine learning model to **predict loan loss severity**, defined as the dollar amount a lender loses conditional on borrower default. Unlike traditional credit risk models that focus on binary default classification, this work estimates **continuous loss values**, providing a more realistic and actionable view of credit risk.

A **Random Forest regression model** is trained and evaluated using **Mean Absolute Error (MAE)**. The final model is used to generate predictions for an unlabeled test dataset, producing a submission-ready file:  
`submission_loss_predictions.csv`.

---

##  Business Motivation

Accurate loss prediction supports key financial decision-making processes:

- **Underwriting & Approval** – Adjust loan amounts or reject applications with high expected loss  
- **Risk-Based Pricing** – Set interest rates and fees proportional to predicted loss  
- **Capital Planning & Stress Testing** – Estimate portfolio losses under adverse scenarios  
- **Portfolio Optimization** – Identify and rebalance high-loss portfolio segments  

By modeling loss severity directly, this project aligns closely with real-world credit risk management practices.

---

##  Dataset Description

Three datasets are used:

| Dataset | Description |
|------|------------|
| `train_v3.csv` | Training data containing predictors and target variable `loss` |
| `test_v3.csv` | Labeled test data used for model evaluation |
| `test__no_lossv3.csv` | Unlabeled test data used for final predictions |

**Key characteristics:**
- Hundreds of numeric predictor variables  
- Highly skewed loss distribution  
- Significant missing values across features  

---

##  Exploratory Data Analysis (EDA)

Key insights from EDA:

- Over **90% of loans have zero loss**
- Loss distribution is **highly right-skewed with a long tail**
- Several predictors have **15–18% missing values**
- Individual features show weak linear correlation with loss

 Visual outputs include:
- Histogram of the loss distribution  
- Missing value summaries  
- Feature importance plots  

---

##  Data Preprocessing

The following preprocessing steps were applied:

- **Median imputation** for all numeric predictors  
  - Medians computed using training data only (prevents data leakage)
- Removal of accidental index columns (e.g., `"Unnamed"`)
- Ensured identical predictor sets across all datasets
- No feature scaling required (tree-based model)

---

##  Modeling Strategy

Two models were implemented and compared:

### 1️⃣ Baseline Mean Model
- Predicts the mean loss from the training data
- Establishes a minimum performance benchmark

### 2️⃣ Random Forest Regression (Ranger)
- Captures nonlinear relationships and interactions
- Robust to skewness and outliers
- Scales well to high-dimensional data
- Provides feature importance measures

---

##  Model Training & Hyperparameter Tuning

- **Train/Validation Split:** 80% / 20%
- **Evaluation Metric:** Mean Absolute Error (MAE)
- **Hyperparameters Tuned:**
  - `mtry`
  - `min.node.size`
- Tuning performed on a **30,000-row sample** for computational efficiency

**Best hyperparameters selected:**
- `mtry = 27`
- `min.node.size = 10`

---

##  Model Performance

| Dataset | Baseline MAE | Random Forest MAE |
|-------|--------------|------------------|
| Validation | ~1.45 | ~1.64 |
| Labeled Test | ~1.46 | ~1.64 |

> Although the Random Forest did not outperform the baseline in MAE, the model captures complex nonlinear structure and provides valuable feature importance insights, highlighting the noisy and highly skewed nature of loss data.

---

##  Feature Importance

- Top predictors were identified using impurity-based importance
- These features consistently help differentiate high-loss and low-loss loans
- Insights can support:
  - Risk segmentation
  - Policy design
  - Stress testing and monitoring

---

##  Business Use Cases

- **Underwriting:** Identify loans with excessive predicted loss  
- **Pricing:** Adjust rates using expected loss
- - **Capital Planning:** Estimate portfolio losses under stress scenarios  
- **Collections:** Prioritize high-risk, high-loss accounts
  
---

## ⚠️ Limitations

- Dataset lacks external credit bureau or income data
- Single-stage loss modeling may conflate zero-loss and positive-loss behavior
- Random Forest interpretability is limited
- Hyperparameter tuning scope was constrained




