# 💳 FINRISK 360 : Credit Risk Analytics & Decision Support System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-337AB7)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-red)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)

**An end-to-end machine learning based credit risk analytics and decision-support system for predicting loan default, understanding risk drivers, evaluating affordability, detecting early-warning signals, and generating portfolio-level risk insights.**

FINRISK 360 is designed to go beyond simply predicting whether a loan application is likely to default. The project combines **data quality analysis, exploratory data analysis, feature engineering, machine learning, probability-of-default scoring, affordability analysis, early warning signals, application-level fraud indicators, portfolio risk analysis, and explainable AI** into a unified credit-risk workflow.

---

## 📌 Project Highlights

- Built an end-to-end **credit risk analytics pipeline** using Python and machine learning
- Performed detailed **data quality auditing, cleaning, validation, and exploratory data analysis**
- Engineered domain-relevant features for **credit risk and affordability analysis**
- Developed machine learning models for **probability of default prediction**
- Used **XGBoost** as the primary tree-based modelling approach
- Converted predicted default probabilities into interpretable **Low, Medium, High, and Severe risk tiers**
- Identified **loan-to-income burden** as one of the strongest observed risk signals
- Built an **Affordability Advisor prototype** for evaluating different loan amounts against applicant income
- Designed a point-in-time **Early Warning System (EWS)** using model risk and supporting risk factors
- Developed transparent **application-level fraud indicator rules**
- Designed portfolio-level **Expected Loss, concentration risk, and stress-testing analysis**
- Integrated **SHAP-based explainability** for understanding individual model predictions
- Maintained explicit data and modelling limitations instead of creating unsupported financial assumptions
- Designed the project as a foundation for a future **Risk Decision Engine and Streamlit-based interface**

---

# 🎯 Problem Statement

Credit-risk assessment is not only about determining whether a borrower may default.

A practical risk analytics system should also answer:

- How risky is the application?
- What factors are contributing to the predicted risk?
- Is the requested loan reasonable relative to the applicant's income?
- Which applications require additional attention?
- Are there unusual combinations of application characteristics?
- Which segments contribute the most portfolio exposure?
- How sensitive is the portfolio to changes in lending conditions?
- Can the model's decision be explained to a human decision-maker?

FINRISK 360 addresses these questions through a structured machine-learning and risk-analytics workflow.

---

# 🧩 Overall Architecture

```text
                         FINRISK 360
                              │
                              ▼
                   Credit Risk Dataset
                              │
                              ▼
                  Data Quality & Cleaning
                              │
                              ▼
                     Exploratory Analysis
                              │
                              ▼
                    Feature Engineering
                              │
                              ▼
                  Machine Learning Models
                              │
                              ▼
                  Probability of Default
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
          Risk Scoring   Affordability    Portfolio
                │           Analysis         Risk
                │
                ▼
       Early Warning System
                │
                ▼
      Application-Level Fraud
             Indicators
                │
                ▼
        SHAP Explainability
                │
                ▼
       Risk Decision Support
```

---

# 📊 Dataset

The project uses the public **Credit Risk Dataset** containing borrower and loan-application characteristics.

The dataset is treated as **public/simulated credit-risk data** and is not represented as real bank customer data.

### Dataset Overview

| Property | Value |
|---|---:|
| Original Applications | 32,581 |
| Original Features | 12 |
| Cleaned Applications | 32,416 |
| Target Variable | `loan_status` |
| Non-default | `0` |
| Default | `1` |
| Final Default Rate | ~21.86% |

### Major Variables

The dataset contains information related to:

- Applicant age
- Applicant income
- Home ownership
- Employment length
- Loan intent
- Loan grade
- Loan amount
- Interest rate
- Loan status
- Loan-to-income burden
- Previous credit default indicator
- Credit history length

---

# 🧹 Data Cleaning & Quality Analysis

Before modelling, the dataset was subjected to a structured data-quality process.

The cleaning workflow included:

```text
Raw Dataset
     │
     ├── Missing Value Analysis
     │
     ├── Duplicate Detection
     │
     ├── Invalid Value Detection
     │
     ├── Categorical Validation
     │
     ├── Numerical Validation
     │
     └── Feature Quality Checks
             │
             ▼
       Cleaned Dataset
```

### Identified Data Quality Issues

The initial dataset contained:

- Missing employment-length values
- Missing interest-rate values
- Duplicate records
- Unrealistic age values
- Invalid employment-length values
- Strongly skewed income distribution
- Extreme but potentially meaningful income observations

### Treatment Strategy

The project avoids blindly deleting observations.

Examples:

- Exact duplicates → removed
- Invalid age values → treated as missing and imputed
- Invalid employment length → treated as missing and imputed
- Missing employment length → median imputation with missingness indicator
- Missing interest rate → median imputation with missingness indicator
- Extreme income → flagged and investigated rather than automatically deleted
- High loan-to-income burden → retained because it may represent meaningful risk information

---

# 📈 Exploratory Data Analysis

EDA was performed to understand both the distribution of variables and their relationship with loan default.

### Key Areas Analysed

- Numerical feature distributions
- Categorical distributions
- Loan-grade default rates
- Loan-intent default rates
- Home-ownership default rates
- Employment stability
- Debt-burden tiers
- Feature correlations with default
- Interaction between debt burden and loan intent
- Outlier and feature validation

---

# 💡 Key Risk Finding — Loan-to-Income Burden

One of the strongest observed risk patterns was associated with:

```text
loan_percent_income
```

This variable represents the available **loan-to-income burden proxy**.

It is calculated conceptually as:

```text
                 Loan Amount
Loan-to-Income = ─────────────
                 Annual Income
```

Because the dataset does not contain loan tenure or repayment schedules, FINRISK 360 **does not calculate actual EMI-to-income**.

Instead, the project consistently uses `loan_percent_income` as a **loan-to-income burden proxy**.

---

# 🚦 Debt-Burden Risk Tiers

The loan-to-income burden is grouped into four interpretable categories.

| Loan-to-Income Burden | Risk Tier |
|---:|---|
| `< 20%` | 🟢 Low |
| `20% – <35%` | 🟡 Moderate |
| `35% – 50%` | 🟠 High |
| `> 50%` | 🔴 Severe |

### Observed Default Rates

The cleaned dataset showed a strong increase in observed default rate as debt burden increased:

| Debt Burden Tier | Applications | Default Rate |
|---|---:|---:|
| Low | 21,363 | 13.43% |
| Moderate | 8,634 | 29.05% |
| High | 2,172 | 69.94% |
| Severe | 247 | 78.54% |

This became one of the central risk signals used throughout the project.

> These are observed associations within the project dataset and should not be interpreted as causal relationships.

---

# 🧠 Feature Engineering

Domain-relevant features were created to improve model representation and support downstream risk analytics.

### Engineered Features

| Feature | Purpose |
|---|---|
| `loan_percent_income` | Loan-to-income burden proxy |
| `age_invalid_flag` | Identifies originally invalid age observations |
| `emp_length_invalid_flag` | Identifies invalid employment values |
| `emp_length_missing_flag` | Preserves employment missingness information |
| `loan_int_rate_missing_flag` | Preserves interest-rate missingness information |
| `income_bracket` | Income segmentation |
| `loan_amount_bracket` | Loan-size segmentation |
| `debt_burden_tier` | Interpretable affordability/risk segmentation |
| `loan_grade_encoded` | Ordinal encoding of loan grade |
| `cb_person_default_on_file_encoded` | Binary credit-history indicator |
| `grade_burden_interaction` | Interaction between grade and debt burden |
| `employment_stability_flag` | Identifies short employment history |
| `credit_history_depth_tier` | Segments credit-history depth |
| One-hot encoded variables | Represents nominal categorical variables |

---

# ⚖️ Handling Class Imbalance

The target variable is moderately imbalanced.

Approximately:

```text
Non-default → 78.14%
Default     → 21.86%
```

The project therefore uses **class weighting** as the primary imbalance-handling strategy.

The current approach avoids automatically applying SMOTE because synthetic oversampling is not necessary as the default strategy for this workflow.

Threshold adjustment is also treated as a separate decision-stage consideration rather than the primary imbalance solution.

---

# 🤖 Machine Learning

The modelling stage transforms the engineered application data into a probability-of-default prediction.

### Modelling Workflow

```text
Feature-Engineered Dataset
          │
          ▼
     Train / Test Split
          │
          ▼
   Model Training
          │
          ▼
 Probability Prediction
          │
          ▼
 Risk Tier Assignment
          │
          ▼
 Decision-Support Layer
```

The project uses tree-based machine learning, with **XGBoost** used as the primary model in the current workflow.

The modelling stage is designed to support both:

- Predictive performance
- Downstream interpretability and decision support

---

# 📊 Probability of Default

The model produces a probability rather than only a binary prediction.

For example:

```text
Predicted Probability of Default
               │
               ▼
              0.72
               │
               ▼
          72% predicted
          default probability
```

This probability can then be converted into an interpretable risk tier.

---

# 🚦 Risk Tier Framework

FINRISK 360 uses a consistent risk-tier language across its downstream components.

| Predicted Default Probability | Risk Tier |
|---:|---|
| `< 20%` | 🟢 Low |
| `20% – <35%` | 🟡 Medium |
| `35% – 50%` | 🟠 High |
| `> 50%` | 🔴 Severe |

This provides a common framework for:

- Risk scoring
- Early warnings
- Affordability analysis
- Customer-facing explanations
- Portfolio segmentation

---

# 💰 Affordability Advisor

The Affordability Advisor evaluates the impact of different loan amounts relative to applicant income.

For a given annual income, the system can test multiple loan amounts:

```text
Applicant Income
       │
       ▼
Candidate Loan Amounts
       │
       ▼
Loan-to-Income Calculation
       │
       ▼
Debt-Burden Tier
       │
       ▼
Affordability Interpretation
```

### Example

A system can evaluate:

```text
Income = ₹500,000

Loan Amount
     │
     ├── ₹75,000  → Low burden
     ├── ₹100,000 → Moderate burden
     ├── ₹175,000 → High burden
     └── ₹300,000 → Severe burden
```

The exact classification is determined using the project's predefined burden thresholds.

> This is an affordability analysis based on the available loan-to-income proxy. It is not an EMI calculator because loan tenure is not available in the dataset.

---

# 🚨 Early Warning System

The Early Warning System is designed as a **point-in-time risk flagging layer**.

It is not a longitudinal delinquency-monitoring system because the dataset does not contain:

- Payment history
- Transaction history
- Delinquency timelines
- Historical account behaviour

### Trigger Logic

The EWS combines:

```text
Model-Predicted Risk
        +
Supporting Risk Drivers
        │
        ▼
Early Warning Flag
```

Supporting factors include:

- High or Severe debt-burden tier
- Loan grade D, E, F, or G
- Short employment history (<2 years)

A stronger warning is generated when elevated model risk is accompanied by one or more supporting risk drivers.

---

# 🔎 Application-Level Fraud Indicators

FINRISK 360 includes an application-level fraud-screening concept.

The current dataset does **not** contain transaction records or confirmed fraud labels.

Therefore, the project does not claim to implement a full transaction-monitoring AML system.

Instead, transparent rule-based indicators are used to identify unusual application patterns.

### Current Indicators

#### 1. High Loan-to-Income Relative to Peer Group

Applications with unusually high loan-to-income burden compared with similar loan-intent groups can receive a fraud indicator.

#### 2. Income–Employment Inconsistency

Potentially unusual combinations such as:

```text
Very High Income
        +
Very Short Employment History
```

can contribute to the indicator score.

#### 3. Risk-Factor Clustering

Multiple risk-elevating characteristics appearing together on a single application can increase the score.

#### 4. Extreme Feature Combinations

Statistically unusual combinations that were not already removed during data cleaning can be flagged for review.

### Fraud Indicator Score

Each rule contributes a point:

```text
Rule 1 → +1
Rule 2 → +1
Rule 3 → +1
Rule 4 → +1
        │
        ▼
Fraud Indicator Score
```

This is a **screening mechanism**, not proof that fraud has occurred.

---

# 🏦 Portfolio Risk Analysis

FINRISK 360 extends individual-level predictions into portfolio-level risk analysis.

The portfolio layer considers:

- Expected Loss
- Loan-grade concentration
- Loan-intent concentration
- Home-ownership concentration
- Higher-risk exposure concentration
- Interest-rate stress sensitivity

---

# 💵 Expected Loss

The portfolio framework uses:

```text
Expected Loss = PD × EAD × LGD
```

where:

- **PD** = Model-predicted probability of default
- **EAD** = Exposure at Default
- **LGD** = Loss Given Default

Because the dataset does not contain observed EAD or LGD fields, the project uses explicit assumptions/proxies rather than presenting them as directly observed values.

### EAD

`loan_amnt` is used as an **EAD proxy** for the portfolio analysis.

### LGD

LGD is treated as an **external benchmark assumption**, not as a quantity estimated from this dataset.

Therefore:

> Expected Loss results are explicitly assumption-dependent.

---

# 📊 Portfolio Concentration

The project analyses portfolio exposure across:

### Loan Grade

```text
A → B → C → D → E → F → G
```

Higher grades are examined for their contribution to overall loan volume and risk.

### Loan Intent

The portfolio is segmented by:

- Debt Consolidation
- Education
- Home Improvement
- Medical
- Personal
- Venture

### Home Ownership

The portfolio is also segmented by:

- Rent
- Mortgage
- Own
- Other

The objective is to identify whether a particular segment represents an outsized proportion of portfolio exposure or expected loss.

---

# 📉 Stress Testing

A simple model sensitivity scenario is used to evaluate portfolio behaviour under changing lending conditions.

### Scenario

```text
Interest Rate +2 Percentage Points
```

For example:

```text
10.5% → 12.5%
```

The model is then re-run to compare:

- Baseline probability of default
- Stressed probability of default
- Risk-tier distribution
- Risk-tier migration
- High/Severe risk share
- Expected-loss sensitivity

```text
Baseline Portfolio
       │
       ▼
Baseline PD Distribution
       │
       │
       │ +2 percentage-point
       │ interest-rate shock
       ▼
Stressed Portfolio
       │
       ▼
Stressed PD Distribution
       │
       ▼
Risk Migration Analysis
```

This is a **model sensitivity scenario**, not a complete macroeconomic stress-testing framework.

---

# 🔍 Explainable AI with SHAP

A credit-risk prediction is much more useful when the system can explain **why** a prediction was made.

FINRISK 360 uses **SHAP (SHapley Additive exPlanations)** for model interpretability.

```text
Application
     │
     ▼
XGBoost Model
     │
     ▼
Default Probability
     │
     ▼
SHAP Explanation
     │
     ▼
Top Risk Drivers
```

### SHAP Analysis

The explainability layer is designed to provide:

- Global feature importance
- Feature contribution analysis
- Individual customer explanations
- Low-risk explanations
- High-risk explanations
- Borderline-case explanations
- Top risk-driving factors

Example:

```text
Predicted Risk: 72%

Major contributing factors:
├── High loan-to-income burden
├── Higher loan grade risk
└── Short employment history
```

SHAP values represent **model contributions**, not causal explanations.

---

# 🧑‍💼 Risk Decision Support Vision

The long-term objective is to transform model output into a practical decision-support workflow.

```text
                    Application
                         │
                         ▼
                  Risk Prediction
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
          Risk Tier  Affordability  EWS
             │           │           │
             └───────────┼───────────┘
                         │
                         ▼
                   SHAP Explanation
                         │
                         ▼
                 Risk Decision Engine
                         │
                         ▼
               Actionable Recommendation
```

The future decision engine can combine:

- Predicted risk
- Risk tier
- Affordability
- Risk drivers
- Early-warning indicators
- Explainability
- Portfolio context

---

# 🛠️ Technology Stack

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Explainable AI

- SHAP

### Model Persistence

- Joblib

### Development Environment

- Google Colab
- Jupyter Notebook
- Git
- GitHub

### Future Application Layer

- Streamlit

---

# 📁 Current Repository Structure

The repository is intentionally lightweight during the current development stage.

```text
FINRISK-360/
│
├── README.md
├── FINRISK_ML.py
├── requirements.txt
└── .gitignore
```

As additional components are finalized, the repository will be expanded gradually rather than adding unnecessary files prematurely.

---

# 📌 Project Progress

| Component | Status |
|---|---|
| Dataset Understanding | ✅ Completed |
| Data Quality Analysis | ✅ Completed |
| Data Cleaning | ✅ Completed |
| Exploratory Data Analysis | ✅ Completed |
| Feature Engineering | ✅ Completed |
| Machine Learning Modelling | ✅ Completed |
| Risk Scoring | ✅ Completed |
| Affordability Analysis | ✅ Completed |
| Early Warning System | 🚧 Developing |
| Portfolio Risk Analysis | 🚧 Developing |
| Fraud Indicators | 🚧 Developing |
| SHAP Explainability | 🚧 Developing |
| Recovery Agent | 🔜 Planned |
| Risk Decision Engine | 🔜 Planned |
| Streamlit Interface | 🔜 Planned |

---

# ⚠️ Limitations

FINRISK 360 deliberately avoids making unsupported assumptions from unavailable data.

The current dataset does not contain:

- Loan tenure
- Repayment schedules
- Actual EMI
- Transaction history
- Transaction networks
- Confirmed fraud labels
- Observed LGD
- Observed EAD
- Longitudinal delinquency information

Therefore:

- Actual EMI-to-income is not calculated
- `loan_percent_income` is used as a loan-to-income burden proxy
- Transaction-based AML is outside the current scope
- Fraud indicators are screening signals, not fraud determinations
- Expected Loss is assumption-dependent
- EAD is represented using a project-defined proxy
- LGD is based on an external assumption
- Stress testing represents model sensitivity rather than a complete macroeconomic scenario analysis
- SHAP explanations describe model behaviour rather than causality

---

# 🔮 Future Roadmap

## Phase 1 — Risk Modelling

- Complete model evaluation
- Model calibration
- Threshold optimization
- Robust validation
- Model monitoring

## Phase 2 — Explainability

- Complete SHAP global analysis
- Individual SHAP explanations
- Automated risk-driver extraction
- Integration with Recovery Agent

## Phase 3 — Decision Engine

- Combine risk score and affordability
- Generate actionable recommendations
- Build customer-level risk summaries
- Develop bank-facing decision support

## Phase 4 — Application

- Streamlit dashboard
- Interactive loan-risk prediction
- Customer-level explanation interface
- Portfolio monitoring dashboard

## Phase 5 — Advanced Risk & AML

With suitable transaction-level data:

- Transaction monitoring
- Velocity checks
- Behavioural anomaly detection
- Network analysis
- Counterparty analysis
- Transaction-based AML detection

---

# 📚 Project Philosophy

FINRISK 360 follows three principles:

### 1. Predict

Use machine learning to estimate credit risk.

### 2. Explain

Make model outputs understandable through interpretable risk tiers and SHAP-based explanations.

### 3. Act

Convert predictions into affordability insights, early-warning signals, portfolio analysis, and actionable decision support.

```text
             PREDICT
                │
                ▼
             EXPLAIN
                │
                ▼
              ACT
```

---

# 🚧 Project Status

**Active Development**

FINRISK 360 is being developed incrementally, with each component validated before integration into the larger credit-risk decision-support workflow.

The current repository represents the ongoing development of the machine-learning core and will evolve as additional risk-analytics and explainability components are finalized.

---

# 👨‍💻 Author

## Debashis Kar

**B.Tech — Computer Science & Engineering**

### Areas of Interest

- Machine Learning
- Data Science
- Financial Risk Analytics
- Explainable AI
- Applied Machine Learning
- Decision Support Systems

---

## ⭐ FINRISK 360

> **Predict Risk. Explain Risk. Act on Risk.**
