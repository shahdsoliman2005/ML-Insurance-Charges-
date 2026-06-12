# ML-Insurance-Charges-
# 📊 Insurance Charges — Regression Models

> Predict medical insurance charges using a wide range of regression algorithms and ensemble methods, with thorough exploratory data analysis and a structured model comparison.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models Covered](#models-covered)
- [Results](#results)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Next Steps](#next-steps)

---

## Overview

This notebook walks through the full machine learning pipeline — from raw data exploration to training and comparing multiple regression models — on a real-world insurance dataset. The goal is to accurately predict individual medical charges based on demographic and health features.

Key highlights:
- In-depth **Exploratory Data Analysis (EDA)** with visualisations
- Clean **preprocessing pipeline** (imputation, encoding, scaling)
- Side-by-side comparison of **base models** and **ensemble methods**
- Evaluation using MAE, MSE, RMSE, and R² on both train and test sets

---

## Dataset

**File:** `insurance.csv`

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the insured person |
| `sex` | Categorical | Gender (`male` / `female`) |
| `bmi` | Numerical | Body Mass Index |
| `children` | Numerical | Number of dependents |
| `smoker` | Categorical | Smoking status (`yes` / `no`) |
| `region` | Categorical | US residential region |
| `charges` | **Target** | Individual medical costs (USD) |

> ⚠️ The target variable (`charges`) is right-skewed. Smoker status is the single strongest predictor of high charges.

---

## Project Structure

```
├── Regression_Models_Enhanced.ipynb   # Main notebook
├── insurance.csv                      # Dataset
└── README.md
```

---

## Models Covered

### Base Models
| Model | Notes |
|---|---|
| Linear Regression | Baseline linear approach |
| Lasso / Ridge | Regularised linear models |
| K-Nearest Neighbours (KNN) | `k=5`, Manhattan distance |
| Support Vector Regression (SVR) | Linear kernel, `C=100` |
| Decision Tree | `max_depth=10`, squared error criterion |

### Ensemble Methods
| Model | Notes |
|---|---|
| Random Forest | Bagging of decision trees |
| Extra Trees | Extremely randomised trees |
| Bagging Regressor | Bootstrap aggregation |
| AdaBoost | Adaptive boosting |
| Gradient Boosting | Sequential boosting |
| HistGradient Boosting | Fast histogram-based boosting |
| Voting (Simple) | Equal-weight average of LR, KNN, DT |
| Voting (Weighted) | LR ×2, KNN ×1, DT ×2 |
| Stacking | Meta-learner on top of base models |

---

## Results

| Takeaway | Detail |
|---|---|
| **Best predictor** | `smoker` status dominates — largest correlation with charges |
| **Top model** | Decision Tree (depth=10) and Voting ensembles generally outperform linear models |
| **Overfitting watch** | Decision Tree shows a large Train / Test R² gap — consider pruning or Random Forest |

Models are ranked by **R² Test Score** in the final comparison table and bar charts.

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost tqdm
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook Regression_Models_Enhanced.ipynb
   ```

---

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- catboost
- tqdm

---

## Next Steps

- [ ] Apply **log-transform** on the target (`charges`) to reduce skewness
- [ ] Tune hyperparameters with **GridSearchCV** or **Optuna**
- [ ] Add **XGBoost / LightGBM / CatBoost** models
- [ ] Use **cross-validation** instead of a single train/test split
- [ ] Build a simple **prediction interface** (e.g. Streamlit app)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
