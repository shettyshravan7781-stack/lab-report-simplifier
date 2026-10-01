# Lab Report Simplifier — ML Core Pipeline

An end-to-end Machine Learning pipeline that processes medical laboratory panel data to deliver explainable multi-organ risk predictions (Anemia, Diabetes, Kidney, Lipid, Liver, and Overall Health).

## 📌 Features

- **Data Processing:** Automated feature engineering and missingness flagging on raw lab values.
- **Risk Modeling:** Multi-target XGBoost ensemble classifiers trained and benchmarked against standard baselines.
- **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) values and visual confusion matrices for clinical interpretability.

## 📁 Project Structure

```text
├── output/               # Model evaluation outputs
│   ├── eda_plots/        # Correlation heatmaps and distributions
│   ├── model_plots/      # Confusion matrices
│   ├── saved_models/     # Trained XGBoost models (.pkl)
│   └── shap_plots/       # SHAP feature importance charts
├── *.py                  # Core pipeline scripts (merge, feature eng, train, explain)
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
```
