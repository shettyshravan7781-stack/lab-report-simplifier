import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def run_step4_shap_analysis():
    print("==============================================================")
    print("   Starting Step 4: SHAP Explainability & Feature Importance  ")
    print("==============================================================\n")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    SHAP_PLOTS_DIR = os.path.join(OUTPUT_DIR, "shap_plots")
    os.makedirs(SHAP_PLOTS_DIR, exist_ok=True)

    dataset_path = os.path.join(OUTPUT_DIR, "ML_Ready_Hospital_Dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(BASE_DIR, "ML_Ready_Hospital_Dataset.csv")

    df = pd.read_csv(dataset_path, low_memory=False)

    ignore_cols = ['MedID', 'LabReference', 'Sample ID', 'Collected', 'Time', 'Reported Time', 
                   'Source_File', 'First Name', 'Last Name', 'Gender', 'Diet', 'Patient_Explanation_Summary']
    target_cols = [c for c in df.columns if c.startswith('Target_')]
    feature_cols = [c for c in df.columns if c not in ignore_cols and c not in target_cols]

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Perform SHAP analysis for key multi-disease targets
    for target in target_cols:
        print(f"Generating SHAP Summary Plot for: {target}...")
        y = df[target].values
        num_classes = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if num_classes == 2:
            model = XGBClassifier(eval_metric='logloss', random_state=42)
        else:
            model = XGBClassifier(eval_metric='mlogloss', random_state=42)

        model.fit(X_train, y_train)

        # Compute SHAP Values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        # Plot SHAP Summary
        plt.figure(figsize=(10, 6))
        
        # If multi-class, shap_values has shape (N, Features, Classes)
        if len(shap_values.shape) == 3:
            shap.summary_plot(shap_values[:, :, 1], X_test, show=False, max_display=12)
        else:
            shap.summary_plot(shap_values, X_test, show=False, max_display=12)

        plt.title(f"SHAP Biomarker Importance: {target}", fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(SHAP_PLOTS_DIR, f"shap_{target}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" ✓ Saved plot to: {save_path}")

    print("\n==============================================================")
    print("   Step 4 Complete! All SHAP plots saved in 'output/shap_plots/' ")
    print("==============================================================")

if __name__ == "__main__":
    run_step4_shap_analysis()