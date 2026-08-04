import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def save_xgboost_models():
    print("==============================================================")
    print("      Saving Trained XGBoost Models & Generating Feature List  ")
    print("==============================================================\n")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "saved_models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    dataset_path = os.path.join(OUTPUT_DIR, "ML_Ready_Hospital_Dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(BASE_DIR, "ML_Ready_Hospital_Dataset.csv")

    df = pd.read_csv(dataset_path, low_memory=False)

    # Exclude non-feature columns
    ignore_cols = ['MedID', 'LabReference', 'Sample ID', 'Collected', 'Time', 'Reported Time', 
                   'Source_File', 'First Name', 'Last Name', 'Gender', 'Diet', 'Patient_Explanation_Summary']
    target_cols = [c for c in df.columns if c.startswith('Target_')]
    feature_cols = [c for c in df.columns if c not in ignore_cols and c not in target_cols]

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Save each XGBoost target model
    for target in target_cols:
        print(f"Training and saving model for: {target}...")
        y = df[target].values
        num_classes = len(np.unique(y))

        if num_classes == 2:
            model = XGBClassifier(eval_metric='logloss', random_state=42)
        else:
            model = XGBClassifier(eval_metric='mlogloss', random_state=42)

        model.fit(X, y)

        model_filename = f"xgboost_{target}.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"  ✓ Saved model: {model_path}")

    # Save feature names list to JSON for OCR teammate
    import json
    feature_list_path = os.path.join(OUTPUT_DIR, "feature_names.json")
    with open(feature_list_path, "w") as f:
        json.dump(feature_cols, f, indent=4)

    print("\n==============================================================")
    print("  SUCCESS! Models saved in 'output/saved_models/'")
    print(f"  Feature list saved in: {feature_list_path}")
    print("==============================================================")

if __name__ == "__main__":
    save_xgboost_models()