import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from xgboost import XGBClassifier

def run_step3_model_training():
    print("==============================================================")
    print("   Starting Step 3: Model Training & Performance Evaluation   ")
    print("==============================================================\n")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "model_plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    dataset_path = os.path.join(OUTPUT_DIR, "ML_Ready_Hospital_Dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(BASE_DIR, "ML_Ready_Hospital_Dataset.csv")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)

    # Define Feature Matrix (X) and Target Variables (y)
    ignore_cols = ['MedID', 'LabReference', 'Sample ID', 'Collected', 'Time', 'Reported Time', 
                   'Source_File', 'First Name', 'Last Name', 'Gender', 'Diet', 'Patient_Explanation_Summary']
    target_cols = [c for c in df.columns if c.startswith('Target_')]

    feature_cols = [c for c in df.columns if c not in ignore_cols and c not in target_cols]

    # Convert features to numeric
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    print(f"Feature Space Dimensions: {X.shape[1]} clinical biomarkers")
    print(f"Target Tasks ({len(target_cols)}): {target_cols}\n")

    results = []

    for target in target_cols:
        print(f"--------------------------------------------------------------")
        print(f" Training Models for Task: {target}")
        print(f"--------------------------------------------------------------")
        
        y = df[target].values
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scaler for Neural Net
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Configure XGBoost dynamically for Binary vs Multi-class tasks
        if num_classes == 2:
            xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
        else:
            xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": xgb_model,
            "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        }

        for name, model in models.items():
            if name == "MLP Neural Net":
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')

            print(f"  [{name:15s}] Accuracy: {acc*100:.2f}% | F1-Score: {f1*100:.2f}%")

            results.append({
                'Target Task': target,
                'Model Algorithm': name,
                'Accuracy (%)': round(acc * 100, 2),
                'Precision (%)': round(prec * 100, 2),
                'Recall (%)': round(rec * 100, 2),
                'F1-Score (%)': round(f1 * 100, 2)
            })

            # Plot Confusion Matrix for XGBoost
            if name == "XGBoost":
                cm = confusion_matrix(y_test, preds)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title(f'XGBoost Confusion Matrix: {target}')
                plt.xlabel('Predicted Class')
                plt.ylabel('True Class')
                plt.tight_layout()
                plt.savefig(os.path.join(PLOTS_DIR, f'cm_{target}.png'), dpi=300)
                plt.close()

    # Save benchmark results to Excel
    results_df = pd.DataFrame(results)
    excel_path = os.path.join(OUTPUT_DIR, "Model_Benchmarking_Results.xlsx")
    results_df.to_excel(excel_path, index=False)

    print("\n==============================================================")
    print(f"   Step 3 Complete! Results saved to '{excel_path}'")
    print("==============================================================")

if __name__ == "__main__":
    run_step3_model_training()