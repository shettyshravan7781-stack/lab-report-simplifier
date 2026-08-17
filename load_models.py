import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import subprocess

# =========================================================
# AUTOMATION STEP: Run OCR on newest image before loading model
# =========================================================
print("Running OCR on newest lab report...")
try:
    subprocess.run(["python", "OCR/report_parser.py"], check=True)
except Exception as e:
    print(f"Warning: OCR script execution failed: {e}")
# =========================================================

# 1. Folder Setup
BASE_DIR = r"C:\AI_Lab_Report"
MODELS_DIR = os.path.join(BASE_DIR, "output", "saved_models")
JSON_PATH = os.path.join(BASE_DIR, "OCR", "output", "report_output.json")
SHAP_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "shap_plots")
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# 2. Read OCR Data
with open(JSON_PATH, "r") as f:
    ocr_data = json.load(f)

extracted_tests = {}
abnormal_findings = []

tests_list = ocr_data.get("tests", []) or ocr_data.get("lab_tests", [])

for item in tests_list:
    val_str = str(item.get("value", "")).strip()
    try:
        val_float = float(val_str)
        test_name = item.get("test") or item.get("test_name", "")
        if test_name:
            extracted_tests[test_name] = val_float
            if item.get("flag") in ["High", "Low"]:
                abnormal_findings.append({
                    "test": test_name, 
                    "val": val_float, 
                    "unit": item.get('unit', ''), 
                    "flag": item.get('flag')
                })
    except ValueError:
        continue

# 3. Comprehensive Mapping for ALL 10 Lab Report Panels
mapping = {
    # 1. Complete Blood Count (CBC)
    "Haemoglobin": "Hemoglobin (g/dL)", "Hemoglobin": "Hemoglobin (g/dL)",
    "RBC Count": "RBC Count (mil/µL)", "Hematocrit": "Hematocrit %", "Hct": "Hematocrit %",
    "MCV": "MCV (fL)", "MCH": "MCH (pg)", "MCHC": "MCHC (g/dL)",
    "RDW-CV": "RDW-CV %", "RDW-SD": "RDW-SD (fL)",
    "Total Leucocyte Count": "WBC (cells/µL)", "WBC": "WBC (cells/µL)",
    "Neutrophils": "Neutrophils %", "Lymphocytes": "Lymphocytes %",
    "Eosinophils": "Eosinophils %", "Monocytes": "Monocytes %", "Basophils": "Basophils %",
    "Platelet Count": "Platelet Count (×10^3/µL)", "MPV": "MPV (fL)",

    # 2. Liver Function Test (LFT)
    "Total Protein": "Protein Total (g/dL)", "Albumin": "Albumin (g/dL)",
    "Globulin": "Globulin (g/dL)", "Albumin/Globulin Ratio": "Ratio_Albumin_Globulin",
    "A/G Ratio": "A/G Ratio", "Total Bilirubin": "Bilirubin Total (mg/dL)",
    "Direct Bilirubin": "Bilirubin Direct (mg/dL)", "Indirect Bilirubin": "Bilirubin Indirect (mg/dL)",
    "SGOT (AST)": "AST/SGOT (U/L)", "AST": "AST/SGOT (U/L)",
    "SGPT (ALT)": "ALT/SGPT (U/L)", "ALT": "ALT/SGPT (U/L)",
    "Alkaline Phosphatase (ALP)": "ALP (U/L)", "ALP": "ALP (U/L)",
    "Gamma Glutamyl Transferase (GGT)": "GGT (U/L)", "GGT": "GGT (U/L)",

    # 3. Kidney Function Test (KFT / Renal)
    "Serum Creatinine": "Creatinine (mg/dL)", "Creatinine": "Creatinine (mg/dL)",
    "Blood Urea": "Urea (mg/dL)", "Urea": "Urea (mg/dL)", "BUN": "BUN (mg/dL)",
    "BUN/Creatinine Ratio": "BUN/Creatinine Ratio", "Uric Acid": "Uric Acid (mg/dL)",
    "eGFR": "eGFR (mL/min/1.73m²)",

    # 4. Lipid Profile
    "Total Cholesterol": "Total Cholesterol (mg/dL)", "HDL Cholesterol": "HDL (mg/dL)",
    "LDL Cholesterol": "LDL (mg/dL)", "VLDL": "VLDL (mg/dL)",
    "Triglycerides": "Triglycerides (mg/dL)", "Non-HDL": "Non-HDL (mg/dL)",

    # 5. Diabetes Profile
    "Fasting Blood Sugar": "FBS (mg/dL)", "FBS": "FBS (mg/dL)",
    "Postprandial Blood Sugar": "PLBS (mg/dL)", "PLBS": "PLBS (mg/dL)",
    "HbA1c": "HbA1c %", "Estimated Average Glucose": "Estimated Avg Glucose (mg/dL)",

    # 6. Thyroid Profile
    "TSH": "TSH (µIU/mL)", "Total T3": "TT3 (ng/dL)", "TT3": "TT3 (ng/dL)",
    "Total T4": "TT4 (µg/dL)", "TT4": "TT4 (µg/dL)",

    # 7. Electrolytes & Bone Profile
    "Sodium": "Sodium (mmol/L)", "Potassium": "Potassium (mmol/L)",
    "Chloride": "Chloride (mmol/L)", "Calcium": "Calcium (mg/dL)",
    "Phosphorus": "Phosphorus (mg/dL)",

    # 8. Iron Profile
    "Serum Iron": "Iron (µg/dL)", "Iron": "Iron (µg/dL)",
    "TIBC": "TIBC (µg/dL)", "UIBC": "UIBC (µg/dL)",
    "Transferrin Saturation": "Transferrin Saturation %",

    # 9. Urine Microalbumin Panel
    "Urine Albumin": "Urine Albumin (mg/L)", "Urine Creatinine": "Urine Creatinine (mg/dL)",
    "Albumin/Creatinine Ratio": "Albumin/Creatinine Ratio",

    # 10. Urinalysis Parameters
    "Specific Gravity": "Specific Gravity", "pH": "pH"
}

# Auto-detect panel type based on present tests
raw_panel = str(ocr_data.get('report_type', 'N/A')).upper()

def detect_panel(tests_keys, raw_type):
    joined = " ".join(tests_keys).upper() + " " + raw_type
    if "HBA1C" in joined or "GLUCOSE" in joined or "FBS" in joined: return "Diabetes Profile"
    if "CREATININE" in joined or "UREA" in joined or "BUN" in joined: return "Kidney Function Test (KFT)"
    if "CHOLESTEROL" in joined or "TRIGLYCERIDES" in joined or "HDL" in joined: return "Lipid Profile"
    if "BILIRUBIN" in joined or "ALT" in joined or "AST" in joined or "SGOT" in joined: return "Liver Function Test (LFT)"
    if "TSH" in joined or "TT3" in joined or "TT4" in joined: return "Thyroid Profile"
    if "IRON" in joined or "TIBC" in joined: return "Iron Profile"
    if "URINE" in joined or "SPECIFIC GRAVITY" in joined: return "Urinalysis"
    if "HEMOGLOBIN" in joined or "WBC" in joined or "RBC" in joined or "PLATELET" in joined: return "Complete Blood Count (CBC)"
    return "General Clinical Panel"

panel_type = detect_panel(list(extracted_tests.keys()), raw_panel)

# Select SHAP target dynamically for the report type
SHAP_TARGET_MAP = {
    "Diabetes Profile": "Target_Diabetes_Risk",
    "Kidney Function Test (KFT)": "Target_Kidney_Risk",
    "Lipid Profile": "Target_Lipid_Risk",
    "Liver Function Test (LFT)": "Target_Liver_Risk",
    "Complete Blood Count (CBC)": "Target_Anemia_Risk",
    "Iron Profile": "Target_Anemia_Risk"
}
target_for_shap = SHAP_TARGET_MAP.get(panel_type, "Target_Overall_Health_Risk")

print("\n" + "=" * 65)
print("     AI CLINICAL DIAGNOSTIC PIPELINE — INTEGRATED AUDIT")
print("=" * 65)
print(f"Patient Name : {ocr_data.get('patient', {}).get('name', 'N/A')}")
print(f"Age / Gender : {ocr_data.get('patient', {}).get('age', 'N/A')} / {ocr_data.get('patient', {}).get('gender', 'N/A')}")
print(f"Panel Type   : {panel_type}")
print("-" * 65)

print("\n[SECTION 1: ORGAN RISK PROBABILITIES]")

for pkl_file in sorted(os.listdir(MODELS_DIR)):
    if pkl_file.endswith(".pkl"):
        target_name = pkl_file.replace("xgboost_", "").replace(".pkl", "")
        model_path = os.path.join(MODELS_DIR, pkl_file)
        model = joblib.load(model_path)

        feature_dict = {feat: np.nan for feat in model.feature_names_in_}
        
        for ocr_k, model_k in mapping.items():
            if ocr_k in extracted_tests and model_k in feature_dict:
                feature_dict[model_k] = extracted_tests[ocr_k]
                
        if "patient" in ocr_data and "age" in ocr_data["patient"] and "Age" in feature_dict:
            raw_age = ocr_data["patient"]["age"]
            try:
                clean_age = float(''.join(c for c in str(raw_age) if c.isdigit() or c == '.'))
                feature_dict["Age"] = clean_age
            except ValueError:
                feature_dict["Age"] = 0.0

        df_in = pd.DataFrame([feature_dict])
        df_in = df_in.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        prob = model.predict_proba(df_in)[0][1] if hasattr(model, "predict_proba") else model.predict(df_in)[0]
        
        # Clinical heuristic rule flags
        if target_name == "Target_Liver_Risk" and any(item["test"] in ["Gamma Glutamyl Transferase (GGT)", "Alkaline Phosphatase (ALP)"] for item in abnormal_findings):
            prob = max(prob, 0.85)
        elif target_name == "Target_Anemia_Risk" and any("Hb" in item["test"] or "Hemoglobin" in item["test"] for item in abnormal_findings):
            prob = max(prob, 0.85)

        risk_pct = prob * 100
        status = "HIGH RISK ⚠️" if risk_pct > 50 else "LOW RISK  ✅"
        print(f" • {target_name:28s}: {risk_pct:6.2f}%  [{status}]")

        # Generate SHAP explanation plot dynamically for detected target
        if target_name == target_for_shap or (target_for_shap not in [f.replace("xgboost_", "").replace(".pkl", "") for f in os.listdir(MODELS_DIR)] and target_name == "Target_Overall_Health_Risk"):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer(df_in)
            plt.figure(figsize=(10, 4))
            if len(shap_vals.shape) == 3:
                shap.plots.bar(shap_vals[0, :, 1], max_display=8, show=False)
            else:
                shap.plots.bar(shap_vals[0], max_display=8, show=False)
            plt.title(f"SHAP Biomarker Attribution: {panel_type} Risk Profile", fontsize=11)
            plt.tight_layout()
            shap_path = os.path.join(SHAP_OUTPUT_DIR, "patient_risk_shap_bar.png")
            plt.savefig(shap_path, dpi=300)
            plt.close()

print("\n[SECTION 2: GENERATED VISUAL & AUDIT ARTIFACTS]")
print(f" ✓ SHAP Plot Saved         : {os.path.join(SHAP_OUTPUT_DIR, 'patient_risk_shap_bar.png')}")
print(f" ✓ Model Benchmarks Report: {os.path.join(BASE_DIR, 'output', 'Model_Benchmarking_Results.xlsx')}")

print("\n[SECTION 3: CLINICAL SUMMARY]")
if abnormal_findings:
    print(" • Detected Biomarker Abnormalities:")
    for item in abnormal_findings:
        print(f"   - {item['test']}: {item['val']} {item['unit']} ({item['flag']})")
    
    print("\n • Diagnostic Assessment:")
    print(f"   Abnormal findings identified in {panel_type}.")
    print("   Recommendation: Medical consultation for panel-specific evaluation.")
else:
    print(" • No out-of-range biomarkers detected.")
    print("\n • Diagnostic Assessment:")
    print("   All measured parameters are within standard biological reference ranges.")
print("=" * 65)