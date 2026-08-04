import os
import pandas as pd
import numpy as np

def run_feature_engineering():
    print("==============================================================")
    print("   Starting Step 2: Feature Engineering & Clinical Labeling   ")
    print("==============================================================\n")
    
    # 1. Paths Setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    dataset_path = os.path.join(OUTPUT_DIR, "Master_Hospital_Dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(BASE_DIR, "Master_Hospital_Dataset.csv")
        
    print(f"Loading master dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"Dataset loaded successfully. Shape: {df.shape}\n")

    # Ensure numerical conversion for core diagnostic markers
    core_cols = [
        'HbA1c %', 'Fasting Glucose (mg/dL)', 'Postprandial Glucose (mg/dL)',
        'Creatinine (mg/dL)', 'eGFR (mL/min/1.73m²)', 'Urea (mg/dL)', 'BUN (mg/dL)',
        'ALT/SGPT (U/L)', 'AST/SGOT (U/L)', 'Bilirubin Total (mg/dL)',
        'Total Cholesterol (mg/dL)', 'Triglycerides (mg/dL)', 'LDL (mg/dL)', 'HDL (mg/dL)',
        'Hemoglobin (g/dL)', 'RBC Count (mil/µL)', 'Albumin (g/dL)', 'Globulin (g/dL)'
    ]
    for col in core_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # -----------------------------------------------------------------
    # A. ENGINEER CLINICAL DIAGNOSTIC RATIOS
    # -----------------------------------------------------------------
    print("1. Computing Derived Biomarker Ratios...")

    if 'BUN (mg/dL)' in df.columns and 'Creatinine (mg/dL)' in df.columns:
        df['Ratio_BUN_Creatinine'] = (df['BUN (mg/dL)'] / df['Creatinine (mg/dL)'].replace(0, np.nan)).round(2)

    if 'Total Cholesterol (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
        df['Ratio_Cholesterol_HDL'] = (df['Total Cholesterol (mg/dL)'] / df['HDL (mg/dL)'].replace(0, np.nan)).round(2)

    if 'LDL (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
        df['Ratio_LDL_HDL'] = (df['LDL (mg/dL)'] / df['HDL (mg/dL)'].replace(0, np.nan)).round(2)

    if 'Albumin (g/dL)' in df.columns and 'Globulin (g/dL)' in df.columns:
        df['Ratio_Albumin_Globulin'] = (df['Albumin (g/dL)'] / df['Globulin (g/dL)'].replace(0, np.nan)).round(2)

    print("   ✓ 4 diagnostic biomarker ratios calculated.")

    # -----------------------------------------------------------------
    # B. GENERATE TARGET RISK LABELS (CLINICAL GUIDELINES)
    # -----------------------------------------------------------------
    print("\n2. Generating Target Labels according to International Medical Guidelines...")

    # 1. Diabetes Risk (ADA Guidelines)
    def calc_diabetes_risk(row):
        hba1c = row.get('HbA1c %', np.nan)
        fbs = row.get('Fasting Glucose (mg/dL)', np.nan)
        if pd.isna(hba1c) or pd.isna(fbs): return 0
        if hba1c >= 6.5 or fbs >= 126:
            return 2  # Diabetes
        elif hba1c >= 5.7 or fbs >= 100:
            return 1  # Prediabetes
        else:
            return 0  # Normal

    # 2. Kidney Risk (KDIGO Guidelines)
    def calc_kidney_risk(row):
        egfr = row.get('eGFR (mL/min/1.73m²)', np.nan)
        creat = row.get('Creatinine (mg/dL)', np.nan)
        if pd.isna(egfr) or pd.isna(creat): return 0
        if egfr < 60 or creat > 1.5:
            return 2  # High Risk (Stage 3+ CKD)
        elif egfr < 90 or creat > 1.2:
            return 1  # Mild Risk (Stage 1-2 CKD)
        else:
            return 0  # Normal

    # 3. Liver Risk (Elevated Enzymes)
    def calc_liver_risk(row):
        alt = row.get('ALT/SGPT (U/L)', np.nan)
        ast = row.get('AST/SGOT (U/L)', np.nan)
        bili = row.get('Bilirubin Total (mg/dL)', np.nan)
        if pd.isna(alt) or pd.isna(ast) or pd.isna(bili): return 0
        if alt > 45 or ast > 40 or bili > 1.2:
            return 1  # Elevated Enzyme Risk
        else:
            return 0  # Normal

    # 4. Lipid / Cardiovascular Risk (NCEP ATP III)
    def calc_lipid_risk(row):
        tc = row.get('Total Cholesterol (mg/dL)', np.nan)
        tg = row.get('Triglycerides (mg/dL)', np.nan)
        ldl = row.get('LDL (mg/dL)', np.nan)
        hdl = row.get('HDL (mg/dL)', np.nan)
        if pd.isna(tc) or pd.isna(tg) or pd.isna(ldl) or pd.isna(hdl): return 0
        if tc >= 240 or tg >= 200 or ldl >= 160 or hdl < 40:
            return 2  # High Risk
        elif tc >= 200 or tg >= 150 or ldl >= 130:
            return 1  # Borderline High
        else:
            return 0  # Desirable

    # 5. Anemia Risk (WHO Criteria)
    def calc_anemia_risk(row):
        hb = row.get('Hemoglobin (g/dL)', np.nan)
        rbc = row.get('RBC Count (mil/µL)', np.nan)
        gender = str(row.get('Gender', '')).strip()
        if pd.isna(hb) or pd.isna(rbc): return 0
        cutoff = 13.0 if gender == 'Male' else 12.0
        if hb < 10.0 or rbc < 3.5:
            return 2  # Moderate/Severe Anemia
        elif hb < cutoff or rbc < 4.2:
            return 1  # Mild Anemia
        else:
            return 0  # Normal

    # Apply row-by-row labeling
    df['Target_Diabetes_Risk'] = df.apply(calc_diabetes_risk, axis=1)
    df['Target_Kidney_Risk'] = df.apply(calc_kidney_risk, axis=1)
    df['Target_Liver_Risk'] = df.apply(calc_liver_risk, axis=1)
    df['Target_Lipid_Risk'] = df.apply(calc_lipid_risk, axis=1)
    df['Target_Anemia_Risk'] = df.apply(calc_anemia_risk, axis=1)

    # 6. Overall Composite Risk Index
    def calc_overall_risk(row):
        score = (row['Target_Diabetes_Risk'] + 
                 row['Target_Kidney_Risk'] + 
                 row['Target_Liver_Risk'] + 
                 row['Target_Lipid_Risk'] + 
                 row['Target_Anemia_Risk'])
        if score >= 4:
            return 2  # High Multi-System Risk
        elif score >= 2:
            return 1  # Moderate Risk
        else:
            return 0  # Low Risk

    df['Target_Overall_Health_Risk'] = df.apply(calc_overall_risk, axis=1)
    print("   ✓ 6 Target Risk Labels generated successfully.")

    # -----------------------------------------------------------------
    # C. SAVE ML-READY DATASET & EXPORT DISTRIBUTION SUMMARY
    # -----------------------------------------------------------------
    print("\n3. Saving ML-Ready Dataset & Distribution Report...")
    
    # Save updated CSV
    ml_csv_path = os.path.join(OUTPUT_DIR, "ML_Ready_Hospital_Dataset.csv")
    df.to_csv(ml_csv_path, index=False)
    print(f"   ✓ ML-Ready Dataset saved to: '{ml_csv_path}' (Shape: {df.shape})")

    # Generate Target Distribution Summary Table for Paper
    target_summary = []
    target_cols = [
        ('Target_Diabetes_Risk', {0: 'Normal', 1: 'Prediabetes', 2: 'Diabetes'}),
        ('Target_Kidney_Risk', {0: 'Normal', 1: 'Mild Risk (Stage 1-2)', 2: 'High Risk (Stage 3+)'}),
        ('Target_Liver_Risk', {0: 'Normal', 1: 'Elevated Enzymes'}),
        ('Target_Lipid_Risk', {0: 'Desirable', 1: 'Borderline High', 2: 'High Risk'}),
        ('Target_Anemia_Risk', {0: 'Normal', 1: 'Mild Anemia', 2: 'Moderate/Severe Anemia'}),
        ('Target_Overall_Health_Risk', {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'})
    ]

    for col_name, mapping in target_cols:
        counts = df[col_name].value_counts().to_dict()
        for code, label in mapping.items():
            cnt = counts.get(code, 0)
            pct = round((cnt / len(df)) * 100, 2)
            target_summary.append({
                'Target Variable': col_name,
                'Class Code': code,
                'Class Label': label,
                'Patient Sample Count': cnt,
                'Percentage (%)': pct
            })

    summary_df = pd.DataFrame(target_summary)
    
    excel_report_path = os.path.join(OUTPUT_DIR, "Target_Labels_Distribution_Report.xlsx")
    with pd.ExcelWriter(excel_report_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Target Distributions', index=False)
        
    print(f"   ✓ Target Labels Report saved to: '{excel_report_path}'")

    print("\n==============================================================")
    print("   Step 2 Complete! Feature Engineering Finished Successfully.")
    print("==============================================================")

if __name__ == "__main__":
    run_feature_engineering()