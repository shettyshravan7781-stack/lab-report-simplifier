import os
import glob
import pandas as pd
import numpy as np

def merge_clean_and_impute_pipeline():
    print("==============================================================")
    print("Starting Hospital Data Merging, Cleaning & Imputation Pipeline")
    print("==============================================================\n")
    
    # 1. Define Folder Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
    MONTHLY_DIR = os.path.join(DATASET_DIR, "Monthly_Files")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Merge 12 Monthly Files
    monthly_files = sorted(glob.glob(os.path.join(MONTHLY_DIR, "*Values.csv")))
    print(f"Found {len(monthly_files)} monthly files in '{MONTHLY_DIR}'")
    
    monthly_dfs = []
    for f in monthly_files:
        filename = os.path.basename(f)
        raw_df = pd.read_csv(f, header=None, low_memory=False)
        
        col_names = raw_df.iloc[1].tolist()
        for idx, col in enumerate(col_names):
            if pd.isna(col) or str(col).strip() in ['Reported', '']:
                if idx == 4:
                    col_names[idx] = 'Time'
                elif idx == 5:
                    col_names[idx] = 'Reported Time'
                else:
                    col_names[idx] = f'Unnamed_{idx}'
            else:
                col_names[idx] = str(col).strip()
        
        df_data = raw_df.iloc[2:].copy()
        df_data.columns = col_names
        df_data['Source_File'] = filename
        monthly_dfs.append(df_data)
    
    combined_df = pd.concat(monthly_dfs, ignore_index=True)
    initial_rows = len(combined_df)
    
    # Clean redundant headers & exact row duplicates
    combined_df = combined_df[combined_df['MedID'] != 'MedID']
    num_duplicates = len(combined_df[combined_df.duplicated()])
    combined_df = combined_df.drop_duplicates()
    final_unique_rows = len(combined_df)
    
    # Standardize missing value symbols
    missing_placeholders = ['NaN', 'nan', 'N/A', 'null', 'NULL', 'None', '', ' ', 'Nil', '-', '–']
    combined_df = combined_df.replace(missing_placeholders, np.nan)
    
    # Normalize MedID as string
    combined_df['MedID'] = combined_df['MedID'].astype(str).str.strip()
    
    # 3. Merge SubDetails.csv (Demographics)
    sub_details_path = os.path.join(DATASET_DIR, "SubDetails.csv")
    if os.path.exists(sub_details_path):
        sub_df = pd.read_csv(sub_details_path)
        if 'Zender' in sub_df.columns:
            sub_df = sub_df.rename(columns={'Zender': 'Gender'})
        sub_df['MedID'] = sub_df['MedID'].astype(str).str.strip()
        sub_clean = sub_df.drop(columns=['LabReference'], errors='ignore').drop_duplicates(subset=['MedID'])
        master_df = pd.merge(combined_df, sub_clean, on='MedID', how='left')
        print("✓ Successfully merged metadata from SubDetails.csv")
    else:
        master_df = combined_df

    # 4. Attempt Merging ESR & Vitamin D Files
    extra_files = [
        "ESR_Report_Dataset_1000_Enhanced.csv",
        "Vitamin_D_Profile_Enhanced.csv"
    ]
    possible_id_cols = ['MedID', 'Patient_ID', 'PatientID', 'ID', 'Sample ID', 'Subject_ID']

    for extra_file in extra_files:
        extra_path = os.path.join(DATASET_DIR, extra_file)
        if os.path.exists(extra_path):
            try:
                extra_df = pd.read_csv(extra_path, low_memory=False)
                id_col = next((col for col in possible_id_cols if col in extra_df.columns), None)
                if id_col:
                    extra_df = extra_df.rename(columns={id_col: 'MedID'})
                    extra_df['MedID'] = extra_df['MedID'].astype(str).str.strip()
                    cols_to_add = [c for c in extra_df.columns if c not in master_df.columns or c == 'MedID']
                    if len(cols_to_add) > 1:
                        extra_sub = extra_df[cols_to_add].drop_duplicates(subset=['MedID'])
                        master_df = pd.merge(master_df, extra_sub, on='MedID', how='left')
                        print(f"✓ Merged columns from '{extra_file}'")
            except Exception as e:
                print(f"❌ Error merging '{extra_file}': {e}")

    print(f"\nPre-Imputation Master Dataset Shape: {master_df.shape}")

    # 5. AUTOMATED MISSING VALUE IMPUTATION
    print("\n--- Executing Intelligent Missing Value Imputation ---")
    
    # A. Drop columns that are 100% missing (e.g. completely un-matched research profiles)
    all_null_cols = [col for col in master_df.columns if master_df[col].isnull().sum() == len(master_df)]
    if all_null_cols:
        master_df = master_df.drop(columns=all_null_cols)
        print(f"• Dropped {len(all_null_cols)} columns with 100% missing entries (unmatched keys).")

    # B. Specific Clinical Rules Imputation
    if 'Crystals' in master_df.columns:
        master_df['Crystals'] = master_df['Crystals'].fillna('Absent')
        print("• Imputed 'Crystals' missing values -> 'Absent'")
        
    if 'Casts' in master_df.columns:
        master_df['Casts'] = master_df['Casts'].fillna('Absent')
        print("• Imputed 'Casts' missing values -> 'Absent'")
        
    if 'Others' in master_df.columns:
        master_df['Others'] = master_df['Others'].fillna('NIL')
        print("• Imputed 'Others' missing values -> 'NIL'")

    # C. Numeric Columns Imputation (Median Strategy)
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude ID/Key columns from numeric imputation if any
    numeric_cols = [c for c in numeric_cols if c not in ['MedID', 'Sample ID']]
    
    num_imputed_count = 0
    for col in numeric_cols:
        if master_df[col].isnull().sum() > 0:
            median_val = master_df[col].median()
            master_df[col] = master_df[col].fillna(median_val)
            num_imputed_count += 1
            
    print(f"• Imputed {num_imputed_count} numeric features using Median Strategy.")

    # D. Categorical/Object Columns Imputation (Mode / 'Unknown' Strategy)
    categorical_cols = master_df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_imputed_count = 0
    for col in categorical_cols:
        if master_df[col].isnull().sum() > 0:
            mode_series = master_df[col].mode()
            fill_val = mode_series[0] if not mode_series.empty else 'Unknown'
            master_df[col] = master_df[col].fillna(fill_val)
            cat_imputed_count += 1
            
    print(f"• Imputed {cat_imputed_count} categorical features using Mode Strategy.")

    remaining_nulls = master_df.isnull().sum().sum()
    print(f"\nPost-Imputation Total Missing Values Remaining: {remaining_nulls}")

    # 6. Save Cleaned & Imputed Master Dataset
    master_csv_path = os.path.join(OUTPUT_DIR, "Master_Hospital_Dataset.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"\nSaved Fully Imputed Master Dataset: '{master_csv_path}' (Shape: {master_df.shape})")

    # 7. Generate Quality Reports
    dq_report = pd.DataFrame([{
        'Dataset Name': 'Master_Hospital_Dataset',
        'Total Records': len(master_df),
        'Total Features': len(master_df.columns),
        'Total Missing Values': master_df.isnull().sum().sum(),
        'Duplicate Rows Dropped': num_duplicates,
        'Memory Footprint (MB)': round(master_df.memory_usage().sum() / (1024 * 1024), 2)
    }])
    dq_report.to_excel(os.path.join(OUTPUT_DIR, "Data_Quality_Report.xlsx"), index=False)

    missing_counts = master_df.isnull().sum()
    missing_pcts = (missing_counts / len(master_df)) * 100
    missing_report = pd.DataFrame({
        'Feature Name': missing_counts.index,
        'Missing Count': missing_counts.values,
        'Missing Percentage (%)': missing_pcts.values.round(2)
    }).sort_values(by='Missing Count', ascending=False)
    missing_report.to_excel(os.path.join(OUTPUT_DIR, "Missing_Value_Report.xlsx"), index=False)

    duplicate_report = pd.DataFrame([{
        'Initial Total Rows': initial_rows,
        'Duplicates Dropped': num_duplicates,
        'Final Unique Rows': final_unique_rows
    }])
    duplicate_report.to_excel(os.path.join(OUTPUT_DIR, "Duplicate_Report.xlsx"), index=False)

    print("\nUpdated Excel Quality Reports Generated in 'output/':")
    print(" - Data_Quality_Report.xlsx")
    print(" - Missing_Value_Report.xlsx")
    print(" - Duplicate_Report.xlsx")
    print("\nPipeline Execution Completed Successfully!")

if __name__ == "__main__":
    merge_clean_and_impute_pipeline()