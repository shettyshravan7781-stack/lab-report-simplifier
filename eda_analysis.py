import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_exploratory_data_analysis():
    print("==============================================================")
    print("   Starting Step 1: Exploratory Data Analysis (EDA) & EDA     ")
    print("==============================================================\n")
    
    # 1. Paths Setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "eda_plots")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load Master Dataset
    dataset_path = os.path.join(OUTPUT_DIR, "Master_Hospital_Dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(BASE_DIR, "Master_Hospital_Dataset.csv")
        
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"Dataset successfully loaded. Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})\n")

    # Set plot aesthetic style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # -----------------------------------------------------------------
    # A. BIOMARKER CORRELATION HEATMAPS
    # -----------------------------------------------------------------
    print("1. Generating Biomarker Correlation Heatmaps...")
    
    # Define primary clinical biomarkers across major panels
    primary_biomarkers = [
        'HbA1c %', 'Fasting Glucose (mg/dL)', 'Postprandial Glucose (mg/dL)',
        'Creatinine (mg/dL)', 'eGFR (mL/min/1.73m²)', 'Urea (mg/dL)',
        'ALT/SGPT (U/L)', 'AST/SGOT (U/L)', 'Bilirubin Total (mg/dL)',
        'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'LDL (mg/dL)', 'Triglycerides (mg/dL)',
        'Hemoglobin (g/dL)', 'RBC Count (mil/µL)', 'Age', 'Registration Weight'
    ]
    
    # Filter columns present in dataset and convert to numeric
    avail_biomarkers = [col for col in primary_biomarkers if col in df.columns]
    for col in avail_biomarkers:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    corr_matrix = df[avail_biomarkers].corr()

    # Plot 1: Overall Biomarker Correlation Heatmap
    plt.figure(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="vlag", 
                          linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
    plt.title("Biomarker Inter-Correlation Heatmap (Primary Clinical Panels)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "01_biomarker_correlation_heatmap.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"   ✓ Saved Heatmap: '{plot1_path}'")

    # -----------------------------------------------------------------
    # B. SPECIFIC PAIRWISE BIOMARKER SCATTER / REGRESSION PLOTS
    # -----------------------------------------------------------------
    print("2. Generating Pairwise Biomarker Scatter Plots...")
    
    pairs = [
        ('HbA1c %', 'Fasting Glucose (mg/dL)', 'Glycemic Correlation: HbA1c vs Fasting Glucose'),
        ('Creatinine (mg/dL)', 'eGFR (mL/min/1.73m²)', 'Renal Function Correlation: Creatinine vs eGFR'),
        ('ALT/SGPT (U/L)', 'Triglycerides (mg/dL)', 'Hepato-Metabolic Link: ALT/SGPT vs Triglycerides'),
        ('Hemoglobin (g/dL)', 'RBC Count (mil/µL)', 'Hematology Link: Hemoglobin vs RBC Count')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (x_col, y_col, title) in enumerate(pairs):
        if x_col in df.columns and y_col in df.columns:
            sns.regplot(data=df, x=x_col, y=y_col, ax=axes[idx],
                        scatter_kws={'alpha':0.3, 'color':'#2b5c8f', 's':15}, 
                        line_kws={'color':'#d95f02', 'linewidth':2})
            axes[idx].set_title(title, fontweight='bold')
            axes[idx].set_xlabel(x_col)
            axes[idx].set_ylabel(y_col)

    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "02_pairwise_biomarker_regressions.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"   ✓ Saved Regression Plots: '{plot2_path}'")

    # -----------------------------------------------------------------
    # C. DEMOGRAPHIC DISTRIBUTIONS & SUBGROUP ANALYSIS
    # -----------------------------------------------------------------
    print("3. Analyzing Demographic Distributions (Age, Gender, Diet)...")

    # Plot 3: Age & Weight Distribution across Gender
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'Age' in df.columns and 'Gender' in df.columns:
        sns.kdeplot(data=df, x='Age', hue='Gender', common_norm=False, fill=True, alpha=0.3, ax=axes[0])
        axes[0].set_title("Age Density Distribution by Gender", fontweight='bold')
        
    if 'Registration Weight' in df.columns and 'Diet' in df.columns:
        sns.boxplot(data=df, x='Diet', y='Registration Weight', palette='Set2', ax=axes[1])
        axes[1].set_title("Registration Weight across Diet Groups", fontweight='bold')

    plt.tight_layout()
    plot3_path = os.path.join(PLOTS_DIR, "03_demographic_distributions.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"   ✓ Saved Demographic Plots: '{plot3_path}'")

    # Plot 4: Metabolic Indicators across Diet Subgroups
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    diet_biomarkers = ['Total Cholesterol (mg/dL)', 'Triglycerides (mg/dL)', 'HbA1c %']
    for idx, bio in enumerate(diet_biomarkers):
        if bio in df.columns and 'Diet' in df.columns:
            sns.boxplot(data=df, x='Diet', y=bio, palette='Blues', ax=axes[idx])
            axes[idx].set_title(f"{bio} by Diet Pattern", fontweight='bold')

    plt.tight_layout()
    plot4_path = os.path.join(PLOTS_DIR, "04_biomarkers_by_diet.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"   ✓ Saved Diet Comparison Plots: '{plot4_path}'")

    # -----------------------------------------------------------------
    # D. EXPORT STATISTICAL SUMMARY REPORTS TO EXCEL
    # -----------------------------------------------------------------
    print("4. Exporting Summary Statistics Report...")

    # Summary Stats
    summary_stats = df[avail_biomarkers].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    summary_stats = summary_stats.rename(columns={'50%': 'median'}).round(2)

    # Demographic Breakdown
    gender_counts = df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Patient Visit Count']

    diet_counts = df['Diet'].value_counts().reset_index()
    diet_counts.columns = ['Diet Pattern', 'Patient Visit Count']

    excel_report_path = os.path.join(OUTPUT_DIR, "EDA_Summary_Report.xlsx")
    with pd.ExcelWriter(excel_report_path, engine='openpyxl') as writer:
        summary_stats.to_excel(writer, sheet_name='Biomarker Statistics')
        corr_matrix.round(3).to_excel(writer, sheet_name='Correlation Matrix')
        gender_counts.to_excel(writer, sheet_name='Gender Distribution', index=False)
        diet_counts.to_excel(writer, sheet_name='Diet Distribution', index=False)

    print(f"   ✓ Saved Summary Report: '{excel_report_path}'")
    print("\n==============================================================")
    print("   Step 1 Complete! All EDA charts generated successfully.    ")
    print("==============================================================")

if __name__ == "__main__":
    run_exploratory_data_analysis()