import os
import pandas as pd
import numpy as np

# ======================================
# Folder containing all datasets
# ======================================
DATASET_FOLDER = r"C:\AI_Lab_Report\Dataset"

# ======================================
# Find every CSV file recursively
# ======================================
csv_files = []

for root, dirs, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print(f"\nFound {len(csv_files)} CSV files\n")

# ======================================
# Read every dataset
# ======================================
datasets = []

for file in csv_files:
    print(f"Reading : {os.path.basename(file)}")

    df = pd.read_csv(file)

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    datasets.append(df)

# ======================================
# Merge datasets
# ======================================
master_df = pd.concat(
    datasets,
    ignore_index=True,
    sort=False
)

# ======================================
# Remove duplicate rows
# ======================================
master_df.drop_duplicates(inplace=True)

# ======================================
# Replace empty strings with NaN
# ======================================
master_df.replace(
    ["", " ", "NA", "N/A", "null", "None"],
    np.nan,
    inplace=True
)

# ======================================
# Save
# ======================================
output_file = os.path.join(DATASET_FOLDER, "Master_Lab_Dataset.csv")

master_df.to_csv(output_file, index=False)

print("\n=================================")
print("Dataset merged successfully")
print("Rows :", master_df.shape[0])
print("Columns :", master_df.shape[1])
print("Saved at :", output_file)
print("=================================")