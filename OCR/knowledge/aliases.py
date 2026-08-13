TEST_ALIASES = {

    # ==========================
    # CBC
    # ==========================

    "HB": "Hemoglobin",
    "HGB": "Hemoglobin",
    "HAEMOGLOBIN": "Hemoglobin",

    "TLC": "Total Leukocyte Count",
    "WBC": "Total Leukocyte Count",
    "TOTAL WBC COUNT": "Total Leukocyte Count",

    "RBC": "Total RBC Count",
    "TOTAL RBC": "Total RBC Count",

    "PCV": "Hematocrit (HCT)",
    "HCT": "Hematocrit (HCT)",

    "MCV": "Mean Corpuscular Volume (MCV)",
    "MCH": "Mean Cell Hemoglobin (MCH)",
    "MCHC": "Mean Cell Hemoglobin Concentration (MCHC)",

    "PLATELETS": "Platelet Count",

    "NEUTROPHIL": "Neutrophils",
    "LYMPHOCYTE": "Lymphocytes",
    "EOSINOPHIL": "Eosinophils",
    "MONOCYTE": "Monocytes",
    "BASOPHIL": "Basophils",

    # ==========================
    # LFT & Common OCR Typos
    # ==========================

    "SGOT": "AST",
    "AST (SGOT)": "AST",
    "SOOT (AST)": "AST",
    "SOOT": "AST",

    "SGPT": "ALT",
    "ALT (SGPT)": "ALT",
    "SGIPT": "ALT",

    "SGOT / SGPT RATIO": "AST/ALT Ratio",
    "SGOT/SGPT RATIO": "AST/ALT Ratio",
    "AST/ALT RATIO": "AST/ALT Ratio",
    "SOOT /SGIPT RAT I0": "AST/ALT Ratio",
    "SOOT /SGIPT RATIO": "AST/ALT Ratio",
    "SOOT /SGIPT RAT I0": "AST/ALT Ratio",

    # ==========================
    # KFT
    # ==========================

    "UREA": "Blood Urea",
    "BLOOD UREA": "Blood Urea",

    "CREATININE": "Serum Creatinine",
    "SERUM CREATININE": "Serum Creatinine",

    "URIC ACID": "Uric Acid",

    "NA": "Sodium",
    "K": "Potassium",
    "CL": "Chloride",

    # ==========================
    # Lipid
    # ==========================

    "TOTAL CHOLESTEROL": "Total Cholesterol",

    "TRIGLYCERIDES": "Triglycerides",
    "TG": "Triglycerides",

    "HDL": "HDL Cholesterol",
    "LDL": "LDL Cholesterol",
    "VLDL": "VLDL Cholesterol",

    # ==========================
    # Diabetes
    # ==========================

    "FBS": "Fasting Blood Sugar",
    "RBS": "Random Blood Sugar",
    "PPBS": "Postprandial Blood Sugar",

    "HBA1C": "HbA1c",

    # ==========================
    # Thyroid
    # ==========================

    "FT3": "Free T3",
    "FT4": "Free T4",

    "T3": "T3",
    "T4": "T4",
    "TSH": "TSH",

    # ==========================
    # Vitamins
    # ==========================

    "B12": "Vitamin B12",

    "25 OH VITAMIN D": "Vitamin D",
    "25-OH VITAMIN D": "Vitamin D"
}

# Helper function to match text ignoring case
def get_clean_test_name(raw_name: str) -> str:
    cleaned = raw_name.strip().upper()
    
    # Check exact match in aliases
    if cleaned in TEST_ALIASES:
        return TEST_ALIASES[cleaned]
        
    # Check partial matches for noisy OCR strings
    if "SOOT" in cleaned or "SGOT" in cleaned:
        if "RAT" in cleaned or "RATIO" in cleaned:
            return "AST/ALT Ratio"
        return "AST"
        
    if "SGPT" in cleaned or "SGIPT" in cleaned:
        return "ALT"
        
    return raw_name