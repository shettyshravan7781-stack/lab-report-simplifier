import os
import sys
import json
import joblib
import re
import time
import logging
import warnings
import subprocess
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Suppress runtime deprecation and non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress Google GenAI SDK verbose internal loggers
logging.getLogger("google.genai").setLevel(logging.ERROR)

# -------------------------------------------------------------------------
# Dynamic Google Gemini API Key Setup
# -------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------------------------------------------------------
# 1. ENVIRONMENT & PATH SETUP
# -------------------------------------------------------------------------
BASE_DIR = r"C:\AI_Lab_Report"
MODELS_DIR = os.path.join(BASE_DIR, "output", "saved_models")
JSON_PATH = os.path.join(BASE_DIR, "OCR", "output", "report_output.json")
SHAP_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "shap_plots")
INPUT_DIR = os.path.join(BASE_DIR, "input")
PARSER_SCRIPT = os.path.join(BASE_DIR, "OCR", "report_parser.py")

os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# 2. UNIVERSAL CLINICAL REFERENCE DICTIONARY
# -------------------------------------------------------------------------
FALLBACK_REF_RANGES = {
    "Hemoglobin": "13.0 - 17.0", "Haemoglobin": "13.0 - 17.0", "Hb": "13.0 - 17.0",
    "RBC Count": "4.5 - 5.5", "Hematocrit": "40.0 - 50.0", "Hct": "40.0 - 50.0",
    "MCV": "81.0 - 101.0", "MCH": "27.0 - 32.0", "MCHC": "31.5 - 34.5",
    "RDW-CV": "11.6 - 14.0", "RDW-SD": "39.0 - 46.0", "WBC": "4000 - 10000",
    "Total Leucocyte Count": "4000 - 10000", "Platelet Count": "150000 - 410000",
    "PCT": "0.10 - 0.40", "MPV": "7.5 - 11.5", "PDW": "9.0 - 17.0",
    "Neutrophils": "40 - 70", "Lymphocytes": "20 - 40", "Eosinophils": "1 - 6",
    "Monocytes": "2 - 10", "Basophils": "0 - 1",
    "Bilirubin Total": "0.2 - 1.2", "Bilirubin Direct": "0.0 - 0.3", "Bilirubin Indirect": "0.2 - 0.8",
    "ALT": "7 - 56", "ALT/SGPT": "7 - 56", "SGPT": "7 - 56",
    "AST": "10 - 40", "AST/SGOT": "10 - 40", "SGOT": "10 - 40",
    "ALP": "44 - 147", "GGT": "9 - 48", "Protein Total": "6.0 - 8.3",
    "Albumin": "3.5 - 5.0", "Globulin": "2.0 - 3.5",
    "Creatinine": "0.7 - 1.3", "Urea": "15 - 45", "Blood Urea": "15 - 45",
    "BUN": "7 - 20", "Blood Urea Nitrogen": "7 - 20", "eGFR": "90 - 120", 
    "Estimated GFR": "90 - 120", "Uric Acid": "3.5 - 7.2",
    "Total Cholesterol": "125 - 200", "Triglycerides": "30 - 150",
    "HDL": "40 - 60", "LDL": "50 - 100", "VLDL": "5 - 40", "Non-HDL": "0 - 130",
    "HbA1c": "4.0 - 5.6", "FBS": "70 - 99", "Fasting Glucose": "70 - 99",
    "PPBS": "70 - 140", "Postprandial Glucose": "70 - 140",
    "TT3": "75 - 175", "T3, TOTAL": "80 - 200", "Total T3": "80 - 200",
    "TT4": "4.5 - 12.0", "T4, TOTAL": "4.5 - 12.5", "Total T4": "4.5 - 12.5",
    "TSH": "0.4 - 4.2", "Specific Gravity": "1.005 - 1.030", "pH": "4.6 - 8.0",
    "Serum Iron": "60 - 170", "Iron": "60 - 170", "TIBC": "240 - 450", 
    "UIBC": "111 - 343", "Transferrin Saturation": "20 - 50",
    "Sodium": "137 - 145", "Potassium": "3.5 - 5.1", "Chloride": "98 - 107",
    "Calcium": "8.5 - 10.5", "Phosphorus": "2.5 - 4.5",
    "Urine Albumin": "0 - 20", "Urine Creatinine": "20 - 320", "Albumin/Creatinine Ratio": "0 - 30",
    "Prothrombin Time": "11.0 - 13.5", "Bleeding Time": "2.0 - 7.0", "Clotting Time": "3.0 - 9.0", "APTT": "25.0 - 35.0"
}

# -------------------------------------------------------------------------
# 3. PURGE STALE CACHE & EXECUTE OCR PIPELINE
# -------------------------------------------------------------------------
if os.path.exists(JSON_PATH):
    try:
        os.remove(JSON_PATH)
        print("[INIT] Flushed stale OCR output cache.")
    except Exception as e:
        print(f"[WARN] Failed to flush cache: {e}")

print("[OCR] Executing OCR Parser Engine...")
try:
    subprocess.run([sys.executable, PARSER_SCRIPT], check=True)
except Exception as e:
    print(f"[ERROR] OCR execution error: {e}")

# -------------------------------------------------------------------------
# 4. READ & INGEST EXTRACTED OCR DATA
# -------------------------------------------------------------------------
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"CRITICAL: Failed to generate {JSON_PATH}. Check parser script paths.")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw_ocr = json.load(f)

all_tests_list = []
patient_info = {}

if isinstance(raw_ocr, list):
    for page in raw_ocr:
        all_tests_list.extend(page.get("tests", []) or page.get("lab_tests", []))
        if page.get("patient"):
            patient_info.update(page.get("patient"))
elif isinstance(raw_ocr, dict):
    all_tests_list = raw_ocr.get("tests", []) or raw_ocr.get("lab_tests", [])
    patient_info = raw_ocr.get("patient", {})

extracted_tests = {}
abnormal_findings = []

def parse_age_gender(p_dict, tests_list):
    raw_age = p_dict.get('age') or p_dict.get('Age') or ""
    raw_gender = p_dict.get('gender') or p_dict.get('Gender') or p_dict.get('sex') or ""
    
    combined_str = f"{raw_age} {raw_gender}"
    
    if not str(raw_age).strip() or str(raw_age).strip().upper() in ["N/A", "NONE", ""]:
        for item in tests_list:
            t_str = str(item.get("test") or "") + " " + str(item.get("value") or "")
            if re.search(r"\b(age|yrs|years)\b", t_str, re.IGNORECASE):
                combined_str += " " + t_str

    age_match = re.search(r"\b([1-9][0-9]?|1[01][0-9]|120)\s*(?:yrs|years|y/o|y)?\b", combined_str, re.IGNORECASE)
    extracted_age = age_match.group(1) if age_match else "N/A"
    extracted_gender = "Female" if re.search(r"\b(female|fem|f)\b", combined_str, re.IGNORECASE) else "Male" if re.search(r"\b(male|m)\b", combined_str, re.IGNORECASE) else "N/A"
        
    return extracted_age, extracted_gender

clean_age, clean_gender = parse_age_gender(patient_info, all_tests_list)

def extract_numeric(val):
    if val is None:
        return None
    match = re.search(r"([\d]+\.?[\d]*)", str(val))
    return float(match.group(1)) if match else None

def normalize_biomarker_value(test_name, val):
    if val is None:
        return None
    name_upper = test_name.upper()
    if "PCT" in name_upper and val > 2.0:
        return round(val / 100.0, 3)
    if "PLATELET" in name_upper and val > 1000000:
        return round(val / 1000.0, 1)
    return val

def parse_ocr_entry(item):
    raw_test = str(item.get("test") or item.get("test_name") or "").strip()
    raw_val = str(item.get("value") or "").strip()
    raw_ref = str(item.get("reference") or item.get("reference_range") or item.get("unit") or "").strip()

    num_val = extract_numeric(raw_val)
    clean_test_name = raw_test

    if num_val is None:
        match_in_test = re.search(r"^(.*?)\s+([\d]+\.?[\d]*)$", raw_test)
        if match_in_test:
            clean_test_name = match_in_test.group(1).strip()
            num_val = float(match_in_test.group(2))

    if not raw_ref or raw_ref.upper() in ["N/A", ""]:
        for k, ref_v in FALLBACK_REF_RANGES.items():
            if k.lower() in clean_test_name.lower():
                raw_ref = ref_v
                break

    return clean_test_name, num_val, raw_val, raw_ref

def is_out_of_range(val, ref_range):
    if not ref_range or str(ref_range).strip().upper() in ["N/A", ""]:
        return False
    ref_str = str(ref_range).strip().lower()

    try:
        num_val = float(val)
    except (ValueError, TypeError):
        return False

    if "<" in ref_str:
        lim = extract_numeric(ref_str)
        return num_val >= lim if lim is not None else False
    if ">" in ref_str:
        lim = extract_numeric(ref_str)
        return num_val <= lim if lim is not None else False

    numbers = re.findall(r"[\d]+\.?[\d]*", ref_str)
    if len(numbers) >= 2:
        low, high = float(numbers[0]), float(numbers[1])
        if low > high:
            low, high = high, low
        return num_val < low or num_val > high

    return False

GARBAGE_KEYWORDS = [
    "SMART VISION", "COMPLEX", "ROAD", "MUMBAI", "CIRCADIAN", "VARIATION", 
    "REGISTERED", "COLLECTED", "REPORTED ON", "SAMPLE TYPE", "TAT", "AGE :"
]

for item in all_tests_list:
    test_name, num_val, raw_val_str, ref_range = parse_ocr_entry(item)
    
    if any(gb in test_name.upper() for gb in GARBAGE_KEYWORDS) or len(test_name) < 2:
        continue

    num_val = normalize_biomarker_value(test_name, num_val)
    
    if test_name and num_val is not None:
        extracted_tests[test_name] = num_val
        if is_out_of_range(num_val, ref_range):
            abnormal_findings.append({
                "test": test_name,
                "val": num_val,
                "unit": item.get("unit", ""),
                "flag": f"Out of Range (Ref: {ref_range})"
            })

raw_panel = str(patient_info.get('report_type', 'N/A')).upper()

def detect_panel(tests_keys, raw_type):
    joined = " ".join(tests_keys).upper() + " " + raw_type.upper()
    detected = []
    
    if re.search(r"\b(PROTHROMBIN|BLEEDING TIME|CLOTTING TIME|APTT|PT|INR|FIBRINOGEN|COAGULATION)\b", joined):
        detected.append("Coagulation / Hemostasis Profile")
    if re.search(r"\b(T3|T4|TSH|THYROID)\b", joined): 
        detected.append("Thyroid Profile")
    if re.search(r"\b(HBA1C|GLUCOSE|FBS|PPBS|BLOOD SUGAR)\b", joined): 
        detected.append("Diabetes Profile")
    if re.search(r"\b(CREATININE|UREA|BUN|EGFR|URIC ACID)\b", joined): 
        detected.append("Kidney Function Test (KFT)")
    if re.search(r"\b(CHOLESTEROL|TRIGLYCERIDES|HDL|LDL|VLDL)\b", joined): 
        detected.append("Lipid Profile")
    if re.search(r"\b(BILIRUBIN|ALT|AST|SGOT|SGPT|GGT|ALP)\b", joined): 
        detected.append("Liver Function Test (LFT)")
    if re.search(r"\b(HEMOGLOBIN|HAEMOGLOBIN|WBC|LEUCOCYTE|RBC|PLATELET|NEUTROPHILS|MPV|MCV)\b", joined): 
        detected.append("Complete Blood Count (CBC)")
    if re.search(r"\b(URINE|URINALYSIS|SPECIFIC GRAVITY)\b", joined): 
        detected.append("Urinalysis")

    if len(detected) > 1:
        return "Comprehensive Multi-Panel Assessment (" + ", ".join(detected) + ")"
    elif len(detected) == 1:
        return detected[0]
    return "General Clinical Panel"

panel_type = detect_panel(list(extracted_tests.keys()), raw_panel)

# -------------------------------------------------------------------------
# 5. TERMINAL HEADER
# -------------------------------------------------------------------------
patient_name = patient_info.get('name', 'N/A')
if patient_name == 'N/A' or any(gb in patient_name.upper() for gb in ["STEROID", "THERAPY", "ROAD"]):
    patient_name = "Dynamic Patient"

print("\n" + "═" * 75)
print("     AI CLINICAL DIAGNOSTIC PIPELINE — INTEGRATED AUDIT REPORT")
print("═" * 75)
print(f" Patient Name : {patient_name}")
print(f" Age / Gender : {clean_age} / {clean_gender}")
print(f" Target Panel : {panel_type}")
print("─" * 75)

# -------------------------------------------------------------------------
# 6. DISPLAY EXTRACTED PARAMETERS TABLE
# -------------------------------------------------------------------------
print("\n[SECTION 1: EXTRACTED LABORATORY BIOMARKERS]")
print(f"{'Biomarker Parameter':<32} | {'Extracted Value':<18} | {'Status':<15}")
print("─" * 75)
for test_k, val_v in extracted_tests.items():
    is_abnormal = any(ab["test"] == test_k for ab in abnormal_findings)
    status_str = "⚠️ OUT OF RANGE" if is_abnormal else "✅ NORMAL"
    print(f"{test_k:<32} | {str(val_v):<18} | {status_str:<15}")

# -------------------------------------------------------------------------
# 7. DISPLAY ABNORMAL CRITICAL FLAGS
# -------------------------------------------------------------------------
print("\n[SECTION 2: CRITICAL CLINICAL FLAGS & OUT-OF-RANGE PARSING]")
if abnormal_findings:
    for item in abnormal_findings:
        print(f" 🚨 FLAG DETECTED: {item['test']} = {item['val']} {item['unit']} ➔ {item['flag']}")
else:
    print(" ✅ No parameter reference range deviations detected.")

# -------------------------------------------------------------------------
# 8. DYNAMIC FUZZY MATCHING MACHINE LEARNING ENGINE (XGBoost)
# -------------------------------------------------------------------------
print("\n[SECTION 3: XGBoost MACHINE LEARNING ORGAN RISK PROBABILITIES]")
calculated_risks = []

def fuzzy_map_feature(extracted_dict, target_feature):
    tf_clean = re.sub(r'[^a-zA-Z0-9]', '', target_feature).lower()
    
    for ext_k, val in extracted_dict.items():
        ext_clean = re.sub(r'[^a-zA-Z0-9]', '', ext_k).lower()
        if ext_clean in tf_clean or tf_clean in ext_clean:
            return val
            
    for ext_k, val in extracted_dict.items():
        ek_l = ext_k.lower()
        tf_l = target_feature.lower()
        if ("hb" in ek_l or "hemoglobin" in ek_l) and "hemoglobin" in tf_l: return val
        if ("wbc" in ek_l or "leucocyte" in ek_l) and "wbc" in tf_l: return val
        if "creatinine" in ek_l and "creatinine" in tf_l: return val
        if "urea" in ek_l and "urea" in tf_l: return val
        if "tsh" in ek_l and "tsh" in tf_l: return val
        if ("fbs" in ek_l or "fasting" in ek_l) and ("fbs" in tf_l or "fasting" in tf_l): return val
        if "hba1c" in ek_l and "hba1c" in tf_l: return val

    return np.nan

if os.path.exists(MODELS_DIR):
    for pkl_file in sorted(os.listdir(MODELS_DIR)):
        if pkl_file.endswith(".pkl"):
            target_name = pkl_file.replace("xgboost_", "").replace(".pkl", "")
            model_path = os.path.join(MODELS_DIR, pkl_file)
            model = joblib.load(model_path)

            if not hasattr(model, "feature_names_in_"):
                continue

            feature_dict = {feat: np.nan for feat in model.feature_names_in_}
            
            for feat in model.feature_names_in_:
                val = fuzzy_map_feature(extracted_tests, feat)
                if not np.isnan(val):
                    feature_dict[feat] = val

            if clean_age != 'N/A' and "Age" in feature_dict:
                try:
                    feature_dict["Age"] = float(clean_age)
                except ValueError:
                    feature_dict["Age"] = 0.0

            df_in = pd.DataFrame([feature_dict]).apply(pd.to_numeric, errors='coerce').fillna(0.0)
            prob = model.predict_proba(df_in)[0][1] if hasattr(model, "predict_proba") else model.predict(df_in)[0]

            abnormal_names = [item["test"].lower() for item in abnormal_findings]
            if target_name == "Target_Kidney_Risk" and any(kw in name for name in abnormal_names for kw in ["creatinine", "urea", "bun", "egfr"]):
                prob = max(prob, 0.85)

            risk_pct = prob * 100
            status = "HIGH RISK ⚠️" if risk_pct > 50 else "LOW RISK   ✅"
            print(f" • {target_name:28s}: {risk_pct:6.2f}%  [{status}]")
            calculated_risks.append(f"{target_name}: {risk_pct:.2f}% ({status})")
else:
    print(" [WARN] No saved models found in directory.")

# -------------------------------------------------------------------------
# 9. DETAILED CLINICAL NARRATIVE GENERATOR (GEMINI LLM WITH RETRY & FALLBACK)
# -------------------------------------------------------------------------
def generate_gemini_summary(p_name, age, gender, p_type, extracted_dict, abnormal_list, risk_summary):
    if not GEMINI_API_KEY:
        return "Clinical summary generation skipped: GEMINI_API_KEY is not set."

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    You are an expert Medical Pathologist and Clinical AI Consultant.
    Analyze this lab report payload and generate a structured, easy-to-read assessment.

    PATIENT: {p_name} (Age: {age}, Gender: {gender})
    PANEL TYPE: {p_type}

    EXTRACTED LAB TESTS:
    {json.dumps(extracted_dict, indent=2)}

    OUT OF RANGE PARAMETERS:
    {json.dumps(abnormal_list, indent=2)}

    ML ORGAN RISK PREDICTIONS:
    {risk_summary}

    PROVIDE THESE 4 DETAILED SECTIONS IN PLAIN TEXT:

    EXECUTIVE SUMMARY
    - High-level overview of the patient's diagnostic profile.

    BIOMARKER ANALYSIS
    - Plain-language explanation of extracted biomarker values and their diagnostic meaning.

    HEALTH THREATS & POTENTIAL RISKS
    - Explicitly state potential diseases, organ dysfunctions, or systemic risks associated with any abnormal levels.

    RECOMMENDED ACTION PLAN
    - 4 to 5 clear, realistic next steps (lifestyle changes, follow-up tests, medical specialist consults).
    """

    candidate_models = ['gemini-3.6-flash']
    max_retries = 3
    retry_delays = [3, 6, 12]
    last_error = ""

    for model_name in candidate_models:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                last_error = str(e)
                wait_time = retry_delays[attempt - 1] if attempt <= len(retry_delays) else 5
                err_msg = last_error.upper()
                
                if "NOT_FOUND" in err_msg or "404" in err_msg:
                    break  # Skip model if invalid model name
                
                if ("503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg):
                    if attempt < max_retries:
                        print(f" [Section 4 Gemini Attempt {attempt}/{max_retries}] API spike detected ({err_msg[:30]}...). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        break
                else:
                    break

    # Automated Local Fallback Clinical Narrative
    abnormal_str = ", ".join([item["test"] for item in abnormal_list]) if abnormal_list else "None"
    return f"""EXECUTIVE SUMMARY
-----------------
Diagnostic assessment generated for {p_name} ({age}, {gender}) evaluating a {p_type}. 
{"Lab biomarkers show notable out-of-range parameters requiring clinical attention: " + abnormal_str if abnormal_list else "All extracted lab biomarkers fall within standard reference ranges."}

BIOMARKER ANALYSIS
------------------
- Extracted Parameters: {len(extracted_dict)} parameters successfully analyzed.
- Out-of-Range Parameters: {abnormal_str}.

HEALTH THREATS & POTENTIAL RISKS
--------------------------------
- Organ Risk Assessment: Overall machine learning organ risk predictions remain low.
{"- Flagged Parameters: Require clinical correlation with patient symptoms and medication history." if abnormal_list else "- Physiological Status: Normal overall biomarker trends."}

RECOMMENDED ACTION PLAN
-----------------------
1. Review results with a primary care physician or appropriate specialist.
2. Correlate laboratory findings with clinical signs, medication history, and symptoms.
3. Consider repeating tests after 2–4 weeks if clinically indicated.
4. Maintain routine health screenings as recommended by clinical practice guidelines.
"""

print("\n[SECTION 4: DETAILED CLINICAL NARRATIVE & THREAT ASSESSMENT]")
print("─" * 75)
risk_text = "\n".join(calculated_risks) if calculated_risks else "No ML models loaded."
summary_output = generate_gemini_summary(patient_name, clean_age, clean_gender, panel_type, extracted_tests, abnormal_findings, risk_text)
print(summary_output)
print("=" * 75)