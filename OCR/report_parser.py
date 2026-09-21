import os
import re
import json
import time
import pymupdf
import warnings
import logging
from dotenv import load_dotenv
from google import genai

# Suppress SDK warnings and internal loggers from printing to stdout
warnings.filterwarnings("ignore")
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)

# Load environment variables from .env
load_dotenv()

BASE_DIR = r"C:\AI_Lab_Report"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "OCR", "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "report_output.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def extract_text_row_by_row(pdf_path):
    doc = pymupdf.open(pdf_path)
    structured_lines = []

    for page in doc:
        words = page.get_text("words")  # (x0, y0, x1, y1, word)
        rows = {}
        for w in words:
            y_pos = round(w[1] / 4) * 4
            if y_pos not in rows:
                rows[y_pos] = []
            rows[y_pos].append(w)

        for y in sorted(rows.keys()):
            row_words = sorted(rows[y], key=lambda x: x[0])
            line_text = " ".join([w[4] for w in row_words]).strip()
            if line_text:
                structured_lines.append(line_text)

    return "\n".join(structured_lines)


def parse_demographics(raw_text):
    patient_info = {"name": "Unknown Patient", "age": "N/A", "gender": "N/A"}

    name_match = re.search(
        r"(?:Patient\s*Name|Name|Client\s*Name)\s*[:\-]?\s*([A-Za-z\s\.]+)",
        raw_text,
        re.IGNORECASE,
    )
    if name_match:
        extracted = name_match.group(1).split("\n")[0].strip()
        for drop_kw in [
            "LAB",
            "DIAGNOSTICS",
            "REPORT",
            "DRLOGY",
            "AGE",
            "GENDER",
            "SEX",
            "DATE",
        ]:
            if re.search(rf"\b{drop_kw}\b", extracted, re.IGNORECASE):
                extracted = re.split(
                    rf"\b{drop_kw}\b", extracted, flags=re.IGNORECASE
                )[0].strip()
        if len(extracted) > 2:
            patient_info["name"] = extracted

    age_match = re.search(
        r"\b([1-9][0-9]?)\s*(?:Yrs|Years|Y/O|Y)\b", raw_text, re.IGNORECASE
    )
    if age_match:
        patient_info["age"] = age_match.group(1)

    if re.search(r"\b(Female|Fem|F)\b", raw_text, re.IGNORECASE):
        patient_info["gender"] = "Female"
    elif re.search(r"\b(Male|M)\b", raw_text, re.IGNORECASE):
        patient_info["gender"] = "Male"

    return patient_info


def parse_tests_dynamic_fallback(raw_text):
    """
    Refined fallback parser that filters out phone numbers, addresses, UHID strings,
    and non-biomarker header text.
    """
    tests = []
    lines = raw_text.split("\n")

    IGNORE_KEYWORDS = [
        "PATIENT", "DOCTOR", "DATE", "REF", "UHID", "ROAD", "COMPLEX", "MUMBAI", 
        "PHONE", "TEL", "ADDRESS", "DRLOGY", "REGISTERED", "COLLECTED", "REPORTED", 
        "SAMPLE", "GENDER", "AGE", "0123456789", "SMART VISION"
    ]

    VALID_TEST_KEYWORDS = [
        "HAEMOGLOBIN", "LEUCOCYTE", "NEUTROPHILS", "LYMPHOCYTES", "EOSINOPHILS",
        "MONOCYTES", "BASOPHILS", "RBC", "MCV", "MCH", "MCHC", "HCT", "RDW",
        "PLATELET", "PCT", "MPV", "PDW", "PROTHROMBIN", "PT", "INR"
    ]

    for line in lines:
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 3:
            continue

        if any(kw in line_clean.upper() for kw in IGNORE_KEYWORDS):
            if not any(tk in line_clean.upper() for tk in VALID_TEST_KEYWORDS):
                continue

        match = re.search(
            r"^(.*?)\s+([\d]+\.?[\d]*)\s*(sec|seconds|mins|min|mg/dL|g/dL|uIU/mL|%|INR|fL|/dL|10\^3/uL|10\^6/uL)?\s*(.*)$",
            line_clean,
            re.IGNORECASE
        )

        if match:
            test_name = match.group(1).strip()
            val = match.group(2).strip()
            unit = match.group(3).strip() if match.group(3) else "N/A"
            ref = match.group(4).strip() if match.group(4) else "N/A"

            if re.match(r"^[\d\s\-\|]+$", test_name) or len(test_name) < 2:
                continue

            tests.append({
                "test": test_name,
                "value": val,
                "unit": unit,
                "reference": ref
            })

    return tests


def parse_report_with_gemini(raw_text):
    if not GEMINI_API_KEY:
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    You are an expert medical data extraction system. Extract ALL laboratory test parameters, numeric values, units, and reference ranges from this OCR text.
    Ignore hospital headers, phone numbers, addresses, and UHID codes.
    Also extract patient metadata (Name, Age, Gender). Do NOT fabricate data.

    Return ONLY raw valid JSON with this exact structure (do NOT wrap in ```json markdown codeblocks):
    {{
      "patient": {{ "name": "extracted name or Unknown Patient", "age": "extracted age or N/A", "gender": "extracted gender or N/A" }},
      "tests": [
        {{ "test": "Parameter Name", "value": "123.4", "unit": "unit", "reference": "range" }}
      ]
    }}

    OCR TEXT:
    {raw_text[:4000]}
    """

    candidate_models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name, 
                contents=prompt
            )

            res_text = response.text.strip()
            # Robust JSON extraction: Strip potential markdown backticks
            res_text = re.sub(r"^```json\s*", "", res_text, flags=re.MULTILINE)
            res_text = re.sub(r"^```\s*", "", res_text, flags=re.MULTILINE)
            res_text = res_text.strip("`").strip()

            # Find the JSON boundaries explicitly
            start_idx = res_text.find("{")
            end_idx = res_text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                res_text = res_text[start_idx:end_idx + 1]

            parsed = json.loads(res_text)
            if parsed and parsed.get("tests"):
                print(f"[OCR Gemini] Successfully extracted test data using {model_name}.")
                return parsed

        except Exception:
            continue

    return None


if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in input directory: {INPUT_DIR}")

    target_pdf = os.path.join(INPUT_DIR, pdf_files[0])
    print(f"[OCR] Target PDF: {target_pdf}")

    raw_text = extract_text_row_by_row(target_pdf)
    parsed_data = parse_report_with_gemini(raw_text)

    if not parsed_data or not parsed_data.get("tests"):
        print("[OCR] Executing dynamic fallback parser...")
        patient_data = parse_demographics(raw_text)
        tests_data = parse_tests_dynamic_fallback(raw_text)
        parsed_data = {"patient": patient_data, "tests": tests_data}

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2)

    print(f"[OCR] Output written to: {OUTPUT_JSON}")