import os
import re
import json
import time
import pymupdf
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

BASE_DIR = r"C:\AI_Lab_Report"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "OCR", "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "report_output.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dynamic Gemini API Key lookup
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

    # Known noise keywords to ignore entirely
    IGNORE_KEYWORDS = [
        "PATIENT", "DOCTOR", "DATE", "REF", "UHID", "ROAD", "COMPLEX", "MUMBAI", 
        "PHONE", "TEL", "ADDRESS", "DRLOGY", "REGISTERED", "COLLECTED", "REPORTED", 
        "SAMPLE", "GENDER", "AGE", "0123456789", "SMART VISION"
    ]

    # Valid clinical biomarker patterns
    VALID_TEST_KEYWORDS = [
        "PROTHROMBIN", "PT", "INR", "APTT", "PTT", "FIBRINOGEN", "THROMBIN",
        "BLEEDING TIME", "CLOTTING TIME", "PLATELET", "FACTOR"
    ]

    for line in lines:
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 3:
            continue

        # Skip lines matching noise keywords
        if any(kw in line_clean.upper() for kw in IGNORE_KEYWORDS):
            # Exception: allow if it explicitly contains a known test parameter
            if not any(tk in line_clean.upper() for tk in VALID_TEST_KEYWORDS):
                continue

        # Extract parameters using matching regex
        # Look for [Test Name] [Value] [Optional Unit/Reference]
        match = re.search(
            r"^(.*?)\s+([\d]+\.?[\d]*)\s*(sec|seconds|mins|min|mg/dL|g/dL|uIU/mL|%|INR)?\s*(.*)$",
            line_clean,
            re.IGNORECASE
        )

        if match:
            test_name = match.group(1).strip()
            val = match.group(2).strip()
            unit = match.group(3).strip() if match.group(3) else "N/A"
            ref = match.group(4).strip() if match.group(4) else "N/A"

            # Filter out pure digits or meaningless short names
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
        print("[OCR Gemini] Skipping LLM parsing: No GEMINI_API_KEY found.")
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
    You are an expert medical data extraction system. Extract ALL laboratory test parameters, numeric values, units, and reference ranges from this OCR text.
    Ignore hospital headers, phone numbers, addresses, and UHID codes.
    Also extract patient metadata (Name, Age, Gender). Do NOT fabricate data.

    Return ONLY valid JSON with this exact structure:
    {{
      "patient": {{ "name": "extracted name or Unknown Patient", "age": "extracted age or N/A", "gender": "extracted gender or N/A" }},
      "tests": [
        {{ "test": "Parameter Name", "value": "123.4", "unit": "unit", "reference": "range" }}
      ]
    }}

    OCR TEXT:
    {raw_text[:4000]}
    """

    # Retry loop with exponential backoff (3s -> 6s -> 12s) to handle Gemini 503 capacity spikes
    max_retries = 3
    retry_delays = [3, 6, 12]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash", 
                contents=prompt
            )

            res_text = response.text
            res_text = re.sub(r"```json\s*", "", res_text)
            res_text = re.sub(r"```\s*", "", res_text).strip()

            parsed = json.loads(res_text)
            if parsed and parsed.get("tests") and len(parsed["tests"]) > 0:
                print("[OCR Gemini] Successfully extracted test data using gemini-3.6-flash.")
                return parsed

        except Exception as e:
            wait_time = retry_delays[attempt - 1]
            print(f"[OCR Gemini Attempt {attempt}/{max_retries}] Exception: {str(e)}. Retrying in {wait_time}s...")
            if attempt < max_retries:
                time.sleep(wait_time)

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