import os
import re
import json
import pymupdf
from google import genai



BASE_DIR = r"C:\AI_Lab_Report"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "OCR", "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "report_output.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Google Gemini API Key
GEMINI_API_KEY = "AQ.Ab8RN6JwuJYiIdUVQMBjdBeDlHGkcRFTZKeDEAwpzzhcKklBLQ"

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

    name_match = re.search(r"(?:Patient\s*Name|Name|Client\s*Name)\s*[:\-]?\s*([A-Za-z\s\.]+)", raw_text, re.IGNORECASE)
    if name_match:
        extracted = name_match.group(1).split("\n")[0].strip()
        for drop_kw in ["LAB", "DIAGNOSTICS", "REPORT", "DRLOGY", "AGE", "GENDER", "SEX", "DATE"]:
            if re.search(rf"\b{drop_kw}\b", extracted, re.IGNORECASE):
                extracted = re.split(rf"\b{drop_kw}\b", extracted, flags=re.IGNORECASE)[0].strip()
        if len(extracted) > 2:
            patient_info["name"] = extracted

    age_match = re.search(r"\b([1-9][0-9]?)\s*(?:Yrs|Years|Y/O|Y)\b", raw_text, re.IGNORECASE)
    if age_match:
        patient_info["age"] = age_match.group(1)

    if re.search(r"\b(Female|Fem|F)\b", raw_text, re.IGNORECASE):
        patient_info["gender"] = "Female"
    elif re.search(r"\b(Male|M)\b", raw_text, re.IGNORECASE):
        patient_info["gender"] = "Male"

    return patient_info

def parse_tests_dynamic_fallback(raw_text):
    tests = []
    lines = raw_text.split("\n")
    
    skip_header_words = ["PATIENT", "DOCTOR", "DATE", "REF", "BIOLOGICAL", "PARAMETER", "REPORTED", "SAMPLE", "TAT", "AGE"]
    stop_keywords = ["end of report", "notes:", "disclaimer", "methodology", "technician", "instrumentation"]

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        if any(sk in line_clean.lower() for sk in stop_keywords):
            break

        if any(hw in line_clean.upper() for hw in skip_header_words) and not re.search(r"\b(T3|T4|TSH|HbA1c|WBC|RBC|Hb|Hemoglobin)\b", line_clean, re.IGNORECASE):
            continue

        tokens = line_clean.split()
        num_idx = -1
        
        for idx, token in enumerate(tokens):
            if re.match(r"^\d+(\.\d+)?$", token.strip(",")) and not (idx > 0 and tokens[idx-1].upper() in ["TOTAL", "FREE"]):
                num_idx = idx
                break

        if num_idx > 0:
            raw_test_name = " ".join(tokens[:num_idx]).strip()
            val = tokens[num_idx].strip(",")
            remainder = tokens[num_idx+1:] if num_idx + 1 < len(tokens) else []

            clean_test_name = re.sub(r"\s+[\d\.]+\s*(?:Normal|High|Low)?.*$", "", raw_test_name, flags=re.IGNORECASE).strip()

            unit = "N/A"
            ref = "N/A"

            if remainder:
                if remainder[0] in ["ng/dL", "ug/dL", "uIU/mL", "µIU/mL", "mg/dL", "g/dL", "%", "mmol/L", "cells/cu.mm", "g/dl"]:
                    unit = remainder[0]
                    ref = " ".join(remainder[1:]) if len(remainder) > 1 else "N/A"
                else:
                    ref = " ".join(remainder)

            if len(clean_test_name) >= 2:
                tests.append({
                    "test": clean_test_name,
                    "value": val,
                    "unit": unit,
                    "reference": ref
                })

    return tests

def parse_report_with_gemini(raw_text):
    if not GEMINI_API_KEY:
        print("[OCR Gemini] Skipping LLM parsing: No API key provided.")
        return None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = f"""
        You are an expert medical data extraction system. Extract ALL laboratory test parameters, numeric values, units, and reference ranges from this OCR text.
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

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )

        res_text = response.text
        # Clean any accidental markdown code wrappers
        res_text = re.sub(r'```json\s*', '', res_text)
        res_text = re.sub(r'```\s*', '', res_text).strip()
        
        parsed = json.loads(res_text)
        if parsed and parsed.get("tests") and len(parsed["tests"]) > 0:
            print("[OCR Gemini] Successfully extracted test data using gemini-2.5-flash.")
            return parsed

    except Exception as e:
        print(f"[OCR Gemini Error] API call exception: {str(e)}")

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
        parsed_data = {
            "patient": patient_data,
            "tests": tests_data
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2)

    print(f"[OCR] Output written to: {OUTPUT_JSON}")