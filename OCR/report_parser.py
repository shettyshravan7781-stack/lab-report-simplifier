import os
import sys
import re
import json
import warnings
import logging
import pymupdf
from dotenv import load_dotenv

# Reconfigure stdout/stderr for Unicode support on Windows terminals
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress SDK warnings and internal loggers
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore")

for logger_name in ["google", "google.genai", "google_genai", "google_genai.models", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "OCR", "output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "report_output.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Section boundary markers where biomarker extraction MUST stop
CUTOFF_KEYWORDS = [
    "NOTE:", "NOTES:", "COMMENTS:", "COMMENT:", "INTERPRETATION:",
    "CLINICAL INTERPRETATION", "CLINICAL SIGNIFICANCE:", 
    "RECOMMENDED THERAPEUTIC RANGE", "CLASSIFICATION OF", 
    "REMARKS:", "IN CASE OF ANY UNEXPECTED", "DISCLAIMER",
    "DR.", "MD, PATHOLOGIST", "THANKS FOR REFERENCE", "****END OF REPORT****",
    "PARTIAL THROMBOPLASTIN TIME", "PROTHROMBIN TIME MEASURES",
    "DEGREE OF PROLONGATION", "RESULTS SHOULD BE CLINICALLY",
    "MEDICAL LAB TECHNICIAN", "TO CHECK REPORT AUTHENTICITY",
    "ADA DIAGNOSTIC", "PREDIABETES", "DIAGNOSTIC CRITERIA", "LOINC:", "DATASET PANEL"
]
STOP_SECTION_KEYWORDS = CUTOFF_KEYWORDS

# Keywords representing non-test guidance, footers, tables, or meta-comments to discard
IGNORE_KEYWORDS = [
    "ADA DIAGNOSTIC", "PREDIABETES", "DIAGNOSTIC CRITERIA", "LOINC:", 
    "DATASET PANEL", "CLASSIFICATION OF", "REFERENCE INTERVAL", 
    "BIOLOGICAL REFERENCE", "METHOD:", "SAMPLE TYPE:"
]

TABLE_HEADER_KEYWORDS = [
    "INVESTIGATION", "OBSERVATION", "TEST NAME", "PARAMETER", 
    "ANALYTE", "TEST DESCRIPTION", "DESCRIPTION"
]

METHOD_NOISE_TERMS = [
    "Ivy's Method", "Lee & White", "Photo optical Clot Detection",
    "Calculated", "Urease, Colorimetric", "Enzymatic", "Arsenazo III",
    "Phosphomolybdate reduction", "ISE Direct", "Calculated By CKD-EPI(2021)",
    "Calculated By CKD-EPI", "Spectrophotometry", "Immunoturbidimetry",
    "Chemiluminescence", "Dry Chemistry", "Hexokinase", "GOD-POD",
    "Jaffe's Method", "Biuret Method", "Bromocresol Green", "Direct ISE"
]


def parse_demographics_spatial(doc):
    """
    Extracts Patient Name, Age, and Gender using structural layout
    anchors and multi-line aware regex boundaries.
    """
    patient_info = {"name": "Unknown Patient", "age": "N/A", "gender": "N/A"}
    p0 = doc[0]
    page_text = p0.get_text("text")
    blocks = p0.get_text("blocks")

    # 1. Patient Name Extraction
    name_found = False

    # Truncate footer and disclaimer zones to isolate header demographics
    ignore_footers = [
        r"\bNote\s*:", r"\bInterpretation\s*:", r"\bDisclaimer\s*:", 
        r"\bThanks\s+for\s+Reference\b", r"\bDr\.\s+[A-Za-z]+", 
        r"\bPage\s+\d+\s+of\b", r"\bEnd\s+of\s+Report\b"
    ]
    header_text = page_text
    for pat in ignore_footers:
        m = re.search(pat, header_text, re.IGNORECASE)
        if m:
            header_text = header_text[:m.start()]

    # Labeled pattern with explicit word boundaries
    name_m = re.search(
        r"\b(?:Patient\s*Name|Client\s*Name|Pt\.?\s*Name|Patient|Client)\b\s*[:\-]?\s*(?:\n\s*)?([^\n\r]+)",
        header_text,
        re.IGNORECASE
    )
    if name_m:
        cand = name_m.group(1).strip()
        cand = re.sub(r"^(?:Mr|Mrs|Ms|Miss|Master|Dr|Pt|Shri|Smt)\.?\s+", "", cand, flags=re.IGNORECASE).strip()
        cand = re.split(
            r"\b(?:Lab|UHID|ID|Age|Sex|Gender|Date|Ref|Dr|Patient\s*ID|Center|Centre|Visit|Referred)\b", 
            cand, 
            flags=re.IGNORECASE
        )[0].strip()
        # Clean trailing colon or punctuation
        cand = re.sub(r"^[\s:\-]+|[\s:\-]+$", "", cand).strip()
        invalid_starts = r"^(?:s\s+|on\s+|id\s+|referred\s*by\b)"
        if (len(cand) >= 2 and 
            not re.search(invalid_starts, cand, re.IGNORECASE) and
            not re.match(r"^(?:UNKNOWN|PATIENT(?:\s*NAME)?|CLIENT(?:\s*NAME)?)$", cand, re.IGNORECASE)):
            patient_info["name"] = cand
            name_found = True

    # Drlogy-style / Header Block fallback (Name printed as standalone heading before demographics)
    if not name_found:
        for b in blocks[:6]:
            btxt = b[4].strip()
            lines = [l.strip() for l in btxt.split("\n") if l.strip()]
            if not lines:
                continue
            first_line = lines[0]
            if any(kw in first_line.upper() for kw in [
                "LAB", "PATHOLOGY", "DIAGNOSTIC", "ROAD", "MUMBAI", "COMPLEX", 
                "SMART", "PHONE", "WWW", "DRLOGY", "ACCURATE", "SAMPLE", "CLINICAL"
            ]):
                continue
            words = first_line.split()
            if 2 <= len(words) <= 5 and all(w[0].isupper() or w.endswith(".") for w in words):
                if not any(kw in first_line.upper() for kw in [
                    "REPORT", "INVESTIGATION", "PROFILE", "TEST", "NORMAL", "COLLECTION", "BLOOD", "URINE"
                ]):
                    patient_info["name"] = first_line
                    name_found = True
                    break

    # 2. Age and Gender Extraction
    # Pattern A: Inline combined "Age / Sex : 66 Y / M" or "Age / Gender: 21 Years / Female"
    inline_m = re.search(
        r"(?:Age\s*[/\\&]?\s*(?:Sex|Gender)|Sex\s*[/\\&]?\s*Age)\s*[:\-]?\s*(?:\n\s*)?(\d{1,3})\s*(?:Y(?:ears?|rs)?)?\s*[/\\,\-]\s*([A-Za-z]+)",
        page_text,
        re.IGNORECASE
    )
    if inline_m:
        patient_info["age"] = inline_m.group(1).strip()
        g_raw = inline_m.group(2).strip().upper()
        patient_info["gender"] = "Female" if g_raw.startswith("F") else "Male" if g_raw.startswith("M") else "N/A"
    else:
        # Pattern B: Format like "20/Male", "21 Years / Female", "25 Y / F", "66 Y/M"
        inline_m2 = re.search(
            r"\b(\d{1,3})\s*(?:Y(?:ears?|rs)?)?\s*[/\\|\-]\s*(Male|Female|Fem|M|F)\b", 
            page_text, 
            re.IGNORECASE
        )
        if inline_m2:
            patient_info["age"] = inline_m2.group(1).strip()
            g_raw = inline_m2.group(2).strip().upper()
            patient_info["gender"] = "Female" if g_raw.startswith("F") else "Male" if g_raw.startswith("M") else "N/A"

    # Pattern C: Separate Age field
    if patient_info["age"] == "N/A":
        age_m = re.search(r"\bAge\s*[:\-]?\s*(?:\n\s*)?(\d{1,3})\s*(?:Y(?:ears?|rs)?)?\b", page_text, re.IGNORECASE)
        if age_m:
            patient_info["age"] = age_m.group(1).strip()
        else:
            age_m2 = re.search(r"\b([1-9][0-9]?)\s*(?:Yrs|Years|Y/O)\b", page_text, re.IGNORECASE)
            if age_m2:
                patient_info["age"] = age_m2.group(1).strip()

    # Pattern D: Separate Gender/Sex field
    if patient_info["gender"] == "N/A":
        gen_m = re.search(r"\b(?:Sex|Gender)\s*[:\-]?\s*(?:\n\s*)?(Female|Male|Fem|F|M)\b", page_text, re.IGNORECASE)
        if gen_m:
            g_raw = gen_m.group(1).strip().upper()
            patient_info["gender"] = "Female" if g_raw.startswith("F") else "Male" if g_raw.startswith("M") else "N/A"

    return patient_info


def parse_tests_spatial_layout(doc):
    """
    Layout-aware parser using PyMuPDF word bounding boxes, column boundary
    alignment, and explicit section boundary termination.
    """
    all_tests = []

    for page_idx, page in enumerate(doc):
        words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
        if not words:
            continue

        # Step 1: Detect Table Header Row and Stop Section Boundary
        header_words = []
        header_y0 = None
        header_y1 = None

        words_sorted_y = sorted(words, key=lambda x: (x[1], x[0]))

        for w in words_sorted_y:
            wu = w[4].upper()
            if any(hk in wu for hk in TABLE_HEADER_KEYWORDS) and header_y0 is None:
                line_w = [ow for ow in words if abs(ow[1] - w[1]) < 6.0]
                line_txt = " ".join(ow[4].upper() for ow in sorted(line_w, key=lambda x: x[0]))
                if any(rk in line_txt for rk in ["RESULT", "VALUE", "OBSERVED"]):
                    header_words = line_w
                    header_y0 = min(ow[1] for ow in line_w)
                    header_y1 = max(ow[3] for ow in line_w)
                    break

        if header_y1 is None:
            header_y1 = 160.0

        # Step 2: Detect Stop Boundary below table header
        stop_y0 = page.rect.height
        for w in words_sorted_y:
            if w[1] > header_y1 + 5.0:
                line_w = [ow for ow in words if abs(ow[1] - w[1]) < 5.0]
                line_txt = " ".join(ow[4].upper() for ow in sorted(line_w, key=lambda x: x[0]))
                if any(sk in line_txt for sk in STOP_SECTION_KEYWORDS):
                    if w[1] < stop_y0:
                        stop_y0 = w[1]
                        break

        # Step 3: Determine Column Horizontal Boundaries
        col_result_x = None
        col_ref_x = None
        col_unit_x = None

        if header_words:
            for hw in sorted(header_words, key=lambda x: x[0]):
                hwu = hw[4].upper()
                if any(k in hwu for k in ["RESULT", "VALUE", "OBSERVED"]) and col_result_x is None:
                    col_result_x = hw[0]
                elif any(k in hwu for k in ["REF", "BIOLOGICAL", "NORMAL", "REFERENCE"]) and col_ref_x is None:
                    col_ref_x = hw[0]
                elif "UNIT" in hwu and col_unit_x is None:
                    col_unit_x = hw[0]

        if col_result_x is None:
            col_result_x = 180.0

        # Step 4: Extract table words within the bounded table region
        table_words = [
            w for w in words 
            if (header_y1 + 2.0) <= w[1] < (stop_y0 - 2.0)
            and not (w[4] in ["Report", "Sample"] and (w[2] - w[0]) > 60)  # Filter background watermarks
        ]

        # Group words into physical horizontal rows using vertical tolerance
        lines_map = {}
        for w in sorted(table_words, key=lambda x: (x[1], x[0])):
            found = False
            for ly in sorted(lines_map.keys()):
                if abs(w[1] - ly) < 3.5:
                    lines_map[ly].append(w)
                    found = True
                    break
            if not found:
                lines_map[w[1]] = [w]

        lines = []
        for ly in sorted(lines_map.keys()):
            lw = sorted(lines_map[ly], key=lambda x: x[0])
            l_text = " ".join(w[4] for w in lw)
            # Skip metadata rows and sub-panel dividers
            if any(skip in l_text.upper() for skip in [
                "PRIMARY SAMPLE TYPE", "TAT:", "ELECTROLYTES", "SERUM SAMPLE", "---"
            ]):
                continue
            lines.append({
                "y": ly,
                "words": lw,
                "text": l_text
            })

        # Step 5: Assign cells into columns and assemble multi-line test rows
        tests = []
        current_test = None

        for item in lines:
            param_words = [w[4] for w in item["words"] if w[0] < col_result_x - 10]
            result_words = []
            unit_words = []
            ref_words = []

            for w in item["words"]:
                wx = w[0]
                wt = w[4]
                if wx >= col_result_x - 10:
                    if col_unit_x and col_ref_x and col_unit_x < col_ref_x:
                        # Layout: Parameter | Result | Unit | Reference
                        if wx < col_unit_x - 10:
                            result_words.append(wt)
                        elif wx < col_ref_x - 10:
                            unit_words.append(wt)
                        else:
                            ref_words.append(wt)
                    elif col_ref_x and col_unit_x and col_ref_x < col_unit_x:
                        # Layout: Parameter | Result | Reference | Unit
                        if wx < col_ref_x - 10:
                            result_words.append(wt)
                        elif wx < col_unit_x - 10:
                            ref_words.append(wt)
                        else:
                            unit_words.append(wt)
                    else:
                        boundary = col_ref_x if col_ref_x else (col_unit_x if col_unit_x else 300.0)
                        if wx < boundary - 10:
                            result_words.append(wt)
                        else:
                            ref_words.append(wt)

            res_str = " ".join(result_words).strip()
            num_m = re.search(r"([\d]+\.?[\d]*)", res_str)

            if param_words and num_m:
                # Complete row on a single line
                p_name = " ".join(param_words)
                current_test = {
                    "test": p_name,
                    "value": num_m.group(1),
                    "unit": " ".join(unit_words).strip() or "N/A",
                    "reference": " ".join(ref_words).strip() or "N/A",
                    "y": item["y"]
                }
                tests.append(current_test)

            elif param_words and not num_m:
                p_name = " ".join(param_words)
                METHOD_KEYWORDS = ["METHOD", "DETECTION", "CALCULATED", "COLORIMETRIC", "DIRECT", "ENZYMATIC", "WHITE", "IVY"]
                is_method = any(mk in p_name.upper() for mk in METHOD_KEYWORDS)
                
                # Check if this line continues a multi-line parameter title
                if current_test and current_test.get("value") and not is_method and (item["y"] - current_test.get("y", 0)) < 15.0:
                    current_test["test"] += " " + p_name
                elif not current_test or (current_test and current_test.get("value")):
                    current_test = {
                        "test": p_name,
                        "value": None,
                        "unit": "N/A",
                        "reference": "N/A",
                        "y": item["y"]
                    }
                    tests.append(current_test)
                else:
                    current_test["test"] += " " + p_name

            elif not param_words and num_m:
                # Value row corresponding to the pending parameter name above
                if current_test and current_test["value"] is None:
                    current_test["value"] = num_m.group(1)
                    current_test["unit"] = " ".join(unit_words).strip() or "N/A"
                    current_test["reference"] = " ".join(ref_words).strip() or "N/A"
                    current_test["y"] = item["y"]

        # Step 6: Dynamic Parameter Key and Reference Range Sanitation
        for t in tests:
            if not t.get("value"):
                continue
            t_name = t["test"]
            for mt in METHOD_NOISE_TERMS:
                t_name = re.sub(re.escape(mt), "", t_name, flags=re.IGNORECASE).strip()
            
            # Dynamic strip of leading/trailing digits, symbols, and status flags
            t_name = re.sub(r"^[\d\.\-\s:]+", "", t_name).strip()
            t_name = re.sub(r"[\d\.\-\s:]+$", "", t_name).strip()
            t_name = re.sub(r"\b(Normal|High|Low|Abnormal)\b", "", t_name, flags=re.IGNORECASE).strip()

            t_ref = t["reference"]
            for mt in METHOD_NOISE_TERMS:
                t_ref = re.sub(re.escape(mt), "", t_ref, flags=re.IGNORECASE).strip()
            t_ref = re.sub(r"\b(Normal|High|Low)\b", "", t_ref, flags=re.IGNORECASE).strip()

            if len(t_name) >= 2 and not any(ik in t_name.upper() for ik in IGNORE_KEYWORDS):
                all_tests.append({
                    "test": t_name,
                    "value": t["value"],
                    "unit": t["unit"],
                    "reference": t_ref.strip() or "N/A"
                })

    return all_tests


def parse_report_with_gemini(raw_text):
    """
    LLM-based structured extractor using Google GenAI SDK.
    """
    if not GEMINI_API_KEY:
        return None

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        return None

    prompt = f"""
    You are an expert medical data extraction system. Extract ALL laboratory test parameters, numeric values, units, and reference ranges from this OCR text.
    STOP extracting biomarkers when notes, comments, clinical interpretation, or recommendations start.
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

    candidate_models = [
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-pro-preview"
    ]

    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name, 
                contents=prompt,
                config=types.GenerateContentConfig(tools=[])
            )
            res_text = response.text.strip()
            res_text = re.sub(r"^```json\s*", "", res_text, flags=re.MULTILINE)
            res_text = re.sub(r"^```\s*", "", res_text, flags=re.MULTILINE)
            res_text = res_text.strip("`").strip()

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


def extract_report(pdf_path, use_llm=False):
    """
    Unified extraction pipeline: executes deterministic layout-aware spatial parser.
    Executes 100% locally with zero external network calls by default.
    """
    doc = pymupdf.open(pdf_path)
    patient_data = parse_demographics_spatial(doc)
    spatial_tests = parse_tests_spatial_layout(doc)

    if use_llm and GEMINI_API_KEY:
        try:
            raw_text = "\n".join([page.get_text("text") for page in doc])
            parsed_data = parse_report_with_gemini(raw_text)
            if parsed_data and parsed_data.get("tests"):
                if parsed_data.get("patient", {}).get("name") in ["Unknown Patient", "N/A", None]:
                    parsed_data["patient"] = patient_data
                return parsed_data
        except Exception:
            pass

    print("[OCR] Executing layout-aware spatial bounding-box parser...")
    return {
        "patient": patient_data,
        "tests": spatial_tests
    }


if __name__ == "__main__":
    target_pdf = None
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if os.path.isabs(arg_path) and os.path.exists(arg_path):
            target_pdf = arg_path
        elif os.path.exists(os.path.join(INPUT_DIR, arg_path)):
            target_pdf = os.path.join(INPUT_DIR, arg_path)

    if not target_pdf:
        pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in input directory: {INPUT_DIR}")
        target_pdf = os.path.join(INPUT_DIR, pdf_files[0])

    print(f"[OCR] Target PDF: {target_pdf}")
    # Local deterministic spatial extraction: executes 100% locally with zero network calls
    result = extract_report(target_pdf, use_llm=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[OCR] Output successfully written to: {OUTPUT_JSON}")