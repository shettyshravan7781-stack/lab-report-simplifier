import os
import sys
from pathlib import Path

# Fix module lookup paths dynamically
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

for path_item in [str(PROJECT_ROOT), str(CURRENT_DIR)]:
    if path_item not in sys.path:
        sys.path.append(path_item)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

# Safe Module Imports with Relative Fallbacks
try:
    from preprocess import ImagePreprocessor
    from ocr_engine import OCREngine
    from layout.layout_parser import LayoutParser
    from layout.row_merger import RowMerger
    from layout.table_detector import TableDetector
    from classifier.report_classifier import ReportClassifier
    from parser.PatientParser import PatientParser
    from parser.parser_factory import ParserFactory
    from knowledge.medical_engine import MedicalEngine
    from validator.validation_engine import ValidationEngine
except ImportError:
    from OCR.preprocess import ImagePreprocessor
    from OCR.ocr_engine import OCREngine
    from OCR.layout.layout_parser import LayoutParser
    from OCR.layout.row_merger import RowMerger
    from OCR.layout.table_detector import TableDetector
    from OCR.classifier.report_classifier import ReportClassifier
    from OCR.parser.PatientParser import PatientParser
    from OCR.parser.parser_factory import ParserFactory
    from OCR.knowledge.medical_engine import MedicalEngine
    from OCR.validator.validation_engine import ValidationEngine

# Initialize FastAPI Application
app = FastAPI(
    title="LabReportAI API",
    description="Automated Extraction, Normalization, and Medical Simplifier Engine for Diagnostic Blood Reports",
    version="1.0.0"
)

# Enable CORS for Frontend/Client Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pipeline Singletons
ocr_engine = OCREngine()
preprocessor = ImagePreprocessor()
layout_parser = LayoutParser()
row_merger = RowMerger()
table_detector = TableDetector()
classifier = ReportClassifier()
patient_parser = PatientParser()
medical_engine = MedicalEngine()
validation_engine = ValidationEngine()


def generate_patient_summary(report_type: str, patient: dict, tests: list) -> dict:
    """Converts extracted structured lab data into clear, patient-friendly insights."""
    abnormal_tests = [
        t for t in tests 
        if str(t.get("flag", "NORMAL")).upper() in ["HIGH", "LOW", "ABNORMAL"]
    ]
    
    # 1. Overall Status Summary
    if not abnormal_tests:
        overall_status = "All analyzed parameters appear to be within normal reference ranges."
    else:
        abnormal_names = ", ".join([t.get("test", t.get("standard_test_name", "Parameter")) for t in abnormal_tests])
        overall_status = f"Attention recommended: The following parameter(s) are outside expected reference ranges: [{abnormal_names}]."

    # 2. Plain-English Explanations per Test
    simplified_tests = []
    for test in tests:
        name = test.get("test") or test.get("standard_test_name") or "Unknown Test"
        val = test.get("value", "N/A")
        unit = test.get("unit", "")
        flag = test.get("flag", "NORMAL").upper()
        desc = test.get("description", "A standard clinical lab measurement.")

        display_val = f"{val} {unit}".strip()

        # Custom medical explanation mapping
        if "AST/ALT RATIO" in name.upper() or "SGOT/SGPT RATIO" in name.upper():
            explanation = f"Your AST/ALT Ratio is {display_val}. Doctors analyze this proportion to evaluate overall liver health."
        elif "AST" in name.upper() or "SGOT" in name.upper():
            explanation = f"Your AST level is {display_val} ({flag}). AST is an enzyme synthesized in the liver; levels within range indicate absence of acute hepatic cellular stress."
        elif "ALT" in name.upper() or "SGPT" in name.upper():
            explanation = f"Your ALT level is {display_val} ({flag}). ALT is a key liver-specific enzyme used to measure liver cell integrity."
        elif "TOTAL BILIRUBIN" in name.upper():
            explanation = f"Your Total Bilirubin level is {display_val} ({flag}). Bilirubin measures red blood cell breakdown and bile processing efficiency."
        elif "CREATININE" in name.upper():
            explanation = f"Your Serum Creatinine level is {display_val} ({flag}). Creatinine is a waste product filtered by the kidneys, indicating renal filtration efficiency."
        elif "HBA1C" in name.upper():
            explanation = f"Your HbA1c level is {display_val} ({flag}). HbA1c reflects average blood sugar control over the past 2 to 3 months."
        else:
            explanation = f"Your {name} level is {display_val} ({flag}). {desc}"

        simplified_tests.append({
            "test_name": name,
            "result_value": display_val,
            "status": flag,
            "plain_english_explanation": explanation
        })

    # 3. Actionable Clinical Advice
    if abnormal_tests:
        advice = "Please consult a registered healthcare provider or physician to clinically correlate these out-of-range parameters."
    else:
        advice = "Maintain routine health monitoring and consult your physician for periodic evaluation."

    return {
        "overall_health_summary": overall_status,
        "simplified_results": simplified_tests,
        "recommendation": advice
    }


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "LabReportAI Engine API",
        "supported_formats": ["PNG", "JPG", "JPEG", "TIFF", "BMP"]
    }


@app.post("/api/v1/analyze")
async def analyze_report(file: UploadFile = File(...)):
    """Uploads a diagnostic report image, extracts tabular data via OCR, and normalizes output."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload a clear image (PNG, JPG, JPEG, TIFF, BMP)."
        )

    temp_dir = CURRENT_DIR / "output"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        # Read and decode uploaded file into OpenCV format
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image file.")

        # Save temporarily for preprocessing
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)

        # Step 1: Preprocessing
        try:
            processed_image = preprocessor.preprocess(str(temp_path))
        except Exception:
            processed_image = image

        # Step 2: OCR Bounding Box Extraction & Row Alignment
        ocr_result = ocr_engine.extract_boxes(processed_image)
        rows = ocr_engine.group_into_rows(ocr_result)
        
        if hasattr(row_merger, 'merge'):
            rows = row_merger.merge(rows)

        # Step 3: Table Layout & Profile Classification
        columns = table_detector.detect(rows)

        report_type = "GENERAL"
        if hasattr(classifier, 'classify'):
            report_type = classifier.classify(rows)

        # Step 4: Patient Metadata & Table Parsing
        patient = patient_parser.parse(rows)

        parser = ParserFactory.get_parser(report_type, columns)
        if hasattr(parser, 'parse'):
            tests = parser.parse(rows)
        elif hasattr(parser, 'parse_table'):
            tests = parser.parse_table(rows)
        else:
            from parser.base_parser import BaseParser
            tests = BaseParser(columns=columns).parse_table(rows)

        # Step 5: Standardization & Flag Validation
        if isinstance(tests, list):
            standardized_tests = [
                medical_engine.standardize(t) if not isinstance(t, dict) or "flag" not in t else t 
                for t in tests
            ]
        else:
            standardized_tests = [medical_engine.standardize(tests)]

        if hasattr(validation_engine, 'validate'):
            standardized_tests = validation_engine.validate(standardized_tests)

        # Step 6: Patient-Friendly Simplification Engine
        simplification = generate_patient_summary(report_type, patient, standardized_tests)

        return {
            "success": True,
            "filename": file.filename,
            "report_type": report_type,
            "patient": patient,
            "detected_columns": columns,
            "raw_tests": standardized_tests,
            "total_tests": len(standardized_tests),
            "patient_friendly_summary": simplification
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Report processing failed: {str(e)}"
        )

    finally:
        # Cleanup temporary files safely
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass