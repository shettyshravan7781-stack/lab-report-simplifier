import os
import json
from OCR.knowledge.medical_engine import MedicalEngine

# Sample extracted OCR test rows mimicking PaddleOCR / table extraction output
MOCK_EXTRACTED_RAW_TESTS = [
    {"test": "Total Protein", "value": "7.2", "unit": "g/dL", "reference": "6.5-7.8", "method": "Biuret, No Serum Blank"},
    {"test": "Albumin", "value": "3.9", "unit": "g/dL", "reference": "3.9 - 5.0", "method": "Bromocresol Green"},
    {"test": "Globulin", "value": "3.3", "unit": "gm/dL", "reference": "2.0-3.5", "method": "Calculated"},
    {"test": "A/G Ratio", "value": "1.18", "unit": "Ratio", "reference": "1.5-2.5", "method": "Calculated"},
    {"test": "Total Bilirubin", "value": "0.73", "unit": "mg/dL", "reference": "0.2-1.3", "method": "Azobilirubin/dyphylline"},
    {"test": "Conjugated Bilirubin", "value": "0.35", "unit": "mg/dL", "reference": "<0.3", "method": "Calculated"},
    {"test": "Unconjugated Bilirubin", "value": "0.38", "unit": "mg/dL", "reference": "<1.1", "method": "Spectrophotometry"},
    {"test": "SGOT (AST)", "value": "37", "unit": "U/L", "reference": "18-34", "method": "Enzymatic Colorimetric"},
    {"test": "SGPT (ALT)", "value": "25", "unit": "U/L", "reference": "4-35", "method": "UV with P5P"},
    {"test": "SGOT/SGPT Ratio", "value": "1.48", "unit": "Ratio", "reference": "", "method": "Calculated"},
    {"test": "Alkaline Phosphatase", "value": "154", "unit": "U/L", "reference": "46 - 122", "method": "PNPP, AMP buffer"},
    {"test": "Gamma Glutamyl Transferase", "value": "153", "unit": "U/L", "reference": "10 - 54", "method": "G-glutamyl-p-nitroanilide"}
]

def run_pipeline():
    print("========================================")
    print("    LAB REPORT AI PIPELINE RUNNER      ")
    print("========================================")
    
    patient_info = {
        "name": "Demo Patient Name",
        "age": 52,
        "gender": "Female",
        "uhid": "UHID",
        "date": "20-Jan-25",
        "lab_name": "H·O·D"
    }

    engine = MedicalEngine()
    standardized_tests = []

    print("[1/2] Standardizing and normalizing test data...")
    for item in MOCK_EXTRACTED_RAW_TESTS:
        clean_entry = engine.standardize(item)
        standardized_tests.append(clean_entry)

    print("[2/2] Classifying report panel type...")
    detected_report_type = engine.classify_report_type(standardized_tests)

    final_payload = {
        "report_type": detected_report_type,
        "patient": patient_info,
        "tests": standardized_tests,
        "total_tests": len(standardized_tests)
    }

    print("\n========== FINAL OUTPUT JSON ==========\n")
    print(json.dumps(final_payload, indent=4))

    # Export to output file directory
    output_dir = os.path.join("OCR", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, "report_output.json")
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=4)
        
    print(f"\n[SUCCESS] Exported report output to: {output_filepath}")

if __name__ == "__main__":
    run_pipeline()