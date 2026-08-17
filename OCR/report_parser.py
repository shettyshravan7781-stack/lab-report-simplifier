import os

# Disable oneDNN CPU graph flags before importing engine modules
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import sys
import glob
import json
import pymupdf  # PyMuPDF for automatic PDF reading

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from OCR.ocr_engine import OCREngine
from OCR.parser.parser_factory import ParserFactory

def process_latest_report():
    image_dir = os.path.join(PROJECT_ROOT, "OCR", "images")
    output_json_path = os.path.join(PROJECT_ROOT, "OCR", "output", "report_output.json")

    # Locate image and PDF files in OCR/images
    valid_extensions = ("*.pdf", "*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in valid_extensions:
        files.extend(glob.glob(os.path.join(image_dir, ext)))

    # Ignore internal temporary files
    files = [f for f in files if not os.path.basename(f).startswith("_temp_")]

    if not files:
        print(f"❌ Error: No image or PDF files found in {image_dir}")
        return

    latest_file = max(files, key=os.path.getmtime)
    print(f"\n📄 Processing latest report file: {os.path.basename(latest_file)}")

    target_image_path = latest_file

    if latest_file.lower().endswith(".pdf"):
        print("⚡ PDF detected! Automatically converting to image in background...")
        doc = pymupdf.open(latest_file)
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        temp_img_path = os.path.join(image_dir, "_temp_ocr_page.png")
        pix.save(temp_img_path)
        target_image_path = temp_img_path

    # Step 1: Execute OCR engine to get bounding boxes and rows
    ocr = OCREngine()
    ocr_results = ocr.process_report(target_image_path)

    # Step 2: Extract report type and column bounds from OCR output
    report_type = ocr_results.get("patient", {}).get("report_type", "CBC")
    columns = ocr_results.get("detected_columns", {})

    # Step 3: Instantiate parser using ParserFactory
    try:
        parser = ParserFactory.get_parser(report_type, columns)
        rows = ocr.group_into_rows(ocr.extract_boxes(target_image_path))
        parsed_lab_tests = parser.parse_table(rows) if hasattr(parser, "parse_table") else parser.parse(rows)
        
        parsed_data = {
            "patient": ocr_results.get("patient", {}),
            "lab_tests": parsed_lab_tests,
            "total_tests_found": len(parsed_lab_tests)
        }
    except Exception as e:
        print(f"⚠️ Factory parse warning ({e}). Falling back to baseline OCR results.")
        parsed_data = ocr_results

    # Step 4: Write finalized structured JSON output
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=4)

    temp_file = os.path.join(image_dir, "_temp_ocr_page.png")
    if os.path.exists(temp_file):
        os.remove(temp_file)

    print(f"✅ OCR extraction complete! Saved to report_output.json")

if __name__ == "__main__":
    process_latest_report()