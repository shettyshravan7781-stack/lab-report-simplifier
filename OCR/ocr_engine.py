import os
import math

# Force disable oneDNN and PIR engine flags before paddle imports
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError("PaddleOCR is required. Install via `pip install paddleocr`.")

# Safe import for config settings
try:
    from OCR.config import LANGUAGE, MIN_CONFIDENCE
except ImportError:
    try:
        from config import LANGUAGE, MIN_CONFIDENCE
    except ImportError:
        LANGUAGE = 'en'
        MIN_CONFIDENCE = 0.50

# Internal Module Imports with fallback logic
try:
    from OCR.parser.PatientParser import PatientParser
    from OCR.layout.table_detector import TableDetector
    from OCR.parser.base_parser import BaseParser
except ImportError:
    from parser.PatientParser import PatientParser
    from layout.table_detector import TableDetector
    from parser.base_parser import BaseParser


class OCREngine:
    """
    Core engine that interfaces with PaddleOCR to extract text boxes,
    group them into logical horizontal rows, and execute baseline report parsing.
    """

    def __init__(self):
        self.ocr = PaddleOCR(
            lang=LANGUAGE,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False  # Kills oneDNN static graph runtime crash
        )
        self.patient_parser = PatientParser()
        self.table_detector = TableDetector()

    def extract_boxes(self, image):
        """
        Runs PaddleOCR and extracts normalized bounding box dictionaries.
        Supports both traditional PaddleOCR lists and newer predict dict structures.
        """
        extracted_data = []

        # Execute OCR inference without outdated cls keyword argument
        try:
            results = self.ocr.ocr(image)
        except Exception:
            results = self.ocr.predict(image)

        if not results:
            return extracted_data

        # Parse PaddleOCR list format: [[box, (text, score)], ...]
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            for line in results[0]:
                if not line or len(line) < 2:
                    continue
                box_pts, text_info = line[0], line[1]
                text, score = text_info[0], float(text_info[1])

                if score < MIN_CONFIDENCE or not text.strip():
                    continue

                xs = [pt[0] for pt in box_pts]
                ys = [pt[1] for pt in box_pts]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

                extracted_data.append({
                    "text": text.strip(),
                    "confidence": float(score),
                    "box": [x_min, y_min, x_max, y_max],
                    "x": x_min,
                    "y": y_min,
                    "w": x_max - x_min,
                    "h": y_max - y_min,
                    "cy": y_min + (y_max - y_min) / 2.0  # Vertical center point
                })

        # Parse PaddleOCR dict format: {"rec_boxes": ..., "rec_texts": ..., "rec_scores": ...}
        elif isinstance(results, list):
            for page in results:
                if not isinstance(page, dict):
                    continue
                boxes = page.get("rec_boxes", [])
                texts = page.get("rec_texts", [])
                scores = page.get("rec_scores", [])

                for box, text, score in zip(boxes, texts, scores):
                    score = float(score)
                    if score < MIN_CONFIDENCE or not str(text).strip():
                        continue

                    if hasattr(box, "tolist"):
                        box = box.tolist()

                    if len(box) == 4 and not isinstance(box[0], (list, tuple)):
                        x_min, y_min, x_max, y_max = box
                    else:
                        xs = [pt[0] for pt in box]
                        ys = [pt[1] for pt in box]
                        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

                    extracted_data.append({
                        "text": str(text).strip(),
                        "confidence": score,
                        "box": [x_min, y_min, x_max, y_max],
                        "x": x_min,
                        "y": y_min,
                        "w": x_max - x_min,
                        "h": y_max - y_min,
                        "cy": y_min + (y_max - y_min) / 2.0
                    })

        return extracted_data

    def group_into_rows(self, boxes, y_threshold=12):
        """
        Groups bounding boxes into horizontal line rows using vertical center points.
        Boxes within `y_threshold` pixels are placed in the same row and sorted left-to-right.
        """
        if not boxes:
            return []

        # Sort all boxes primarily by vertical position (Y centroid)
        sorted_boxes = sorted(boxes, key=lambda b: b.get("cy", b["y"]))
        rows = []
        current_row = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            prev_cy = current_row[-1].get("cy", current_row[-1]["y"])
            curr_cy = box.get("cy", box["y"])

            if abs(curr_cy - prev_cy) <= y_threshold:
                current_row.append(box)
            else:
                current_row.sort(key=lambda b: b["x"])
                rows.append(current_row)
                current_row = [box]

        if current_row:
            current_row.sort(key=lambda b: b["x"])
            rows.append(current_row)

        return rows

    def process_report(self, image):
        """
        Runs the standard pipeline:
        Image -> Bounding Boxes -> Horizontal Rows -> Metadata & Table Parsing.
        """
        raw_boxes = self.extract_boxes(image)
        rows = self.group_into_rows(raw_boxes)

        patient_info = self.patient_parser.parse(rows) if hasattr(self.patient_parser, "parse") else {}
        column_bounds = self.table_detector.detect(rows) if hasattr(self.table_detector, "detect") else {}

        table_parser = BaseParser(columns=column_bounds)
        test_results = table_parser.parse_table(rows) if hasattr(table_parser, "parse_table") else []

        return {
            "patient": patient_info,
            "lab_tests": test_results,
            "detected_columns": column_bounds,
            "total_tests_found": len(test_results)
        }