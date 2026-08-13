"""
OCR/layout/table_detector.py
Detects horizontal column bounds for test table columns.
"""

class TableDetector:

    HEADER_KEYWORDS = [
        "OBSERVATION", "TEST", "PARAMETER", "INVESTIGATION", 
        "RESULT", "VALUE", "BIOLOGICAL", "REFERENCE", "UNITS", "METHOD"
    ]

    def detect(self, rows):
        """
        Calculates column X ranges based on OCR boxes below the header.
        """
        if not rows:
            return self._default_bounds()

        table_rows = []
        table_started = False

        for row in rows:
            row_text = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item) 
                for item in row
            ).upper()

            if any(kw in row_text for kw in self.HEADER_KEYWORDS):
                table_started = True
                continue

            if table_started:
                if any(kw in row_text for kw in ["CLINICAL NOTES", "END OF REPORT", "INTERPRETATION", "NOTE:"]):
                    break
                if len(row) > 0:
                    table_rows.append(row)

        if not table_rows:
            table_rows = rows

        # Calculate bounding width across rows
        x_min_all = []
        x_max_all = []

        for row in table_rows:
            for item in row:
                if isinstance(item, dict):
                    x = item.get("x", 0)
                    w = item.get("w", 0)
                    x_min_all.append(x)
                    x_max_all.append(x + w)

        if not x_min_all or not x_max_all:
            return self._default_bounds()

        min_x = min(x_min_all)
        max_x = max(x_max_all)
        width = max_x - min_x

        # Proportional column layout (Test | Value | Unit | Reference | Method)
        return {
            "test": [min_x, min_x + int(width * 0.42)],
            "value": [min_x + int(width * 0.42), min_x + int(width * 0.58)],
            "unit": [min_x + int(width * 0.58), min_x + int(width * 0.70)],
            "reference": [min_x + int(width * 0.70), min_x + int(width * 0.85)],
            "method": [min_x + int(width * 0.85), max_x]
        }

    def _default_bounds(self):
        return {
            "test": [0, 260],
            "value": [260, 360],
            "unit": [360, 440],
            "reference": [440, 560],
            "method": [560, 800]
        }