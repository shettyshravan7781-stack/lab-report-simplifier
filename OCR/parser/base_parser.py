import re

class BaseParser:

    def __init__(self, columns=None):
        """
        columns expected shape from TableDetector:
        {
            'test': [x_min, x_max],
            'value': [x_min, x_max],
            'unit': [x_min, x_max],
            'reference': [x_min, x_max],
            'method': [x_min, x_max]
        }
        """
        self.columns = columns or {}

    def _get_cell_x_mid(self, cell):
        """Returns the horizontal center of a cell bbox."""
        if "box" in cell:
            box = cell["box"]
            if isinstance(box[0], (list, tuple)):
                xs = [pt[0] for pt in box]
                return (min(xs) + max(xs)) / 2.0
            return (box[0] + box[2]) / 2.0
        x = cell.get("x", 0)
        w = cell.get("w", cell.get("width", 0))
        return x + (w / 2.0)

    def _assign_column(self, cell_x):
        """Assigns a cell coordinate to the closest matching column range."""
        if not self.columns:
            return "test"

        best_col = None
        min_distance = float("inf")

        for col_name, bounds in self.columns.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
                continue
            x_min, x_max = bounds[0], bounds[1]

            # Direct hit within bounds
            if x_min <= cell_x <= x_max:
                return col_name

            # Track distance to center of range for nearest match fallback
            col_mid = (x_min + x_max) / 2.0
            dist = abs(cell_x - col_mid)
            if dist < min_distance:
                min_distance = dist
                best_col = col_name

        return best_col or "test"

    def parse_table(self, rows):
        table = []
        table_started = False

        for row in rows:
            if not row:
                continue

            # Sort row cells left-to-right
            sorted_row = sorted(row, key=lambda c: self._get_cell_x_mid(c))
            line = " ".join(str(cell.get("text", "")) for cell in sorted_row).upper()

            # -----------------------------
            # Detect beginning of table
            # -----------------------------
            if any(keyword in line for keyword in [
                "TOTAL", "PROTEIN", "ALBUMIN", "GLOBULIN", "A/G",
                "BILIRUBIN", "SGOT", "AST", "SGPT", "ALT",
                "ALKALINE", "PHOSPHATASE", "GAMMA", "GGT",
                "HEMOGLOBIN", "CHOLESTEROL", "GLUCOSE", "UREA", "CREATININE"
            ]):
                table_started = True

            if not table_started:
                continue

            # -----------------------------
            # Detect end of report
            # -----------------------------
            if any(keyword in line for keyword in [
                "CLINICAL NOTES", "END OF REPORT", "SPECIMEN",
                "COMMENT", "INTERPRETATION", "NOTE:"
            ]):
                break

            # Ignore table header rows
            if any(word in line for word in ["OBSERVATION", "RESULT", "REFERENCE", "METHOD", "UNIT", "TEST NAME"]):
                continue

            record = {
                "test": "",
                "value": "",
                "unit": "",
                "reference": "",
                "method": ""
            }

            # -----------------------------
            # Assign content to columns dynamically
            # -----------------------------
            for cell in sorted_row:
                text = str(cell.get("text", "")).strip()
                if not text:
                    continue

                cell_x = self._get_cell_x_mid(cell)
                target_col = self._assign_column(cell_x)
                
                if target_col in record:
                    record[target_col] += text + " "

            for key in record:
                record[key] = record[key].strip()

            # -----------------------------
            # Skip noise / empty rows
            # -----------------------------
            if len(record["test"]) < 2:
                continue

            if re.fullmatch(r"[\W_]+", record["test"]):
                continue

            # Skip rows where no value or reference was detected (e.g., section subheadings)
            if not record["value"] and not record["reference"]:
                continue

            # -----------------------------
            # Clean numeric values
            # -----------------------------
            record["value"] = self.clean_number(record["value"])
            record["reference"] = record["reference"].replace("†", "-")

            table.append(record)

        return table

    def clean_number(self, value):
        value = value.replace(",", ".")
        value = value.strip()

        match = re.search(r"-?\d+(?:\.\d+)?", value)

        if match:
            return match.group()

        return value