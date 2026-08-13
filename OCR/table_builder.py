"""
TableBuilder converts raw OCR bounding box dictionaries into 2D horizontal rows 
and structured table cells based on spatial coordinates.
"""

class TableBuilder:
    """
    Groups OCR bounding boxes into structured horizontal line rows 
    sorted left-to-right and maps them to vertical table columns.
    """

    def __init__(self, y_threshold=15):
        self.y_threshold = y_threshold

    def build(self, parsed_data):
        """
        Groups bounding boxes into horizontal line rows.

        Args:
            parsed_data (list): List of dicts containing 'box' coordinates or 'x','y','w','h'.

        Returns:
            list: 2D list where each element is a row of left-to-right sorted text boxes.
        """
        if not parsed_data:
            return []

        # -----------------------------------------------------------------
        # 1. Standardize and Pre-Sort All Items Vertically by Y-Centroid
        # -----------------------------------------------------------------
        normalized_items = []
        for item in parsed_data:
            box = item.get("box", [])
            if len(box) >= 4:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cy = y1 + (y2 - y1) / 2.0
            else:
                x1 = item.get("x", 0)
                y1 = item.get("y", 0)
                h = item.get("h", 10)
                cy = y1 + h / 2.0

            item_copy = dict(item)
            item_copy["_x"] = x1
            item_copy["_cy"] = cy
            normalized_items.append(item_copy)

        # Essential step: Pre-sort all items strictly top-to-bottom
        normalized_items.sort(key=lambda item: item["_cy"])

        # -----------------------------------------------------------------
        # 2. Cluster into Horizontal Line Rows
        # -----------------------------------------------------------------
        rows = []
        current_row = []
        running_y_sum = 0.0

        for item in normalized_items:
            cy = item["_cy"]

            if not current_row:
                current_row.append(item)
                running_y_sum = cy
                continue

            # Compare against current row's average Y centroid
            avg_y = running_y_sum / len(current_row)

            if abs(cy - avg_y) <= self.y_threshold:
                current_row.append(item)
                running_y_sum += cy
            else:
                # Sort completed row left-to-right by X coordinate
                current_row.sort(key=lambda x: x["_x"])
                rows.append(self._clean_row(current_row))

                # Start new row
                current_row = [item]
                running_y_sum = cy

        # Add remaining trailing row
        if current_row:
            current_row.sort(key=lambda x: x["_x"])
            rows.append(self._clean_row(current_row))

        return rows

    def _clean_row(self, row):
        """Removes internal temporary sorting keys before returning."""
        cleaned = []
        for item in row:
            item_dict = dict(item)
            item_dict.pop("_x", None)
            item_dict.pop("_cy", None)
            cleaned.append(item_dict)
        return cleaned

    def build_grid(self, rows, column_bounds):
        """
        Maps horizontal text rows into structured dict columns based on 
        detected x-coordinate column boundaries.

        Args:
            rows (list): 2D list of horizontal text boxes.
            column_bounds (dict): E.g., {"test_name": [0, 250], "result": [250, 400], ...}

        Returns:
            list: List of row dictionaries with key-value mapped columns.
        """
        structured_table = []

        for row in rows:
            grid_row = {col_name: [] for col_name in column_bounds.keys()}

            for item in row:
                box = item.get("box", [item.get("x", 0), item.get("y", 0), 0, 0])
                x_center = box[0] + (box[2] - box[0]) / 2.0 if len(box) >= 4 else box[0]

                # Assign item to column whose x-boundaries encompass x_center
                assigned = False
                for col_name, (x_min, x_max) in column_bounds.items():
                    if x_min <= x_center <= x_max:
                        grid_row[col_name].append(item.get("text", "").strip())
                        assigned = True
                        break

                if not assigned and column_bounds:
                    # Fallback to closest column
                    closest_col = min(
                        column_bounds.keys(),
                        key=lambda k: abs(x_center - (column_bounds[k][0] + column_bounds[k][1]) / 2.0)
                    )
                    grid_row[closest_col].append(item.get("text", "").strip())

            # Join cell tokens into readable strings
            row_dict = {col: " ".join(tokens).strip() for col, tokens in grid_row.items()}
            if any(row_dict.values()):
                structured_table.append(row_dict)

        return structured_table