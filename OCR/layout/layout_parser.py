class LayoutParser:

    def __init__(self):
        pass

    def group_rows(self, ocr_data, y_threshold=8):
        """
        Groups OCR words into horizontal rows based on vertical alignment.
        Uses the average Y position of the current row to prevent row-drift
        on slightly tilted or slanted text lines.
        """

        if not ocr_data:
            return []

        # Sort top -> bottom by Y position
        ocr_data = sorted(ocr_data, key=lambda x: x["y"])

        rows = []
        current_row = [ocr_data[0]]

        for item in ocr_data[1:]:
            y = item["y"]
            
            # Compare item Y with the running average Y of current_row
            avg_y = sum(w["y"] for w in current_row) / len(current_row)

            if abs(y - avg_y) <= y_threshold:
                current_row.append(item)
            else:
                # Sort words in completed row left -> right by X position
                current_row.sort(key=lambda x: x["x"])
                rows.append(current_row)

                current_row = [item]

        # Append final row sorted left -> right
        if current_row:
            current_row.sort(key=lambda x: x["x"])
            rows.append(current_row)

        return rows