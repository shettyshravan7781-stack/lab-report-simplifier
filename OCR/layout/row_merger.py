"""
OCR/layout/row_merger.py
Groups OCR boxes into horizontal lines using precise vertical Y-centroids.
"""

class RowMerger:

    def __init__(self, y_threshold=6):
        self.y_threshold = y_threshold

    def merge_rows(self, boxes):
        if not boxes:
            return []

        # Add Y-centroid for precise line checking
        normalized_boxes = []
        for box in boxes:
            b = dict(box)
            if "cy" not in b:
                b["cy"] = b.get("y", 0) + (b.get("h", 10) / 2.0)
            normalized_boxes.append(b)

        # Sort top-to-bottom
        sorted_boxes = sorted(normalized_boxes, key=lambda b: b["cy"])

        rows = []
        current_row = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            prev_cy = current_row[-1]["cy"]
            curr_cy = box["cy"]

            # Keep items in same row if within strict Y distance
            if abs(curr_cy - prev_cy) <= self.y_threshold:
                current_row.append(box)
            else:
                # Sort line items left-to-right
                current_row.sort(key=lambda b: b.get("x", 0))
                rows.append(current_row)
                current_row = [box]

        if current_row:
            current_row.sort(key=lambda b: b.get("x", 0))
            rows.append(current_row)

        return rows