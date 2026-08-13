"""
Report section splitter that divides grouped text rows into 
patient header rows, table body rows, and footer rows.
"""

class ReportParser:
    """
    Splits document rows into 3 logical structural regions:
    1. patient: Header rows above the lab report table.
    2. table: Main tabular data containing test parameters and values.
    3. footer: Lab signatures, notes, and disclaimers below the table.
    """

    # Multi-keyword triggers for robust boundary detection
    TABLE_HEADER_KEYWORDS = [
        ("TEST", "RESULT"),
        ("TEST", "VALUE"),
        ("PARAMETER", "VALUE"),
        ("PARAMETER", "RESULT"),
        ("INVESTIGATION", "RESULT"),
        ("TEST NAME", "OBSERVED"),
    ]

    FOOTER_KEYWORDS = [
        "CLINICAL NOTES",
        "END OF REPORT",
        "INTERPRETATION",
        "DR.",
        "PATHOLOGIST",
        "LABORATORY DIRECTOR",
        "THANK YOU FOR REFERRAL",
        "PAGE 1 OF"
    ]

    def __init__(self):
        pass

    def split_sections(self, rows):
        """
        Splits grouped text rows into patient, table, and footer sections.

        Args:
            rows (list): List of text box lists representing horizontal lines.

        Returns:
            dict: {
                "patient": list of patient header rows,
                "table": list of core table rows,
                "footer": list of footer/signature rows
            }
        """
        patient_rows = []
        table_rows = []
        footer_rows = []

        inside_table = False

        for row in rows:
            # Concatenate row items into a single uppercase string for keyword evaluation
            row_text = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in row
            ).upper()

            # -----------------------------------------------------------------
            # 1. Check for Table Start Boundary
            # -----------------------------------------------------------------
            if not inside_table:
                # Check if any pair of header keywords exist in the row
                for kw1, kw2 in self.TABLE_HEADER_KEYWORDS:
                    if kw1 in row_text and kw2 in row_text:
                        inside_table = True
                        break

            # -----------------------------------------------------------------
            # 2. Check for Footer / Exit Boundary
            # -----------------------------------------------------------------
            if inside_table:
                for footer_kw in self.FOOTER_KEYWORDS:
                    if footer_kw in row_text:
                        inside_table = False
                        break

            # -----------------------------------------------------------------
            # 3. Route Row to Appropriate Bucket
            # -----------------------------------------------------------------
            if inside_table:
                table_rows.append(row)
            elif not table_rows:
                patient_rows.append(row)
            else:
                footer_rows.append(row)

        return {
            "patient": patient_rows,
            "table": table_rows,
            "footer": footer_rows
        }