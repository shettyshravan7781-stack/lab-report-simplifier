class ReportClassifier:

    def classify(self, rows):

        text = " ".join(
            item["text"].upper()
            for row in rows
            for item in row
        )

        # CBC
        if (
            "COMPLETE BLOOD COUNT" in text
            or "CBC" in text
            or "HAEMOGLOBIN" in text
            or "HEMOGLOBIN" in text
            or "TOTAL LEUKOCYTE COUNT" in text
        ):
            return "CBC"

        # LFT
        if (
            "LIVER FUNCTION TEST" in text
            or "LIVER PROFILE" in text
            or "SGOT" in text
            or "SGPT" in text
            or "AST" in text
            or "ALT" in text
            or "BILIRUBIN" in text
        ):
            return "LFT"

        # KFT
        if (
            "KIDNEY FUNCTION TEST" in text
            or "KFT" in text
            or "CREATININE" in text
            or "UREA" in text
            or "URIC ACID" in text
        ):
            return "KFT"

        # Lipid Profile
        if (
            "LIPID PROFILE" in text
            or "CHOLESTEROL" in text
            or "HDL" in text
            or "LDL" in text
            or "TRIGLYCERIDES" in text
        ):
            return "Lipid Profile"

        # Thyroid
        if (
            "THYROID" in text
            or "TSH" in text
            or "T3" in text
            or "T4" in text
        ):
            return "Thyroid"

        # HbA1c
        if (
            "HBA1C" in text
            or "GLYCATED HEMOGLOBIN" in text
        ):
            return "HbA1c"

        # Urine
        if (
            "URINE" in text
            or "URINE ROUTINE" in text
            or "PUS CELLS" in text
            or "EPITHELIAL CELLS" in text
        ):
            return "Urine Routine"

        return "Unknown"