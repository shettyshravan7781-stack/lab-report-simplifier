import re


class MedicalEngine:
    def __init__(self):
        # Specific / longer key strings MUST come before shorter substrings!
        self.canonical_map = [
            # ----------------------------------------------------
            # 1. LIVER FUNCTION TEST (LFT)
            # ----------------------------------------------------
            ("SGOT/SGPT RATIO", "SGOT/SGPT Ratio"),
            ("SGOT /SGPT", "SGOT/SGPT Ratio"),
            ("SOOT /SGIPT", "SGOT/SGPT Ratio"),
            ("A/G RATIO", "Albumin/Globulin Ratio"),
            ("ALBUMIN/GLOBULIN RATIO", "Albumin/Globulin Ratio"),
            ("UNCONJUGATED BILIRUBIN", "Indirect Bilirubin"),
            ("INDIRECT BILIRUBIN", "Indirect Bilirubin"),
            ("CONJUGATED BILIRUBIN", "Direct Bilirubin"),
            ("DIRECT BILIRUBIN", "Direct Bilirubin"),
            ("TOTAL BILIRUBIN", "Total Bilirubin"),
            ("TOTAL PROTEIN", "Total Protein"),
            ("ALBUMIN", "Albumin"),
            ("GLOBULIN", "Globulin"),
            ("SGOT (AST)", "SGOT (AST)"),
            ("SGOT", "SGOT (AST)"),
            ("SOOT", "SGOT (AST)"),
            ("AST", "SGOT (AST)"),
            ("SGPT (ALT)", "SGPT (ALT)"),
            ("SGPT", "SGPT (ALT)"),
            ("ALT", "SGPT (ALT)"),
            ("ALKALINE PHOSPHATASE", "Alkaline Phosphatase (ALP)"),
            ("ALP", "Alkaline Phosphatase (ALP)"),
            ("GAMMA GLUTAMYL TRANSFERASE", "Gamma Glutamyl Transferase (GGT)"),
            ("GPRUMA CHULZIMY", "Gamma Glutamyl Transferase (GGT)"),
            ("GGT", "Gamma Glutamyl Transferase (GGT)"),

            # ----------------------------------------------------
            # 2. KIDNEY FUNCTION TEST (KFT / RFT)
            # ----------------------------------------------------
            ("BLOOD UREA NITROGEN", "Blood Urea Nitrogen (BUN)"),
            ("BUN", "Blood Urea Nitrogen (BUN)"),
            ("UREA", "Blood Urea"),
            ("SERUM CREATININE", "Serum Creatinine"),
            ("CREATININE", "Serum Creatinine"),
            ("URIC ACID", "Uric Acid"),
            ("SODIUM", "Serum Sodium"),
            ("POTASSIUM", "Serum Potassium"),
            ("CHLORIDE", "Serum Chloride"),
            ("BUN/CREATININE RATIO", "BUN/Creatinine Ratio"),

            # ----------------------------------------------------
            # 3. LIPID PROFILE
            # ----------------------------------------------------
            ("TOTAL CHOLESTEROL", "Total Cholesterol"),
            ("CHOLESTEROL", "Total Cholesterol"),
            ("TRIGLYCERIDES", "Triglycerides"),
            ("HDL CHOLESTEROL", "HDL Cholesterol"),
            ("HDL", "HDL Cholesterol"),
            ("LDL CHOLESTEROL", "LDL Cholesterol"),
            ("LDL", "LDL Cholesterol"),
            ("VLDL CHOLESTEROL", "VLDL Cholesterol"),
            ("VLDL", "VLDL Cholesterol"),
            ("NON-HDL CHOLESTEROL", "Non-HDL Cholesterol"),
            ("TOTAL CHOLESTEROL/HDL RATIO", "Total Cholesterol/HDL Ratio"),
            ("LDL/HDL RATIO", "LDL/HDL Ratio"),

            # ----------------------------------------------------
            # 4. COMPLETE BLOOD COUNT (CBC) & PLATELET PROFILE
            # ----------------------------------------------------
            ("HEMOGLOBIN", "Hemoglobin"),
            ("HB", "Hemoglobin"),
            ("TOTAL LEUKOCYTE COUNT", "Total Leukocyte Count (TLC)"),
            ("TOTAL WBC COUNT", "Total Leukocyte Count (TLC)"),
            ("TLC", "Total Leukocyte Count (TLC)"),
            ("WBC", "Total Leukocyte Count (TLC)"),
            ("RED BLOOD CELL COUNT", "Total RBC Count"),
            ("RBC COUNT", "Total RBC Count"),
            ("RBC", "Total RBC Count"),
            ("HEMATOCRIT", "Hematocrit (HCT)"),
            ("PCV", "Packed Cell Volume (PCV)"),
            ("MCV", "Mean Corpuscular Volume (MCV)"),
            ("MCH", "Mean Cell Hemoglobin (MCH)"),
            ("MCHC", "Mean Cell Hemoglobin Concentration (MCHC)"),
            ("RDW-CV", "RDW-CV"),
            ("RDW-SD", "RDW-SD"),
            ("NEUTROPHILS", "Neutrophils"),
            ("LYMPHOCYTES", "Lymphocytes"),
            ("MONOCYTES", "Monocytes"),
            ("EOSINOPHILS", "Eosinophils"),
            ("BASOPHILS", "Basophils"),
            ("PLATELET COUNT", "Platelet Count"),
            ("PLATELETS", "Platelet Count"),
            ("MPV", "Mean Platelet Volume (MPV)"),
            ("PDW", "Platelet Distribution Width (PDW)"),
            ("PCT", "Plateletcrit (PCT)"),

            # ----------------------------------------------------
            # 5. DIABETES MARKERS (HbA1c & GLUCOSE)
            # ----------------------------------------------------
            ("HBA1C", "HbA1c (Glycated Hemoglobin)"),
            ("GLYCOSYLATED HEMOGLOBIN", "HbA1c (Glycated Hemoglobin)"),
            ("AVERAGE BLOOD GLUCOSE", "Estimated Average Glucose (eAG)"),
            ("ESTIMATED AVERAGE GLUCOSE", "Estimated Average Glucose (eAG)"),
            ("FASTING BLOOD SUGAR", "Fasting Blood Glucose"),
            ("FASTING BLOOD GLUCOSE", "Fasting Blood Glucose"),
            ("FASTING GLUCOSE", "Fasting Blood Glucose"),
            ("POSTPRANDIAL BLOOD SUGAR", "Postprandial Blood Glucose"),
            ("POST PRANDIAL GLUCOSE", "Postprandial Blood Glucose"),
            ("PP GLUCOSE", "Postprandial Blood Glucose"),
            ("RANDOM BLOOD SUGAR", "Random Blood Glucose"),

            # ----------------------------------------------------
            # 6. THYROID PROFILE
            # ----------------------------------------------------
            ("THYROID STIMULATING HORMONE", "TSH"),
            ("TSH", "TSH"),
            ("FREE T3", "Free T3 (FT3)"),
            ("FREE T4", "Free T4 (FT4)"),
            ("TOTAL T3", "Total T3"),
            ("TOTAL T4", "Total T4"),
            ("TRIIODOTHYRONINE", "Total T3"),
            ("THYROXINE", "Total T4"),

            # ----------------------------------------------------
            # 7. IRON PROFILE
            # ----------------------------------------------------
            ("SERUM IRON", "Serum Iron"),
            ("TOTAL IRON BINDING CAPACITY", "TIBC"),
            ("TIBC", "TIBC"),
            ("UNSATURATED IRON BINDING CAPACITY", "UIBC"),
            ("UIBC", "UIBC"),
            ("PERCENT TRANSFERRIN SATURATION", "Transferrin Saturation"),
            ("TRANSFERRIN SATURATION", "Transferrin Saturation"),
            ("SERUM FERRITIN", "Serum Ferritin"),
            ("FERRITIN", "Serum Ferritin"),

            # ----------------------------------------------------
            # 8. CALCIUM & PHOSPHORUS
            # ----------------------------------------------------
            ("SERUM CALCIUM", "Serum Calcium"),
            ("CALCIUM", "Serum Calcium"),
            ("SERUM PHOSPHORUS", "Serum Phosphorus"),
            ("PHOSPHORUS", "Serum Phosphorus"),
            ("INORGANIC PHOSPHATE", "Serum Phosphorus"),
            ("IONIZED CALCIUM", "Ionized Calcium"),

            # ----------------------------------------------------
            # 9. URINE ACR (ALBUMIN TO CREATININE RATIO)
            # ----------------------------------------------------
            ("URINE ALBUMIN", "Microalbumin (Urine)"),
            ("MICROALBUMIN", "Microalbumin (Urine)"),
            ("URINE CREATININE", "Urine Creatinine"),
            ("ALBUMIN/CREATININE RATIO", "Urine ACR"),
            ("URINE ACR", "Urine ACR"),

            # ----------------------------------------------------
            # 10. URINE ROUTINE & EXAMINATION
            # ----------------------------------------------------
            ("URINE COLOR", "Urine Color"),
            ("URINE APPEARANCE", "Urine Appearance"),
            ("URINE PH", "Urine pH"),
            ("SPECIFIC GRAVITY", "Specific Gravity"),
            ("URINE PROTEIN", "Urine Protein"),
            ("URINE SUGAR", "Urine Sugar"),
            ("URINE KETONES", "Urine Ketones"),
            ("URINE BILIRUBIN", "Urine Bilirubin"),
            ("UROBILINOGEN", "Urobilinogen"),
            ("URINE NITRITE", "Urine Nitrite"),
            ("PUS CELLS", "Pus Cells (WBCs)"),
            ("EPITHELIAL CELLS", "Epithelial Cells"),
            ("RED BLOOD CELLS", "Urine RBCs"),
            ("CASTS", "Urine Casts"),
            ("CRYSTALS", "Urine Crystals")
        ]

        self.known_units = [
            "g/dL", "gm/dL", "g/dl", "mg/dL", "U/L", "IU/L", "Ratio", "%",
            "mmol/L", "mEq/L", "uIU/mL", "μIU/mL", "ng/dL", "pg/mL",
            "μg/dL", "ug/dL", "10^3/uL", "10^6/uL", "fL", "pg", "mEql/L",
            "mg/g", "mg/mmol", "/hpf", "/lpf", "cumm", "lakhs/cumm", "million/cumm", "Pg", "eport"
        ]

    def compute_flag(self, test_name, value_str, ref_str):
        """
        Evaluates test value against reference range string or fallback clinical safety limits.
        """
        try:
            val = float(value_str)
        except (ValueError, TypeError):
            return "Normal"

        # 1. If reference string exists, parse it normally
        if ref_str:
            upper_match = re.search(r"<\s*(\d+(?:\.\d+)?)", ref_str)
            if upper_match:
                return "High" if val > float(upper_match.group(1)) else "Normal"

            lower_match = re.search(r">\s*(\d+(?:\.\d+)?)", ref_str)
            if lower_match:
                return "Low" if val < float(lower_match.group(1)) else "Normal"

            range_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", ref_str)
            if range_match:
                low_b, high_b = float(range_match.group(1)), float(range_match.group(2))
                if val < low_b: return "Low"
                if val > high_b: return "High"
                return "Normal"

        # 2. SAFETY FALLBACK LIMITS (If reference box was missed by OCR)
        name_upper = test_name.upper()
        if "SGOT" in name_upper or "AST" in name_upper or "SOOT" in name_upper:
            return "High" if val > 40 else "Normal"
        if "SGPT" in name_upper or "ALT" in name_upper:
            return "High" if val > 45 else "Normal"
        if "ALKALINE PHOSPHATASE" in name_upper or "ALP" in name_upper:
            return "High" if val > 125 else "Normal"
        if "BILIRUBIN" in name_upper and "TOTAL" in name_upper:
            return "High" if val > 1.2 else "Normal"
        if "GGT" in name_upper or "GLUTAMYL" in name_upper or "GPRUMA" in name_upper:
            return "High" if val > 55 else "Normal"

        return "Normal"

    def classify_report_type(self, extracted_tests):
        counts = {
            "Liver Function Test (LFT)": 0,
            "Kidney Function Test (KFT)": 0,
            "Complete Blood Count (CBC)": 0,
            "Lipid Profile": 0,
            "Thyroid Profile": 0,
            "Iron Profile": 0,
            "Diabetes Profile": 0,
            "Urine Routine": 0,
            "Urine ACR Profile": 0
        }

        lft_keys = {"SGOT (AST)", "SGPT (ALT)", "Total Bilirubin", "Direct Bilirubin", "Indirect Bilirubin", "Alkaline Phosphatase (ALP)", "Gamma Glutamyl Transferase (GGT)"}
        kft_keys = {"Blood Urea", "Blood Urea Nitrogen (BUN)", "Serum Creatinine", "Uric Acid", "Serum Sodium", "Serum Potassium"}
        cbc_keys = {"Hemoglobin", "Total Leukocyte Count (TLC)", "Total RBC Count", "Packed Cell Volume (PCV)", "Platelet Count", "Neutrophils", "Mean Corpuscular Volume (MCV)"}
        lipid_keys = {"Total Cholesterol", "Triglycerides", "HDL Cholesterol", "LDL Cholesterol", "VLDL Cholesterol"}
        thyroid_keys = {"TSH", "Free T3 (FT3)", "Free T4 (FT4)", "Total T3", "Total T4"}
        iron_keys = {"Serum Iron", "TIBC", "UIBC", "Serum Ferritin", "Transferrin Saturation"}
        diabetes_keys = {"HbA1c (Glycated Hemoglobin)", "Fasting Blood Glucose", "Postprandial Blood Glucose"}
        urine_keys = {"Urine Color", "Urine Protein", "Specific Gravity", "Pus Cells (WBCs)"}

        for t in extracted_tests:
            name = t.get("test")
            if name in lft_keys: counts["Liver Function Test (LFT)"] += 1
            elif name in kft_keys: counts["Kidney Function Test (KFT)"] += 1
            elif name in cbc_keys: counts["Complete Blood Count (CBC)"] += 1
            elif name in lipid_keys: counts["Lipid Profile"] += 1
            elif name in thyroid_keys: counts["Thyroid Profile"] += 1
            elif name in iron_keys: counts["Iron Profile"] += 1
            elif name in diabetes_keys: counts["Diabetes Profile"] += 1
            elif name in urine_keys: counts["Urine Routine"] += 1

        best_match = max(counts, key=counts.get)
        return best_match if counts[best_match] > 0 else "General Health Panel"

    def standardize(self, item):
        if isinstance(item, str):
            item = {"test": item, "value": "", "unit": "", "reference": "", "method": ""}
        elif not isinstance(item, dict):
            item = {"test": str(item), "value": "", "unit": "", "reference": "", "method": ""}

        raw_test = item.get("test", "").strip()
        raw_val = item.get("value", "").strip()
        extracted_unit = item.get("unit", "").strip()
        extracted_ref = item.get("reference", "").strip()
        extracted_method = item.get("method", "").strip()

        # 1. Extract trailing numerical value if stuck to raw_test string
        extracted_value = ""
        val_match = re.search(r"(\d+(?:\.\d+)?)\s*$", raw_test)
        if val_match:
            extracted_value = val_match.group(1)
            raw_test = raw_test[:val_match.start()].strip()

        # 2. Extract Unit from raw_val field if combined
        for u in self.known_units:
            if re.search(r'\b' + re.escape(u) + r'\b', raw_val, flags=re.IGNORECASE):
                extracted_unit = u
                raw_val = re.sub(re.escape(u), "", raw_val, flags=re.IGNORECASE).strip()
                break

        # AUTO-REPAIR: If reference column accidentally grabbed a unit string due to column shift
        if extracted_ref and not extracted_unit:
            for u in self.known_units:
                if u.lower() in extracted_ref.lower():
                    extracted_unit = extracted_ref
                    extracted_ref = ""
                    break

        # Combine loose text fields for isolated classification
        raw_ref_method = f"{extracted_ref} {extracted_method} {raw_val}".strip()
        if extracted_ref == extracted_unit:
            extracted_ref = ""

        if raw_ref_method and extracted_ref != extracted_unit:
            ref_match = re.search(r"(?:<|>|<=|>=)\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?", raw_ref_method)
            
            if ref_match:
                extracted_ref = ref_match.group(0).strip()
                extracted_method = (raw_ref_method[:ref_match.start()] + " " + raw_ref_method[ref_match.end():]).strip()
                extracted_method = re.sub(r'\bUV\s+with\s+P\s+P\b', 'UV with P5P', extracted_method, flags=re.IGNORECASE)
            elif any(qual in raw_ref_method.upper() for qual in ["NEGATIVE", "NIL", "NORMAL", "ABSENT"]):
                extracted_ref = "Negative"
                extracted_method = re.sub(r'(?i)negative|nil|normal|absent', '', raw_ref_method).strip()
            else:
                extracted_method = raw_ref_method

        final_value = extracted_value if extracted_value else raw_val

        # 3. Canonical name mapping
        raw_upper = raw_test.upper()
        canonical_name = raw_test
        for key, value in self.canonical_map:
            if key in raw_upper:
                canonical_name = value
                break

        # ORPHANED ROW SAFEGUARD: If OCR completely ate the test name but value/method exists
        if not canonical_name or len(canonical_name.strip()) < 2:
            combined_context = f"{extracted_method} {extracted_ref}".upper()
            if "PUPP" in combined_context or "APMP" in combined_context or "ALP" in combined_context:
                canonical_name = "Alkaline Phosphatase (ALP)"
            else:
                canonical_name = "Unknown Clinical Parameter"

        # 4. Compute High / Low / Normal flag (incorporating safety boundaries)
        status_flag = self.compute_flag(canonical_name, final_value, extracted_ref)
        validation_text = "Within Expected Range" if status_flag == "Normal" else "Out of Range"

        return {
            "test": canonical_name,
            "raw_test_name": raw_test,
            "value": final_value,
            "unit": extracted_unit if extracted_unit != "eport" else "",
            "reference": extracted_ref,
            "flag": status_flag,
            "validation": validation_text,
            "method": extracted_method
        }