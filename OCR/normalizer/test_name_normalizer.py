try:
    from OCR.knowledge.aliases import get_clean_test_name
except ImportError:
    try:
        from knowledge.aliases import get_clean_test_name
    except ImportError:
        def get_clean_test_name(name):
            return name


class TestNameNormalizer:
    """
    Normalizes raw extracted test names into standard Title Case medical names.
    Leverages mapping dictionaries and knowledge alias fallbacks for OCR typos.
    """

    def __init__(self):
        self.mapping = {
            # CBC Variations
            "HEMOGLOBIN": "Hemoglobin",
            "HAEMOGLOBIN": "Hemoglobin",
            "TOTAL LEUKOCYTE COUNT": "Total Leukocyte Count",
            "TLC": "Total Leukocyte Count",
            "NEUTROPHILS": "Neutrophils",
            "LYMPHOCYTE": "Lymphocytes",
            "LYMPHOCYTES": "Lymphocytes",
            "EOSINOPHILS": "Eosinophils",
            "MONOCYTES": "Monocytes",
            "BASOPHILS": "Basophils",
            "PLATELET COUNT": "Platelet Count",
            "TOTAL RBC COUNT": "Total RBC Count",
            "RBC COUNT": "Total RBC Count",
            "HEMATOCRIT VALUE, HCT": "Hematocrit (HCT)",
            "HEMATOCRIT": "Hematocrit (HCT)",
            "HCT": "Hematocrit (HCT)",
            "MEAN CORPUSCULAR VOLUME, MCV": "Mean Corpuscular Volume (MCV)",
            "MCV": "Mean Corpuscular Volume (MCV)",
            "MEAN CELL HAEMOGLOBIN, MCH": "Mean Cell Hemoglobin (MCH)",
            "MCH": "Mean Cell Hemoglobin (MCH)",
            "MEAN CELL HAEMOGLOBIN CON, MCHC H": "Mean Cell Hemoglobin Concentration (MCHC)",
            "MCHC": "Mean Cell Hemoglobin Concentration (MCHC)",

            # LFT Variations
            "SGOT": "AST",
            "SGOT / SGPT RATIO": "AST/ALT Ratio",
            "SGPT": "ALT",
            "TOTAL BILIRUBIN": "Total Bilirubin",
            "DIRECT BILIRUBIN": "Direct Bilirubin",
            "INDIRECT BILIRUBIN": "Indirect Bilirubin",
            "ALKALINE PHOSPHATASE": "Alkaline Phosphatase",
            "TOTAL PROTEIN": "Total Protein",
            "ALBUMIN": "Albumin",
            "GLOBULIN": "Globulin",
            "A/G RATIO": "Albumin/Globulin Ratio"
        }

    def normalize(self, test: str) -> str:
        if not test:
            return ""

        clean_input = test.strip().upper()

        # 1. Check exact map match
        if clean_input in self.mapping:
            return self.mapping[clean_input]

        # 2. Fallback to knowledge aliases (handles OCR typos like "SOOT")
        clean_alias = get_clean_test_name(test)
        if clean_alias != test:
            return clean_alias

        # 3. Default return original
        return test