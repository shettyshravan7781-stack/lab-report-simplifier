import re

from parser.base_parser import BaseParser


class CBCParser(BaseParser):

    def __init__(self, columns):
        super().__init__()
        self.columns = columns

    def normalize_test(self, test):

        mapping = {

            "HEMOGLOBIN": "Hemoglobin",

            "TOTAL LEUKOCYTE COUNT": "Total Leukocyte Count",

            "NEUTROPHILS": "Neutrophils",

            "LYMPHOCYTE": "Lymphocytes",
            "LYMPHOCYTES": "Lymphocytes",

            "EOSINOPHILS": "Eosinophils",

            "MONOCYTES": "Monocytes",

            "BASOPHILS": "Basophils",

            "PLATELET COUNT": "Platelet Count",

            "TOTAL RBC COUNT": "Total RBC Count",

            "HEMATOCRIT VALUE, HCT": "Hematocrit (HCT)",
            "HEMATOCRIT (HCT)": "Hematocrit (HCT)",

            "MEAN CORPUSCULAR VOLUME, MCV": "Mean Corpuscular Volume (MCV)",
            "MEAN CORPUSCULAR VOLUME (MCV)": "Mean Corpuscular Volume (MCV)",

            "MEAN CELL HAEMOGLOBIN, MCH": "Mean Cell Hemoglobin (MCH)",
            "MEAN CELL HEMOGLOBIN (MCH)": "Mean Cell Hemoglobin (MCH)",

            "MEAN CELL HAEMOGLOBIN CON, MCHC H": "Mean Cell Hemoglobin Concentration (MCHC)",
            "MEAN CELL HAEMOGLOBIN CONCENTRATION, MCHC": "Mean Cell Hemoglobin Concentration (MCHC)",
            "MEAN CELL HEMOGLOBIN CONCENTRATION (MCHC)": "Mean Cell Hemoglobin Concentration (MCHC)"

        }

        return mapping.get(test.upper(), test.title())

    def fix_unit(self, test, unit):

        unit = unit.lower()

        corrections = {

            "gd": "g/dL",
            "gdl": "g/dL",

            "lakts/cumm": "lakhs/cumm",
            "lakhs/cumm": "lakhs/cumm",

            "million/oumm": "million/cumm",
            "million/cumm": "million/cumm",

            "pg": "pg",

            "%": "%",

            "fl": "fL"
        }

        if "Mean Corpuscular Volume" in test:
            return "fL"

        return corrections.get(unit, unit)

    def parse(self, rows):

        table = self.parse_table(rows)

        tests = []

        ignore_rows = {

            "DIFFERENTIAL LEUCOCYTE COUNT",
            "DIFFERENTIAL LEUKOCYTE COUNT",

            "HAEMATOLOGY",
            "HEMATOLOGY",

            "COMPLETE BLOOD COUNT (CBC)",

            "TEST"

        }

        for row in table:

            test = self.normalize_test(
                row.get("test", "").strip()
            )

            if test.upper() in ignore_rows:
                continue

            value = row.get("value", "").replace(",", "").strip()

            flag = "Normal"

            if value.startswith("H "):
                flag = "High"
                value = value[2:].strip()

            elif value.startswith("L "):
                flag = "Low"
                value = value[2:].strip()

            elif value == "H":
                flag = "High"
                value = ""

            elif value == "L":
                flag = "Low"
                value = ""

            unit = self.fix_unit(
                test,
                row.get("unit", "").strip()
            )

            tests.append({

                "test": test,

                "value": value,

                "unit": unit,

                "reference": row.get("reference", "").strip(),

                "flag": flag

            })

        return tests