try:
    from OCR.knowledge.units import clean_unit
except ImportError:
    try:
        from knowledge.units import clean_unit
    except ImportError:
        clean_unit = None


class UnitNormalizer:
    """
    Normalizes raw extracted unit strings into standard medical format.
    Fixes common OCR distortion noise on unit symbols.
    """

    def __init__(self):
        self.mapping = {
            "gd": "g/dL",
            "gdl": "g/dL",
            "g/d": "g/dL",
            "g/dl": "g/dL",
            "gm/dl": "g/dL",
            "lakts/cumm": "lakhs/cumm",
            "lakhs/cumm": "lakhs/cumm",
            "lakh/cumm": "lakhs/cumm",
            "million/oumm": "million/cumm",
            "million/cumm": "million/cumm",
            "mill/cumm": "million/cumm",
            "pg": "pg",
            "%": "%",
            "u/l": "U/L",
            "iu/l": "IU/L",
            "cumm": "cumm",
            "fl": "fL",
            "mg/dl": "mg/dL"
        }

    def normalize(self, unit: str) -> str:
        if not unit or not isinstance(unit, str):
            return ""

        cleaned = unit.strip().lower()

        # 1. Check exact map match
        if cleaned in self.mapping:
            return self.mapping[cleaned]

        # 2. Fallback to knowledge units module if available
        if clean_unit:
            return clean_unit(unit)

        return unit