UNIT_ALIASES = {
    "gd": "g/dL",
    "g/dl": "g/dL",
    "pg": "pg",
    "%": "%",
    "cumm": "cumm",
    "lakts/cumm": "lakhs/cumm",
    "lakhs/cumm": "lakhs/cumm",
    "million/oumm": "million/cumm",
    "million/cumm": "million/cumm",
    "u/l": "U/L",
    "iu/l": "IU/L",
}

def clean_unit(raw_unit: str) -> str:
    """Cleans up raw OCR unit strings using unit aliases."""
    if not raw_unit:
        return ""
    
    cleaned = raw_unit.strip().lower()
    return UNIT_ALIASES.get(cleaned, raw_unit)