import re


class ValueCleaner:
    """
    Sanitizes raw OCR result values into clean numeric strings.
    Removes extraneous punctuation, colon prefixes, and inline unit noise
    while preserving leading comparison operators (<, >) and valid decimals.
    """

    def clean(self, value: str) -> str:
        if value is None:
            return ""

        if not isinstance(value, str):
            value = str(value)

        # 1. Remove commas and basic whitespace
        cleaned = value.replace(",", "").strip()

        if not cleaned:
            return ""

        # 2. Match leading operators (<, >) followed by digits/decimals (e.g. "< 0.05" -> "<0.05")
        operator_match = re.search(r"([<>]=?)\s*(\d+(?:\.\d+)?)", cleaned)
        if operator_match:
            op, val = operator_match.groups()
            return f"{op}{val}"

        # 3. Extract pure floating-point or integer number from text noise (e.g. ": 21.5 U/L" -> "21.5")
        number_match = re.search(r"\d+(?:\.\d+)?", cleaned)
        if number_match:
            return number_match.group(0)

        # 4. Fallback for non-numeric qualitative results (e.g. "NEGATIVE", "PRESENT", "NIL")
        return cleaned.strip(":= -")