import re


class ValidationEngine:
    """
    Validates numeric result values against reference ranges and sets
    'flag' ('High', 'Low', 'Normal') and 'validation' messaging.
    """

    def extract_number(self, text):
        if text is None:
            return None

        # Strips out operator prefixes if present, extracts raw float
        match = re.search(r"\d+(?:\.\d+)?", str(text).replace(",", ""))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None

        return None

    def validate(self, tests):
        if not tests or not isinstance(tests, list):
            return tests

        for test in tests:
            value = self.extract_number(test.get("value", ""))
            ref_raw = str(test.get("reference", "")).strip()

            test["flag"] = "Normal"
            test["validation"] = "Within Expected Range"

            if value is None or not ref_raw:
                continue

            try:
                # -------------------------------------------------------------
                # Pattern 1: Range format (e.g. "13.0 - 17.0", "13.0-17.0 g/dL")
                # -------------------------------------------------------------
                range_match = re.search(r"(\d+(?:\.\d+)?)\s*[\-\–\—\to]+\s*(\d+(?:\.\d+)?)", ref_raw, re.IGNORECASE)
                if range_match:
                    low = float(range_match.group(1))
                    high = float(range_match.group(2))

                    if value < low:
                        test["flag"] = "Low"
                        test["validation"] = "Below Expected Range"
                    elif value > high:
                        test["flag"] = "High"
                        test["validation"] = "Above Expected Range"
                    continue

                # -------------------------------------------------------------
                # Pattern 2: Less than threshold (e.g. "< 2.0", "<= 1.5", "UP TO 1.0")
                # -------------------------------------------------------------
                less_match = re.search(r"(?:<|<=|up\s*to)\s*(\d+(?:\.\d+)?)", ref_raw, re.IGNORECASE)
                if less_match:
                    limit = float(less_match.group(1))
                    if value > limit:
                        test["flag"] = "High"
                        test["validation"] = "Above Expected Range"
                    continue

                # -------------------------------------------------------------
                # Pattern 3: Greater than threshold (e.g. "> 5.0", ">= 10")
                # -------------------------------------------------------------
                greater_match = re.search(r"(?:>|>=)\s*(\d+(?:\.\d+)?)", ref_raw, re.IGNORECASE)
                if greater_match:
                    limit = float(greater_match.group(1))
                    if value < limit:
                        test["flag"] = "Low"
                        test["validation"] = "Below Expected Range"
                    continue

            except Exception:
                # Fallback to default Normal if unexpected range string structure
                pass

        return tests