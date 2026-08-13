import re

class PatientParser:
    """
    Extracts patient demographic metadata from header rows using clean RegEx patterns.
    """

    def parse(self, rows):
        text_lines = []
        for row in rows[:10]:  # Examine top 10 header lines
            line_str = " ".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in row)
            text_lines.append(line_str)

        full_header_text = "\n".join(text_lines)

        return {
            "name": self._extract_name(full_header_text),
            "age": self._extract_age(full_header_text),
            "gender": self._extract_gender(full_header_text),
            "uhid": self._extract_uhid(full_header_text),
            "date": self._extract_date(full_header_text),
            "lab_name": self._extract_lab_name(full_header_text)
        }

    def _extract_name(self, text):
        match = re.search(r'(?:Patient\s*Name|Name)\s*[:\-]?\s*([A-Za-z\s\.]+)', text, re.IGNORECASE)
        if match:
            raw_name = match.group(1).strip()
            # Truncate at neighboring label keywords
            for delimiter in ["Lab No", "Visit", "Age", "Sex", "Gender", "Ref", "UHID", "Date"]:
                if delimiter.lower() in raw_name.lower():
                    raw_name = re.split(re.escape(delimiter), raw_name, flags=re.IGNORECASE)[0]
            return raw_name.strip()
        return "Unknown"

    def _extract_age(self, text):
        match = re.search(r'(\d{1,3})\s*(?:Y|Yrs|Years|Yr)', text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_gender(self, text):
        if re.search(r'\b(Female|F)\b', text, re.IGNORECASE):
            return "Female"
        elif re.search(r'\b(Male|M)\b', text, re.IGNORECASE):
            return "Male"
        return "Unknown"

    def _extract_uhid(self, text):
        match = re.search(r'(?:UHID|Patient\s*ID)\s*[:\-]?\s*([A-Za-z0-9\-]+)', text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_date(self, text):
        match = re.search(r'\b\d{1,2}[\.\/\-](?:\d{1,2}|[A-Za-z]{3})[\.\/\-]\d{2,4}\b', text)
        return match.group(0) if match else None

    def _extract_lab_name(self, text):
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return lines[0] if lines else None