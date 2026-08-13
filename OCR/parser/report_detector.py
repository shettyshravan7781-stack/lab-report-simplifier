class ReportDetector:

    def __init__(self):
        pass

    def detect(self, rows):

        text = ""

        for row in rows:
            for item in row:
                text += item["text"].upper() + " "

        if "COMPLETE BLOOD COUNT" in text or "CBC" in text:
            return "CBC"

        elif "LIVER FUNCTION TEST" in text or "LFT" in text:
            return "LFT"

        elif "KIDNEY FUNCTION TEST" in text or "KFT" in text:
            return "KFT"

        elif "THYROID" in text:
            return "THYROID"

        return "UNKNOWN"

if __name__ == "__main__":

    sample = [
        [
            {"text": "Complete"},
            {"text": "Blood"},
            {"text": "Count"},
            {"text": "(CBC)"}
        ]
    ]

    detector = ReportDetector()

    print(detector.detect(sample))