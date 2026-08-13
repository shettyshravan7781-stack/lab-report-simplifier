from parser.report_detector import ReportDetector
from parser.PatientParser import PatientParser
from parser.cbc_parser import CBCParser


class MedicalParser:

    def __init__(self):

        self.detector = ReportDetector()
        self.patient_parser = PatientParser()
        self.cbc_parser = CBCParser()

    def parse(self, rows):

        report_type = self.detector.detect(rows)

        report = {
            "report_type": report_type,
            "patient": {},
            "tests": []
        }

        # Patient Details
        report["patient"] = self.patient_parser.parse(rows)

        # Report Parser
        if report_type == "CBC":
            report["tests"] = self.cbc_parser.parse(rows)

        else:
            print(f"{report_type} parser not implemented.")

        return report