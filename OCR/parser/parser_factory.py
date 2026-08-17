import os
import sys

# Ensure project root is available for sub-module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from OCR.parser.cbc_parser import CBCParser
    from OCR.parser.lft_parser import LFTParser
except ImportError:
    from parser.cbc_parser import CBCParser
    from parser.lft_parser import LFTParser


class ParserFactory:

    @staticmethod
    def get_parser(report_type, columns=None):
        report_type_upper = str(report_type).upper()

        if "CBC" in report_type_upper or "HAEMATOLOGY" in report_type_upper or "HEMATOLOGY" in report_type_upper:
            return CBCParser(columns)
        elif "LFT" in report_type_upper or "LIVER" in report_type_upper:
            return LFTParser(columns)

        # Fallback default to CBC if report type is unrecognized
        return CBCParser(columns)