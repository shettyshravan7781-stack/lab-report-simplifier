from parser.cbc_parser import CBCParser
from parser.lft_parser import LFTParser


class ParserFactory:

    @staticmethod
    def get_parser(report_type, columns):

        if report_type == "CBC":
            return CBCParser(columns)

        elif report_type == "LFT":
            return LFTParser(columns)

        raise ValueError(f"No parser available for '{report_type}'")