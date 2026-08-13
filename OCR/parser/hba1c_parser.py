from parser.base_parser import BaseParser


class HbA1cParser(BaseParser):

    def __init__(self):
        super().__init__()

    def parse(self, rows):
        return self.parse_table(rows)