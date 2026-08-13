from parser.base_parser import BaseParser


class LipidParser(BaseParser):

    def __init__(self):
        super().__init__()

    def parse(self, rows):
        return self.parse_table(rows)