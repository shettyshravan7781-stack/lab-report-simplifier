from parser.base_parser import BaseParser


class LFTParser(BaseParser):

    def __init__(self, columns):
        super().__init__(columns)

    def parse(self, rows):

        table = self.parse_table(rows)

        tests = []

        for row in table:

            if len(row["test"]) < 3:
                continue

            if "_" in row["test"]:
                continue

            tests.append({
                "test": row["test"],
                "value": self.clean_number(row["value"]),
                "unit": row["unit"],
                "reference": row["reference"]
            })

        return tests