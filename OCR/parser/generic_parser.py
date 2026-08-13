class GenericParser:

    def __init__(self, columns):
        self.columns = columns

    def nearest_column(self, x):

        nearest = None
        distance = 999999

        for name, cx in self.columns.items():

            d = abs(x - cx)

            if d < distance:
                distance = d
                nearest = name

        return nearest

    def parse_table(self, rows):

        records = []

        started = False

        for row in rows:

            line = " ".join(item["text"] for item in row).upper()

            if "TEST" in line and "REFERENCE" in line:
                started = True
                continue

            if not started:
                continue

            if "CLINICAL NOTES" in line:
                break

            record = {}

            for item in row:

                column = self.nearest_column(item["box"][0])

                if column not in record:
                    record[column] = item["text"]

                else:
                    record[column] += " " + item["text"]

            if record:
                records.append(record)

        return records