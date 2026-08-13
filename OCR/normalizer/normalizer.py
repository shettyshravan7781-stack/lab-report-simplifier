from normalizer.unit_normalizer import UnitNormalizer
from normalizer.test_name_normalizer import TestNameNormalizer
from normalizer.value_cleaner import ValueCleaner


class Normalizer:

    def __init__(self):
        self.unit = UnitNormalizer()
        self.test = TestNameNormalizer()
        self.value = ValueCleaner()

    def normalize(self, report):
        if not report:
            return report

        # Support both 'raw_tests' and 'tests' keys safely
        test_list_key = "raw_tests" if "raw_tests" in report else "tests"

        if test_list_key in report and isinstance(report[test_list_key], list):
            for test in report[test_list_key]:
                if "test" in test:
                    test["test"] = self.test.normalize(test["test"])

                if "unit" in test:
                    test["unit"] = self.unit.normalize(test["unit"])

                if "value" in test:
                    test["value"] = self.value.clean(test["value"])

        return report