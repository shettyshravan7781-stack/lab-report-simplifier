import json
from pathlib import Path


class Exporter:
    """
    Handles serializing and saving pipeline dictionary outputs into structured JSON files.
    """

    def __init__(self, output_dir: str = "OCR/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, data: dict, filename: str = "report.json") -> Path:
        """
        Exports the provided data dictionary to a JSON file.
        """
        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                data, 
                f, 
                indent=4, 
                ensure_ascii=False, 
                default=str  # Fallback for non-serializable objects like Path or datetime
            )

        print("\n====================================")
        print("Report exported successfully!")
        print(f"Saved to: {output_file.resolve()}")
        print("====================================")

        return output_file