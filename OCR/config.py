"""
Configuration file for LabReportAI pipeline parameters and path definitions.
"""
from pathlib import Path

# =============================================================================
# Project Directory Paths
# =============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent
IMAGE_DIR: Path = BASE_DIR / "images"
OUTPUT_DIR: Path = BASE_DIR / "output"
TEMP_DIR: Path = BASE_DIR / "temp"

# Ensure all operational directories exist
for directory in [IMAGE_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# OCR Engine Settings
# =============================================================================
LANGUAGE: str = "en"
MIN_CONFIDENCE: float = 0.50

# =============================================================================
# Image Preprocessing Settings
# =============================================================================
RESIZE_WIDTH: int = 1600
SAVE_PREPROCESSED: bool = True

# =============================================================================
# Layout & Table Extraction Thresholds
# =============================================================================
ROW_Y_THRESHOLD: int = 12       # Vertical distance (pixels) to group OCR boxes into one row
COLUMN_X_MARGIN: int = 15       # Margin allowance (pixels) for column boundary matching

# =============================================================================
# Medical Knowledge Engine Settings
# =============================================================================
FUZZY_MATCH_THRESHOLD: float = 0.60  # Minimum similarity score for fuzzy string matching (0.0 to 1.0)