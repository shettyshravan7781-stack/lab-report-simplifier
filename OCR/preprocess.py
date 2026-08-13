import os
import cv2
import numpy as np
from pathlib import Path

# Safe import for config settings regardless of package invocation
try:
    from OCR.config import RESIZE_WIDTH, SAVE_PREPROCESSED, OUTPUT_DIR
except ImportError:
    try:
        from config import RESIZE_WIDTH, SAVE_PREPROCESSED, OUTPUT_DIR
    except ImportError:
        RESIZE_WIDTH = 1600
        SAVE_PREPROCESSED = True
        OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ImagePreprocessor:
    """
    Preprocesses raw lab report images to optimize OCR accuracy.
    Includes aspect-ratio scaling, CLAHE contrast enhancement, 
    and median noise filtering.
    """

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def preprocess(self, image_input):
        """
        Preprocesses an image.

        Args:
            image_input (str | Path | np.ndarray): File path string, Path object, 
                                                  or loaded OpenCV BGR image array.

        Returns:
            np.ndarray: Preprocessed 3-channel BGR image optimized for PaddleOCR.
        """
        # Load image if input is a file path
        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at path: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Unable to decode image file at: {image_path}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise TypeError("Expected image_input to be a file path (str/Path) or cv2 image array (np.ndarray).")

        # ------------------------------------
        # 1. Resize (only if wider than RESIZE_WIDTH)
        # ------------------------------------
        h, w = image.shape[:2]
        if w > RESIZE_WIDTH:
            ratio = RESIZE_WIDTH / float(w)
            new_h = int(h * ratio)
            image = cv2.resize(
                image,
                (RESIZE_WIDTH, new_h),
                interpolation=cv2.INTER_AREA
            )

        # ------------------------------------
        # 2. Convert to Grayscale
        # ------------------------------------
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ------------------------------------
        # 3. CLAHE Local Contrast Adaptive Histogram
        # ------------------------------------
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        gray = clahe.apply(gray)

        # ------------------------------------
        # 4. Remove Fine Noise
        # ------------------------------------
        gray = cv2.medianBlur(gray, 3)

        # ------------------------------------
        # 5. Re-convert to 3-channel BGR format
        # Note: Avoid binary thresholding as deep-learning OCR models 
        # (PaddleOCR/Tesseract) achieve better accuracy on grayscale ranges.
        # ------------------------------------
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if SAVE_PREPROCESSED:
            save_path = OUTPUT_DIR / "preprocessed.png"
            cv2.imwrite(str(save_path), processed)

        return processed

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """
        Optional helper method to detect document tilt and deskew the image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # If angle is minor, don't perform rotation
        if abs(angle) < 0.5 or abs(angle) > 15.0:
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated