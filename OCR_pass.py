
# ----------- Standard imports -----------

import os                          # used to validate file paths
import cv2                         # OpenCV for image loading & preprocessing
import numpy as np                 # images are handled as numpy arrays

# ----------- PaddleOCR import -----------

from paddleocr import PaddleOCR    # main OCR engine


class PassportOCRExtractor:
    """
    This class handles:
    - Loading an image
    - Preprocessing it for OCR
    - Running PaddleOCR
    - Returning extracted text as a single string

    You can call this class from your Gemini code
    and pass the returned text directly to the LLM.
    """

    def __init__(self):
        """
        Initialize PaddleOCR once.
        This is important for performance and memory.
        """

        self.ocr_model = PaddleOCR(
            use_angle_cls=True,    # enables rotated text detection
            lang="en"              # English OCR
        )

    def _load_image(self, image_path: str):
        """
        Loads image from disk using OpenCV.
        Returns image as a NumPy array.
        """

        # Check if image path exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image in BGR format (OpenCV default)
        image = cv2.imread(image_path)

        # If OpenCV fails to load the image
        if image is None:
            raise ValueError("Failed to load image. Unsupported or corrupted file.")

        return image

    def _preprocess_for_ocr(self, image):
        """
        PaddleOCR v3+ expects a 3-channel BGR image.
        Do NOT convert to grayscale or threshold here.
        """

        # If a grayscale image is accidentally passed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image

    def extract_text(self, image_path: str) -> str:
        """
        Main public method.
        Takes an image path and returns extracted OCR text.
        """

        # Load image from disk
        image = self._load_image(image_path)

        # Preprocess image for OCR
        image = self._preprocess_for_ocr(image)

        # Run PaddleOCR (v3.x uses predict)
        results = self.ocr_model.predict(image)

        extracted_texts = []

        # Iterate over OCR results
        for page in results:
            # rec_texts contains detected text strings
            for text in page.get("rec_texts", []):
                if text.strip():           # ignore empty strings
                    extracted_texts.append(text)

        # Combine all detected text into one string
        full_text = " ".join(extracted_texts)

        return full_text


# ----------- Example usage (optional test) -----------

if __name__ == "__main__":
    image_path = "/home/neosoft/Downloads/passport/Image.jpeg"

    ocr_extractor = PassportOCRExtractor()
    extracted_text = ocr_extractor.extract_text(image_path)

    print("\n========== OCR TEXT ==========\n")
    print(extracted_text)
