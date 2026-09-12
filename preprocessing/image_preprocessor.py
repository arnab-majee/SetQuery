from pathlib import Path
from PIL import Image

class ImagePreprocessor:
    def validate(self, image_path: Path) -> bool:
        if not image_path.exists():
            return False
        try:
            with Image.open(image_path) as image:
                image.verify()  # Verify that it is an image
            return True

        except Exception:
            return False

    def get_size(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            return image.size  # Returns (width & height)

image_preprocessor = ImagePreprocessor()