from typing import List
import uuid
from mock import patch


class ImagesMocker:
    """HACK ALERT: I needed a way to call the booste API without storing the images first
     (as that is not allowed in streamlit sharing). If you have a better idea on hwo to this let me know!"""

    def __init__(self):
        self.pil_patch = patch('PIL.Image.open', lambda x: self.image_id2image(x))
        self.path_patch = patch('os.path.exists', lambda x: True)
        self.image_id2image_lookup = {}

    def start_mocking(self):
        self.pil_patch.start()
        self.path_patch.start()

    def stop_mocking(self):
        self.pil_patch.stop()
        self.path_patch.stop()

    def image_id2image(self, image_id: str):
        return self.image_id2image_lookup[image_id]

    def calculate_image_id2image_lookup(self, images: List):
        self.image_id2image_lookup = {str(uuid.uuid4()) + ".png": image for image in images}

    @property
    def image_ids(self):
        return list(self.image_id2image_lookup.keys())
