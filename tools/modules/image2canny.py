import cv2
import numpy as np
from PIL import Image

from ..utils import prompts, get_new_image_name


class Image2Canny:
    def __init__(self, device):
        print("Initializing Image2Canny")
        self.low_threshold = 100
        self.high_threshold = 200

    @prompts(
        name="Edge Detection On Image",
        description="useful when you want to detect the edge of the image. "
        "like: detect the edges of this image, or canny detection on image, "
        "or perform edge detection on this image, or detect the canny image of this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        updated_image_path = get_new_image_name(inputs, func_name="edge")
        canny.save(updated_image_path)
        print(
            f"\nProcessed Image2Canny, Input Image: {inputs}, Output Text: {updated_image_path}"
        )
        return updated_image_path
