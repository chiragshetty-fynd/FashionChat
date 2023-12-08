import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

from ..utils import prompts, get_new_image_name


class Image2Normal:
    def __init__(self, device):
        print("Initializing Image2Normal")
        self.depth_estimator = pipeline(
            "depth-estimation", model="Intel/dpt-hybrid-midas"
        )
        self.bg_threhold = 0.4

    @prompts(
        name="Predict Normal Map On Image",
        description="useful when you want to detect norm map of the image. "
        "like: generate normal map from this image, or predict normal map of this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        original_size = image.size
        image = self.depth_estimator(image)["predicted_depth"][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < self.bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < self.bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize(original_size)
        updated_image_path = get_new_image_name(inputs, func_name="normal-map")
        image.save(updated_image_path)
        print(
            f"\nProcessed Image2Normal, Input Image: {inputs}, Output Depth: {updated_image_path}"
        )
        return updated_image_path
