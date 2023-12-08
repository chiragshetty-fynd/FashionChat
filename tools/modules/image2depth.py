import numpy as np
from PIL import Image
from transformers import pipeline

from ..utils import prompts, get_new_image_name


class Image2Depth:
    def __init__(self, device):
        print("Initializing Image2Depth")
        self.depth_estimator = pipeline("depth-estimation")

    @prompts(
        name="Predict Depth On Image",
        description="useful when you want to detect depth of the image. like: generate the depth from this image, "
        "or detect the depth map on this image, or predict the depth for this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        depth = self.depth_estimator(image)["depth"]
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)
        depth = Image.fromarray(depth)
        updated_image_path = get_new_image_name(inputs, func_name="depth")
        depth.save(updated_image_path)
        print(
            f"\nProcessed Image2Depth, Input Image: {inputs}, Output Depth: {updated_image_path}"
        )
        return updated_image_path
