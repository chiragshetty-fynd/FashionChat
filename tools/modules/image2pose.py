from PIL import Image
from controlnet_aux import OpenposeDetector

from ..utils import prompts, get_new_image_name


class Image2Pose:
    def __init__(self, device):
        print("Initializing Image2Pose")
        self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    @prompts(
        name="Pose Detection On Image",
        description="useful when you want to detect the human pose of the image. "
        "like: generate human poses of this image, or generate a pose image from this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        pose = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="human-pose")
        pose.save(updated_image_path)
        print(
            f"\nProcessed Image2Pose, Input Image: {inputs}, Output Pose: {updated_image_path}"
        )
        return updated_image_path
