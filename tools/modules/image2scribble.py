from PIL import Image
from controlnet_aux import HEDdetector

from ..utils import get_new_image_name, prompts


class Image2Scribble:
    def __init__(self, device):
        print("Initializing Image2Scribble")
        self.detector = HEDdetector.from_pretrained("lllyasviel/Annotators")

    @prompts(
        name="Sketch Detection On Image",
        description="useful when you want to generate a scribble of the image. "
        "like: generate a scribble of this image, or generate a sketch from this image, "
        "detect the sketch from this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        scribble = self.detector(image, scribble=True)
        updated_image_path = get_new_image_name(inputs, func_name="scribble")
        scribble.save(updated_image_path)
        print(
            f"\nProcessed Image2Scribble, Input Image: {inputs}, Output Scribble: {updated_image_path}"
        )
        return updated_image_path
