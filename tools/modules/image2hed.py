from PIL import Image
from controlnet_aux import HEDdetector

from ..utils import prompts, get_new_image_name


class Image2Hed:
    def __init__(self, device):
        print("Initializing Image2Hed")
        self.detector = HEDdetector.from_pretrained("lllyasviel/Annotators")

    @prompts(
        name="Hed Detection On Image",
        description="useful when you want to detect the soft hed boundary of the image. "
        "like: detect the soft hed boundary of this image, or hed boundary detection on image, "
        "or perform hed boundary detection on this image, or detect soft hed boundary image of this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        hed = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="hed-boundary")
        hed.save(updated_image_path)
        print(
            f"\nProcessed Image2Hed, Input Image: {inputs}, Output Hed: {updated_image_path}"
        )
        return updated_image_path
