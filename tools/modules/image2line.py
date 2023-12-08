from PIL import Image
from controlnet_aux import MLSDdetector

from ..utils import get_new_image_name, prompts


class Image2Line:
    def __init__(self, device):
        print("Initializing Image2Line")
        self.detector = MLSDdetector.from_pretrained("lllyasviel/Annotators")

    @prompts(
        name="Line Detection On Image",
        description="useful when you want to detect the straight line of the image. "
        "like: detect the straight lines of this image, or straight line detection on image, "
        "or perform straight line detection on this image, or detect the straight line image of this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        mlsd = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="line-of")
        mlsd.save(updated_image_path)
        print(
            f"\nProcessed Image2Line, Input Image: {inputs}, Output Line: {updated_image_path}"
        )
        return updated_image_path
