from PIL import Image
from .object_segmenting import ObjectSegmenting

from ..utils import prompts, get_new_image_name


class BackgroundRemoving:
    """
    using to remove the background of the given picture
    """

    template_model = True

    def __init__(
        self,
        VisualQuestionAnswering,
        Text2Box,
        Segmenting,
    ):
        self.vqa = VisualQuestionAnswering
        self.obj_segmenting = ObjectSegmenting(Text2Box, Segmenting)

    @prompts(
        name="Remove the background",
        description="useful when you want to extract the object or remove the background,"
        "the input should be a string image_path",
    )
    def inference(self, image_path):
        """
        given a image, return the picture only contains the extracted main object
        """
        updated_image_path = None

        mask = self.get_mask(image_path)

        image = Image.open(image_path)
        mask = Image.fromarray(mask)
        image.putalpha(mask)

        updated_image_path = get_new_image_name(
            image_path, func_name="detect-something"
        )
        image.save(updated_image_path)

        return updated_image_path

    def get_mask(self, image_path):
        """
        Description:
            given an image path, return the mask of the main object.
        Args:
            image_path (string): the file path of the image
        Outputs:
            mask (numpy.ndarray): H x W
        """
        vqa_input = f"{image_path}, what is the main object in the image?"
        text_prompt = self.vqa.inference(vqa_input)

        mask = self.obj_segmenting.get_mask(image_path, text_prompt)

        return mask
