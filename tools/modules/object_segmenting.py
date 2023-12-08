import cv2
import torch
import numpy as np
from PIL import Image

from ..utils import prompts


class ObjectSegmenting:
    template_model = True  # Add this line to show this is a template model.

    def __init__(self, Text2Box, Segmenting):
        # self.llm = OpenAI(temperature=0)
        self.grounding = Text2Box
        self.sam = Segmenting

    @prompts(
        name="Segment the given object",
        description="useful when you only want to segment the certain objects in the picture"
        "according to the given text"
        "like: segment the cat,"
        "or can you segment an obeject for me"
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path, the text description of the object to be found",
    )
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        print(f"image_path={image_path}, text_prompt={det_prompt}")
        image_pil, image = self.grounding.load_image(image_path)

        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(image, det_prompt)
        updated_image_path = self.sam.segment_image_with_boxes(
            image_pil, image_path, boxes_filt, pred_phrases
        )
        print(
            f"\nProcessed ObejectSegmenting, Input Image: {image_path}, Object to be Segment {det_prompt}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path

    def merge_masks(self, masks):
        """
        Args:
            mask (numpy.ndarray): shape N x 1 x H x W
        Outputs:
            new_mask (numpy.ndarray): shape H x W
        """
        if type(masks) == torch.Tensor:
            x = masks
        elif type(masks) == np.ndarray:
            x = torch.tensor(masks, dtype=int)
        else:
            raise TypeError(
                "the type of the input masks must be numpy.ndarray or torch.tensor"
            )
        x = x.squeeze(dim=1)
        value, _ = x.max(dim=0)
        new_mask = value.cpu().numpy()
        new_mask.astype(np.uint8)
        return new_mask

    def get_mask(self, image_path, text_prompt):
        print(f"image_path={image_path}, text_prompt={text_prompt}")
        # image_pil (PIL.Image.Image) -> size: W x H
        # image (numpy.ndarray) -> H x W x 3
        image_pil, image = self.grounding.load_image(image_path)

        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(
            image, text_prompt
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam.sam_predictor.set_image(image)

        # masks (torch.tensor) -> N x 1 x H x W
        masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)

        # merged_mask -> H x W
        merged_mask = self.merge_masks(masks)
        # draw output image

        for mask in masks:
            image = self.sam.show_mask(
                mask[0].cpu().numpy(), image, random_color=True, transparency=0.3
            )

        merged_mask_image = Image.fromarray(merged_mask)
        return merged_mask
