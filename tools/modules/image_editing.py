import cv2
import torch
import numpy as np
from PIL import Image

from ..utils import prompts, get_new_image_name


class ImageEditing:
    template_model = True

    def __init__(self, Text2Box, Segmenting, Inpainting):
        print(f"Initializing ImageEditing")
        self.sam = Segmenting
        self.grounding = Text2Box
        self.inpaint = Inpainting

    def pad_edge(self, mask, padding):
        # mask Tensor [H,W]
        mask = mask.numpy()
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(
                slice(max(0, i - padding), i + padding + 1) for i in idx
            )
            mask_array[padded_slice] = True
        new_mask = (mask_array * 255).astype(np.uint8)
        # new_mask
        return new_mask

    @prompts(
        name="Remove Something From The Photo",
        description="useful when you want to remove object or something from the photo "
        "from its description or location. "
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path and the object need to be removed. ",
    )
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        return self.inference_replace_sam(
            f"{image_path},{to_be_removed_txt},background"
        )

    @prompts(
        name="Replace Something From The Photo",
        description="useful when you want to replace an object from the object description or "
        "location with another object from its description. "
        "The input to this tool should be a comma separated string of three, "
        "representing the image_path, the object to be replaced, the object to be replaced with ",
    )
    def inference_replace_sam(self, inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")

        print(f"image_path={image_path}, to_be_replaced_txt={to_be_replaced_txt}")
        image_pil, image = self.grounding.load_image(image_path)
        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(
            image, to_be_replaced_txt
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam.sam_predictor.set_image(image)
        masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)
        mask = torch.sum(masks, dim=0).unsqueeze(0)
        mask = torch.where(mask > 0, True, False)
        mask = mask.squeeze(0).squeeze(0).cpu()  # tensor

        mask = self.pad_edge(mask, padding=20)  # numpy
        mask_image = Image.fromarray(mask)

        updated_image = self.inpaint(
            prompt=replace_with_txt, image=image_pil, mask_image=mask_image
        )
        updated_image_path = get_new_image_name(
            image_path, func_name="replace-something"
        )
        updated_image = updated_image.resize(image_pil.size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path
