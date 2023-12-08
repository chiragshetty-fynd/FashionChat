import os
import uuid
import requests

from .image_captioning import image_captioning
from ..utils import prompts, get_new_image_name


class DalleInpainting:
    def __init__(self, device):
        self.url = "http://openai_hub:80/inpaint"
        self.image_captioning = image_captioning
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

    @prompts(
        name="Dalle Inpainting tool to replace or edit highlighted/masked/selected region",
        description="useful when you want to edit and/or replace a part of the user image highlighted by the mask."
        " like: remove the round collar from highlighted part and replace it a v-neck"
        " or replace the tiger highlighted in the image to a lion. "
        " like: edit the a real image to modify an object or something masked in this image,"
        " or modify an highlighted object in an image using this masked image. "
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path and the user description. ",
    )
    def inference(self, inputs):
        img_path, prompt = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        description = self.image_captioning.inference(img_path)
        prompt = f"{description}. {prompt}"
        updated_img_path = get_new_image_name(img_path, func_name="dalleinpainting")
        data = {
            "prompt": prompt,
            "img_path": img_path,
            "updated_img_path": updated_img_path,
        }
        response = requests.post(self.url, headers=self.headers, data=data)
        if response.status_code == 200:
            print(
                f"\nProcessed Dalle Text2Image, Input Text: {data['prompt']}, Input Image: {data['img_path']}, Output Image: {data['updated_img_path']}"
            )
        else:
            print(
                f"\nError while processing Dalle Text2Image, Input Text: {data['prompt']}, Input Image: {data['img_path']}, Output Image: {data['updated_img_path']}"
            )
        return data["updated_img_path"]
