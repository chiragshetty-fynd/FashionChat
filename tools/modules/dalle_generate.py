import os
import uuid
import requests

from ..utils import prompts


class DalleGenerate:
    def __init__(self, device):
        self.url = "http://openai_hub:80/generate"
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

    @prompts(
        name="Dalle Generate Image From User Input Text",
        description="useful when you want to generate an image using openai dalle-3 from a user input text and save it to a file. "
        "like: generate an image of an object or something, or generate an image that includes some objects. "
        "The input to this tool should be a string, representing the text used to generate image. ",
    )
    def inference(self, prompt):
        img_path = os.path.join("image", f"{str(uuid.uuid4())[:8]}.png")
        data = {
            "prompt": prompt,
            "img_path": img_path,
        }
        response = requests.post(self.url, headers=self.headers, data=data)
        if response.status_code == 200:
            print(
                f"\nProcessed Dalle Text2Image, Input Text: {data['prompt']}, Output Image: {data['img_path']}"
            )
        else:
            print(
                f"\nError while processing Dalle Text2Image, Input Text: {data['prompt']}, Output Image: {data['img_path']}"
            )
        return data["img_path"]
