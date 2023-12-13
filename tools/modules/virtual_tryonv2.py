import os
import requests
from ..utils import prompts


class VirtualTryon:
    def __init__(self, device):
        self.device = device
        self.url = "http://virtual_tryon:8888/tryon"
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

    @prompts(
        name="Virtual Tryon tool to visualize a fashion or clothing item on a person",
        description="useful when you have a fashion item"
        " generated and want ot visualize it on a person "
        " like: try on this image/cloth.png on this image/person_image.png person"
        " like: visualize this image/cloth.png on this image/person_image.png man"
        " like: tryon this image/cloth.png on this image/person_image.png woman"
        " The input to this tool should be a comma separated string of two, "
        " representing the image_path to the clothing or fashion item and image_path of the person. ",
    )
    def inference(self, inputs):
        cloth_path, img_path = inputs.split(",")
        data = {"img_path": img_path, "cloth_path": cloth_path}
        response = requests.post(self.url, headers=self.headers, data=data)

        img_fp = os.path.basename(img_path)
        cloth_fp = os.path.basename(cloth_path)
        _, ext = os.path.splitext(img_fp)
        to_path = os.path.join("image", img_fp.replace(ext, f"_{cloth_fp}"))

        if response.status_code == 200:
            print(
                f"\nProcessed Virtual Tryon, Input Cloth: {data['cloth_path']}, Input Image: {data['img_path']}, Output Image: {to_path}"
            )
        else:
            print(
                f"\nError while processing Dalle Text2Image, Input Cloth: {data['cloth_path']}, Input Image: {data['img_path']}, Output Image: {to_path}"
            )

        return to_path
