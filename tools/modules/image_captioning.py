import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from ..utils import prompts


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained(
            # "Salesforce/blip-image-captioning-base"
            "Salesforce/blip-image-captioning-large"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            # "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large",
            torch_dtype=self.torch_dtype,
        ).to(self.device)

    @prompts(
        name="Get Photo Description",
        description="useful when you want to know what is inside the photo. receives image_path as input. "
        "The input to this tool should be a string, representing the image_path. ",
    )
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(
            f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}"
        )
        return captions


image_captioning = ImageCaptioning("cuda:0")
