import torch
import random
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from ..utils import prompts, get_new_image_name, seed_everything


class HedText2Image:
    def __init__(self, device):
        print(f"Initializing HedText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image Condition On Soft Hed Boundary Image",
        description="useful when you want to generate a new real image from both the user description "
        "and a soft hed boundary image. "
        "like: generate a real image of a object or something from this soft hed boundary image, "
        "or generate a new real image of a object or something from this hed boundary. "
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path and the user description",
    )
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = f"{instruct_text}, {self.a_prompt}"
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="hed2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed HedText2Image, Input Hed: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path
