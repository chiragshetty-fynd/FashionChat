import os
import uuid
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from ..utils import prompts
from ..ehancer import PromptEnhancer


class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            # self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            # "dreamlike-art/dreamlike-photoreal-2.0",
            # "SG161222/RealVisXL_V2.0",
            torch_dtype=self.torch_dtype,
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed, realistic"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality, cartoonish, animated"
            "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch"
            "cartoon, drawing, anime), text, cropped, out of frame, worst quality"
            "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated"
            "extra fingers, mutated hands, poorly drawn hands, poorly drawn face"
            "mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions"
            "extra limbs, cloned face, disfigured, gross proportions, malformed limbs"
            "missing arms, missing legs, extra arms, extra legs, fused fingers"
            "too many fingers, long neck, folded clothing, creases, folds, ugly, "
            "extra print on clothes, nude, naked"
        )

    @prompts(
        name="SD Generate (Stable Diffusion), Image From User Input Text",
        description="useful when you want to generate an image from a user input text and save it to a file. "
        "like: generate an image of an object or something, or generate an image that includes some objects. "
        "The input to this tool should be a string, representing the text used to generate image. ",
    )
    def inference(self, text, guidance_scale=7.5, enhance=True):
        image_filename = os.path.join("image", f"{str(uuid.uuid4())[:8]}.png")
        prompt = text + ", " + self.a_prompt
        prompt = prompt if not enhance else PromptEnhancer.enhance(prompt)
        prompt = f"{prompt}, {self.a_prompt}"
        image = self.pipe(
            prompt, guidance_scale=guidance_scale, negative_prompt=self.n_prompt
        ).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}"
        )
        torch.cuda.empty_cache()
        return image_filename
