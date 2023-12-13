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


class ScribbleText2Image:
    def __init__(self, device):
        print(f"Initializing ScribbleText2Image to {device}")
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            # "fusing/stable-diffusion-v1-5-controlnet-scribble",
            "lllyasviel/sd-controlnet-scribble",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            # "CompVis/stable-diffusion-v1-4",
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
        self.a_prompt = "best quality, extremely detailed, realistic"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
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
        name="Generate Image Condition On Sketch or Scribble Image",
        description="useful when you want to generate a new real image from both the user description and "
        "a scribble image or a sketch image. "
        "like: Use the reference image to generate an image of a man wearing a jacket"
        "like: use the sketch to generate an image of a girl with a spaghetti dress."
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path and the user description",
    )
    def inference(self, inputs, num_inference_steps=50, guidance_Scale=7.5):
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
            eta=0.0,
            num_inference_steps=num_inference_steps,
            negative_prompt=self.n_prompt,
            guidance_scale=guidance_Scale,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed ScribbleText2Image, Input Scribble: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path
