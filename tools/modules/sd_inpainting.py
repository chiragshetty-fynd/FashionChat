import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from .image_captioning import image_captioning
from ..utils import prompts, get_new_image_name
from ..ehancer import PromptEnhancer


class SDInpainting:
    def __init__(self, device):
        self.device = device
        self.image_captioning = image_captioning
        self.revision = "fp16" if "cuda" in self.device else None
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            # "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=self.torch_dtype,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
        ).to(device)
        self.inpaint.enable_vae_slicing()
        self.inpaint.enable_freeu(s1=0.65, s2=0.45, b1=1.16, b2=1.26)
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
        name="SD Inpaint (Stable Diffusion) tool to replace or edit highlighted/masked/selected region",
        description="useful when you want to edit and/or replace a part of the user image highlighted by the mask."
        " like: remove the round collar from highlighted part and replace it a v-neck"
        " or replace the tiger highlighted in the image to a lion. "
        " like: edit the a real image to modify an object or something masked in this image,"
        " or modify an highlighted object in an image using this masked image. "
        "The input to this tool should be a comma separated string of two, "
        "representing the image_path and the user description. ",
    )
    def inference(
        self,
        inputs,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=20.0,
        strength=0.99,
        enhance=True,
    ):
        image_path, prompt = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        description = self.image_captioning.inference(image_path)
        prompt = f"{description}. {prompt}"
        mask_path = image_path.replace(".png", "_mask.png")
        mask = Image.open(mask_path)
        image = Image.open(image_path)
        prompt = prompt if not enhance else PromptEnhancer.enhance(prompt)
        prompt = f"{prompt}, {self.a_prompt}"
        image = self.inpaint(
            prompt=prompt,
            negative_prompt=self.n_prompt,
            image=image.resize((width, height)),
            mask_image=mask.resize((width, height)),
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="sdinpainting")
        image.save(updated_image_path)
        print(
            f"\nProcessed SDInpainting, Input: {image_path}, Input Text: {prompt}, "
            f"Output Text: {updated_image_path}"
        )
        return updated_image_path
