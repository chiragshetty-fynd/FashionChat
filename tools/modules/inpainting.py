import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker


class Inpainting:
    def __init__(self, device):
        self.device = device
        self.revision = "fp16" if "cuda" in self.device else None
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
        ).to(device)

    def __call__(
        self, prompt, image, mask_image, height=512, width=512, num_inference_steps=50
    ):
        update_image = self.inpaint(
            prompt=prompt,
            image=image.resize((width, height)),
            mask_image=mask_image.resize((width, height)),
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        ).images[0]
        return update_image
