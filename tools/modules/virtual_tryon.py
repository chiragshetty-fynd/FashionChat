import torch
from PIL import Image
from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from .image_captioning import image_captioning
from ..utils import prompts, get_new_image_name
from .pipeline.reference import StableDiffusionReferencePipeline


class VirtualTryon:
    def __init__(self, device):
        self.device = device
        self.image_captioning = image_captioning
        self.revision = "fp16" if "cuda" in self.device else None
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        self.tryon = StableDiffusionReferencePipeline.from_pretrained(
            "dreamlike-art/dreamlike-photoreal-2.0",
            # revision=self.revision,
            torch_dtype=self.torch_dtype,
            # safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            #     "CompVis/stable-diffusion-safety-checker"
            # ),
        ).to(device)
        self.tryon.scheduler = UniPCMultistepScheduler.from_config(
            self.tryon.scheduler.config
        )
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
            "extra print on clothes, nude, naked, back, back facing"
        )

    @prompts(
        name="Virtual Tryon tool to visualize a fashion or clothing item on a person",
        description="usefule when you have a fashion item"
        " generated and want ot visualize it on a person "
        " like: tryon this image on a male fashion model"
        " like: visualize this image on a female asian model"
        " The input to this tool should be a comma separated string of two, "
        " representing the image_path and the user description. ",
    )
    def inference(
        self,
        inputs,
        height=512,
        width=512,
        num_inference_steps=50,
        attention_auto_machine_weight=1.0,
        gn_auto_machine_weight=1.0,
        style_fidelity=0.95,
        guidance_scale=6.5,
        reference_attn=True,
        reference_adain=False,
    ):
        image_path, prompt = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        description = self.image_captioning.inference(image_path)
        prompt = f"{prompt} {description}, {self.a_prompt}"
        image = Image.open(image_path)
        image = self.tryon(
            prompt=prompt,
            ref_image=image.resize((width, height)),
            negative_prompt=self.n_prompt,
            num_inference_steps=num_inference_steps,
            attention_auto_machine_weight=attention_auto_machine_weight,
            gn_auto_machine_weight=gn_auto_machine_weight,
            style_fidelity=style_fidelity,
            guidance_scale=guidance_scale,
            reference_attn=reference_attn,
            reference_adain=reference_adain,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="sdinpainting")
        image.save(updated_image_path)
        print(
            f"\nProcessed SDTryon, Input: {image_path}, Input Text: {prompt}, "
            f"Output Text: {updated_image_path}"
        )
        return updated_image_path
