import torch
from PIL import Image
from diffusers.utils import export_to_video
from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from ..utils import prompts


class SDVideo:
    def __init__(self, device):
        self.device = device
        self.revision = "fp16" if "cuda" in self.device else None
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=self.torch_dtype,
            variant="fp16",
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
        ).to(device)
        self.pipe.enable_model_cpu_offload()

    @prompts(
        name="SDVideo (Stable Diffusion) generate video or gif or live photos from image",
        description="useful when you want to generate short videos, gifs or live photos with images"
        "The input to this tool should be a string representing the image path.",
    )
    def inference(
        self,
        image_path,
        height=512,
        width=512,
        decode_chunk_size=8,
        motion_bucket_id=100,
        noise_aug_strength=0.001,
        fps=7,
    ):
        image = Image.open(image_path)
        frames = self.pipe(
            image=image.resize((width, height)),
            height=height,
            width=width,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
        ).frames[0]
        video_path = image_path.replace(".png", ".mp4")
        export_to_video(frames, video_path, fps=fps)
        print(f"\nProcessed SDVideo, Input: {image_path} " f"Output Text: {video_path}")
        return video_path
