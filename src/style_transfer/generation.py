import random
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
)
from PIL import Image, ImageOps

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"
IP_ADAPTER_MODEL = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sd15.safetensors"
IMAGE_SIZE = 512


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_image(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def make_canny_image(
    image: Image.Image, low_threshold: int = 100, high_threshold: int = 200
) -> Image.Image:
    gray = cv2.cvtColor(np.asarray(prepare_image(image)), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(edges).convert("RGB")


def load_pipeline(cache_dir: Path, device: str = "mps") -> StableDiffusionControlNetPipeline:
    dtype = torch.float32 if device == "cpu" else torch.float16
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
        cache_dir=cache_dir,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_ip_adapter(
        IP_ADAPTER_MODEL,
        subfolder="models",
        weight_name=IP_ADAPTER_WEIGHT,
        cache_dir=cache_dir,
    )
    return pipeline.to(device)


@torch.inference_mode()
def generate_image(
    pipeline: StableDiffusionControlNetPipeline,
    photo: Image.Image,
    style_image: Image.Image,
    prompt: str,
    *,
    use_controlnet: bool = True,
    seed: int = 42,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    ip_adapter_scale: float = 0.7,
    controlnet_scale: float = 1.0,
    canny_low: int = 100,
    canny_high: int = 200,
    negative_prompt: str = "low quality, blurry, text, watermark",
) -> Image.Image:
    seed_everything(seed)
    active_pipeline = pipeline
    controlnet_kwargs = {}
    if use_controlnet:
        controlnet_kwargs = {
            "image": make_canny_image(photo, canny_low, canny_high),
            "controlnet_conditioning_scale": controlnet_scale,
        }
    else:
        active_pipeline = StableDiffusionPipeline.from_pipe(pipeline, torch_dtype=pipeline.dtype)

    active_pipeline.set_ip_adapter_scale(ip_adapter_scale)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    return active_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=ImageOps.exif_transpose(style_image).convert("RGB"),
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        **controlnet_kwargs,
    ).images[0]
