from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, BaseImageProcessor, Dinov2Model

from style_transfer.generation import make_canny_image, prepare_image

DINO_MODEL = "facebook/dinov2-small"


def load_dino(cache_dir: Path, device: str = "mps") -> tuple[BaseImageProcessor, Dinov2Model]:
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL, cache_dir=cache_dir, use_fast=False)
    model = Dinov2Model.from_pretrained(DINO_MODEL, cache_dir=cache_dir).to(device).eval()
    return processor, model


def canny_f1(
    source: Image.Image,
    generated: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
    tolerance: int = 3,
) -> float:
    source_edges = np.asarray(make_canny_image(source, low_threshold, high_threshold))[:, :, 0] > 0
    generated_edges = (
        np.asarray(make_canny_image(generated, low_threshold, high_threshold))[:, :, 0] > 0
    )
    if not source_edges.any() or not generated_edges.any():
        return 0.0

    source_distance = cv2.distanceTransform(
        (~source_edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    generated_distance = cv2.distanceTransform(
        (~generated_edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    precision = (source_distance[generated_edges] <= tolerance).mean()
    recall = (generated_distance[source_edges] <= tolerance).mean()
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


@torch.inference_mode()
def dino_cosine(
    source: Image.Image,
    generated: Image.Image,
    processor: BaseImageProcessor,
    model: Dinov2Model,
) -> float:
    inputs = processor(
        images=[prepare_image(source), prepare_image(generated)], return_tensors="pt"
    ).to(model.device)
    features = F.normalize(model(**inputs).last_hidden_state[:, 0], dim=-1)
    return (features[0] @ features[1]).item()
