"""
Utilities to load a CLIP model and compute similarity between two images.
"""

from __future__ import annotations

import platform
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


def _mps_is_available() -> bool:
    """Detect whether MPS (Apple Silicon GPU) is usable."""
    if not hasattr(torch.backends, "mps"):
        return False
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()


def select_device(preferred: Optional[str] = None) -> str:
    """
    Pick an inference device with sensible fallbacks.

    - Windows: prefer CUDA when available.
    - macOS on Apple Silicon: prefer MPS.
    - Otherwise: CUDA when available, else CPU.
    """
    if preferred:
        return preferred

    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows" and torch.cuda.is_available():
        return "cuda"

    if system == "Darwin" and machine == "arm64" and _mps_is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


class ImageSimilarityModel:
    """Encapsulates CLIP model loading and similarity scoring."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = select_device(device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def compute_similarity(self, image_a: Image.Image, image_b: Image.Image) -> float:
        """
        Compute cosine similarity between two PIL images using CLIP embeddings.
        """
        inputs = self.processor(images=[image_a, image_b], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        # Normalize to unit length then take cosine similarity.
        normalized = torch.nn.functional.normalize(features, p=2, dim=-1)
        similarity = torch.nn.functional.cosine_similarity(normalized[0:1], normalized[1:2])
        return similarity.item()

