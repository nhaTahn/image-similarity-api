"""
Utilities to load a CLIP model and compute similarity between two images.
"""

from __future__ import annotations

import platform
import threading
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_MODE = "semantic"
HASH_SIZE = 8
HYBRID_SEMANTIC_WEIGHT = 0.7
ALLOWED_MODELS = {
    "openai/clip-vit-base-patch32": "CLIP ViT-B/32 (fast)",
    "openai/clip-vit-large-patch14": "CLIP ViT-L/14 (higher quality)",
}
ALLOWED_MODES = {
    "semantic": "Semantic (CLIP)",
    "strict": "Strict (image hash)",
    "hybrid": "Hybrid (CLIP + hash)",
    "feature": "Feature (ORB/SIFT match)",
}


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


def _dhash(image: Image.Image, hash_size: int = HASH_SIZE) -> np.ndarray:
    """
    Difference hash for quick pixel-level similarity.
    Returns a boolean matrix of shape (hash_size, hash_size).
    """
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized)
    return pixels[:, 1:] > pixels[:, :-1]


def compute_hash_similarity(image_a: Image.Image, image_b: Image.Image) -> float:
    """Return image hash similarity in [0, 1]."""
    hash_a = _dhash(image_a)
    hash_b = _dhash(image_b)
    distance = np.count_nonzero(hash_a != hash_b)
    total = hash_a.size
    return 1.0 - (distance / total)


def normalize_score(score: float) -> float:
    """Map cosine similarity [-1, 1] to [0, 1]."""
    return (score + 1.0) / 2.0


def denormalize_score(score: float) -> float:
    """Map [0, 1] scores back to [-1, 1]."""
    return (score * 2.0) - 1.0


def compute_hybrid_score(semantic_score: float, hash_score: float) -> float:
    """Blend semantic and hash scores on a normalized scale."""
    semantic_norm = normalize_score(semantic_score)
    blended = (semantic_norm * HYBRID_SEMANTIC_WEIGHT) + (hash_score * (1.0 - HYBRID_SEMANTIC_WEIGHT))
    return denormalize_score(blended)


class ModelRegistry:
    """Cache CLIP models so we can switch without reloading every request."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = select_device(device)
        self._models: dict[str, ImageSimilarityModel] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str) -> ImageSimilarityModel:
        if model_name not in ALLOWED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")

        if model_name in self._models:
            return self._models[model_name]

        with self._lock:
            if model_name in self._models:
                return self._models[model_name]
            model = ImageSimilarityModel(model_name=model_name, device=self.device)
            self._models[model_name] = model
            return model
