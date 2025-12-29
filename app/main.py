"""
FastAPI application exposing an image similarity endpoint backed by CLIP.
"""

import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.models import SimilarityResponse
from app.similarity import (
    ALLOWED_MODELS,
    ALLOWED_MODES,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODE,
    ModelRegistry,
    compute_hash_similarity,
    compute_hybrid_score,
    select_device,
)

app = FastAPI(
    title="Image Similarity API",
    description="Compare two images using CLIP embeddings and return a similarity score.",
    version="0.1.0",
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

similarity_registry: Optional[ModelRegistry] = None


@app.on_event("startup")
async def load_model() -> None:
    """Load the CLIP model once at startup."""
    global similarity_registry
    similarity_registry = ModelRegistry(device=select_device())
    similarity_registry.get(DEFAULT_MODEL_NAME)


@app.get("/", response_class=FileResponse, status_code=status.HTTP_200_OK)
async def index() -> FileResponse:
    """Serve the simple UI for uploading images."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/compare", response_model=SimilarityResponse, status_code=status.HTTP_200_OK)
async def compare_images(
    image1: UploadFile = File(..., description="First image file"),
    image2: UploadFile = File(..., description="Second image file"),
    model_name: str = Form(DEFAULT_MODEL_NAME, description="CLIP model identifier"),
    mode: str = Form(DEFAULT_MODE, description="Similarity mode: semantic, strict, hybrid"),
) -> SimilarityResponse:
    """
    Accept two image uploads and return their cosine similarity.
    """
    if similarity_registry is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not image1.content_type.startswith("image") or not image2.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Both files must be valid images")

    if model_name not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model name")

    if mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail="Unsupported similarity mode")

    try:
        image_bytes_1 = await image1.read()
        image_bytes_2 = await image2.read()

        pil_image_1 = Image.open(io.BytesIO(image_bytes_1)).convert("RGB")
        pil_image_2 = Image.open(io.BytesIO(image_bytes_2)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Unable to read image files") from exc

    hash_similarity = compute_hash_similarity(pil_image_1, pil_image_2)
    semantic_similarity = None
    score = None

    if mode in {"semantic", "hybrid"}:
        model = similarity_registry.get(model_name)
        semantic_similarity = model.compute_similarity(pil_image_1, pil_image_2)
    else:
        model = None

    if mode == "semantic":
        score = semantic_similarity
    elif mode == "strict":
        score = (hash_similarity * 2.0) - 1.0
    else:
        score = compute_hybrid_score(semantic_similarity, hash_similarity)

    return SimilarityResponse(
        similarity_score=score,
        device=similarity_registry.device,
        model_name=model_name,
        mode=mode,
        semantic_similarity=semantic_similarity,
        hash_similarity=hash_similarity,
    )


@app.get("/models", status_code=status.HTTP_200_OK)
async def list_models() -> dict[str, object]:
    """Return supported CLIP models for the UI."""
    return {
        "default": DEFAULT_MODEL_NAME,
        "models": [{"name": name, "label": label} for name, label in ALLOWED_MODELS.items()],
        "default_mode": DEFAULT_MODE,
        "modes": [{"name": name, "label": label} for name, label in ALLOWED_MODES.items()],
    }


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    """Lightweight health check endpoint."""
    device = similarity_registry.device if similarity_registry else "uninitialized"
    return {"status": "ok", "device": device}
