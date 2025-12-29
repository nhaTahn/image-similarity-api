"""
FastAPI application exposing an image similarity endpoint backed by CLIP.
"""

import io
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image

from app.models import SimilarityResponse
from app.similarity import ImageSimilarityModel, select_device

app = FastAPI(
    title="Image Similarity API",
    description="Compare two images using CLIP embeddings and return a similarity score.",
    version="0.1.0",
)

similarity_model: Optional[ImageSimilarityModel] = None


@app.on_event("startup")
async def load_model() -> None:
    """Load the CLIP model once at startup."""
    global similarity_model
    similarity_model = ImageSimilarityModel(device=select_device())


@app.post("/compare", response_model=SimilarityResponse, status_code=status.HTTP_200_OK)
async def compare_images(
    image1: UploadFile = File(..., description="First image file"),
    image2: UploadFile = File(..., description="Second image file"),
) -> SimilarityResponse:
    """
    Accept two image uploads and return their cosine similarity.
    """
    if similarity_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not image1.content_type.startswith("image") or not image2.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Both files must be valid images")

    try:
        image_bytes_1 = await image1.read()
        image_bytes_2 = await image2.read()

        pil_image_1 = Image.open(io.BytesIO(image_bytes_1)).convert("RGB")
        pil_image_2 = Image.open(io.BytesIO(image_bytes_2)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Unable to read image files") from exc

    score = similarity_model.compute_similarity(pil_image_1, pil_image_2)
    return SimilarityResponse(
        similarity_score=score,
        device=similarity_model.device,
        model_name=similarity_model.model_name,
    )


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    """Lightweight health check endpoint."""
    device = similarity_model.device if similarity_model else "uninitialized"
    return {"status": "ok", "device": device}

