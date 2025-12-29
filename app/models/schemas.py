from typing import Optional

from pydantic import BaseModel, Field


class SimilarityResponse(BaseModel):
    """Response payload for image similarity results."""

    similarity_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Similarity score in [-1, 1] for the selected mode."
    )
    device: str = Field(..., description="Torch device used for inference.")
    model_name: str = Field(..., description="Identifier of the CLIP model used.")
    mode: str = Field(..., description="Similarity mode used for the comparison.")
    semantic_similarity: Optional[float] = Field(
        default=None, description="Raw CLIP cosine similarity (semantic mode)."
    )
    hash_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Image hash similarity.")
