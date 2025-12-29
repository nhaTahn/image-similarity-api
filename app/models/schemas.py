from pydantic import BaseModel, Field


class SimilarityResponse(BaseModel):
    """Response payload for image similarity results."""

    similarity_score: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity between the two images.")
    device: str = Field(..., description="Torch device used for inference.")
    model_name: str = Field(..., description="Identifier of the CLIP model used.")

