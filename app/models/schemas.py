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
    feature_similarity: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Feature match similarity in [0, 1]."
    )
    feature_method: Optional[str] = Field(default=None, description="Feature method used when mode=feature.")
    feature_match_count: Optional[int] = Field(default=None, ge=0, description="Number of good feature matches.")
    feature_inlier_count: Optional[int] = Field(default=None, ge=0, description="Inlier matches after RANSAC.")


class MatchResponse(BaseModel):
    """Response payload for feature matching visualization."""

    method: str = Field(..., description="Feature matching method used.")
    match_count: int = Field(..., ge=0, description="Number of good matches after filtering.")
    keypoints_a: int = Field(..., ge=0, description="Keypoints detected in the first image.")
    keypoints_b: int = Field(..., ge=0, description="Keypoints detected in the second image.")
    image_base64: str = Field(..., description="Base64-encoded PNG match visualization.")
