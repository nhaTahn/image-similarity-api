"""
Feature matching utilities (ORB/SIFT) for visualizing correspondences.
"""

from __future__ import annotations

import base64
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

DEFAULT_MATCH_METHOD = "orb"
ALLOWED_MATCH_METHODS = {
    "orb": "ORB (fast)",
    "sift": "SIFT (higher quality)",
}
RATIO_TEST = 0.75
INLIER_WEIGHT = 0.6


def _to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _create_detector(method: str) -> cv2.Feature2D:
    if method == "orb":
        return cv2.ORB_create(nfeatures=800)
    if method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise ValueError("SIFT not available in this OpenCV build")
        return cv2.SIFT_create(nfeatures=800)
    raise ValueError(f"Unsupported match method: {method}")


def _create_matcher(method: str) -> cv2.BFMatcher:
    norm = cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2
    return cv2.BFMatcher(norm, crossCheck=False)


def _side_by_side(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    height = max(image_a.shape[0], image_b.shape[0])

    def pad_to_height(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == height:
            return image
        padding = height - image.shape[0]
        return cv2.copyMakeBorder(
            image,
            0,
            padding,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=(240, 240, 240),
        )

    padded_a = pad_to_height(image_a)
    padded_b = pad_to_height(image_b)
    return cv2.hconcat([padded_a, padded_b])


def _compute_good_matches(
    bgr_a: np.ndarray,
    bgr_b: np.ndarray,
    method: str,
) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
    detector = _create_detector(method)
    keypoints_a, descriptors_a = detector.detectAndCompute(bgr_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(bgr_b, None)
    keypoints_a = keypoints_a or []
    keypoints_b = keypoints_b or []

    if descriptors_a is None or descriptors_b is None:
        return keypoints_a, keypoints_b, []

    matcher = _create_matcher(method)
    raw_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = []
    for first, second in raw_matches:
        if first.distance < RATIO_TEST * second.distance:
            good_matches.append(first)
    return keypoints_a, keypoints_b, good_matches


def compute_feature_similarity(
    image_a: Image.Image,
    image_b: Image.Image,
    method: str = DEFAULT_MATCH_METHOD,
) -> Tuple[float, int, int, int, int]:
    """
    Return a feature similarity score in [0, 1] with match statistics.
    """
    bgr_a = _to_bgr(image_a)
    bgr_b = _to_bgr(image_b)

    keypoints_a, keypoints_b, good_matches = _compute_good_matches(bgr_a, bgr_b, method)
    keypoints_a_count = len(keypoints_a)
    keypoints_b_count = len(keypoints_b)
    match_count = len(good_matches)

    min_keypoints = min(keypoints_a_count, keypoints_b_count)
    if min_keypoints == 0 or match_count == 0:
        return 0.0, match_count, keypoints_a_count, keypoints_b_count, 0

    match_ratio = match_count / min_keypoints
    inlier_count = 0

    if match_count >= 4:
        src_pts = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_count = int(mask.sum())

    inlier_ratio = inlier_count / match_count if match_count else 0.0
    score = (INLIER_WEIGHT * inlier_ratio) + ((1.0 - INLIER_WEIGHT) * match_ratio)
    return min(1.0, score), match_count, keypoints_a_count, keypoints_b_count, inlier_count


def compute_match_map(
    image_a: Image.Image,
    image_b: Image.Image,
    method: str = DEFAULT_MATCH_METHOD,
) -> Tuple[str, int, int, int]:
    """
    Return a base64 PNG image with match lines plus match/keypoint counts.
    """
    bgr_a = _to_bgr(image_a)
    bgr_b = _to_bgr(image_b)
    keypoints_a, keypoints_b, good_matches = _compute_good_matches(bgr_a, bgr_b, method)

    if not good_matches:
        rendered = _side_by_side(bgr_a, bgr_b)
        match_count = 0
    else:
        match_count = len(good_matches)
        rendered = cv2.drawMatches(
            bgr_a,
            keypoints_a,
            bgr_b,
            keypoints_b,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    ok, buffer = cv2.imencode(".png", rendered)
    if not ok:
        raise RuntimeError("Failed to encode match map image")

    image_base64 = base64.b64encode(buffer).decode("ascii")
    return image_base64, match_count, len(keypoints_a), len(keypoints_b)
