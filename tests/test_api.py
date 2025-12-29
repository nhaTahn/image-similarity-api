"""
Lightweight client script to exercise the /compare endpoint.
Run the API locally (e.g., `uvicorn app.main:app --reload`) before executing.
"""

from __future__ import annotations

import io

import requests
from PIL import Image


def _make_image(color: str) -> io.BytesIO:
    """Create an in-memory square PNG of a single color."""
    img = Image.new("RGB", (128, 128), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def main() -> None:
    url = "http://localhost:8000/compare"
    files = {
        "image1": ("img1.png", _make_image("red"), "image/png"),
        "image2": ("img2.png", _make_image("blue"), "image/png"),
    }

    response = requests.post(url, files=files, timeout=30)
    response.raise_for_status()
    print("Similarity response:", response.json())


if __name__ == "__main__":
    main()

