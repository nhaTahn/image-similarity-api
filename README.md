# Image Similarity API

FastAPI service that compares two images using CLIP embeddings and returns a cosine similarity score.

## Features
- `/compare` endpoint accepts two image uploads and returns a similarity score, model name, and device.
- Simple web UI at `/` for drag-and-drop uploads, similarity mode selection, model selection, and live results.
- Automatic device choice: CUDA on Windows (when available), MPS on Apple Silicon macOS, otherwise CPU.
- Dockerfile for containerized runs on macOS and Windows.
- Minimal test client in `tests/test_api.py` that crafts sample images and hits the API.

## Quickstart (local)
1. Create a virtual environment (recommended) and install dependencies:
   ```bash
   cd image-similarity-api
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
   > Note: If you want GPU acceleration, install the torch build appropriate for your platform/CUDA stack before running the app.

2. Start the API:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Call the `/compare` endpoint with two images:
   ```bash
   curl -X POST "http://localhost:8000/compare" \
     -F "image1=@/path/to/image_a.jpg" \
     -F "image2=@/path/to/image_b.jpg"
   ```

4. Optional: open the UI at `http://localhost:8000/` or run the test client (server must be running):
   ```bash
   python tests/test_api.py
   ```

## Docker usage
Build the image (from the project root):
```bash
docker build -t image-similarity-api .
```

Run the container:
```bash
docker run --rm -p 8000:8000 image-similarity-api
```

> GPU passthrough varies by platform/host. The app will fall back to CPU inside the container if no GPU is exposed.

## Project structure
```
app/
  main.py            # FastAPI app and routes
  similarity.py      # CLIP loading + similarity scoring
  static/
    index.html       # Simple UI
    styles.css       # UI styling
    app.js           # UI interactions
  models/
    schemas.py       # Pydantic response models
requirements.txt     # Runtime dependencies
Dockerfile           # Container build
tests/test_api.py    # Simple client for exercising the API
```

## Endpoint
- `POST /compare`: multipart form with `image1` and `image2` files. Returns JSON:
  ```json
  {
    "similarity_score": 0.87,
    "device": "cuda",
    "model_name": "openai/clip-vit-base-patch32",
    "mode": "semantic",
    "semantic_similarity": 0.87,
    "hash_similarity": 0.42
  }
  ```
