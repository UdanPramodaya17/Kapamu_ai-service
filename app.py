"""
SmartSalon — FastAPI Application
=================================
Endpoints:
  GET  /          → health check
  POST /predict   → multipart image upload → face-shape + hairstyle JSON
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Read config for max upload size
_CONFIG_PATH = Path(__file__).parent / "models" / "inference_config_v5.json"
with open(_CONFIG_PATH) as _f:
    _CONFIG = json.load(_f)

MAX_UPLOAD_BYTES: int = _CONFIG.get("max_upload_bytes", 5 * 1024 * 1024)  # default 5 MB
MODEL_VERSION: str = _CONFIG.get("model_version", "ensemble-v5")

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

app = FastAPI(
    title="SmartSalon Face Shape API",
    description=(
        "Upload a face photo to get your face shape classification "
        "and personalised hairstyle recommendations."
    ),
    version="1.0.0",
)

# Allow the React dev server (and any localhost port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "https://kapamu-frontend.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model_version": MODEL_VERSION,
            "confidence_threshold": _CONFIG.get("confidence_threshold"),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
        }
    )


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.post("/predict", summary="Predict face shape and get hairstyle recommendations")
async def predict(
    file: UploadFile = File(..., description="Face photo (JPEG, PNG, WebP, BMP, TIFF)"),
    gender: str = Form(
        default="all",
        description="Filter hairstyle suggestions: 'all' | 'women' | 'men'",
    ),
) -> JSONResponse:
    # --- Validate content type ---
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=415,
            content={
                "error": True,
                "code": "unsupported_media_type",
                "message": (
                    f"Unsupported file type '{content_type}'. "
                    f"Please upload a JPEG, PNG, WebP, BMP, or TIFF image."
                ),
            },
        )

    # --- Validate gender parameter ---
    gender = gender.strip().lower()
    if gender not in ("all", "women", "men"):
        return JSONResponse(
            status_code=422,
            content={
                "error": True,
                "code": "invalid_gender",
                "message": "gender must be 'all', 'women', or 'men'.",
            },
        )

    # --- Read & size-check image bytes ---
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "code": "empty_file",
                "message": "Uploaded file is empty.",
            },
        )
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        return JSONResponse(
            status_code=413,
            content={
                "error": True,
                "code": "file_too_large",
                "message": f"File exceeds the {mb:.0f} MB limit.",
            },
        )

    # --- Run inference (all errors are caught and returned as JSON) ---
    try:
        from inference import predict_face_shape_and_recommend

        result = predict_face_shape_and_recommend(image_bytes, gender=gender)
    except Exception as exc:  # noqa: BLE001
        # Never expose a raw stack trace to the client
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "code": "inference_error",
                "message": f"Inference failed: {type(exc).__name__}: {exc}",
            },
        )

    # Errors returned as dicts (no_face, no_landmarks, etc.) get a 422
    if result.get("error"):
        return JSONResponse(status_code=422, content=result)

    return JSONResponse(content=result)
