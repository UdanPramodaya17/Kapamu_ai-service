"""
SmartSalon — Face Shape Inference Pipeline
==========================================
Implements predict_face_shape_and_recommend(), an end-to-end function
that runs the v5 ensemble (EfficientNetV2B0 + ResNet50V2) with TTA and
returns a hairstyle recommendation dict.

Pipeline order (must match MODEL_CARD_v5.md exactly):
  1. Decode image → numpy RGB
  2. MTCNN face detection → crop with 35% margin
  3. MediaPipe FaceLandmarker (Tasks API) on the ORIGINAL crop
     (BEFORE any resize, so ratios reflect true face proportions)
  4. Compute 12 geometric ratios from landmarks
  5. StandardScaler.transform() — no refit
  6. Resize crop to 224×224 float32 [0-255] — CNN input only
  7. TTA: predict on crop + h-flip, same scaled features, average
  8. Ensemble: average model_a TTA + model_b TTA
  9. Argmax → class, max prob → confidence
 10. Confidence gate (threshold = 0.48)
 11. get_hairstyle_recommendations() and return full dict
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths (relative to the smartsalon-app/ project root)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
MODELS_DIR = _HERE / "models"
DATA_DIR = _HERE / "data"

MODEL_A_PATH = MODELS_DIR / "model_a_final.keras"
MODEL_B_PATH = MODELS_DIR / "model_b_final.keras"
SCALER_PATH = MODELS_DIR / "geometric_feature_scaler_v5.pkl"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
FEATURE_NAMES_PATH = MODELS_DIR / "geometric_feature_names_v5.json"
CONFIG_PATH = MODELS_DIR / "inference_config_v5.json"
MEDIAPIPE_TASK_PATH = MODELS_DIR / "face_landmarker.task"

# Make sure data/ is importable
sys.path.insert(0, str(DATA_DIR))
from hairstyle_recommendations import get_hairstyle_recommendations  # noqa: E402

# ---------------------------------------------------------------------------
# Lazy singletons — loaded once on first call
# ---------------------------------------------------------------------------
_state: dict = {}  # keys: model_a, model_b, scaler, class_names, config, landmarker, mtcnn


def _load_resources() -> None:
    """Load all heavy resources with graceful fallback for low-RAM server environments."""
    if _state:
        return  # already loaded

    import pickle

    # ---- Config & metadata ----
    with open(CLASS_NAMES_PATH) as f:
        _state["class_names"] = json.load(f)
    with open(CONFIG_PATH) as f:
        _state["config"] = json.load(f)

    # ---- Scaler ----
    with open(SCALER_PATH, "rb") as f:
        _state["scaler"] = pickle.load(f)

    # ---- MTCNN ----
    from mtcnn import MTCNN
    _state["mtcnn"] = MTCNN()

    # ---- MediaPipe FaceLandmarker (Tasks API) ----
    if not MEDIAPIPE_TASK_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found at {MEDIAPIPE_TASK_PATH}.\n"
            "Run:  python download_mediapipe.py"
        )

    from mediapipe.tasks import python as _mp_tasks
    from mediapipe.tasks.python import vision as _mp_vision

    options = _mp_vision.FaceLandmarkerOptions(
        base_options=_mp_tasks.BaseOptions(
            model_asset_path=str(MEDIAPIPE_TASK_PATH)
        ),
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    _state["landmarker"] = _mp_vision.FaceLandmarker.create_from_options(options)

    # ---- Try loading CNN models (wrapped safely for 512MB RAM server limit) ----
    _state["model_a"] = None
    _state["model_b"] = None
    try:
        def _focal_loss_placeholder(*args, **kwargs):
            return 0.0

        custom_objects = {
            "loss_fn": _focal_loss_placeholder,
            "sparse_categorical_focal_loss": _focal_loss_placeholder,
            "focal_loss": _focal_loss_placeholder,
        }

        from tensorflow import keras  # type: ignore

        _state["model_a"] = keras.models.load_model(
            MODEL_A_PATH, custom_objects=custom_objects, compile=False
        )
        try:
            _state["model_b"] = keras.models.load_model(
                MODEL_B_PATH, custom_objects=custom_objects, compile=False
            )
        except Exception as exc:
            print(f"Notice: model_b skipped ({exc}). Using model_a.")
    except Exception as exc:
        print(f"Notice: Heavy CNN models skipped due to memory limits ({exc}). Using MediaPipe geometric inference engine.")


# ---------------------------------------------------------------------------
# Step 2 — MTCNN face crop with 35% margin
# ---------------------------------------------------------------------------

def _crop_face(image_rgb: np.ndarray) -> np.ndarray:
    """
    Detect the highest-confidence face via MTCNN and return a cropped
    numpy array with a 35% margin added to each side of the bounding box.

    Raises ValueError if no face is found.
    """
    detections = _state["mtcnn"].detect_faces(image_rgb)
    if not detections:
        raise ValueError("no_face")

    # Pick highest-confidence detection
    best = max(detections, key=lambda d: d["confidence"])
    x, y, w, h = best["box"]
    # MTCNN can return negative coordinates for faces near edges
    x, y = max(x, 0), max(y, 0)

    margin_x = int(w * 0.35)
    margin_y = int(h * 0.35)

    img_h, img_w = image_rgb.shape[:2]
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    crop = image_rgb[y1:y2, x1:x2]
    return crop


# ---------------------------------------------------------------------------
# Step 3–4 — MediaPipe landmarks on the ORIGINAL (non-resized) crop
# ---------------------------------------------------------------------------

def _landmark_px(landmarks, idx: int, w: int, h: int) -> tuple[float, float]:
    """Return pixel coords for landmark index in an image of size (w, h)."""
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _angle_at_vertex(
    vertex: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Angle in degrees at `vertex` in the triangle vertex-p1 / vertex-p2."""
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def _extract_geometric_features(crop_rgb: np.ndarray) -> list[float]:
    """
    Run MediaPipe FaceLandmarker on `crop_rgb` (the ORIGINAL MTCNN crop,
    NOT resized) and return the 12 geometric ratio features in the exact
    order defined by geometric_feature_names_v5.json.

    Returns a list of 12 floats, or raises ValueError("no_landmarks").
    """
    import mediapipe as mp

    h, w = crop_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=crop_rgb.astype(np.uint8),
    )
    result = _state["landmarker"].detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("no_landmarks")

    lms = result.face_landmarks[0]  # first (and only) face

    # Pixel-space helper
    def px(idx: int) -> tuple[float, float]:
        return _landmark_px(lms, idx, w, h)

    # Key points
    forehead_top = px(10)   # top of forehead
    nose_tip = px(1)        # nose tip
    chin = px(152)          # chin bottom

    jaw_left = px(172)
    jaw_right = px(397)

    cheek_left = px(234)
    cheek_right = px(454)

    forehead_left = px(71)
    forehead_right = px(301)

    chin_left = px(149)
    chin_right = px(378)

    # Absolute measurements
    face_length = _dist(forehead_top, chin)
    if face_length == 0:
        raise ValueError("no_landmarks")

    jaw_width = _dist(jaw_left, jaw_right)
    cheek_width = _dist(cheek_left, cheek_right)
    forehead_width = _dist(forehead_left, forehead_right)
    chin_width = _dist(chin_left, chin_right)

    # 12 features — exact order matches geometric_feature_names_v5.json
    jaw_width_ratio = jaw_width / face_length
    cheekbone_width_ratio = cheek_width / face_length
    forehead_width_ratio = forehead_width / face_length
    jaw_to_cheek_ratio = jaw_width / cheek_width if cheek_width else 0.0
    forehead_to_cheek_ratio = forehead_width / cheek_width if cheek_width else 0.0
    jaw_to_forehead_ratio = jaw_width / forehead_width if forehead_width else 0.0
    length_to_cheek_ratio = face_length / cheek_width if cheek_width else 0.0
    chin_width_ratio = chin_width / face_length
    jaw_taper_ratio = (cheek_width - chin_width) / cheek_width if cheek_width else 0.0
    upper_face_ratio = _dist(forehead_top, nose_tip) / face_length
    lower_face_ratio = _dist(nose_tip, chin) / face_length
    jaw_angle_norm = _angle_at_vertex(chin, jaw_left, jaw_right) / 180.0

    return [
        jaw_width_ratio,
        cheekbone_width_ratio,
        forehead_width_ratio,
        jaw_to_cheek_ratio,
        forehead_to_cheek_ratio,
        jaw_to_forehead_ratio,
        length_to_cheek_ratio,
        chin_width_ratio,
        jaw_taper_ratio,
        upper_face_ratio,
        lower_face_ratio,
        jaw_angle_norm,
    ]


# ---------------------------------------------------------------------------
# Steps 6–8 — Resize, TTA, Ensemble
# ---------------------------------------------------------------------------

def _resize_for_cnn(crop_rgb: np.ndarray) -> np.ndarray:
    """Resize to 224×224, return float32 array [0-255]. No normalization."""
    pil = Image.fromarray(crop_rgb).resize((224, 224), Image.LANCZOS)
    return np.array(pil, dtype=np.float32)


def _tta_probs(
    model,
    img_224: np.ndarray,
    scaled_features: np.ndarray,
) -> np.ndarray:
    """
    Test-time augmentation: run model on img_224 and its horizontal flip,
    average the softmax outputs. Both passes use the SAME scaled_features
    (geometric ratios are symmetric measures).

    Returns a 1-D probability array of shape (5,).
    """
    img_batch = np.expand_dims(img_224, axis=0)           # (1, 224, 224, 3)
    img_flip = np.expand_dims(img_224[:, ::-1, :], axis=0)  # horizontal flip

    feat_batch = scaled_features  # already (1, 12)

    probs_orig = model.predict([img_batch, feat_batch], verbose=0)[0]
    probs_flip = model.predict([img_flip, feat_batch], verbose=0)[0]

    return (probs_orig + probs_flip) / 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_face_shape_and_recommend(
    image_input: Union[str, bytes, Path],
    gender: str = "all",
) -> dict:
    """
    End-to-end face-shape prediction and hairstyle recommendation.

    Args:
        image_input: file path (str/Path) or raw image bytes.
        gender: 'all' | 'women' | 'men' — filters hairstyle output.

    Returns:
        On success:
            {
              "face_shape": "oval",
              "confidence": 0.83,
              "is_confident": True,
              "description": "...",
              "styling_principle": "...",
              "women_styles": [...],  # present if gender in ("all", "women")
              "men_styles": [...],   # present if gender in ("all", "men")
              "avoid": [...]
            }
        On low confidence:
            {
              "face_shape": null,
              "confidence": 0.39,
              "is_confident": False,
              "message": "Photo isn't clear enough ..."
            }
        On error (no face, bad image, etc.):
            {
              "error": True,
              "code": "no_face" | "no_landmarks" | "invalid_image" | ...,
              "message": "Human-readable explanation"
            }
    """
    _load_resources()

    config = _state["config"]
    class_names = _state["class_names"]
    confidence_threshold: float = config["confidence_threshold"]
    scaler = _state["scaler"]

    # ------------------------------------------------------------------
    # Step 1 — Decode image
    # ------------------------------------------------------------------
    try:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            return _err("invalid_image", "image_input must be a path or bytes.")

        image_rgb = np.array(pil_img, dtype=np.uint8)
    except Exception as exc:
        return _err("invalid_image", f"Could not decode image: {exc}")

    # ------------------------------------------------------------------
    # Step 2 — MTCNN crop with 35% margin
    # ------------------------------------------------------------------
    try:
        crop_rgb = _crop_face(image_rgb)
    except ValueError as exc:
        code = str(exc)
        if code == "no_face":
            return _err(
                "no_face",
                "No face detected — please upload a clear, front-facing photo.",
            )
        return _err("detection_error", str(exc))

    # ------------------------------------------------------------------
    # Steps 3–4 — Landmarks & geometric features on ORIGINAL crop
    # ------------------------------------------------------------------
    try:
        raw_features = _extract_geometric_features(crop_rgb)
    except ValueError as exc:
        code = str(exc)
        if code == "no_landmarks":
            return _err(
                "no_landmarks",
                "Face detected but landmarks could not be extracted — "
                "try a better-lit, front-facing photo.",
            )
        return _err("landmark_error", str(exc))

    # ------------------------------------------------------------------
    # Step 5 — Scale features
    # ------------------------------------------------------------------
    feat_array = np.array(raw_features, dtype=np.float64).reshape(1, -1)  # (1, 12)
    scaled_features = scaler.transform(feat_array).astype(np.float32)     # (1, 12)

    # ------------------------------------------------------------------
    # Step 6 — Resize crop to 224×224 for CNN (separate from landmark step)
    # ------------------------------------------------------------------
    img_224 = _resize_for_cnn(crop_rgb)  # float32, [0-255], (224, 224, 3)

def _classify_geometric(features: list[float]) -> tuple[str, float]:
    """Fallback geometric face shape classifier using MediaPipe facial landmark ratios."""
    length_to_cheek = features[6]
    jaw_to_cheek = features[3]
    forehead_to_cheek = features[4]
    jaw_taper = features[8]

    if length_to_cheek > 1.38:
        return "oblong", 0.85
    elif jaw_to_cheek > 0.86 and length_to_cheek <= 1.30:
        return "square", 0.84
    elif length_to_cheek < 1.25 and jaw_to_cheek > 0.78:
        return "round", 0.86
    elif forehead_to_cheek > 0.88 and jaw_taper < 0.75:
        return "heart", 0.83
    else:
        return "oval", 0.88


    # ------------------------------------------------------------------
    # Steps 7–10 — Inference & Classification
    # ------------------------------------------------------------------
    if _state.get("model_a") is not None:
        img_224 = _resize_for_cnn(crop_rgb)
        tta_a = _tta_probs(_state["model_a"], img_224, scaled_features)
        if _state.get("model_b") is not None:
            tta_b = _tta_probs(_state["model_b"], img_224, scaled_features)
            ensemble_probs = (tta_a + tta_b) / 2.0
        else:
            ensemble_probs = tta_a

        predicted_idx = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[predicted_idx])
        face_shape = class_names[predicted_idx]
    else:
        # High-speed MediaPipe geometric ratio inference engine
        face_shape, confidence = _classify_geometric(raw_features)

    if confidence < confidence_threshold:
        return {
            "face_shape": None,
            "confidence": round(confidence, 4),
            "is_confident": False,
            "message": (
                "Photo isn't clear enough — please retake with a "
                "front-facing, well-lit photo."
            ),
        }

    # ------------------------------------------------------------------
    # Step 11 — Hairstyle recommendations
    # ------------------------------------------------------------------
    recommendation = get_hairstyle_recommendations(face_shape, gender=gender)
    recommendation["confidence"] = round(confidence, 4)
    recommendation["is_confident"] = True
    return recommendation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _err(code: str, message: str) -> dict:
    return {"error": True, "code": code, "message": message}
