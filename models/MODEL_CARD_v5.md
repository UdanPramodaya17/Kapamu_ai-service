# SmartSalon Face-Shape Model Card (v5 — Ensemble + Extended Geometry + Focal Loss + TTA)

## Final held-out results (ensemble, with TTA)
- Model A (EfficientNetV2B0) alone: 0.7340
- Model B (ResNet50V2) alone: 0.7550
- **Ensemble accuracy: 0.7740**
- **Ensemble macro-F1: 0.7698**
- Test images: 1000
- Uncertainty threshold: 0.48

## Architecture
Two independently-trained hybrid (image + 12 geometric ratios) models:
1. EfficientNetV2B0 backbone
2. ResNet50V2 backbone (different family, for ensemble diversity)
Both use focal loss (gamma=2.0) instead of plain cross-entropy.

## IMPORTANT for deployment — reproduce EXACTLY in this order:
1. MTCNN face detection + crop (35% margin)
2. MediaPipe FaceLandmarker (Tasks API) on the crop
3. Compute all 12 geometric ratios: jaw_width_ratio, cheekbone_width_ratio, forehead_width_ratio, jaw_to_cheek_ratio, forehead_to_cheek_ratio, jaw_to_forehead_ratio, length_to_cheek_ratio, chin_width_ratio, jaw_taper_ratio, upper_face_ratio, lower_face_ratio, jaw_angle_norm
4. Standardize with the SAVED scaler (geometric_feature_scaler_v5.pkl) — do not refit
5. For EACH model: run inference on the crop AND its horizontal flip, average
   the two softmax outputs (test-time augmentation)
6. Average Model A's TTA output and Model B's TTA output (ensemble)
7. Apply confidence_threshold — below it, return "uncertain" to the user

## Important limitations
- Face-shape labels are subjective and overlapping rather than medical facts.
- Dataset is celebrity-focused and predominantly female; demographic
  generalization is not guaranteed.
- Inference is ~2x slower than a single model due to the ensemble + TTA
  (4 forward passes per image total). Acceptable for a consultation app,
  not for real-time video.
