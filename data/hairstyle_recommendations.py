# =============================================================================
# SmartSalon — Hairstyle Recommendation Engine
#
# Takes the face-shape prediction from the v5 ensemble model
# (heart / oblong / oval / round / square + confidence score) and returns
# curated hairstyle recommendations for men and women, plus styling
# principles and cuts to avoid.
#
# This is pure lookup/domain-knowledge logic — no ML needed here, it plugs
# directly onto the output of your v5 inference pipeline.
# =============================================================================

HAIRSTYLE_RECOMMENDATIONS = {

    "oval": {
        "description": (
            "Oval face shape is considered the most balanced — forehead, "
            "cheekbones and jaw are close to equal width, with a length "
            "somewhat longer than the width. Most hairstyles suit this shape."
        ),
        "styling_principle": (
            "Maintain the natural balance — avoid styles that add excessive "
            "height or width, since the proportions are already even."
        ),
        "women": [
            "Long layers with soft waves",
            "Blunt bob (chin-length or shoulder-length)",
            "Sleek high ponytail",
            "Curtain bangs / fringe",
            "Pixie cut (oval faces carry short cuts especially well)",
        ],
        "men": [
            "Classic side part",
            "Crew cut",
            "Textured quiff",
            "Slicked-back undercut",
            "Buzz cut",
        ],
        "avoid": [
            "Heavy, blunt bangs that shorten the face excessively",
            "Styles that add too much height on top (can make the face look longer)",
        ],
    },

    "round": {
        "description": (
            "Round face shape has soft curves with similar width and length, "
            "full cheeks, and a rounded jawline with no sharp angles."
        ),
        "styling_principle": (
            "Add height at the crown and length/angles to visually elongate "
            "the face; avoid width at the cheekbone level."
        ),
        "women": [
            "Long layered cuts past the shoulders",
            "High volume at the crown with a side part",
            "Asymmetrical bob (longer on one side)",
            "Long side-swept bangs",
            "Voluminous top with tapered/undercut sides",
        ],
        "men": [
            "Pompadour (height at the crown)",
            "Faux hawk",
            "Textured crop with volume on top",
            "Undercut with longer top",
            "Angular fringe",
        ],
        "avoid": [
            "Chin-length blunt bobs (adds width, emphasizes roundness)",
            "Center parts with no volume",
            "Full, rounded fringes",
            "Very short buzz cuts with no shape on top",
        ],
    },

    "square": {
        "description": (
            "Square face shape has a strong, angular jawline, broad forehead, "
            "and minimal curve — width at the forehead, cheekbone and jaw is "
            "similar."
        ),
        "styling_principle": (
            "Soften the strong jawline with layers, waves, or rounded "
            "silhouettes; avoid styles that add more sharp width."
        ),
        "women": [
            "Soft layered waves starting at the cheekbone",
            "Side-swept fringe",
            "Long layers that fall past the jaw",
            "Textured lob (long bob) with soft ends",
            "Curly or wavy shoulder-length cuts",
        ],
        "men": [
            "Textured crop with soft, tousled top",
            "Side part with soft fringe",
            "Medium-length waves",
            "Fade with longer, textured top (avoid flat-top styles)",
        ],
        "avoid": [
            "Blunt, straight-across bangs",
            "Sleek center-part styles with no volume",
            "Very short buzz cuts that expose the full jawline",
            "Geometric, sharp-edged cuts",
        ],
    },

    "heart": {
        "description": (
            "Heart face shape has a wider forehead and cheekbones tapering "
            "to a narrow, often pointed chin — sometimes with a widow's peak."
        ),
        "styling_principle": (
            "Add width and fullness near the jawline/chin to balance the "
            "wider forehead; avoid extra volume at the crown or temples."
        ),
        "women": [
            "Chin-length bob (adds width where the face narrows)",
            "Side-swept fringe to soften/minimize forehead width",
            "Long layers with waves starting at the jaw",
            "Textured lob with volume at the ends",
        ],
        "men": [
            "Side part with volume kept low, not at the crown",
            "Textured fringe covering part of the forehead",
            "Medium-length cuts with fullness near the jaw/beard line",
            "Beard styling to add visual width at the chin (if applicable)",
        ],
        "avoid": [
            "Slicked-back styles that expose the full, wide forehead",
            "High volume at the crown (exaggerates forehead width)",
            "Very short, tapered-chin cuts with no width at the jaw",
        ],
    },

    "oblong": {
        "description": (
            "Oblong (long) face shape is longer than it is wide, with a "
            "fairly straight cheek line and sometimes a long, narrow chin."
        ),
        "styling_principle": (
            "Add width at the sides and avoid extra length/height on top; "
            "the goal is to visually shorten the face."
        ),
        "women": [
            "Blunt, straight-across bangs (shortens the visual face length)",
            "Chin- to shoulder-length bob with waves for width",
            "Layered cuts with volume at the sides, not the crown",
            "Curly or wavy styles that add horizontal fullness",
        ],
        "men": [
            "Side part with fullness at the sides",
            "Textured crop kept short on top",
            "Fringe/bangs to break up forehead length",
            "Avoid tall pompadours or quiffs",
        ],
        "avoid": [
            "Long, straight styles with no layers (exaggerates length)",
            "High volume at the crown",
            "Very short sides with height on top",
        ],
    },
}


def get_hairstyle_recommendations(face_shape: str, gender: str = "all", top_n: int = 5) -> dict:
    """
    Look up curated hairstyle recommendations for a predicted face shape.

    Args:
        face_shape: one of 'heart', 'oblong', 'oval', 'round', 'square'
                    (must match CLASS_NAMES from the v5 model exactly).
        gender: 'women', 'men', or 'all' (default) to include both.
        top_n: max number of style suggestions to return per gender.

    Returns:
        dict with description, styling_principle, recommended styles, and
        cuts to avoid. Raises ValueError for an unrecognized face_shape.
    """
    face_shape = face_shape.strip().lower()
    if face_shape not in HAIRSTYLE_RECOMMENDATIONS:
        raise ValueError(
            f"Unknown face shape '{face_shape}'. Expected one of: "
            f"{list(HAIRSTYLE_RECOMMENDATIONS.keys())}"
        )

    entry = HAIRSTYLE_RECOMMENDATIONS[face_shape]
    result = {
        "face_shape": face_shape,
        "description": entry["description"],
        "styling_principle": entry["styling_principle"],
        "avoid": entry["avoid"],
    }

    if gender in ("women", "all"):
        result["women_styles"] = entry["women"][:top_n]
    if gender in ("men", "all"):
        result["men_styles"] = entry["men"][:top_n]

    return result


# -----------------------------------------------------------------------------
# Integration with the v5 ensemble inference pipeline
# -----------------------------------------------------------------------------
def get_full_recommendation(
    image_np,
    mtcnn_detector,
    face_landmarker,
    model_a,
    model_b,
    scaler,
    class_names,
    geometric_feature_names,
    confidence_threshold: float,
    gender: str = "all",
) -> dict:
    """
    End-to-end: raw uploaded image -> MTCNN crop -> landmark geometry ->
    ensemble TTA prediction -> hairstyle recommendations.

    This function shows the SHAPE of the full pipeline described in
    MODEL_CARD_v5.md. Wire in your actual crop_face(), extract_geometric_features(),
    and TTA-averaging helper functions from the v5 training script — they are
    not reproduced here to avoid duplicating ~150 lines already defined
    elsewhere in your codebase.

    Returns a dict like:
        {
            "face_shape": "oval",
            "confidence": 0.81,
            "is_confident": True,
            "description": "...",
            "styling_principle": "...",
            "women_styles": [...],
            "men_styles": [...],
            "avoid": [...],
        }
    or, if confidence is below threshold:
        {
            "face_shape": None,
            "confidence": 0.42,
            "is_confident": False,
            "message": "Photo isn't clear enough — please retake with a "
                       "front-facing, well-lit photo."
        }
    """
    # 1. MTCNN crop (use crop_face()/get_or_make_crop() logic from v3/v5 script)
    # cropped_image = crop_face(image_np, mtcnn_detector.detect_faces(image_np)[0]['box'])

    # 2. Extract 12 geometric features (use extract_geometric_features() from v5 script)
    # raw_features = extract_geometric_features(cropped_image)
    # scaled_features = scaler.transform([raw_features])[0]

    # 3. Run both models with TTA and average (use get_tta_probs() logic from v5 script)
    # probs_a = get_tta_probs(model_a, cropped_image, scaled_features)
    # probs_b = get_tta_probs(model_b, cropped_image, scaled_features)
    # ensemble_probs = (probs_a + probs_b) / 2.0

    # 4. Interpret prediction
    # predicted_idx = ensemble_probs.argmax()
    # confidence = float(ensemble_probs[predicted_idx])
    # face_shape = class_names[predicted_idx]

    # --- Placeholder wiring below; replace with the real steps above ---
    raise NotImplementedError(
        "Wire in your actual MTCNN/landmark/TTA/ensemble steps from the v5 "
        "training script here — see the numbered comments above for the "
        "exact order (must match MODEL_CARD_v5.md exactly)."
    )

    # if confidence < confidence_threshold:
    #     return {
    #         "face_shape": None,
    #         "confidence": confidence,
    #         "is_confident": False,
    #         "message": "Photo isn't clear enough — please retake with a "
    #                    "front-facing, well-lit photo.",
    #     }
    #
    # recommendation = get_hairstyle_recommendations(face_shape, gender=gender)
    # recommendation["confidence"] = confidence
    # recommendation["is_confident"] = True
    # return recommendation


if __name__ == "__main__":
    # Quick manual test of the lookup logic (no model needed)
    for shape in HAIRSTYLE_RECOMMENDATIONS:
        rec = get_hairstyle_recommendations(shape, gender="all", top_n=3)
        print(f"\n=== {shape.upper()} ===")
        print("Principle:", rec["styling_principle"])
        print("Women:", rec["women_styles"])
        print("Men:", rec["men_styles"])
        print("Avoid:", rec["avoid"])
