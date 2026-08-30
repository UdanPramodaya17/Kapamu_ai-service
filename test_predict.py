"""
test_predict.py
---------------
Sends a test image to the running SmartSalon /predict endpoint and
prints the JSON response.

Usage:
  # 1. Start the server in another terminal:
  #       uvicorn app:app --reload
  #
  # 2. Run this script with a real face photo:
  #       python test_predict.py path/to/face.jpg
  #       python test_predict.py path/to/face.jpg --gender women
  #
  # Equivalent curl command (shown at the end of each run):
  #       curl -X POST http://localhost:8000/predict \
  #            -F "file=@path/to/face.jpg" \
  #            -F "gender=all"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the SmartSalon /predict endpoint."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to a face photo (JPEG/PNG/WebP).",
    )
    parser.add_argument(
        "--gender",
        default="all",
        choices=["all", "women", "men"],
        help="Filter hairstyle suggestions (default: all).",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running server (default: http://localhost:8000).",
    )
    args = parser.parse_args()

    import requests

    base_url = args.url.rstrip("/")

    # ---- Health check ----
    print(f"\n{'='*60}")
    print(f"  SmartSalon API Test  →  {base_url}")
    print(f"{'='*60}")

    try:
        health = requests.get(f"{base_url}/", timeout=10)
        health.raise_for_status()
        print("\n[GET /] Health check:")
        print(json.dumps(health.json(), indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ Could not reach server at {base_url}: {exc}")
        print("  Start the server first:  uvicorn app:app --reload")
        sys.exit(1)

    # ---- Predict ----
    if args.image is None:
        print(
            "\n⚠  No image path provided — skipping /predict test.\n"
            "   Usage:  python test_predict.py path/to/face.jpg\n"
        )
        return

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"\n✗ Image not found: {image_path}")
        sys.exit(1)

    print(f"\n[POST /predict]  image={image_path}  gender={args.gender}")
    with open(image_path, "rb") as img_f:
        response = requests.post(
            f"{base_url}/predict",
            files={"file": (image_path.name, img_f, "image/jpeg")},
            data={"gender": args.gender},
            timeout=120,
        )

    print(f"HTTP {response.status_code}")
    print(json.dumps(response.json(), indent=2))

    # ---- Show equivalent curl command ----
    print(f"\n{'─'*60}")
    print("Equivalent curl command:")
    print(
        f'  curl -X POST {base_url}/predict \\\n'
        f'       -F "file=@{image_path}" \\\n'
        f'       -F "gender={args.gender}"'
    )
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
