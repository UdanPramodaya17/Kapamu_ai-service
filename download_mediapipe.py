"""
download_mediapipe.py
---------------------
One-time helper: downloads the MediaPipe FaceLandmarker model file
into models/face_landmarker.task.

Run once before starting the server:
    python download_mediapipe.py
"""

import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DEST = Path(__file__).parent / "models" / "face_landmarker.task"


def download() -> None:
    if DEST.exists():
        print(f"✓ Already exists: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"Downloading MediaPipe FaceLandmarker model …\n  {URL}")
    DEST.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(URL, DEST, reporthook=_reporthook)
    print(f"\n✓ Saved to {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    download()
