"""Live snapshot: face + anti-spoof on camera frame."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import face_antispoof  # noqa: E402


def main() -> int:
    snap = config.DATA_DIR / "test_snap_cam73.jpg"
    if not snap.exists():
        print("FAIL: snapshot yo'q", snap)
        return 1
    img = cv2.imread(str(snap))
    if img is None:
        print("FAIL: snapshot o'qilmadi")
        return 1
    print(f"snapshot: {img.shape[1]}x{img.shape[0]}")

    from insightface.app import FaceAnalysis

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)
    faces = app.get(img)
    print(f"faces detected: {len(faces)}")

    asp = face_antispoof.AntiSpoofEnsemble(config)
    for i, face in enumerate(faces[:5]):
        x1, y1, x2, y2 = [int(v) for v in face.bbox.tolist()]
        r = asp.predict(img, (x1, y1, x2, y2))
        emb_score = float(np.max(face.normed_embedding @ face.normed_embedding))  # dummy
        print(
            f"  face{i+1}: bbox=({x1},{y1},{x2},{y2}) det={face.det_score:.2f} "
            f"antispoof real={r.real_prob:.3f} is_real={r.is_real}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
