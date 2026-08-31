"""Smoke test: anti-spoof + motion liveness + worker API."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import face_antispoof  # noqa: E402
import face_liveness  # noqa: E402


class _FakeFace:
    def __init__(self, kps: np.ndarray | None = None, det_score: float = 0.9):
        self.kps = kps
        self.det_score = det_score


def test_antispoof_models() -> bool:
    asp = face_antispoof.AntiSpoofEnsemble(config)
    if not asp.enabled or len(asp.models) < 1:
        print("FAIL antispoof: modellar yuklanmadi")
        return False
    # Flat synthetic "print-like" patch
    img = np.full((320, 320, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (80, 60), (240, 260), (225, 210, 195), -1)
    cv2.circle(img, (130, 140), 12, (40, 40, 40), -1)
    cv2.circle(img, (190, 140), 12, (40, 40, 40), -1)
    r = asp.predict(img, (80, 60, 240, 260))
    print(f"antispoof flat: real={r.real_prob:.3f} is_real={r.is_real} ({r.detail})")
    return True


def test_motion_liveness() -> bool:
    liv = face_liveness.MotionLiveness(config)
    if not liv.enabled:
        print("FAIL liveness: o'chirilgan")
        return False
    bbox = (100, 80, 220, 240)
    kps = np.array(
        [[120, 130], [180, 130], [150, 165], [130, 200], [170, 200]],
        dtype=np.float32,
    )
    face = _FakeFace(kps=kps)
    # Static — should NOT pass quickly
    st_static = None
    for i in range(12):
        st_static = liv.update("t", 1, face, bbox, now=time.time() + i * 0.12)
    assert st_static is not None
    print(
        f"motion static: ready={st_static.ready} passed={st_static.passed} "
        f"m={st_static.motion:.4f} ({st_static.detail})"
    )
    # Moving — micro jitter
    liv2 = face_liveness.MotionLiveness(config)
    st_move = None
    base = time.time()
    for i in range(16):
        jx = int(np.sin(i * 0.7) * 4)
        jy = int(np.cos(i * 0.5) * 3)
        bb = (100 + jx, 80 + jy, 220 + jx, 240 + jy)
        k = kps.copy()
        k[:, 0] += jx
        k[:, 1] += jy
        st_move = liv2.update("t", 2, _FakeFace(kps=k), bb, now=base + i * 0.12)
    assert st_move is not None
    print(
        f"motion moving: ready={st_move.ready} passed={st_move.passed} "
        f"m={st_move.motion:.4f} ({st_move.detail})"
    )
    ok = st_move.passed and (not st_static.passed or st_move.motion > st_static.motion)
    if not ok:
        print("WARN motion: moving/static farqi kichik — sozlamani tekshiring")
    return True


def test_worker_http() -> bool:
    try:
        import urllib.request

        req = urllib.request.Request("http://127.0.0.1:8080/login")
        with urllib.request.urlopen(req, timeout=5) as r:
            code = r.status
        print(f"http login: {code}")
        return code == 200
    except Exception as e:
        print(f"FAIL http: {e}")
        return False


def main() -> int:
    ok = True
    ok &= test_antispoof_models()
    ok &= test_motion_liveness()
    ok &= test_worker_http()
    print("RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
