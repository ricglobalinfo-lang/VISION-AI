"""Reject false face detections (furniture, shadows, noise) before unknown-person save."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def _bbox_area(b: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _face_inside_person(
    face_box: tuple[int, int, int, int],
    people: Iterable[tuple[int, int, int, int]],
) -> bool:
    """Face center must lie in a person box, or IoU with person must be reasonable."""
    x1, y1, x2, y2 = face_box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    fw, fh = max(1, x2 - x1), max(1, y2 - y1)
    for px1, py1, px2, py2 in people:
        # Face center in upper 70% of person body
        if px1 <= cx <= px2 and py1 <= cy <= py1 + 0.7 * max(1, py2 - py1):
            # Face shouldn't be huge vs person (false blob)
            pw, ph = max(1, px2 - px1), max(1, py2 - py1)
            if fw <= pw * 0.95 and fh <= ph * 0.75:
                return True
        if _iou(face_box, (px1, py1, px2, py2)) >= 0.08:
            if cy <= py1 + 0.75 * max(1, py2 - py1):
                return True
    return False


def is_real_face(
    face,
    frame_bgr: np.ndarray,
    person_boxes: list[tuple[int, int, int, int]] | None = None,
    *,
    min_det_score: float = 0.68,
    min_side: int = 48,
    min_brightness: float = 28.0,
    min_sharpness: float = 18.0,
    require_person: bool = True,
) -> tuple[bool, str]:
    """
    Return (ok, reason). reason is empty when ok.
    Filters out dark blobs / furniture that InsightFace sometimes labels as faces.
    """
    det = float(getattr(face, "det_score", 0.0) or 0.0)
    if det < min_det_score:
        return False, f"det_score={det:.2f}<{min_det_score}"

    x1, y1, x2, y2 = [int(v) for v in face.bbox.tolist()]
    w, h = x2 - x1, y2 - y1
    if w < min_side or h < min_side:
        return False, f"tiny={w}x{h}"

    aspect = h / max(w, 1)
    if aspect < 0.75 or aspect > 2.2:
        return False, f"aspect={aspect:.2f}"

    fh, fw = frame_bgr.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(fw, x2), min(fh, y2)
    if x2c - x1c < 8 or y2c - y1c < 8:
        return False, "crop_empty"
    crop = frame_bgr[y1c:y2c, x1c:x2c]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < min_brightness:
        return False, f"dark={brightness:.1f}"

    # Very flat / noisy dark textures often look like faces
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharp < min_sharpness:
        return False, f"blur={sharp:.1f}"

    # Landmark sanity: eyes above mouth roughly
    kps = getattr(face, "kps", None)
    if kps is not None:
        try:
            pts = np.asarray(kps, dtype=np.float32)
            if pts.shape[0] >= 3:
                # InsightFace 5 pts: left_eye, right_eye, nose, left_mouth, right_mouth
                le, re, nose = pts[0], pts[1], pts[2]
                eye_y = 0.5 * (le[1] + re[1])
                if nose[1] + 2 < eye_y:
                    return False, "landmarks_inverted"
                eye_dist = float(np.hypot(le[0] - re[0], le[1] - re[1]))
                if eye_dist < min_side * 0.12:
                    return False, "eyes_too_close"
        except Exception:
            pass

    box = (x1, y1, x2, y2)
    if require_person:
        people = person_boxes or []
        if not people:
            return False, "no_person"
        if not _face_inside_person(box, people):
            return False, "not_on_person"

    return True, ""
