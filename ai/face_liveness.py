"""Temporal micro-motion liveness — static print/screen often has no natural movement."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MotionSample:
    cx: float
    cy: float
    nose_x: float
    nose_y: float
    eye_x: float
    eye_y: float
    bw: float
    bh: float
    ts: float


@dataclass
class MotionState:
    ready: bool = False
    passed: bool = False
    motion: float = 0.0
    frames: int = 0
    detail: str = ""


@dataclass
class _TrackMotion:
    samples: deque = field(default_factory=deque)
    passed: bool = False
    last_seen: float = 0.0
    last_bbox: tuple[int, int, int, int] | None = None


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(1, ax2 - ax1) * max(1, ay2 - ay1) + max(1, bx2 - bx1) * max(1, by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


class MotionLiveness:
    def __init__(self, cfg: Any) -> None:
        self.enabled = bool(getattr(cfg, "FACE_LIVENESS_ENABLED", True))
        self.min_frames = int(getattr(cfg, "FACE_LIVENESS_MIN_FRAMES", 4))
        self.min_sec = float(getattr(cfg, "FACE_LIVENESS_MIN_SEC", 0.45))
        self.motion_thresh = float(getattr(cfg, "FACE_LIVENESS_MOTION_THRESH", 0.08))
        self.soft_thresh = float(getattr(cfg, "FACE_LIVENESS_SOFT_THRESH", 0.035))
        self.window_sec = float(getattr(cfg, "FACE_LIVENESS_WINDOW_SEC", 3.5))
        self.track_merge_iou = float(getattr(cfg, "FACE_LIVENESS_TRACK_MERGE_IOU", 0.22))
        self._tracks: dict[str, dict[int, _TrackMotion]] = defaultdict(dict)
        self._cam_live: set[str] = set()

    def _track(self, cam_id: str, track_id: int) -> _TrackMotion:
        bucket = self._tracks[cam_id]
        if track_id not in bucket:
            bucket[track_id] = _TrackMotion(samples=deque(maxlen=64))
        return bucket[track_id]

    def _resolve_track(
        self,
        cam_id: str,
        track_id: int,
        bbox: tuple[int, int, int, int],
    ) -> int:
        bucket = self._tracks[cam_id]
        if track_id in bucket:
            return track_id
        best_tid, best_iou = track_id, self.track_merge_iou
        for tid, st in bucket.items():
            if st.last_bbox is None:
                continue
            v = _bbox_iou(bbox, st.last_bbox)
            if v > best_iou:
                best_iou, best_tid = v, tid
        return best_tid

    @staticmethod
    def _landmarks(face: Any, bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        nx, ny, ex, ey = cx, cy, cx, cy
        kps = getattr(face, "kps", None)
        if kps is None:
            return nx, ny, ex, ey
        try:
            pts = np.asarray(kps, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                nx, ny = float(pts[2, 0]), float(pts[2, 1])
            if pts.ndim == 2 and pts.shape[0] >= 2:
                ex = 0.5 * (float(pts[0, 0]) + float(pts[1, 0]))
                ey = 0.5 * (float(pts[0, 1]) + float(pts[1, 1]))
        except Exception:
            pass
        return nx, ny, ex, ey

    def _score_motion(self, samples: deque) -> float:
        if len(samples) < 2:
            return 0.0
        arr = list(samples)
        bw = max(1.0, float(np.mean([s.bw for s in arr])))
        bh = max(1.0, float(np.mean([s.bh for s in arr])))
        norm = max(10.0, 0.5 * (bw + bh))

        cx = np.array([s.cx for s in arr], dtype=np.float32)
        cy = np.array([s.cy for s in arr], dtype=np.float32)
        nx = np.array([s.nose_x for s in arr], dtype=np.float32)
        ny = np.array([s.nose_y for s in arr], dtype=np.float32)
        ex = np.array([s.eye_x for s in arr], dtype=np.float32)
        ey = np.array([s.eye_y for s in arr], dtype=np.float32)

        center_std = float(np.std(cx) + np.std(cy)) / norm
        nose_std = float(np.std(nx) + np.std(ny)) / norm
        eye_std = float(np.std(ex) + np.std(ey)) / norm
        bw_std = float(np.std([s.bw for s in arr])) / norm

        path = 0.0
        peak = 0.0
        for i in range(1, len(arr)):
            dx = (arr[i].cx - arr[i - 1].cx) / norm
            dy = (arr[i].cy - arr[i - 1].cy) / norm
            step = float(np.hypot(dx, dy))
            path += step
            peak = max(peak, step)
            ndx = (arr[i].nose_x - arr[i - 1].nose_x) / norm
            ndy = (arr[i].nose_y - arr[i - 1].nose_y) / norm
            peak = max(peak, float(np.hypot(ndx, ndy)))

        return (
            center_std * 0.35
            + nose_std * 0.25
            + eye_std * 0.20
            + bw_std * 0.05
            + min(0.25, path * 0.55)
            + min(0.20, peak * 1.6)
        )

    def update(
        self,
        cam_id: str,
        track_id: int,
        face: Any,
        bbox: tuple[int, int, int, int],
        now: float | None = None,
    ) -> MotionState:
        if not self.enabled:
            return MotionState(ready=True, passed=True, motion=1.0, frames=1, detail="disabled")
        now = time.time() if now is None else now
        if cam_id in self._cam_live:
            return MotionState(ready=True, passed=True, motion=1.0, frames=1, detail="live-ok")

        tid = self._resolve_track(cam_id, track_id, bbox)
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        nx, ny, ex, ey = self._landmarks(face, bbox)
        st = self._track(cam_id, tid)
        st.last_seen = now
        st.last_bbox = bbox
        st.samples.append(
            MotionSample(
                cx=0.5 * (x1 + x2),
                cy=0.5 * (y1 + y2),
                nose_x=nx,
                nose_y=ny,
                eye_x=ex,
                eye_y=ey,
                bw=bw,
                bh=bh,
                ts=now,
            )
        )
        while st.samples and now - st.samples[0].ts > self.window_sec:
            st.samples.popleft()

        frames = len(st.samples)
        span = (st.samples[-1].ts - st.samples[0].ts) if frames > 1 else 0.0
        motion = self._score_motion(st.samples)
        ready = frames >= self.min_frames and span >= self.min_sec
        passed = st.passed or (ready and motion >= self.motion_thresh)
        if not passed and ready and motion >= self.soft_thresh and frames >= self.min_frames + 1:
            passed = True
        if passed:
            st.passed = True
            self._cam_live.add(cam_id)
        detail = f"m={motion:.3f} f={frames} t={span:.1f}s"
        return MotionState(ready=ready, passed=passed, motion=motion, frames=frames, detail=detail)

    def mark_cam_live(self, cam_id: str, now: float | None = None) -> None:
        self._cam_live.add(cam_id)

    def clear_cam(self, cam_id: str) -> None:
        """Xona bo‘shaganda jonlilikni qayta tekshirish uchun."""
        self._cam_live.discard(cam_id)
        for tid in list(self._tracks.get(cam_id, {}).keys()):
            self._tracks[cam_id].pop(tid, None)
        self._tracks.pop(cam_id, None)

    def is_cam_live(self, cam_id: str) -> bool:
        return cam_id in self._cam_live

    def prune(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        ttl = max(self.window_sec * 2.0, 6.0)
        for cam_id, bucket in list(self._tracks.items()):
            dead = [tid for tid, st in bucket.items() if now - st.last_seen > ttl]
            for tid in dead:
                bucket.pop(tid, None)
            if not bucket:
                self._tracks.pop(cam_id, None)
