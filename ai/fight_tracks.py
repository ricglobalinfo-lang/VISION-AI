"""
Per-camera pose tracking + skeleton sequence buffers (ByteTrack via Ultralytics).

Stores last N frames of COCO-17 keypoints per track_id for action recognition.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fight_detect import PersonBox


@dataclass
class TrackedPose:
    track_id: int
    box: PersonBox
    kpt: np.ndarray  # (17, 3) x,y,conf in display coords
    ts: float


@dataclass
class CamTrackState:
    seq_len: int = 48
    skeletons: dict[int, deque] = field(default_factory=dict)
    boxes: dict[int, deque] = field(default_factory=dict)
    last_seen: dict[int, float] = field(default_factory=dict)
    ttl_sec: float = 1.5

    def push(self, track_id: int, kpt: np.ndarray, box: PersonBox, ts: float | None = None) -> None:
        ts = time.time() if ts is None else ts
        if track_id not in self.skeletons:
            self.skeletons[track_id] = deque(maxlen=self.seq_len)
            self.boxes[track_id] = deque(maxlen=self.seq_len)
        # Fill gaps with last frame if short dropouts
        self.skeletons[track_id].append(kpt.astype(np.float32))
        self.boxes[track_id].append(box)
        self.last_seen[track_id] = ts

    def prune(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        dead = [tid for tid, t in self.last_seen.items() if now - t > self.ttl_sec]
        for tid in dead:
            self.skeletons.pop(tid, None)
            self.boxes.pop(tid, None)
            self.last_seen.pop(tid, None)

    def ready_ids(self, min_len: int | None = None) -> list[int]:
        need = min_len if min_len is not None else max(16, self.seq_len // 2)
        return [tid for tid, buf in self.skeletons.items() if len(buf) >= need]

    def sequence(self, track_id: int, length: int | None = None) -> np.ndarray | None:
        """Return (T, 17, 3) float32 or None."""
        buf = self.skeletons.get(track_id)
        if not buf:
            return None
        arr = np.stack(list(buf), axis=0)
        if length is not None and len(arr) < length:
            # pad by repeating first frame
            pad_n = length - len(arr)
            pad = np.repeat(arr[:1], pad_n, axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        elif length is not None and len(arr) > length:
            arr = arr[-length:]
        return arr.astype(np.float32)

    def latest_box(self, track_id: int) -> PersonBox | None:
        buf = self.boxes.get(track_id)
        if not buf:
            return None
        return buf[-1]

    def close_pairs(
        self,
        max_norm_dist: float = 0.95,
        min_len: int = 24,
    ) -> list[tuple[int, int, float]]:
        """Return list of (id_a, id_b, norm_dist) for nearby tracks with enough history."""
        ids = self.ready_ids(min_len)
        pairs: list[tuple[int, int, float]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                ba = self.latest_box(ids[i])
                bb = self.latest_box(ids[j])
                if ba is None or bb is None:
                    continue
                avg_h = 0.5 * (ba.h + bb.h)
                dist = float(np.hypot(ba.cx - bb.cx, ba.cy - bb.cy) / max(avg_h, 1.0))
                if dist > max_norm_dist * 1.25:
                    continue
                # Bitta odam ikki track: boxlar deyarli ustma-ust
                ix1 = max(ba.x1, bb.x1)
                iy1 = max(ba.y1, bb.y1)
                ix2 = min(ba.x2, bb.x2)
                iy2 = min(ba.y2, bb.y2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ua = ba.w * ba.h + bb.w * bb.h - inter
                iou = float(inter / ua) if ua > 0 else 0.0
                if iou >= 0.38:
                    continue
                pairs.append((ids[i], ids[j], dist))
        pairs.sort(key=lambda x: x[2])
        return pairs


class PoseTrackBank:
    """Multi-camera track/skeleton store."""

    def __init__(self, seq_len: int = 48) -> None:
        self.seq_len = int(seq_len)
        self._cams: dict[str, CamTrackState] = defaultdict(
            lambda: CamTrackState(seq_len=self.seq_len)
        )

    def cam(self, cam_id: str) -> CamTrackState:
        st = self._cams[cam_id]
        st.seq_len = self.seq_len
        return st

    def update_from_ultralytics(
        self,
        cam_id: str,
        result: Any,
        inv: float = 1.0,
        min_conf: float = 0.45,
    ) -> list[TrackedPose]:
        """
        Parse Ultralytics pose track result into tracked poses + buffer.
        result: single ultralytics Results object (already tracked).
        """
        out: list[TrackedPose] = []
        st = self.cam(cam_id)
        now = time.time()
        if result is None or result.boxes is None or result.keypoints is None:
            st.prune(now)
            return out

        boxes = result.boxes
        kobj = result.keypoints
        xyxy = boxes.xyxy
        confs = boxes.conf
        ids = boxes.id
        kdata = getattr(kobj, "data", None)
        if xyxy is None or kdata is None:
            st.prune(now)
            return out

        xyxy_np = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        conf_np = confs.cpu().numpy() if confs is not None and hasattr(confs, "cpu") else (
            np.ones((len(xyxy_np),), dtype=np.float32)
        )
        id_np = None
        if ids is not None:
            id_np = ids.cpu().numpy().astype(int) if hasattr(ids, "cpu") else np.asarray(ids, dtype=int)
        k_np = kdata.cpu().numpy() if hasattr(kdata, "cpu") else np.asarray(kdata)

        for i in range(len(xyxy_np)):
            conf = float(conf_np[i]) if i < len(conf_np) else 1.0
            if conf < min_conf:
                continue
            tid = int(id_np[i]) if id_np is not None and i < len(id_np) else -(i + 1)
            x1, y1, x2, y2 = xyxy_np[i].tolist()
            x1 = int(x1 * inv)
            y1 = int(y1 * inv)
            x2 = int(x2 * inv)
            y2 = int(y2 * inv)
            kpt = k_np[i].copy()
            kpt[:, 0] *= inv
            kpt[:, 1] *= inv
            box = PersonBox(x1, y1, x2, y2, conf=conf, track_id=tid)
            st.push(tid, kpt, box, now)
            out.append(TrackedPose(track_id=tid, box=box, kpt=kpt, ts=now))

        st.prune(now)
        return out
