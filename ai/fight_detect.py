"""
Fight / violence detector — strict multi-signal + temporal confirmation.

Hard rules (precision-first):
  - Proximity alone NEVER triggers (side-by-side standing must not alert)
  - Require strong pose aggression OR extreme motion
  - Side-by-side horizontal overlap is penalized, not rewarded
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class PersonBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0
    track_id: int = -1

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def w(self) -> float:
        return max(1.0, float(self.x2 - self.x1))

    @property
    def h(self) -> float:
        return max(1.0, float(self.y2 - self.y1))

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class FightState:
    active: bool = False
    score: float = 0.0
    confirm_ratio: float = 0.0
    pair: tuple[PersonBox, PersonBox] | None = None
    detail: str = ""
    just_confirmed: bool = False
    fight_prob: float = 0.0


@dataclass
class _CamHist:
    prev_gray: np.ndarray | None = None
    prev_centers: list[tuple[float, float]] = field(default_factory=list)
    hit_times: list[float] = field(default_factory=list)
    last_confirm_ts: float = 0.0
    was_active: bool = False


def _iou(a: PersonBox, b: PersonBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = a.w * a.h + b.w * b.h - inter
    return float(inter / ua) if ua > 0 else 0.0


def nms_people(people: list[PersonBox], iou_thresh: float = 0.40) -> list[PersonBox]:
    """Bitta odam ikki box bo‘lsa — kattaroq/aniqrog‘ini qoldirish."""
    if len(people) < 2:
        return people
    ordered = sorted(people, key=lambda p: (p.conf, p.w * p.h), reverse=True)
    kept: list[PersonBox] = []
    for p in ordered:
        if any(_iou(p, k) >= iou_thresh for k in kept):
            continue
        kept.append(p)
    return kept


def _union(a: PersonBox, b: PersonBox) -> tuple[int, int, int, int]:
    return (
        min(a.x1, b.x1),
        min(a.y1, b.y1),
        max(a.x2, b.x2),
        max(a.y2, b.y2),
    )


def _norm_distance(a: PersonBox, b: PersonBox) -> float:
    avg_h = 0.5 * (a.h + b.h)
    dx = a.cx - b.cx
    dy = a.cy - b.cy
    return float(np.hypot(dx, dy) / max(avg_h, 1.0))


def _flow_energy(prev_gray: np.ndarray | None, gray: np.ndarray, region: tuple[int, int, int, int]) -> float:
    if prev_gray is None or prev_gray.shape != gray.shape:
        return 0.0
    x1, y1, x2, y2 = region
    h, w = gray.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return 0.0
    a = prev_gray[y1:y2, x1:x2]
    b = gray[y1:y2, x1:x2]
    if a.size == 0 or b.size == 0:
        return 0.0
    scale = 96.0 / max(a.shape[0], a.shape[1])
    if scale < 1.0:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 15, 2, 5, 1.1, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.percentile(mag, 90))


def _center_jitter(prev: list[tuple[float, float]], cur: list[tuple[float, float]], scale_h: float) -> float:
    if not prev or not cur or scale_h <= 1:
        return 0.0
    used = set()
    total = 0.0
    n = 0
    for cx, cy in cur:
        best_i, best_d = -1, 1e18
        for i, (px, py) in enumerate(prev):
            if i in used:
                continue
            d = (cx - px) ** 2 + (cy - py) ** 2
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            used.add(best_i)
            total += np.sqrt(best_d) / scale_h
            n += 1
    return float(total / n) if n else 0.0


def _pose_aggression(pose_kpts: list[np.ndarray] | None, people: list[PersonBox]) -> float:
    """Cross-person wrist → head/torso intrusion (0..1)."""
    if not pose_kpts or len(pose_kpts) < 2 or len(people) < 2:
        return 0.0
    matched: list[tuple[PersonBox, np.ndarray]] = []
    used_p = set()
    for kpt in pose_kpts:
        if kpt is None or kpt.shape[0] < 11:
            continue
        pts = []
        for idx in (0, 5, 6):
            if kpt[idx, 2] >= 0.35:
                pts.append(kpt[idx, :2])
        if not pts:
            continue
        cx, cy = np.mean(pts, axis=0)
        best_i, best_d = -1, 1e18
        for i, p in enumerate(people):
            if i in used_p:
                continue
            d = (p.cx - cx) ** 2 + (p.cy - cy) ** 2
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d < (people[best_i].h * 0.8) ** 2:
            used_p.add(best_i)
            matched.append((people[best_i], kpt))
    if len(matched) < 2:
        return 0.0

    hits = 0
    checks = 0
    for i in range(len(matched)):
        for j in range(len(matched)):
            if i == j:
                continue
            _pa, ka = matched[i]
            pb, _kb = matched[j]
            for wi in (9, 10):
                if ka[wi, 2] < 0.40:
                    continue
                wx, wy = float(ka[wi, 0]), float(ka[wi, 1])
                checks += 1
                tx1 = pb.x1 - 0.04 * pb.w
                ty1 = pb.y1 - 0.04 * pb.h
                tx2 = pb.x2 + 0.04 * pb.w
                ty2 = pb.y1 + 0.58 * pb.h
                if tx1 <= wx <= tx2 and ty1 <= wy <= ty2:
                    hits += 1
    if checks == 0:
        return 0.0
    return min(1.0, hits / max(3.0, float(checks)))


class FightDetector:
    """Legacy/heuristic backend (precision-hardened)."""

    def __init__(self, cfg: Any):
        self.min_people = int(getattr(cfg, "FIGHT_MIN_PEOPLE", 2))
        self.max_norm_dist = float(getattr(cfg, "FIGHT_MAX_NORM_DIST", 0.95))
        self.min_iou = float(getattr(cfg, "FIGHT_MIN_IOU", 0.05))
        self.flow_thresh = float(getattr(cfg, "FIGHT_FLOW_THRESH", 3.5))
        self.jitter_thresh = float(getattr(cfg, "FIGHT_JITTER_THRESH", 0.10))
        self.score_thresh = float(getattr(cfg, "FIGHT_SCORE_THRESH", 0.82))
        self.confirm_sec = float(getattr(cfg, "FIGHT_CONFIRM_SEC", 1.8))
        self.hold_sec = float(getattr(cfg, "FIGHT_HOLD_SEC", 2.5))
        self.cooldown_sec = float(getattr(cfg, "FIGHT_COOLDOWN_SEC", 15.0))
        self.min_person_conf = float(getattr(cfg, "FIGHT_MIN_PERSON_CONF", 0.50))
        self.require_pose = bool(getattr(cfg, "FIGHT_REQUIRE_POSE", True))
        self.min_pose = float(getattr(cfg, "FIGHT_MIN_POSE_SCORE", 0.55))
        self.min_flow = float(getattr(cfg, "FIGHT_MIN_FLOW_SCORE", 0.70))
        self.side_penalty = bool(getattr(cfg, "FIGHT_SIDE_BY_SIDE_PENALTY", True))
        self.max_pair_iou = float(getattr(cfg, "FIGHT_MAX_PAIR_IOU", 0.38))
        self.min_intrusion = float(getattr(cfg, "FIGHT_MIN_INTRUSION", 0.22))
        self.action_thresh = float(getattr(cfg, "FIGHT_ACTION_THRESH", 0.78))
        self._cams: dict[str, _CamHist] = {}

    def _hist(self, cam_id: str) -> _CamHist:
        if cam_id not in self._cams:
            self._cams[cam_id] = _CamHist()
        return self._cams[cam_id]

    def update(
        self,
        cam_id: str,
        frame_bgr: np.ndarray,
        people: list[PersonBox],
        pose_kpts: list[np.ndarray] | None = None,
        now: float | None = None,
        fight_prob: float | None = None,
    ) -> FightState:
        now = time.time() if now is None else now
        hist = self._hist(cam_id)
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        people = [p for p in people if p.conf >= self.min_person_conf and p.w >= 20 and p.h >= 40]
        people = nms_people(people, iou_thresh=self.max_pair_iou)
        people.sort(key=lambda p: p.w * p.h, reverse=True)
        people = people[:6]

        state = FightState()
        if fight_prob is not None:
            state.fight_prob = float(fight_prob)

        if len(people) < self.min_people:
            hist.prev_gray = gray
            hist.prev_centers = [(p.cx, p.cy) for p in people]
            hist.hit_times = [t for t in hist.hit_times if now - t < self.confirm_sec * 2]
            if hist.was_active and now - hist.last_confirm_ts < self.hold_sec:
                state.active = True
                state.score = 0.55
                state.detail = "urush (davom etmoqda)"
            else:
                hist.was_active = False
            return state

        best_pair = None
        best_prox = 0.0
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                a, b = people[i], people[j]
                dist = _norm_distance(a, b)
                iou = _iou(a, b)
                if iou >= self.max_pair_iou:
                    continue  # bitta odam
                prox = 0.0
                if dist <= self.max_norm_dist:
                    prox += 0.55 * (1.0 - dist / self.max_norm_dist)
                if iou >= self.min_iou:
                    prox += min(0.35, iou * 1.0)
                # Yonma-yon turish: bonus YO'Q — jarima
                overlap_x = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) / max(min(a.w, b.w), 1)
                overlap_y = max(0, min(a.y2, b.y2) - max(a.y1, b.y1)) / max(min(a.h, b.h), 1)
                if self.side_penalty and overlap_x > 0.2 and overlap_y < 0.35 and iou < 0.12:
                    prox *= 0.45
                if prox > best_prox:
                    best_prox = prox
                    best_pair = (a, b)

        if best_pair is None or best_prox < 0.40:
            hist.prev_gray = gray
            hist.prev_centers = [(p.cx, p.cy) for p in people]
            hist.hit_times = [t for t in hist.hit_times if now - t < self.confirm_sec * 2]
            if hist.was_active and now - hist.last_confirm_ts < self.hold_sec:
                state.active = True
                state.score = 0.55
                state.detail = "urush (davom etmoqda)"
                state.pair = best_pair
            else:
                hist.was_active = False
            return state

        a, b = best_pair
        ux1, uy1, ux2, uy2 = _union(a, b)
        pad = int(0.08 * max(ux2 - ux1, uy2 - uy1))
        region = (
            max(0, ux1 - pad),
            max(0, uy1 - pad),
            min(w, ux2 + pad),
            min(h, uy2 + pad),
        )
        flow = _flow_energy(hist.prev_gray, gray, region)
        avg_h = 0.5 * (a.h + b.h)
        jitter = _center_jitter(hist.prev_centers, [(p.cx, p.cy) for p in people], avg_h)
        pose_s = _pose_aggression(pose_kpts, people)

        flow_s = min(1.0, flow / max(self.flow_thresh, 0.1))
        jitter_s = min(1.0, jitter / max(self.jitter_thresh, 0.01))

        # Motion-heavy score; proximity is weak
        score = (
            0.12 * min(1.0, best_prox)
            + 0.40 * flow_s
            + 0.18 * jitter_s
            + 0.30 * pose_s
        )

        # HARD GATE: yaqlik yetarli emas
        aggression_ok = pose_s >= self.min_pose or flow_s >= self.min_flow
        geo_ok = fight_prob is not None and float(fight_prob) >= self.action_thresh
        if self.require_pose and pose_s < self.min_pose and flow_s < self.min_flow and not geo_ok:
            score = 0.0
            aggression_ok = False
        elif not aggression_ok and not geo_ok:
            score = 0.0
        elif flow_s < 0.40 and pose_s < 0.50 and not geo_ok:
            score = 0.0

        if fight_prob is not None:
            fp = float(fight_prob)
            if fp >= self.action_thresh and aggression_ok:
                score = max(score * 0.40, fp)
            elif fp >= self.score_thresh:
                score = max(score, fp * 0.70)
            else:
                # past geo — heuristic’ni oshirmaslik
                score = min(score, 0.45)

        state.score = float(score)
        state.pair = best_pair
        state.detail = (
            f"prox={best_prox:.2f} flow={flow:.1f} jitter={jitter:.2f} "
            f"pose={pose_s:.2f} fight_prob={state.fight_prob:.2f}"
        )

        hit = score >= self.score_thresh and aggression_ok
        if hit:
            hist.hit_times.append(now)
        hist.hit_times = [t for t in hist.hit_times if now - t <= max(self.confirm_sec, 1.2)]
        if hist.hit_times:
            span = max(hist.hit_times) - min(hist.hit_times) if len(hist.hit_times) > 1 else 0.0
            density = len(hist.hit_times)
            need_n = max(5, int(self.confirm_sec * 3.0))
            state.confirm_ratio = min(1.0, density / need_n)
            confirmed = density >= need_n and span >= self.confirm_sec * 0.60
        else:
            confirmed = False
            state.confirm_ratio = 0.0

        if confirmed:
            state.active = True
            if not hist.was_active and (now - hist.last_confirm_ts) >= self.cooldown_sec:
                state.just_confirmed = True
                hist.last_confirm_ts = now
            elif hist.was_active:
                hist.last_confirm_ts = now
            hist.was_active = True
        elif hist.was_active and now - hist.last_confirm_ts < self.hold_sec:
            state.active = True
        else:
            hist.was_active = False

        hist.prev_gray = gray
        hist.prev_centers = [(p.cx, p.cy) for p in people]
        return state
