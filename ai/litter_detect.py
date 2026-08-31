"""
Litter / axlat detector — ikki rejim:

  A) Tashlash: qo‘lda (elevated) → yerga → tinch + (ixtiyoriy) pose
  B) Statik: pol zonasida uzoq yotgan bottle/cup/... (odam shart emas)

COCO’da alohida "trash" klassi yo‘q — litter-like klasslar + temporal/pol filtrlari.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fight_detect import PersonBox


@dataclass
class ObjectBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0
    name: str = ""

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

    @property
    def side(self) -> float:
        return min(self.w, self.h)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class LitterState:
    active: bool = False
    score: float = 0.0
    confirm_ratio: float = 0.0
    obj: ObjectBox | None = None
    person: PersonBox | None = None
    detail: str = ""
    just_confirmed: bool = False
    mode: str = ""  # "throw" | "static" | ""


@dataclass
class _Track:
    tid: int
    name: str
    cx: float
    cy: float
    box: ObjectBox
    seen_at: float
    elevated: bool = False
    ground_since: float | None = None
    last_ground_xy: tuple[float, float] | None = None
    person_key: tuple[int, int] | None = None
    miss: int = 0
    throw_score: float = 0.0


@dataclass
class _StaticTrack:
    tid: int
    name: str
    cx: float
    cy: float
    box: ObjectBox
    seen_at: float
    floor_since: float | None = None
    last_xy: tuple[float, float] | None = None
    miss: int = 0


@dataclass
class _CamHist:
    tracks: list[_Track] = field(default_factory=list)
    static_tracks: list[_StaticTrack] = field(default_factory=list)
    next_id: int = 1
    next_static_id: int = 1
    hit_times: list[float] = field(default_factory=list)
    static_hit_times: list[float] = field(default_factory=list)
    last_confirm_ts: float = 0.0
    last_static_confirm_ts: float = 0.0
    was_active: bool = False
    was_static_active: bool = False
    last_obj: ObjectBox | None = None
    last_person: PersonBox | None = None
    last_score: float = 0.0
    last_detail: str = ""
    last_mode: str = ""
    wrist_hist: dict[tuple[int, int], list[tuple[float, float]]] = field(default_factory=dict)


def _person_key(p: PersonBox) -> tuple[int, int]:
    return (int(p.cx // 40), int(p.cy // 40))


def _norm_dist_obj_person(obj: ObjectBox, person: PersonBox) -> float:
    return float(np.hypot(obj.cx - person.cx, obj.cy - person.cy) / max(person.h, 1.0))


def _height_ratio(obj: ObjectBox, person: PersonBox) -> float:
    """0 at person top, 1 at person bottom."""
    return float((obj.cy - person.y1) / max(person.h, 1.0))


def _match_pose_to_person(
    pose_kpts: list[np.ndarray] | None,
    person: PersonBox,
) -> np.ndarray | None:
    if not pose_kpts:
        return None
    best = None
    best_d = 1e18
    for kpt in pose_kpts:
        if kpt is None or getattr(kpt, "shape", (0,))[0] < 11:
            continue
        pts = []
        for idx in (0, 5, 6):
            if float(kpt[idx, 2]) >= 0.30:
                pts.append(kpt[idx, :2])
        if not pts:
            continue
        mean = np.mean(pts, axis=0)
        cx, cy = float(mean[0]), float(mean[1])
        d = (person.cx - cx) ** 2 + (person.cy - cy) ** 2
        if d < best_d:
            best_d = d
            best = kpt
    if best is None or best_d > (person.h * 0.9) ** 2:
        return None
    return best


class LitterDetector:
    def __init__(self, cfg: Any):
        self.classes = set(getattr(cfg, "LITTER_CLASSES", ("bottle", "cup", "bowl", "wine glass")))
        self.min_conf = float(getattr(cfg, "LITTER_MIN_CONF", 0.40))
        self.min_person_conf = float(getattr(cfg, "LITTER_MIN_PERSON_CONF", 0.35))
        self.assoc_dist = float(getattr(cfg, "LITTER_ASSOC_DIST", 1.35))
        self.hand_max = float(getattr(cfg, "LITTER_HAND_MAX_RATIO", 0.72))
        self.ground_min = float(getattr(cfg, "LITTER_GROUND_MIN_RATIO", 0.88))
        self.still_px = float(getattr(cfg, "LITTER_GROUND_STILL_PX", 18.0))
        self.ground_hold = float(getattr(cfg, "LITTER_GROUND_HOLD_SEC", 1.0))
        self.confirm_sec = float(getattr(cfg, "LITTER_CONFIRM_SEC", 1.4))
        self.hold_sec = float(getattr(cfg, "LITTER_HOLD_SEC", 3.0))
        self.cooldown_sec = float(getattr(cfg, "LITTER_COOLDOWN_SEC", 20.0))
        self.score_thresh = float(getattr(cfg, "LITTER_SCORE_THRESH", 0.62))
        self.require_floor = bool(getattr(cfg, "LITTER_REQUIRE_FLOOR", True))
        self.floor_ratio = float(getattr(cfg, "LITTER_FLOOR_RATIO", 0.68))
        self.use_pose = bool(getattr(cfg, "LITTER_USE_POSE", True))
        self.throw_drop = float(getattr(cfg, "LITTER_THROW_DROP_RATIO", 0.10))
        self.throw_window = float(getattr(cfg, "LITTER_THROW_WINDOW_SEC", 0.8))
        self.throw_min = float(getattr(cfg, "LITTER_THROW_MIN", 0.30))
        # Static / Mode B
        self.static_enabled = bool(getattr(cfg, "LITTER_STATIC_ENABLED", True))
        self.static_require_person = bool(getattr(cfg, "LITTER_STATIC_REQUIRE_PERSON", False))
        self.static_floor_ratio = float(getattr(cfg, "LITTER_STATIC_FLOOR_RATIO", 0.70))
        self.static_min_conf = float(getattr(cfg, "LITTER_STATIC_MIN_CONF", 0.40))
        self.static_hold = float(getattr(cfg, "LITTER_STATIC_HOLD_SEC", 3.5))
        self.static_confirm = float(getattr(cfg, "LITTER_STATIC_CONFIRM_SEC", 2.0))
        self.static_cooldown = float(getattr(cfg, "LITTER_STATIC_COOLDOWN_SEC", 45.0))
        self.static_score_thresh = float(getattr(cfg, "LITTER_STATIC_SCORE_THRESH", 0.55))
        self.static_still_px = float(getattr(cfg, "LITTER_STATIC_STILL_PX", 22.0))
        self.static_min_side = float(getattr(cfg, "LITTER_STATIC_MIN_SIDE", 10))
        self.static_max_side_frac = float(getattr(cfg, "LITTER_STATIC_MAX_SIDE_FRAC", 0.22))
        self._cams: dict[str, _CamHist] = {}

    def _hist(self, cam_id: str) -> _CamHist:
        if cam_id not in self._cams:
            self._cams[cam_id] = _CamHist()
        return self._cams[cam_id]

    def _update_wrist_hist(
        self,
        hist: _CamHist,
        people: list[PersonBox],
        pose_kpts: list[np.ndarray] | None,
        now: float,
    ) -> None:
        if not self.use_pose or not pose_kpts:
            return
        for p in people:
            kpt = _match_pose_to_person(pose_kpts, p)
            if kpt is None:
                continue
            wys = []
            for wi in (9, 10):
                if float(kpt[wi, 2]) < 0.35:
                    continue
                wy = float(kpt[wi, 1])
                wys.append((wy - p.y1) / max(p.h, 1.0))
            if not wys:
                continue
            key = _person_key(p)
            series = hist.wrist_hist.get(key, [])
            series.append((now, float(max(wys))))
            series = [(t, y) for t, y in series if now - t <= self.throw_window * 2.5]
            hist.wrist_hist[key] = series[-20:]
        stale = [k for k, s in hist.wrist_hist.items() if not s or now - s[-1][0] > 3.0]
        for k in stale:
            hist.wrist_hist.pop(k, None)

    def _throw_score_for_person(self, hist: _CamHist, person: PersonBox, now: float) -> float:
        if not self.use_pose:
            return 0.5
        key = _person_key(person)
        series = hist.wrist_hist.get(key) or []
        if len(series) < 2:
            for dk in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)):
                alt = (key[0] + dk[0], key[1] + dk[1])
                series = hist.wrist_hist.get(alt) or series
                if len(series) >= 2:
                    break
        if len(series) < 2:
            return 0.0
        recent = [(t, y) for t, y in series if now - t <= self.throw_window]
        if len(recent) < 2:
            recent = series[-4:]
        drop = recent[-1][1] - recent[0][1]
        if drop < self.throw_drop * 0.5:
            return 0.0
        return float(min(1.0, drop / max(self.throw_drop, 0.05)))

    def update(
        self,
        cam_id: str,
        people: list[PersonBox],
        objects: list[ObjectBox],
        frame_h: int,
        frame_w: int | None = None,
        now: float | None = None,
        pose_kpts: list[np.ndarray] | None = None,
    ) -> LitterState:
        now = time.time() if now is None else now
        hist = self._hist(cam_id)
        state = LitterState()

        people = [p for p in people if p.conf >= self.min_person_conf and p.h >= 40]
        objects = [
            o for o in objects
            if o.name in self.classes and o.conf >= self.min_conf and o.w >= 8 and o.h >= 8
        ]
        if frame_w is None or frame_w <= 0:
            frame_w = max(1, int(frame_h * 16 / 9))

        self._update_wrist_hist(hist, people, pose_kpts, now)

        throw_state = self._update_throw(hist, people, objects, frame_h, now, pose_kpts)
        static_state = (
            self._update_static(hist, people, objects, frame_h, frame_w, now)
            if self.static_enabled
            else LitterState()
        )

        # Prefer throw if both active/high; else best score
        pick = throw_state
        if static_state.score > throw_state.score or (
            static_state.active and not throw_state.active
        ):
            if static_state.score >= throw_state.score or static_state.active:
                pick = static_state
        if throw_state.active and throw_state.score >= static_state.score * 0.95:
            pick = throw_state

        # Hold display from previous confirm
        if not pick.active and not pick.just_confirmed:
            if hist.was_active and now - hist.last_confirm_ts < self.hold_sec:
                pick = LitterState(
                    active=True,
                    score=max(0.5, hist.last_score),
                    obj=hist.last_obj,
                    person=hist.last_person,
                    detail=hist.last_detail or "axlat (davom etmoqda)",
                    mode=hist.last_mode or "throw",
                )
            elif hist.was_static_active and now - hist.last_static_confirm_ts < self.hold_sec:
                pick = LitterState(
                    active=True,
                    score=max(0.5, hist.last_score),
                    obj=hist.last_obj,
                    person=hist.last_person,
                    detail=hist.last_detail or "axlat yotgan (davom)",
                    mode="static",
                )

        return pick

    def _finalize_hits(
        self,
        hist: _CamHist,
        score: float,
        thresh: float,
        confirm_sec: float,
        cooldown: float,
        obj: ObjectBox | None,
        person: PersonBox | None,
        detail: str,
        mode: str,
        now: float,
        hit_attr: str,
        was_attr: str,
        ts_attr: str,
    ) -> LitterState:
        state = LitterState(
            score=float(score),
            obj=obj,
            person=person,
            detail=detail,
            mode=mode,
        )
        hits: list[float] = getattr(hist, hit_attr)
        hit = score >= thresh and obj is not None
        if hit:
            hits.append(now)
        hits = [t for t in hits if now - t <= confirm_sec]
        setattr(hist, hit_attr, hits)

        if hits:
            density = len(hits)
            need_n = max(2, int(confirm_sec * 2.0))
            span = max(hits) - min(hits) if density > 1 else 0.0
            state.confirm_ratio = min(1.0, density / need_n)
            confirmed = density >= need_n and (span >= confirm_sec * 0.45 or density >= need_n + 1)
        else:
            confirmed = False
            state.confirm_ratio = 0.0

        last_ts = float(getattr(hist, ts_attr))
        was = bool(getattr(hist, was_attr))
        if confirmed:
            state.active = True
            if not was and (now - last_ts) >= cooldown:
                state.just_confirmed = True
                setattr(hist, ts_attr, now)
            elif was:
                setattr(hist, ts_attr, now)
            setattr(hist, was_attr, True)
            hist.last_score = state.score
            hist.last_obj = obj
            hist.last_person = person
            hist.last_detail = detail
            hist.last_mode = mode
        else:
            setattr(hist, was_attr, False)
        return state

    def _update_throw(
        self,
        hist: _CamHist,
        people: list[PersonBox],
        objects: list[ObjectBox],
        frame_h: int,
        now: float,
        pose_kpts: list[np.ndarray] | None,
    ) -> LitterState:
        if not people or not objects:
            self._age_tracks(hist, now, matched=set())
            hist.hit_times = [t for t in hist.hit_times if now - t < self.confirm_sec * 2]
            return LitterState(mode="throw")

        matched_tids = self._match_tracks(hist, objects, now)
        self._age_tracks(hist, now, matched=matched_tids)

        best_score = 0.0
        best_obj: ObjectBox | None = None
        best_person: PersonBox | None = None
        best_detail = ""
        floor_y = float(frame_h) * self.floor_ratio

        for tr in hist.tracks:
            person = self._nearest_person(tr, people)
            if person is None:
                continue
            dist = _norm_dist_obj_person(tr.box, person)
            if dist > self.assoc_dist * 1.25:
                continue

            ratio = _height_ratio(tr.box, person)
            on_floor = tr.cy >= floor_y
            frame_ground = tr.cy >= frame_h * 0.72
            elevated = ratio <= self.hand_max and not frame_ground and not on_floor
            on_ground = (ratio >= self.ground_min or frame_ground or on_floor) and ratio > self.hand_max * 0.85

            throw_s = self._throw_score_for_person(hist, person, now)
            tr.throw_score = max(tr.throw_score * 0.85, throw_s)

            if elevated:
                tr.elevated = True
                tr.ground_since = None
                tr.last_ground_xy = None
                tr.person_key = _person_key(person)
                continue

            if not (tr.elevated and on_ground):
                continue

            if self.require_floor and not on_floor and not frame_ground:
                continue

            if tr.person_key is not None and tr.person_key != _person_key(person):
                if dist > self.assoc_dist * 0.9:
                    continue

            still = True
            if tr.last_ground_xy is not None:
                dx = tr.cx - tr.last_ground_xy[0]
                dy = tr.cy - tr.last_ground_xy[1]
                still = (dx * dx + dy * dy) ** 0.5 <= self.still_px
            tr.last_ground_xy = (tr.cx, tr.cy)

            if not still:
                tr.ground_since = now
                continue

            if tr.ground_since is None:
                tr.ground_since = now
            ground_age = now - tr.ground_since

            drop_s = 1.0 if tr.elevated else 0.0
            ground_s = min(1.0, ground_age / max(self.ground_hold, 0.1))
            prox_s = max(0.0, 1.0 - dist / max(self.assoc_dist, 0.1))
            conf_s = min(1.0, tr.box.conf / 0.7)
            floor_s = 1.0 if on_floor else (0.7 if frame_ground else 0.2)
            pose_s = tr.throw_score

            score = (
                0.28 * drop_s
                + 0.22 * ground_s
                + 0.15 * prox_s
                + 0.10 * conf_s
                + 0.12 * floor_s
                + 0.13 * pose_s
            )
            if ground_age < self.ground_hold * 0.6:
                score *= 0.5
            if self.use_pose and pose_kpts is not None and pose_s < self.throw_min:
                score *= 0.72
            elif pose_s >= 0.55:
                score = min(1.0, score + 0.08)

            if score > best_score:
                best_score = score
                best_obj = tr.box
                best_person = person
                best_detail = (
                    f"throw {tr.name} ground={ground_age:.1f}s floor={on_floor} "
                    f"throw={pose_s:.2f} dist={dist:.2f}"
                )

        return self._finalize_hits(
            hist,
            best_score,
            self.score_thresh,
            self.confirm_sec,
            self.cooldown_sec,
            best_obj,
            best_person,
            best_detail,
            "throw",
            now,
            "hit_times",
            "was_active",
            "last_confirm_ts",
        )

    def _update_static(
        self,
        hist: _CamHist,
        people: list[PersonBox],
        objects: list[ObjectBox],
        frame_h: int,
        frame_w: int,
        now: float,
    ) -> LitterState:
        floor_y = float(frame_h) * self.static_floor_ratio
        max_side = float(frame_w) * self.static_max_side_frac

        floor_objs: list[ObjectBox] = []
        for o in objects:
            if o.conf < self.static_min_conf:
                continue
            if o.cy < floor_y:
                continue
            if o.side < self.static_min_side:
                continue
            if max(o.w, o.h) > max_side:
                continue
            floor_objs.append(o)

        if self.static_require_person and not people:
            self._age_static(hist, now, matched=set())
            return LitterState(mode="static")

        matched = self._match_static(hist, floor_objs, now)
        self._age_static(hist, now, matched=matched)

        best_score = 0.0
        best_obj: ObjectBox | None = None
        best_person: PersonBox | None = None
        best_detail = ""

        for tr in hist.static_tracks:
            # optional person proximity (informational / soft boost)
            person = None
            dist = 9.0
            if people:
                person = self._nearest_person_box(tr.box, people)
                if person is not None:
                    dist = _norm_dist_obj_person(tr.box, person)

            if self.static_require_person and person is None:
                continue

            still = True
            if tr.last_xy is not None:
                dx = tr.cx - tr.last_xy[0]
                dy = tr.cy - tr.last_xy[1]
                still = (dx * dx + dy * dy) ** 0.5 <= self.static_still_px
            tr.last_xy = (tr.cx, tr.cy)

            if not still:
                tr.floor_since = now
                continue

            if tr.floor_since is None:
                tr.floor_since = now
            age = now - tr.floor_since

            dwell = min(1.0, age / max(self.static_hold, 0.1))
            conf_s = min(1.0, tr.box.conf / 0.75)
            floor_s = min(1.0, (tr.cy - floor_y) / max(frame_h * 0.15, 1.0))
            size_s = float(np.clip(tr.box.side / 28.0, 0.35, 1.0))
            score = 0.40 * dwell + 0.25 * conf_s + 0.20 * floor_s + 0.15 * size_s
            if age < self.static_hold * 0.5:
                score *= 0.55
            if person is not None and dist < 1.8:
                score = min(1.0, score + 0.05)

            if score > best_score:
                best_score = score
                best_obj = tr.box
                best_person = person
                best_detail = (
                    f"static {tr.name} floor={age:.1f}s conf={tr.box.conf:.2f} "
                    f"cy={tr.cy:.0f}/{floor_y:.0f}"
                )

        return self._finalize_hits(
            hist,
            best_score,
            self.static_score_thresh,
            self.static_confirm,
            self.static_cooldown,
            best_obj,
            best_person,
            best_detail,
            "static",
            now,
            "static_hit_times",
            "was_static_active",
            "last_static_confirm_ts",
        )

    def _nearest_person(self, tr: _Track, people: list[PersonBox]) -> PersonBox | None:
        return self._nearest_person_box(tr.box, people)

    def _nearest_person_box(self, box: ObjectBox, people: list[PersonBox]) -> PersonBox | None:
        best = None
        best_d = 1e18
        for p in people:
            d = _norm_dist_obj_person(box, p)
            if d < best_d:
                best_d = d
                best = p
        if best is None or best_d > self.assoc_dist * 1.4:
            return None
        return best

    def _match_tracks(self, hist: _CamHist, objects: list[ObjectBox], now: float) -> set[int]:
        matched: set[int] = set()
        used_obj: set[int] = set()
        for tr in hist.tracks:
            best_i, best_cost = -1, 1e18
            for i, obj in enumerate(objects):
                if i in used_obj or obj.name != tr.name:
                    continue
                dx = abs(tr.cx - obj.cx)
                dy = abs(tr.cy - obj.cy)
                max_dx = max(48.0, 0.8 * max(tr.box.w, obj.w))
                max_dy = max(240.0, 4.5 * max(tr.box.h, obj.h))
                if dx > max_dx or dy > max_dy:
                    continue
                cost = dx + 0.35 * dy
                if cost < best_cost:
                    best_cost, best_i = cost, i
            if best_i >= 0:
                obj = objects[best_i]
                used_obj.add(best_i)
                matched.add(tr.tid)
                tr.cx, tr.cy = obj.cx, obj.cy
                tr.box = obj
                tr.seen_at = now
                tr.miss = 0
        for i, obj in enumerate(objects):
            if i in used_obj:
                continue
            inherited = False
            person_key = None
            for tr in hist.tracks:
                if tr.name != obj.name or tr.miss == 0:
                    continue
                if abs(tr.cx - obj.cx) <= 80 and tr.elevated:
                    inherited = True
                    person_key = tr.person_key
                    break
            tid = hist.next_id
            hist.next_id += 1
            hist.tracks.append(
                _Track(
                    tid=tid,
                    name=obj.name,
                    cx=obj.cx,
                    cy=obj.cy,
                    box=obj,
                    seen_at=now,
                    elevated=inherited,
                    person_key=person_key,
                )
            )
            matched.add(tid)
        return matched

    def _match_static(self, hist: _CamHist, objects: list[ObjectBox], now: float) -> set[int]:
        matched: set[int] = set()
        used: set[int] = set()
        for tr in hist.static_tracks:
            best_i, best_cost = -1, 1e18
            for i, obj in enumerate(objects):
                if i in used or obj.name != tr.name:
                    continue
                dx = abs(tr.cx - obj.cx)
                dy = abs(tr.cy - obj.cy)
                max_d = max(60.0, 1.2 * max(tr.box.w, obj.w, tr.box.h, obj.h))
                if dx > max_d or dy > max_d:
                    continue
                cost = dx + dy
                if cost < best_cost:
                    best_cost, best_i = cost, i
            if best_i >= 0:
                obj = objects[best_i]
                used.add(best_i)
                matched.add(tr.tid)
                tr.cx, tr.cy = obj.cx, obj.cy
                tr.box = obj
                tr.seen_at = now
                tr.miss = 0
        for i, obj in enumerate(objects):
            if i in used:
                continue
            tid = hist.next_static_id
            hist.next_static_id += 1
            hist.static_tracks.append(
                _StaticTrack(
                    tid=tid,
                    name=obj.name,
                    cx=obj.cx,
                    cy=obj.cy,
                    box=obj,
                    seen_at=now,
                    floor_since=now,
                    last_xy=(obj.cx, obj.cy),
                )
            )
            matched.add(tid)
        return matched

    def _age_tracks(self, hist: _CamHist, now: float, matched: set[int]) -> None:
        alive: list[_Track] = []
        for tr in hist.tracks:
            if tr.tid not in matched:
                tr.miss += 1
            if tr.miss <= 8 and now - tr.seen_at < 4.0:
                alive.append(tr)
        hist.tracks = alive

    def _age_static(self, hist: _CamHist, now: float, matched: set[int]) -> None:
        alive: list[_StaticTrack] = []
        for tr in hist.static_tracks:
            if tr.tid not in matched:
                tr.miss += 1
            # Keep a bit longer so brief YOLO misses don't reset dwell timer hard
            if tr.miss <= 12 and now - tr.seen_at < 8.0:
                alive.append(tr)
        hist.static_tracks = alive
