"""
Multi-frame face recognition bank.

Per (camera, track) we keep recent face embeddings and decide identity from the
best-quality samples — so a blurry/partial frame can be corrected by a clearer
one 0.5–2s later.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import face_match


@dataclass
class FaceSample:
    emb: np.ndarray
    quality: float
    bbox: tuple[int, int, int, int]
    ts: float
    det_score: float = 0.0


@dataclass
class RoomOccupant:
    name: str
    score: float
    last_person_bbox: tuple[int, int, int, int]
    last_face_bbox: tuple[int, int, int, int] | None
    last_seen: float
    confirmed_at: float


@dataclass
class TrackFaceState:
    samples: deque = field(default_factory=lambda: deque(maxlen=24))
    confirmed_name: str | None = None
    confirmed_score: float = 0.0
    confirmed_ts: float = 0.0
    candidate_name: str | None = None
    candidate_hits: int = 0
    candidate_score: float = 0.0
    last_bbox: tuple[int, int, int, int] | None = None
    last_seen: float = 0.0


@dataclass
class FaceDecision:
    name: str  # display / log name; "Noma'lum" if unknown
    score: float
    bbox: tuple[int, int, int, int]
    confirmed: bool  # strong identity (log known)
    is_unknown: bool  # treat as unknown for save/log
    pending: bool  # soft candidate, waiting better frames
    quality: float
    detail: str = ""
    emb: np.ndarray | None = None  # best emb (for unknown save)
    just_confirmed: bool = False  # newly confirmed this frame (for logging)
    log_unknown: bool = False  # emit unknown event this frame


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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


def face_sample_quality(face: Any, bbox: tuple[int, int, int, int]) -> float:
    """0..1-ish quality for ranking embeddings."""
    det = float(getattr(face, "det_score", 0.0) or 0.0)
    x1, y1, x2, y2 = bbox
    side = float(min(max(1, x2 - x1), max(1, y2 - y1)))
    size_s = min(1.0, side / 90.0)
    # landmark / pose hint if present
    pose_s = 1.0
    kps = getattr(face, "kps", None)
    if kps is not None:
        try:
            pts = np.asarray(kps, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] >= 5:
                # eye distance as frontal proxy
                eye = float(np.linalg.norm(pts[0] - pts[1]))
                pose_s = float(np.clip(eye / max(side * 0.22, 1.0), 0.35, 1.0))
        except Exception:
            pose_s = 0.8
    return float(np.clip(0.55 * det + 0.30 * size_s + 0.15 * pose_s, 0.0, 1.0))


class SpatialIdTracker:
    """Lightweight IoU tracker when ByteTrack id is missing."""

    def __init__(self, iou_thresh: float = 0.25, ttl: float = 1.8) -> None:
        self.iou_thresh = iou_thresh
        self.ttl = ttl
        self._next = 1
        self._tracks: dict[int, tuple[tuple[int, int, int, int], float]] = {}

    def assign(self, bbox: tuple[int, int, int, int], now: float | None = None) -> int:
        now = time.time() if now is None else now
        dead = [tid for tid, (_b, t) in self._tracks.items() if now - t > self.ttl]
        for tid in dead:
            self._tracks.pop(tid, None)
        best_tid, best_iou = -1, self.iou_thresh
        for tid, (bb, _t) in self._tracks.items():
            v = _iou(bbox, bb)
            if v > best_iou:
                best_iou, best_tid = v, tid
        if best_tid >= 0:
            self._tracks[best_tid] = (bbox, now)
            return best_tid
        tid = self._next
        self._next += 1
        self._tracks[tid] = (bbox, now)
        return tid


class FaceBank:
    def __init__(self, cfg: Any) -> None:
        self.window_sec = float(getattr(cfg, "FACE_TEMPORAL_SEC", 2.0))
        self.match_thresh = float(getattr(cfg, "FACE_MATCH_THRESHOLD", 0.36))
        self.soft_thresh = float(getattr(cfg, "FACE_SOFT_THRESHOLD", 0.28))
        self.match_margin = float(getattr(cfg, "FACE_MATCH_MARGIN", 0.05))
        self.match_margin_soft = float(getattr(cfg, "FACE_MATCH_MARGIN_SOFT", 0.03))
        self.confirm_hits = int(getattr(cfg, "FACE_CONFIRM_HITS", 2))
        self.hold_sec = float(getattr(cfg, "FACE_IDENTITY_HOLD_SEC", 4.0))
        self.min_quality_unknown = float(getattr(cfg, "FACE_UNKNOWN_MIN_QUALITY", 0.45))
        self.top_k = int(getattr(cfg, "FACE_TEMPORAL_TOP_K", 5))
        self.room_enabled = bool(getattr(cfg, "FACE_ROOM_SESSION_ENABLED", True))
        self.room_empty_sec = float(getattr(cfg, "FACE_ROOM_EMPTY_SEC", 120.0))
        self.room_match_iou = float(getattr(cfg, "FACE_ROOM_MATCH_IOU", 0.10))
        self.room_max_occ = int(getattr(cfg, "FACE_ROOM_MAX_OCCUPANTS", 8))
        self._cams: dict[str, dict[int, TrackFaceState]] = defaultdict(dict)
        self._spatial: dict[str, SpatialIdTracker] = defaultdict(
            lambda: SpatialIdTracker(
                iou_thresh=float(getattr(cfg, "FACE_SPATIAL_IOU_THRESH", 0.20)),
                ttl=float(getattr(cfg, "FACE_SPATIAL_TTL_SEC", 12.0)),
            )
        )
        self._room: dict[str, list[RoomOccupant]] = defaultdict(list)
        self._room_empty_since: dict[str, float | None] = {}

    def register_room_identity(
        self,
        cam_id: str,
        name: str,
        score: float,
        person_bbox: tuple[int, int, int, int],
        face_bbox: tuple[int, int, int, int] | None = None,
        now: float | None = None,
    ) -> None:
        if not self.room_enabled or not name or name in ("Noma'lum", "Rasm?"):
            return
        now = time.time() if now is None else now
        room = self._room[cam_id]
        best_i, best_v = -1, self.room_match_iou
        for i, occ in enumerate(room):
            if occ.name == name:
                best_i, best_v = i, 1.0
                break
            v = _iou(person_bbox, occ.last_person_bbox)
            if v > best_v:
                best_i, best_v = i, v
        if best_i >= 0:
            occ = room[best_i]
            occ.name = name
            occ.score = max(occ.score, score)
            occ.last_person_bbox = person_bbox
            if face_bbox is not None:
                occ.last_face_bbox = face_bbox
            occ.last_seen = now
        else:
            room.append(
                RoomOccupant(
                    name=name,
                    score=score,
                    last_person_bbox=person_bbox,
                    last_face_bbox=face_bbox,
                    last_seen=now,
                    confirmed_at=now,
                )
            )
        if len(room) > self.room_max_occ:
            room.sort(key=lambda o: o.last_seen, reverse=True)
            del room[self.room_max_occ :]

    def match_room_person(
        self,
        cam_id: str,
        person_bbox: tuple[int, int, int, int],
        now: float | None = None,
        allow_single_fallback: bool = True,
    ) -> tuple[str, float] | None:
        if not self.room_enabled:
            return None
        now = time.time() if now is None else now
        room = self._room.get(cam_id) or []
        if not room:
            return None
        best: RoomOccupant | None = None
        best_iou = self.room_match_iou
        for occ in room:
            if now - occ.last_seen > self.room_empty_sec:
                continue
            v = _iou(person_bbox, occ.last_person_bbox)
            if v > best_iou:
                best_iou, best = v, occ
        if best is None and allow_single_fallback:
            # Yagona xonada yagona odam — orqa tomonda ham ushlab turish
            active = [o for o in room if now - o.last_seen <= self.room_empty_sec]
            if len(active) == 1:
                cx = 0.5 * (person_bbox[0] + person_bbox[2])
                cy = 0.5 * (person_bbox[1] + person_bbox[3])
                ox1, oy1, ox2, oy2 = active[0].last_person_bbox
                ocx, ocy = 0.5 * (ox1 + ox2), 0.5 * (oy1 + oy2)
                pw = max(1.0, person_bbox[2] - person_bbox[0])
                ph = max(1.0, person_bbox[3] - person_bbox[1])
                dist = ((cx - ocx) / pw) ** 2 + ((cy - ocy) / ph) ** 2
                if dist <= 1.1:
                    best = active[0]
        if best is None:
            return None
        best.last_person_bbox = person_bbox
        best.last_seen = now
        return best.name, best.score

    def sync_room(
        self,
        cam_id: str,
        people_bboxes: list[tuple[int, int, int, int]],
        now: float | None = None,
    ) -> bool:
        """Xona bo‘shasa True qaytaradi (sessiya tugadi)."""
        if not self.room_enabled:
            return False
        now = time.time() if now is None else now
        if not people_bboxes:
            if cam_id not in self._room_empty_since or self._room_empty_since[cam_id] is None:
                self._room_empty_since[cam_id] = now
            elif now - float(self._room_empty_since[cam_id]) >= self.room_empty_sec:
                self._room[cam_id].clear()
                self._room_empty_since[cam_id] = None
                return True
            return False
        self._room_empty_since[cam_id] = None
        matched_names: set[str] = set()
        for pb in people_bboxes:
            hit = self.match_room_person(cam_id, pb, now=now)
            if hit is not None:
                matched_names.add(hit[0])
        # Uzoq vaqt ko‘rinmagan xona a’zolarini olib tashlash
        self._room[cam_id] = [
            o for o in self._room[cam_id]
            if o.name in matched_names or now - o.last_seen <= self.room_empty_sec * 0.5
        ]
        return False

    def resolve_track_id(
        self,
        cam_id: str,
        bbox: tuple[int, int, int, int],
        hint_id: int | None = None,
    ) -> int:
        if hint_id is not None and hint_id >= 0:
            return int(hint_id)
        return self._spatial[cam_id].assign(bbox)

    def _state(self, cam_id: str, track_id: int) -> TrackFaceState:
        bucket = self._cams[cam_id]
        if track_id not in bucket:
            bucket[track_id] = TrackFaceState()
        return bucket[track_id]

    def _prune_samples(self, st: TrackFaceState, now: float) -> None:
        while st.samples and now - st.samples[0].ts > self.window_sec:
            st.samples.popleft()

    def _best_emb(self, st: TrackFaceState) -> tuple[np.ndarray | None, float, FaceSample | None]:
        if not st.samples:
            return None, 0.0, None
        ranked = sorted(st.samples, key=lambda s: s.quality, reverse=True)
        top = ranked[: max(1, self.top_k)]
        # quality-weighted mean of top embeddings (L2 renorm)
        w = np.array([max(1e-3, s.quality) for s in top], dtype=np.float32)
        w /= w.sum()
        emb = np.zeros_like(top[0].emb, dtype=np.float32)
        for wi, s in zip(w, top):
            emb += wi * s.emb
        n = float(np.linalg.norm(emb) + 1e-8)
        emb = emb / n
        return emb, float(top[0].quality), top[0]

    def update(
        self,
        cam_id: str,
        track_id: int,
        face: Any,
        emb: np.ndarray,
        bbox: tuple[int, int, int, int],
        db_gallery: face_match.FaceGallery,
        now: float | None = None,
        spoof_ok: bool = True,
        confirm_ok: bool = True,
        live_detail: str = "",
        person_bbox: tuple[int, int, int, int] | None = None,
        allow_single_fallback: bool = True,
    ) -> FaceDecision:
        now = time.time() if now is None else now
        st = self._state(cam_id, track_id)
        st.last_seen = now
        st.last_bbox = bbox
        q = face_sample_quality(face, bbox)
        if not spoof_ok:
            room = (
                self.match_room_person(cam_id, person_bbox, now, allow_single_fallback=allow_single_fallback)
                if person_bbox
                else None
            )
            if room is not None:
                name, score = room
                return FaceDecision(
                    name=name,
                    score=score,
                    bbox=bbox,
                    confirmed=True,
                    is_unknown=False,
                    pending=False,
                    quality=q,
                    detail="room-hold-live",
                    emb=emb,
                )
            return FaceDecision(
                name="Rasm?",
                score=0.0,
                bbox=bbox,
                confirmed=False,
                is_unknown=False,
                pending=True,
                quality=q,
                detail=live_detail or "spoof",
                emb=emb,
            )

        st.samples.append(
            FaceSample(
                emb=emb.astype(np.float32),
                quality=q,
                bbox=bbox,
                ts=now,
                det_score=float(getattr(face, "det_score", 0.0) or 0.0),
            )
        )
        self._prune_samples(st, now)

        best_emb, best_q, best_s = self._best_emb(st)
        if best_emb is None or best_s is None:
            return FaceDecision(
                name="Noma'lum",
                score=0.0,
                bbox=bbox,
                confirmed=False,
                is_unknown=False,
                pending=True,
                quality=q,
                detail="empty",
                emb=emb,
            )

        mr = self._match(best_emb, db_gallery)
        mr1 = self._match(best_s.emb, db_gallery)
        if mr1.score > mr.score:
            mr = mr1
            best_emb = best_s.emb
        name, score = mr.name, mr.score

        # Refresh / keep confirmed identity
        if score >= self.match_thresh and name != "Noma'lum" and not mr.ambiguous:
            if confirm_ok:
                was_new = st.confirmed_name != name or (now - st.confirmed_ts) > self.hold_sec
                st.confirmed_name = name
                st.confirmed_score = score
                st.confirmed_ts = now
                st.candidate_name = None
                st.candidate_hits = 0
                if person_bbox is not None:
                    self.register_room_identity(cam_id, name, score, person_bbox, bbox, now)
                return FaceDecision(
                    name=name,
                    score=score,
                    bbox=best_s.bbox,
                    confirmed=True,
                    is_unknown=False,
                    pending=False,
                    quality=best_q,
                    detail=f"match q={best_q:.2f} n={len(st.samples)}",
                    emb=best_emb,
                    just_confirmed=was_new,
                )
            return FaceDecision(
                name=f"{name}?",
                score=score,
                bbox=best_s.bbox,
                confirmed=False,
                is_unknown=False,
                pending=True,
                quality=best_q,
                detail=live_detail or "liveness-wait",
                emb=best_emb,
            )

        if (
            st.confirmed_name
            and st.confirmed_name != "Noma'lum"
            and now - st.confirmed_ts <= self.hold_sec
        ):
            if person_bbox is not None:
                self.register_room_identity(
                    cam_id, st.confirmed_name, st.confirmed_score, person_bbox, bbox, now
                )
            return FaceDecision(
                name=st.confirmed_name,
                score=st.confirmed_score,
                bbox=bbox,
                confirmed=True,
                is_unknown=False,
                pending=False,
                quality=q,
                detail=f"hold q={q:.2f}",
                emb=emb,
                just_confirmed=False,
            )

        if score >= self.soft_thresh and name != "Noma'lum" and not mr.ambiguous:
            if st.candidate_name == name:
                st.candidate_hits += 1
                st.candidate_score = max(st.candidate_score, score)
            else:
                st.candidate_name = name
                st.candidate_hits = 1
                st.candidate_score = score
            if st.candidate_hits >= self.confirm_hits:
                if not confirm_ok:
                    return FaceDecision(
                        name=f"{name}?",
                        score=st.candidate_score,
                        bbox=best_s.bbox,
                        confirmed=False,
                        is_unknown=False,
                        pending=True,
                        quality=best_q,
                        detail=live_detail or "liveness-wait",
                        emb=best_emb,
                    )
                st.confirmed_name = name
                st.confirmed_score = st.candidate_score
                st.confirmed_ts = now
                st.candidate_name = None
                st.candidate_hits = 0
                if person_bbox is not None:
                    self.register_room_identity(cam_id, name, st.confirmed_score, person_bbox, best_s.bbox, now)
                return FaceDecision(
                    name=name,
                    score=st.confirmed_score,
                    bbox=best_s.bbox,
                    confirmed=True,
                    is_unknown=False,
                    pending=False,
                    quality=best_q,
                    detail=f"soft-confirm q={best_q:.2f}",
                    emb=best_emb,
                    just_confirmed=True,
                )
            return FaceDecision(
                name=f"{name}?",
                score=score,
                bbox=best_s.bbox,
                confirmed=False,
                is_unknown=False,
                pending=True,
                quality=best_q,
                detail=f"candidate {st.candidate_hits}/{self.confirm_hits}",
                emb=best_emb,
            )

        # Unknown — xonada allaqachon tanilgan bo‘lsa, begona deb chiqarmaslik
        room = (
            self.match_room_person(cam_id, person_bbox, now, allow_single_fallback=allow_single_fallback)
            if person_bbox
            else None
        )
        if room is not None:
            rname, rscore = room
            return FaceDecision(
                name=rname,
                score=rscore,
                bbox=best_s.bbox,
                confirmed=True,
                is_unknown=False,
                pending=False,
                quality=best_q,
                detail="room-hold-face",
                emb=best_emb,
            )

        st.candidate_name = None
        st.candidate_hits = 0
        is_unk = best_q >= self.min_quality_unknown and confirm_ok
        unk_detail = mr.detail if mr.ambiguous else f"unknown q={best_q:.2f}"
        if not confirm_ok:
            unk_detail = live_detail or "liveness-wait"
        return FaceDecision(
            name="Noma'lum",
            score=score,
            bbox=best_s.bbox,
            confirmed=False,
            is_unknown=is_unk,
            pending=not is_unk,
            quality=best_q,
            detail=unk_detail,
            emb=best_emb,
            log_unknown=is_unk,
        )

    def _match(self, emb: np.ndarray, gallery: face_match.FaceGallery) -> face_match.MatchResult:
        return face_match.match_gallery(
            emb,
            gallery,
            threshold=self.match_thresh,
            soft_threshold=self.soft_thresh,
            min_margin=self.match_margin,
            min_margin_soft=self.match_margin_soft,
        )

    def prune(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        for cam_id, bucket in list(self._cams.items()):
            dead = [tid for tid, st in bucket.items() if now - st.last_seen > self.window_sec * 3]
            for tid in dead:
                bucket.pop(tid, None)
