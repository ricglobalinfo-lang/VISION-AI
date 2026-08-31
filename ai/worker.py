"""
Multi-camera AI worker:
  RTSP (per camera) -> YOLO + InsightFace -> annotated MJPEG
Low latency: latest-frame grabbers + ffmpeg fallback for HEVC OEM cams.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, request, send_from_directory, session, stream_with_context
from werkzeug.utils import secure_filename
import hmac

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import enroll_faces  # noqa: E402
import face_match  # noqa: E402
import face_bank  # noqa: E402
import face_antispoof  # noqa: E402
import face_liveness  # noqa: E402
import face_quality  # noqa: E402
import fight_action  # noqa: E402
import fight_detect  # noqa: E402
import fight_tracks  # noqa: E402
import litter_detect  # noqa: E402
import unknown_store  # noqa: E402


def _bootstrap_cuda_dlls() -> None:
    """Put torch CUDA DLLs on PATH so onnxruntime-gpu can load cublas/cudnn."""
    try:
        import torch

        lib = Path(torch.__file__).resolve().parent / "lib"
        if not lib.is_dir():
            return
        lib_s = str(lib)
        path = os.environ.get("PATH", "")
        if lib_s not in path.split(os.pathsep):
            os.environ["PATH"] = lib_s + os.pathsep + path
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_s)
            except OSError:
                pass
    except Exception:
        pass


_bootstrap_cuda_dlls()

_events: deque[dict[str, Any]] = deque(maxlen=300)
_events_lock = threading.Lock()
_cooldown: dict[str, float] = {}

_face_db_lock = threading.Lock()
_db_gallery = face_match.FaceGallery.empty()
_reload_faces = threading.Event()
_face_app_holder: dict[str, Any] = {"app": None}
_enroll_lock = threading.Lock()

# Per-camera live state (jpeg separate — avoids lock stalls)
_cam_lock = threading.Lock()
_cam_state: dict[str, dict[str, Any]] = {}
_jpeg_map: dict[str, bytes] = {}
_jpeg_lock = threading.Lock()
_focus_lock = threading.Lock()
_focus_cam_id = config.CAMERAS[0]["id"] if config.CAMERAS else ""
_focus_mode = "single"  # single | grid

_grabbers: dict[str, Any] = {}
_grabber_quality: dict[str, str] = {}  # cam_id -> "main" | "sub"
_grabbers_lock = threading.Lock()


def _init_cam_state() -> None:
    with _cam_lock:
        for cam in config.CAMERAS:
            _cam_state[cam["id"]] = {
                "id": cam["id"],
                "name": cam["name"],
                "ip": cam["ip"],
                "fps": 0.0,
                "lag_ms": 0,
                "running": False,
                "error": "",
            }


def set_cam(cam_id: str, **kwargs: Any) -> None:
    jpeg = kwargs.pop("jpeg", None)
    if jpeg is not None:
        with _jpeg_lock:
            _jpeg_map[cam_id] = jpeg
    if not kwargs:
        return
    with _cam_lock:
        st = _cam_state.get(cam_id)
        if st:
            st.update(kwargs)


def get_jpeg(cam_id: str) -> bytes | None:
    with _jpeg_lock:
        return _jpeg_map.get(cam_id)


def get_cam(cam_id: str) -> dict[str, Any] | None:
    with _cam_lock:
        st = _cam_state.get(cam_id)
        return dict(st) if st else None


def list_cam_status() -> list[dict[str, Any]]:
    with _jpeg_lock:
        has = set(_jpeg_map.keys())
    with _cam_lock:
        out = []
        for cam in config.CAMERAS:
            st = _cam_state.get(cam["id"], {})
            out.append({
                "id": cam["id"],
                "name": cam["name"],
                "ip": cam["ip"],
                "running": bool(st.get("running")),
                "fps": float(st.get("fps") or 0),
                "lag_ms": int(st.get("lag_ms") or 0),
                "error": st.get("error") or "",
                "has_frame": cam["id"] in has,
            })
        return out


class LatestFrameGrabber:
    """OpenCV grabber — discards backlog for low latency."""

    def __init__(self, url: str, cam_id: str) -> None:
        self.url = url
        self.cam_id = cam_id
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._running = True
        self._error = ""
        self._thread = threading.Thread(target=self._loop, name=f"grab-{cam_id}", daemon=True)
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _loop(self) -> None:
        cap = self._open()
        fail = 0
        while self._running:
            if not cap.isOpened():
                self._error = "RTSP ochilmadi"
                time.sleep(1.5)
                cap.release()
                cap = self._open()
                continue
            grabbed = False
            # 2–3 grab: buffer flush, lekin ts yangilanishi sekinlashmasin
            for _ in range(3):
                if not cap.grab():
                    break
                grabbed = True
            if not grabbed:
                fail += 1
                if fail > 40:
                    self._error = "Kadr yo‘q"
                    cap.release()
                    time.sleep(1.0)
                    cap = self._open()
                    fail = 0
                else:
                    time.sleep(0.01)
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                fail += 1
                continue
            fail = 0
            self._error = ""
            with self._lock:
                self._frame = frame
                self._ts = time.time()
        cap.release()

    def get(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._ts

    @property
    def error(self) -> str:
        return self._error

    def stop(self) -> None:
        self._running = False


class FFmpegFrameGrabber:
    """ffmpeg pipe grabber — reliable for HEVC OEM cameras."""

    def __init__(self, url: str, cam_id: str, width: int = 960) -> None:
        self.url = url
        self.cam_id = cam_id
        self.width = width
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._running = True
        self._error = ""
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._loop, name=f"ffgrab-{cam_id}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        ff = str(config.FFMPEG) if config.FFMPEG.is_file() else "ffmpeg"
        while self._running:
            h = max(270, int(self.width * 9 / 16))
            cmd = [
                ff, "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", "tcp",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-max_delay", "0",
                "-probesize", "32",
                "-analyzeduration", "0",
                "-i", self.url,
                "-an",
                "-vf", (
                    f"scale={self.width}:{h}:force_original_aspect_ratio=decrease:"
                    f"flags=fast_bilinear,pad={self.width}:{h}:(ow-iw)/2:(oh-ih)/2"
                ),
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-",
            ]
            frame_bytes = self.width * h * 3
            try:
                # Kichik pipe — eski kadrlar to‘planib lag oshmasin
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=max(frame_bytes, 65536),
                )
            except Exception as e:
                self._error = f"ffmpeg: {e}"
                time.sleep(2)
                continue
            self._proc = proc

            assert proc.stdout is not None
            while self._running:
                raw = proc.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    self._error = "ffmpeg oqim uzildi"
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, self.width, 3)).copy()
                with self._lock:
                    self._frame = frame
                    self._ts = time.time()
                self._error = ""
            try:
                proc.kill()
            except Exception:
                pass
            self._proc = None
            if self._running:
                time.sleep(1.0)

    def get(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._ts

    @property
    def error(self) -> str:
        return self._error

    def stop(self) -> None:
        self._running = False
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass


def cam_rtsp(cam: dict[str, Any], quality: str) -> str:
    if quality == "main":
        return str(cam.get("rtsp_main") or cam.get("rtsp") or "")
    return str(cam.get("rtsp_sub") or cam.get("rtsp") or "")


def make_grabber(cam: dict[str, Any], quality: str = "sub"):
    url = cam_rtsp(cam, quality)
    width = int(
        getattr(config, "DISPLAY_WIDTH_MAIN", 1440)
        if quality == "main"
        else getattr(config, "DISPLAY_WIDTH_SUB", config.DISPLAY_WIDTH)
    )
    if cam.get("brand") == "oem":
        return FFmpegFrameGrabber(url, cam["id"], width=width)
    return LatestFrameGrabber(url, cam["id"])


def is_hq_view(cam_id: str) -> bool:
    with _focus_lock:
        return _focus_mode == "single" and _focus_cam_id == cam_id


def sync_grabber_streams() -> None:
    """Focus kamerani main ga, qolganlarini sub ga ulaydi."""
    with _focus_lock:
        mode = _focus_mode
        focus = _focus_cam_id
    for cam in config.CAMERAS:
        want = "main" if (mode == "single" and cam["id"] == focus) else "sub"
        with _grabbers_lock:
            cur = _grabber_quality.get(cam["id"])
            if cur == want and cam["id"] in _grabbers:
                continue
            old = _grabbers.get(cam["id"])
        print(f"[stream] {cam['id']} -> {want}")
        new_g = make_grabber(cam, want)
        # biroz ulanishga vaqt
        time.sleep(0.35)
        with _grabbers_lock:
            _grabbers[cam["id"]] = new_g
            _grabber_quality[cam["id"]] = want
        if old is not None:
            try:
                old.stop()
            except Exception:
                pass


def get_grabber(cam_id: str) -> Any | None:
    with _grabbers_lock:
        return _grabbers.get(cam_id)


def log_event(kind: str, label: str, detail: str = "", camera: str = "") -> None:
    now = time.time()
    key = f"{kind}:{camera}:{label}"
    if kind == "unknown":
        cool = config.UNKNOWN_COOLDOWN_SEC
    elif kind == "fight":
        cool = float(getattr(config, "FIGHT_COOLDOWN_SEC", 12.0))
    elif kind == "litter":
        cool = float(getattr(config, "LITTER_COOLDOWN_SEC", 20.0))
    else:
        cool = config.KNOWN_COOLDOWN_SEC
    last = _cooldown.get(key, 0.0)
    if now - last < cool:
        return
    _cooldown[key] = now

    evt = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "epoch": now,
        "kind": kind,
        "label": label,
        "detail": detail,
        "camera": camera,
    }
    with _events_lock:
        _events.appendleft(evt)
    try:
        with open(config.EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except OSError:
        pass
    if kind == "unknown":
        try:
            import winsound
            winsound.Beep(880, 100)
        except Exception:
            pass
    elif kind == "fight":
        try:
            import winsound
            winsound.Beep(1200, 180)
            winsound.Beep(900, 220)
            winsound.Beep(1400, 280)
        except Exception:
            pass
    elif kind == "litter":
        try:
            import winsound
            winsound.Beep(700, 120)
            winsound.Beep(950, 160)
        except Exception:
            pass


def load_face_db() -> face_match.FaceGallery:
    return face_match.FaceGallery.from_npz(config.FACE_DB_PATH)


def apply_face_db(gallery: face_match.FaceGallery) -> None:
    global _db_gallery
    with _face_db_lock:
        _db_gallery = gallery


def get_face_db() -> face_match.FaceGallery:
    with _face_db_lock:
        return _db_gallery


def yolo_infer_kwargs() -> dict[str, Any]:
    """Shared Ultralytics device/half settings."""
    import torch
    raw = getattr(config, "YOLO_DEVICE", 0)
    half = bool(getattr(config, "YOLO_HALF", True))
    if raw == "cpu" or raw == -1:
        return {"device": "cpu", "half": False}
    if not torch.cuda.is_available():
        return {"device": "cpu", "half": False}
    return {"device": int(raw) if str(raw).isdigit() or isinstance(raw, int) else raw, "half": half}


def pose_infer_kwargs() -> dict[str, Any]:
    kw = yolo_infer_kwargs()
    if "FIGHT_POSE_HALF" in dir(config):
        if kw.get("device") != "cpu":
            kw["half"] = bool(getattr(config, "FIGHT_POSE_HALF", True))
    return kw


def match_face(emb: np.ndarray, gallery: face_match.FaceGallery) -> tuple[str, float]:
    kw = face_match.gallery_cfg(config)
    mr = face_match.match_gallery(emb, gallery, **kw)
    return mr.name, mr.score


def yolo_label(name: str) -> str:
    labels = getattr(config, "YOLO_LABELS_UZ", None) or {}
    return labels.get(name, name)


def yolo_keep(name: str, conf: float, xyxy: tuple[float, float, float, float] | list[float]) -> bool:
    """Drop weak / shape-implausible detections (e.g. chair → laptop)."""
    base = float(getattr(config, "YOLO_CONF", 0.45))
    per_class = getattr(config, "YOLO_CLASS_MIN_CONF", None) or {}
    need = float(per_class.get(name, base))
    if conf < need:
        return False
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    aspect = w / h
    wide = getattr(config, "YOLO_WIDE_CLASSES", ()) or ()
    min_ar = float(getattr(config, "YOLO_WIDE_MIN_ASPECT", 0.90))
    if name in wide and aspect < min_ar:
        return False
    tall_false = getattr(config, "YOLO_TALL_FALSE_CLASSES", ()) or ()
    tall_max_ar = float(getattr(config, "YOLO_TALL_MAX_ASPECT", 0.85))
    tall_min_conf = float(getattr(config, "YOLO_TALL_MIN_CONF", 0.78))
    if name in tall_false and aspect < tall_max_ar and conf < tall_min_conf:
        return False
    return True


def _face_ok(face, frame_bgr: np.ndarray, person_boxes: list[tuple[int, int, int, int]]) -> bool:
    ok, _reason = face_quality.is_real_face(
        face,
        frame_bgr,
        person_boxes,
        min_det_score=float(getattr(config, "FACE_MIN_DET_SCORE", 0.50)),
        min_side=int(getattr(config, "FACE_MIN_SIDE", 28)),
        min_brightness=float(getattr(config, "FACE_MIN_BRIGHTNESS", 18.0)),
        min_sharpness=float(getattr(config, "FACE_MIN_SHARPNESS", 8.0)),
        require_person=bool(getattr(config, "FACE_REQUIRE_PERSON", False)),
    )
    return ok


def _iou_box(
    a: tuple[int, int, int, int] | fight_detect.PersonBox,
    b: tuple[int, int, int, int] | fight_detect.PersonBox,
) -> float:
    if hasattr(a, "x1"):
        ax1, ay1, ax2, ay2 = int(a.x1), int(a.y1), int(a.x2), int(a.y2)  # type: ignore[attr-defined]
    else:
        ax1, ay1, ax2, ay2 = a  # type: ignore[misc]
    if hasattr(b, "x1"):
        bx1, by1, bx2, by2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)  # type: ignore[attr-defined]
    else:
        bx1, by1, bx2, by2 = b  # type: ignore[misc]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(1, ax2 - ax1) * max(1, ay2 - ay1) + max(1, bx2 - bx1) * max(1, by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _hint_track_id(
    person: fight_detect.PersonBox,
    track_bank: fight_tracks.PoseTrackBank | None,
    cam_id: str,
) -> int | None:
    if track_bank is None:
        return person.track_id if person.track_id >= 0 else None
    st = track_bank.cam(cam_id)
    best_tid, best = None, 0.15
    for tid in list(st.skeletons.keys()):
        box = st.latest_box(tid)
        if box is None:
            continue
        v = _iou_box(person, box)
        if v > best:
            best, best_tid = v, tid
    if best_tid is not None:
        return int(best_tid)
    return person.track_id if person.track_id >= 0 else None


def _detect_faces_on_people(
    face_app,
    display: np.ndarray,
    people: list[fight_detect.PersonBox],
    cam_id: str = "",
    track_bank: fight_tracks.PoseTrackBank | None = None,
    bank: face_bank.FaceBank | None = None,
) -> list[tuple[Any, tuple[int, int, int, int], int]]:
    """
    Run InsightFace on each YOLO person crop (upper body), upscaled.
    Returns list of (face, display_bbox, track_id).
    """
    if face_app is None or not people:
        return []
    h, w = display.shape[:2]
    out: list[tuple[Any, tuple[int, int, int, int], int]] = []
    ordered = sorted(people, key=lambda p: p.w * p.h, reverse=True)[:6]
    min_side = int(getattr(config, "FACE_CROP_MIN", 224))

    for p in ordered:
        # Head/shoulders — wider crop helps partial faces
        bw, bh = max(1, p.x2 - p.x1), max(1, p.y2 - p.y1)
        pad_x = int(bw * 0.22)
        y1 = max(0, p.y1 - int(bh * 0.12))
        y2 = min(h, p.y1 + int(bh * 0.62))
        x1 = max(0, p.x1 - pad_x)
        x2 = min(w, p.x2 + pad_x)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        crop = display[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        scale = 1.0
        if max(ch, cw) < min_side:
            scale = min_side / float(max(ch, cw))
            crop = cv2.resize(
                crop,
                (max(1, int(cw * scale)), max(1, int(ch * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        try:
            faces = face_app.get(crop)
        except Exception:
            faces = []
        if not faces:
            continue
        # Eng katta + ishonchli yuz (mayda chalkash “yuz”larni emas)
        def _face_rank(f: Any) -> float:
            bb = [float(v) for v in f.bbox.tolist()]
            area = max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
            return area * float(getattr(f, "det_score", 0.0) or 0.0)

        face = max(faces, key=_face_rank)
        # Softer gate for temporal bank; still drop obvious junk
        if not _face_ok(face, crop, [(0, 0, crop.shape[1], crop.shape[0])]):
            continue
        fx1, fy1, fx2, fy2 = [float(v) for v in face.bbox.tolist()]
        dx1 = int(x1 + fx1 / scale)
        dy1 = int(y1 + fy1 / scale)
        dx2 = int(x1 + fx2 / scale)
        dy2 = int(y1 + fy2 / scale)
        bbox = (dx1, dy1, dx2, dy2)
        min_disp = int(getattr(config, "FACE_MIN_DISPLAY_SIDE", 40))
        if min(dx2 - dx1, dy2 - dy1) < min_disp:
            continue
        hint = _hint_track_id(p, track_bank, cam_id)
        if bank is not None:
            tid = bank.resolve_track_id(cam_id or "cam", bbox, hint)
        else:
            tid = int(hint) if hint is not None else -1
        out.append((face, bbox, tid))
    return out


def _person_bbox_for_face(
    face_bbox: tuple[int, int, int, int],
    people: list[fight_detect.PersonBox],
) -> tuple[int, int, int, int] | None:
    fx1, fy1, fx2, fy2 = face_bbox
    fcx = 0.5 * (fx1 + fx2)
    fcy = 0.5 * (fy1 + fy2)
    best: fight_detect.PersonBox | None = None
    best_score = 0.0
    for p in people:
        if p.x1 <= fcx <= p.x2 and p.y1 <= fcy <= p.y1 + 0.75 * max(1, p.y2 - p.y1):
            area = p.w * p.h
            if area > best_score:
                best_score = area
                best = p
        else:
            v = _iou_box(face_bbox, p)
            if v > 0.05 and p.w * p.h > best_score:
                best_score = p.w * p.h
                best = p
    if best is None:
        return None
    return int(best.x1), int(best.y1), int(best.x2), int(best.y2)


def _face_hit_covers_person(
    face_bbox: tuple[int, int, int, int],
    person: fight_detect.PersonBox,
) -> bool:
    fx1, fy1, fx2, fy2 = face_bbox
    fcx = 0.5 * (fx1 + fx2)
    fcy = 0.5 * (fy1 + fy2)
    return (
        person.x1 <= fcx <= person.x2
        and person.y1 <= fcy <= person.y1 + 0.75 * max(1, person.y2 - person.y1)
    )


def _apply_room_presence(
    people: list[fight_detect.PersonBox],
    face_bboxes: list[tuple[int, int, int, int]],
    cam: dict[str, Any],
    items: list[tuple] | None,
    bank: face_bank.FaceBank | None,
    liveness: face_liveness.MotionLiveness | None = None,
) -> None:
    """Yuz ko‘rinmasa ham (orqa tomonda) xonadagi tanilgan shaxsni saqlash."""
    if bank is None:
        return
    people_boxes = [(int(p.x1), int(p.y1), int(p.x2), int(p.y2)) for p in people]
    cleared = bank.sync_room(cam["id"], people_boxes)
    if cleared and liveness is not None:
        liveness.clear_cam(cam["id"])
    if not people:
        return
    used_names: set[str] = set()
    allow_single_fallback = len(people) <= 1
    for p in people:
        pb = (int(p.x1), int(p.y1), int(p.x2), int(p.y2))
        if any(_face_hit_covers_person(fb, p) for fb in face_bboxes):
            continue
        room = bank.match_room_person(
            cam["id"], pb, allow_single_fallback=allow_single_fallback
        )
        if room is None:
            continue
        name, score = room
        if name in used_names:
            continue
        used_names.add(name)
        color = config.COLOR_KNOWN
        tag = f"{name} ({score:.2f})"
        if items is not None:
            items.append((pb[0], pb[1], pb[2], pb[3], color, tag))


def _apply_face_hits(
    hits: list[tuple[Any, tuple[int, int, int, int], int]] | list[tuple[Any, tuple[int, int, int, int]]],
    display: np.ndarray,
    cam: dict[str, Any],
    items: list[tuple] | None,
    draw_direct: bool = False,
    bank: face_bank.FaceBank | None = None,
    antispoof: face_antispoof.AntiSpoofEnsemble | None = None,
    liveness: face_liveness.MotionLiveness | None = None,
    people: list[fight_detect.PersonBox] | None = None,
) -> list[tuple[int, int, int, int]]:
    gallery = get_face_db()
    face_boxes: list[tuple[int, int, int, int]] = []
    people = people or []
    allow_single_fallback = len(people) <= 1
    for hit in hits:
        if len(hit) == 3:
            face, (x1, y1, x2, y2), track_id = hit  # type: ignore[misc]
        else:
            face, (x1, y1, x2, y2) = hit  # type: ignore[misc]
            track_id = -1
        emb = face.normed_embedding.astype(np.float32)
        bbox = (x1, y1, x2, y2)
        face_boxes.append(bbox)
        person_bbox = _person_bbox_for_face(bbox, people)

        if bank is not None:
            tid = bank.resolve_track_id(cam["id"], bbox, track_id if track_id >= 0 else None)
        else:
            tid = track_id

        spoof = (
            antispoof.predict(display, bbox)
            if antispoof is not None
            else face_antispoof.AntiSpoofResult(True, 1.0, 0.0, "disabled")
        )
        motion = (
            liveness.update(cam["id"], tid, face, bbox)
            if liveness is not None
            else face_liveness.MotionState(True, True, 1.0, 1, "disabled")
        )
        spoof_ok = spoof.is_real
        motion_bypass = float(getattr(config, "FACE_ANTISPOOF_MOTION_BYPASS", 0.72))
        motion_ok = motion.passed or spoof.real_prob >= motion_bypass
        confirm_ok = spoof_ok and motion_ok
        live_detail = ""
        if not spoof_ok:
            live_detail = f"spoof {spoof.detail}"
        elif not motion_ok:
            live_detail = f"motion {motion.detail}"

        if bank is not None:
            dec = bank.update(
                cam["id"],
                tid,
                face,
                emb,
                bbox,
                gallery,
                spoof_ok=spoof_ok,
                confirm_ok=confirm_ok,
                live_detail=live_detail,
                person_bbox=person_bbox,
                allow_single_fallback=allow_single_fallback,
            )
            x1, y1, x2, y2 = dec.bbox
            if not spoof_ok:
                color = getattr(config, "COLOR_SPOOF", (0, 140, 255))
                tag = f"RASM {spoof.real_prob:.2f}"
                if draw_direct:
                    draw_detection(display, x1, y1, x2, y2, color, tag)
                elif items is not None:
                    items.append((x1, y1, x2, y2, color, tag))
                continue
            if not confirm_ok and dec.pending and not dec.confirmed:
                color = config.COLOR_KNOWN if "?" in dec.name else config.COLOR_UNKNOWN
                if not motion.ready:
                    tag = f"Jonlilik... {motion.frames}/{getattr(config, 'FACE_LIVENESS_MIN_FRAMES', 4)}"
                elif dec.name not in ("Noma'lum", "Rasm?"):
                    tag = f"{dec.name} ({dec.score:.2f})"
                else:
                    tag = f"Jonlilik? {motion.motion:.2f}"
                if draw_direct:
                    draw_detection(display, x1, y1, x2, y2, color, tag)
                elif items is not None:
                    items.append((x1, y1, x2, y2, color, tag))
                continue
            if dec.pending and not dec.confirmed and not dec.is_unknown:
                # Soft candidate — show on overlay, do not log/save yet
                color = config.COLOR_KNOWN if "?" in dec.name else config.COLOR_UNKNOWN
                tag = f"{dec.name} ({dec.score:.2f})"
                if draw_direct:
                    draw_detection(display, x1, y1, x2, y2, color, tag)
                elif items is not None:
                    items.append((x1, y1, x2, y2, color, tag))
                continue
            if dec.confirmed:
                color = config.COLOR_KNOWN
                if dec.just_confirmed and not str(dec.detail).startswith("room-"):
                    log_event("known", dec.name, f"score={dec.score:.2f} | {dec.detail}", camera=cam["name"])
                    if liveness is not None:
                        liveness.mark_cam_live(cam["id"])
                tag = f"{dec.name} ({dec.score:.2f})"
            elif dec.is_unknown:
                color = config.COLOR_UNKNOWN
                saved = None
                if dec.log_unknown:
                    try:
                        saved = unknown_store.save_unknown(
                            display, (x1, y1, x2, y2), dec.emb if dec.emb is not None else emb,
                            camera_id=cam["id"], camera_name=cam["name"],
                        )
                    except Exception as e:
                        print(f"unknown save error: {e}")
                    label = (saved or {}).get("label") or "Noma'lum odam"
                    detail = f"score={dec.score:.2f} q={dec.quality:.2f}"
                    if saved and saved.get("saved"):
                        detail += " | rasm saqlandi"
                    log_event("unknown", label, detail, camera=cam["name"])
                    tag = label
                else:
                    tag = f"Noma'lum ({dec.score:.2f})"
            else:
                # Weak unknown — draw only
                color = config.COLOR_UNKNOWN
                tag = f"Noma'lum ({dec.score:.2f})"
            if draw_direct:
                draw_detection(display, x1, y1, x2, y2, color, tag)
            elif items is not None:
                items.append((x1, y1, x2, y2, color, tag))
            continue

        # Legacy single-frame path
        name, score = match_face(emb, gallery)
        if name == "Noma'lum":
            color = config.COLOR_UNKNOWN
            saved = None
            try:
                saved = unknown_store.save_unknown(
                    display, (x1, y1, x2, y2), emb,
                    camera_id=cam["id"], camera_name=cam["name"],
                )
            except Exception as e:
                print(f"unknown save error: {e}")
            label = (saved or {}).get("label") or "Noma'lum odam"
            detail = f"score={score:.2f}"
            if saved and saved.get("saved"):
                detail += " | rasm saqlandi"
            log_event("unknown", label, detail, camera=cam["name"])
            tag = label
        else:
            color = config.COLOR_KNOWN
            log_event("known", name, f"score={score:.2f}", camera=cam["name"])
            tag = f"{name} ({score:.2f})"
        if draw_direct:
            draw_detection(display, x1, y1, x2, y2, color, tag)
        elif items is not None:
            items.append((x1, y1, x2, y2, color, tag))
    return face_boxes


def draw_corners(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Faqat burchaklar — obyekt o'rtasi ochiq qoladi."""
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    cl = max(6, min(18, min(w, h) // 4))
    line = cv2.LINE_AA
    # yuqori-chap
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, thickness, line)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, thickness, line)
    # yuqori-o'ng
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, thickness, line)
    cv2.line(img, (x2, y1), (x2, y1 + cl), color, thickness, line)
    # pastki-chap
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, thickness, line)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, thickness, line)
    # pastki-o'ng
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, thickness, line)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, thickness, line)


def draw_label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 4, 2
    box_h = th + baseline + pad_y * 2
    box_w = tw + pad_x * 2
    y1 = max(0, y - box_h - 1)
    y2 = min(img.shape[0], y1 + box_h)
    x1 = max(0, min(x, img.shape[1] - 1))
    x2 = min(img.shape[1], x1 + box_w)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    tint = np.empty_like(roi)
    tint[:] = color
    cv2.addWeighted(tint, 0.42, roi, 0.58, 0, roi)
    tx = x1 + pad_x
    ty = y2 - baseline - pad_y
    cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detection(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    tag: str | None = None,
) -> None:
    draw_corners(img, int(x1), int(y1), int(x2), int(y2), color, thickness=1)
    if tag:
        draw_label(img, tag, int(x1), int(y1), color)


def rebuild_and_reload() -> dict:
    with _enroll_lock:
        app = _face_app_holder.get("app")
        result = enroll_faces.enroll_all(face_app=app)
        gallery = load_face_db()
        apply_face_db(gallery)
        _reload_faces.set()
        return result


def process_frame(frame: np.ndarray, yolo, class_ids, face_app, cam: dict[str, Any]) -> np.ndarray:
    h0, w0 = frame.shape[:2]
    if w0 > config.DISPLAY_WIDTH:
        ds = config.DISPLAY_WIDTH / float(w0)
        display = cv2.resize(frame, (config.DISPLAY_WIDTH, int(h0 * ds)), interpolation=cv2.INTER_AREA)
    else:
        display = frame

    dh, dw = display.shape[:2]
    if dw > config.AI_WIDTH:
        ai_scale = config.AI_WIDTH / float(dw)
        ai_frame = cv2.resize(display, (config.AI_WIDTH, int(dh * ai_scale)), interpolation=cv2.INTER_AREA)
    else:
        ai_frame = display
        ai_scale = 1.0
    inv = 1.0 / ai_scale if ai_scale else 1.0

    results = yolo.predict(
        ai_frame,
        conf=config.YOLO_CONF,
        classes=class_ids if class_ids else None,
        imgsz=config.YOLO_IMGSZ,
        verbose=False,
        **yolo_infer_kwargs(),
    )
    person_boxes_ai: list[tuple[int, int, int, int]] = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw = yolo.names.get(cls_id, str(cls_id))
            ax1, ay1, ax2, ay2 = box.xyxy[0].tolist()
            if not yolo_keep(raw, conf, (ax1, ay1, ax2, ay2)):
                continue
            label = yolo_label(raw)
            if raw == "person":
                person_boxes_ai.append((int(ax1), int(ay1), int(ax2), int(ay2)))
            x1, y1, x2, y2 = int(ax1 * inv), int(ay1 * inv), int(ax2 * inv), int(ay2 * inv)
            color = config.COLOR_PERSON_BOX if raw == "person" else config.COLOR_OBJECT
            draw_detection(display, x1, y1, x2, y2, color, f"{label} {conf:.2f}")

    if face_app is not None:
        people_disp = [
            fight_detect.PersonBox(x1, y1, x2, y2, 1.0)
            for x1, y1, x2, y2 in [
                (int(a[0] * inv), int(a[1] * inv), int(a[2] * inv), int(a[3] * inv))
                for a in person_boxes_ai
            ]
        ]
        hits = _detect_faces_on_people(face_app, display, people_disp)
        _apply_face_hits(hits, display, cam, items=None, draw_direct=True)

    return display


# Last AI overlays per camera: list of (x1,y1,x2,y2,color,tag)
_overlays: dict[str, list[tuple]] = {}
_overlays_lock = threading.Lock()


def _resize_display(frame: np.ndarray, max_w: int | None = None) -> np.ndarray:
    if max_w is None:
        max_w = int(getattr(config, "DISPLAY_WIDTH_SUB", config.DISPLAY_WIDTH))
    h0, w0 = frame.shape[:2]
    if w0 > max_w:
        ds = max_w / float(w0)
        return cv2.resize(frame, (max_w, int(h0 * ds)), interpolation=cv2.INTER_AREA)
    return frame


def _draw_overlays(display: np.ndarray, cam: dict[str, Any], lag_ms: int, fps: float) -> np.ndarray:
    with _overlays_lock:
        items = list(_overlays.get(cam["id"], []))
    for x1, y1, x2, y2, color, tag in items:
        draw_detection(display, x1, y1, x2, y2, color, tag)
    hud = f"{cam['name']} | {fps:.1f}fps | lag~{lag_ms}ms"
    cv2.putText(display, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
    return display


def preview_loop(cam: dict[str, Any]) -> None:
    """Fast path: always publish newest frame (~1:1), reuse last AI boxes."""
    cid = cam["id"]
    n = 0
    t0 = time.time()
    while True:
        grabber = get_grabber(cid)
        if grabber is None:
            time.sleep(0.05)
            continue
        frame, cap_ts = grabber.get()
        if frame is None:
            set_cam(cid, running=False, error=grabber.error or "kadr kutilmoqda...")
            time.sleep(0.05)
            continue
        lag_ms = max(0, int((time.time() - cap_ts) * 1000)) if cap_ts else 0
        hq = is_hq_view(cid)
        max_w = int(
            getattr(config, "DISPLAY_WIDTH_MAIN", 1440)
            if hq
            else getattr(config, "DISPLAY_WIDTH_SUB", config.DISPLAY_WIDTH)
        )
        jpeg_q = int(
            getattr(config, "JPEG_QUALITY_MAIN", 92)
            if hq
            else getattr(config, "JPEG_QUALITY_SUB", config.JPEG_QUALITY)
        )
        # Grabber allaqachon kerakli kenglikda (ffmpeg scale) — ortiqcha resize yo‘q
        h0, w0 = frame.shape[:2]
        if w0 > max_w:
            display = _resize_display(frame, max_w=max_w)
        else:
            display = frame
        st = get_cam(cid) or {}
        fps = float(st.get("fps") or 0)
        display = _draw_overlays(display, cam, lag_ms, fps)
        ok_jpg, buf = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        if ok_jpg:
            set_cam(cid, jpeg=buf.tobytes(), running=True, error="", lag_ms=lag_ms)
        n += 1
        dt = time.time() - t0
        if dt >= 1.0:
            set_cam(cid, fps=n / dt)
            n = 0
            t0 = time.time()
        # Qisqa sleep — yangi kadrni tezroq olish (lag ↓)
        time.sleep(0.02 if hq else 0.025)


def _extract_people_from_yolo(results, names, inv: float) -> list[fight_detect.PersonBox]:
    people: list[fight_detect.PersonBox] = []
    if not results or results[0].boxes is None:
        return people
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        raw = names.get(cls_id, str(cls_id))
        if raw != "person":
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        people.append(
            fight_detect.PersonBox(
                int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv), conf
            )
        )
    return people


def _extract_litter_from_yolo(results, names, inv: float) -> list[litter_detect.ObjectBox]:
    litter_names = set(getattr(config, "LITTER_CLASSES", ("bottle", "cup", "bowl", "wine glass")))
    out: list[litter_detect.ObjectBox] = []
    if not results or results[0].boxes is None:
        return out
    min_conf = float(getattr(config, "LITTER_MIN_CONF", 0.40))
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        raw = names.get(cls_id, str(cls_id))
        if raw not in litter_names:
            continue
        conf = float(box.conf[0])
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        # Litter classes skip the strict yolo_keep wide-aspect gate used for laptops
        out.append(
            litter_detect.ObjectBox(
                int(x1 * inv),
                int(y1 * inv),
                int(x2 * inv),
                int(y2 * inv),
                conf,
                raw,
            )
        )
    return out


def _pose_track(
    pose_models: dict[str, Any],
    bank: fight_tracks.PoseTrackBank,
    cam_id: str,
    ai_frame: np.ndarray,
    inv: float,
) -> list[fight_tracks.TrackedPose]:
    """Run YOLO-Pose + ByteTrack for one camera (dedicated model instance)."""
    model = pose_models.get(cam_id)
    if model is None:
        return []
    try:
        res = model.track(
            ai_frame,
            conf=max(0.40, float(getattr(config, "FIGHT_MIN_PERSON_CONF", 0.50))),
            imgsz=int(getattr(config, "FIGHT_POSE_IMGSZ", config.YOLO_IMGSZ)),
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            **pose_infer_kwargs(),
        )
    except Exception as e:
        # Fallback to predict without IDs
        try:
            res = model.predict(
                ai_frame,
                conf=max(0.40, float(getattr(config, "FIGHT_MIN_PERSON_CONF", 0.50))),
                imgsz=int(getattr(config, "FIGHT_POSE_IMGSZ", config.YOLO_IMGSZ)),
                verbose=False,
                **pose_infer_kwargs(),
            )
        except Exception:
            print(f"pose track error [{cam_id}]: {e}")
            return []
    if not res:
        return []
    return bank.update_from_ultralytics(
        cam_id,
        res[0],
        inv=inv,
        min_conf=float(getattr(config, "FIGHT_MIN_PERSON_CONF", 0.50)),
    )


def _run_fight(
    detector: fight_detect.FightDetector,
    pose_models: dict[str, Any],
    track_bank: fight_tracks.PoseTrackBank | None,
    action_model: fight_action.FightActionModel | None,
    cam: dict[str, Any],
    display: np.ndarray,
    ai_frame: np.ndarray,
    inv: float,
    people: list[fight_detect.PersonBox],
    items: list[tuple],
    pose_kpts: list[np.ndarray] | None = None,
    tracked_people: list[fight_detect.PersonBox] | None = None,
) -> list[np.ndarray] | None:
    if not getattr(config, "FIGHT_ENABLED", True):
        return pose_kpts

    backend = str(getattr(config, "FIGHT_BACKEND", "stgcn")).lower()
    fight_prob = None
    detail_extra = ""
    unique_people = fight_detect.nms_people(
        people,
        iou_thresh=float(getattr(config, "FIGHT_MAX_PAIR_IOU", 0.38)),
    )
    if tracked_people is None:
        tracked_people = unique_people
    else:
        tracked_people = fight_detect.nms_people(
            tracked_people,
            iou_thresh=float(getattr(config, "FIGHT_MAX_PAIR_IOU", 0.38)),
        )

    # Bitta odam (Muhammad kabinetda) — urush mumkin emas
    if len(unique_people) < 2:
        return pose_kpts

    if pose_kpts is None and track_bank is not None and pose_models:
        tracked = _pose_track(pose_models, track_bank, cam["id"], ai_frame, inv)
        if tracked:
            tracked_people = [t.box for t in tracked]
            pose_kpts = [t.kpt for t in tracked]

    if track_bank is not None and pose_models:
        if backend == "stgcn" and action_model is not None and track_bank is not None:
            st = track_bank.cam(cam["id"])
            pairs = st.close_pairs(
                max_norm_dist=float(getattr(config, "FIGHT_MAX_NORM_DIST", 0.95)),
                min_len=int(getattr(config, "FIGHT_PAIR_MIN_LEN", 16)),
            )
            best = None
            for tid_a, tid_b, _dist in pairs[:3]:
                seq_a = st.sequence(tid_a, length=int(getattr(config, "FIGHT_SEQ_LEN", 48)))
                seq_b = st.sequence(tid_b, length=int(getattr(config, "FIGHT_SEQ_LEN", 48)))
                if seq_a is None or seq_b is None:
                    continue
                scored = action_model.score_pair(seq_a, seq_b)
                if best is None or scored["fight_prob"] > best["fight_prob"]:
                    best = scored
                    best["tid_a"] = tid_a
                    best["tid_b"] = tid_b
                    ba = st.latest_box(tid_a)
                    bb = st.latest_box(tid_b)
                    if ba and bb:
                        tracked_people = [ba, bb]
            if best is not None:
                intr = float(best.get("intrusion", 0.0))
                min_intr = float(getattr(config, "FIGHT_MIN_INTRUSION", 0.22))
                # Tezkor qo‘l + 0 intrusion = urush emas
                if intr < min_intr:
                    fight_prob = 0.0
                else:
                    fight_prob = float(best["fight_prob"])
                detail_extra = (
                    f"tracks={best.get('tid_a')}/{best.get('tid_b')} "
                    f"geo={best.get('geo', 0):.2f} gcn={best.get('gcn', 0):.2f} "
                    f"intr={intr:.2f} wspd={best.get('wrist_speed', 0):.2f}"
                )

    st_out = detector.update(
        cam["id"],
        display,
        tracked_people,
        pose_kpts=pose_kpts,
        fight_prob=fight_prob,
    )
    if detail_extra:
        st_out.detail = f"{st_out.detail} | {detail_extra}"

    # ST-GCN: past fight_prob — overlay ham, event ham yo‘q
    if backend == "stgcn" and fight_prob is not None:
        action_thresh = float(getattr(config, "FIGHT_ACTION_THRESH", 0.78))
        if fight_prob < action_thresh:
            st_out.active = False
            st_out.just_confirmed = False
            if st_out.score < action_thresh:
                return pose_kpts

    if not st_out.active and not st_out.just_confirmed:
        return pose_kpts
    color = config.COLOR_FIGHT
    label_score = fight_prob if fight_prob is not None else st_out.score
    tag = "URUSH" if st_out.just_confirmed or st_out.active else "URUSH?"
    if st_out.pair:
        a, b = st_out.pair
        ux1, uy1, ux2, uy2 = (
            min(a.x1, b.x1),
            min(a.y1, b.y1),
            max(a.x2, b.x2),
            max(a.y2, b.y2),
        )
        items.append((ux1, uy1, ux2, uy2, color, f"{tag} {label_score:.2f}"))
    else:
        items.append((8, 40, display.shape[1] - 8, 90, color, f"{tag} {label_score:.2f}"))
    if st_out.just_confirmed:
        log_event(
            "fight",
            "⚠ URUSH ANIQLANDI",
            f"score={st_out.score:.2f} fight_prob={st_out.fight_prob:.2f} | {st_out.detail}",
            camera=cam["name"],
        )
        try:
            fight_dir = config.DATA_DIR / "fight"
            fight_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = fight_dir / f"{cam['id']}_{ts}.jpg"
            cv2.imwrite(str(path), display)
        except Exception as e:
            print(f"fight save error: {e}")
    return pose_kpts


def _run_litter(
    detector: litter_detect.LitterDetector,
    cam: dict[str, Any],
    display: np.ndarray,
    people: list[fight_detect.PersonBox],
    litter_objs: list[litter_detect.ObjectBox],
    items: list[tuple],
    pose_kpts: list[np.ndarray] | None = None,
) -> None:
    if not getattr(config, "LITTER_ENABLED", True):
        return
    st = detector.update(
        cam["id"],
        people,
        litter_objs,
        frame_h=display.shape[0],
        frame_w=display.shape[1],
        pose_kpts=pose_kpts,
    )
    if not st.active and st.score < 0.50:
        return
    color = getattr(config, "COLOR_LITTER", (0, 165, 255))
    mode = st.mode or ""
    tag_score = "AXLAT-YER" if mode == "static" else "AXLAT"
    if st.obj is not None:
        o = st.obj
        label_uz = yolo_label(o.name)
        items.append((o.x1, o.y1, o.x2, o.y2, color, f"{tag_score} {st.score:.2f}"))
        if st.person is not None:
            p = st.person
            items.append((p.x1, p.y1, p.x2, p.y2, color, f"axlat:{label_uz}"))
    else:
        items.append((8, 50, display.shape[1] - 8, 100, color, f"{tag_score}? {st.score:.2f}"))
    if st.just_confirmed:
        detail = st.detail or ""
        if st.obj is not None:
            detail = f"{yolo_label(st.obj.name)} | {detail}".strip(" |")
        evt_label = "⚠ AXLAT (YERDA)" if mode == "static" else "⚠ AXLAT TASHLANDI"
        log_event(
            "litter",
            evt_label,
            detail,
            camera=cam["name"],
        )
        try:
            litter_dir = config.DATA_DIR / "litter"
            litter_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = litter_dir / f"{cam['id']}_{ts}.jpg"
            cv2.imwrite(str(path), display)
        except Exception as e:
            print(f"litter save error: {e}")


_face_infer_lock = threading.Lock()
_fight_infer_lock = threading.Lock()
_yolo_infer_lock = threading.Lock()


def ai_loop(
    grabbers: dict[str, Any],
    yolo,
    class_ids,
    face_app,
    pose_models: dict[str, Any] | None = None,
    fight_detector=None,
    litter_detector=None,
    track_bank: fight_tracks.PoseTrackBank | None = None,
    action_model: fight_action.FightActionModel | None = None,
    face_memory: face_bank.FaceBank | None = None,
    antispoof: face_antispoof.AntiSpoofEnsemble | None = None,
    liveness: face_liveness.MotionLiveness | None = None,
) -> None:
    """Full AI on all cameras (batch YOLO + parallel post)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pose_models = pose_models or {}
    grid_idx = 0
    person_only = None
    if class_ids is None:
        name_to_id = {v: k for k, v in yolo.names.items()}
        person_only = [name_to_id["person"]] if "person" in name_to_id else [0]
        litter_ids = [name_to_id[n] for n in getattr(config, "LITTER_CLASSES", ()) if n in name_to_id]
        scan_classes = sorted(set(person_only + litter_ids)) if litter_ids else person_only
    else:
        person_only = class_ids
        scan_classes = class_ids

    full_all = bool(getattr(config, "AI_FULL_ALL_CAMS", True))
    batch_yolo = bool(getattr(config, "AI_BATCH_YOLO", True))
    post_workers = max(1, int(getattr(config, "AI_POST_WORKERS", 4)))
    use_pipeline = bool(getattr(config, "AI_PIPELINE", True))
    focus_refine = bool(getattr(config, "AI_FOCUS_REFINE", True))
    focus_imgsz = int(getattr(config, "AI_FOCUS_IMGSZ", 1280))
    bg_every_n = max(1, int(getattr(config, "AI_BG_EVERY_N", 3)))
    post_pool = ThreadPoolExecutor(max_workers=post_workers)
    pending_futs: list[Any] = []
    loop_tick = 0
    print(
        f"AI loop: full_all={full_all} batch_yolo={batch_yolo} "
        f"post_workers={post_workers} pipeline={use_pipeline} "
        f"bg_every_n={bg_every_n} focus_refine={focus_refine}@{focus_imgsz} "
        f"ai_width={config.AI_WIDTH} imgsz={config.YOLO_IMGSZ}"
    )

    def _prepare_cam(cam: dict[str, Any]) -> dict[str, Any] | None:
        grabber = get_grabber(cam["id"])
        if grabber is None:
            return None
        frame, _cap_ts = grabber.get()
        if frame is None:
            return None
        hq = is_hq_view(cam["id"])
        max_w = int(
            getattr(config, "DISPLAY_WIDTH_MAIN", 1440)
            if hq
            else getattr(config, "DISPLAY_WIDTH_SUB", config.DISPLAY_WIDTH)
        )
        display = _resize_display(frame, max_w=max_w)
        dh, dw = display.shape[:2]
        if dw > config.AI_WIDTH:
            ai_scale = config.AI_WIDTH / float(dw)
            ai_frame = cv2.resize(
                display, (config.AI_WIDTH, int(dh * ai_scale)), interpolation=cv2.INTER_AREA
            )
        else:
            ai_frame = display
            ai_scale = 1.0
        inv = 1.0 / ai_scale if ai_scale else 1.0
        return {
            "cam": cam,
            "display": display,
            "ai_frame": ai_frame,
            "inv": inv,
            "full_ai": True if full_all else False,
        }

    def _post_one(job: dict[str, Any], result) -> None:
        cam = job["cam"]
        display = job["display"]
        ai_frame = job["ai_frame"]
        inv = job["inv"]
        full_ai = job["full_ai"]
        items: list[tuple] = []
        try:
            people = _extract_people_from_yolo([result] if result is not None else [], yolo.names, inv)
            litter_objs = _extract_litter_from_yolo([result] if result is not None else [], yolo.names, inv)

            if full_ai and result is not None and result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    raw = yolo.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    if not yolo_keep(raw, conf, (x1, y1, x2, y2)):
                        continue
                    label = yolo_label(raw)
                    x1, y1, x2, y2 = int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv)
                    color = config.COLOR_PERSON_BOX if raw == "person" else config.COLOR_OBJECT
                    items.append((x1, y1, x2, y2, color, f"{label} {conf:.2f}"))

            if full_ai and face_app is not None and people:
                with _face_infer_lock:
                    hits = _detect_faces_on_people(
                        face_app,
                        display,
                        people,
                        cam_id=cam["id"],
                        track_bank=track_bank,
                        bank=face_memory,
                    )
                    face_boxes = _apply_face_hits(
                        hits,
                        display,
                        cam,
                        items=items,
                        draw_direct=False,
                        bank=face_memory,
                        antispoof=antispoof,
                        liveness=liveness,
                        people=people,
                    )
                    _apply_room_presence(people, face_boxes, cam, items, face_memory, liveness)
                    if face_memory is not None:
                        face_memory.prune()
                    if liveness is not None:
                        liveness.prune()
            elif full_ai and face_memory is not None:
                _apply_room_presence(people or [], [], cam, items, face_memory, liveness)

            pose_kpts: list[np.ndarray] | None = None
            need_pose = (
                (fight_detector is not None and getattr(config, "FIGHT_ENABLED", True))
                or (
                    litter_detector is not None
                    and getattr(config, "LITTER_ENABLED", True)
                    and getattr(config, "LITTER_USE_POSE", True)
                )
            )
            if need_pose and pose_models and track_bank is not None:
                with _fight_infer_lock:
                    tracked = _pose_track(pose_models, track_bank, cam["id"], ai_frame, inv)
                    if tracked:
                        pose_kpts = [t.kpt for t in tracked]

            if fight_detector is not None:
                with _fight_infer_lock:
                    pose_kpts = _run_fight(
                        fight_detector,
                        pose_models,
                        track_bank,
                        action_model,
                        cam,
                        display,
                        ai_frame,
                        inv,
                        people,
                        items,
                        pose_kpts=pose_kpts,
                    )

            if litter_detector is not None:
                _run_litter(
                    litter_detector,
                    cam,
                    display,
                    people,
                    litter_objs,
                    items,
                    pose_kpts=pose_kpts,
                )

            with _overlays_lock:
                if full_ai:
                    _overlays[cam["id"]] = items
                else:
                    prev = list(_overlays.get(cam["id"], []))
                    alert_items = [
                        it for it in items
                        if "URUSH" in str(it[5]) or "AXLAT" in str(it[5]) or str(it[5]).startswith("axlat:")
                    ]
                    prev = [
                        it for it in prev
                        if "URUSH" not in str(it[5])
                        and "AXLAT" not in str(it[5])
                        and not str(it[5]).startswith("axlat:")
                    ]
                    _overlays[cam["id"]] = prev + alert_items
        except Exception as e:
            set_cam(cam["id"], error=str(e))

    while True:
        if _reload_faces.is_set():
            _reload_faces.clear()
            gallery = load_face_db()
            apply_face_db(gallery)
            print(f"Face DB reloaded: {len(names)}")

        with _focus_lock:
            mode = _focus_mode
            focus = _focus_cam_id

        loop_tick += 1
        run_background = (loop_tick % bg_every_n) == 0

        if full_all:
            cams = list(config.CAMERAS)
        elif mode == "grid":
            primary = config.CAMERAS[grid_idx % len(config.CAMERAS)]
            grid_idx += 1
            cams = [primary]
        else:
            cams = [next((c for c in config.CAMERAS if c["id"] == focus), config.CAMERAS[0])]

        jobs: list[dict[str, Any]] = []
        for cam in cams:
            is_focus_cam = cam["id"] == focus
            # Background skip: focus always; others only every Nth tick (efir silliqligi)
            if full_all and not is_focus_cam and not run_background:
                continue
            prepared = _prepare_cam(cam)
            if prepared is None:
                continue
            prepared["full_ai"] = True
            jobs.append(prepared)

        if not jobs:
            time.sleep(0.03)
            continue

        results_list: list[Any]
        try:
            if batch_yolo and len(jobs) >= 1:
                batch = [j["ai_frame"] for j in jobs]
                with _yolo_infer_lock:
                    pred = yolo.predict(
                        batch if len(batch) > 1 else batch[0],
                        conf=config.YOLO_CONF,
                        classes=class_ids if class_ids else None,
                        imgsz=config.YOLO_IMGSZ,
                        verbose=False,
                        **yolo_infer_kwargs(),
                    )
                results_list = list(pred) if isinstance(pred, (list, tuple)) else [pred]
                if len(batch) == 1 and len(results_list) >= 1:
                    results_list = [results_list[0]]
                while len(results_list) < len(jobs):
                    results_list.append(None)
                results_list = results_list[: len(jobs)]
            else:
                results_list = []
                for j in jobs:
                    use_classes = class_ids if j["full_ai"] else scan_classes
                    with _yolo_infer_lock:
                        one = yolo.predict(
                            j["ai_frame"],
                            conf=config.YOLO_CONF if j["full_ai"] else max(0.30, min(config.YOLO_CONF, 0.40)),
                            classes=use_classes if use_classes else None,
                            imgsz=config.YOLO_IMGSZ,
                            verbose=False,
                            **yolo_infer_kwargs(),
                        )
                    results_list.append(one[0] if one else None)

            # Focus camera: second higher-res YOLO pass (more GPU + accuracy)
            if focus_refine and mode == "single" and focus_imgsz > int(config.YOLO_IMGSZ):
                for i, j in enumerate(jobs):
                    if j["cam"]["id"] != focus:
                        continue
                    with _yolo_infer_lock:
                        one = yolo.predict(
                            j["ai_frame"],
                            conf=config.YOLO_CONF,
                            classes=class_ids if class_ids else None,
                            imgsz=focus_imgsz,
                            verbose=False,
                            **yolo_infer_kwargs(),
                        )
                    if one:
                        results_list[i] = one[0]
                    break
        except Exception as e:
            print(f"YOLO batch error: {e}")
            time.sleep(0.05)
            continue

        # Pipeline: previous face/pose posts run in parallel with this YOLO above.
        # Drain them before queueing the next wave (bound latency / backlog).
        if use_pipeline and pending_futs:
            for f in as_completed(pending_futs):
                exc = f.exception()
                if exc:
                    print(f"AI post error: {exc}")
            pending_futs = []

        if post_workers <= 1 or len(jobs) == 1:
            for j, res in zip(jobs, results_list):
                _post_one(j, res)
            pending_futs = []
        else:
            futs = [post_pool.submit(_post_one, j, res) for j, res in zip(jobs, results_list)]
            if use_pipeline:
                pending_futs = futs
            else:
                for f in as_completed(futs):
                    exc = f.exception()
                    if exc:
                        print(f"AI post error: {exc}")
                pending_futs = []

        time.sleep(0.001)


def inference_loop() -> None:
    _init_cam_state()
    import torch
    print("Loading YOLO...")
    from ultralytics import YOLO

    print(
        f"CUDA: avail={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
        f"yolo_kw={yolo_infer_kwargs()}"
    )

    yolo = YOLO(config.YOLO_MODEL)
    # Warmup on target device
    try:
        _ = yolo.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            verbose=False,
            **yolo_infer_kwargs(),
        )
        print(f"YOLO warmup OK: {config.YOLO_MODEL}")
    except Exception as e:
        print(f"YOLO warmup warn: {e}")
    name_to_id = {v: k for k, v in yolo.names.items()}
    if config.YOLO_CLASSES:
        class_ids = [name_to_id[n] for n in config.YOLO_CLASSES if n in name_to_id]
    else:
        class_ids = None  # all 80 COCO classes
    print(f"YOLO classes: {'ALL' if class_ids is None else len(class_ids)}")

    pose_models: dict[str, Any] = {}
    fight_detector = None
    track_bank = None
    action_model = None
    if getattr(config, "FIGHT_ENABLED", True):
        print("Loading YOLO-Pose + ByteTrack (urush aniqlash)...")
        pose_name = getattr(config, "FIGHT_POSE_MODEL", "yolov8n-pose.pt")
        try:
            # Per-camera model so ByteTrack persist state does not mix streams
            for cam in config.CAMERAS:
                pose_models[cam["id"]] = YOLO(pose_name)
            print(f"Fight pose models: {len(pose_models)} cams OK")
        except Exception as e:
            print(f"Fight pose model failed (heuristic fallback): {e}")
            pose_models = {}
        fight_detector = fight_detect.FightDetector(config)
        track_bank = fight_tracks.PoseTrackBank(
            seq_len=int(getattr(config, "FIGHT_SEQ_LEN", 48))
        )
        if str(getattr(config, "FIGHT_BACKEND", "stgcn")).lower() == "stgcn":
            try:
                action_model = fight_action.FightActionModel(config)
                print(
                    f"Fight ActionGCN ON | device={action_model.device} "
                    f"thresh>={config.FIGHT_ACTION_THRESH}"
                )
            except Exception as e:
                print(f"Fight ActionGCN failed, heuristic only: {e}")
                action_model = None
        print(
            f"Fight detector ON | backend={getattr(config, 'FIGHT_BACKEND', 'stgcn')} "
            f"confirm={config.FIGHT_CONFIRM_SEC}s score>={config.FIGHT_SCORE_THRESH}"
        )

    litter_detector = None
    if getattr(config, "LITTER_ENABLED", True):
        litter_detector = litter_detect.LitterDetector(config)
        print(
            f"Litter detector ON | classes={list(getattr(config, 'LITTER_CLASSES', ()))} "
            f"confirm={config.LITTER_CONFIRM_SEC}s floor={getattr(config, 'LITTER_REQUIRE_FLOOR', True)}"
            f"@{getattr(config, 'LITTER_FLOOR_RATIO', 0.68)} pose={getattr(config, 'LITTER_USE_POSE', True)} "
            f"static={getattr(config, 'LITTER_STATIC_ENABLED', True)}"
            f"@{getattr(config, 'LITTER_STATIC_HOLD_SEC', 3.5)}s"
        )

    print("Loading InsightFace...")
    face_app = None
    face_memory = face_bank.FaceBank(config)
    antispoof = face_antispoof.AntiSpoofEnsemble(config)
    liveness = face_liveness.MotionLiveness(config)
    try:
        from insightface.app import FaceAnalysis
        providers: list[str] = []
        if getattr(config, "FACE_USE_GPU", True):
            try:
                import onnxruntime as ort
                avail = set(ort.get_available_providers())
                if "CUDAExecutionProvider" in avail:
                    providers.append("CUDAExecutionProvider")
            except Exception:
                pass
        providers.append("CPUExecutionProvider")
        face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
        face_app.prepare(
            ctx_id=ctx_id,
            det_size=config.FACE_DET_SIZE,
            det_thresh=float(getattr(config, "FACE_DET_THRESH", 0.40)),
        )
        print(
            f"Face: providers={providers} det_size={config.FACE_DET_SIZE} "
            f"thresh={config.FACE_DET_THRESH} match>={config.FACE_MATCH_THRESHOLD} "
            f"soft>={getattr(config, 'FACE_SOFT_THRESHOLD', 0.26)} "
            f"temporal={getattr(config, 'FACE_TEMPORAL_SEC', 2.0)}s"
        )
        if antispoof.enabled:
            print(
                f"Anti-spoof ON | models={len(antispoof.models)} "
                f"real>={getattr(config, 'FACE_ANTISPOOF_REAL_THRESH', 0.55)}"
            )
        if liveness.enabled:
            print(
                f"Liveness ON | frames>={getattr(config, 'FACE_LIVENESS_MIN_FRAMES', 4)} "
                f"motion>={getattr(config, 'FACE_LIVENESS_MOTION_THRESH', 0.08)} "
                f"bypass>={getattr(config, 'FACE_ANTISPOOF_MOTION_BYPASS', 0.72)}"
            )
        if getattr(config, "FACE_ROOM_SESSION_ENABLED", True):
            print(
                f"Room session ON | empty={getattr(config, 'FACE_ROOM_EMPTY_SEC', 180)}s "
                f"max_occ={getattr(config, 'FACE_ROOM_MAX_OCCUPANTS', 8)}"
            )
    except Exception as e:
        print(f"InsightFace failed: {e}")
        face_app = None

    _face_app_holder["app"] = face_app
    gallery = load_face_db()
    apply_face_db(gallery)
    print(
        f"Face DB: {gallery.count} identities ({gallery.proto_count} protos) | "
        f"Cameras: {len(config.CAMERAS)}"
    )

    # Start all on sub, then promote focused cam to main
    with _grabbers_lock:
        for cam in config.CAMERAS:
            _grabbers[cam["id"]] = make_grabber(cam, "sub")
            _grabber_quality[cam["id"]] = "sub"
    sync_grabber_streams()
    time.sleep(1.5)

    for cam in config.CAMERAS:
        threading.Thread(
            target=preview_loop, args=(cam,),
            name=f"preview-{cam['id']}", daemon=True,
        ).start()

    # AI in this thread (round-robin)
    ai_loop(
        _grabbers,
        yolo,
        class_ids,
        face_app,
        pose_models=pose_models,
        fight_detector=fight_detector,
        litter_detector=litter_detector,
        track_bank=track_bank,
        action_model=action_model,
        face_memory=face_memory,
        antispoof=antispoof,
        liveness=liveness,
    )


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.secret_key = os.environ.get("AUTH_SECRET") or getattr(
        config, "AUTH_SECRET", "CHANGE_ME_SESSION_SECRET_KEY"
    )
    auth_on = bool(getattr(config, "AUTH_ENABLED", True))
    auth_user = os.environ.get("AUTH_USERNAME") or getattr(config, "AUTH_USERNAME", "admin")
    auth_pass = os.environ.get("AUTH_PASSWORD") or getattr(config, "AUTH_PASSWORD", "CHANGE_ME_PASSWORD")

    def _auth_ok() -> bool:
        if not auth_on:
            return True
        return bool(session.get("auth"))

    def _check_creds(username: str, password: str) -> bool:
        u_ok = hmac.compare_digest(str(username or ""), str(auth_user))
        p_ok = hmac.compare_digest(str(password or ""), str(auth_pass))
        return u_ok and p_ok

    @app.before_request
    def _require_auth():
        if not auth_on:
            return None
        path = request.path or "/"
        if path in ("/login", "/logout") or path.startswith("/login"):
            return None
        if _auth_ok():
            return None
        # API / streams → 401 JSON; pages → login redirect
        if path.startswith("/ai/") or path.startswith("/vendor/"):
            return jsonify({"ok": False, "error": "login kerak"}), 401
        nxt = path if path.startswith("/") else "/ai.html"
        return redirect(f"/login?next={nxt}")

    @app.get("/login")
    def login_page():
        if _auth_ok():
            return redirect("/ai.html")
        return send_from_directory(config.WWW_DIR, "login.html")

    @app.post("/login")
    def login_post():
        data = request.get_json(silent=True) or {}
        username = data.get("username") or request.form.get("username", "")
        password = data.get("password") or request.form.get("password", "")
        if not _check_creds(username, password):
            return jsonify({"ok": False, "error": "Login yoki parol noto‘g‘ri"}), 401
        session.clear()
        session["auth"] = True
        session["user"] = auth_user
        session.permanent = True
        return jsonify({"ok": True})

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect("/login")

    @app.get("/")
    def root():
        return send_from_directory(config.WWW_DIR, "ai.html")

    @app.get("/ai.html")
    def ai_page():
        return send_from_directory(config.WWW_DIR, "ai.html")

    @app.get("/vendor/<path:filename>")
    def vendor(filename: str):
        return send_from_directory(config.WWW_DIR / "vendor", filename)

    @app.get("/ai/cameras")
    def cameras():
        with _focus_lock:
            focus = _focus_cam_id
            fmode = _focus_mode
        return jsonify({
            "ok": True,
            "cameras": list_cam_status(),
            "faces_loaded": get_face_db().count,
            "focus": focus,
            "mode": fmode,
        })

    @app.post("/ai/focus")
    def set_focus():
        global _focus_cam_id, _focus_mode
        data = request.get_json(silent=True) or {}
        mode = data.get("mode") or "single"
        cam_id = data.get("camera") or config.CAMERAS[0]["id"]
        if mode not in ("single", "grid"):
            mode = "single"
        ids = {c["id"] for c in config.CAMERAS}
        if cam_id not in ids:
            cam_id = config.CAMERAS[0]["id"]
        with _focus_lock:
            _focus_mode = mode
            _focus_cam_id = cam_id
        threading.Thread(target=sync_grabber_streams, name="stream-sync", daemon=True).start()
        return jsonify({"ok": True, "mode": mode, "camera": cam_id})

    @app.get("/ai/status")
    def status():
        cams = list_cam_status()
        running_any = any(c["running"] for c in cams)
        return jsonify({
            "running": running_any,
            "faces_loaded": get_face_db().count,
            "cameras": cams,
            "error": next((c["error"] for c in cams if c["error"]), ""),
            "fps": float(np.mean([c["fps"] for c in cams]) if cams else 0),
            "lag_ms": int(np.mean([c["lag_ms"] for c in cams]) if cams else 0),
        })

    @app.get("/ai/events")
    def events():
        with _events_lock:
            items = list(_events)
        return jsonify(items)

    @app.get("/ai/stream")
    @app.get("/ai/stream/<cam_id>")
    def stream(cam_id: str | None = None):
        if not cam_id:
            cam_id = config.CAMERAS[0]["id"]

        @stream_with_context
        def gen():
            boundary = b"--frame"
            last = None
            try:
                while True:
                    # Client uzilsa waitress/werkzeug GeneratorExit beradi
                    data = get_jpeg(cam_id)
                    if data is not None and data is not last:
                        last = data
                        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                    time.sleep(0.08)
            except GeneratorExit:
                return
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

        resp = Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    @app.get("/ai/snapshot.jpg")
    @app.get("/ai/snapshot/<cam_id>.jpg")
    def snapshot(cam_id: str | None = None):
        if not cam_id:
            cam_id = config.CAMERAS[0]["id"]
        data = get_jpeg(cam_id)
        if not data:
            return Response(b"", status=503)
        return Response(data, mimetype="image/jpeg")

    @app.get("/ai/people")
    def people_list():
        return jsonify({"ok": True, "people": enroll_faces.list_people()})

    @app.post("/ai/people")
    def people_add():
        fio = enroll_faces.sanitize_fio(request.form.get("fio", ""))
        if not fio:
            return jsonify({"ok": False, "error": "FIO kiriting"}), 400
        files = request.files.getlist("photos")
        if not files:
            return jsonify({"ok": False, "error": "Kamida 1 ta yuz rasmi kerak"}), 400
        person_dir = config.FACES_DIR / fio
        person_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for i, f in enumerate(files):
            if not f or not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                ext = ".jpg"
            safe = secure_filename(Path(f.filename).stem) or f"photo_{i+1}"
            dest = person_dir / f"{safe}_{int(time.time())}_{i}{ext}"
            f.save(str(dest))
            img = cv2.imread(str(dest))
            if img is None:
                dest.unlink(missing_ok=True)
                continue
            app_face = _face_app_holder.get("app")
            if app_face is not None and not app_face.get(img):
                dest.unlink(missing_ok=True)
                continue
            saved += 1
        if saved == 0:
            if person_dir.exists() and not any(person_dir.iterdir()):
                person_dir.rmdir()
            return jsonify({"ok": False, "error": "Rasmlarda yuz topilmadi"}), 400
        try:
            result = rebuild_and_reload()
        except Exception as e:
            return jsonify({"ok": False, "error": f"O‘rgatish xatosi: {e}"}), 500
        return jsonify({"ok": True, "fio": fio, "saved_photos": saved, "faces_loaded": result.get("count", 0)})

    @app.delete("/ai/people/<path:fio>")
    def people_delete(fio: str):
        name = enroll_faces.sanitize_fio(fio)
        person_dir = config.FACES_DIR / name
        if not person_dir.is_dir():
            return jsonify({"ok": False, "error": "Topilmadi"}), 404
        shutil.rmtree(person_dir, ignore_errors=True)
        result = rebuild_and_reload()
        return jsonify({"ok": True, "faces_loaded": result.get("count", 0)})

    @app.get("/ai/unknowns")
    def unknowns_list():
        return jsonify({"ok": True, "people": unknown_store.list_unknowns()})

    @app.get("/ai/unknowns/<pid>/<filename>")
    def unknowns_photo(pid: str, filename: str):
        pid = Path(pid).name
        filename = Path(filename).name
        folder = config.UNKNOWN_DIR / pid
        if not (folder / filename).is_file():
            return Response(b"", status=404)
        return send_from_directory(folder, filename)

    @app.delete("/ai/unknowns/<pid>")
    def unknowns_delete(pid: str):
        if not unknown_store.delete_unknown(pid):
            return jsonify({"ok": False, "error": "Topilmadi"}), 404
        return jsonify({"ok": True})

    @app.post("/ai/unknowns/<pid>/promote")
    def unknowns_promote(pid: str):
        data = request.get_json(silent=True) or {}
        fio = data.get("fio") or request.form.get("fio", "")
        result = unknown_store.promote_to_known(pid, fio)
        if not result.get("ok"):
            return jsonify(result), 400
        enrolled = rebuild_and_reload()
        unknown_store.delete_unknown(pid)
        return jsonify({"ok": True, "fio": result["fio"], "copied": result["copied"], "faces_loaded": enrolled.get("count", 0)})

    if auth_on:
        print(f"Auth ON | user={auth_user} | /login")
    return app


def main() -> int:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.FACES_DIR.mkdir(parents=True, exist_ok=True)
    config.UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    config.WWW_DIR.mkdir(parents=True, exist_ok=True)
    if not config.FACE_DB_PATH.is_file():
        np.savez_compressed(
            config.FACE_DB_PATH,
            names=np.array([], dtype=object),
            embeddings=np.zeros((0, 512), dtype=np.float32),
        )

    removed = unknown_store.purge_expired(force=True)
    ttl_h = float(getattr(config, "UNKNOWN_TTL_SEC", 7200)) / 3600.0
    print(f"Unknown TTL: {ttl_h:g} soat | eski begonalar o‘chirildi: {removed}")

    def _unknown_purge_loop() -> None:
        while True:
            time.sleep(float(getattr(config, "UNKNOWN_PURGE_INTERVAL_SEC", 60.0)))
            try:
                n = unknown_store.purge_expired(force=True)
                if n:
                    print(f"Unknown TTL: {n} ta begona o‘chirildi")
            except Exception as e:
                print(f"Unknown purge xato: {e}")

    threading.Thread(target=_unknown_purge_loop, name="unknown-purge", daemon=True).start()

    t = threading.Thread(target=inference_loop, name="inference", daemon=True)
    t.start()
    app = create_app()
    print(f"AI UI: http://127.0.0.1:{config.PORT}/ai.html")
    try:
        from waitress import serve
        print("Serving with waitress (multi-thread)")
        serve(app, host=config.HOST, port=config.PORT, threads=12, channel_timeout=30)
    except Exception as e:
        print(f"waitress fallback to flask: {e}")
        app.run(host=config.HOST, port=config.PORT, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
