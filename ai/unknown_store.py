"""Store and cluster unknown (stranger) faces detected by the camera."""
from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config

_lock = threading.Lock()
_INDEX = "index.json"
_EMB = "embedding.npy"
_last_purge_at = 0.0


def _ensure_dirs() -> None:
    config.UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)


def _person_dir(pid: str) -> Path:
    return config.UNKNOWN_DIR / pid


def _load_index(pid: str) -> dict[str, Any]:
    path = _person_dir(pid) / _INDEX
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_index(pid: str, data: dict[str, Any]) -> None:
    path = _person_dir(pid) / _INDEX
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_id() -> str:
    existing = []
    if config.UNKNOWN_DIR.is_dir():
        for p in config.UNKNOWN_DIR.iterdir():
            if p.is_dir() and p.name.startswith("begona_"):
                try:
                    existing.append(int(p.name.split("_", 1)[1]))
                except ValueError:
                    pass
    n = max(existing) + 1 if existing else 1
    return f"begona_{n:03d}"


def _crop_face(frame: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.25) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _parse_seen(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        return ts if ts > 0 else None
    text = str(value).strip()
    if not text:
        return None
    for fmt, n in (
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y-%m-%dT%H:%M:%S", 19),
        ("%Y%m%d_%H%M%S", 15),
    ):
        try:
            return datetime.strptime(text[:n], fmt).timestamp()
        except ValueError:
            continue
    return None


def _folder_mtime(person: Path) -> float:
    latest = 0.0
    try:
        latest = max(latest, person.stat().st_mtime)
    except OSError:
        pass
    try:
        for p in person.iterdir():
            try:
                latest = max(latest, p.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return latest


def person_age_epoch(pid: str, idx: dict[str, Any] | None = None) -> float:
    """Most recent activity timestamp for TTL (last_seen preferred)."""
    d = _person_dir(pid)
    data = idx if idx is not None else _load_index(pid)
    for key in ("last_seen", "first_seen"):
        ts = _parse_seen(data.get(key))
        if ts is not None:
            return ts
    ts = _parse_seen(data.get("last_save_epoch"))
    if ts is not None:
        return ts
    return _folder_mtime(d)


def purge_expired(force: bool = False) -> int:
    """
    Delete unknown (begona) folders older than UNKNOWN_TTL_SEC.
    Returns number of deleted people.
    """
    global _last_purge_at
    ttl = float(getattr(config, "UNKNOWN_TTL_SEC", 2 * 3600))
    interval = float(getattr(config, "UNKNOWN_PURGE_INTERVAL_SEC", 60.0))
    now = time.time()
    if not force and (now - _last_purge_at) < interval:
        return 0

    _ensure_dirs()
    deleted = 0
    with _lock:
        _last_purge_at = now
        if not config.UNKNOWN_DIR.is_dir():
            return 0
        for person in list(config.UNKNOWN_DIR.iterdir()):
            if not person.is_dir():
                continue
            idx = _load_index(person.name)
            age_ts = person_age_epoch(person.name, idx)
            if age_ts <= 0 or (now - age_ts) < ttl:
                continue
            shutil.rmtree(person, ignore_errors=True)
            deleted += 1
    return deleted


def list_unknowns() -> list[dict[str, Any]]:
    _ensure_dirs()
    purge_expired()
    out: list[dict[str, Any]] = []
    for person in sorted(config.UNKNOWN_DIR.iterdir(), reverse=True):
        if not person.is_dir():
            continue
        idx = _load_index(person.name)
        photos = sorted(
            [p.name for p in person.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
            reverse=True,
        )
        if not photos and not idx:
            continue
        out.append({
            "id": person.name,
            "label": idx.get("label") or person.name.replace("_", " ").title(),
            "first_seen": idx.get("first_seen", ""),
            "last_seen": idx.get("last_seen", ""),
            "count": idx.get("count", len(photos)),
            "photos": photos[:12],
            "thumb": photos[0] if photos else "",
            "cameras": idx.get("cameras") or [],
        })
    # newest last_seen first
    out.sort(key=lambda x: x.get("last_seen") or "", reverse=True)
    return out


def delete_unknown(pid: str) -> bool:
    pid = Path(pid).name
    d = _person_dir(pid)
    if not d.is_dir():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def save_unknown(
    display_frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    emb: np.ndarray,
    camera_id: str = "",
    camera_name: str = "",
) -> dict[str, Any] | None:
    """
    Cluster by embedding and save a face crop.
    Returns {id, label, is_new, photo} or None if skipped (cooldown / fail).
    """
    purge_expired()
    _ensure_dirs()
    emb = emb.astype(np.float32).reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    now = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _lock:
        best_id = None
        best_score = -1.0
        for person in config.UNKNOWN_DIR.iterdir():
            if not person.is_dir():
                continue
            emb_path = person / _EMB
            if not emb_path.is_file():
                continue
            try:
                other = np.load(str(emb_path)).astype(np.float32).reshape(-1)
                other = other / (np.linalg.norm(other) + 1e-9)
                score = float(np.dot(emb, other))
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_id = person.name

        is_new = False
        if best_id is None or best_score < config.UNKNOWN_MATCH_THRESHOLD:
            best_id = _next_id()
            is_new = True
            d = _person_dir(best_id)
            d.mkdir(parents=True, exist_ok=True)
            np.save(str(d / _EMB), emb)
            idx = {
                "id": best_id,
                "label": f"Begona {best_id.split('_')[-1]}",
                "first_seen": now_str,
                "last_seen": now_str,
                "count": 0,
                "last_save_epoch": 0.0,
                "cameras": [],
            }
        else:
            d = _person_dir(best_id)
            idx = _load_index(best_id) or {
                "id": best_id,
                "label": f"Begona {best_id.split('_')[-1]}",
                "first_seen": now_str,
                "count": 0,
                "last_save_epoch": 0.0,
                "cameras": [],
            }
            # refresh embedding (running mean)
            emb_path = d / _EMB
            if emb_path.is_file():
                try:
                    old = np.load(str(emb_path)).astype(np.float32).reshape(-1)
                    mean = old + emb
                    mean = mean / (np.linalg.norm(mean) + 1e-9)
                    np.save(str(emb_path), mean.astype(np.float32))
                except Exception:
                    np.save(str(emb_path), emb)

        last_save = float(idx.get("last_save_epoch") or 0.0)
        photos = list(d.glob("*.jpg"))
        # First photo always; later on interval; hard cap
        should_save = is_new or (now - last_save >= config.UNKNOWN_SAVE_INTERVAL_SEC)
        if len(photos) >= config.UNKNOWN_MAX_PHOTOS_PER_PERSON:
            should_save = False

        photo_name = ""
        if should_save:
            crop = _crop_face(display_frame, bbox)
            if crop is None or crop.size == 0:
                return None
            photo_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{int(now * 1000) % 1000:03d}.jpg"
            ok = cv2.imwrite(str(d / photo_name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return None
            idx["last_save_epoch"] = now
            idx["count"] = int(idx.get("count") or 0) + 1
        else:
            # still update last_seen without new file
            idx["last_seen"] = now_str
            _save_index(best_id, idx)
            return {
                "id": best_id,
                "label": idx.get("label", best_id),
                "is_new": False,
                "photo": "",
                "saved": False,
            }

        idx["last_seen"] = now_str
        if is_new or not idx.get("first_seen"):
            idx["first_seen"] = now_str
        cams = list(idx.get("cameras") or [])
        tag = camera_name or camera_id
        if tag and tag not in cams:
            cams.append(tag)
            idx["cameras"] = cams
        _save_index(best_id, idx)

        return {
            "id": best_id,
            "label": idx.get("label", best_id),
            "is_new": is_new,
            "photo": photo_name,
            "saved": True,
            "cameras": cams,
        }


def promote_to_known(pid: str, fio: str) -> dict[str, Any]:
    """Copy unknown photos into faces/<fio>/ for enrollment."""
    import enroll_faces

    fio = enroll_faces.sanitize_fio(fio)
    pid = Path(pid).name
    src = _person_dir(pid)
    if not fio:
        return {"ok": False, "error": "FIO kiriting"}
    if not src.is_dir():
        return {"ok": False, "error": "Begona topilmadi"}

    dest = config.FACES_DIR / fio
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in src.iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        shutil.copy2(p, dest / p.name)
        copied += 1
    if copied == 0:
        return {"ok": False, "error": "Rasm yo‘q"}
    return {"ok": True, "fio": fio, "copied": copied, "unknown_id": pid}
