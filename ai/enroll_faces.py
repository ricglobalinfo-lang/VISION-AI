"""Enroll known faces from faces/<FIO>/*.jpg into data/face_db.npz (multi-prototype gallery)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from face_bank import face_sample_quality  # noqa: E402

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_fio(fio: str) -> str:
    name = " ".join((fio or "").strip().split())
    name = _INVALID.sub("", name).strip(" .")
    return name[:120]


def build_app():
    """Enrollment uchun: katta det_size ba’zi portretlarda yuzni yo‘qotadi — 640 ishonchliroq."""
    import os
    from pathlib import Path

    from insightface.app import FaceAnalysis

    try:
        import torch

        lib = Path(torch.__file__).resolve().parent / "lib"
        if lib.is_dir():
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(lib))
                except OSError:
                    pass
    except Exception:
        pass

    providers = ["CPUExecutionProvider"]
    ctx_id = -1
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in set(ort.get_available_providers()):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
    except Exception:
        pass

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    enroll_size = tuple(getattr(config, "FACE_ENROLL_DET_SIZE", (640, 640)))
    enroll_thresh = float(getattr(config, "FACE_ENROLL_DET_THRESH", 0.35))
    app.prepare(ctx_id=ctx_id, det_size=enroll_size, det_thresh=enroll_thresh)
    return app


def _read_image(img_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(img_path))
    if img is not None:
        return img
    try:
        raw = np.fromfile(str(img_path), dtype=np.uint8)
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _detect_faces(app, img: np.ndarray) -> list:
    faces = app.get(img) or []
    if faces:
        return faces
    try:
        app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.30)
        faces = app.get(img) or []
    except Exception:
        faces = []
    return faces or []


def _best_face(faces: list) -> Any | None:
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _face_bbox_int(face) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in face.bbox.tolist()]
    return x1, y1, x2, y2


def _collect_embeddings(app, img: np.ndarray, min_quality: float) -> list[tuple[np.ndarray, float]]:
    """Extract quality-filtered embeddings from image (+ horizontal flip variant)."""
    out: list[tuple[np.ndarray, float]] = []
    variants = [img, cv2.flip(img, 1)]
    seen: set[tuple[float, ...]] = set()

    for variant in variants:
        faces = _detect_faces(app, variant)
        face = _best_face(faces)
        if face is None:
            continue
        bbox = _face_bbox_int(face)
        q = face_sample_quality(face, bbox)
        if q < min_quality:
            continue
        emb = face.normed_embedding.astype(np.float32)
        key = tuple(np.round(emb[:8], 4).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append((emb, q))
    return out


def _pick_top_prototypes(
    items: list[tuple[np.ndarray, float]], max_protos: int
) -> list[np.ndarray]:
    ranked = sorted(items, key=lambda x: x[1], reverse=True)
    return [emb for emb, _q in ranked[:max_protos]]


def _find_duplicate_warnings(
    names: list[str],
    embeddings: np.ndarray,
    person_ids: np.ndarray,
    warn_thresh: float,
) -> list[str]:
    """Warn when two different people share very similar gallery prototypes."""
    warns: list[str] = []
    if len(names) < 2:
        return warns
    # One representative embedding per person (best proto)
    reps: list[np.ndarray] = []
    for pid in range(len(names)):
        mask = person_ids == pid
        if not np.any(mask):
            continue
        sub = embeddings[mask]
        reps.append(sub[0])
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            sim = float(reps[i] @ reps[j])
            if sim >= warn_thresh:
                warns.append(f"DIQQAT: '{names[i]}' va '{names[j]}' juda o‘xshash ({sim:.2f}) — bitta odam bo‘lishi mumkin")
    return warns


def enroll_all(face_app=None) -> dict:
    """Rebuild face_db.npz from faces/ folders. Returns stats dict."""
    from face_match import FaceGallery

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.FACES_DIR.mkdir(parents=True, exist_ok=True)

    people = [p for p in sorted(config.FACES_DIR.iterdir()) if p.is_dir()]
    if not people:
        FaceGallery.empty().save_npz(config.FACE_DB_PATH)
        return {"ok": True, "count": 0, "names": [], "skipped": [], "prototypes": 0, "warnings": []}

    app = build_app()
    min_quality = float(getattr(config, "FACE_ENROLL_MIN_QUALITY", 0.40))
    max_protos = int(getattr(config, "FACE_GALLERY_MAX_PROTOS", 16))
    dup_warn = float(getattr(config, "FACE_ENROLL_DUPLICATE_WARN", 0.82))

    names: list[str] = []
    all_embeddings: list[np.ndarray] = []
    all_person_ids: list[int] = []
    skipped: list[str] = []

    for person_dir in people:
        name = person_dir.name
        images = [
            p
            for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        if not images:
            skipped.append(f"{name}: rasm yo‘q")
            continue

        person_items: list[tuple[np.ndarray, float]] = []
        for img_path in images:
            img = _read_image(img_path)
            if img is None:
                skipped.append(f"{name}/{img_path.name}: o‘qilmadi")
                continue
            found = _collect_embeddings(app, img, min_quality)
            if not found:
                skipped.append(f"{name}/{img_path.name}: yuz topilmadi yoki sifati past")
                continue
            person_items.extend(found)

        protos = _pick_top_prototypes(person_items, max_protos)
        if not protos:
            skipped.append(f"{name}: yaroqli yuz yo‘q")
            continue

        pid = len(names)
        names.append(name)
        for emb in protos:
            all_embeddings.append(emb.astype(np.float32))
            all_person_ids.append(pid)

    if not all_embeddings:
        FaceGallery.empty().save_npz(config.FACE_DB_PATH)
        return {"ok": True, "count": 0, "names": [], "skipped": skipped, "prototypes": 0, "warnings": []}

    emb_arr = np.stack(all_embeddings, axis=0)
    pid_arr = np.array(all_person_ids, dtype=np.int32)
    gallery = FaceGallery(embeddings=emb_arr, names=names, person_ids=pid_arr)
    gallery.save_npz(config.FACE_DB_PATH)

    warnings = _find_duplicate_warnings(names, emb_arr, pid_arr, dup_warn)
    proto_counts = {n: int(np.sum(pid_arr == i)) for i, n in enumerate(names)}

    return {
        "ok": True,
        "count": len(names),
        "names": names,
        "skipped": skipped,
        "prototypes": int(emb_arr.shape[0]),
        "proto_counts": proto_counts,
        "warnings": warnings,
    }


def list_people() -> list[dict]:
    config.FACES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for person_dir in sorted(config.FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        photos = [
            p.name
            for p in person_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        out.append({"fio": person_dir.name, "photos": len(photos), "files": photos})
    return out


def main() -> int:
    print("Loading InsightFace (first run downloads models)...")
    result = enroll_all()
    print(f"Saved {result['count']} identities ({result.get('prototypes', 0)} prototypes) -> {config.FACE_DB_PATH}")
    for n, c in (result.get("proto_counts") or {}).items():
        print(f"  {n}: {c} prototype")
    for w in result.get("warnings") or []:
        print(f"  ! {w}")
    for s in result.get("skipped") or []:
        print(f"  skip: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
