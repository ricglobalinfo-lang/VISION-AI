"""Gallery-based face matching with per-person prototypes and margin checks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FaceGallery:
    """Multi-prototype gallery: each person may have several embeddings."""

    embeddings: np.ndarray  # (N, 512) L2-normalized
    names: list[str]
    person_ids: np.ndarray  # (N,) index into names

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def proto_count(self) -> int:
        return int(self.embeddings.shape[0]) if self.embeddings.size else 0

    @classmethod
    def empty(cls) -> FaceGallery:
        return cls(
            embeddings=np.zeros((0, 512), dtype=np.float32),
            names=[],
            person_ids=np.zeros((0,), dtype=np.int32),
        )

    @classmethod
    def from_npz(cls, path: Path) -> FaceGallery:
        if not path.is_file():
            return cls.empty()
        data = np.load(path, allow_pickle=True)
        names = [str(n) for n in data["names"].tolist()]
        embs = data["embeddings"].astype(np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1) if embs.size else np.zeros((0, 512), dtype=np.float32)
        if "person_ids" in data:
            person_ids = data["person_ids"].astype(np.int32)
        else:
            # Legacy: one embedding per person
            person_ids = np.arange(len(names), dtype=np.int32)
        if embs.shape[0] != person_ids.shape[0]:
            n = min(embs.shape[0], person_ids.shape[0])
            embs = embs[:n]
            person_ids = person_ids[:n]
        return cls(embeddings=embs, names=names, person_ids=person_ids)

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            names=np.array(self.names, dtype=object),
            embeddings=self.embeddings.astype(np.float32),
            person_ids=self.person_ids.astype(np.int32),
        )


@dataclass(frozen=True)
class MatchResult:
    name: str
    score: float
    margin: float
    second_name: str
    second_score: float
    ambiguous: bool
    detail: str = ""


def _person_best_scores(
    emb: np.ndarray, gallery: FaceGallery
) -> list[tuple[int, float]]:
    """Return [(person_id, best_proto_score), ...] sorted by score desc."""
    if gallery.proto_count == 0 or not gallery.names:
        return []
    sims = gallery.embeddings @ emb.reshape(-1)
    best: dict[int, float] = {}
    for i, pid in enumerate(gallery.person_ids.tolist()):
        s = float(sims[i])
        prev = best.get(int(pid))
        if prev is None or s > prev:
            best[int(pid)] = s
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return ranked


def match_gallery(
    emb: np.ndarray,
    gallery: FaceGallery,
    *,
    threshold: float,
    soft_threshold: float,
    min_margin: float,
    min_margin_soft: float = 0.03,
) -> MatchResult:
    """
    Match embedding against gallery.

    Uses max-over-prototypes per person, then requires a margin between
    the top two identities to avoid confusing similar people.
    """
    ranked = _person_best_scores(emb, gallery)
    if not ranked:
        return MatchResult(
            name="Noma'lum",
            score=0.0,
            margin=0.0,
            second_name="",
            second_score=0.0,
            ambiguous=False,
            detail="empty-gallery",
        )

    best_pid, best_score = ranked[0]
    best_name = gallery.names[best_pid]
    second_name = ""
    second_score = 0.0
    if len(ranked) > 1:
        second_pid, second_score = ranked[1]
        second_name = gallery.names[second_pid]
    margin = best_score - second_score

    if best_score >= threshold and margin >= min_margin:
        return MatchResult(
            name=best_name,
            score=best_score,
            margin=margin,
            second_name=second_name,
            second_score=second_score,
            ambiguous=False,
            detail=f"match m={margin:.2f}",
        )

    ambiguous = margin < min_margin and best_score >= soft_threshold
    if best_score >= soft_threshold and margin >= min_margin_soft:
        return MatchResult(
            name=best_name,
            score=best_score,
            margin=margin,
            second_name=second_name,
            second_score=second_score,
            ambiguous=ambiguous,
            detail=f"soft m={margin:.2f}" + (" ambiguous" if ambiguous else ""),
        )

    if ambiguous:
        return MatchResult(
            name="Noma'lum",
            score=best_score,
            margin=margin,
            second_name=second_name,
            second_score=second_score,
            ambiguous=True,
            detail=f"ambiguous {best_name} vs {second_name} m={margin:.2f}",
        )

    return MatchResult(
        name="Noma'lum",
        score=best_score,
        margin=margin,
        second_name=second_name,
        second_score=second_score,
        ambiguous=False,
        detail=f"low score={best_score:.2f}",
    )


def gallery_cfg(cfg: Any) -> dict[str, float]:
    return {
        "threshold": float(getattr(cfg, "FACE_MATCH_THRESHOLD", 0.36)),
        "soft_threshold": float(getattr(cfg, "FACE_SOFT_THRESHOLD", 0.28)),
        "min_margin": float(getattr(cfg, "FACE_MATCH_MARGIN", 0.05)),
        "min_margin_soft": float(getattr(cfg, "FACE_MATCH_MARGIN_SOFT", 0.03)),
    }
