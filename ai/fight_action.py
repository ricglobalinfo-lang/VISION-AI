"""
Skeleton action recognition for fight detection.

Combines:
  1) Lightweight ST-GCN (spatiotemporal graph conv on COCO-17)
  2) Geometric temporal scorer (wrist intrusion, relative speed) — primary precision gate

Output: fight_prob in [0, 1]. Standing side-by-side → low score.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# COCO-17 skeleton edges (undirected)
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


def _normalize_pair_seq(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """
    seq_*: (T, 17, 3) → relative pair tensor (T, 17, 6)
    channels: ax, ay, ac, bx_rel, by_rel, bc  (b relative to a hip/shoulder center)
    """
    T = min(len(seq_a), len(seq_b))
    a = seq_a[-T:].astype(np.float32).copy()
    b = seq_b[-T:].astype(np.float32).copy()

    def center_scale(sk: np.ndarray) -> tuple[np.ndarray, float]:
        # use shoulders mid if visible else bbox of visible joints
        conf = sk[:, :, 2]
        xy = sk[:, :, :2]
        mid = []
        for t in range(len(sk)):
            pts = []
            for j in (5, 6, 11, 12):
                if conf[t, j] >= 0.2:
                    pts.append(xy[t, j])
            if pts:
                mid.append(np.mean(pts, axis=0))
            else:
                vis = conf[t] >= 0.2
                mid.append(xy[t, vis].mean(axis=0) if vis.any() else np.zeros(2))
        mid_arr = np.asarray(mid, dtype=np.float32)
        # scale by median shoulder width / torso
        scales = []
        for t in range(len(sk)):
            if conf[t, 5] >= 0.2 and conf[t, 6] >= 0.2:
                scales.append(float(np.linalg.norm(xy[t, 5] - xy[t, 6])))
            elif conf[t, 11] >= 0.2 and conf[t, 12] >= 0.2:
                scales.append(float(np.linalg.norm(xy[t, 11] - xy[t, 12])))
        scale = float(np.median(scales)) if scales else 50.0
        scale = max(scale, 20.0)
        out = sk.copy()
        out[:, :, 0] = (xy[:, :, 0] - mid_arr[:, None, 0]) / scale
        out[:, :, 1] = (xy[:, :, 1] - mid_arr[:, None, 1]) / scale
        return out, scale

    a_n, _ = center_scale(a)
    b_n, _ = center_scale(b)
    # b relative to a frame origin already per-person; also add a-b delta in a's frame approx
    rel = np.concatenate([a_n, b_n], axis=-1)  # (T,17,6)
    return rel


class GraphConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, adj: torch.Tensor):
        super().__init__()
        self.register_buffer("adj", adj)
        self.linear = nn.Linear(in_c, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, C)
        x = torch.einsum("vw,btwc->btvc", self.adj, x)
        return self.linear(x)


class STGCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, adj: torch.Tensor, t_kernel: int = 5):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, adj)
        pad = t_kernel // 2
        self.tcn = nn.Conv2d(out_c, out_c, kernel_size=(t_kernel, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(out_c)
        self.res = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,V,C)
        y = self.gcn(x)
        res = self.res(x)
        # TCN expects (B,C,T,V)
        y2 = y.permute(0, 3, 1, 2)
        y2 = self.bn(F.relu(self.tcn(y2)))
        y = y2.permute(0, 2, 3, 1)
        return F.relu(y + res)


def build_adjacency(num_joints: int = 17) -> torch.Tensor:
    a = np.eye(num_joints, dtype=np.float32)
    for i, j in COCO_EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    # normalize
    d = np.sum(a, axis=1, keepdims=True)
    a = a / np.maximum(d, 1e-6)
    return torch.from_numpy(a)


class ActionGCN(nn.Module):
    """Lightweight ST-GCN binary fight classifier."""

    def __init__(self, in_c: int = 6, hid: int = 64):
        super().__init__()
        adj = build_adjacency(17)
        self.b1 = STGCNBlock(in_c, hid, adj)
        self.b2 = STGCNBlock(hid, hid, adj)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,V,C)
        x = self.b1(x)
        x = self.b2(x)
        # mean over T,V
        x = x.mean(dim=(1, 2))
        return self.fc(x).squeeze(-1)


def geometric_fight_score(seq_a: np.ndarray, seq_b: np.ndarray) -> dict[str, float]:
    """
    Precision-oriented temporal geometry on two skeleton sequences (T,17,3).
    Standing side-by-side → low wrist_intrusion + low approach_speed.
    """
    T = min(len(seq_a), len(seq_b))
    if T < 8:
        return {"fight_prob": 0.0, "intrusion": 0.0, "wrist_speed": 0.0, "approach": 0.0}

    a = seq_a[-T:]
    b = seq_b[-T:]

    def torso_box(sk_t: np.ndarray) -> tuple[float, float, float, float] | None:
        # sk_t: (17,3)
        conf = sk_t[:, 2]
        idxs = [j for j in (0, 5, 6, 11, 12) if conf[j] >= 0.25]
        if len(idxs) < 2:
            return None
        xs = sk_t[idxs, 0]
        ys = sk_t[idxs, 1]
        # expand toward head
        pad_x = 0.25 * (xs.max() - xs.min() + 1)
        y0 = ys.min() - 0.15 * (ys.max() - ys.min() + 1)
        y1 = ys.min() + 0.75 * (ys.max() - ys.min() + 1)
        return float(xs.min() - pad_x), float(y0), float(xs.max() + pad_x), float(y1)

    def center(sk_t: np.ndarray) -> np.ndarray:
        conf = sk_t[:, 2]
        pts = []
        for j in (5, 6, 11, 12):
            if conf[j] >= 0.25:
                pts.append(sk_t[j, :2])
        if not pts:
            vis = conf >= 0.25
            if not vis.any():
                return np.zeros(2, dtype=np.float32)
            return sk_t[vis, :2].mean(axis=0)
        return np.mean(pts, axis=0)

    intrusion_hits = 0
    intrusion_checks = 0
    wrist_speeds: list[float] = []
    centers_a = []
    centers_b = []

    for t in range(T):
        ca = center(a[t])
        cb = center(b[t])
        centers_a.append(ca)
        centers_b.append(cb)
        scale = max(float(np.linalg.norm(ca - cb)), 30.0)

        for src, dst in ((a[t], b[t]), (b[t], a[t])):
            box = torso_box(dst)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            for wi in (9, 10):
                if src[wi, 2] < 0.35:
                    continue
                intrusion_checks += 1
                wx, wy = float(src[wi, 0]), float(src[wi, 1])
                if x1 <= wx <= x2 and y1 <= wy <= y2:
                    intrusion_hits += 1

        if t > 0:
            for wi in (9, 10):
                if a[t, wi, 2] >= 0.35 and a[t - 1, wi, 2] >= 0.35:
                    wrist_speeds.append(float(np.linalg.norm(a[t, wi, :2] - a[t - 1, wi, :2]) / scale))
                if b[t, wi, 2] >= 0.35 and b[t - 1, wi, 2] >= 0.35:
                    wrist_speeds.append(float(np.linalg.norm(b[t, wi, :2] - b[t - 1, wi, :2]) / scale))

    centers_a = np.asarray(centers_a)
    centers_b = np.asarray(centers_b)
    dist = np.linalg.norm(centers_a - centers_b, axis=1)
    # approach: distance decreasing with variance (not static close)
    approach = 0.0
    if len(dist) >= 8:
        d0 = float(np.median(dist[: max(3, T // 4)]))
        d1 = float(np.median(dist[-max(3, T // 4) :]))
        shrink = max(0.0, (d0 - d1) / max(d0, 1.0))
        jitter = float(np.std(np.diff(dist))) / max(float(np.median(dist)), 1.0)
        approach = float(np.clip(0.6 * shrink + 0.8 * min(jitter, 0.5), 0.0, 1.0))

    intrusion = float(intrusion_hits / max(intrusion_checks, 1))
    # require sustained intrusion, not a single frame
    if intrusion_checks >= 8:
        intrusion = float(intrusion_hits / intrusion_checks)
    else:
        intrusion *= 0.5

    wspeed = float(np.percentile(wrist_speeds, 80)) if wrist_speeds else 0.0
    wspeed_s = float(np.clip(wspeed / 0.12, 0.0, 1.0))

    # Precision: tezkor qo‘l YETMAYDI — boshqa odam tanasiga kirish kerak
    fight = 0.0
    if intrusion >= 0.18:
        fight = 0.52 * intrusion + 0.30 * wspeed_s + 0.18 * approach
        if intrusion >= 0.40 and wspeed_s >= 0.25:
            fight = max(fight, 0.72)
        if intrusion >= 0.50 and wspeed_s >= 0.35:
            fight = max(fight, 0.84)
        if intrusion >= 0.35 and wspeed_s >= 0.50 and approach >= 0.12:
            fight = max(fight, 0.80)
    # Hard reject: suhbat, o‘tirish, bitta odam qo‘l harakati
    if intrusion < 0.18:
        fight = 0.0
    elif wspeed_s < 0.18 and approach < 0.12:
        fight *= 0.25

    fight = float(np.clip(fight, 0.0, 1.0))
    return {
        "fight_prob": fight,
        "intrusion": intrusion,
        "wrist_speed": wspeed_s,
        "approach": approach,
    }


class FightActionModel:
    """ST-GCN + geometric scorer."""

    def __init__(self, cfg: Any):
        self.thresh = float(getattr(cfg, "FIGHT_ACTION_THRESH", 0.75))
        self.seq_len = int(getattr(cfg, "FIGHT_SEQ_LEN", 48))
        raw_dev = getattr(cfg, "FIGHT_ACTION_DEVICE", 0)
        if raw_dev == -1 or raw_dev == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{int(raw_dev)}")
        self.net = ActionGCN(in_c=6, hid=64).to(self.device)
        self.net.eval()
        # Untrained ST-GCN is not trusted alone — geometric score is primary.
        # ST-GCN provides a mild temporal smoothness prior (random → ~0.5, so we
        # center it and keep small weight).
        self.gcn_weight = 0.18
        self.geo_weight = 0.82

    @torch.inference_mode()
    def score_pair(self, seq_a: np.ndarray, seq_b: np.ndarray) -> dict[str, float]:
        geo = geometric_fight_score(seq_a, seq_b)
        geo_p = float(geo["fight_prob"])

        gcn_p = 0.5
        try:
            rel = _normalize_pair_seq(seq_a, seq_b)
            if len(rel) >= 12:
                # resample / pad to seq_len
                if len(rel) < self.seq_len:
                    pad = np.repeat(rel[:1], self.seq_len - len(rel), axis=0)
                    rel = np.concatenate([pad, rel], axis=0)
                else:
                    rel = rel[-self.seq_len :]
                x = torch.from_numpy(rel).unsqueeze(0).to(self.device)  # (1,T,17,6)
                logit = self.net(x)
                gcn_p = float(torch.sigmoid(logit).item())
        except Exception:
            gcn_p = 0.5

        # Center untrained GCN around neutral
        gcn_centered = abs(gcn_p - 0.5) * 2.0  # not used as fight evidence alone
        # Only let GCN boost when geometric already suspicious
        if geo_p >= 0.45:
            fight_prob = self.geo_weight * geo_p + self.gcn_weight * max(gcn_p, geo_p)
        else:
            fight_prob = geo_p * 0.85

        fight_prob = float(np.clip(fight_prob, 0.0, 1.0))
        return {
            "fight_prob": fight_prob,
            "geo": geo_p,
            "gcn": gcn_p,
            "intrusion": float(geo["intrusion"]),
            "wrist_speed": float(geo["wrist_speed"]),
            "approach": float(geo["approach"]),
            "gcn_centered": float(gcn_centered),
        }
