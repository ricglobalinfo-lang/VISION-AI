"""MiniFASNet anti-spoof (print/screen) via ONNXRuntime."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_MODEL_SPECS: tuple[tuple[str, float, str], ...] = (
    (
        "MiniFASNetV2.onnx",
        2.7,
        "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
    ),
    (
        "MiniFASNetV1SE.onnx",
        4.0,
        "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx",
    ),
)


@dataclass
class AntiSpoofResult:
    is_real: bool
    real_prob: float
    fake_prob: float
    detail: str = ""


class _OnnxAntiSpoof:
    def __init__(self, model_path: Path, scale: float, providers: list[str]) -> None:
        import onnxruntime as ort

        self.scale = float(scale)
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = (int(inp.shape[3]), int(inp.shape[2]))  # W, H

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _crop_face(self, image: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        x, y, box_w, box_h = bbox_xywh
        box_w = max(1, int(box_w))
        box_h = max(1, int(box_h))
        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, self.scale)
        new_w = box_w * scale
        new_h = box_h * scale
        center_x = x + box_w / 2.0
        center_y = y + box_h / 2.0
        x1 = max(0, int(center_x - new_w / 2.0))
        y1 = max(0, int(center_y - new_h / 2.0))
        x2 = min(src_w - 1, int(center_x + new_w / 2.0))
        y2 = min(src_h - 1, int(center_y + new_h / 2.0))
        cropped = image[y1 : y2 + 1, x1 : x2 + 1]
        if cropped.size == 0:
            cropped = image
        return cv2.resize(cropped, self.input_size, interpolation=cv2.INTER_LINEAR)

    def real_prob(self, image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        bbox_xywh = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        face = self._crop_face(image_bgr, bbox_xywh)
        tensor = face.astype(np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        out = self.session.run(None, {self.input_name: tensor})[0]
        probs = self._softmax(np.asarray(out, dtype=np.float32))
        # yakhyo/MiniFAS: idx 0 = Fake, 1 = Real
        if probs.shape[1] >= 2:
            return float(probs[0, 1])
        return float(probs[0, 0])


def ensure_antispoof_models(model_dir: Path) -> list[Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, _scale, url in _MODEL_SPECS:
        dst = model_dir / name
        if dst.exists() and dst.stat().st_size > 100_000:
            paths.append(dst)
            continue
        print(f"Anti-spoof: yuklanmoqda {name} ...")
        try:
            urllib.request.urlretrieve(url, dst)
            paths.append(dst)
            print(f"Anti-spoof: {name} OK")
        except Exception as e:
            print(f"Anti-spoof: {name} yuklanmadi: {e}")
    return paths


class AntiSpoofEnsemble:
    def __init__(self, cfg: Any) -> None:
        self.enabled = bool(getattr(cfg, "FACE_ANTISPOOF_ENABLED", True))
        self.real_thresh = float(getattr(cfg, "FACE_ANTISPOOF_REAL_THRESH", 0.55))
        self.models: list[_OnnxAntiSpoof] = []
        if not self.enabled:
            return
        model_dir = Path(getattr(cfg, "FACE_ANTISPOOF_MODEL_DIR", cfg.DATA_DIR / "antispoof_models"))
        providers: list[str] = []
        if getattr(cfg, "FACE_USE_GPU", True):
            try:
                import onnxruntime as ort

                if "CUDAExecutionProvider" in set(ort.get_available_providers()):
                    providers.append("CUDAExecutionProvider")
            except Exception:
                pass
        providers.append("CPUExecutionProvider")
        specs = {name: scale for name, scale, _url in _MODEL_SPECS}
        for path in ensure_antispoof_models(model_dir):
            scale = specs.get(path.name, 2.7)
            try:
                self.models.append(_OnnxAntiSpoof(path, scale, providers))
            except Exception as e:
                print(f"Anti-spoof load fail {path.name}: {e}")
        if not self.models:
            print("Anti-spoof: modellar yo'q — o'chirildi")
            self.enabled = False

    def predict(self, frame_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> AntiSpoofResult:
        if not self.enabled or not self.models:
            return AntiSpoofResult(True, 1.0, 0.0, "disabled")
        probs = [m.real_prob(frame_bgr, bbox_xyxy) for m in self.models]
        real_prob = float(np.mean(probs))
        fake_prob = 1.0 - real_prob
        is_real = real_prob >= self.real_thresh
        detail = f"real={real_prob:.2f} models={len(probs)}"
        return AntiSpoofResult(is_real=is_real, real_prob=real_prob, fake_prob=fake_prob, detail=detail)
