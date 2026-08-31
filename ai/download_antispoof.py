"""Download MiniFASNet ONNX anti-spoof weights."""
from __future__ import annotations

import face_antispoof
import config


def main() -> None:
    model_dir = config.DATA_DIR / "antispoof_models"
    paths = face_antispoof.ensure_antispoof_models(model_dir)
    if paths:
        print(f"OK: {len(paths)} model(lar) -> {model_dir}")
    else:
        print("XATO: hech qanday model yuklanmadi")


if __name__ == "__main__":
    main()
