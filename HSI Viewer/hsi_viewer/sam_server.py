"""SAM3 inference server that runs in a separate *process*.

Why a process and not a thread: loading sam3.pt (3.4 GB) and building the model
hold the Python GIL for seconds at a time, which freezes the Qt event loop and makes
Windows show "응답 없음". In its own process the GUI never blocks, and CUDA state is
isolated from the viewer.

Protocol (multiprocessing.Pipe, pickled dicts):
    -> {"cmd": "load", "model": path}
    <- {"ev": "status", "text": ...} ... {"ev": "ready", "ok": bool, "error": str|None}
    -> {"cmd": "segment", "rgb": uint8 HxWx3 RGB, "key": any, "prompts": [...], "conf": f,
        "nms_iou": f}
    <- {"ev": "status", ...} ... {"ev": "result", "masks": packed bits (N, H*W/8), "shape": (H, W),
        "scores": (N,), "cls": (N,), "boxes": (N,4), "names": [...], "elapsed": f}
       or {"ev": "error", "error": str}
    -> {"cmd": "quit"}
"""
from __future__ import annotations

import time
import traceback


def serve(conn, model_path: str):
    import numpy as np

    predictor = None
    image_key = None

    def send(**kw):
        try:
            conn.send(kw)
        except Exception:
            pass

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        cmd = msg.get("cmd")
        if cmd == "quit":
            break

        if cmd == "load":
            if predictor is not None:
                send(ev="ready", ok=True, error=None)
                continue
            try:
                t0 = time.time()
                send(ev="status", text="SAM3 모델 로딩 중… (최초 1회, 약 10~20초)")
                from ultralytics.models.sam import SAM3SemanticPredictor

                predictor = SAM3SemanticPredictor(
                    overrides={
                        "conf": 0.2,
                        "task": "segment",
                        "mode": "predict",
                        "model": msg.get("model", model_path),
                        "save": False,
                        "verbose": False,
                    }
                )
                predictor.setup_model()
                send(ev="status", text=f"SAM3 준비 완료 ({time.time() - t0:.1f}s)")
                send(ev="ready", ok=True, error=None)
            except Exception as e:  # noqa: BLE001
                predictor = None
                send(ev="ready", ok=False, error=f"{e}\n{traceback.format_exc()}")
            continue

        if cmd == "segment":
            try:
                if predictor is None:
                    conn.send({"ev": "error", "error": "model not loaded"})
                    continue
                t0 = time.time()
                rgb = msg["rgb"]
                prompts = list(msg.get("prompts") or [])
                predictor.args.conf = float(msg.get("conf", 0.2))
                predictor.args.iou = float(msg.get("nms_iou", 0.7))
                key = msg.get("key")
                if key is None or key != image_key:
                    send(ev="status", text="이미지 인코딩 중…")
                    predictor.set_image(np.ascontiguousarray(rgb[:, :, ::-1]))
                    image_key = key
                bboxes = msg.get("bboxes")
                if bboxes:
                    send(ev="status", text=f"박스 예시 {len(bboxes)}개로 탐지 중…")
                    r = predictor(bboxes=np.asarray(bboxes, dtype=np.float32).reshape(-1, 4))[0]
                else:
                    send(ev="status", text="탐지 중: " + ", ".join(prompts))
                    r = predictor(text=prompts)[0]
                h, w = rgb.shape[:2]
                if r.masks is not None and len(r.boxes):
                    masks = r.masks.data.cpu().numpy().astype(bool)
                    out = {
                        "ev": "result",
                        "masks": np.packbits(masks.reshape(masks.shape[0], -1), axis=1),
                        "shape": (h, w),
                        "scores": r.boxes.conf.cpu().numpy().astype(np.float32),
                        "cls": r.boxes.cls.cpu().numpy().astype(np.int32),
                        "boxes": r.boxes.xyxy.cpu().numpy().round().astype(np.int32),
                        "names": list(r.names) if isinstance(r.names, (list, tuple)) else [r.names[i] for i in sorted(r.names)],
                        "elapsed": time.time() - t0,
                    }
                else:
                    out = {
                        "ev": "result",
                        "masks": np.zeros((0, (h * w + 7) // 8), np.uint8),
                        "shape": (h, w),
                        "scores": np.zeros(0, np.float32),
                        "cls": np.zeros(0, np.int32),
                        "boxes": np.zeros((0, 4), np.int32),
                        "names": prompts,
                        "elapsed": time.time() - t0,
                    }
                conn.send(out)
            except Exception as e:  # noqa: BLE001
                send(ev="error", error=f"{e}\n{traceback.format_exc()}")
            continue


def _entry(conn, model_path):
    try:
        serve(conn, model_path)
    finally:
        try:
            conn.close()
        except Exception:
            pass
