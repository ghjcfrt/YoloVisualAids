from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

MIN_SIDE_PX = 8


def select_device(requested: str | None = None) -> str:
    if requested and requested.lower() not in {"", "auto"}:
        return requested
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda"
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except (AttributeError, RuntimeError):
            pass
    return "cpu"


def parse_img_size(s: str | None) -> list[int] | None:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(',') if p.strip()]
    try:
        arr = [int(x) for x in parts]
    except (TypeError, ValueError):
        return None
    else:
        return arr or None


@dataclass
class YOLOOpts:
    # 默认指向仓库内 models/yolo/yolo11n.pt
    model_path: str = "models/yolo/yolo11n.pt"
    conf: float = 0.5
    img_size: list[int] | None = None
    device: str = "auto"
    class_id: int = 9
    first_only: bool = False


class YOLOAutoDetector:
    def __init__(self, opts: YOLOOpts):
        self.opts = opts
        self.device = select_device(opts.device)
        if YOLO is None:
            msg = "未安装 ultralytics，请先安装: pip install ultralytics"
            raise ImportError(msg)
        self.model = YOLO(opts.model_path)
        self.last_state: str = "init"

    def _predict_and_collect(self, frame) -> list[tuple[int, int, int, int, float]]:
        if frame is None or not isinstance(frame, np.ndarray):
            msg = "frame 必须是 numpy 图像"
            raise TypeError(msg)

        imgsz = self.opts.img_size
        if imgsz is None:
            h, w = frame.shape[:2]
            imgsz = [h, w]

        res = self.model.predict(frame, imgsz=imgsz, conf=self.opts.conf, device=self.device, verbose=False)
        r = res[0]
        return self._collect_candidate_boxes(r, frame.shape)

    def _collect_candidate_boxes(self, r, frame_shape: tuple[int, ...]) -> list[tuple[int, int, int, int, float]]:
        h_img, w_img = frame_shape[:2]
        out: list[tuple[int, int, int, int, float]] = []
        for box in getattr(r, 'boxes', []):
            try:
                cls_id = int(box.cls.item())
                if cls_id != int(self.opts.class_id):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
            except (AttributeError, ValueError, TypeError, IndexError):
                continue

            x1i = max(0, min(w_img - 1, int(x1)))
            y1i = max(0, min(h_img - 1, int(y1)))
            x2i = max(0, min(w_img, int(x2)))
            y2i = max(0, min(h_img, int(y2)))
            if x2i <= x1i or y2i <= y1i:
                continue
            if (y2i - y1i) < MIN_SIDE_PX or (x2i - x1i) < MIN_SIDE_PX:
                continue
            out.append((x1i, y1i, x2i, y2i, conf))
        out.sort(key=operator.itemgetter(4), reverse=True)
        return out

    @staticmethod
    def _is_vertical(b: tuple[int, int, int, int, float]) -> bool:
        x1, y1, x2, y2, _ = b
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return h > w

    @staticmethod
    def _center_distance2(b: tuple[int, int, int, int, float], cx_img: float, cy_img: float) -> float:
        x1, y1, x2, y2, _ = b
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = cx - cx_img
        dy = cy - cy_img
        return dx * dx + dy * dy

    def detect(self, frame) -> list[tuple[int, int, int, int, float]]:
        boxes = self._predict_and_collect(frame)
        if not boxes:
            self.last_state = "none"
            return []
        vertical_boxes = [b for b in boxes if self._is_vertical(b)]
        if not vertical_boxes:
            self.last_state = "no_vertical"
            return []
        h_img, w_img = frame.shape[:2]
        cx_img, cy_img = w_img / 2.0, h_img / 2.0
        best_box = min(
            vertical_boxes,
            key=lambda b: (self._center_distance2(b, cx_img, cy_img), -b[4]),
        )
        self.last_state = "ok"
        return [best_box]

    def detect_orientations(
        self, frame
    ) -> tuple[list[tuple[int, int, int, int, float]], list[tuple[int, int, int, int, float]]]:
        boxes = self._predict_and_collect(frame)
        if not boxes:
            self.last_state = "none"
            return [], []
        vertical = [b for b in boxes if self._is_vertical(b)]
        horizontal = [b for b in boxes if not self._is_vertical(b)]
        self.last_state = "ok" if vertical else "no_vertical"
        return vertical, horizontal
