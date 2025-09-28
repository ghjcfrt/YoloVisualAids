"""YOLO 推理工具封装。

提供：
- select_device: 自动/指定设备选择
- parse_img_size: 将 '640' 或 '640,640' 字符串解析为 list[int]
- YOLOAutoDetector: 简易封装，按给定阈值与类别检出框
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


def select_device(requested: str | None = None) -> str:
    try:
        import torch  # noqa: WPS433
        if requested and requested.lower() not in {"", "auto"}:
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def parse_img_size(s: str | None) -> Optional[List[int]]:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(',') if p.strip()]
    try:
        arr = [int(x) for x in parts]
        return arr if arr else None
    except Exception:
        return None


@dataclass
class YOLOOpts:
    model_path: str = "yolo11n.pt"
    conf: float = 0.5
    img_size: Optional[List[int]] = None
    device: str = "auto"
    class_id: int = 9
    first_only: bool = False


class YOLOAutoDetector:
    def __init__(self, opts: YOLOOpts):
        from ultralytics import YOLO  # 延迟导入

        self.opts = opts
        self.device = select_device(opts.device)
        self.model = YOLO(opts.model_path)

    def detect(self, frame) -> List[Tuple[int, int, int, int, float]]:
        """返回 (x1,y1,x2,y2,conf) 仅包含指定 class-id，按置信度降序。"""
        import numpy as np

        assert frame is not None and isinstance(frame, (np.ndarray,)), "frame 必须是 numpy 图像"
        imgsz = self.opts.img_size
        if imgsz is None:
            h, w = frame.shape[:2]
            imgsz = [h, w]
        res = self.model.predict(frame, imgsz=imgsz, conf=self.opts.conf, device=self.device, verbose=False)
        r = res[0]
        boxes: List[Tuple[int, int, int, int, float]] = []
        try:
            h_img, w_img = frame.shape[:2]
            for box in getattr(r, 'boxes', []):
                try:
                    cls_id = int(box.cls.item())
                    if cls_id != int(self.opts.class_id):
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
                except Exception:
                    continue
                x1i = max(0, min(w_img - 1, int(x1)))
                y1i = max(0, min(h_img - 1, int(y1)))
                x2i = max(0, min(w_img, int(x2)))
                y2i = max(0, min(h_img, int(y2)))
                if x2i <= x1i or y2i <= y1i:
                    continue
                if (y2i - y1i) < 8 or (x2i - x1i) < 8:
                    continue
                boxes.append((x1i, y1i, x2i, y2i, conf))
        except Exception:
            pass
        boxes.sort(key=lambda b: b[4], reverse=True)
        if self.opts.first_only and boxes:
            return [boxes[0]]
        return boxes
