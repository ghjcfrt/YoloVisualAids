"""YOLO 推理工具封装。

提供：
- select_device: 自动/指定设备选择
- parse_img_size: 将 '640' 或 '640,640' 字符串解析为 list[int]
- YOLOAutoDetector: 简易封装，按给定阈值与类别检出框
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - 环境可能没有 torch
    torch = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - 允许在未安装时被延迟导入报错
    YOLO = None  # type: ignore[assignment]


# 最小检测框边长像素（用于过滤极小噪声框）
MIN_SIDE_PX = 8


def select_device(requested: str | None = None) -> str:
    if requested and requested.lower() not in {"", "auto"}:
        return requested
    if torch is not None:
        try:
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                return "cuda"
            backends = getattr(torch, "backends", None)
            if getattr(backends, "mps", None) and backends.mps.is_available():  # type: ignore[union-attr]
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
    model_path: str = "yolo11n.pt"
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
        # 最近一次检测状态："init" | "none" | "no_vertical" | "ok"
        self.last_state: str = "init"

    def _collect_candidate_boxes(self, r, frame_shape: tuple[int, ...]) -> list[tuple[int, int, int, int, float]]:
        """从 YOLO 结果中提取候选框，仅保留指定类别并进行边界/尺寸约束。"""
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
        # 置信度降序
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
        """返回单个最佳框或空：优先竖向、再按距中心最近。

        规则：
        - 只考虑竖向红绿灯（高>宽）。
        - 多个竖向时，选距离图像中心最近的一个（次序再参考置信度）。
        - 无竖向则返回空列表。
        """
        if frame is None or not isinstance(frame, np.ndarray):
            msg = "frame 必须是 numpy 图像"
            raise TypeError(msg)

        imgsz = self.opts.img_size
        if imgsz is None:
            h, w = frame.shape[:2]
            imgsz = [h, w]

        res = self.model.predict(frame, imgsz=imgsz, conf=self.opts.conf, device=self.device, verbose=False)
        r = res[0]

        boxes = self._collect_candidate_boxes(r, frame.shape)
        if not boxes:
            self.last_state = "none"
            return []

        vertical_boxes = [b for b in boxes if self._is_vertical(b)]
        if not vertical_boxes:
            # 有检测，但没有竖直红绿灯
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
