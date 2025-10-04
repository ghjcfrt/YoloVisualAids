"""交通灯状态判定逻辑

根据检测得到的候选框（含坐标与置信度），并结合 ROI 颜色识别结果，
在竖直/水平两种信号灯样式之间做分支，输出最终的中文状态文案。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .color_detection import detect_traffic_light_color

if TYPE_CHECKING:
    from collections.abc import Iterable

Box = tuple[int, int, int, int, float]


def _is_vertical(b: Box) -> bool:
    """判定候选框是否竖直（高>宽）。"""
    x1, y1, x2, y2, _ = b
    return (y2 - y1) > (x2 - x1)


def _best_by_center(img_shape, boxes: Iterable[Box]) -> Box | None:
    """从候选中选择离图像中心最近（次关键按置信度倒序）的框。"""
    boxes = list(boxes)
    if not boxes:
        return None
    h, w = img_shape[:2]
    cx_img, cy_img = w / 2.0, h / 2.0

    def center_dist2(b: Box) -> float:
        x1, y1, x2, y2, _ = b
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx, dy = cx - cx_img, cy - cy_img
        return dx * dx + dy * dy

    return min(boxes, key=lambda b: (center_dist2(b), -b[4]))


def _color_of_box(frame, b: Box) -> str:
    """对框内 ROI 进行颜色判定，返回 red/yellow/green/unknown。"""
    x1, y1, x2, y2, _ = b
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    roi = frame[y1:y2, x1:x2]
    return detect_traffic_light_color(roi)


def _decide_from_vertical(frame, vertical: list[Box]) -> str | None:
    """针对竖直灯样式的判定策略。"""
    if not vertical:
        return None
    if len(vertical) > 1:
        vb = _best_by_center(frame.shape, vertical)
        if vb is None:
            return "无红绿灯"
        c = _color_of_box(frame, vb)
        return c if c in {"red", "yellow", "green"} else "红绿灯不工作"
    vb = vertical[0]
    c = _color_of_box(frame, vb)
    if c in {"red", "yellow", "green"}:
        return c
    return None


def _decide_from_horizontal(frame, horizontal: list[Box]) -> str | None:
    """针对水平灯样式的判定策略。"""
    if not horizontal:
        return None
    if len(horizontal) > 1:
        colors = [_color_of_box(frame, b) for b in horizontal]
        lit = [c for c in colors if c in {"red", "yellow", "green"}]
        if not lit:
            return "红绿灯不工作"
        uniq = set(lit)
        return next(iter(uniq)) if len(uniq) == 1 else "颜色不同"
    hb = horizontal[0]
    c = _color_of_box(frame, hb)
    return c if c in {"red", "yellow", "green"} else "红绿灯不工作"


def decide_traffic_status(frame, boxes: list[Box]) -> str:
    """根据候选框集合与颜色检测结果生成状态文案。"""
    if not boxes:
        return "无红绿灯"

    vertical = [b for b in boxes if _is_vertical(b)]
    horizontal = [b for b in boxes if not _is_vertical(b)]

    v_res = _decide_from_vertical(frame, vertical)
    if v_res is not None:
        if v_res is None and not horizontal:  # 保障分支
            return "红绿灯不工作"
        return v_res

    h_res = _decide_from_horizontal(frame, horizontal)
    if h_res is not None:
        return h_res

    if vertical and not horizontal:
        return "红绿灯不工作"
    return "无红绿灯"
