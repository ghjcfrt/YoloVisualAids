"""
红绿灯决策逻辑（不依赖 YOLO 推理器）：

提供 decide_traffic_status(frame, boxes) -> str
- frame: OpenCV BGR 图
- boxes: [(x1,y1,x2,y2,conf), ...]，只传入交通灯类别的检测框

返回：
- 'red' | 'yellow' | 'green' | '颜色不同' | '红绿灯不工作' | '无红绿灯'

规则：参见用户需求说明（竖/横向 + 数量 + 颜色一致性）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型检查使用，避免运行期开销
    from collections.abc import Iterable

from color_detction import detect_traffic_light_color

Box = tuple[int, int, int, int, float]


def _is_vertical(b: Box) -> bool:
    x1, y1, x2, y2, _ = b
    return (y2 - y1) > (x2 - x1)


def _best_by_center(img_shape, boxes: Iterable[Box]) -> Box | None:
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
    x1, y1, x2, y2, _ = b
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    roi = frame[y1:y2, x1:x2]
    return detect_traffic_light_color(roi)


def _decide_from_vertical(frame, vertical: list[Box]) -> str | None:
    """基于竖向灯做一次判定。
    返回 red/yellow/green 或 "红绿灯不工作"；当需继续看横向时返回 None。
    """
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
    # 竖向存在但不亮：交由外层决定是否需要 fallback 到横向
    return None


def _decide_from_horizontal(frame, horizontal: list[Box]) -> str | None:
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
    if not boxes:
        return "无红绿灯"

    vertical = [b for b in boxes if _is_vertical(b)]
    horizontal = [b for b in boxes if not _is_vertical(b)]

    # 1) 先看竖向
    v_res = _decide_from_vertical(frame, vertical)
    if v_res is not None:
        # 如果竖向明确不亮，且没有横向，则给出不工作
        if v_res is None and not horizontal:  # pragma: no cover - 保护分支
            return "红绿灯不工作"
        return v_res

    # 2) 竖向不确定 -> 看横向
    h_res = _decide_from_horizontal(frame, horizontal)
    if h_res is not None:
        return h_res

    # 3) 都没有
    if vertical and not horizontal:
        return "红绿灯不工作"
    return "无红绿灯"
