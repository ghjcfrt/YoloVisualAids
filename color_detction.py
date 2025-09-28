"""
基于 HSV 的交通灯颜色识别

提供 detect_traffic_light_color(bgr_roi) -> str
返回值: 'red' | 'yellow' | 'green' | 'unknown'

实现说明:
- 使用 OpenCV HSV 空间阈值分割红/黄/绿
- 结合面积占比与亮度得分的综合评分，提升鲁棒性
- 对很小或空的 ROI 返回 'unknown'
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def _color_masks(hsv: np.ndarray) -> Dict[str, np.ndarray]:
    """在 HSV 空间生成红/黄/绿的二值掩码。

    OpenCV HSV 范围:
      H: 0~180, S:0~255, V:0~255
    """
    h, s, v = cv2.split(hsv)

    # 通用阈值（可按需微调）
    s_thr = 80
    v_thr = 60

    # 红色包含两个区间 (0-10) 与 (170-180)
    red1 = cv2.inRange(h, np.full_like(h, 0), np.full_like(h, 10))
    red2 = cv2.inRange(h, np.full_like(h, 170), np.full_like(h, 180))
    red = cv2.bitwise_or(red1, red2)
    red = cv2.bitwise_and(
        red,
        cv2.inRange(s, np.full_like(s, s_thr), np.full_like(s, 255)),
    )
    red = cv2.bitwise_and(
        red,
        cv2.inRange(v, np.full_like(v, v_thr), np.full_like(v, 255)),
    )

    # 黄色大致 15~35（饱和度、亮度略高）
    yellow = cv2.inRange(h, np.full_like(h, 15), np.full_like(h, 35))
    yellow = cv2.bitwise_and(
        yellow,
        cv2.inRange(s, np.full_like(s, s_thr), np.full_like(s, 255)),
    )
    yellow = cv2.bitwise_and(
        yellow,
        cv2.inRange(v, np.full_like(v, 80), np.full_like(v, 255)),
    )

    # 绿色大致 40~85
    green = cv2.inRange(h, np.full_like(h, 40), np.full_like(h, 85))
    green = cv2.bitwise_and(
        green,
        cv2.inRange(s, np.full_like(s, 70), np.full_like(s, 255)),
    )
    green = cv2.bitwise_and(
        green,
        cv2.inRange(v, np.full_like(v, v_thr), np.full_like(v, 255)),
    )

    # 形态学去噪
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k, iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k, iterations=1)

    return {"red": red, "yellow": yellow, "green": green}


def detect_traffic_light_color(bgr_roi: np.ndarray) -> str:
    """对裁剪到的交通灯 ROI 进行颜色识别。

    参数:
        bgr_roi: OpenCV BGR 图像 (H,W,3)
    返回:
        'red' | 'yellow' | 'green' | 'unknown'
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return "unknown"
    h, w = bgr_roi.shape[:2]
    if h < 8 or w < 8:
        return "unknown"

    # 适度缩放，降低噪声影响
    max_side = max(h, w)
    if max_side > 320:
        scale = 320.0 / max_side
        bgr_roi = cv2.resize(bgr_roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 轻度平滑
    bgr = cv2.GaussianBlur(bgr_roi, (3, 3), 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    masks = _color_masks(hsv)
    area = hsv.shape[0] * hsv.shape[1]
    v_channel = hsv[:, :, 2]

    scores: Dict[str, float] = {}
    for name, m in masks.items():
        nz = cv2.countNonZero(m)
        if nz == 0:
            scores[name] = 0.0
            continue
        # 亮度得分（只在掩码内统计）
        v_masked = cv2.mean(v_channel, mask=m)[0]  # 0~255
        ratio = nz / max(1, area)
        # 综合评分：面积占比 * 亮度权重
        scores[name] = ratio * ((v_masked / 255.0) ** 1.2)

    # 最小面积占比阈值，避免噪点误判
    min_ratio = 0.002  # 0.2%
    best_key, best_score = max(scores.items(), key=lambda kv: kv[1])
    best_mask = masks[best_key]
    best_ratio = cv2.countNonZero(best_mask) / max(1, area)

    if best_ratio < min_ratio or best_score <= 0.0:
        return "unknown"
    return best_key


__all__ = ["detect_traffic_light_color"]
