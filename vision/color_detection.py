from __future__ import annotations

from operator import itemgetter

import cv2
import numpy as np

MIN_SIDE_PX = 8
MAX_PREPROC_SIDE = 320
S_THR = 80
V_THR = 60
YELLOW_V_MIN = 80
GREEN_S_MIN = 70
MIN_AREA_RATIO = 0.002


def _color_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
    h, s, v = cv2.split(hsv)
    red1 = cv2.inRange(h, np.full_like(h, 0), np.full_like(h, 10))
    red2 = cv2.inRange(h, np.full_like(h, 170), np.full_like(h, 180))
    red = cv2.bitwise_or(red1, red2)
    red = cv2.bitwise_and(red, cv2.inRange(s, np.full_like(s, S_THR), np.full_like(s, 255)))
    red = cv2.bitwise_and(red, cv2.inRange(v, np.full_like(v, V_THR), np.full_like(v, 255)))

    yellow = cv2.inRange(h, np.full_like(h, 15), np.full_like(h, 35))
    yellow = cv2.bitwise_and(yellow, cv2.inRange(s, np.full_like(s, S_THR), np.full_like(s, 255)))
    yellow = cv2.bitwise_and(yellow, cv2.inRange(v, np.full_like(v, YELLOW_V_MIN), np.full_like(v, 255)))

    green = cv2.inRange(h, np.full_like(h, 40), np.full_like(h, 85))
    green = cv2.bitwise_and(green, cv2.inRange(s, np.full_like(s, GREEN_S_MIN), np.full_like(s, 255)))
    green = cv2.bitwise_and(green, cv2.inRange(v, np.full_like(v, V_THR), np.full_like(v, 255)))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k, iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k, iterations=1)
    return {"red": red, "yellow": yellow, "green": green}


def _preprocess_to_hsv(bgr_roi: np.ndarray) -> np.ndarray:
    h, w = bgr_roi.shape[:2]
    max_side = max(h, w)
    if max_side > MAX_PREPROC_SIDE:
        scale = MAX_PREPROC_SIDE / float(max_side)
        bgr_roi = cv2.resize(bgr_roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.GaussianBlur(bgr_roi, (3, 3), 0)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _best_color(hsv: np.ndarray, masks: dict[str, np.ndarray]) -> tuple[str, float, float]:
    area = hsv.shape[0] * hsv.shape[1]
    v_channel = hsv[:, :, 2]
    scores: dict[str, float] = {}
    for name, m in masks.items():
        nz = cv2.countNonZero(m)
        if nz == 0:
            scores[name] = 0.0
            continue
        v_masked = cv2.mean(v_channel, mask=m)[0]
        ratio = nz / max(1, area)
        scores[name] = ratio * ((v_masked / 255.0) ** 1.2)
    best_key, best_score = max(scores.items(), key=itemgetter(1))
    best_ratio = cv2.countNonZero(masks[best_key]) / max(1, area)
    return best_key, best_ratio, best_score


def detect_traffic_light_color(bgr_roi: np.ndarray) -> str:
    if bgr_roi is None or bgr_roi.size == 0:
        return "unknown"
    h, w = bgr_roi.shape[:2]
    if h < MIN_SIDE_PX or w < MIN_SIDE_PX:
        return "unknown"
    hsv = _preprocess_to_hsv(bgr_roi)
    masks = _color_masks(hsv)
    best_key, best_ratio, best_score = _best_color(hsv, masks)
    if best_ratio < MIN_AREA_RATIO or best_score <= 0.0:
        return "unknown"
    return best_key


__all__ = ["detect_traffic_light_color"]
