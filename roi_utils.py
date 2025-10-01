"""ROI/图像通用工具。

"""

from __future__ import annotations

from pathlib import Path

import cv2

# 公共类型别名
ROI = tuple[int, int, int, int]


def clamp_roi(x: int, y: int, w: int, h: int, size: tuple[int, int]) -> ROI:
    """将 (x, y, w, h) 限制在宽高 size 内。"""
    width, height = size
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    return x, y, w, h


def iter_images_from_dir(dir_path: str) -> list[str]:
    """遍历目录中图像文件，返回绝对或相对路径字符串列表。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    base = Path(dir_path)
    files: list[str] = [str(c) for c in sorted(base.iterdir()) if c.is_file() and c.suffix.lower() in exts]
    return files


def pick_roi_interactive(window_name: str, frame) -> ROI | None:
    """打开一个交互窗口选择 ROI，回车确认，C 取消。

    返回 (x, y, w, h) 或 None。
    """
    sel = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = map(int, sel)
    if w <= 0 or h <= 0:
        return None
    x, y, w, h = clamp_roi(x, y, w, h, (frame.shape[1], frame.shape[0]))
    return x, y, w, h
