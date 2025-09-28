"""ROI/图像通用工具。

- clamp_roi: 纠正 ROI 边界到图像尺寸内
- iter_images_from_dir: 遍历目录中的图片文件
- pick_roi_interactive: 使用 OpenCV 交互式选择 ROI
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple


def clamp_roi(x: int, y: int, w: int, h: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    return x, y, w, h


def iter_images_from_dir(dir_path: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for name in sorted(os.listdir(dir_path)):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            files.append(p)
    return files


def pick_roi_interactive(window_name: str, frame) -> Optional[Tuple[int, int, int, int]]:
    """打开一个交互窗口选择 ROI，回车确认，C 取消。

    返回 (x, y, w, h) 或 None。
    """
    import cv2  # 延迟导入，避免无 GUI 环境报错

    sel = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = map(int, sel)
    if w <= 0 or h <= 0:
        return None
    x, y, w, h = clamp_roi(x, y, w, h, frame.shape[1], frame.shape[0])
    return x, y, w, h
