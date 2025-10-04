"""检测 API 门面：统一导出核心类型与函数"""

from __future__ import annotations

from .core import (YOLOConfig, YOLODetector, enumerate_cameras,
                   load_config_from_args, main)

__all__ = [
    "YOLOConfig",
    "YOLODetector",
    "enumerate_cameras",
    "load_config_from_args",
    "main",
]
