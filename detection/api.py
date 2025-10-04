"""检测 API 门面

统一导出核心类型与函数，供 GUI 与 CLI 调用：
- YOLOConfig: 参数配置
- YOLODetector: 核心检测器
- enumerate_cameras: 摄像头探测（OpenCV）
- load_config_from_args: 从命令行参数构建配置
- main: 命令行入口
"""

from __future__ import annotations  # noqa: I001

from .core import (YOLOConfig, YOLODetector, enumerate_cameras,
                   load_config_from_args, main)

__all__ = [
    "YOLOConfig",
    "YOLODetector",
    "enumerate_cameras",
    "load_config_from_args",
    "main",
]
