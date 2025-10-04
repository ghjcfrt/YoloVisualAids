"""设备选择与可用设备枚举工具

提供 list_devices 辅助函数，用于根据可选依赖 torch 的实际可用性，
返回推理设备候选项列表（如: auto/cpu/cuda/cuda:0/.../mps）。

注意：本模块不强制依赖 torch；当未安装或不可用时，仅返回通用项。
"""

from __future__ import annotations

from typing import Any


def list_devices(torch_mod: Any | None) -> list[str]:
    """列出可供选择的推理设备字符串列表。

    参数
    - torch_mod: 传入的 torch 模块对象（或 None）。
      为 None 时，仅返回通用选项；非 None 时会探测 CUDA/MPS。

    返回
    - 形如 ["auto", "cpu", "cuda", "cuda:1", "mps", ...] 的列表，去重保序。
    """
    options: list[str] = ["auto", "cpu"]

    def _detect_cuda() -> list[str]:
        # 探测 CUDA 可用 GPU：
        # - 仅在传入的 torch 可用且 has cuda 时才返回
        if torch_mod is None:
            return []
        try:
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                count = torch_mod.cuda.device_count()
                return ["cuda"] if count == 1 else [f"cuda:{i}" for i in range(count)]
        except (AttributeError, RuntimeError, OSError):
            # 某些环境（驱动/权限）会抛异常，忽略并返回空
            return []
        return []

    def _detect_mps() -> list[str]:
        # 探测 Apple 平台的 Metal Performance Shaders（MPS）
        if torch_mod is None:
            return []
        try:
            mps = getattr(torch_mod.backends, "mps", None)
            if mps and mps.is_available():
                return ["mps"]
        except (AttributeError, RuntimeError):
            # 非 Apple 平台或旧版本 torch 可能没有 backends.mps
            return []
        return []

    options.extend(_detect_cuda())
    options.extend(_detect_mps())
    return list(dict.fromkeys(options))
