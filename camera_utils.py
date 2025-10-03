"""摄像头友好名与设备枚举工具。

提供两类能力：
- DirectShow 设备名获取（Windows，依赖 pygrabber），用于与 OpenCV 的 DirectShow 后端一致的名称/索引对应；
- 基于 WMI 的保守名称映射（仅在唯一设备时映射，避免顺序错位）。
"""
from __future__ import annotations

from typing import Any

try:
    from pygrabber.dshow_graph import \
        FilterGraph as _DS_FilterGraph  # type: ignore[import-not-found]
except (ImportError, OSError, RuntimeError):  # pragma: no cover - 可选依赖缺失
    _DS_FilterGraph = None  # type: ignore[assignment]


def map_indices_to_names(indices: list[int], wmi_enum: Any | None) -> dict[int, str]:
    """给出 opencv 索引列表，返回 {index: friendly_name} 映射。

    注意：OpenCV 的索引顺序与操作系统(WMI)设备列表顺序通常不一致，
    简单按序对齐会导致“名称与实际索引错位”的问题。

    为避免误导，这里采取保守策略：
    - 当仅有 1 个摄像头时，映射它的名称；
    - 否则返回空映射，让上层显示可靠的 "Camera n"。
    """
    name_map: dict[int, str] = {}
    if not indices or wmi_enum is None:
        return name_map
    try:
        infos = wmi_enum(only_working=True)
        if not infos:
            return name_map
        # 仅当唯一设备时采用名称映射（可避免顺序错位）
        if len(indices) == 1:
            nm = (getattr(infos[0], "name", "") or "").strip()
            if nm:
                name_map[indices[0]] = nm
    except (OSError, RuntimeError) as e:  # pragma: no cover
        print("警告: camera_name 枚举失败, 使用默认名称。原因:", repr(e))
    return name_map


def get_directshow_device_names() -> list[str]:
    """获取 DirectShow 摄像头设备名称列表（Windows）。

    返回与 DirectShow 索引一致的名称列表，即：
    - 索引 0 对应列表第 0 个名称；
    - 索引 1 对应列表第 1 个名称；
    - 以此类推。

    当环境不支持（非 Windows 或未安装 pygrabber）时，返回空列表。
    """
    if _DS_FilterGraph is None:
        return []
    try:
        graph = _DS_FilterGraph()
        devices = graph.get_input_devices() or []
        # 统一为 str 列表并去除多余空白
        return [str(name).strip() for name in devices]
    except (OSError, RuntimeError, AttributeError, ValueError):  # pragma: no cover - 保守兜底
        return []
