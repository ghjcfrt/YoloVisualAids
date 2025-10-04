"""摄像头名称辅助

封装两类可选的摄像头名称来源：
1) Windows DirectShow（pygrabber）- get_directshow_device_names
2) Windows WMI（camera_name.enumerate_cameras）- map_indices_to_names

上述依赖均为可选：当缺失时，本模块返回空结果或者仅按索引映射。
"""

from __future__ import annotations

from typing import Any

try:
    from pygrabber.dshow_graph import FilterGraph as _DS_FilterGraph  # 依赖缺失
except (ImportError, OSError, RuntimeError):  # 可选依赖缺失
    _DS_FilterGraph = None


def map_indices_to_names(indices: list[int], wmi_enum: Any | None) -> dict[int, str]:
    """将摄像头索引映射为友好名称（基于可选的 WMI 枚举）。

    参数
    - indices: 需要映射的摄像头索引列表
    - wmi_enum: yva_io.camera_name.enumerate_cameras 函数或兼容实现；
      传入 None 时直接返回空映射。

    返回
    - {index: name} 的字典，仅在能获取到名称时填充（例如只有 1 个索引时取首个）。
    """
    name_map: dict[int, str] = {}
    if not indices or wmi_enum is None:
        return name_map
    try:
        infos = wmi_enum(only_working=True)
        if not infos:
            return name_map
        if len(indices) == 1:
            nm = (getattr(infos[0], "name", "") or "").strip()
            if nm:
                name_map[indices[0]] = nm
    except (OSError, RuntimeError) as e:
        print("警告: camera_name 枚举失败 使用默认名称 原因:", repr(e))
        return name_map
    return name_map


def get_directshow_device_names() -> list[str]:
    """返回 Windows DirectShow 输入设备名称列表。

    要求已安装 pygrabber；若依赖缺失或调用失败则返回空列表。
    """
    if _DS_FilterGraph is None:
        return []
    try:
        graph = _DS_FilterGraph()
        devices = graph.get_input_devices() or []
        return [str(name).strip() for name in devices]
    except (OSError, RuntimeError, AttributeError, ValueError):  # 保险
        return []
