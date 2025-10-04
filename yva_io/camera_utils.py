from __future__ import annotations

from typing import Any

try:
    from pygrabber.dshow_graph import FilterGraph as _DS_FilterGraph  # 依赖缺失
except (ImportError, OSError, RuntimeError):  # 可选依赖缺失
    _DS_FilterGraph = None


def map_indices_to_names(indices: list[int], wmi_enum: Any | None) -> dict[int, str]:
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
        print("警告: camera_name 枚举失败, 使用默认名称。原因:", repr(e))
    return name_map


def get_directshow_device_names() -> list[str]:
    if _DS_FilterGraph is None:
        return []
    try:
        graph = _DS_FilterGraph()
        devices = graph.get_input_devices() or []
        return [str(name).strip() for name in devices]
    except (OSError, RuntimeError, AttributeError, ValueError):  # 保险
        return []
