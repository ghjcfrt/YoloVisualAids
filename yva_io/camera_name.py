from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型检查时导入，避免运行时开销
    from collections.abc import Iterable

try:  # Windows 专用，可选依赖
    import win32com.client
except ImportError:  # 非 Windows / 未安装 pywin32
    win32com = None

# 可选导入 COM 错误类型，以便更精确地捕获异常
try:  # 非 Windows ,环境可能缺失
    from pywintypes import com_error as _ComError
except ImportError:
    _ComError = None

KEYWORD_PATTERN = re.compile(r"(camera|webcam|uvc|video)", re.IGNORECASE)


@dataclass
class CameraDeviceInfo:
    index: int
    name: str
    pnp_class: str | None
    status: str | None
    device_id: str | None
    description: str | None
    source: str

    def to_dict(self):
        return asdict(self)


class CameraEnumError(RuntimeError):
    pass


def _connect_wmi():
    if win32com is None:
        msg = "win32com 未安装，无法执行 WMI 查询，请在 Windows 安装 pywin32"
        raise CameraEnumError(msg)
    locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
    return locator.ConnectServer('.', 'root\\cimv2')


def _exec_query(svc, query: str):
    return svc.ExecQuery(query)


def _primary_query() -> Iterable:
    svc = _connect_wmi()
    return _exec_query(svc, (
        "SELECT DeviceID, Name, Description, Status, PNPClass FROM Win32_PnPEntity "
        "WHERE (PNPClass='Camera' OR PNPClass='Image')"
    ))


def _fallback_query() -> Iterable:
    svc = _connect_wmi()
    return _exec_query(svc, "SELECT DeviceID, Name, Description, Status, PNPClass FROM Win32_PnPEntity")


def _env_verbose(*, verbose: bool | None) -> bool:
    if verbose is not None:
        return verbose
    return os.environ.get("CAM_VERBOSE", "0").lower() not in {"0", "false", ""}


def _wmi_error_types() -> tuple[type[BaseException], ...]:
    errs: tuple[type[BaseException], ...] = (OSError, AttributeError, RuntimeError)
    if _ComError is not None:
        errs = (*errs, _ComError)  # type: ignore[misc]
    return errs


def _to_info(idx: int, d, source: str) -> CameraDeviceInfo:
    name = getattr(d, 'Name', '') or ''
    return CameraDeviceInfo(
        index=idx,
        name=name,
        pnp_class=getattr(d, 'PNPClass', None),
        status=getattr(d, 'Status', None),
        device_id=getattr(d, 'DeviceID', None),
        description=getattr(d, 'Description', None),
        source=source,
    )


def _query_primary_devices(wmi_errs: tuple[type[BaseException], ...], *, verbose: bool) -> list[CameraDeviceInfo]:
    try:
        primary = list(_primary_query())
    except wmi_errs as e:
        if verbose:
            print("[camera_name] Primary WMI 查询失败:", repr(e))
        return []
    if not primary:
        return []
    if verbose:
        print(f"[camera_name] Primary 匹配 {len(primary)} 个设备")
    return [_to_info(i, d, source="PNPClass") for i, d in enumerate(primary)]


def _query_keyword_fallback_devices(
    wmi_errs: tuple[type[BaseException], ...], *, verbose: bool
) -> list[CameraDeviceInfo]:
    fallback = []
    try:
        fallback = list(_fallback_query())
    except wmi_errs as e:
        if verbose:
            print("[camera_name] Fallback WMI 查询失败:", repr(e))
        return []
    devices: list[CameraDeviceInfo] = []
    for d in fallback:
        name = getattr(d, "Name", "") or ""
        if name and KEYWORD_PATTERN.search(name):
            devices.append(_to_info(len(devices), d, source="KeywordFallback"))
    if verbose:
        print(f"[camera_name] Keyword 回退匹配 {len(devices)} 个设备")
    return devices


def _filter_only_working(devices: list[CameraDeviceInfo], *, verbose: bool) -> list[CameraDeviceInfo]:
    filtered: list[CameraDeviceInfo] = []
    for dev in devices:
        st = (dev.status or "").upper()
        if (not st) or (st == "OK") or ("WORKING" in st):
            filtered.append(dev)
    if verbose:
        print(f"[camera_name] 过滤后剩余 {len(filtered)} 个设备")
    return filtered


def enumerate_cameras(*, only_working: bool = True, verbose: bool | None = None) -> list[CameraDeviceInfo]:
    """返回摄像头设备列表（Windows / 需 pywin32）

    若依赖不可用或非 Windows，返回空列表
    """
    if win32com is None:
        return []
    vb = _env_verbose(verbose=verbose)
    wmi_errs = _wmi_error_types()

    devices = _query_primary_devices(wmi_errs, verbose=vb)
    if not devices:
        devices = _query_keyword_fallback_devices(wmi_errs, verbose=vb)
    if only_working:
        devices = _filter_only_working(devices, verbose=vb)
    return devices


__all__ = ["CameraDeviceInfo", "enumerate_cameras"]
