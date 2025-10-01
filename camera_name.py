"""camera_devices
---------------------------------
提供 Windows 下摄像头(相机)设备的枚举函数，支持：
1. 通过 WMI 使用 PNPClass=Camera / Image 精准筛选。
2. 若无结果，回退关键词匹配 (camera/webcam/uvc/video)。
3. 可过滤非工作状态设备。
4. 可作为库函数被 GUI 或其他脚本调用。

使用示例:
    from camera_devices import enumerate_cameras, CameraDeviceInfo
    cams = enumerate_cameras(only_working=True)
    for c in cams:
        print(c.index, c.name)

命令行快速查看:
    python -m camera_devices --verbose
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型检查时导入，避免运行时开销
    from collections.abc import Iterable

try:  # 允许在非 Windows 或未安装 pywin32 时被 import 而不立刻崩溃
    import win32com.client
except ImportError:  # pragma: no cover
    win32com = None

# 可选导入 COM 错误类型，以便更精确地捕获异常
_ComError: type[BaseException] | None
try:  # pragma: no cover - 非 Windows 环境可能缺失
    from pywintypes import com_error as _ComError
except ImportError:  # pragma: no cover
    _ComError = None
KEYWORD_PATTERN = re.compile(r"(camera|webcam|uvc|video)", re.IGNORECASE)


@dataclass
class CameraDeviceInfo:
    index: int                 # 逻辑排序索引(非系统/非 OpenCV index，仅枚举顺序)
    name: str                  # 设备名称（保证非 None，取不到时为空串）
    pnp_class: str | None      # PNPClass (Camera / Image / 其它)
    status: str | None         # 设备状态（OK 等）
    device_id: str | None      # PNP DeviceID
    description: str | None    # Description 字段（可能为空）
    source: str                # 'PNPClass' or 'KeywordFallback'

    def to_dict(self):
        return asdict(self)


class CameraEnumError(RuntimeError):
    pass


def _connect_wmi():
    if win32com is None:
        msg = "win32com 未安装，无法执行 WMI 查询。请安装 pywin32 或在 Windows 环境下运行。"
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
    return os.environ.get("CAM_VERBOSE", "0") not in {"0", "false", "False", ""}


def _wmi_error_types() -> tuple[type[BaseException], ...]:
    errs: tuple[type[BaseException], ...] = (OSError, AttributeError, RuntimeError)
    if _ComError is not None:
        errs = (*errs, _ComError)
    return errs


def _query_primary_devices(wmi_errs: tuple[type[BaseException], ...], *, verbose: bool) -> list[CameraDeviceInfo]:
    try:
        primary = list(_primary_query())
    except wmi_errs as e:  # pragma: no cover
        if verbose:
            print("[camera_devices] Primary WMI 查询失败:", repr(e), file=sys.stderr)
        return []
    if not primary:
        return []
    if verbose:
        print(f"[camera_devices] Primary 匹配 {len(primary)} 个设备")
    return [_to_info(i, d, source="PNPClass") for i, d in enumerate(primary)]


def _query_keyword_fallback_devices(
    wmi_errs: tuple[type[BaseException], ...], *, verbose: bool
) -> list[CameraDeviceInfo]:
    fallback = []
    try:
        fallback = list(_fallback_query())
    except wmi_errs as e:  # pragma: no cover
        if verbose:
            print("[camera_devices] Fallback WMI 查询失败:", repr(e), file=sys.stderr)
        return []
    devices: list[CameraDeviceInfo] = []
    for d in fallback:
        name = getattr(d, "Name", "") or ""
        if name and KEYWORD_PATTERN.search(name):
            devices.append(_to_info(len(devices), d, source="KeywordFallback"))
    if verbose:
        print(f"[camera_devices] Keyword 回退匹配 {len(devices)} 个设备")
    return devices


def _filter_only_working(devices: list[CameraDeviceInfo], *, verbose: bool) -> list[CameraDeviceInfo]:
    filtered: list[CameraDeviceInfo] = []
    for dev in devices:
        st = (dev.status or "").upper()
        if (not st) or (st == "OK") or ("WORKING" in st):
            filtered.append(dev)
    if verbose:
        print(f"[camera_devices] 过滤后剩余 {len(filtered)} 个设备")
    return filtered


def enumerate_cameras(*, only_working: bool = True, verbose: bool | None = None) -> list[CameraDeviceInfo]:
    """返回摄像头设备列表。

    参数：
    ----------
    only_working : bool
        是否仅保留状态为 OK/Working 的设备。
    verbose : Optional[bool]
        详细输出；若为 None，则读取环境变量 CAM_VERBOSE。
    """
    vb = _env_verbose(verbose=verbose)
    wmi_errs = _wmi_error_types()

    devices = _query_primary_devices(wmi_errs, verbose=vb)
    if not devices:
        devices = _query_keyword_fallback_devices(wmi_errs, verbose=vb)
    if only_working:
        devices = _filter_only_working(devices, verbose=vb)
    return devices


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


def print_cameras(*, verbose: bool | None = None, only_working: bool = True) -> int:
    cams = enumerate_cameras(only_working=only_working, verbose=verbose)
    if not cams:
        print("(无摄像头设备)")
        return 1
    for c in cams:
        print(f"{c.index}: {c.name} | {c.pnp_class} | {c.status} | {c.device_id} | {c.source}")
    return 0


def main(argv: list[str] | None = None) -> int:  # CLI 支持
    parser = argparse.ArgumentParser(description="列出本机摄像头设备 (WMI)")
    parser.add_argument('--all', action='store_true', help='包含状态非 OK 的设备')
    parser.add_argument('--verbose', action='store_true', help='输出调试信息')
    args = parser.parse_args(argv)
    return print_cameras(verbose=args.verbose, only_working=not args.all)


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
