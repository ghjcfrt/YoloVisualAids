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

import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional

try:  # 允许在非 Windows 或未安装 pywin32 时被 import 而不立刻崩溃
    import win32com.client  # type: ignore
except Exception:  # pragma: no cover
    win32com = None  # type: ignore

KEYWORD_PATTERN = re.compile(r"(camera|webcam|uvc|video)", re.IGNORECASE)


@dataclass
class CameraDeviceInfo:
    index: int                 # 逻辑排序索引(非系统/非 OpenCV index，仅枚举顺序)
    name: str                  # 设备名称（保证非 None，取不到时为空串）
    pnp_class: Optional[str]   # PNPClass (Camera / Image / 其它)
    status: Optional[str]      # 设备状态（OK 等）
    device_id: Optional[str]   # PNP DeviceID
    description: Optional[str] # Description 字段（可能为空）
    source: str                # 'PNPClass' or 'KeywordFallback'

    def to_dict(self):
        return asdict(self)


class CameraEnumError(RuntimeError):
    pass


def _connect_wmi():
    if win32com is None:
        raise CameraEnumError("win32com 未安装，无法执行 WMI 查询。请安装 pywin32 或在 Windows 环境下运行。")
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


def enumerate_cameras(only_working: bool = True, verbose: bool | None = None) -> List[CameraDeviceInfo]:
    """返回摄像头设备列表。

    Parameters
    ----------
    only_working : bool
        是否仅保留状态为 OK/Working 的设备。
    verbose : Optional[bool]
        详细输出；若为 None，则读取环境变量 CAM_VERBOSE。
    """
    if verbose is None:
        verbose = os.environ.get("CAM_VERBOSE", "0") not in ("0", "false", "False", "")
    devices: List[CameraDeviceInfo] = []
    try:
        primary = list(_primary_query())
    except Exception as e:  # pragma: no cover
        if verbose:
            print("[camera_devices] Primary WMI 查询失败:", repr(e), file=sys.stderr)
        primary = []
    if primary:
        if verbose:
            print(f"[camera_devices] Primary 匹配 {len(primary)} 个设备")
        for idx, d in enumerate(primary):
            devices.append(_to_info(idx, d, source='PNPClass'))
    else:
        # 关键词回退
        fallback = []
        try:
            fallback = list(_fallback_query())
        except Exception as e:  # pragma: no cover
            if verbose:
                print("[camera_devices] Fallback WMI 查询失败:", repr(e), file=sys.stderr)
        for d in fallback:
            name = getattr(d, 'Name', '') or ''
            if name and KEYWORD_PATTERN.search(name):
                devices.append(_to_info(len(devices), d, source='KeywordFallback'))
        if verbose:
            print(f"[camera_devices] Keyword 回退匹配 {len(devices)} 个设备")
    if only_working:
        filtered = []
        for dev in devices:
            st = (dev.status or '').upper()
            if (not st) or st in ("OK",) or "WORKING" in st:
                filtered.append(dev)
        devices = filtered
        if verbose:
            print(f"[camera_devices] 过滤后剩余 {len(devices)} 个设备")
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


def print_cameras(verbose: bool | None = None, only_working: bool = True) -> int:
    cams = enumerate_cameras(only_working=only_working, verbose=verbose)
    if not cams:
        print("(无摄像头设备)")
        return 1
    for c in cams:
        print(f"{c.index}: {c.name} | {c.pnp_class} | {c.status} | {c.device_id} | {c.source}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:  # CLI 支持
    import argparse
    parser = argparse.ArgumentParser(description="列出本机摄像头设备 (WMI)")
    parser.add_argument('--all', action='store_true', help='包含状态非 OK 的设备')
    parser.add_argument('--verbose', action='store_true', help='输出调试信息')
    args = parser.parse_args(argv)
    return print_cameras(verbose=args.verbose, only_working=not args.all)


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
