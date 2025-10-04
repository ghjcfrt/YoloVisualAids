"""项目便捷入口

用法
    - python main.py                  启动 GUI
    - python main.py gui             启动 GUI
    - python main.py detect [args]   运行检测 CLI
    - python main.py traffic [args]  运行交通灯 CLI

此文件仅做转发 规范入口建议使用模块方式
    - python -m yolovisualaids.app
    - python -m yolovisualaids.detection
    - python -m yolovisualaids.vision.traffic_cli
"""
from __future__ import annotations

import sys

from app import main as gui_main
from detection.cli import main as detect_main
from vision.traffic_cli import main as traffic_main


def _run_gui() -> None:
    """启动 GUI"""
    gui_main()


def _run_detect(argv: list[str]) -> None:
    """运行检测 CLI"""
    detect_main(argv)


def _run_traffic(argv: list[str]) -> None:
    """运行交通灯 CLI"""
    traffic_main(argv)


def _print_usage() -> None:
    """打印用法说明"""
    print(
        "用法:\n"
        "  python main.py                 # 启动 GUI（默认）\n"
        "  python main.py gui             # 启动 GUI\n"
        "  python main.py detect [参数]   # 运行检测 CLI\n"
        "  python main.py traffic [参数]  # 运行交通灯 CLI\n",
        end="",
    )


def main() -> None:
    if len(sys.argv) == 1:
        _run_gui()
        return

    cmd = (sys.argv[1] or "").strip().lower()
    rest = sys.argv[2:]

    if cmd == "gui":
        _run_gui()
        return
    if cmd in {"detect", "det", "yolo"}:
        _run_detect(rest)
        return
    if cmd in {"traffic", "tl"}:
        _run_traffic(rest)
        return

    _print_usage()
    sys.exit(2)


if __name__ == "__main__":
    main()
