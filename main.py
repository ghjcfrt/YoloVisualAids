"""Convenience entry point for YoloVisualAids.

Usage:
  - python main.py                 # launch GUI (default)
  - python main.py gui             # launch GUI
  - python main.py detect [args]   # run detection CLI
  - python main.py traffic [args]  # run traffic-light CLI

This is a thin wrapper that forwards to package modules, so the canonical
entries remain:
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
    gui_main()


def _run_detect(argv: list[str]) -> None:
    detect_main(argv)


def _run_traffic(argv: list[str]) -> None:
    traffic_main(argv)


def _print_usage() -> None:
    print(
        "Usage:\n"
        "  python main.py                 # launch GUI (default)\n"
        "  python main.py gui             # launch GUI\n"
        "  python main.py detect [args]   # run detection CLI\n"
        "  python main.py traffic [args]  # run traffic-light CLI\n",
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
