"""命令行检测入口封装"""

from __future__ import annotations

from .core import main as _core_main


def main(argv: list[str] | None = None) -> None:
    _core_main(argv)


if __name__ == "__main__":
    main()
