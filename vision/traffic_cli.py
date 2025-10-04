from __future__ import annotations

import argparse

from . import traffic_mode as _traffic_mode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="交通灯颜色测试（图片/目录/摄像头 + 手动/自动 ROI）")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="单张图片路径")
    src.add_argument("--dir", help="图片目录")
    src.add_argument("--cam", type=int, help="摄像头索引")

    p.add_argument("--win", default="traffic-test", help="窗口标题")
    p.add_argument("--delay", type=int, default=1, help="摄像头显示延迟 ms（>=1）")

    # 手动 ROI
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="ROI: X Y W H，不传则交互选择")

    # 自动 ROI（YOLO）
    p.add_argument("--auto", action="store_true", help="启用 YOLO 自动检测红绿灯并裁剪 ROI")
    p.add_argument("--model", default="models/yolo/yolo11n.pt", help="YOLO 模型权重")
    p.add_argument("--conf", type=float, default=0.5, help="置信度阈值 (0~1)")
    p.add_argument("--img-size", default="", help="输入尺寸，如 640 或 640,640；留空表示原始尺寸")
    p.add_argument("--device", default="auto", help="设备: auto/cuda/cpu/mps")
    p.add_argument("--class-id", type=int, default=9, help="COCO 类别 id（默认 9: traffic light）")
    p.add_argument("--first", action="store_true", help="仅取 1 个最优框（按中心距+置信度）")
    p.add_argument("--save-crops", help="将 ROI 裁剪保存到该目录")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _traffic_mode.run(args)


if __name__ == "__main__":
    main()
