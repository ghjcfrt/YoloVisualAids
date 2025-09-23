import argparse
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

import cv2
import torch
from ultralytics import YOLO

#############################################
# 配置区
# 统一管理所有可调参数，支持：
# 1. 代码内默认值
# 2. 环境变量覆盖（前缀 YV_）
# 3. 命令行参数覆盖
#############################################


ENV_PREFIX = "YV_"  # 环境变量前缀，例如 YV_MODEL_PATH


def _env(name: str, default: Any) -> Any:
    """读取环境变量（若存在则覆盖默认值）。"""
    return os.getenv(f"{ENV_PREFIX}{name}", default)


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_optional_int_list(val: str | None) -> Optional[list[int]]:
    if val in (None, "", "none", "null"):  # 允许不设置
        return None
    parts = [p.strip() for p in val.split(',') if p.strip()]
    out: list[int] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"IMG_SIZE 环境变量/参数包含非整数: {p}")
        out.append(int(p))
    return out


@dataclass
class YOLOConfig:
    # 模型与设备
    model_path: str = field(default_factory=lambda: _env("MODEL_PATH", "yolo11n.pt"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))

    # 运行输入输出
    source: Union[int, str] = field(default_factory=lambda: _parse_source(_env("SOURCE", "0")))
    save_dir: str = field(default_factory=lambda: _env("SAVE_DIR", "results"))
    save_txt: bool = field(default_factory=lambda: _as_bool(_env("SAVE_TXT", False)))

    # 推理参数
    conf: float = field(default_factory=lambda: float(_env("CONF", 0.5)))
    img_size: Optional[list[int]] = field(default_factory=lambda: _as_optional_int_list(_env("IMG_SIZE", "")))  # 例如: "640" 或 "640,640"

    # 界面与输出细节
    window_name: str = field(default_factory=lambda: _env("WINDOW_NAME", "YOLOv11 Detection"))
    timestamp_fmt: str = field(default_factory=lambda: _env("TIMESTAMP_FMT", "%Y%m%d_%H%M%S"))
    exit_key: str = field(default_factory=lambda: _env("EXIT_KEY", "q"))

    # 额外可选：是否显示 FPS / 统计
    show_fps: bool = field(default_factory=lambda: _as_bool(_env("SHOW_FPS", True)))

    def to_dict(self):  # 便于调试打印
        return asdict(self)


def _parse_source(raw: str) -> Union[int, str]:
    # 纯数字且长度<6 认为是摄像头索引
    if raw.isdigit() and len(raw) < 6:
        return int(raw)
    return raw


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv11 实时检测配置参数")
    parser.add_argument("--model", dest="model_path", help="模型权重路径 (默认: yolo11n.pt)")
    parser.add_argument("--device", dest="device", help="运行设备: cuda / cpu / mps / auto")
    parser.add_argument("--source", dest="source", help="视频源: 摄像头索引或视频文件路径")
    parser.add_argument("--save-dir", dest="save_dir", help="结果保存目录")
    parser.add_argument("--save-txt", dest="save_txt", action="store_true", help="保存 YOLO txt 标注文件")
    parser.add_argument("--conf", dest="conf", type=float, help="置信度阈值 (0~1)")
    parser.add_argument("--img-size", dest="img_size", help="输入尺寸: 例如 640 或 640,640")
    parser.add_argument("--window-name", dest="window_name", help="窗口标题")
    parser.add_argument("--timestamp-fmt", dest="timestamp_fmt", help="时间戳格式 strftime")
    parser.add_argument("--exit-key", dest="exit_key", help="退出按键 (默认 q)")
    parser.add_argument("--no-fps", dest="show_fps", action="store_false", help="关闭 FPS 显示")
    return parser


def load_config_from_args(argv: Optional[list[str]] = None) -> YOLOConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = YOLOConfig()  # 先加载默认+环境

    # 仅覆盖用户传入的非 None 值
    for field_name in [
        "model_path", "device", "source", "save_dir", "save_txt", "conf", "img_size",
        "window_name", "timestamp_fmt", "exit_key", "show_fps"
    ]:
        val = getattr(args, field_name, None)
        if val is not None:
            if field_name == "source":
                val = _parse_source(val)
            elif field_name == "img_size" and isinstance(val, str):
                val = _as_optional_int_list(val)
            setattr(cfg, field_name, val)
    return cfg


def _select_device(requested: str | None = None) -> str:
    """自动选择设备（如果要求），或“自动”。
    优先选择 CUDA，其次 MPS（Apple），否则使用 CPU。"""
    if requested and requested.lower() not in {"auto", ""}:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class YOLODetector:
    def __init__(self, cfg: YOLOConfig):
        self.cfg = cfg
        self.device = _select_device(cfg.device)
        self.model: YOLO = YOLO(cfg.model_path)

    def detect_and_save(self):
        cfg = self.cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源： {cfg.source}")

        frame_id = 0
        last_time = datetime.now()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 推理
            if cfg.img_size is not None:
                results = self.model.predict(frame, imgsz=cfg.img_size, conf=cfg.conf, device=self.device, verbose=False)
            else:
                results = self.model.predict(frame, conf=cfg.conf, device=self.device, verbose=False)
            result = results[0]

            annotated_frame = result.plot()

            # FPS 显示
            if cfg.show_fps:
                now = datetime.now()
                dt = (now - last_time).total_seconds()
                last_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(cfg.window_name, annotated_frame)

            timestamp = datetime.now().strftime(cfg.timestamp_fmt)
            base_name = f'frame_{frame_id}_{timestamp}'
            img_path = os.path.join(cfg.save_dir, base_name + '.jpg')
            cv2.imwrite(img_path, annotated_frame)

            if cfg.save_txt:
                txt_path = os.path.join(cfg.save_dir, base_name + '.txt')
                h, w = result.orig_shape
                with open(txt_path, 'w', encoding='utf-8') as f:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x_c = (x1 + x2) / 2 / w
                        y_c = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord(cfg.exit_key):
                break

        cap.release()
        cv2.destroyAllWindows()


def main(argv: Optional[list[str]] = None):
    """供外部脚本调用的主入口，不再在本文件内自动执行。"""
    cfg = load_config_from_args(argv)
    print("[配置] 使用参数: ")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    detector = YOLODetector(cfg)
    detector.detect_and_save()


