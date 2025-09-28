import argparse
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

import cv2
import torch
from ultralytics import YOLO

from color_detction import detect_traffic_light_color

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

    # 摄像头选择（仅启动时）
    select_camera: bool = field(default_factory=lambda: _as_bool(_env("SELECT_CAMERA", False)))  # 启动时交互选择摄像头
    max_cam_index: int = field(default_factory=lambda: int(_env("MAX_CAM_INDEX", 8)))  # 枚举最大索引（0~N-1）

    # 推理参数
    conf: float = field(default_factory=lambda: float(_env("CONF", 0.5)))
    # 输入尺寸；为空(None) 时表示使用原始帧尺寸而不是模型的默认缩放尺寸
    img_size: Optional[list[int]] = field(default_factory=lambda: _as_optional_int_list(_env("IMG_SIZE", "")))  # 例如: "640" 或 "640,640"

    # 界面与输出细节
    window_name: str = field(default_factory=lambda: _env("WINDOW_NAME", "YOLOv11 Detection"))
    timestamp_fmt: str = field(default_factory=lambda: _env("TIMESTAMP_FMT", "%Y%m%d_%H%M%S"))
    exit_key: str = field(default_factory=lambda: _env("EXIT_KEY", "q"))

    # 额外可选：是否显示 FPS / 统计
    show_fps: bool = field(default_factory=lambda: _as_bool(_env("SHOW_FPS", True)))
    # OpenCV 日志抑制（降低无摄像头时的错误输出噪声）
    quiet_cv: bool = field(default_factory=lambda: _as_bool(_env("QUIET_CV", True)))
    # 枚举摄像头连续失败上限（用于提前终止枚举）
    cam_fail_limit: int = field(default_factory=lambda: int(_env("CAM_FAIL_LIMIT", 3)))

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
    # 摄像头相关（仅启动选择，不再支持运行时切换）
    parser.add_argument("--select-camera", dest="select_camera", action="store_true", help="启动时列出并交互选择可用摄像头")
    parser.add_argument("--max-cam", dest="max_cam_index", type=int, help="枚举最大摄像头索引 (默认 8)")
    parser.add_argument("--conf", dest="conf", type=float, help="置信度阈值 (0~1)")
    parser.add_argument("--img-size", dest="img_size", help="输入尺寸: 例如 640 或 640,640")
    parser.add_argument("--window-name", dest="window_name", help="窗口标题")
    parser.add_argument("--timestamp-fmt", dest="timestamp_fmt", help="时间戳格式 strftime")
    parser.add_argument("--exit-key", dest="exit_key", help="退出按键 (默认 q)")
    parser.add_argument("--no-fps", dest="show_fps", action="store_false", help="关闭 FPS 显示")
    parser.add_argument("--quiet-cv", dest="quiet_cv", action="store_true", help="抑制 OpenCV 摄像头错误日志")
    parser.add_argument("--cam-fail-limit", dest="cam_fail_limit", type=int, help="摄像头枚举连续失败上限 (默认 3)")
    return parser


def load_config_from_args(argv: Optional[list[str]] = None) -> YOLOConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = YOLOConfig()  # 先加载默认+环境

    # 仅覆盖用户传入的非 None 值
    for field_name in [
        "model_path", "device", "source", "save_dir", "save_txt",
        "select_camera", "max_cam_index",
        "conf", "img_size", "window_name", "timestamp_fmt", "exit_key", "show_fps",
        "quiet_cv", "cam_fail_limit"
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

    def detect_and_save(self, stop_event: Optional[Any] = None):
        cfg = self.cfg
        os.makedirs(cfg.save_dir, exist_ok=True)
        # 可选抑制 OpenCV 日志（仅在本进程）
        if cfg.quiet_cv:
            try:  # 兼容不同 OpenCV 版本
                utils_mod = getattr(cv2, 'utils', None)
                logging_mod = getattr(utils_mod, 'logging', None) if utils_mod else None
                set_level = getattr(logging_mod, 'setLogLevel', None) if logging_mod else None
                level_const = getattr(logging_mod, 'LOG_LEVEL_SILENT', None) if logging_mod else None
                if callable(set_level) and level_const is not None:
                    set_level(level_const)
            except Exception:  # pragma: no cover
                pass
        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源： {cfg.source}")

        frame_id = 0
        last_time = datetime.now()
        fps = 0.0

        while True:
            if stop_event is not None and getattr(stop_event, 'is_set', lambda: False)():
                break
            ret, frame = cap.read()
            if not ret:
                # 连续读取失败处理：允许短暂失败（如摄像头热插拔）
                fail = getattr(self, '_read_fail_count', 0) + 1
                self._read_fail_count = fail
                if fail >= 10:
                    print(f"[错误] 连续 {fail} 次无法读取帧，结束检测。")
                    break
                # 小延迟（可选）: 不引入 sleep 以保持响应
                continue
            else:
                if hasattr(self, '_read_fail_count'):
                    self._read_fail_count = 0

            # 推理: 若未指定 img_size 则使用当前帧原始尺寸 (保持分辨率)，否则使用用户设定尺寸
            if cfg.img_size is not None:
                results = self.model.predict(
                    frame,
                    imgsz=cfg.img_size,
                    conf=cfg.conf,
                    device=self.device,
                    verbose=False,
                )
            else:
                h, w = frame.shape[:2]
                # 使用 [h, w] 让 ultralytics 以原始分辨率为目标进行 letterbox（会自动适配 stride）
                results = self.model.predict(
                    frame,
                    imgsz=[h, w],
                    conf=cfg.conf,
                    device=self.device,
                    verbose=False,
                )
            result = results[0]

            annotated_frame = result.plot()

            # 交通灯检测与裁剪分类（COCO 类别 9: traffic light）
            try:
                h_img, w_img = frame.shape[:2]
                for bi, box in enumerate(getattr(result, 'boxes', [])):
                    try:
                        cls_id = int(box.cls.item())
                    except Exception:
                        continue
                    if cls_id != 9:
                        continue
                    # 坐标与边界裁剪
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                    except Exception:
                        continue
                    x1i = max(0, min(w_img - 1, int(x1)))
                    y1i = max(0, min(h_img - 1, int(y1)))
                    x2i = max(0, min(w_img, int(x2)))
                    y2i = max(0, min(h_img, int(y2)))
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    roi = frame[y1i:y2i, x1i:x2i]
                    if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
                        continue
                    # 颜色判别
                    tl_state = detect_traffic_light_color(roi)
                    # 中文标签与颜色
                    state_cn = {
                        'red': '红灯',
                        'yellow': '黄灯',
                        'green': '绿灯',
                        'unknown': '未知',
                    }.get(tl_state, '未知')
                    color_map = {
                        'red': (0, 0, 255),
                        'yellow': (0, 255, 255),
                        'green': (0, 255, 0),
                        'unknown': (255, 255, 255),
                    }
                    cv2.putText(
                        annotated_frame,
                        state_cn,
                        (x1i, max(10, y1i - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color_map.get(tl_state, (255, 255, 255)),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    # 保存裁剪图
                    try:
                        tl_dir = os.path.join(cfg.save_dir, 'traffic_lights')
                        os.makedirs(tl_dir, exist_ok=True)
                        timestamp = datetime.now().strftime(cfg.timestamp_fmt)
                        crop_name = f"tl_{frame_id}_{bi}_{timestamp}_{tl_state}.jpg"
                        cv2.imwrite(os.path.join(tl_dir, crop_name), roi)
                    except Exception:
                        pass
            except Exception:
                # 分类或裁剪过程中出错不影响主流程
                pass

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


def enumerate_cameras(max_index: int = 8) -> list[int]:
    """探测可用摄像头索引。
    通过尝试 0..max_index-1 打开并读取一帧判断是否可用。

    改进:
      1. 可通过环境变量 YV_SUPPRESS_ENUM_ERRORS(=1) 临时抑制 OpenCV 低层枚举错误日志，减少
         obsensor / MSMF 等后端的噪声 (不会影响后续真正运行时的日志等级)。
      2. 遵循 YV_CAM_FAIL_LIMIT 连续失败上限，发现至少 1 个可用摄像头后若连续失败达到上限提前停止。
    """
    available: list[int] = []
    consecutive_fail = 0
    try:
        fail_limit = int(os.getenv(f"{ENV_PREFIX}CAM_FAIL_LIMIT", '3'))
    except ValueError:
        fail_limit = 3

    suppress = os.getenv(f"{ENV_PREFIX}SUPPRESS_ENUM_ERRORS", '1').lower() not in ('0', 'false', '')
    restore_level = None
    set_level = None
    try:
        if suppress:
            utils_mod = getattr(cv2, 'utils', None)
            logging_mod = getattr(utils_mod, 'logging', None) if utils_mod else None
            set_level = getattr(logging_mod, 'setLogLevel', None) if logging_mod else None
            get_level = getattr(logging_mod, 'getLogLevel', None) if logging_mod else None
            silent_const = getattr(logging_mod, 'LOG_LEVEL_SILENT', None)
            if callable(set_level) and silent_const is not None:
                try:
                    if callable(get_level):
                        restore_level = get_level()
                except Exception:
                    restore_level = None
                set_level(silent_const)
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                cap.release()
                consecutive_fail += 1
                if available and consecutive_fail >= fail_limit:
                    break
                continue
            ret, _ = cap.read()
            if ret:
                available.append(i)
                consecutive_fail = 0
            else:
                consecutive_fail += 1
            cap.release()
            if available and consecutive_fail >= fail_limit:
                break
    finally:
        if suppress and callable(set_level) and restore_level is not None:
            try:
                set_level(restore_level)
            except Exception:
                pass
    return available


def interactive_select_camera(max_index: int) -> int:
    cams = enumerate_cameras(max_index)
    if not cams:
        raise RuntimeError("未检测到任何可用摄像头。")
    print("可用摄像头: " + ", ".join(str(c) for c in cams))
    while True:
        sel = input("请输入要使用的摄像头索引(回车默认0): ").strip()
        if sel == "":
            sel = "0"
        if sel.isdigit():
            idx = int(sel)
            if idx in cams:
                print(f"[信息] 选择摄像头 {idx}")
                return idx
        print("输入无效，请重新输入。")


def main(argv: Optional[list[str]] = None):
    """供外部脚本调用的主入口，不再在本文件内自动执行。"""
    cfg = load_config_from_args(argv)
    # 交互式摄像头选择
    if cfg.select_camera:
        try:
            cfg.source = interactive_select_camera(cfg.max_cam_index)
        except Exception as e:
            print(f"[错误] 摄像头选择失败: {e}")
            return
    print("[配置] 使用参数: ")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    detector = YOLODetector(cfg)
    detector.detect_and_save()





