"""YOLO 检测核心实现

"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO

from vision import (TL_COLOR_MAP, TL_STATE_CN, decide_traffic_status,
                    detect_traffic_light_color)
from voice import Announcer

ENV_PREFIX = "YV_"  # 环境变量前缀，例如 YV_MODEL_PATH

# 常量
TRAFFIC_LIGHT_CLASS_ID = 9
MIN_ROI_SIDE = 8
READ_FAIL_LIMIT = 10
LABEL_Y_OFFSET = 6
TEXT_MARGIN_MIN = 10
MAX_INDEX_DIGITS = 6


def _env(name: str, default: Any) -> Any:
    """读取环境变量（若存在则覆盖默认值）"""
    return os.getenv(f"{ENV_PREFIX}{name}", default)


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_optional_int_list(val: str | None) -> list[int] | None:
    if val is None:
        return None
    low = val.strip().lower()
    if low in {"", "none", "null"}:  # 允许不设置
        return None
    parts = [p.strip() for p in val.split(',') if p.strip()]
    out: list[int] = []
    for p in parts:
        if not p.isdigit():
            msg = f"IMG_SIZE 环境变量/参数包含非整数: {p}"
            raise ValueError(msg)
        out.append(int(p))
    return out


@dataclass
class YOLOConfig:
    # 模型与设备
    model_path: str = field(default_factory=lambda: _env("MODEL_PATH", "models/yolo/yolo11n.pt"))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))

    # 运行输入输出
    source: int | str = field(default_factory=lambda: _parse_source(_env("SOURCE", "0")))
    save_dir: str = field(default_factory=lambda: _env("SAVE_DIR", "results"))
    save_txt: bool = field(default_factory=lambda: _as_bool(_env("SAVE_TXT", default=False)))

    # 摄像头选择（仅启动时）
    select_camera: bool = field(default_factory=lambda: _as_bool(_env("SELECT_CAMERA", default=False)))
    max_cam_index: int = field(default_factory=lambda: int(_env("MAX_CAM_INDEX", 8)))

    # 推理参数
    conf: float = field(default_factory=lambda: float(_env("CONF", 0.5)))
    # 输入尺寸；为空(None) 时表示使用原始帧尺寸而不是模型的默认缩放尺寸
    img_size: list[int] | None = field(default_factory=lambda: _as_optional_int_list(_env("IMG_SIZE", "")))

    # 界面与输出细节
    window_name: str = field(default_factory=lambda: _env("WINDOW_NAME", "YOLOv11 Detection"))
    timestamp_fmt: str = field(default_factory=lambda: _env("TIMESTAMP_FMT", "%Y%m%d_%H%M%S"))
    exit_key: str = field(default_factory=lambda: _env("EXIT_KEY", "q"))

    # 额外可选：是否显示 FPS / 统计
    show_fps: bool = field(default_factory=lambda: _as_bool(_env("SHOW_FPS", default=True)))
    # OpenCV 日志抑制（降低无摄像头时的错误输出噪声）
    quiet_cv: bool = field(default_factory=lambda: _as_bool(_env("QUIET_CV", default=True)))
    # 枚举摄像头连续失败上限（用于提前终止枚举）
    cam_fail_limit: int = field(default_factory=lambda: int(_env("CAM_FAIL_LIMIT", 3)))

    # 播报/节流/黄闪参数
    ann_min_interval: float = field(default_factory=lambda: float(_env("ANN_MIN_INTERVAL", 1.5)))
    ann_flash_window: float = field(default_factory=lambda: float(_env("ANN_FLASH_WINDOW", 3.0)))
    ann_flash_min_events: int = field(default_factory=lambda: int(_env("ANN_FLASH_MIN_EVENTS", 6)))
    ann_flash_yellow_ratio: float = field(default_factory=lambda: float(_env("ANN_FLASH_YELLOW_RATIO", 0.9)))
    ann_flash_cooldown: float = field(default_factory=lambda: float(_env("ANN_FLASH_COOLDOWN", 5.0)))

    def to_dict(self):  # 便于调试打印
        return asdict(self)


def _parse_source(raw: str) -> int | str:
    # 纯数字且长度<6 认为是摄像头索引
    if raw.isdigit() and len(raw) < MAX_INDEX_DIGITS:
        return int(raw)
    return raw


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv11 实时检测配置参数")
    parser.add_argument("--model", dest="model_path", help="模型权重路径 (默认: models/yolo/yolo11n.pt)")
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
    # 播报/节流/黄闪参数
    parser.add_argument("--ann-min-interval", dest="ann_min_interval", type=float, help="同句最小播报间隔(秒)")
    parser.add_argument("--ann-flash-window", dest="ann_flash_window", type=float, help="黄闪判定时间窗口(秒)")
    parser.add_argument("--ann-flash-min-events", dest="ann_flash_min_events", type=int, help="黄闪判定最少采样数")
    parser.add_argument("--ann-flash-yellow-ratio", dest="ann_flash_yellow_ratio", type=float, help="黄灯占比阈值(0~1)")
    parser.add_argument("--ann-flash-cooldown", dest="ann_flash_cooldown", type=float, help="黄闪播报冷却时间(秒)")
    return parser


def load_config_from_args(argv: list[str] | None = None) -> YOLOConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = YOLOConfig()  # 先加载默认+环境

    # 仅覆盖用户传入的非 None 值
    for field_name in [
        "model_path",
        "device",
        "source",
        "save_dir",
        "save_txt",
        "select_camera",
        "max_cam_index",
        "conf",
        "img_size",
        "window_name",
        "timestamp_fmt",
        "exit_key",
        "show_fps",
        "quiet_cv",
        "cam_fail_limit",
        "ann_min_interval",
        "ann_flash_window",
        "ann_flash_min_events",
        "ann_flash_yellow_ratio",
        "ann_flash_cooldown",
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
    """自动选择设备（如果要求），或“自动”； 优先选择 CUDA，其次 MPS（Apple），否则使用 CPU"""
    if requested and requested.lower() not in {"auto", ""}:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YOLODetector:
    def __init__(self, cfg: YOLOConfig):
        self.cfg = cfg
        self.device = _select_device(cfg.device)
        self.model: YOLO = YOLO(cfg.model_path)
        # FPS 相关状态
        self._last_time = datetime.now(UTC)
        self._fps = 0.0
        # TTS 播报器（可配置：去重/限流/黄闪参数）
        self._ann = Announcer(
            min_interval_sec=self.cfg.ann_min_interval,
            flash_window_sec=self.cfg.ann_flash_window,
            flash_min_events=self.cfg.ann_flash_min_events,
            flash_yellow_ratio=self.cfg.ann_flash_yellow_ratio,
            flash_cooldown_sec=self.cfg.ann_flash_cooldown,
        )

    @staticmethod
    def _should_stop(stop_event: Any | None) -> bool:
        return bool(stop_event is not None and getattr(stop_event, "is_set", lambda: False)())

    def _inc_read_fail_and_should_break(self) -> bool:
        fail = getattr(self, "_read_fail_count", 0) + 1
        self._read_fail_count = fail
        if fail >= READ_FAIL_LIMIT:
            print(f"[错误] 连续 {fail} 次无法读取帧，结束检测")
            return True
        return False

    def _reset_read_fail(self) -> None:
        if hasattr(self, "_read_fail_count"):
            self._read_fail_count = 0

    def _say_non_tl_counts(self, result) -> None:
        counts: dict[int, int] = {}
        for b in getattr(result, "boxes", []):
            try:
                cid = int(b.cls.item())
            except (AttributeError, ValueError, TypeError):
                continue
            if cid == TRAFFIC_LIGHT_CLASS_ID:
                continue
            counts[cid] = counts.get(cid, 0) + 1
        if counts:
            self._ann.say_non_tl(counts)

    def _process_frame(self, frame, frame_id: int):
        result, annotated_frame = self._predict(frame)
        self._annotate_traffic_lights(frame, result, annotated_frame, frame_id)
        self._say_non_tl_counts(result)
        self._update_and_draw_fps(annotated_frame)
        cv2.imshow(self.cfg.window_name, annotated_frame)
        self._save_result(frame_id, annotated_frame, result)
        return annotated_frame

    def _quiet_opencv_logs(self) -> None:
        """按需抑制 OpenCV 日志"""
        if not self.cfg.quiet_cv:
            return
        utils_mod = getattr(cv2, "utils", None)
        logging_mod = getattr(utils_mod, "logging", None) if utils_mod else None
        set_level = getattr(logging_mod, "setLogLevel", None) if logging_mod else None
        level_const = getattr(logging_mod, "LOG_LEVEL_SILENT", None) if logging_mod else None
        if callable(set_level) and level_const is not None:
            try:
                set_level(level_const)
            except (cv2.error, RuntimeError) as err:  # 记录一次即可
                print(f"[警告] 设置 OpenCV 日志等级失败: {err}")

    def _predict(self, frame) -> tuple[Any, Any]:
        """执行模型推理，返回 (result, annotated_frame)"""
        cfg = self.cfg
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
            results = self.model.predict(
                frame,
                imgsz=[h, w],
                conf=cfg.conf,
                device=self.device,
                verbose=False,
            )
        result = results[0]
        annotated_frame = result.plot()
        return result, annotated_frame

    def _annotate_traffic_lights(self, frame, result, annotated_frame, frame_id: int) -> None:
        """检测交通灯并标注、裁剪保存"""
        h_img, w_img = frame.shape[:2]
        tl_boxes: list[tuple[int, int, int, int, float]] = []
        for bi, box in enumerate(getattr(result, "boxes", [])):
            try:
                cls_id = int(box.cls.item())
            except (AttributeError, ValueError, TypeError) as err:
                print(f"[警告] 无法解析类别: {err}")
                continue
            if cls_id != TRAFFIC_LIGHT_CLASS_ID:
                continue
            try:
                coords = box.xyxy[0].tolist()
            except (AttributeError, ValueError, TypeError, IndexError) as err:
                print(f"[警告] 无法解析边界框: {err}")
                continue
            bound = (
                max(0, min(w_img - 1, int(coords[0]))),
                max(0, min(h_img - 1, int(coords[1]))),
                max(0, min(w_img, int(coords[2]))),
                max(0, min(h_img, int(coords[3]))),
            )
            if bound[2] <= bound[0] or bound[3] <= bound[1]:
                continue
            try:
                conf_val = float(box.conf.item()) if hasattr(box, "conf") else 0.0
            except (AttributeError, ValueError, TypeError):
                conf_val = 0.0
            tl_boxes.append((bound[0], bound[1], bound[2], bound[3], conf_val))
            roi = frame[bound[1] : bound[3], bound[0] : bound[2]]
            if roi.size == 0 or roi.shape[0] < MIN_ROI_SIDE or roi.shape[1] < MIN_ROI_SIDE:
                continue
            tl_state = detect_traffic_light_color(roi)
            cv2.putText(
                annotated_frame,
                TL_STATE_CN.get(tl_state, "未知"),
                (bound[0], max(TEXT_MARGIN_MIN, bound[1] - LABEL_Y_OFFSET)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TL_COLOR_MAP.get(tl_state, (255, 255, 255)),
                2,
                lineType=cv2.LINE_AA,
            )
            self._save_tl_crop(roi, frame_id, bi, tl_state)
        if tl_boxes:
            status = decide_traffic_status(frame, tl_boxes)
            self._ann.say_traffic(status)

    def _save_tl_crop(self, roi, frame_id: int, bi: int, tl_state: str) -> None:
        tl_dir = Path(self.cfg.save_dir) / "traffic_lights"
        tl_dir.mkdir(parents=True, exist_ok=True)
        crop_name = f"tl_{frame_id}_{bi}_{datetime.now(UTC).strftime(self.cfg.timestamp_fmt)}_{tl_state}.jpg"
        try:
            cv2.imwrite(str(tl_dir / crop_name), roi)
        except (cv2.error, OSError) as err:
            print(f"[警告] 保存交通灯裁剪失败: {err}")

    def _update_and_draw_fps(self, annotated_frame) -> None:
        if not self.cfg.show_fps:
            return
        now = datetime.now(UTC)
        dt = (now - self._last_time).total_seconds()
        self._last_time = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps > 0 else 1.0 / dt
        cv2.putText(
            annotated_frame,
            f"FPS: {self._fps:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    def _save_result(self, frame_id: int, annotated_frame, result) -> None:
        ts = datetime.now(UTC).strftime(self.cfg.timestamp_fmt)
        base_name = f"frame_{frame_id}_{ts}"
        cv2.imwrite(str(Path(self.cfg.save_dir) / f"{base_name}.jpg"), annotated_frame)
        if self.cfg.save_txt:
            txt_path = Path(self.cfg.save_dir) / f"{base_name}.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                for line in _format_boxes_yolo(result):
                    f.write(line)

    def detect_and_save(self, stop_event: Any | None = None):
        cfg = self.cfg
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        self._quiet_opencv_logs()
        if isinstance(cfg.source, int) and os.name == "nt":
            cap = cv2.VideoCapture(cfg.source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            msg = f"无法打开视频源： {cfg.source}"
            raise RuntimeError(msg)

        frame_id = 0
        while True:
            if self._should_stop(stop_event):
                break
            ret, frame = cap.read()
            if not ret:
                if self._inc_read_fail_and_should_break():
                    break
                continue
            self._reset_read_fail()
            self._process_frame(frame, frame_id)
            frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord(cfg.exit_key):
                break

        cap.release()
        cv2.destroyAllWindows()


def _format_boxes_yolo(result) -> list[str]:
    """将检测框转换为 YOLO txt 每行字符串"""
    lines: list[str] = []
    h, w = result.orig_shape
    for box in result.boxes:
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_c = (x1 + x2) / 2 / w
        y_c = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
    return lines


@contextmanager
def _opencv_enum_log_suppressed(*, enable: bool):
    """在枚举摄像头时临时抑制 OpenCV 低层错误日志"""
    if not enable:
        yield
        return
    utils_mod = getattr(cv2, "utils", None)
    logging_mod = getattr(utils_mod, "logging", None) if utils_mod else None
    set_level = getattr(logging_mod, "setLogLevel", None) if logging_mod else None
    get_level = getattr(logging_mod, "getLogLevel", None) if logging_mod else None
    silent_const = getattr(logging_mod, "LOG_LEVEL_SILENT", None)
    restore_level = None
    try:
        if callable(set_level) and silent_const is not None:
            try:
                if callable(get_level):
                    restore_level = get_level()
            except (cv2.error, RuntimeError) as err:
                print(f"[警告] 读取 OpenCV 日志等级失败: {err}")
                restore_level = None
            try:
                set_level(silent_const)
            except (cv2.error, RuntimeError) as err:
                print(f"[警告] 设置 OpenCV 日志等级失败: {err}")
        yield
    finally:
        if callable(set_level) and restore_level is not None:
            try:
                set_level(restore_level)
            except (cv2.error, RuntimeError) as err:
                print(f"[警告] 恢复 OpenCV 日志等级失败: {err}")


def _camera_is_usable(idx: int) -> bool:
    cap = cv2.VideoCapture(idx)
    try:
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        return bool(ret)
    finally:
        cap.release()


def enumerate_cameras(max_index: int = 8) -> list[int]:
    """探测可用摄像头索引
    尝试 0..max_index-1 打开并读取一帧判断是否可用；发现至少 1 个后，若连续失败达阈值提前结束
    """
    available: list[int] = []
    consecutive_fail = 0
    try:
        fail_limit = int(os.getenv(f"{ENV_PREFIX}CAM_FAIL_LIMIT", "3"))
    except ValueError:
        fail_limit = 3
    suppress = os.getenv(f"{ENV_PREFIX}SUPPRESS_ENUM_ERRORS", "1").lower() not in {"0", "false", ""}
    with _opencv_enum_log_suppressed(enable=suppress):
        for i in range(max_index):
            if _camera_is_usable(i):
                available.append(i)
                consecutive_fail = 0
            else:
                consecutive_fail += 1
                if available and consecutive_fail >= fail_limit:
                    break
    return available


def interactive_select_camera(max_index: int) -> int:
    cams = enumerate_cameras(max_index)
    if not cams:
        msg = "未检测到任何可用摄像头"
        raise RuntimeError(msg)
    print("可用摄像头: " + ", ".join(str(c) for c in cams))
    while True:
        sel = input("请输入要使用的摄像头索引(回车默认0): ").strip()
        if not sel:
            sel = "0"
        if sel.isdigit():
            idx = int(sel)
            if idx in cams:
                print(f"[信息] 选择摄像头 {idx}")
                return idx
        print("输入无效，请重新输入")


def main(argv: list[str] | None = None):
    """供外部脚本调用的主入口"""
    cfg = load_config_from_args(argv)
    if cfg.select_camera:
        try:
            cfg.source = interactive_select_camera(cfg.max_cam_index)
        except RuntimeError as err:
            print(f"[错误] 摄像头选择失败: {err}")
            return
    print("[配置] 使用参数: ")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    detector = YOLODetector(cfg)
    detector.detect_and_save()


__all__ = [
    "YOLOConfig",
    "YOLODetector",
    "enumerate_cameras",
    "load_config_from_args",
    "main",
]
