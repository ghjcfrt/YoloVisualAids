"""
功能特性:
 - 一般 / 高级 两种模式 (一般模式仅允许修改模型路径)
 - 设备下拉: auto / cpu / cuda(/cuda:N) / mps (若可用)
 - 恢复默认 (重新生成 YOLOConfig, 考虑环境变量覆盖)
 - 后台线程运行推理, 使用 stop_event 实现 GUI 主动停止
 - 状态栏实时显示当前状态
"""
from __future__ import annotations

import sys
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QStatusBar, QVBoxLayout,
                               QWidget)

import YOLO_detection
from YOLO_detection import YOLOConfig, enumerate_cameras, load_config_from_args

# 可选依赖：torch 和 camera_name（用于摄像头友好名）
try:
    import torch
except (ImportError, OSError, RuntimeError):  # pragma: no cover - 环境可能没有 GPU 依赖
    torch = None  # type: ignore[assignment]

try:
    from camera_name import enumerate_cameras as wmi_enum
except (ImportError, OSError):
    wmi_enum = None  # type: ignore[assignment]


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)


class DetectionWorker(threading.Thread):
    def __init__(self, argv: list[str], signals: WorkerSignals, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.argv = argv
        self.signals = signals
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            cfg = load_config_from_args(self.argv)
            det = YOLO_detection.YOLODetector(cfg)
            det.detect_and_save(stop_event=self.stop_event)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:  # pragma: no cover
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


STATUS_BAR_HEIGHT = 26   # 统一固定状态栏高度，避免随内容/字体变化
MODE_BOX_HEIGHT = 58     # 模式选择区域固定高度 (可按需调整)
EPSILON = 1e-9


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 参数配置 - PySide6")
        self.resize(760, 620)

        self.defaults = YOLOConfig()
        self.worker: DetectionWorker | None = None
        self.stop_event: threading.Event | None = None
        # 友好名称映射（显示文本 -> 实际摄像头索引）
        self._simple_cam_name_to_index: dict[str, int] = {}
        self._adv_cam_name_to_index: dict[str, int] = {}

        self._build_ui()

    # ---------------- UI -----------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._init_status_bar(layout)
        self._init_mode_box(layout)
        self._init_general_box(layout)
        self._init_adv_box(layout)
        self._init_buttons(layout)
        self._update_mode_visibility()

    def _init_status_bar(self, layout: QVBoxLayout) -> None:
        self.status = QStatusBar()
        self.status.setSizeGripEnabled(False)
        self.status.setFixedHeight(STATUS_BAR_HEIGHT)
        self.status.setStyleSheet("QStatusBar {padding-left:6px; font-size:12px;}")
        layout.addWidget(self.status)
        self.status.showMessage("就绪")

    def _init_mode_box(self, layout: QVBoxLayout) -> None:
        mode_box = QGroupBox("模式")
        mode_layout = QHBoxLayout(mode_box)
        mode_layout.setContentsMargins(8, 6, 8, 6)
        mode_layout.setSpacing(10)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["一般", "高级"])
        self.mode_combo.currentIndexChanged.connect(self._update_mode_visibility)
        mode_layout.addWidget(QLabel("选择:"))
        mode_layout.addWidget(self.mode_combo, 1)
        mode_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        mode_box.setFixedHeight(MODE_BOX_HEIGHT)
        layout.addWidget(mode_box)

    def _init_general_box(self, layout: QVBoxLayout) -> None:
        self.general_box = QGroupBox("一般 (模型 + 摄像头)")
        gen_form = QFormLayout(self.general_box)
        self.model_line_simple = QLineEdit(self.defaults.model_path)
        btn_model_simple = QPushButton("选择模型...")
        btn_model_simple.clicked.connect(self._choose_model_simple)
        row_simple = QHBoxLayout()
        row_simple.addWidget(self.model_line_simple)
        row_simple.addWidget(btn_model_simple)
        gen_form.addRow(QLabel("模型权重"), row_simple)

        self.cam_combo_simple = QComboBox()
        self.refresh_cam_btn = QPushButton("刷新")
        self.refresh_cam_btn.clicked.connect(lambda: self._refresh_cams(
            combo=self.cam_combo_simple,
            mapping=self._simple_cam_name_to_index,
            current_source=self.defaults.source,
        ))
        self.cam_combo_simple.currentIndexChanged.connect(
            lambda: self._on_cam_selected(self.cam_combo_simple, self._simple_cam_name_to_index)
        )
        cam_row = QHBoxLayout()
        cam_row.addWidget(self.cam_combo_simple)
        cam_row.addWidget(self.refresh_cam_btn)
        gen_form.addRow(QLabel("摄像头"), cam_row)
        self._refresh_cams(
            combo=self.cam_combo_simple,
            mapping=self._simple_cam_name_to_index,
            current_source=self.defaults.source,
        )
        layout.addWidget(self.general_box)

    def _init_adv_box(self, layout: QVBoxLayout) -> None:
        self.adv_box = QGroupBox("高级参数")
        form = QFormLayout(self.adv_box)
        self._add_model_row(form)
        self._add_device_row(form)
        self._add_source_row(form)
        self._add_adv_cam_row(form)
        self._add_save_dir_row(form)
        self._add_save_txt_row(form)
        self._add_conf_row(form)
        self._add_img_size_row(form)
        self._add_window_name_row(form)
        self._add_timestamp_row(form)
        self._add_exit_key_row(form)
        self._add_show_fps_row(form)
        layout.addWidget(self.adv_box)

    def _add_model_row(self, form: QFormLayout) -> None:
        self.model_line = QLineEdit(self.defaults.model_path)
        btn_model = QPushButton("选择...")
        btn_model.clicked.connect(self._choose_model)
        mh = QHBoxLayout()
        mh.addWidget(self.model_line)
        mh.addWidget(btn_model)
        form.addRow("模型权重", mh)

    def _add_device_row(self, form: QFormLayout) -> None:
        self.device_combo = QComboBox()
        self._populate_devices()
        form.addRow("运算设备", self.device_combo)

    def _add_source_row(self, form: QFormLayout) -> None:
        self.source_line = QLineEdit(str(self.defaults.source))
        btn_source = QPushButton("视频文件...")
        btn_source.clicked.connect(self._choose_source)
        sh = QHBoxLayout()
        sh.addWidget(self.source_line)
        sh.addWidget(btn_source)
        form.addRow("视频源", sh)

    def _add_adv_cam_row(self, form: QFormLayout) -> None:
        self.adv_cam_combo = QComboBox()
        self.adv_cam_refresh_btn = QPushButton("刷新")
        self.adv_cam_refresh_btn.clicked.connect(lambda: self._refresh_cams(
            combo=self.adv_cam_combo,
            mapping=self._adv_cam_name_to_index,
            current_source=self.defaults.source,
        ))
        self.adv_cam_combo.currentIndexChanged.connect(
            lambda: self._on_cam_selected(self.adv_cam_combo, self._adv_cam_name_to_index)
        )
        adv_cam_row = QHBoxLayout()
        adv_cam_row.addWidget(self.adv_cam_combo)
        adv_cam_row.addWidget(self.adv_cam_refresh_btn)
        form.addRow("摄像头列表", adv_cam_row)
        self._refresh_cams(
            combo=self.adv_cam_combo,
            mapping=self._adv_cam_name_to_index,
            current_source=self.defaults.source,
        )

    def _add_save_dir_row(self, form: QFormLayout) -> None:
        self.save_dir_line = QLineEdit(self.defaults.save_dir)
        btn_dir = QPushButton("目录...")
        btn_dir.clicked.connect(self._choose_dir)
        dh = QHBoxLayout()
        dh.addWidget(self.save_dir_line)
        dh.addWidget(btn_dir)
        form.addRow("保存目录", dh)

    def _add_save_txt_row(self, form: QFormLayout) -> None:
        self.save_txt_chk = QCheckBox("保存 txt 标注")
        self.save_txt_chk.setChecked(self.defaults.save_txt)
        form.addRow("保存TXT", self.save_txt_chk)

    def _add_conf_row(self, form: QFormLayout) -> None:
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.defaults.conf * 100))
        self.conf_value_label = QLabel(f"{self.defaults.conf:.2f}")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value_label.setText(f"{v / 100:.2f}")
        )
        ch = QHBoxLayout()
        ch.addWidget(self.conf_slider)
        ch.addWidget(self.conf_value_label)
        form.addRow("置信度", ch)

    def _add_img_size_row(self, form: QFormLayout) -> None:
        self.img_size_line = QLineEdit(
            "" if self.defaults.img_size is None else ",".join(str(i) for i in self.defaults.img_size)
        )
        form.addRow("输入尺寸", self.img_size_line)

    def _add_window_name_row(self, form: QFormLayout) -> None:
        self.window_name_line = QLineEdit(self.defaults.window_name)
        form.addRow("窗口标题", self.window_name_line)

    def _add_timestamp_row(self, form: QFormLayout) -> None:
        self.timestamp_fmt_line = QLineEdit(self.defaults.timestamp_fmt)
        form.addRow("时间戳格式", self.timestamp_fmt_line)

    def _add_exit_key_row(self, form: QFormLayout) -> None:
        self.exit_key_line = QLineEdit(self.defaults.exit_key)
        self.exit_key_line.setMaxLength(2)
        form.addRow("退出按键", self.exit_key_line)

    def _add_show_fps_row(self, form: QFormLayout) -> None:
        self.show_fps_chk = QCheckBox("显示 FPS")
        self.show_fps_chk.setChecked(self.defaults.show_fps)
        form.addRow("显示FPS", self.show_fps_chk)

    def _init_buttons(self, layout: QVBoxLayout) -> None:
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("启动检测")
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.clicked.connect(self.on_reset)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.reset_btn)
        layout.addLayout(btn_row)

    # --------- 设备枚举 ---------
    def _populate_devices(self) -> None:
        self.device_combo.clear()
        options = ["auto", "cpu"]

        def _detect_cuda() -> list[str]:
            if torch is None:
                return []
            try:
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    count = torch.cuda.device_count()
                    return ["cuda"] if count == 1 else [f"cuda:{i}" for i in range(count)]
            except (AttributeError, RuntimeError, OSError):  # pragma: no cover
                return []
            return []

        def _detect_mps() -> list[str]:
            if torch is None:
                return []
            try:
                mps = getattr(torch.backends, "mps", None)
                if mps and mps.is_available():
                    return ["mps"]
            except (AttributeError, RuntimeError):  # pragma: no cover
                return []
            return []

        options.extend(_detect_cuda())
        options.extend(_detect_mps())

        # 保持顺序去重
        uniq = list(dict.fromkeys(options))
        self.device_combo.addItems(uniq)
        idx = self.device_combo.findText(self.defaults.device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)

    # --------- 文件对话框 ---------
    def _choose_model_simple(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            filter="模型文件 (*.pt *.onnx *.engine *.tflite);;所有文件 (*.*)",
        )
        if path:
            self.model_line_simple.setText(path)

    def _choose_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            filter="模型文件 (*.pt *.onnx *.engine *.tflite);;所有文件 (*.*)",
        )
        if path:
            self.model_line.setText(path)

    def _choose_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "视频文件",
            filter="视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)",
        )
        if path:
            self.source_line.setText(path)

    def _choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if path:
            self.save_dir_line.setText(path)

    # --------- 模式切换 ---------
    def _update_mode_visibility(self):
        advanced = self.mode_combo.currentIndex() == 1
        self.adv_box.setVisible(advanced)
        self.general_box.setVisible(not advanced)

    # --------- 恢复默认 ---------
    def on_reset(self):
        self.defaults = YOLOConfig()
        # 一般
        self.model_line_simple.setText(self.defaults.model_path)
        # 重新枚举一般模式摄像头
        self._refresh_cams(
            combo=self.cam_combo_simple,
            mapping=self._simple_cam_name_to_index,
            current_source=self.defaults.source,
        )
        if isinstance(self.defaults.source, int):
            for friendly, cam_idx in self._simple_cam_name_to_index.items():
                if cam_idx == self.defaults.source:
                    pos = self.cam_combo_simple.findText(friendly)
                    if pos >= 0:
                        self.cam_combo_simple.setCurrentIndex(pos)
                    break
        # 高级
        self.model_line.setText(self.defaults.model_path)
        idx = self.device_combo.findText(self.defaults.device)
        if idx < 0:
            self.device_combo.addItem(self.defaults.device)
            idx = self.device_combo.findText(self.defaults.device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.source_line.setText(str(self.defaults.source))
        if hasattr(self, 'adv_cam_combo'):
            self._refresh_cams(
                combo=self.adv_cam_combo,
                mapping=self._adv_cam_name_to_index,
                current_source=self.defaults.source,
            )
        self.save_dir_line.setText(self.defaults.save_dir)
        self.save_txt_chk.setChecked(self.defaults.save_txt)
        self.conf_slider.setValue(int(self.defaults.conf * 100))
        self.img_size_line.setText(
            "" if self.defaults.img_size is None else ",".join(str(i) for i in self.defaults.img_size)
        )
        self.window_name_line.setText(self.defaults.window_name)
        self.timestamp_fmt_line.setText(self.defaults.timestamp_fmt)
        self.exit_key_line.setText(self.defaults.exit_key)
        self.show_fps_chk.setChecked(self.defaults.show_fps)
        self.status.showMessage("已恢复默认")

    # --------- 构建 CLI 参数 ---------
    def _build_argv(self, *, advanced: bool) -> list[str]:
        return self._build_argv_advanced() if advanced else self._build_argv_simple()

    def _build_argv_simple(self) -> list[str]:
        base = YOLOConfig()
        args: list[str] = []
        m = self.model_line_simple.text().strip()
        if m and m != base.model_path:
            args += ["--model", m]
        cam_text = self.cam_combo_simple.currentText().strip()
        if cam_text and cam_text != "None":
            if cam_text in self._simple_cam_name_to_index:
                sel_cam = self._simple_cam_name_to_index[cam_text]
            else:
                sel_cam = None
                if cam_text.endswith(")") and "(" in cam_text:
                    maybe = cam_text[cam_text.rfind("(") + 1 : -1]
                    if maybe.isdigit():
                        sel_cam = int(maybe)
                elif cam_text.isdigit():
                    sel_cam = int(cam_text)
            if sel_cam is not None and (
                (isinstance(base.source, int) and sel_cam != base.source) or not isinstance(base.source, int)
            ):
                args += ["--source", str(sel_cam)]
        return args

    def _build_argv_advanced(self) -> list[str]:
        base = YOLOConfig()
        args: list[str] = []

        def add_if_changed(flag: str, val: str, default: str) -> None:
            val_s = val.strip()
            if val_s and val_s != default:
                args.extend([flag, val_s])

        add_if_changed("--model", self.model_line.text(), base.model_path)
        add_if_changed("--device", self.device_combo.currentText(), base.device)
        add_if_changed("--source", self.source_line.text(), str(base.source))
        add_if_changed("--save-dir", self.save_dir_line.text(), base.save_dir)
        if self.save_txt_chk.isChecked() and not base.save_txt:
            args.append("--save-txt")

        conf_val = self.conf_slider.value() / 100.0
        if abs(conf_val - base.conf) > EPSILON:
            args.extend(["--conf", f"{conf_val}"])

        base_img = "" if base.img_size is None else ",".join(str(i) for i in base.img_size)
        img_text = self.img_size_line.text().strip()
        if img_text and img_text != base_img:
            args.extend(["--img-size", img_text])

        add_if_changed("--window-name", self.window_name_line.text(), base.window_name)
        add_if_changed("--timestamp-fmt", self.timestamp_fmt_line.text(), base.timestamp_fmt)

        ek = self.exit_key_line.text().strip()
        if ek and ek != base.exit_key:
            args.extend(["--exit-key", ek])
        if not self.show_fps_chk.isChecked() and base.show_fps:
            args.append("--no-fps")
        return args

    # --------- 启动 / 停止 ---------
    def on_start(self):
        if self.worker and self.worker.is_alive():
            QMessageBox.information(self, "提示", "检测已在运行中")
            return
        argv = self._build_argv(advanced=self.mode_combo.currentIndex() == 1)
        self.stop_event = threading.Event()
        signals = WorkerSignals()
        signals.finished.connect(self._on_finished)
        signals.error.connect(self._on_error)
        self.worker = DetectionWorker(argv, signals, self.stop_event)
        self.worker.start()
        self.status.showMessage("运行中: " + " ".join(argv) if argv else "运行中 (默认配置)")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)

    def on_stop(self):
        if not self.worker or not self.worker.is_alive():
            QMessageBox.information(self, "提示", "当前没有正在运行的检测")
            return
        if self.stop_event:
            self.stop_event.set()
        self.status.showMessage("请求停止... (等待当前帧) ")

    # --------- 线程回调 ---------
    def _on_finished(self):
        self.status.showMessage("已结束")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.worker = None
        self.stop_event = None

    # --------- 错误回调 ---------
    def _on_error(self, msg: str):
        QMessageBox.critical(self, "错误", msg)
        self.status.showMessage("错误: " + msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.worker = None
        self.stop_event = None

    # --------- 通用摄像头刷新逻辑 ---------
    def _refresh_cams(
        self,
        combo: QComboBox,
        mapping: dict[str, int],
        current_source,
    ):
        combo.clear()
        mapping.clear()
        cams = self._enumerate_cameras_with_names()
        if not cams:
            combo.addItem("None")
            combo.setEnabled(False)
            return
        combo.setEnabled(True)
        combo.addItem("None")
        for idx, name in cams:
            display = name
            if display in mapping:
                display = f"{display} ({idx})"
            mapping[display] = idx
            combo.addItem(display)
        if isinstance(current_source, int):
            for friendly, cam_idx in mapping.items():
                if cam_idx == current_source:
                    i = combo.findText(friendly)
                    if i >= 0:
                        combo.setCurrentIndex(i)
                    break
        self._maybe_warn_generic_names(cams)

    # --------- 统一摄像头选择回调 ---------
    def _on_cam_selected(self, combo: QComboBox, mapping: dict[str, int]):
        """统一摄像头选择回调: 将所选名称对应索引写入 source_line (若有效)。"""
        if combo.currentIndex() < 0:
            return
        text = combo.currentText().strip()
        if text == "None":
            return
        if text in mapping:
            if hasattr(self, "source_line"):
                self.source_line.setText(str(mapping[text]))
            return
        if text.endswith(")") and "(" in text:
            maybe = text[text.rfind("(") + 1 : -1]
            if maybe.isdigit() and hasattr(self, "source_line"):
                self.source_line.setText(maybe)

    # --------- 带名称摄像头枚举 ---------
    def _enumerate_cameras_with_names(self) -> list[tuple[int, str]]:
        """返回 (opencv_index, friendly_name)。使用 camera_name (WMI) + fallback。"""
        try:
            indices = enumerate_cameras(
                self.defaults.max_cam_index if hasattr(self.defaults, "max_cam_index") else 8
            )
        except (OSError, RuntimeError, ValueError):
            return []
        if not indices:
            return []
        name_map: dict[int, str] = {}
        if wmi_enum is not None:
            try:
                infos = wmi_enum(only_working=True)
                if infos:
                    for i, cam_idx in enumerate(indices):
                        if i >= len(infos):
                            break
                        nm = (getattr(infos[i], "name", "") or "").strip()
                        if nm:
                            name_map[cam_idx] = nm
            except (OSError, RuntimeError) as e:  # pragma: no cover
                print("警告: camera_name 枚举失败, 使用默认名称。原因:", repr(e))
        return [(idx, name_map.get(idx, f"Camera {idx}")) for idx in indices]

    # --------- 辅助: 检测名称是否全部为默认并提示 ---------
    def _maybe_warn_generic_names(self, cams: list[tuple[int, str]]):
        if not cams or not hasattr(self, "status"):
            return
        if all(name.strip() == f"Camera {idx}" for idx, name in cams):
            print("未获取到系统摄像头名称 (需 Windows + pywin32)，已使用默认 Camera n。")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
