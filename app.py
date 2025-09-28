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
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QStatusBar, QVBoxLayout,
                               QWidget)

import YOLO_detection
from YOLO_detection import YOLOConfig, enumerate_cameras, load_config_from_args

try:  # 枚举可能的 GPU / MPS
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)


class DetectionWorker(threading.Thread):
    def __init__(self, argv: List[str], signals: WorkerSignals, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.argv = argv
        self.signals = signals
        self.stop_event = stop_event

    def run(self):  # noqa: D401
        try:
            cfg = load_config_from_args(self.argv)
            det = YOLO_detection.YOLODetector(cfg)
            det.detect_and_save(stop_event=self.stop_event)
        except Exception as e:  # pragma: no cover
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


STATUS_BAR_HEIGHT = 26   # 统一固定状态栏高度，避免随内容/字体变化
MODE_BOX_HEIGHT = 58     # 模式选择区域固定高度 (可按需调整)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 参数配置 - PySide6")
        self.resize(760, 620)

        self.defaults = YOLOConfig()
        self.worker: Optional[DetectionWorker] = None
        self.stop_event: Optional[threading.Event] = None
        # 友好名称映射（显示文本 -> 实际摄像头索引）
        self._simple_cam_name_to_index: dict[str, int] = {}
        self._adv_cam_name_to_index: dict[str, int] = {}

        self._build_ui()

    # ---------------- UI -----------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 状态栏 (提前创建, 以便后续刷新摄像头时可用)
        self.status = QStatusBar()
        # 固定高度，禁止高度随布局变化；SizeGrip 在普通 QWidget 中无意义可关闭
        self.status.setSizeGripEnabled(False)
        self.status.setFixedHeight(STATUS_BAR_HEIGHT)
        # 轻微内边距并确保文本单行（可根据需要调整样式）
        self.status.setStyleSheet(
            "QStatusBar {padding-left:6px; font-size:12px;}"
        )
        layout.addWidget(self.status)
        self.status.showMessage("就绪")
        # 模式选择 (固定高度)
        mode_box = QGroupBox("模式")
        mode_layout = QHBoxLayout(mode_box)
        mode_layout.setContentsMargins(8, 6, 8, 6)
        mode_layout.setSpacing(10)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["一般", "高级"])
        self.mode_combo.currentIndexChanged.connect(self._update_mode_visibility)
        mode_layout.addWidget(QLabel("选择:"))
        mode_layout.addWidget(self.mode_combo, 1)
        # 固定高度 + 限制垂直扩展
        mode_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        mode_box.setFixedHeight(MODE_BOX_HEIGHT)
        layout.addWidget(mode_box)

        # 一般模式
        self.general_box = QGroupBox("一般 (模型 + 摄像头)")
        gen_form = QFormLayout(self.general_box)
        self.model_line_simple = QLineEdit(self.defaults.model_path)
        btn_model_simple = QPushButton("选择模型...")
        btn_model_simple.clicked.connect(self._choose_model_simple)
        row_simple = QHBoxLayout()
        row_simple.addWidget(self.model_line_simple)
        row_simple.addWidget(btn_model_simple)
        gen_form.addRow(QLabel("模型权重"), row_simple)

        # 摄像头下拉 (显示友好名称 + None)
        self.cam_combo_simple = QComboBox()
        self.refresh_cam_btn = QPushButton("刷新")
        self.refresh_cam_btn.clicked.connect(lambda: self._refresh_cams(
            combo=self.cam_combo_simple,
            mapping=self._simple_cam_name_to_index,
            current_source=self.defaults.source,
        ))
        # 选择时更新 source_line (统一逻辑)
        self.cam_combo_simple.currentIndexChanged.connect(
            lambda: self._on_cam_selected(self.cam_combo_simple, self._simple_cam_name_to_index)
        )
        cam_row = QHBoxLayout()
        cam_row.addWidget(self.cam_combo_simple)
        cam_row.addWidget(self.refresh_cam_btn)
        gen_form.addRow(QLabel("摄像头"), cam_row)
        # 初始化一般模式摄像头列表
        self._refresh_cams(
            combo=self.cam_combo_simple,
            mapping=self._simple_cam_name_to_index,
            current_source=self.defaults.source,
        )
        layout.addWidget(self.general_box)

        # 高级参数
        self.adv_box = QGroupBox("高级参数")
        form = QFormLayout(self.adv_box)

        # 模型路径 (高级)
        self.model_line = QLineEdit(self.defaults.model_path)
        btn_model = QPushButton("选择...")
        btn_model.clicked.connect(self._choose_model)
        mh = QHBoxLayout()
        mh.addWidget(self.model_line)
        mh.addWidget(btn_model)
        form.addRow("模型权重", mh)

        # 运算设备
        self.device_combo = QComboBox()
        self._populate_devices()
        form.addRow("运算设备", self.device_combo)

        # 视频源输入 + 文件按钮
        self.source_line = QLineEdit(str(self.defaults.source))
        btn_source = QPushButton("视频文件...")
        btn_source.clicked.connect(self._choose_source)
        sh = QHBoxLayout()
        sh.addWidget(self.source_line)
        sh.addWidget(btn_source)
        form.addRow("视频源", sh)

        # 摄像头列表 (高级)
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

        # 保存目录
        self.save_dir_line = QLineEdit(self.defaults.save_dir)
        btn_dir = QPushButton("目录...")
        btn_dir.clicked.connect(self._choose_dir)
        dh = QHBoxLayout()
        dh.addWidget(self.save_dir_line)
        dh.addWidget(btn_dir)
        form.addRow("保存目录", dh)

        # 保存 txt
        self.save_txt_chk = QCheckBox("保存 txt 标注")
        self.save_txt_chk.setChecked(self.defaults.save_txt)
        form.addRow("保存TXT", self.save_txt_chk)

        # 置信度
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.defaults.conf * 100))
        self.conf_value_label = QLabel(f"{self.defaults.conf:.2f}")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_value_label.setText(f"{v/100:.2f}"))
        ch = QHBoxLayout()
        ch.addWidget(self.conf_slider)
        ch.addWidget(self.conf_value_label)
        form.addRow("置信度", ch)

        # 输入尺寸
        self.img_size_line = QLineEdit(
            "" if self.defaults.img_size is None else ",".join(str(i) for i in self.defaults.img_size)
        )
        form.addRow("输入尺寸", self.img_size_line)

        # 窗口标题
        self.window_name_line = QLineEdit(self.defaults.window_name)
        form.addRow("窗口标题", self.window_name_line)

        # 时间戳格式
        self.timestamp_fmt_line = QLineEdit(self.defaults.timestamp_fmt)
        form.addRow("时间戳格式", self.timestamp_fmt_line)

        # 退出按键
        self.exit_key_line = QLineEdit(self.defaults.exit_key)
        self.exit_key_line.setMaxLength(2)
        form.addRow("退出按键", self.exit_key_line)

        # 显示 FPS
        self.show_fps_chk = QCheckBox("显示 FPS")
        self.show_fps_chk.setChecked(self.defaults.show_fps)
        form.addRow("显示FPS", self.show_fps_chk)

        layout.addWidget(self.adv_box)

        # 操作按钮
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

        self._update_mode_visibility()

    # --------- 设备枚举 ---------
    def _populate_devices(self):
        self.device_combo.clear()
        options = ["auto", "cpu"]
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    count = torch.cuda.device_count()
                    if count == 1:
                        options.append("cuda")
                    else:
                        for i in range(count):
                            options.append(f"cuda:{i}")
            except Exception:  # pragma: no cover
                pass
            try:
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    options.append("mps")
            except Exception:  # pragma: no cover
                pass
        seen = set()
        uniq: List[str] = []
        for o in options:
            if o not in seen:
                seen.add(o)
                uniq.append(o)
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
    def _build_argv(self, advanced: bool) -> List[str]:
        base = YOLOConfig()
        args: List[str] = []
        if not advanced:
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
                        maybe = cam_text[cam_text.rfind("(")+1:-1]
                        if maybe.isdigit():
                            sel_cam = int(maybe)
                    elif cam_text.isdigit():
                        sel_cam = int(cam_text)
                if sel_cam is not None:
                    if (isinstance(base.source, int) and sel_cam != base.source) or not isinstance(base.source, int):
                        args += ["--source", str(sel_cam)]
            return args

        def add(flag: str, val, default):
            if val is None:
                return
            if isinstance(val, str):
                if val.strip() == "" or val == default:
                    return
            if val != default:
                args.extend([flag, str(val)])

        add("--model", self.model_line.text().strip(), base.model_path)
        add("--device", self.device_combo.currentText().strip(), base.device)
        add("--source", self.source_line.text().strip(), str(base.source))
        add("--save-dir", self.save_dir_line.text().strip(), base.save_dir)
        if self.save_txt_chk.isChecked() and not base.save_txt:
            args.append("--save-txt")
        conf_val = self.conf_slider.value() / 100.0
        if abs(conf_val - base.conf) > 1e-9:
            args += ["--conf", f"{conf_val}"]
        img_text = self.img_size_line.text().strip()
        if img_text and img_text != (
            "" if base.img_size is None else ",".join(str(i) for i in base.img_size)
        ):
            args += ["--img-size", img_text]
        add("--window-name", self.window_name_line.text().strip(), base.window_name)
        add("--timestamp-fmt", self.timestamp_fmt_line.text().strip(), base.timestamp_fmt)
        ek = self.exit_key_line.text().strip()
        if ek and ek != base.exit_key:
            args += ["--exit-key", ek]
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
            if hasattr(self, 'source_line'):
                self.source_line.setText(str(mapping[text]))
            return
        if text.endswith(")") and "(" in text:
            maybe = text[text.rfind("(")+1:-1]
            if maybe.isdigit():
                if hasattr(self, 'source_line'):
                    self.source_line.setText(maybe)

    # --------- 带名称摄像头枚举 ---------
    def _enumerate_cameras_with_names(self) -> List[tuple[int, str]]:
        """返回 (opencv_index, friendly_name)。使用 camera_name (WMI) + fallback。"""
        try:
            indices = enumerate_cameras(
                self.defaults.max_cam_index if hasattr(self.defaults, 'max_cam_index') else 8
            )
        except Exception:
            return []
        if not indices:
            return []
        name_map: dict[int, str] = {}
        try:
            from camera_name import \
                enumerate_cameras as wmi_enum  # type: ignore
            infos = wmi_enum(only_working=True)
            if infos:
                for i, cam_idx in enumerate(indices):
                    if i >= len(infos):
                        break
                    nm = (getattr(infos[i], 'name', '') or '').strip()
                    if nm:
                        name_map[cam_idx] = nm
        except Exception as e:  # pragma: no cover
            print("警告: camera_name 枚举失败, 使用默认名称。原因:", repr(e))
        return [(idx, name_map.get(idx, f"Camera {idx}")) for idx in indices]

    # --------- 辅助: 检测名称是否全部为默认并提示 ---------
    def _maybe_warn_generic_names(self, cams: List[tuple[int, str]]):
        if not cams or not hasattr(self, 'status'):
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

