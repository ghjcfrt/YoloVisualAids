"""GUI 启动"""

from __future__ import annotations  # noqa: I001

import contextlib
import logging
import pathlib
import re
import sys
import threading

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QStatusBar, QVBoxLayout,
                               QWidget)

from app.worker import DetectionWorker, WorkerSignals
from detection.api import YOLOConfig, enumerate_cameras
from voice import set_speaker
from voice.tts_queue import TTSManager
from voice.voice_control import VoiceController
from yva_io.camera_utils import (get_directshow_device_names,
                                 map_indices_to_names)
from yva_io.device_utils import list_devices

# 语音播报（可选）
try:
    from voice import tts as _tts_mod
except (ImportError, OSError, RuntimeError):  # 无 TTS 依赖时不播报
    _tts_mod = None  # type: ignore[assignment]

# 可选依赖：torch 和 camera_name（用于摄像头友好名）
try:
    import torch
except (ImportError, OSError, RuntimeError):  # 环境可能没有 GPU 依赖
    torch = None

try:
    from yva_io.camera_name import enumerate_cameras as wmi_enum
except (ImportError, OSError):
    wmi_enum = None

# --- 语音/TTS 专用作用域日志（不启用全局 DEBUG） ---
_voice_log = logging.getLogger("YVA.Voice")
_tts_log = logging.getLogger("YVA.TTS")
for _lg in (_voice_log, _tts_log):
    _lg.setLevel(logging.DEBUG)
    if not _lg.handlers:
        _h = logging.StreamHandler()
        _h.setLevel(logging.DEBUG)
        _h.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
        _lg.addHandler(_h)
    _lg.propagate = False

STATUS_BAR_HEIGHT = 26
MODE_BOX_HEIGHT = 58
EPSILON = 1e-9
DUP_SPEECH_WINDOW = 1.2
GENERIC_CAM_RE = re.compile(r"^\s*Camera\s+\d+\s*$", re.IGNORECASE)


class MainWindow(QWidget):
    """应用主窗口：提供参数配置、摄像头选择、语音控制与检测启动/停止。"""

    voiceStart = Signal()
    voiceStop = Signal()
    statusFlash = Signal(str, int, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 参数配置 - PySide6")
        self.resize(760, 620)

        self.defaults = YOLOConfig()
        self.worker: DetectionWorker | None = None
        self.stop_event: threading.Event | None = None
        self._simple_cam_name_to_index: dict[str, int | str] = {}
        self._adv_cam_name_to_index: dict[str, int | str] = {}

        self._build_ui()
        self.voiceStart.connect(self._on_voice_start)
        self.voiceStop.connect(self._on_voice_stop)
        self.statusFlash.connect(self._flash_status)
        self._tts = TTSManager(tts_module=_tts_mod, logger=_tts_log, dup_window=DUP_SPEECH_WINDOW)
        self._tts.start()
        set_speaker(self._tts.speak)
        self._voice_ctrl = None
        self._init_voice_control()
        # 默认关闭调试日志输出
        self._apply_debug_logging(enabled=False)

    def _build_ui(self) -> None:
        """构建主界面布局"""
        layout = QVBoxLayout(self)
        self._init_status_bar(layout)
        self._init_mode_box(layout)
        self._init_general_box(layout)
        self._init_adv_box(layout)
        self._init_buttons(layout)
        self._update_mode_visibility()

    def _init_status_bar(self, layout: QVBoxLayout) -> None:
        """初始化状态栏"""
        self.status = QStatusBar()
        self.status.setSizeGripEnabled(False)
        self.status.setFixedHeight(STATUS_BAR_HEIGHT)
        self.status.setStyleSheet("QStatusBar {padding-left:6px; font-size:12px;}")
        layout.addWidget(self.status)
        self.status.showMessage("就绪")

    def _flash_status(self, msg: str, ms: int = 2000, fallback: str = "就绪") -> None:
        """临时显示一条状态消息 到期恢复为 fallback"""
        self.status.showMessage(msg)
        expected = msg

        def _restore() -> None:
            with contextlib.suppress(Exception):
                if self.status.currentMessage() == expected:
                    self.status.showMessage(fallback)

        QTimer.singleShot(ms, _restore)

    def _init_mode_box(self, layout: QVBoxLayout) -> None:
        """初始化模式选择区 一般与高级"""
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
        """初始化一般模式 参数包含模型与摄像头"""
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
        """初始化高级参数区域"""
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
        self._add_debug_log_row(form)
        self._add_announcer_rows(form)
        layout.addWidget(self.adv_box)

    def _add_model_row(self, form: QFormLayout) -> None:
        """添加模型权重选择行"""
        self.model_line = QLineEdit(self.defaults.model_path)
        btn_model = QPushButton("选择...")
        btn_model.clicked.connect(self._choose_model)
        mh = QHBoxLayout()
        mh.addWidget(self.model_line)
        mh.addWidget(btn_model)
        form.addRow("模型权重", mh)

    def _add_device_row(self, form: QFormLayout) -> None:
        """添加运算设备选择行"""
        self.device_combo = QComboBox()
        self._populate_devices()
        form.addRow("运算设备", self.device_combo)

    def _add_source_row(self, form: QFormLayout) -> None:
        """添加视频源输入行 支持索引或文件路径"""
        self.source_line = QLineEdit(str(self.defaults.source))
        btn_source = QPushButton("视频文件...")
        btn_source.clicked.connect(self._choose_source)
        sh = QHBoxLayout()
        sh.addWidget(self.source_line)
        sh.addWidget(btn_source)
        form.addRow("视频源", sh)

    def _add_adv_cam_row(self, form: QFormLayout) -> None:
        """添加高级模式下的摄像头列表"""
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
        """添加保存目录选择行"""
        self.save_dir_line = QLineEdit(self.defaults.save_dir)
        btn_dir = QPushButton("目录...")
        btn_dir.clicked.connect(self._choose_dir)
        dh = QHBoxLayout()
        dh.addWidget(self.save_dir_line)
        dh.addWidget(btn_dir)
        form.addRow("保存目录", dh)

    def _add_save_txt_row(self, form: QFormLayout) -> None:
        """添加保存 txt 标注开关"""
        self.save_txt_chk = QCheckBox("保存 txt 标注")
        self.save_txt_chk.setChecked(self.defaults.save_txt)
        form.addRow("保存TXT", self.save_txt_chk)

    def _add_conf_row(self, form: QFormLayout) -> None:
        """添加置信度阈值滑条"""
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
        """添加输入尺寸设置 留空表示使用原始尺寸"""
        self.img_size_line = QLineEdit(
            "" if self.defaults.img_size is None else ",".join(str(i) for i in self.defaults.img_size)
        )
        form.addRow("输入尺寸", self.img_size_line)

    def _add_window_name_row(self, form: QFormLayout) -> None:
        """添加窗口标题设置"""
        self.window_name_line = QLineEdit(self.defaults.window_name)
        form.addRow("窗口标题", self.window_name_line)

    def _add_timestamp_row(self, form: QFormLayout) -> None:
        """添加时间戳格式设置"""
        self.timestamp_fmt_line = QLineEdit(self.defaults.timestamp_fmt)
        form.addRow("时间戳格式", self.timestamp_fmt_line)

    def _add_exit_key_row(self, form: QFormLayout) -> None:
        """添加退出按键设置"""
        self.exit_key_line = QLineEdit(self.defaults.exit_key)
        self.exit_key_line.setMaxLength(2)
        form.addRow("退出按键", self.exit_key_line)

    def _add_show_fps_row(self, form: QFormLayout) -> None:
        """添加是否显示 FPS 的开关"""
        self.show_fps_chk = QCheckBox("显示 FPS")
        self.show_fps_chk.setChecked(self.defaults.show_fps)
        form.addRow("显示FPS", self.show_fps_chk)

    def _add_debug_log_row(self, form: QFormLayout) -> None:
        """添加调试日志开关 仅影响本应用的语音相关日志"""
        self.debug_log_chk = QCheckBox("调试日志输出")
        self.debug_log_chk.setChecked(False)
        form.addRow("调试日志", self.debug_log_chk)

    def _add_announcer_rows(self, form: QFormLayout) -> None:
        """添加播报与黄闪相关参数行"""
        self.ann_min_interval_line = QLineEdit(str(getattr(self.defaults, 'ann_min_interval', 1.5)))
        form.addRow("播报最小间隔(秒)", self.ann_min_interval_line)
        self.ann_flash_window_line = QLineEdit(str(getattr(self.defaults, 'ann_flash_window', 3.0)))
        form.addRow("黄闪窗口(秒)", self.ann_flash_window_line)
        self.ann_flash_min_events_line = QLineEdit(str(getattr(self.defaults, 'ann_flash_min_events', 6)))
        form.addRow("黄闪最少采样", self.ann_flash_min_events_line)
        self.ann_flash_yellow_ratio_line = QLineEdit(str(getattr(self.defaults, 'ann_flash_yellow_ratio', 0.9)))
        form.addRow("黄灯占比阈值(0~1)", self.ann_flash_yellow_ratio_line)
        self.ann_flash_cooldown_line = QLineEdit(str(getattr(self.defaults, 'ann_flash_cooldown', 5.0)))
        form.addRow("黄闪冷却(秒)", self.ann_flash_cooldown_line)

    def _init_buttons(self, layout: QVBoxLayout) -> None:
        """初始化启动 停止 恢复默认 按钮"""
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

    def _init_voice_control(self) -> None:
        """初始化语音控制 如可用则启动关键词监听"""
        # 模型目录相对项目根: <repo_root>/models/vosk/vosk-model-small-cn-0.22
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        model_dir = repo_root / "models" / "vosk" / "vosk-model-small-cn-0.22"
        self._voice_ctrl = VoiceController(model_dir)
        if not self._voice_ctrl.available():
            self.status.showMessage("语音控制未启用（缺少依赖或模型目录）")
            return

        def _on_voice(text: str) -> None:
            t = (text or "").strip()
            _voice_log.debug("检测到关键词: '%s'", t)
            if "启动辅助系统" in t:
                _voice_log.debug("发出 voiceStart 信号")
                self.voiceStart.emit()
                self.statusFlash.emit("语音指令：启动辅助系统", 2000, "就绪")
            elif "关闭辅助系统" in t:
                _voice_log.debug("发出 voiceStop 信号")
                self.voiceStop.emit()
                self.statusFlash.emit("语音指令：关闭辅助系统", 2000, "就绪")

        self._voice_ctrl.start(["启动辅助系统", "关闭辅助系统"], _on_voice)
        _voice_log.debug("语音监听已启动")
        self.status.showMessage("语音控制已启用（可说：启动辅助系统 / 关闭辅助系统）")

    def _on_voice_start(self) -> None:
        """语音触发启动检测 若已运行则提示"""
        _voice_log.debug("收到语音启动请求，当前是否已运行: %s", bool(self.worker and self.worker.is_alive()))
        if self.worker and self.worker.is_alive():
            # 如果正在播报检测到xxx，应抑制“检测到xxx”，但仍播报“检测已在运行中”
            if self._is_detection_speaking():
                _voice_log.debug("已在运行中，且当前为检测播报 -> 清理/抑制检测播报，并保留状态提示")
                self._suppress_detection_announcements()
            _voice_log.debug("已在运行中 -> 语音提示 '检测已在运行中'")
            self._speak("检测已在运行中")
            return
        _voice_log.debug("未运行 -> 调用 on_start")
        self.on_start()

    def _on_voice_stop(self) -> None:
        """语音触发停止检测 若未运行则提示"""
        _voice_log.debug("收到语音停止请求，当前是否已运行: %s", bool(self.worker and self.worker.is_alive()))
        if not self.worker or not self.worker.is_alive():
            _voice_log.debug("未在运行 -> 语音提示 '当前没有正在运行的检测'")
            self._speak("当前没有正在运行的检测")
            return
        _voice_log.debug("运行中 -> 调用 on_stop")
        self.on_stop()

    def _is_detection_speaking(self) -> bool:
        """判断 TTS 是否正在播报以“检测到”开头/包含的检测结果。

        说明：检测播报通常来自 voice.announce 中的 "检测到：..."，
        我们通过查询 TTSManager 的当前文本并去除零宽字符来判断。
        """
        tts_mgr = getattr(self, "_tts", None)
        if not tts_mgr or not hasattr(tts_mgr, "is_busy") or not hasattr(tts_mgr, "get_current_text"):
            return False
        if not tts_mgr.is_busy():
            return False
        cur = tts_mgr.get_current_text() or ""
        # 去除用于去重的零宽字符
        for z in ("\u200b", "\u200c", "\u200d"):
            cur = cur.replace(z, "")
        return "检测到" in cur

    def _suppress_detection_announcements(self) -> None:
        """抑制/清队近期的“检测到 …”播报，避免打断状态提示。"""
        tts_mgr = getattr(self, "_tts", None)
        if not tts_mgr:
            return
        # 清理队列里可能尚未播出的“检测到 …”
        if hasattr(tts_mgr, "clear_pending_substring"):
            tts_mgr.clear_pending_substring("检测到")
        # 在短时间内抑制新的“检测到 …”
        if hasattr(tts_mgr, "suppress_substring"):
            tts_mgr.suppress_substring("检测到", duration_sec=2.0)

    def _speak(self, text: str) -> None:
        """通过 TTS 管理器播报一条文本"""
        if not text:
            _tts_log.debug("跳过播报: 文本为空")
            return
        tts_mgr = getattr(self, "_tts", None)
        if tts_mgr is None:
            _tts_log.debug("跳过播报: TTS 管理器未初始化")
            return
        tts_mgr.speak(text)

    def _populate_devices(self) -> None:
        """刷新设备下拉列表 默认选择配置中的设备"""
        self.device_combo.clear()
        self.device_combo.addItems(list_devices(torch))
        idx = self.device_combo.findText(self.defaults.device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)

    def _choose_model_simple(self):
        """选择模型权重文件 写入一般模式输入框"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            filter="模型文件 (*.pt *.onnx *.engine *.tflite);;所有文件 (*.*)",
        )
        if path:
            self.model_line_simple.setText(path)

    def _choose_model(self):
        """选择模型权重文件 写入高级模式输入框"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            filter="模型文件 (*.pt *.onnx *.engine *.tflite);;所有文件 (*.*)",
        )
        if path:
            self.model_line.setText(path)

    def _choose_source(self):
        """选择视频文件 写入视频源输入框"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "视频文件",
            filter="视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)",
        )
        if path:
            self.source_line.setText(path)

    def _choose_dir(self):
        """选择保存目录 写入保存目录输入框"""
        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if path:
            self.save_dir_line.setText(path)

    def _update_mode_visibility(self):
        """根据模式切换显示一般区或高级区"""
        advanced = self.mode_combo.currentIndex() == 1
        self.adv_box.setVisible(advanced)
        self.general_box.setVisible(not advanced)

    def on_reset(self):
        """恢复所有参数为默认值 并刷新界面"""
        self.defaults = YOLOConfig()
        self.model_line_simple.setText(self.defaults.model_path)
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
        if hasattr(self, 'debug_log_chk'):
            self.debug_log_chk.setChecked(False)
            self._apply_debug_logging(enabled=False)
        if hasattr(self, 'ann_min_interval_line'):
            self.ann_min_interval_line.setText(str(getattr(self.defaults, 'ann_min_interval', 1.5)))
            self.ann_flash_window_line.setText(str(getattr(self.defaults, 'ann_flash_window', 3.0)))
            self.ann_flash_min_events_line.setText(str(getattr(self.defaults, 'ann_flash_min_events', 6)))
            self.ann_flash_yellow_ratio_line.setText(str(getattr(self.defaults, 'ann_flash_yellow_ratio', 0.9)))
            self.ann_flash_cooldown_line.setText(str(getattr(self.defaults, 'ann_flash_cooldown', 5.0)))
        self.status.showMessage("已恢复默认")

    def _build_argv(self, *, advanced: bool) -> list[str]:
        """根据当前 UI 值构建 CLI 参数列表"""
        return self._build_argv_advanced() if advanced else self._build_argv_simple()

    def _build_argv_simple(self) -> list[str]:
        """一般模式下仅包含模型与摄像头相关参数"""
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
        """高级模式下构建完整参数 仅加入与默认不同的项"""
        base = YOLOConfig()
        args: list[str] = []
        self._argv_basic_options(args, base)
        self._argv_conf_and_img(args, base)
        self._argv_window_and_misc(args, base)
        self._argv_announcer(args, base)
        return args

    def _argv_basic_options(self, args: list[str], base: YOLOConfig) -> None:
        """基础选项：模型 设备 源 保存目录 文本标注"""
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

    def _argv_conf_and_img(self, args: list[str], base: YOLOConfig) -> None:
        """阈值与尺寸"""
        conf_val = self.conf_slider.value() / 100.0
        if abs(conf_val - base.conf) > EPSILON:
            args.extend(["--conf", f"{conf_val}"])

        base_img = "" if base.img_size is None else ",".join(str(i) for i in base.img_size)
        img_text = self.img_size_line.text().strip()
        if img_text and img_text != base_img:
            args.extend(["--img-size", img_text])

    def _argv_window_and_misc(self, args: list[str], base: YOLOConfig) -> None:
        """窗口 标题 时间戳 退出按键 FPS"""
        def add_if_changed(flag: str, val: str, default: str) -> None:
            val_s = val.strip()
            if val_s and val_s != default:
                args.extend([flag, val_s])

        add_if_changed("--window-name", self.window_name_line.text(), base.window_name)
        add_if_changed("--timestamp-fmt", self.timestamp_fmt_line.text(), base.timestamp_fmt)

        ek = self.exit_key_line.text().strip()
        if ek and ek != base.exit_key:
            args.extend(["--exit-key", ek])
        if not self.show_fps_chk.isChecked() and base.show_fps:
            args.append("--no-fps")

    def _argv_announcer(self, args: list[str], base: YOLOConfig) -> None:
        """播报相关参数"""
        def add_num(flag: str, text: str, default_val: float):
            s = text.strip()
            if not s:
                return
            try:
                val = float(s)
            except ValueError:
                return
            if abs(val - default_val) > EPSILON:
                args.extend([flag, str(val)])

        def add_int(flag: str, text: str, default_val: int):
            s = text.strip()
            if not s:
                return
            try:
                val = int(s)
            except ValueError:
                return
            if val != default_val:
                args.extend([flag, str(val)])

        add_num("--ann-min-interval", self.ann_min_interval_line.text(), float(getattr(base, 'ann_min_interval', 1.5)))
        add_num("--ann-flash-window", self.ann_flash_window_line.text(), float(getattr(base, 'ann_flash_window', 3.0)))
        add_int("--ann-flash-min-events", self.ann_flash_min_events_line.text(), int(getattr(base, 'ann_flash_min_events', 6)))
        add_num("--ann-flash-yellow-ratio", self.ann_flash_yellow_ratio_line.text(), float(getattr(base, 'ann_flash_yellow_ratio', 0.9)))
        add_num("--ann-flash-cooldown", self.ann_flash_cooldown_line.text(), float(getattr(base, 'ann_flash_cooldown', 5.0)))

    def on_start(self):
        """启动后台检测线程 并更新按钮与状态"""
        if self.worker and self.worker.is_alive():
            QMessageBox.information(self, "提示", "检测已在运行中")
            return
        # 根据勾选应用调试日志等级
        if hasattr(self, 'debug_log_chk'):
            self._apply_debug_logging(enabled=self.debug_log_chk.isChecked())
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
        """请求停止后台检测 若未运行则提示"""
        if not self.worker or not self.worker.is_alive():
            _voice_log.debug("点击停止但未在运行 -> 弹出提示框")
            QMessageBox.information(self, "提示", "当前没有正在运行的检测")
            return
        if self.stop_event:
            _voice_log.debug("停止请求 -> 设置 stop_event")
            self.stop_event.set()
        self.status.showMessage("请求关闭... (等待当前帧) ")

    def _on_finished(self):
        """后台线程结束后的收尾与按钮复位"""
        _voice_log.debug("工作线程已结束")
        self.status.showMessage("已结束")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.worker = None
        self.stop_event = None

    def closeEvent(self, event) -> None:
        """关闭窗口时停止语音与 TTS 管理器 然后交由父类处理"""
        try:
            vc = getattr(self, "_voice_ctrl", None)
            if vc is not None:
                _voice_log.debug("关闭窗口 -> 停止语音监听")
                vc.stop()
        except (AttributeError, RuntimeError):
            logging.exception("关闭语音监听时异常")
        try:
            tts_mgr = getattr(self, "_tts", None)
            if tts_mgr is not None:
                _tts_log.debug("关闭窗口 -> 停止 TTS 管理器")
                tts_mgr.stop()
        except (AttributeError, RuntimeError):
            logging.exception("关闭 TTS 管理器时异常")
        return super().closeEvent(event)

    def _on_error(self, msg: str):
        """处理后台错误 弹窗并重置按钮"""
        QMessageBox.critical(self, "错误", msg)
        self.status.showMessage("错误: " + msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.worker = None
        self.stop_event = None

    def _refresh_cams(
        self,
        combo: QComboBox,
        mapping: dict[str, int | str],
        current_source,
    ):
        """刷新摄像头下拉列表 支持 DirectShow 名称或 WMI 名称"""
        combo.clear()
        mapping.clear()
        friendly_names: list[str] = []
        try:
            indices = enumerate_cameras(
                self.defaults.max_cam_index if hasattr(self.defaults, "max_cam_index") else 8
            )
        except (OSError, RuntimeError, ValueError):
            indices = []
        if not indices:
            combo.addItem("None")
            combo.setEnabled(False)
            return
        combo.setEnabled(True)
        combo.addItem("None")
        ds_names = get_directshow_device_names()
        if ds_names:
            for idx in indices:
                nm = ds_names[idx] if 0 <= idx < len(ds_names) else f"Camera {idx}"
                display = f"Camera {idx}  |  {nm}" if nm else f"Camera {idx}"
                mapping[display] = idx
                combo.addItem(display)
                friendly_names.append(nm or "")
        else:
            name_map = map_indices_to_names(indices, wmi_enum)
            for idx in indices:
                nm = name_map.get(idx, f"Camera {idx}")
                display = f"Camera {idx}  |  {nm}" if nm else f"Camera {idx}"
                mapping[display] = idx
                combo.addItem(display)
                friendly_names.append(nm or "")
        if isinstance(current_source, int):
            for friendly, cam_idx in mapping.items():
                if cam_idx == current_source:
                    i = combo.findText(friendly)
                    if i >= 0:
                        combo.setCurrentIndex(i)
                    break
        self._maybe_warn_generic_names(friendly_names)

    def _on_cam_selected(self, combo: QComboBox, mapping: dict[str, int | str]):
        """当选择某摄像头时 同步到视频源输入框"""
        if combo.currentIndex() < 0:
            return
        text = combo.currentText().strip()
        if text == "None":
            return
        if text in mapping:
            if hasattr(self, "source_line"):
                self.source_line.setText(str(mapping[text]))
            return
        if "Camera " in text:
            parts = text.split()
            for p in parts:
                if p.isdigit():
                    if hasattr(self, "source_line"):
                        self.source_line.setText(p)
                    return

    def _maybe_warn_generic_names(self, names: list[str]):
        """若摄像头名称均为默认 Camera n 或为空 则输出提示"""
        if not names or not hasattr(self, "status"):
            return
    # 若全部名称均为默认样式 "Camera n" 或为空，则提示；否则不提示

        def _is_generic(s: str) -> bool:
            t = (s or "").strip()
            if not t:
                return True
            return bool(GENERIC_CAM_RE.match(t))

        if all(_is_generic(n) for n in names):
            print("未获取到系统摄像头名称（可选依赖：pygrabber 或 Windows + pywin32），已使用默认 Camera n")

    @staticmethod
    def _apply_debug_logging(*, enabled: bool) -> None:
        """根据开关启用或关闭调试日志 仅影响本应用内置语音日志"""
        level = logging.DEBUG if enabled else logging.WARNING
        try:
            _voice_log.setLevel(level)
            _tts_log.setLevel(level)
        except Exception:
            logging.exception("设置语音日志等级失败")
        # 同步调整已存在 handler 的等级
        for lg in (_voice_log, _tts_log):
            for h in lg.handlers:
                with contextlib.suppress(Exception):
                    h.setLevel(level)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
