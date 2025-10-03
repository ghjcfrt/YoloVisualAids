"""
功能特性:
 - 一般 / 高级 两种模式 (一般模式仅允许修改模型路径)
 - 设备下拉: auto / cpu / cuda(/cuda:N) / mps (若可用)
 - 恢复默认 (重新生成 YOLOConfig, 考虑环境变量覆盖)
 - 后台线程运行推理, 使用 stop_event 实现 GUI 主动停止
 - 状态栏实时显示当前状态
"""
from __future__ import annotations  # noqa: I001

import contextlib
import logging
import pathlib
import sys
import threading

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QStatusBar, QVBoxLayout,
                               QWidget)

from announce import set_speaker
from camera_utils import get_directshow_device_names, map_indices_to_names
from detection_worker import DetectionWorker, WorkerSignals
from device_utils import list_devices
from tts_queue import TTSManager
from voice_control import VoiceController
from YOLO_detection import YOLOConfig, enumerate_cameras

# 语音播报（可选）
try:
    import tts as _tts_mod
except (ImportError, OSError, RuntimeError):  # pragma: no cover - 无 TTS 依赖时不播报
    _tts_mod = None  # type: ignore[assignment]

# 可选依赖：torch 和 camera_name（用于摄像头友好名）
try:
    import torch
except (ImportError, OSError, RuntimeError):  # pragma: no cover - 环境可能没有 GPU 依赖
    torch = None  # type: ignore[assignment]

try:
    from camera_name import enumerate_cameras as wmi_enum
except (ImportError, OSError):
    wmi_enum = None  # type: ignore[assignment]

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

STATUS_BAR_HEIGHT = 26   # 统一固定状态栏高度，避免随内容/字体变化
MODE_BOX_HEIGHT = 58     # 模式选择区域固定高度 (可按需调整)
EPSILON = 1e-9
DUP_SPEECH_WINDOW = 1.2  # 相同文本在该窗口内视为重复，添加零宽字符强制区分


class MainWindow(QWidget):
    """应用主窗口：提供参数配置、摄像头选择、语音控制与检测启动/停止。"""
    # 将语音触发通过 Qt 信号转到主线程，避免跨线程操作 UI
    voiceStart = Signal()
    voiceStop = Signal()
    # 在主线程刷新状态栏（消息, 持续毫秒, 恢复文本）
    statusFlash = Signal(str, int, str)

    def __init__(self):
        """初始化 UI、默认配置、TTS/语音功能与线程资源。"""
        super().__init__()
        self.setWindowTitle("YOLO 参数配置 - PySide6")
        self.resize(760, 620)

        self.defaults = YOLOConfig()
        self.worker: DetectionWorker | None = None
        self.stop_event: threading.Event | None = None
        # 显示文本 -> 实际视频源（int 索引或 "video=友好名" 字符串）
        self._simple_cam_name_to_index: dict[str, int | str] = {}
        self._adv_cam_name_to_index: dict[str, int | str] = {}

        self._build_ui()
        # 连接语音信号到专用处理逻辑（在其中调用 on_start/on_stop，并做语音播报）
        self.voiceStart.connect(self._on_voice_start)
        self.voiceStop.connect(self._on_voice_stop)
        # 将跨线程状态刷新通过信号切换到主线程
        self.statusFlash.connect(self._flash_status)
        # 初始化 TTS 管理器（串行播报，避免重叠或被内部聚合导致静音）
        self._tts = TTSManager(tts_module=_tts_mod, logger=_tts_log, dup_window=DUP_SPEECH_WINDOW)
        self._tts.start()
        set_speaker(self._tts.speak)
        # 初始化语音控制（若可用）
        self._voice_ctrl = None
        self._init_voice_control()

    # ---------------- UI -----------------
    def _build_ui(self) -> None:
        """构建主界面结构：状态栏、模式区、一般/高级参数与按钮区。"""
        layout = QVBoxLayout(self)
        self._init_status_bar(layout)
        self._init_mode_box(layout)
        self._init_general_box(layout)
        self._init_adv_box(layout)
        self._init_buttons(layout)
        self._update_mode_visibility()

    def _init_status_bar(self, layout: QVBoxLayout) -> None:
        """初始化状态栏并设置默认消息。"""
        self.status = QStatusBar()
        self.status.setSizeGripEnabled(False)
        self.status.setFixedHeight(STATUS_BAR_HEIGHT)
        self.status.setStyleSheet("QStatusBar {padding-left:6px; font-size:12px;}")
        layout.addWidget(self.status)
        self.status.showMessage("就绪")

    def _flash_status(self, msg: str, ms: int = 2000, fallback: str = "就绪") -> None:
        """短暂显示状态栏消息，一段时间后若消息未被其他更新覆盖，则恢复为 fallback。"""
        self.status.showMessage(msg)
        expected = msg

        def _restore() -> None:
            # 仅当消息仍是这条临时消息时才恢复，避免覆盖其他更新
            with contextlib.suppress(Exception):
                if self.status.currentMessage() == expected:
                    self.status.showMessage(fallback)
        QTimer.singleShot(ms, _restore)

    def _init_mode_box(self, layout: QVBoxLayout) -> None:
        """初始化模式选择（一般/高级）。"""
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
        """初始化“一般”参数区（模型权重与摄像头）。"""
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
        """初始化“高级参数”区。"""
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
        # 播报/节流/黄闪参数
        self._add_announcer_rows(form)
        layout.addWidget(self.adv_box)

    def _add_model_row(self, form: QFormLayout) -> None:
        """添加模型权重行。"""
        self.model_line = QLineEdit(self.defaults.model_path)
        btn_model = QPushButton("选择...")
        btn_model.clicked.connect(self._choose_model)
        mh = QHBoxLayout()
        mh.addWidget(self.model_line)
        mh.addWidget(btn_model)
        form.addRow("模型权重", mh)

    def _add_device_row(self, form: QFormLayout) -> None:
        """添加设备行并填充可用设备。"""
        self.device_combo = QComboBox()
        self._populate_devices()
        form.addRow("运算设备", self.device_combo)

    def _add_source_row(self, form: QFormLayout) -> None:
        """添加视频源行。"""
        self.source_line = QLineEdit(str(self.defaults.source))
        btn_source = QPushButton("视频文件...")
        btn_source.clicked.connect(self._choose_source)
        sh = QHBoxLayout()
        sh.addWidget(self.source_line)
        sh.addWidget(btn_source)
        form.addRow("视频源", sh)

    def _add_adv_cam_row(self, form: QFormLayout) -> None:
        """添加高级摄像头列表与刷新按钮。"""
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
        """添加保存目录选择行。"""
        self.save_dir_line = QLineEdit(self.defaults.save_dir)
        btn_dir = QPushButton("目录...")
        btn_dir.clicked.connect(self._choose_dir)
        dh = QHBoxLayout()
        dh.addWidget(self.save_dir_line)
        dh.addWidget(btn_dir)
        form.addRow("保存目录", dh)

    def _add_save_txt_row(self, form: QFormLayout) -> None:
        """添加“保存 txt 标注”开关。"""
        self.save_txt_chk = QCheckBox("保存 txt 标注")
        self.save_txt_chk.setChecked(self.defaults.save_txt)
        form.addRow("保存TXT", self.save_txt_chk)

    def _add_conf_row(self, form: QFormLayout) -> None:
        """添加置信度设置行。"""
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
        """添加输入尺寸行。"""
        self.img_size_line = QLineEdit(
            "" if self.defaults.img_size is None else ",".join(str(i) for i in self.defaults.img_size)
        )
        form.addRow("输入尺寸", self.img_size_line)

    def _add_window_name_row(self, form: QFormLayout) -> None:
        """添加窗口标题行。"""
        self.window_name_line = QLineEdit(self.defaults.window_name)
        form.addRow("窗口标题", self.window_name_line)

    def _add_timestamp_row(self, form: QFormLayout) -> None:
        """添加时间戳格式行。"""
        self.timestamp_fmt_line = QLineEdit(self.defaults.timestamp_fmt)
        form.addRow("时间戳格式", self.timestamp_fmt_line)

    def _add_exit_key_row(self, form: QFormLayout) -> None:
        """添加退出按键行。"""
        self.exit_key_line = QLineEdit(self.defaults.exit_key)
        self.exit_key_line.setMaxLength(2)
        form.addRow("退出按键", self.exit_key_line)

    def _add_show_fps_row(self, form: QFormLayout) -> None:
        """添加“显示 FPS”开关行。"""
        self.show_fps_chk = QCheckBox("显示 FPS")
        self.show_fps_chk.setChecked(self.defaults.show_fps)
        form.addRow("显示FPS", self.show_fps_chk)

    def _add_announcer_rows(self, form: QFormLayout) -> None:
        """添加与语音播报/黄闪判断相关的参数设置行。"""
        # 最小间隔
        self.ann_min_interval_line = QLineEdit(str(getattr(self.defaults, 'ann_min_interval', 1.5)))
        form.addRow("播报最小间隔(秒)", self.ann_min_interval_line)
        # 黄闪窗口
        self.ann_flash_window_line = QLineEdit(str(getattr(self.defaults, 'ann_flash_window', 3.0)))
        form.addRow("黄闪窗口(秒)", self.ann_flash_window_line)
        # 黄闪最少采样
        self.ann_flash_min_events_line = QLineEdit(str(getattr(self.defaults, 'ann_flash_min_events', 6)))
        form.addRow("黄闪最少采样", self.ann_flash_min_events_line)
        # 黄灯占比
        self.ann_flash_yellow_ratio_line = QLineEdit(str(getattr(self.defaults, 'ann_flash_yellow_ratio', 0.9)))
        form.addRow("黄灯占比阈值(0~1)", self.ann_flash_yellow_ratio_line)
        # 黄闪冷却
        self.ann_flash_cooldown_line = QLineEdit(str(getattr(self.defaults, 'ann_flash_cooldown', 5.0)))
        form.addRow("黄闪冷却(秒)", self.ann_flash_cooldown_line)

    def _init_buttons(self, layout: QVBoxLayout) -> None:
        """初始化启动/停止/恢复默认按钮区域。"""
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

    # --------- 语音控制接入 ---------
    def _init_voice_control(self) -> None:
        """初始化关键词语音控制（若依赖与模型可用）。"""
        # 默认模型目录：项目根下 models/vosk-model-small-cn-0.22
        model_dir = pathlib.Path(__file__).resolve().parent / "models" / "vosk-model-small-cn-0.22"
        self._voice_ctrl = VoiceController(model_dir)
        if not self._voice_ctrl.available():
            self.status.showMessage("语音控制未启用（缺少依赖或模型目录）")
            return

        def _on_voice(text: str) -> None:
            """语音回调：检测到关键词时发出对应信号。"""
            t = (text or "").strip()
            _voice_log.debug("检测到关键词: '%s'", t)
            # 简单包含匹配，提升鲁棒性
            if "启动辅助系统" in t:
                _voice_log.debug("发出 voiceStart 信号")
                self.voiceStart.emit()
                # 注意：该回调在监听线程里，不能直接调用 UI；通过信号路由到主线程
                self.statusFlash.emit("语音指令：启动辅助系统", 2000, "就绪")
            elif "关闭辅助系统" in t:
                _voice_log.debug("发出 voiceStop 信号")
                self.voiceStop.emit()
                self.statusFlash.emit("语音指令：关闭辅助系统", 2000, "就绪")

        self._voice_ctrl.start(["启动辅助系统", "关闭辅助系统"], _on_voice)
        _voice_log.debug("语音监听已启动")
        self.status.showMessage("语音控制已启用（可说：启动辅助系统 / 关闭辅助系统）")

    # --------- 语音触发的启动/关闭（带播报） ---------
    def _on_voice_start(self) -> None:
        """语音触发启动：若已在运行则语音播报提示，否则调用按钮同逻辑。"""
        _voice_log.debug("收到语音启动请求，当前是否已运行: %s", bool(self.worker and self.worker.is_alive()))
        if self.worker and self.worker.is_alive():
            # 与按钮提示一致，并做语音播报
            _voice_log.debug("已在运行中 -> 语音提示 '检测已在运行中'")
            self._speak("检测已在运行中")
            return
        # 未运行则走正常启动
        _voice_log.debug("未运行 -> 调用 on_start")
        self.on_start()

    def _on_voice_stop(self) -> None:
        """语音触发关闭：若未在运行则语音播报提示，否则调用按钮同逻辑。"""
        _voice_log.debug("收到语音停止请求，当前是否已运行: %s", bool(self.worker and self.worker.is_alive()))
        if not self.worker or not self.worker.is_alive():
            _voice_log.debug("未在运行 -> 语音提示 '当前没有正在运行的检测'")
            self._speak("当前没有正在运行的检测")
            return
        # 若正在运行，则先语音提示正在关闭，再执行关闭
        _voice_log.debug("运行中 -> 调用 on_stop")
        self.on_stop()

    def _speak(self, text: str) -> None:
        """通过 TTSManager 串行播报，内部自动处理重复抑制与异常。"""
        if not text:
            _tts_log.debug("跳过播报: 文本为空")
            return
        if getattr(self, "_tts", None) is None:
            _tts_log.debug("跳过播报: TTS 管理器未初始化")
            return
        self._tts.speak(text)

    # --------- 设备枚举 ---------
    def _populate_devices(self) -> None:
        """枚举可用运算设备并填入设备下拉框，尽量保持默认项。"""
        self.device_combo.clear()
        self.device_combo.addItems(list_devices(torch))
        idx = self.device_combo.findText(self.defaults.device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)

    # --------- 文件对话框 ---------
    def _choose_model_simple(self):
        """在“一般”模式下选择模型文件并写回文本框。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            filter="模型文件 (*.pt *.onnx *.engine *.tflite);;所有文件 (*.*)",
        )
        if path:
            self.model_line_simple.setText(path)

    def _choose_model(self):
        """在“高级”模式下选择模型文件并写回文本框。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            filter="模型文件 (*.pt *.onnx *.engine *.tflite);;所有文件 (*.*)",
        )
        if path:
            self.model_line.setText(path)

    def _choose_source(self):
        """选择视频源文件并写回文本框（摄像头请在下拉中选择）。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "视频文件",
            filter="视频文件 (*.mp4 *.avi *.mkv);;所有文件 (*.*)",
        )
        if path:
            self.source_line.setText(path)

    def _choose_dir(self):
        """弹出目录选择并写入保存目录输入框。"""
        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if path:
            self.save_dir_line.setText(path)

    # --------- 模式切换 ---------
    def _update_mode_visibility(self):
        """根据模式选择（一般/高级）切换对应区域的可见性。"""
        advanced = self.mode_combo.currentIndex() == 1
        self.adv_box.setVisible(advanced)
        self.general_box.setVisible(not advanced)

    # --------- 恢复默认 ---------
    def on_reset(self):
        """恢复所有控件到默认配置，并尽力匹配默认摄像头项。"""
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
        # 播报参数
        if hasattr(self, 'ann_min_interval_line'):
            self.ann_min_interval_line.setText(str(getattr(self.defaults, 'ann_min_interval', 1.5)))
            self.ann_flash_window_line.setText(str(getattr(self.defaults, 'ann_flash_window', 3.0)))
            self.ann_flash_min_events_line.setText(str(getattr(self.defaults, 'ann_flash_min_events', 6)))
            self.ann_flash_yellow_ratio_line.setText(str(getattr(self.defaults, 'ann_flash_yellow_ratio', 0.9)))
            self.ann_flash_cooldown_line.setText(str(getattr(self.defaults, 'ann_flash_cooldown', 5.0)))
        self.status.showMessage("已恢复默认")

    # --------- 构建 CLI 参数 ---------
    def _build_argv(self, *, advanced: bool) -> list[str]:
        """从当前 UI 读取配置，构造命令行参数列表。"""
        return self._build_argv_advanced() if advanced else self._build_argv_simple()

    def _build_argv_simple(self) -> list[str]:
        """仅打包“一般”模式下的变更项：模型路径与摄像头索引。"""
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

    def _build_argv_advanced(self) -> list[str]:  # noqa: C901 - 逻辑较多但清晰
        """打包“高级”模式下所有可能变更的参数，未变化项将被忽略。"""
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
        # 播报/节流/黄闪参数

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
        return args

    # --------- 启动 / 关闭 ---------
    def on_start(self):
        """启动检测线程：校验未重复启动，构造 argv，更新状态栏与按钮状态。"""
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
        """请求停止后台检测线程，等待当前帧处理完毕后退出。"""
        if not self.worker or not self.worker.is_alive():
            _voice_log.debug("点击停止但未在运行 -> 弹出提示框")
            QMessageBox.information(self, "提示", "当前没有正在运行的检测")
            return
        if self.stop_event:
            _voice_log.debug("停止请求 -> 设置 stop_event")
            self.stop_event.set()
        self.status.showMessage("请求关闭... (等待当前帧) ")

    # --------- 线程回调 ---------
    def _on_finished(self):
        """检测线程结束后的收尾：恢复按钮状态，清空线程引用。"""
        _voice_log.debug("工作线程已结束")
        self.status.showMessage("已结束")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.worker = None
        self.stop_event = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """窗口关闭时清理语音控制与 TTS 管理器，然后继续默认关闭流程。"""
        # 停止语音监听
        try:
            if getattr(self, "_voice_ctrl", None) is not None:
                _voice_log.debug("关闭窗口 -> 停止语音监听")
                self._voice_ctrl.stop()  # type: ignore[union-attr]
        except (AttributeError, RuntimeError):
            logging.exception("关闭语音监听时异常")
        # 停止 TTS 管理器
        try:
            if getattr(self, "_tts", None) is not None:
                _tts_log.debug("关闭窗口 -> 停止 TTS 管理器")
                self._tts.stop()
        except Exception:  # pragma: no cover - 防御性
            logging.exception("关闭 TTS 管理器时异常")
        return super().closeEvent(event)

    # --------- 错误回调 ---------
    def _on_error(self, msg: str):
        """处理检测线程返回的错误：提示用户并重置 UI 状态。"""
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
        mapping: dict[str, int | str],
        current_source,
    ):
        """刷新摄像头下拉框与名称映射，并尽量定位当前默认源。"""
        combo.clear()
        mapping.clear()
        cams = self._enumerate_cameras_with_names()
        if not cams:
            combo.addItem("None")
            combo.setEnabled(False)
            return
        combo.setEnabled(True)
        combo.addItem("None")
        for idx, nm in cams:
            left = f"Camera {idx}"
            friendly = (nm or '').strip()
            # 仅用于显示友好名；实际 source 仍使用 OpenCV 索引
            display = f"{left}  |  {friendly}" if friendly else left
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
    def _on_cam_selected(self, combo: QComboBox, mapping: dict[str, int | str]):
        """当选择摄像头下拉项时，将解析到的索引写回 source_line（若存在）。"""
        if combo.currentIndex() < 0:
            return
        text = combo.currentText().strip()
        if text == "None":
            return
        # 直接从映射表获取索引
        if text in mapping:
            if hasattr(self, "source_line"):
                self.source_line.setText(str(mapping[text]))
            return
        # 兼容旧格式（包含括号索引）或前缀 Camera n
        if "Camera " in text:
            parts = text.split()
            for p in parts:
                if p.isdigit():
                    if hasattr(self, "source_line"):
                        self.source_line.setText(p)
                    return

    # --------- 带名称摄像头枚举 ---------
    def _enumerate_cameras_with_names(self) -> list[tuple[int, str]]:
        """枚举摄像头索引并尽量关联友好名称，不可用时使用默认“Camera n”。"""
        try:
            indices = enumerate_cameras(
                self.defaults.max_cam_index if hasattr(self.defaults, "max_cam_index") else 8
            )
        except (OSError, RuntimeError, ValueError):
            return []
        if not indices:
            return []
        # 优先使用 DirectShow 名称（若可用）
        ds_names = get_directshow_device_names()
        if ds_names:
            pairs: list[tuple[int, str]] = []
            for idx in indices:
                nm = ds_names[idx] if 0 <= idx < len(ds_names) else f"Camera {idx}"
                pairs.append((idx, nm))
            return pairs
        # 次选：保守的 WMI 名称映射（仅唯一设备时）
        name_map = map_indices_to_names(indices, wmi_enum)
        return [(idx, name_map.get(idx, f"Camera {idx}")) for idx in indices]

    # --------- 辅助: 检测名称是否全部为默认并提示 ---------
    def _maybe_warn_generic_names(self, cams: list[tuple[int, str]]):
        """若所有名称均为默认“Camera n”，打印一次提醒（依赖 Windows+pywin32 获取友好名）。"""
        if not cams or not hasattr(self, "status"):
            return
        if all(name.strip() == f"Camera {idx}" for idx, name in cams):
            print("未获取到系统摄像头名称（可选依赖：pygrabber 或 Windows + pywin32）。已使用默认 Camera n。")


def main():
    """Qt 应用入口：创建并展示主窗口。"""
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
