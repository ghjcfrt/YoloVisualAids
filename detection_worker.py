"""检测线程与信号封装。

将后台检测逻辑从 GUI 代码中分离，便于维护与测试。
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from YOLO_detection import YOLODetector, load_config_from_args


class WorkerSignals(QObject):
    """工作线程信号封装。

    - finished: 检测线程完成信号（正常结束或异常后触发）。
    - error(str): 将错误信息发回主线程以便弹窗和状态栏提示。
    """

    finished = Signal()
    error = Signal(str)


class DetectionWorker(threading.Thread):
    """后台检测线程：解析参数、创建检测器并在收到停止事件前执行检测。"""

    def __init__(self, argv: list[str], signals: WorkerSignals, stop_event: threading.Event):
        """保存 CLI 参数、信号对象与停止事件，并以守护线程方式运行。"""
        super().__init__(daemon=True)
        self.argv = argv
        self.signals = signals
        self.stop_event = stop_event

    def run(self) -> None:
        """线程入口：构建配置与检测器并运行，异常通过信号返回。"""
        try:
            cfg = load_config_from_args(self.argv)
            det = YOLODetector(cfg)
            det.detect_and_save(stop_event=self.stop_event)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:  # pragma: no cover
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
