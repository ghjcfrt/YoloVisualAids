"""GUI 检测线程与信号

移除了对根目录 `detection_worker.py` 的依赖，保持相同对外 API：
- WorkerSignals: Qt 信号封装
- DetectionWorker: 后台推理线程
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from detection.api import YOLODetector, load_config_from_args


class WorkerSignals(QObject):
    """工作线程信号封装

    - finished: 检测线程完成信号（正常结束或异常后触发）
    - error(str): 将错误信息发回主线程以便弹窗和状态栏提示
    """

    finished = Signal()
    error = Signal(str)


class DetectionWorker(threading.Thread):
    """后台检测线程：解析参数、创建检测器并在收到停止事件前执行检测"""

    def __init__(self, argv: list[str], signals: WorkerSignals, stop_event: threading.Event):
        """保存 CLI 参数、信号对象与停止事件，并以守护线程方式运行"""
        super().__init__(daemon=True)
        self.argv = argv
        self.signals = signals
        self.stop_event = stop_event

    def run(self) -> None:
        """线程入口：构建配置与检测器并运行，异常通过信号返回"""
        try:
            cfg = load_config_from_args(self.argv)
            det = YOLODetector(cfg)
            det.detect_and_save(stop_event=self.stop_event)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


__all__ = ["DetectionWorker", "WorkerSignals"]
