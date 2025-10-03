"""TTS 队列管理器：串行消费文本并调用 tts 引擎播报。

特点：
- 独立线程+队列，保证播报顺序且不重叠；
- 可选重复文本抑制：短时间内同文追加零宽字符；
- 依赖可选：tts 模块缺失时自动跳过。
"""
from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time


class TTSManager:
    """简易 TTS 播报管理器（线程+队列）。"""

    def __init__(
        self,
        *,
        tts_module: object | None,
        logger: logging.Logger | None = None,
        dup_window: float = 1.2,
        max_queue: int = 32,
        max_age_sec: float | None = 2.0,
    ) -> None:
        self._tts = tts_module
        self._log = logger or logging.getLogger("TTSManager")
        self._dup_window = float(dup_window)
        self._max_age = None if max_age_sec is None else float(max_age_sec)

        # 队列元素为 (text, enqueue_time) 或 None 作为停止信号
        self._queue: queue.Queue[tuple[str, float] | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        # 重复文本抑制
        self._last_text: str | None = None
        self._last_time: float = 0.0
        self._zw_variants = ("\u200b", "\u200c", "\u200d")
        self._zw_idx = 0

    def start(self) -> None:
        """启动后台线程。"""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """请求停止并等待线程退出。"""
        with contextlib.suppress(Exception):
            self._queue.put_nowait(None)
        t = self._thread
        if t and t.is_alive():
            with contextlib.suppress(Exception):
                t.join(timeout=3)
        self._thread = None

    def speak(self, text: str) -> None:
        """入队一个要播报的文本（若 tts 不可用则忽略）。"""
        if not text:
            return
        if self._tts is None:
            self._log.debug("跳过播报：TTS 模块不可用")
            return
        # 重复抑制：短时间内同文追加零宽字符避免被聚合
        now = time.time()
        eff = text
        if self._last_text == text and (now - self._last_time) < self._dup_window:
            zw = self._zw_variants[self._zw_idx % len(self._zw_variants)]
            self._zw_idx += 1
            eff = text + zw
            self._log.debug("在 %.1fs 内重复文本 -> 追加零宽字符 %r 区分", self._dup_window, zw)
        self._last_text = text
        self._last_time = now
        # 若队列满则优先丢弃最旧项，确保最新状态尽快播报
        try:
            self._queue.put_nowait((eff, now))
        except queue.Full:
            with contextlib.suppress(Exception):
                _ = self._queue.get_nowait()
            with contextlib.suppress(Exception):
                self._queue.put_nowait((eff, now))

    # 内部线程：串行消费并调用 tts.speak 阻塞播报
    def _worker(self) -> None:
        while True:
            try:
                item = self._queue.get()
            except (RuntimeError, ValueError):
                self._log.exception("TTS 队列读取异常")
                continue
            if item is None:
                self._log.debug("收到停止信号，TTS 线程将退出")
                break
            if self._tts is None:
                self._log.debug("跳过播报：TTS 模块不可用")
                continue
            text, ts = item
            # 过期丢弃：若消息过旧且队列中还有新消息，跳过当前项
            if self._max_age is not None and (time.time() - ts) > self._max_age and not self._queue.empty():
                self._log.debug("丢弃过期播报: %r", text)
                continue
            with contextlib.suppress(Exception):
                self._log.debug("开始播报: %r", text)
                # 使用同步 speak 以确保严格串行与不重叠
                self._tts.speak(text)  # type: ignore[attr-defined]
                self._log.debug("播报完成: %r", text)
