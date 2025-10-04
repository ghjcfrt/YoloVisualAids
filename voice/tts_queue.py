"""TTS 队列管理器 串行消费文本并调用 TTS 引擎播报"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from typing import Protocol, cast


class TTSLike(Protocol):
    def speak(self, text: str) -> None:
        ...


class TTSManager:
    """管理 TTS 队列与后台线程 支持去重窗口与过期丢弃"""
    def __init__(
        self,
        *,
        tts_module: TTSLike | None,
        logger: logging.Logger | None = None,
        dup_window: float = 1.2,
        max_queue: int = 32,
        max_age_sec: float | None = 2.0,
    ) -> None:
        self._tts: TTSLike | None = tts_module
        self._log = logger or logging.getLogger("TTSManager")
        self._dup_window = float(dup_window)
        self._max_age = None if max_age_sec is None else float(max_age_sec)

        self._queue: queue.Queue[tuple[str, float] | None] = queue.Queue(maxsize=max_queue)
        self._thread: threading.Thread | None = None
        self._last_text: str | None = None
        self._last_time: float = 0.0
        self._zw_variants = ("\u200b", "\u200c", "\u200d")
        self._zw_idx = 0
        # 当前播报状态（用于外部判断是否应抑制某些提示语）
        self._is_speaking: bool = False
        self._current_text: str | None = None
        # 基于子串的临时抑制规则：{substr: expire_ts}
        self._suppress_until = cast("dict[str, float]", {})

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self._queue.put_nowait(None)
        t = self._thread
        if t and t.is_alive():
            with contextlib.suppress(Exception):
                t.join(timeout=3)
        self._thread = None

    def speak(self, text: str) -> None:
        if not text:
            return
        if self._tts is None:
            self._log.debug("跳过播报：TTS 模块不可用")
            return
        now = time.time()
        eff = text
        # 若文本命中抑制规则，则直接丢弃
        if self._is_suppressed(eff, now):
            self._log.debug("根据抑制规则丢弃播报: %r", eff)
            return
        if self._last_text == text and (now - self._last_time) < self._dup_window:
            zw = self._zw_variants[self._zw_idx % len(self._zw_variants)]
            self._zw_idx += 1
            eff = text + zw
            self._log.debug("在 %.1fs 内重复文本 -> 追加零宽字符 %r 区分", self._dup_window, zw)
        self._last_text = text
        self._last_time = now
        try:
            self._queue.put_nowait((eff, now))
        except queue.Full:
            with contextlib.suppress(Exception):
                _ = self._queue.get_nowait()
            with contextlib.suppress(Exception):
                self._queue.put_nowait((eff, now))

    # 供外部查询的只读接口
    def is_busy(self) -> bool:
        """是否正在执行播报。"""
        return bool(self._is_speaking)

    def get_current_text(self) -> str | None:
        """返回当前正在播报的原始文本（可能包含零宽字符）。"""
        return self._current_text

    # 抑制与清理接口
    def suppress_substring(self, substr: str, duration_sec: float = 2.0) -> None:
        """在给定时间窗口内抑制包含某子串的播报。"""
        if not substr:
            return
        expire = time.time() + max(0.0, float(duration_sec))
        self._suppress_until[substr] = expire
        self._log.debug("添加抑制规则: %r 至 %.3f", substr, expire)

    def clear_pending_substring(self, substr: str) -> None:
        """移除队列中所有包含子串的待播报项。"""
        if not substr:
            return
        # 将队列全部取出，过滤后再塞回
        buf: list[tuple[str, float]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                # 保留停止标记在队尾
                buf.append(item)  # type: ignore[arg-type]
                continue
            text, ts = item
            if substr in text:
                self._log.debug("清除待播报项: %r", text)
                continue
            buf.append((text, ts))
        for it in buf:
            with contextlib.suppress(Exception):
                self._queue.put_nowait(it)

    def _is_suppressed(self, text: str, now: float | None = None) -> bool:
        if not self._suppress_until:
            return False
        t = time.time() if now is None else now
        expired: list[str] = []
        for k, exp in self._suppress_until.items():
            if t > exp:
                expired.append(k)
            elif k and k in text:
                return True
        for k in expired:
            self._suppress_until.pop(k, None)
        return False

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
            if self._max_age is not None and (time.time() - ts) > self._max_age and not self._queue.empty():
                self._log.debug("丢弃过期播报: %r", text)
                continue
            # 再次检查抑制规则（防止排队期间变为受抑制）
            if self._is_suppressed(text):
                self._log.debug("根据抑制规则跳过播报: %r", text)
                continue
            with contextlib.suppress(Exception):
                self._is_speaking = True
                self._current_text = text
                self._log.debug("开始播报: %r", text)
                try:
                    self._tts.speak(text)
                finally:
                    self._log.debug("播报完成: %r", text)
                    self._is_speaking = False
                    self._current_text = None
