"""
本地离线 TTS 工具（Windows 首选 SAPI5），基于 pyttsx3。

功能：
- speak(text): 朗读传入文本（优先选择中文语音，若无则使用默认语音）。
- list_voices(): 列出系统可用语音。
- speak_async(text): 后台线程朗读。

注意：
- 为了简单可靠，默认使用阻塞式朗读（runAndWait）。如需非阻塞，请用 speak_async。
- 在同一进程内重复调用是安全的（内部有锁）。
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time as _time
from collections.abc import Iterable
from typing import Any, cast

import pyttsx3

# Scoped logger to align with app logs
_log = logging.getLogger("YVA.TTS")

_lock = threading.RLock()
_DUP_WINDOW = 2.5  # seconds: window to consider texts as duplicates
ZW_CHARS = {"\u200b", "\u200c", "\u200d", "\u200e", "\u200f"}

# 可选“隔离模式”：每次播报都新建并销毁引擎，等效于用户的 safe_speak。
# 默认开启（更稳），如需关闭可设置 YV_TTS_ISOLATED=0/false。
_iso_env = os.getenv("YV_TTS_ISOLATED")
_ISOLATED = True if _iso_env is None else _iso_env.strip().lower() in {"1", "true", "on", "yes"}


class _DedupState:
    __slots__ = ("last_text_norm", "last_time", "rate_toggle")

    def __init__(self) -> None:
        self.last_text_norm: str | None = None
        self.last_time: float = 0.0
        self.rate_toggle: int = 0  # 0 -> +1, 1 -> -1


_DEDUP = _DedupState()

# 每线程持有独立的 pyttsx3 引擎，避免跨线程复用 SAPI5 导致后续静音
_TL = threading.local()

# 保留占位（不再使用队列/单例线程策略，避免跨线程复用引擎引发静音）
_ASYNC_TH: threading.Thread | None = None


def _normalize_text_for_dedup(text: str) -> str:
    # 去掉常见零宽字符，尽量贴近引擎实际发声文本
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        if ch in ZW_CHARS:
            continue
        out.append(ch)
    return "".join(out).strip()


def _apply_rate_jitter(engine: pyttsx3.Engine, text: str, explicit_rate: int | None) -> None:
    """在重复文本短时间内触发时，轻微抖动语速，避免底层去重。"""
    if explicit_rate is not None:
        # 调用方已指定语速，不做抖动
        engine.setProperty("rate", int(explicit_rate))
        return
    # 未指定语速时，根据重复情况轻微抖动
    now = _time.time()
    norm = _normalize_text_for_dedup(text)
    last = _DEDUP.last_time
    last_norm = _DEDUP.last_text_norm
    try:
        base_rate_obj = engine.getProperty("rate")
        if isinstance(base_rate_obj, int):
            base_rate = base_rate_obj
        else:
            try:
                base_rate = int(base_rate_obj)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                base_rate = 175
    except AttributeError:
        base_rate = 175  # 兜底默认
    if last_norm == norm and (now - last) < _DUP_WINDOW:
        delta = 1 if _DEDUP.rate_toggle == 0 else -1
        engine.setProperty("rate", max(50, base_rate + delta))
        _DEDUP.rate_toggle ^= 1
    # 更新状态
    _DEDUP.last_text_norm = norm
    _DEDUP.last_time = now


def _get_engine() -> pyttsx3.Engine:
    """获取当前线程专属的 TTS 引擎实例；若无则在本线程创建。"""
    eng = getattr(_TL, "engine", None)
    if eng is None:
        eng = pyttsx3.init()  # Windows 下默认使用 SAPI5
        _TL.engine = eng  # type: ignore[attr-defined]
        _log.debug("created thread-local TTS engine in thread %s", threading.current_thread().name)
    return eng


def _reset_engine_for_current_thread() -> None:
    """丢弃当前线程的引擎实例，供异常后重建。"""
    try:
        eng = getattr(_TL, "engine", None)
        if eng is not None:
            with contextlib.suppress(Exception):
                eng.stop()
        if hasattr(_TL, "engine"):
            delattr(_TL, "engine")
    except Exception:
        logging.exception("reset engine for current thread failed")


def list_voices() -> list[dict[str, Any]]:
    """返回系统可用语音的信息列表。"""
    eng = _get_engine()
    items: list[dict[str, Any]] = []
    # pyttsx3 的 getProperty("voices") 在类型标注上为 object，这里做显式 Iterable 判定
    voices_obj: object = eng.getProperty("voices")
    iter_voices: Iterable[Any] = cast("Iterable[Any]", voices_obj) if isinstance(voices_obj, Iterable) else ()

    for v in iter_voices:
        # pyttsx3 的 languages 可能是 bytes[] 或 str[]，尽量转成可读字符串
        langs: list[str] = []
        raw_obj = getattr(v, "languages", None)
        raw_iter: Iterable[Any] = (
            cast("Iterable[Any]", raw_obj)
            if isinstance(raw_obj, Iterable) and not isinstance(raw_obj, (str, bytes, bytearray))
            else ()
        )

        for it in raw_iter:
            try:
                if isinstance(it, (bytes, bytearray)):
                    langs.append(bytes(it).decode("utf-8", errors="ignore"))
                else:
                    langs.append(str(it))
            except (UnicodeDecodeError, AttributeError, TypeError):
                langs.append(str(it))
        items.append(
            {
                "id": getattr(v, "id", None),
                "name": getattr(v, "name", None),
                "languages": langs,
                "gender": getattr(v, "gender", None),
                "age": getattr(v, "age", None),
            }
        )
    return items


def _pick_zh_voice_id() -> str | None:
    """尽量选择中文语音；若找不到则返回 None。"""
    candidates = list_voices()
    # 常见中文标识包含 zh、chinese、chs 等
    keywords = ("zh", "chinese", "chs", "cn")
    # 先按 name 再按 languages/id 匹配
    for key in keywords:
        key_lower = key.lower()
        for v in candidates:
            name = str(v.get("name") or "").lower()
            if key_lower in name:
                return str(v.get("id"))
    for key in keywords:
        key_lower = key.lower()
        for v in candidates:
            langs = ",".join(v.get("languages") or []).lower()
            vid = str(v.get("id") or "").lower()
            if key_lower in langs or key_lower in vid:
                return str(v.get("id"))
    return None


def _speak_once(
    text: str,
    *,
    rate: int | None,
    volume: float | None,
    voice: str | None,
) -> None:
    """单次播报：每次新建引擎 -> 设置参数 -> runAndWait -> stop。

    仅供后台线程使用，避免主线程直接驱动引擎。
    """
    if not text:
        return
    with _lock:
        eng = pyttsx3.init()
        # 先尝试清理可能残留的队列，避免后续调用被静默吞掉
        with contextlib.suppress(Exception):
            eng.stop()
        # 设置属性与发声
        if voice is None:
            zh_id = _pick_zh_voice_id()
            if zh_id:
                eng.setProperty("voice", zh_id)
        else:
            eng.setProperty("voice", voice)
        _apply_rate_jitter(eng, text, rate)
        if volume is not None:
            v = max(0.0, min(1.0, float(volume)))
            eng.setProperty("volume", v)
        eng.say(text)
        eng.runAndWait()
        with contextlib.suppress(Exception):
            eng.stop()


def speak(
    text: str,
    *,
    rate: int | None = None,
    volume: float | None = None,
    voice: str | None = None,
) -> None:
    """朗读文本（阻塞直至朗读完成）。内部通过新线程调用，避免主线程直接驱动引擎。"""
    t = speak_async(text, rate=rate, volume=volume, voice=voice)
    t.join()


# 旧的队列式异步策略已移除


def speak_async(
    text: str,
    *,
    rate: int | None = None,
    volume: float | None = None,
    voice: str | None = None,
) -> threading.Thread:
    """在新线程中调用 speak（每次独立引擎/线程，最稳妥）。"""

    def _runner() -> None:
        try:
            _speak_once(text, rate=rate, volume=volume, voice=voice)
        except Exception:
            logging.exception("speak_async failed")

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    # 简单自测：隔离模式默认开启，可持续发声
    def _safe_speak(text: str) -> None:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    _safe_speak('第一次测试')
    print("1")
    _safe_speak('第二次测试')
    print("2")
    _safe_speak('第三次测试')
    print("3")
