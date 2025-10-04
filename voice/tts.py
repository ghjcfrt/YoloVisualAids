"""本地离线 TTS 工具（Windows 首选 SAPI5），基于 pyttsx3

特性：
- 自动尝试中文语音（按名称/语言/ID 关键词匹配）
- 文本去零宽字符与短时间重复语句“音速轻微抖动”防重复感
- 同步 speak（阻塞）与 speak_async（后台线程）两种调用方式
- 可通过环境变量 YV_TTS_ISOLATED 控制是否“隔离实例”（每次 speak 使用新 engine）
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

_log = logging.getLogger("YVA.TTS")
_lock = threading.RLock()
_DUP_WINDOW = 2.5
ZW_CHARS = {"\u200b", "\u200c", "\u200d", "\u200e", "\u200f"}
_iso_env = os.getenv("YV_TTS_ISOLATED")
_ISOLATED = True if _iso_env is None else _iso_env.strip().lower() in {"1", "true", "on", "yes"}


class _DedupState:
    __slots__ = ("last_text_norm", "last_time", "rate_toggle")

    def __init__(self) -> None:
        self.last_text_norm: str | None = None
        self.last_time: float = 0.0
        self.rate_toggle: int = 0


_DEDUP = _DedupState()
_TL = threading.local()
_ASYNC_TH: threading.Thread | None = None


def _normalize_text_for_dedup(text: str) -> str:
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        if ch in ZW_CHARS:
            continue
        out.append(ch)
    return "".join(out).strip()


def _apply_rate_jitter(engine: pyttsx3.Engine, text: str, explicit_rate: int | None) -> None:
    if explicit_rate is not None:
        engine.setProperty("rate", int(explicit_rate))
        return
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
                base_rate = int(cast("Any", base_rate_obj))
            except (TypeError, ValueError):
                base_rate = 175
    except AttributeError:
        base_rate = 175
    if last_norm == norm and (now - last) < _DUP_WINDOW:
        delta = 1 if _DEDUP.rate_toggle == 0 else -1
        engine.setProperty("rate", max(50, base_rate + delta))
        _DEDUP.rate_toggle ^= 1
    _DEDUP.last_text_norm = norm
    _DEDUP.last_time = now


def _get_engine() -> pyttsx3.Engine:
    eng = getattr(_TL, "engine", None)
    if eng is None:
        eng = pyttsx3.init()
        _TL.engine = eng
    _log.debug("已在线程 %s 创建线程本地 TTS 引擎", threading.current_thread().name)
    return eng


def _reset_engine_for_current_thread() -> None:
    try:
        eng = getattr(_TL, "engine", None)
        if eng is not None:
            with contextlib.suppress(Exception):
                eng.stop()
        if hasattr(_TL, "engine"):
            delattr(_TL, "engine")
    except Exception:
        logging.exception("重置当前线程的 TTS 引擎失败")


def list_voices() -> list[dict[str, Any]]:
    """列出系统可用的语音包信息（id/name/languages/gender/age）。"""
    eng = _get_engine()
    items: list[dict[str, Any]] = []
    voices_obj: object = eng.getProperty("voices")
    iter_voices: Iterable[Any] = cast("Iterable[Any]", voices_obj) if isinstance(voices_obj, Iterable) else ()
    for v in iter_voices:
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
    """尽力选择中文语音的 ID（按常见关键字匹配）。"""
    candidates = list_voices()
    keywords = ("zh", "chinese", "chs", "cn")
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
    """在当前线程中执行一次性 TTS 输出。"""
    if not text:
        return
    with _lock:
        eng = pyttsx3.init()
        with contextlib.suppress(Exception):
            eng.stop()
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
    """同步朗读：阻塞等待直至播放完成。"""
    t = speak_async(text, rate=rate, volume=volume, voice=voice)
    t.join()


def speak_async(
    text: str,
    *,
    rate: int | None = None,
    volume: float | None = None,
    voice: str | None = None,
) -> threading.Thread:
    """异步朗读：创建后台线程执行播放并立即返回线程对象。"""
    def _runner() -> None:
        try:
            _speak_once(text, rate=rate, volume=volume, voice=voice)
        except Exception:
            logging.exception("异步播报失败")

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t
