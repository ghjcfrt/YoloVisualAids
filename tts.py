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

import threading
from collections.abc import Iterable
from functools import lru_cache
from typing import Any, cast

import pyttsx3

_lock = threading.RLock()


@lru_cache(maxsize=1)
def _get_engine() -> pyttsx3.Engine:
    """获取（并缓存）全局 TTS 引擎实例。"""
    return pyttsx3.init()  # Windows 下默认使用 SAPI5


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


def speak(
    text: str,
    *,
    rate: int | None = None,
    volume: float | None = None,
    voice: str | None = None,
) -> None:
    """朗读文本（阻塞直至朗读完成）。

    参数：
    - text: 要朗读的文本。
    - rate: 语速（整数，约 100~200，默认使用系统值）。
    - volume: 音量（0.0~1.0）。
    - voice: 指定语音 id（使用 list_voices 获取）；若未指定，将优先选择中文语音。
    """
    if not text:
        return
    eng = _get_engine()
    with _lock:
        if voice is None:
            zh_id = _pick_zh_voice_id()
            if zh_id:
                eng.setProperty("voice", zh_id)
        else:
            eng.setProperty("voice", voice)
        if rate is not None:
            eng.setProperty("rate", int(rate))
        if volume is not None:
            v = max(0.0, min(1.0, float(volume)))
            eng.setProperty("volume", v)
        eng.say(text)
        eng.runAndWait()


def speak_async(
    text: str,
    *,
    rate: int | None = None,
    volume: float | None = None,
    voice: str | None = None,
) -> threading.Thread:
    """在后台线程朗读文本；返回线程句柄。"""

    def _runner() -> None:
        speak(text, rate=rate, volume=volume, voice=voice)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import sys

    txt = " ".join(sys.argv[1:]).strip()
    if not txt:
        try:
            txt = input("请输入要朗读的文字：").strip()
        except EOFError:
            txt = ""
    if not txt:
        txt = "你好，我是本地语音助手。"
    speak(txt)
