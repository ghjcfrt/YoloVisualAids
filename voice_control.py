"""语音关键词监听封装。

将 Vosk 模型与 KeywordListener 初始化与启动封装为独立模块，
主窗体仅需提供回调即可。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

try:
    from keyword_listener import KeywordListener, KeywordOptions
except (ImportError, OSError, RuntimeError):  # pragma: no cover - 依赖可选
    KeywordListener = None  # type: ignore[assignment]
    KeywordOptions = None  # type: ignore[assignment]


class VoiceController:
    """极简语音控制封装。"""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._listener = None

    def available(self) -> bool:
        """返回依赖与模型目录是否就绪。"""
        return (
            KeywordListener is not None
            and KeywordOptions is not None
            and self._model_dir.exists()
        )

    def start(self, keywords: Iterable[str], on_keyword: Callable[[str], None]) -> None:
        """启动关键词监听。"""
        if not self.available() or KeywordListener is None or KeywordOptions is None:
            return
        opts = KeywordOptions(
            debug=True,
            match_contains=True,
            use_grammar=False,
            trigger_on_partial=True,
            repeat_cooldown=0.5,
        )
        listener = KeywordListener(model_path=str(self._model_dir), keywords=list(keywords), options=opts)
        self._listener = listener
        listener.start(on_keyword=on_keyword)

    def stop(self) -> None:
        """停止监听（若已启动）。"""
        lst = self._listener
        if lst is not None and hasattr(lst, "stop"):
            lst.stop()  # type: ignore[attr-defined]
        self._listener = None
