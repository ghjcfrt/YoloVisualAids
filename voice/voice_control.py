"""语音关键词监听封装"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

try:
    from .keyword_listener import KeywordListener, KeywordOptions
except (ImportError, OSError, RuntimeError):
    KeywordListener = None
    KeywordOptions = None


class VoiceController:
    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._listener = None

    def available(self) -> bool:
        """是否可用 需关键监听实现与模型目录齐备"""
        return (
            KeywordListener is not None
            and KeywordOptions is not None
            and self._model_dir.exists()
        )

    def start(self, keywords: Iterable[str], on_keyword: Callable[[str], None]) -> None:
        """启动关键字监听 在后台线程回调命中文本"""
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
        """停止监听 并清理内部状态"""
        lst = self._listener
        if lst is not None and hasattr(lst, "stop"):
            lst.stop()
        self._listener = None
