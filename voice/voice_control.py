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
        return (
            KeywordListener is not None
            and KeywordOptions is not None
            and self._model_dir.exists()
        )

    def start(self, keywords: Iterable[str], on_keyword: Callable[[str], None]) -> None:
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
        lst = self._listener
        if lst is not None and hasattr(lst, "stop"):
            lst.stop()
        self._listener = None
