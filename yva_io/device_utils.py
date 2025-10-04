from __future__ import annotations

from typing import Any


def list_devices(torch_mod: Any | None) -> list[str]:
    options: list[str] = ["auto", "cpu"]

    def _detect_cuda() -> list[str]:
        if torch_mod is None:
            return []
        try:
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                count = torch_mod.cuda.device_count()
                return ["cuda"] if count == 1 else [f"cuda:{i}" for i in range(count)]
        except (AttributeError, RuntimeError, OSError):
            return []
        return []

    def _detect_mps() -> list[str]:
        if torch_mod is None:
            return []
        try:
            mps = getattr(torch_mod.backends, "mps", None)
            if mps and mps.is_available():
                return ["mps"]
        except (AttributeError, RuntimeError):
            return []
        return []

    options.extend(_detect_cuda())
    options.extend(_detect_mps())
    return list(dict.fromkeys(options))
