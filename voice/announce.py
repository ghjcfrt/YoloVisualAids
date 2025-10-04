"""播报工具与交通灯播报逻辑

封装对象数量播报与交通灯状态播报 支持最小间隔与黄闪识别
"""

from __future__ import annotations

import time
from collections import deque

try:
    from .tts import speak_async as _speak_async
except (ImportError, OSError, RuntimeError):  # 可选依赖缺失
    _speak_async = None

try:
    # 包内提供中文标签映射
    from detection.coco_labels_cn import coco_labels_cn
except (ImportError, OSError):  # 极端情况下缺失则提供空映射
    coco_labels_cn = {}


def _noop(_: str) -> None:
    return None


_speak_func = _speak_async or _noop


def set_speaker(func) -> None:
    """设置自定义播报函数 例如 TTSManager.speak 建议在应用启动时调用一次"""
    globals()["_speak_func"] = func


def speak_traffic(status: str) -> None:
    mapping = {"red": "红灯", "yellow": "黄灯", "green": "绿灯"}
    _speak_func(mapping.get(status, status))


def _measure_word(label_cn: str) -> str:
    by_name = {
        "人": "名",
        "小汽车": "辆",
        "自行车": "辆",
        "摩托车": "辆",
        "公交车": "辆",
        "卡车": "辆",
        "火车": "列",
        "船": "艘",
        "飞机": "架",
        "狗": "只",
        "猫": "只",
        "鸟": "只",
        "马": "匹",
        "羊": "只",
            """带去重与节流以及黄灯闪烁识别的播报器"""
        "大象": "头",
        "斑马": "只",
        "长颈鹿": "只",
        "盆栽植物": "盆",
        "瓶子": "个",
        "椅子": "把",
        "电视": "台",
        "笔记本电脑": "台",
        "手机": "部",
    }
    return by_name.get(label_cn, "个")


_ONE = 1
_TWO = 2


def _count_to_cn(n: int) -> str:
    if n <= _ONE:
        return "一"
    if n == _TWO:
        return "两"
    return str(n)


THRESHOLD_EXACT_READ = 3  # 小于该阈值播报具体数量，否则使用量词


def compose_non_tl_phrase(counts: dict[int, int]) -> str | None:
    parts: list[str] = []
    for cls_id, cnt in counts.items():
        if cnt <= 0:
            continue
        name = coco_labels_cn.get(cls_id)
        if not name:
            continue
        mw = _measure_word(name)
        if cnt < THRESHOLD_EXACT_READ:
            parts.append(f"{_count_to_cn(cnt)}{mw}{name}")
        else:
            parts.append(f"多{mw}{name}")
    if not parts:
        return None
    return "，".join(parts)


def speak_non_tl(counts: dict[int, int], prefix: str | None = "检测到") -> None:
    phrase = compose_non_tl_phrase(counts)
    if not phrase:
        return
    text = f"{prefix}：{phrase}" if prefix else phrase
    _speak_func(text)


class Announcer:
    """带去重与节流以及黄灯闪烁识别的播报器"""

    def __init__(
        self,
        min_interval_sec: float = 1.5,
        *,
        flash_window_sec: float | None = None,
        flash_min_events: int | None = None,
        flash_yellow_ratio: float | None = None,
        flash_cooldown_sec: float | None = None,
    ) -> None:
        self._last_text: str | None = None
        self._last_t: float = 0.0
        self._min_interval = float(min_interval_sec)
        # 交通灯闪烁/变化相关
        self._last_tl_status: str | None = None
        self._last_tl_change_t: float = 0.0
        self._last_flash_announce_t: float = 0.0
        self._tl_history: deque[tuple[float, str]] = deque(maxlen=120)
        self._flash_window_sec = 3.0 if flash_window_sec is None else float(flash_window_sec)
        self._flash_min_events = 6 if flash_min_events is None else int(flash_min_events)
        self._flash_yellow_ratio = 0.9 if flash_yellow_ratio is None else float(flash_yellow_ratio)
        self._flash_cooldown_sec = 5.0 if flash_cooldown_sec is None else float(flash_cooldown_sec)

    def say(self, text: str) -> None:
        now = time.time()
        if now - self._last_t < self._min_interval:
            return
        self._last_text = text
        self._last_t = now
        _speak_func(text)

    def say_non_tl(self, counts: dict[int, int]) -> None:
        phrase = compose_non_tl_phrase(counts)
        if not phrase:
            return
        self.say(f"检测到：{phrase}")

    def _push_tl_history(self, status: str, now: float) -> None:
        self._tl_history.append((now, status))
        cutoff = now - self._flash_window_sec
        while self._tl_history and self._tl_history[0][0] < cutoff:
            self._tl_history.popleft()

    def _is_flashing_yellow(self, now: float) -> bool:
        window = [(t, s) for (t, s) in self._tl_history if now - t <= self._flash_window_sec]
        statuses = [s for _, s in window if s in {"red", "yellow", "green"}]
        if len(statuses) < self._flash_min_events:
            return False
        if any(s in {"red", "green"} for s in statuses):
            return False
        yellow_cnt = sum(1 for s in statuses if s == "yellow")
        ratio = yellow_cnt / len(statuses)
        return ratio >= self._flash_yellow_ratio

    def say_traffic(self, status: str) -> None:
        mapping = {"red": "红灯", "yellow": "黄灯", "green": "绿灯"}
        now = time.time()
        self._push_tl_history(status, now)
        is_color = status in {"red", "yellow", "green"}

        if is_color and status != self._last_tl_status:
            self._last_tl_status = status
            self._last_tl_change_t = now
            text = mapping.get(status, status)
            self._last_text = text
            self._last_t = now
            _speak_func(text)
            return

        if (
            is_color
            and status == "yellow"
            and self._is_flashing_yellow(now)
            and (now - self._last_flash_announce_t >= self._flash_cooldown_sec)
        ):
            self._last_flash_announce_t = now
            phrase = "黄灯闪烁"
            self._last_text = phrase
            self._last_t = now
            _speak_func(phrase)
            return

        self.say(mapping.get(status, status))
