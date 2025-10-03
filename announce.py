"""
语音播报辅助：
- 非红绿灯：按类别中文名与数量组合语句；数量<3 读出具体数量，>=3 只读量词；
- 红绿灯：直接朗读结果中文描述。

可注入 TTS 播报函数，默认使用 tts.speak_async；
并提供全局限流，避免积压导致“物体消失仍在播报”。
"""

from __future__ import annotations

import time
from collections import deque

from coco_labels_cn import coco_labels_cn

# 可注入的播报函数：默认使用 tts.speak_async，若不可用则降级为空操作。
try:
    from tts import speak_async as _default_speak_async
except (ImportError, OSError, RuntimeError):  # pragma: no cover - 依赖可能不存在
    _default_speak_async = None  # type: ignore[assignment]


def _noop(_: str) -> None:
    return None


def _wrap_speak_async():
    if _default_speak_async is None:
        return _noop

    def _adapter(text: str) -> None:
        if _default_speak_async is not None:
            _default_speak_async(text)
    return _adapter


_speak_func = _wrap_speak_async()


def set_speaker(func) -> None:
    """设置自定义播报函数（例如 TTSManager.speak）。"""
    # 避免使用 global 的副作用，这里仍更新模块级状态用于简化使用
    # 注意：这是线程不安全的，应在应用启动时设置一次。
    globals()['_speak_func'] = func


def _measure_word(label_cn: str) -> str:
    """返回该类别合适的量词，默认“个”。"""
    # 常用量词映射
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
        "奶牛": "头",
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


def _count_to_cn(n: int) -> str:
    ONE = 1
    TWO = 2
    if n <= ONE:
        return "一"
    if n == TWO:
        return "两"
    return str(n)


THRESHOLD_EXACT_READ = 3  # 小于该阈值播报具体数量，否则使用量词


def compose_non_tl_phrase(counts: dict[int, int]) -> str | None:
    """将非交通灯的类别计数转换为中文短语。"""
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


def speak_traffic(status: str) -> None:
    # 直接朗读状态；red/yellow/green -> 红灯/黄灯/绿灯
    mapping = {"red": "红灯", "yellow": "黄灯", "green": "绿灯"}
    _speak_func(mapping.get(status, status))


class Announcer:
    """简单去重+全局限流的播报器：
    - 最小间隔内只允许一条播报（不论文本内容是否变化），减少积压；
    - 交通灯颜色变化时仍可即时播报。
    """

    def __init__(
        self,
        min_interval_sec: float = 1.5,
        *,
        flash_window_sec: float | None = None,
        flash_min_events: int | None = None,
        flash_yellow_ratio: float | None = None,
        flash_cooldown_sec: float | None = None,
    ):
        self._last_text: str | None = None
        self._last_t: float = 0.0
        self._min_interval = float(min_interval_sec)
        # 交通灯专用状态
        self._last_tl_status: str | None = None
        self._last_tl_change_t: float = 0.0
        self._last_flash_announce_t: float = 0.0
        # 最近状态历史：保存(时间戳, 状态)
        self._tl_history: deque[tuple[float, str]] = deque(maxlen=120)
        # 黄灯闪烁判定窗口与门限（时间与数量自适应，不依赖帧率）
        self._flash_window_sec = 3.0 if flash_window_sec is None else float(flash_window_sec)
        self._flash_min_events = 6 if flash_min_events is None else int(flash_min_events)  # 窗口内至少 N 次采样
        self._flash_yellow_ratio = 0.9 if flash_yellow_ratio is None else float(flash_yellow_ratio)  # 黄灯占比≥该阈值，且无红/绿
        self._flash_cooldown_sec = 5.0 if flash_cooldown_sec is None else float(flash_cooldown_sec)  # “黄灯闪烁”播报的最小间隔

    def say(self, text: str) -> None:
        now = time.time()
        # 全局限流：在最小间隔内，无论文本是否变化都跳过，避免积压
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
        # 丢弃窗口外数据
        cutoff = now - self._flash_window_sec
        while self._tl_history and self._tl_history[0][0] < cutoff:
            self._tl_history.popleft()

    def _is_flashing_yellow(self, now: float) -> bool:
        # 在窗口内统计（仅统计颜色状态，忽略“颜色不同/无红绿灯/不工作”等非颜色信号）
        window = [(t, s) for (t, s) in self._tl_history if now - t <= self._flash_window_sec]
        statuses = [s for _, s in window if s in {"red", "yellow", "green"}]
        if len(statuses) < self._flash_min_events:
            return False
        # 存在红或绿则不认为是黄闪
        if any(s in {"red", "green"} for s in statuses):
            return False
        yellow_cnt = sum(1 for s in statuses if s == "yellow")
        ratio = yellow_cnt / len(statuses)
        return ratio >= self._flash_yellow_ratio

    def say_traffic(self, status: str) -> None:
        """交通灯播报，具备：
        - 去重与限流；
        - 颜色变化时立即播报；
        - 持续黄灯（黄灯闪烁）特殊播报。
        """
        mapping = {"red": "红灯", "yellow": "黄灯", "green": "绿灯"}
        now = time.time()
        # 记录历史，用于闪烁判定
        self._push_tl_history(status, now)

        is_color = status in {"red", "yellow", "green"}

        # 颜色变化 -> 立即播报（绕过限流）
        if is_color and status != self._last_tl_status:
            self._last_tl_status = status
            self._last_tl_change_t = now
            text = mapping.get(status, status)
            # 直接异步播报，不触发通用文本的去重窗口
            self._last_text = text
            self._last_t = now
            _speak_func(text)
            return

        # 黄灯闪烁判定：窗口内基本全为黄灯，且与上一次“黄灯闪烁”播报相隔足够久
        if (
            is_color
            and status == "yellow"
            and self._is_flashing_yellow(now)
            and (now - self._last_flash_announce_t >= self._flash_cooldown_sec)
        ):
            self._last_flash_announce_t = now
            phrase = "黄灯闪烁"
            # 覆盖去重窗口，确保能够播报
            self._last_text = phrase
            self._last_t = now
            _speak_func(phrase)
            return

        # 其他情况：走通用去重限流
        self.say(mapping.get(status, status))
