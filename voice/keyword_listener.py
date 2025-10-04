"""
离线关键词监听（特定词汇语音识别）——包内版本

基于 Vosk + sounddevice 实现：
- 仅针对给定关键词/短语进行识别（语法约束，降低误识别率）
- 完全离线；Windows/macOS/Linux 通用
- 提供同步阻塞的 listen() 与后台线程 start()/stop() 两种方式
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import sounddevice as sd
from vosk import KaldiRecognizer, Model

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# 独立日志器，避免污染全局 root logger
_kl_log = logging.getLogger("YVA.KWD")
if not _kl_log.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
    _kl_log.addHandler(_h)
_kl_log.setLevel(logging.DEBUG)
_kl_log.propagate = False


@dataclass
class AudioConfig:
    sample_rate: int = 16000  # Vosk 推荐 16000
    channels: int = 1         # 单声道
    blocksize: int = 8000     # 每次回调帧数（略大可降低 CPU）
    dtype: str = "int16"


# 默认关键词（可在此处集中维护）
DEFAULT_KEYWORDS: list[str] = [
    "启动辅助系统",
    "停止辅助系统",
]


@dataclass
class KeywordOptions:
    device: int | str | None = None
    max_queue: int = 8
    debug: bool = False
    match_contains: bool = True
    use_grammar: bool = True
    trigger_on_partial: bool = False  # 是否允许根据 partial 触发（默认关闭避免短词误触发）
    repeat_cooldown: float = 1.0      # 同一关键词再次触发的最小时间间隔（秒），允许无静音的重复命中
    suppress_final_after_partial: bool = True  # 若已由 partial 触发，则在窗口内抑制随后 final
    partial_final_window: float = 1.2          # partial 与 final 的抑制窗口（秒）


class KeywordListener:
    """基于 Vosk 的本地关键词监听器"""

    def __init__(
        self,
        *,
        model_path: str,
        keywords: Iterable[str] | None = None,
        audio: AudioConfig | None = None,
        options: KeywordOptions | None = None,
    ) -> None:
        # 使用默认关键词（若未传入）
        if not keywords:
            keywords = DEFAULT_KEYWORDS
        self.model = Model(model_path)
        # 构造受限语法（仅允许这些关键词）
        # 传入 JSON 数组字符串，允许包含空词来提升稳定度
        self.grammar = json.dumps(list(dict.fromkeys([str(k).strip() for k in keywords if str(k).strip()])))
        self.cfg = audio or AudioConfig()
        opt = options or KeywordOptions()
        self.device = opt.device
        self.debug = opt.debug
        self.match_contains = opt.match_contains
        self.use_grammar = opt.use_grammar
        self.trigger_on_partial = opt.trigger_on_partial
        self.repeat_cooldown = max(0.0, float(getattr(opt, "repeat_cooldown", 0.0)))
        self.suppress_final_after_partial = bool(getattr(opt, "suppress_final_after_partial", True))
        self.partial_final_window = max(0.0, float(getattr(opt, "partial_final_window", 1.2)))
        # 归一化关键词集合（便于匹配）
        self._keywords = [self._normalize_text(k) for k in keywords]
        self._keywords_set = set(self._keywords)
        # 根据 use_grammar 决定是否传入受限语法
        if self.use_grammar:
            self._rec = KaldiRecognizer(self.model, self.cfg.sample_rate, self.grammar)
        else:
            self._rec = KaldiRecognizer(self.model, self.cfg.sample_rate)
        self._q = queue.Queue(maxsize=opt.max_queue)
        self._stop = threading.Event()
        self._th = None  # will hold threading.Thread
        self._last_partial_hit = None
        self._last_hit_at = {}
        self._last_partial_at = {}

    @staticmethod
    def _normalize_text(text: str) -> str:
        # 去除空白，统一为全字符串比较
        return "".join(str(text).split())

    def _is_hit(self, text: str) -> str | None:
        t = self._normalize_text(text)
        if not t:
            return None
        # 优先精确匹配
        if t in self._keywords_set:
            # 返回标准化后的关键词（与 self._keywords 对齐）
            for k in self._keywords:
                if k == t:
                    return k
            return t
        # 包含匹配仅允许“识别文本包含完整关键词”，不允许“关键词包含识别文本”
        if self.match_contains:
            for k in self._keywords:
                if k and (k in t):
                    return k
        return None

    def _audio_callback(self, indata, _frames, _time, status) -> None:
        if status:
            # 可按需打印或记录 status
            _kl_log.debug("sounddevice status: %s", status)
        try:
            self._q.put(bytes(indata), block=False)
        except queue.Full:
            # 丢弃旧数据，尽量保持实时
            with contextlib.suppress(queue.Empty):
                _ = self._q.get_nowait()
            with contextlib.suppress(queue.Full):
                self._q.put_nowait(bytes(indata))

    def _worker(self, on_keyword: Callable[[str], None] | None) -> None:
        # 打开输入流
        with sd.RawInputStream(
            samplerate=self.cfg.sample_rate,
            blocksize=self.cfg.blocksize,
            device=self.device,
            dtype=self.cfg.dtype,
            channels=self.cfg.channels,
            callback=self._audio_callback,
        ):
            if self.debug:
                try:
                    default_in = None
                    try:
                        default_in = sd.default.device[0]
                    except (TypeError, IndexError, AttributeError):
                        default_in = None
                    dev_info = sd.query_devices(self.device, 'input') if self.device is not None else (
                        sd.query_devices(default_in, 'input') if default_in is not None else {}
                    )
                except (sd.PortAudioError, ValueError, TypeError, OSError):
                    dev_info = {}
                _kl_log.debug("Using input device: %s", dev_info)
                _kl_log.debug(
                    "Recognizer mode: %s, sample_rate=%s, channels=%s",
                    "grammar" if self.use_grammar else "free",
                    self.cfg.sample_rate,
                    self.cfg.channels,
                )
            while not self._stop.is_set():
                data = self._get_next_chunk()
                if data is None:
                    continue
                if self._rec.AcceptWaveform(data):
                    self._handle_final(on_keyword)
                else:
                    self._handle_partial(on_keyword)

    def _get_next_chunk(self) -> bytes | None:
        try:
            return self._q.get(timeout=0.3)
        except queue.Empty:
            return None

    def _handle_final(self, on_keyword: Callable[[str], None] | None) -> None:
        try:
            res = json.loads(self._rec.Result())
        except json.JSONDecodeError:
            res = {}
        text = str(res.get("text", "")).strip()
        if self.debug:
            _kl_log.debug("final: %s", text)
        hit = self._is_hit(text)
        if hit and on_keyword:
            now = time.time()
            # 若 partial 刚刚触发过相同关键词，且在窗口内，则抑制本次 final 触发，避免一次话语触发两次
            if self.suppress_final_after_partial and self._last_partial_hit == hit:
                last_p = float(self._last_partial_at.get(hit, 0.0))
                if (now - last_p) < self.partial_final_window:
                    if self.debug:
                        _kl_log.debug(
                            "final suppressed due to recent partial: %s (%.2fs since partial, window=%.2fs)",
                            hit,
                            now - last_p,
                            self.partial_final_window,
                        )
                    self._last_partial_hit = None
                    return
            last = self._last_hit_at.get(hit, 0.0)
            if now - last >= self.repeat_cooldown:
                self._last_hit_at[hit] = now
                on_keyword(hit)
            elif self.debug:
                _kl_log.debug(
                    "final suppressed by cooldown: %s (%.2fs remaining)",
                    hit,
                    self.repeat_cooldown - (now - last),
                )
        self._last_partial_hit = None

    def _handle_partial(self, on_keyword: Callable[[str], None] | None) -> None:
        try:
            partial = json.loads(self._rec.PartialResult()).get("partial", "").strip()
        except json.JSONDecodeError:
            partial = ""
        if self.debug and partial:
            _kl_log.debug("partial: %s", partial)
        # 如果当前为静音/空片段，重置最近一次 partial 命中，方便后续相同指令再次触发
        if not partial:
            self._last_partial_hit = None
            return
        if not self.trigger_on_partial:
            return
        hit = self._is_hit(partial)
        if hit and on_keyword:
            now = time.time()
            last = self._last_hit_at.get(hit, 0.0)
            should_fire = (hit != self._last_partial_hit) or (now - last >= self.repeat_cooldown)
            if should_fire:
                self._last_partial_hit = hit
                self._last_partial_at[hit] = now
                self._last_hit_at[hit] = now
                on_keyword(hit)
            elif self.debug:
                _kl_log.debug(
                    "partial suppressed by cooldown: %s (%.2fs remaining)",
                    hit,
                    self.repeat_cooldown - (now - last),
                )

    def start(self, on_keyword: Callable[[str], None] | None = None) -> None:
        """在后台线程启动监听"""
        if self._th and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._worker, args=(on_keyword,), daemon=True)
        self._th.start()

    def stop(self) -> None:
        """停止监听并等待线程退出"""
        self._stop.set()
        if self._th and self._th.is_alive():
            self._th.join(timeout=2.0)

    def listen(self, timeout: float | None = None) -> str | None:
        """阻塞式等待一次关键词命中，返回文本；超时返回 None"""
        hit: list[str | None] = [None]
        evt = threading.Event()

        def _cb(t: str) -> None:
            hit[0] = t
            evt.set()

        self.start(on_keyword=_cb)
        ok = evt.wait(timeout)
        self.stop()
        return hit[0] if ok else None


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="离线关键词监听（Vosk）")
    p.add_argument("--model", required=False, default=None, help="Vosk 模型目录路径（解压后的文件夹）")
    p.add_argument(
        "--keywords",
        nargs="*",
        help="关键词列表（留空则使用内置默认关键词）",
    )
    p.add_argument("--device", default=None, help="录音设备索引或名称（留空为默认）")
    p.add_argument("--timeout", type=float, default=None, help="阻塞等待超时（秒）；不传则持续监听")
    p.add_argument("--debug", action="store_true", help="打印调试信息（设备、partial/final 文本）")
    # 关键词包含匹配默认启用，如需关闭可改代码或扩展参数
    p.add_argument("--list-devices", action="store_true", help="列出可用输入设备并退出")
    p.add_argument("--free", action="store_true", help="自由识别模式：取消语法约束（已为默认）")
    p.add_argument("--grammar", action="store_true", help="启用受限语法（非默认）")
    args = p.parse_args()

    if args.list_devices:
        print("默认设备:", sd.default.device)
        print("输入设备列表(仅输入>=1通道):")
        all_devs = sd.query_devices()
        for idx in range(len(all_devs)):
            try:
                info = sd.query_devices(idx)
                info_map = cast("dict[str, Any]", info)
                max_in_val = int(info_map.get("max_input_channels", 0))
                name = str(info_map.get("name", f"device {idx}"))
            except (sd.PortAudioError, ValueError, TypeError, OSError) as exc:
                logging.debug("query device %s failed: %s", idx, exc)
                continue
            if max_in_val >= 1:
                print(f"[{idx}] {name} (inputs={max_in_val})")
        raise SystemExit(0)

    use_keywords = args.keywords or DEFAULT_KEYWORDS
    print("使用的关键词：", ", ".join(use_keywords))

    # 解析模型路径：命令行 > 环境变量 > 常见默认路径 > 交互输入
    model_path = args.model or os.environ.get("VOSK_MODEL")
    if not model_path:
        candidates = [
            Path("models") / "vosk" / "vosk-model-small-cn-0.22"
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                model_path = str(c)
                break
    if not model_path:
        try:
            model_path = input("未提供 --model，请输入 Vosk 模型目录路径（直接回车取消）：").strip()
        except EOFError:
            model_path = ""
    if not model_path:
        print("未提供模型路径。可通过 --model 指定，或设置环境变量 VOSK_MODEL，或将模型解压到 ./models/vosk/vosk-model-small-cn-0.22")
        raise SystemExit(2)

    # 默认使用 free 模式；若传入 --grammar 则启用受限语法
    use_grammar_flag = bool(args.grammar)
    opts = KeywordOptions(
        device=args.device,
        debug=args.debug,
        match_contains=True,
        use_grammar=use_grammar_flag,
    )
    kl = KeywordListener(model_path=model_path, keywords=use_keywords, options=opts)

    if args.timeout is not None:
        print("等待关键词（超时", args.timeout, "秒）...")
        text = kl.listen(timeout=args.timeout)
        print("命中：", text)
    else:
        print("开始监听，按 Ctrl+C 结束。关键词：", ", ".join(use_keywords))
        try:
            def _on_hit(t: str) -> None:
                print("命中关键词:", t)
            kl.start(on_keyword=_on_hit)
            threading.Event().wait()  # 永久阻塞，直到 Ctrl+C
        except KeyboardInterrupt:
            pass
        finally:
            kl.stop()
