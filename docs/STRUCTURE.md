# 目录结构优化说明

目标：
- 提供稳定的包入口（yolovisualaids），后续重构时不影响调用方式。
- 保持现有脚本可直接运行，兼容当前工作流。

当前结构（新增部分）：

```
## 目录结构与入口说明

本项目采用模块化组织，核心分为 GUI、检测核心、视觉/交通灯模块、语音与设备工具等部分。

顶层关键文件：
- `main.py`：统一入口，根据参数路由到 GUI/检测/交通灯 CLI
- `pyproject.toml`：依赖与工具配置（uv 源、ruff 规则等）

核心目录结构：

```
app/
  gui.py            # PySide6 GUI，参数配置、摄像头选择、语音集成
  worker.py         # GUI 后台检测线程与 Qt 信号封装

detection/
  core.py           # YOLOConfig/YOLODetector，摄像头枚举、推理与保存
  api.py            # 门面导出（供 GUI/CLI 统一调用）
  cli.py            # 命令行入口（python -m detection.cli）

vision/
  color_detection.py# 交通灯颜色 HSV 判定
  traffic_logic.py  # 交通状态融合逻辑（过街/等待等）
  traffic_mode.py   # 统一运行器（来源选择、ROI、YOLO 自动裁剪）
  traffic_cli.py    # 交通灯 CLI（python -m vision.traffic_cli）
  roi_utils.py      # ROI 工具与图像遍历
  visual_styles.py  # 中文标签与颜色样式

voice/              # 可选：离线关键词与 TTS
  keyword_listener.py
  voice_control.py
  tts.py
  tts_queue.py
  announce.py

yva_io/
  camera_utils.py   # DirectShow/WMI 友好名映射
  camera_name.py    # Windows WMI 摄像头名称查询
  device_utils.py   # 设备列表（CUDA/MPS/CPU）

models/             # 放置模型（例如 yolo11n.pt、vosk 模型目录）
results/            # 运行输出（帧、txt、traffic_lights 裁剪等）
docs/STRUCTURE.md   # 本说明
```

可运行入口：
- GUI：`python .\main.py` 或 `python -m app.gui`
- 检测 CLI：`python .\main.py detect ...` 或 `python -m detection.cli ...`
- 交通灯 CLI：`python .\main.py traffic ...` 或 `python -m vision.traffic_cli ...`

命令行与环境变量约定：
- 所有检测参数既可通过命令行提供，也可用 `YV_` 前缀环境变量覆盖默认值（命令行优先）。

输出组织：
- `results/frame_{id}_{ts}.jpg|.txt`：每帧可视化与 YOLO 标签（可选）
- `results/traffic_lights/tl_{frameId}_{boxIndex}_{ts}_{color}.jpg`：自动裁剪的交通灯 ROI

备注：
- Windows 下如需摄像头友好名称，请安装 `pywin32`（WMI 查询）。
- CUDA 使用请确保本机驱动与 `torch==2.5.1+cu121` 兼容；无需 CUDA 可改装 CPU 版 torch。
