# YoloVisualAids —— 基于 YOLOv11 的实时检测与交通灯颜色识别

YoloVisualAids 是一个基于 Ultralytics YOLOv11 的实时目标检测工具，提供图形界面与命令行两种使用方式。支持自动/手动 ROI 的交通灯颜色识别（红/黄/绿），并将检测结果帧、裁剪与可选的 YOLO txt 标签保存到本地。

主要场景：摄像头实时检测、视频离线检测、交通灯颜色判别、Windows 下摄像头友好名称枚举。


## 功能特性

- 实时检测与保存
  - 设备自动选择（auto/cuda/cpu/mps）
  - 每帧保存到 `results/`，可选保存 YOLO txt
  - 可叠加 FPS 显示（开关可控）
- GUI 参数配置（`PySide6`）
  - 一般/高级两种模式（一般模式只改模型与摄像头，高级可配置全部参数）
  - 后台线程推理、可随时停止、恢复默认
  - 摄像头友好名称下拉与刷新（Windows 使用 WMI，需 `pywin32`）
- 交通灯颜色识别
  - HSV 阈值分割 + 面积/亮度综合评分，鲁棒识别红/黄/绿
  - 支持手动 ROI（交互框选）或 YOLO 自动裁剪 ROI
  - 支持图片/目录/摄像头三种来源
  - 自动保存交通灯 ROI 裁剪到 `results/traffic_lights/`
- 环境变量与命令行双配置
  - 环境变量前缀 `YV_`，命令行参数优先
- 摄像头枚举与降噪
  - 枚举 `0..N-1` 摄像头，发现可用后遇到连续失败可提前结束
  - 可在“枚举阶段”抑制 OpenCV 低层错误日志
\n+- 语音关键词识别（离线，可选）
  - 使用 Vosk 本地 ASR + 受限语法，仅识别指定关键词（更稳）
  - 独立模块 `keyword_listener.py`，便于在你的业务中回调触发


## 目录结构（节选）

  GUI 入口位于包内：`yolovisualaids/app/gui.py`
- `yolovisualaids/detection/core.py`：核心检测与参数装载（命令行 + 环境变量）
- `run_detect.py`：检测模式 CLI 入口
- `yolovisualaids/vision/traffic_mode.py`：交通灯测试模式（图片/目录/摄像头 + 手动/自动 ROI）
- `run_traffic.py`：交通灯模式 CLI 入口
- `yolo_utils.py`：设备选择、尺寸解析、YOLO 自动检测封装
- `color_detction.py`：HSV 颜色识别
- `visual_styles.py`：中文标签与颜色样式
- `roi_utils.py`：ROI 选择与工具
- `camera_name.py`：Windows WMI 摄像头枚举（友好名称）
- `results/`：检测输出目录（运行时生成）


## 环境要求

- 操作系统：Windows 10/11（推荐）。Linux/macOS 也可运行 CLI；MPS 需 macOS 支持
- Python：3.11+
- GPU：可选；若使用 CUDA，请确保本机驱动与 `torch==2.5.1+cu121` 匹配
- 可选组件：
  - 摄像头友好名称：`pywin32`


## 安装与依赖

项目使用 `pyproject.toml` 管理依赖，推荐使用 [uv](https://docs.astral.sh/uv/) 安装：

```powershell
# 1) 安装 uv
python -m pip install -U uv

# 2) 同步依赖（已指定 pytorch-cu121 索引）
uv sync

# 3) 验证环境
uv run python -V
```

说明：
- `pyproject.toml` 已将 `torch/torchvision` 指向 CUDA 12.1 轮子。
  - 若无 NVIDIA GPU 或不需 CUDA，可改为 CPU 版（移除自定义索引并安装 CPU 轮子）。
- 仅 CLI 使用时，可不安装 `PySide6`（若用 `uv sync` 会按 `pyproject.toml` 全量安装）。

不使用 uv 的简单方式（需已准备好合适的 torch/torchvision）：

```powershell
python -m pip install -U ultralytics PySide6 pywin32
```

可选：语音关键词识别所需依赖

```powershell
uv sync  # 已包含 vosk 与 sounddevice 依赖
```


## 快速开始

### 1) 图形界面（GUI）

```powershell
uv run python -m yolovisualaids.app
```

说明：
- 一般/高级两种模式，摄像头支持友好名称显示与刷新。
- 启动/停止/恢复默认；结果帧保存到 `results/`，交通灯裁剪保存到 `results/traffic_lights/`。


### 2) 命令行实时检测（YOLO）

```powershell
uv run python -m yolovisualaids.detection --model models/yolo/yolo11n.pt --source 0 --conf 0.5 --save-txt
```

常用参数（更多见下文“参数与环境变量”）：
- `--model` 模型权重（默认 `models/yolo/yolo11n.pt`）
- `--device` 设备：`auto`/`cuda`/`cuda:N`/`cpu`/`mps`
- `--source` 视频源：摄像头索引（如 0）或视频文件路径
- `--save-dir` 输出目录（默认 `results`）
- `--save-txt` 保存 YOLO txt 标签
- `--conf` 置信度阈值（0~1）
- `--img-size` 推理尺寸：`640` 或 `640,640`；留空表示以原始帧尺寸为目标
- `--window-name`/`--timestamp-fmt`/`--exit-key`/`--no-fps` 等

键位：窗口聚焦时按 `q`（或自定义 `--exit-key`）退出。


### 3) 交通灯颜色识别（图片/目录/摄像头）

手动 ROI（交互框选）或 YOLO 自动裁剪 ROI 二选一：

```powershell
# 新的模块入口（推荐）
uv run python -m yolovisualaids.vision.traffic_cli --image .\traffic_img\demo.jpg
uv run python -m yolovisualaids.vision.traffic_cli --dir .\traffic_img
uv run python -m yolovisualaids.vision.traffic_cli --cam 0

# 自动 ROI（YOLO）示例（可叠加 --save-crops 保存裁剪）
uv run python -m yolovisualaids.vision.traffic_cli --image .\traffic_img\demo.jpg --auto --model models/yolo/yolo11n.pt --conf 0.5 --device auto
```

可选参数：
- `--roi X Y W H` 指定矩形 ROI；不传则交互选择
- `--auto` 启用 YOLO 自动检测并裁剪（默认处理 COCO 类别 9：traffic light）
- `--class-id` 目标类别（默认 9），`--first` 仅取 1 个最佳框
- `--img-size`/`--device`/`--conf` 同 YOLO 检测
- `--save-crops DIR` 将自动裁剪的 ROI 保存到指定目录

输出：
- 识别结果绘制在窗口与控制台；
- 自动模式的裁剪保存于 `results/traffic_lights/` 或 `--save-crops` 指定目录。


### 4) 语音关键词识别（离线）

用于只识别一小组“命令词”，离线、稳定、低资源：

1) 下载并解压 Vosk 中文模型（示例：vosk-model-small-cn-0.22）：
   - https://alphacephei.com/vosk/models
2) 运行监听器（麦克风权限需开启）：

```powershell
uv run python -m yolovisualaids.voice.keyword_listener --model ".\\models\\vosk\\vosk-model-small-cn-0.22" --keywords 开始 停止 退出
```

集成到你的代码：

```python
from yolovisualaids.voice.keyword_listener import KeywordListener

kl = KeywordListener(
  model_path=".\\models\\vosk\\vosk-model-small-cn-0.22",
  keywords=["开始", "停止", "保存图片", "退出"],
)

def on_hit(text: str) -> None:
  print("命中关键词:", text)
  # TODO: 在这里触发你的业务逻辑，例如：开始/停止检测、保存当前帧等

kl.start(on_keyword=on_hit)
# ...你的主循环...
# 退出前：
kl.stop()
```


## 参数与环境变量

所有参数既可由命令行传入，也可使用环境变量覆盖默认值（前缀 `YV_`）：

- `MODEL_PATH` → `--model`（默认 `models/yolo/yolo11n.pt`）
- `DEVICE` → `--device`（默认 `auto`）
- `SOURCE` → `--source`（默认 `0`，纯数字且长度<6 视为摄像头索引）
- `SAVE_DIR` → `--save-dir`（默认 `results`）
- `SAVE_TXT` → `--save-txt`（布尔）
- `SELECT_CAMERA` → `--select-camera`（启动时交互选择摄像头）
- `MAX_CAM_INDEX` → `--max-cam`（摄像头枚举最大索引，默认 8）
- `CONF` → `--conf`（默认 0.5）
- `IMG_SIZE` → `--img-size`（如 `640` 或 `640,640`；留空表示用原始帧尺寸）
- `WINDOW_NAME` → `--window-name`（默认 `YOLOv11 Detection`）
- `TIMESTAMP_FMT` → `--timestamp-fmt`（默认 `%Y%m%d_%H%M%S`）
- `EXIT_KEY` → `--exit-key`（默认 `q`）
- `SHOW_FPS` → `--no-fps`（布尔，命令行为“关闭”开关）
- `QUIET_CV` → `--quiet-cv`（抑制 OpenCV 摄像头错误日志）
- `CAM_FAIL_LIMIT` → `--cam-fail-limit`（枚举连续失败上限，默认 3）

摄像头相关额外变量：
- `YV_SUPPRESS_ENUM_ERRORS=1` 在枚举阶段临时降低 OpenCV 日志级别（默认开启）

示例（PowerShell）：

```powershell
$env:YV_MODEL_PATH = ".\\models\\yolo\\yolo11n.pt"
$env:YV_SOURCE = "0"
$env:YV_CONF = "0.45"
uv run python -m yolovisualaids.detection --save-txt
```


## 摄像头选择与枚举

- GUI：下拉框显示友好名称（Windows），可“刷新”；选中后同步到“视频源”。
- CLI：
  - 使用整型索引：`--source 0`
  - 通过 `--select-camera` 启动时交互选择
  - 枚举策略：发现至少 1 个可用摄像头后，若连续失败次数达到 `CAM_FAIL_LIMIT` 则提前结束，提高启动速度。


## 输出与文件组织

- `results/frame_{id}_{timestamp}.jpg`：每帧可视化结果
- `results/frame_{id}_{timestamp}.txt`：可选 YOLO txt（中心点与宽高为归一化值）
- `results/traffic_lights/tl_{frameId}_{boxIndex}_{timestamp}_{color}.jpg`：交通灯 ROI 裁剪


## 常见问题（FAQ）

1) 启动报 CUDA/torch 相关错误？
- 使用 `--device cpu` 强制 CPU；或根据本地环境安装匹配的 `torch/torchvision`（CUDA 版本需与驱动匹配）。

2) 打不开摄像头或黑屏？
- 确认 `--source` 索引正确，尝试换 `0/1/2`；关闭占用摄像头的软件；在 Windows 设备管理器检查摄像头是否可用。

3) GUI 无法启动或报 Qt 相关异常？
- 确认已安装 `PySide6`；无显示环境时请用命令行模式运行。

4) Windows 下摄像头没有显示友好名称？
- 安装 `pywin32`（`camera_name.py` 使用 WMI），否则显示 `Camera n`。

5) 交通灯未检测到或颜色不稳定？
- 调整 `--conf` 与 `--img-size`，或使用手动 ROI；环境光很暗/很亮时可适当调整位置与尺寸。

6) 语音关键词识别没反应或报设备错误？
- 确认已安装依赖并下载模型；在“系统-隐私-麦克风”里允许应用访问；必要时在命令行指定设备索引 `--device`。


## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)


—— 如有问题或建议，欢迎提交 Issue。

