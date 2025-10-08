# YoloVisualAids —— 基于 YOLOv11 的实时检测与交通灯颜色识别

YoloVisualAids 是一个基于 Ultralytics YOLOv11 的实时目标检测工具，提供图形界面与命令行两种使用方式。支持自动/手动 ROI 的交通灯颜色识别（红/黄/绿），并将检测结果帧、裁剪与可选的 YOLO txt 标签保存到本地。

## 简介

YoloVisualAids 是一个基于 Ultralytics YOLOv11 的实时目标检测与交通灯颜色识别项目，支持 GUI 与 CLI 两种使用方式：
- 实时检测摄像头/视频，保存可视化结果和可选 YOLO txt 标签到 `results/`
- 交通灯颜色识别（红/黄/绿），支持手动 ROI 与 YOLO 自动裁剪 ROI
- Windows 下摄像头友好名称显示与刷新（可选依赖）
- 可选离线语音关键词识别（Vosk），用于简单语音控制


## 环境要求

- 操作系统：Windows 10/11 推荐（CLI 在 Linux/macOS 也可运行；MPS 需 macOS 支持）
- Python：3.11+
- GPU：可选；若使用 CUDA，请与 `torch==2.5.1+cu121` 和驱动版本匹配


## 安装

项目使用 `pyproject.toml` 与 [uv](https://docs.astral.sh/uv/) 管理依赖，推荐如下安装：

```powershell
# 安装 uv（若未安装）
python -m pip install -U uv

# 同步依赖（已配置 pytorch-cu121 源）
uv sync

# 验证
uv run python -V
```

说明：
- 若无需 CUDA，可改装 CPU 版 torch/torchvision（移除自定义索引，安装 CPU 轮子）。
- 只用 CLI 时，可跳过 GUI 依赖，但使用 `uv sync` 会按 `pyproject.toml` 全量安装。


## Windows 摄像头友好名称（pywin32/WMI 或 pygrabber）

为在 Windows 下显示更友好的摄像头名称（而不是仅有的 `Camera n` 索引），项目提供两条可选路径：

- WMI（pywin32）：由 `yva_io/camera_name.py` 调用 WMI，返回较“丰富”的设备信息（名称、状态、PNP 类等）。
- DirectShow（pygrabber）：由 `yva_io/camera_utils.py` 直接枚举 DirectShow 输入设备，速度快、依赖少。

依赖状态与默认行为：
- 这两个依赖已在 `pyproject.toml` 声明（`pywin32>=311`、`pygrabber>=0.2`）。在 Windows 上执行 `uv sync` 会自动安装；非 Windows 环境会自动跳过。
- 若两者均不可用或调用失败，界面/CLI 会回退显示 `Camera n`。

安装与验证（PowerShell）：

```powershell
# 安装/同步依赖（Windows 下会包含 pywin32 和 pygrabber）
uv sync

# 可选：单独添加（通常无需，因为已在 pyproject.toml 中）
uv add pywin32 pygrabber

# 自检：打印 DirectShow 设备名（pygrabber）
uv run python -c "from yva_io import get_directshow_device_names as g; print(g())"

# 自检：打印 WMI 设备（pywin32）
uv run python -c "from yva_io import enumerate_cameras as e; print([d.to_dict() for d in e(verbose=True)])"
```

使用建议与注意事项：
- 优先级：WMI 与 DirectShow 信息来源不同，项目内部会按场景择优使用；二者皆无时回退 `Camera n`。
- 顺序与索引：DirectShow 的枚举顺序与 OpenCV 的摄像头索引通常一致，但不保证 100% 对齐；发生不一致时，以能成功打开的索引为准。
- 虚拟摄像头：可能出现重复/虚拟设备（如会议软件虚拟摄像头）；可在系统设备管理器中禁用无关设备以简化列表。
- 故障排查：
  - 运行上面的“自检命令”，确认能否列出设备；
  - 以管理员身份重试（少数机器的 WMI 权限问题会导致失败）；
  - 设置环境变量 `CAM_VERBOSE=1` 查看 `yva_io/camera_name.py` 的详细日志。



## 快速开始

项目提供一个统一入口 `main.py`，以及可直接运行的模块入口。

1) 启动 GUI（PySide6）

```powershell
# 方式 A：统一入口
uv run python .\main.py

# 方式 B：直接运行模块
uv run python -m app.gui
```

2) 命令行实时检测（YOLO）

```powershell
# 方式 A：统一入口
uv run python .\main.py detect --model models\yolo\yolo11n.pt --source 0 --conf 0.5 --save-txt

# 方式 B：直接运行模块
uv run python -m detection.cli --model models\yolo\yolo11n.pt --source 0 --conf 0.5 --save-txt
```

常用参数：
- `--model` 模型权重（默认 `models/yolo/yolo11n.pt`）
- `--device` 设备：`auto`/`cuda`/`cuda:N`/`cpu`/`mps`
- `--source` 视频源：摄像头索引（如 0）或视频文件路径
- `--save-dir` 输出目录（默认 `results`）
- `--save-txt` 保存 YOLO txt 标签
- `--conf` 置信度阈值（0~1）
- `--img-size` 推理尺寸：`640` 或 `640,640`；留空表示以原始帧尺寸为目标
- `--window-name`/`--timestamp-fmt`/`--exit-key`/`--no-fps` 等

窗口聚焦时按 `q`（或 `--exit-key` 指定）退出。

3) 交通灯颜色识别（图片/目录/摄像头）

```powershell
# 方式 A：统一入口
uv run python .\main.py traffic --image .\traffic_img\demo.jpg

# 方式 B：直接运行模块
uv run python -m vision.traffic_cli --image .\traffic_img\demo.jpg
uv run python -m vision.traffic_cli --dir .\traffic_img
uv run python -m vision.traffic_cli --cam 0

# 自动 ROI（YOLO）示例（可叠加 --save-crops 保存裁剪）
uv run python -m vision.traffic_cli --image .\traffic_img\demo.jpg --auto --model models\yolo\yolo11n.pt --conf 0.5 --device auto
```

可选参数：
- `--roi X Y W H` 指定 ROI；不传则交互框选
- `--auto` 启用 YOLO 自动检测并裁剪（默认类别 id=9：traffic light）
- `--class-id` 目标类别（默认 9），`--first` 仅取 1 个最优框
- 其余如 `--img-size`/`--device`/`--conf` 同上
- `--save-crops DIR` 将自动裁剪的 ROI 保存到目录

输出文件：
- `results/frame_{id}_{ts}.jpg`：每帧可视化
- `results/frame_{id}_{ts}.txt`：YOLO txt（可选）
- `results/traffic_lights/tl_{frameId}_{boxIndex}_{ts}_{color}.jpg`：交通灯裁剪


## 环境变量覆盖（前缀 YV_）

除命令行外，也可用环境变量覆盖默认值（命令行优先）：

- `YV_MODEL_PATH` → `--model`
- `YV_DEVICE` → `--device`
- `YV_SOURCE` → `--source`
- `YV_SAVE_DIR` → `--save-dir`
- `YV_SAVE_TXT` → `--save-txt`
- `YV_SELECT_CAMERA` → `--select-camera`
- `YV_MAX_CAM_INDEX` → `--max-cam`
- `YV_CONF` → `--conf`
- `YV_IMG_SIZE` → `--img-size`
- `YV_WINDOW_NAME` → `--window-name`
- `YV_TIMESTAMP_FMT` → `--timestamp-fmt`
- `YV_EXIT_KEY` → `--exit-key`
- `YV_SHOW_FPS` → `--no-fps`（布尔，命令行为“关闭”）
- `YV_QUIET_CV` → `--quiet-cv`
- `YV_CAM_FAIL_LIMIT` → `--cam-fail-limit`

摄像头枚举阶段日志抑制：`YV_SUPPRESS_ENUM_ERRORS=1`（默认开启）。

示例（PowerShell）：

```powershell
$env:YV_MODEL_PATH = ".\models\yolo\yolo11n.pt"
$env:YV_SOURCE = "0"
$env:YV_CONF = "0.45"
uv run python -m detection.cli --save-txt
```


## 目录结构（节选）

```
app/                # GUI 与后台线程
  gui.py            # 主窗口（参数配置、摄像头选择、语音控制集成）
  worker.py         # 检测后台线程与 Qt 信号

detection/          # YOLO 检测核心与 CLI 封装
  core.py           # YOLOConfig/YOLODetector，摄像头枚举、保存、TTS 播报
  api.py            # 门面导出（供 GUI/CLI 复用）
  cli.py            # 命令行入口（python -m detection.cli）

vision/             # 交通灯颜色与相关工具
  color_detection.py# HSV 颜色判定
  traffic_logic.py  # 多框融合判断过街/等待等状态
  traffic_cli.py    # 交通灯模式 CLI
  traffic_mode.py   # 运行器（来源/ROI/YOLO 自动裁剪）
  roi_utils.py      # ROI 与图像遍历工具
  visual_styles.py  # 可视化中文标签与颜色

voice/              # 可选：TTS 与语音关键词
  tts.py, tts_queue.py, announce.py
  keyword_listener.py, voice_control.py

yva_io/             # 设备与摄像头名称工具
  camera_utils.py, camera_name.py, device_utils.py

models/             # 放置模型（例如 models/yolo/yolo11n.pt, models/vosk/...）
results/            # 运行输出
docs/STRUCTURE.md   # 目录说明
main.py             # 统一入口（gui/detect/traffic 路由）
pyproject.toml      # 依赖与工具配置（uv、ruff 等）
```


## 技术实现与算法逻辑

### 整体架构与数据流

- 统一入口 `main.py` 将命令路由至：
  - GUI：`app.gui.MainWindow`（PySide6）
  - YOLO 检测 CLI：`detection.cli` → `detection.core`
  - 交通灯模式 CLI：`vision.traffic_cli` → `vision.traffic_mode`
- 检测核心：`detection/core.py`
  - 构造 `YOLOConfig`（支持命令行 + 环境变量 YV_ 前缀，命令行优先）
  - `YOLODetector` 加载 Ultralytics YOLO 模型，循环读取视频帧并推理
  - 绘制结果、叠加 FPS、保存每帧与可选 YOLO txt；对交通灯 ROI 另存裁剪
  - 结合 `vision` 模块给交通灯打中文标签，并通过 `voice.Announcer` 做 TTS 播报（去重/限流/黄闪识别）
- 交通灯模式：`vision/traffic_mode.py` + `vision/traffic_cli.py`
  - 支持图片/目录/摄像头输入；支持手动 ROI（交互/参数）与 YOLO 自动裁剪 ROI 两种方式
  - 自动模式使用 `detection/yolo_utils.py` 的轻量封装（挑选 class_id=9 的框）
- GUI：`app/gui.py` + `app/worker.py`
  - 主线程纯 UI；检测在 `DetectionWorker` 线程中运行，通过 Qt 信号回传完成/错误
  - 支持摄像头友好名（`yva_io`）、可选离线语音关键字（Vosk）、TTS 队列去抖


### YOLO 检测流水线（detection/core.py）

1) 设备选择：
  - `auto` 优先 `cuda`，再 `mps`，否则 `cpu`（`torch.cuda.is_available()` / `torch.backends.mps.is_available()`）
2) 输入尺寸：
  - 若 `--img-size` 未设，则以“原始帧尺寸”作为目标推理尺寸 `imgsz=[h,w]`，减少拉伸与比例失真
  - 否则按传入的 `640` 或 `640,640` 执行
3) 交通灯候选框：
  - 从 YOLO 结果中筛选 `class_id=9 (traffic light)` 的框，取整像素并裁剪边界
  - 对每个框 ROI 执行颜色判定，绘制中文标签（见下节）并保存裁剪图 `results/traffic_lights/`
4) YOLO txt 导出：
  - 将每帧的检测框转换为归一化 `x_center y_center width height` 格式保存为 `frame_*.txt`
5) FPS 显示：
  - 指数滑动平均平滑 FPS：新 FPS 用 0.1 权重更新，抑制抖动
6) OpenCV 摄像头：
  - Windows 优先 `cv2.CAP_DSHOW` 打开整型索引摄像头；读取失败计数超过阈值提前退出
  - 支持抑制 OpenCV 低层枚举错误日志（仅在“摄像头枚举阶段”临时降低日志级别）


### 交通灯颜色识别（vision/color_detection.py）

目标：对单个 ROI（交通灯框）判断 red/yellow/green/unknown。

核心步骤：
1) 预处理：
  - 限制最长边至 320 像素，并做轻度高斯模糊，转换 BGR→HSV
2) 三色掩码：
  - red：H 通道采用“低段 + 高段”两个范围合并（应对环绕），叠加 S/V 阈值过滤
  - yellow：H≈[15,35]，叠加 S≥阈值、V≥阈值
  - green：H≈[40,85]，叠加 S≥阈值、V≥阈值
  - 对每个掩码做一次开-闭操作去噪
3) 打分与阈值：
  - 对每色计算掩码面积占比 ratio 与掩码区域 V 通道平均亮度 v
  - 记分公式：$score = ratio \times (v/255)^{1.2}$
  - 选择分数最高的颜色作为候选；若面积占比低于阈值或得分为 0，则判定为 unknown

说明：
- 该方法对“亮度显著”的红/黄/绿圆点有较好鲁棒性；复杂场景可根据相机与环境微调 S/V 阈值或次幂因子。


### 交通灯状态融合逻辑（vision/traffic_logic.py）

输入：一帧图像 + 一组 YOLO 候选框（整型像素坐标 + 置信度）。

策略：
1) 方向分类：按“高>宽”为纵向，将候选拆为“竖直灯”和“水平灯”两组
2) 竖直灯：
  - 若多框，先选“距图像中心最近，置信度次排序”的单框，取其颜色
3) 水平灯：
  - 若多框，分别取色，若存在有效色且全部相同则输出该色，否则输出“颜色不同/不工作”类提示
4) 兜底：
  - 无候选 → “无红绿灯”；竖直存在但无法定色 → “红绿灯不工作” 等

可选播报（detection/core.py）：
- `voice.Announcer` 结合窗口与采样计数识别“黄闪”场景；同一内容播报最小间隔限制，避免刷屏与重读。
  - 可调参数：`ann_min_interval`、`ann_flash_window`、`ann_flash_min_events`、`ann_flash_yellow_ratio`、`ann_flash_cooldown`


### 轻量 YOLO 封装（detection/yolo_utils.py）

用于交通灯自动 ROI 模式：
- 仅抽取给定 `class_id`（默认 9）框，并过滤过小框（边长 < 8 像素）
- 先按置信度降序，再按“距图像中心的平方距离 + 置信度次排序”挑选最优竖直框
- 提供 `detect_orientations()` 同时返回竖直/水平两类框，供业务自行融合

设备选择：
- `auto` → CUDA> MPS> CPU，若 `torch` 不可用则回退 CPU。


### GUI 线程与语音集成（app/gui.py, app/worker.py, voice/*）

- 检测运行在后台线程 `DetectionWorker`，通过 Qt 信号回报 `finished/error`，避免阻塞 UI
- 语音关键字：使用 `voice.keyword_listener`（Vosk + sounddevice）离线识别，支持受限语法与包含匹配、冷却抑制
- TTS：`voice.tts_queue.TTSManager` 管理播报队列，提供去重与“抑制包含‘检测到’的播报”能力，避免与状态播报互扰
- 摄像头友好名：`yva_io.camera_utils/camera_name` 结合 DirectShow 或 WMI 获取友好名称；无依赖则回退为 `Camera n`


### 边界与可靠性措施

- 帧读取失败：累计连续失败数，超过阈值提前结束，避免长时间空转
- OpenCV 日志抑制：仅在“枚举摄像头”阶段降低日志级别，避免刷屏，但不影响运行阶段日志
- 文本导出：YOLO txt 使用统一归一化格式，便于再训练或标注复核
- FPS 平滑：指数滑动平均抑制抖动，读数更稳定
- Windows 打开摄像头：整型索引默认使用 `CAP_DSHOW`，兼容性更好


## 常见问题（FAQ）

1) CUDA/torch 报错？
- 使用 `--device cpu`；或安装与你驱动匹配的 CUDA 版 `torch/torchvision`。

2) 打不开摄像头或黑屏？
- 确认索引正确，尝试 0/1/2；关闭占用摄像头的软件；在 Windows 设备管理器检查设备。

3) GUI 启动失败（Qt 异常）？
- 确认安装 `PySide6`；无显示环境改用 CLI。

4) Windows 下无友好名称？
- 安装 `pywin32`（`yva_io/camera_name.py` 使用 WMI），否则显示 `Camera n`。

5) 交通灯颜色不稳定？
- 调整 `--conf` 与 `--img-size`，或使用手动 ROI；注意场景光照与距离。


## 致谢

- Ultralytics YOLO

如有问题或建议，欢迎提交 Issue。
- `EXIT_KEY` → `--exit-key`（默认 `q`）

