# YoloVisualAids —— 基于 YOLOv11 的实时检测与交通灯颜色识别

YoloVisualAids 是一个基于 Ultralytics YOLOv11 的实时目标检测小工具，提供图形界面与命令行两种使用方式，支持自动/手动 ROI 的交通灯颜色识别（红/黄/绿），并将检测结果帧、裁剪与可选的 YOLO txt 标签保存到本地。

主要场景：摄像头实时检测、视频/图片离线检测、交通灯颜色判别、Windows 下摄像头友好名称枚举，以及可选的 Orbbec 深度相机示例。


## 功能特性

- 实时检测与保存
	- 自动选择设备（auto/cuda/cpu/mps）
	- 每帧落盘到 `results/`，可选保存 YOLO txt
	- 状态栏显示与 FPS 叠加（可开关）
- GUI 参数配置（`PySide6`）
	- 一般/高级两种模式（简单只改模型/摄像头，高级可配置全部参数）
	- 后台线程推理、可随时停止、恢复默认
	- 摄像头友好名称下拉与刷新（Windows 使用 WMI，需 `pywin32`）
- 交通灯颜色识别
	- HSV 阈值分割 + 面积/亮度综合评分，鲁棒识别红/黄/绿
	- 支持手动 ROI（交互框选）或 YOLO 自动裁剪 ROI
	- 支持图片/目录/摄像头三种来源
	- 自动保存交通灯 ROI 裁剪到 `results/traffic_lights/`
- 环境变量与命令行双配置
	- 环境变量前缀 `YV_`，易于在不同环境快速切换
	- 命令行参数覆盖默认值与环境值
- 摄像头枚举与降噪
	- 枚举 `0..N-1` 摄像头，按失败次数提前结束
	- 可抑制 OpenCV 低层枚举日志噪声
- 可选：Orbbec 深度相机单帧抓取示例（`test1.py` / `test2.py`）


## 目录结构

（节选）

- `app.py`：GUI 启动与界面交互
- `YOLO_detection.py`：核心检测逻辑与 CLI 参数/环境变量配置
- `color_detction.py`：交通灯颜色识别（HSV）
- `traffic_mode.py`：交通灯测试模式（图片/目录/摄像头 + 手动/自动 ROI）
- `yolo_utils.py`：推理设备选择、简易封装器与工具函数
- `camera_name.py`：Windows 下使用 WMI 枚举摄像头友好名称（需要 `pywin32`）
- `roi_utils.py`：通用 ROI/图像工具
- `test.py`：统一入口（GUI / YOLO CLI / 交通灯模式）
- `test1.py`、`test2.py`：Orbbec 深度相机示例
- `results/`：检测结果输出目录（运行时生成）


## 环境要求

- 操作系统：Windows 10/11（推荐）。Linux/macOS 也可运行 CLI 检测；MPS 需 macOS 配置支持
- Python：3.11+
- 显卡：可选，若使用 CUDA 请安装与驱动匹配的 PyTorch（本项目默认指向 CUDA 12.1 轮子）
- 可选组件：
	- Windows 摄像头友好名称：`pywin32`
	- Orbbec 深度相机示例：`pyorbbecsdk`（另见 `README_CN_pyorbbecsdk.md`）


## 安装与依赖

项目使用 `pyproject.toml` 管理依赖，推荐使用 [uv](https://docs.astral.sh/uv/) 快速安装。

PowerShell（Windows）示例：

```powershell
# 1) 安装 uv（如未安装）
python -m pip install -U uv

# 2) 同步项目依赖
uv sync

# 3) 在虚拟环境中运行（uv 会自动使用隔离环境）
uv run python -V
```

说明：
- `pyproject.toml` 已配置从 `pytorch-cu121` 索引安装 `torch==2.5.1+cu121` 与 `torchvision==0.20.1+cu121`。
	- 若无 NVIDIA GPU 或不需要 CUDA，可自行改为 CPU 轮子（删除自定义索引并安装 CPU 版 torch/torchvision），或在已有环境中单独安装 CPU 版。
- 若仅用命令行且不需要 GUI，可不安装 `PySide6`。

不使用 uv 的替代方案（需已安装合适的 torch/torchvision）：

```powershell
python -m pip install -U ultralytics PySide6 pywin32
```


## 快速开始

### 1) 图形界面（GUI）

```powershell
uv run python test.py --gui
# 或直接运行 GUI 主文件
uv run python app.py
```

GUI 说明：
- 模式：一般/高级。一般模式只需选择模型与摄像头；高级模式可配置设备、输入源、保存目录、置信度、输入尺寸、窗口标题、时间戳格式、退出键、FPS 显示等。
- 摄像头：下拉选择并可刷新，Windows 下显示友好名称（需 `pywin32`）。
- 控制：启动检测、停止、恢复默认。
- 输出：结果帧保存到 `results/`，交通灯裁剪保存到 `results/traffic_lights/`，可选保存 txt 标签。


### 2) 命令行实时检测（YOLO）

两种入口均可：

```powershell
# 通过统一入口（推荐）
uv run python test.py --model yolo11n.pt --source 0 --conf 0.5 --save-txt

# 或直接调用检测脚本
uv run python YOLO_detection.py --model yolo11n.pt --source 0 --conf 0.5 --save-txt
```

常用参数（更多见下文“参数与环境变量”）：
- `--model` 模型权重路径（默认 `yolo11n.pt`）
- `--device` 设备：`auto`/`cuda`/`cuda:N`/`cpu`/`mps`
- `--source` 视频源：摄像头索引（如 0）或视频文件路径
- `--save-dir` 输出目录（默认 `results`）
- `--save-txt` 保存 YOLO txt 标签
- `--conf` 置信度阈值（0~1）
- `--img-size` 推理尺寸：如 `640` 或 `640,640`；留空表示以原始帧尺寸为目标
- `--window-name`/`--timestamp-fmt`/`--exit-key`/`--no-fps` 等

键位：窗口聚焦时按 `q`（或自定义 `--exit-key`）退出。


### 3) 交通灯颜色识别（图片/目录/摄像头）

手动 ROI（交互框选）或 YOLO 自动裁剪 ROI 二选一：

```powershell
# 单张图片（手动框选 ROI）
uv run python test.py traffic --image .\traffic_img\222.JPEG

# 目录批处理（手动 ROI；n 下一张，q 退出，r 重新选 ROI）
uv run python test.py traffic --dir .\traffic_img

# 摄像头（手动 ROI）
uv run python test.py traffic --cam 0

# 使用 YOLO 自动裁剪 ROI 进行颜色识别（可叠加 --save-crops 保存裁剪）
uv run python test.py traffic --image .\traffic_img\222.JPEG --auto --model yolo11n.pt --conf 0.5 --device auto
```

可选参数：
- `--roi X Y W H` 指定矩形 ROI；不传则交互选择
- `--auto` 启用 YOLO 自动检测并裁剪（默认仅处理 COCO 类别 9：traffic light）
- `--class-id` 目标类别（默认 9），`--first` 仅取置信度最高一个
- `--img-size`/`--device`/`--conf` 同 YOLO 检测
- `--save-crops DIR` 将自动裁剪的 ROI 保存到指定目录

输出：
- 识别结果绘制在窗口与控制台；
- 自动模式下的裁剪保存于 `results/traffic_lights/` 或 `--save-crops` 指定目录。


## 参数与环境变量

所有参数既可由命令行传入，也可使用同名环境变量覆盖默认值（前缀 `YV_`）：

- `MODEL_PATH` → `--model`（默认 `yolo11n.pt`）
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

摄像头相关额外环境变量：
- `YV_SUPPRESS_ENUM_ERRORS=1` 在枚举阶段临时降低 OpenCV 日志级别（默认开启）

示例（PowerShell）：

```powershell
$env:YV_MODEL_PATH = "D:\\models\\yolo11n.pt"
$env:YV_SOURCE = "0"
$env:YV_CONF = "0.45"
uv run python YOLO_detection.py --save-txt
```


## 摄像头选择与枚举

- GUI：下拉框内显示友好名称（Windows），可点击“刷新”；选中后会同步到“视频源”。
- CLI：
	- 使用整型索引：`--source 0`
	- 通过 `--select-camera` 启动时交互选择
	- 枚举逻辑：当已发现至少 1 个可用摄像头且连续失败次数达到 `CAM_FAIL_LIMIT` 即提前结束，提高启动速度。


## 输出与文件组织

- `results/frame_{id}_{timestamp}.jpg`：每帧的可视化结果
- `results/frame_{id}_{timestamp}.txt`：可选 YOLO txt（归一化中心点与宽高）
- `results/traffic_lights/tl_{frameId}_{boxIndex}_{timestamp}_{color}.jpg`：交通灯 ROI 裁剪


## Orbbec 深度相机（可选）

- 示例脚本：`test1.py` 与 `test2.py`（非必须）
- 文档：详见 `README_CN_pyorbbecsdk.md`
- Windows 常见驱动提示（摘录）：某些设备需将深度接口改为 WinUSB/libusbK 才能被 SDK 正确枚举。可使用 Zadig 或设备管理器切换，完成后重新插拔设备。


## 常见问题（FAQ）

1) 启动报 CUDA/torch 相关错误？
- 使用 `--device cpu` 强制 CPU；或根据本地环境安装匹配的 `torch/torchvision`（CUDA 版本需与驱动匹配）。

2) 打不开摄像头或黑屏？
- 确认 `--source` 索引正确，尝试换 `0/1/2`；关闭被占用摄像头的软件；在 Windows 设备管理器检查摄像头是否可用。

3) GUI 无法启动或报 Qt/Tk 相关异常？
- 确认已安装 `PySide6`；无显示环境时请用命令行模式运行。

4) Windows 下摄像头没有显示友好名称？
- 需要 `pywin32`（`camera_name.py` 使用 WMI），否则将显示 `Camera n`。

5) 交通灯未检测到或颜色不稳定？
- 调整 `--conf` 与 `--img-size`，或使用手动 ROI；环境光很暗/很亮时可适当调整位置与尺寸。


## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)


---
如有问题或建议，欢迎提交 Issue。希望这个小工具能帮你快速搭建 YOLO 检测与交通灯识别的实验环境。

