# 目录结构优化说明

目标：
- 提供稳定的包入口（yolovisualaids），后续重构时不影响调用方式。
- 保持现有脚本可直接运行，兼容当前工作流。

当前结构（新增部分）：

```
yolovisualaids/
  __init__.py           # 顶层包，暴露 __version__
  app/
    __init__.py         # 导出 gui.main
    gui.py              # GUI 启动（包内实现）
  detection/
    __init__.py         # 导出 cli.main
    core.py             # 检测核心实现
    api.py              # 对外 API（封装 core）
    cli.py              # CLI 封装（调用 core.main）
docs/
  STRUCTURE.md          # 本说明
```

使用方式：

- 启动 GUI：
  - `python -m yolovisualaids.app`
  - 或安装后使用脚本 `yva-app`

- 命令行检测：
  - `python -m yolovisualaids.detection.cli --source 0 --model models/yolo/yolo11n.pt`
  - 或安装后使用脚本 `yva-detect --source 0`

迁移状态：
- 根目录 `YOLO_detection.py` 与 `app.py` 已迁移至包内并删除。
- GUI 与检测 CLI 入口均为包内模块与脚本。

注意：
- 目前封装采用“转发”策略，未移动原文件，避免对现有导入与运行造成影响。
- 若需要彻底清理根目录脚本，请先完成模块化迁移并更新所有引用。
