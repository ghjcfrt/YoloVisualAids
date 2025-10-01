"""
可视化样式常量：用于为检测结果选择展示颜色和中文标签。

将展示层（颜色/文案）与算法层（颜色识别）解耦，便于复用与统一管理。
"""

# 交通灯状态的中文文案
TL_STATE_CN = {
    'red': '红灯',
    'yellow': '黄灯',
    'green': '绿灯',
    'unknown': '未知',
}

# OpenCV BGR 颜色，用于在帧上绘制文本/标注
TL_COLOR_MAP = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'unknown': (255, 255, 255),
}

__all__ = ["TL_COLOR_MAP", "TL_STATE_CN"]
