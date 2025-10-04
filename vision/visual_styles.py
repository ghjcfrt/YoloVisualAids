"""视觉显示风格与映射

提供交通灯中文标签与 BGR 颜色映射常量，供绘制与文案展示使用。
"""
TL_STATE_CN = {
    'red': '红灯',
    'yellow': '黄灯',
    'green': '绿灯',
    'unknown': '未知',
}

TL_COLOR_MAP = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'unknown': (255, 255, 255),
}

__all__ = ["TL_COLOR_MAP", "TL_STATE_CN"]
