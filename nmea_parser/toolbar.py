# -*- coding: utf-8 -*-
"""matplotlib 中文导航工具栏"""

import matplotlib

# 确保 Qt5Agg 后端已设置（config.py 也会设置，这里兜底，防止导入顺序问题）
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)


class ChineseNavToolbar(NavigationToolbar):
    """中文版 matplotlib 导航工具栏"""
    toolitems = [
        ('主页', '重置原始视图', 'home', 'home'),
        ('后退', '返回上一个视图', 'back', 'back'),
        ('前进', '前进到下一个视图', 'forward', 'forward'),
        (None, None, None, None),
        ('平移', '左键平移，右键缩放', 'move', 'pan'),
        ('缩放', '框选缩放', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),
        ('子图', '配置子图布局', 'subplots', 'configure_subplots'),
        (None, None, None, None),
        ('保存', '保存当前图像', 'filesave', 'save_figure'),
    ]
