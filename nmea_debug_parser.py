#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMEA0183 Debug数据解析器 (PyQt5版) —— 启动入口

原单文件脚本已按模块拆分，本文件仅保留启动逻辑。
拆分后的模块位于同目录下的 nmea_parser/ 包中:

nmea_parser/
├── config.py        matplotlib中文字体 / 报文类型颜色 / CSV列定义
├── parser.py        NMEA0183报文解析（纯逻辑，无GUI依赖）
├── statistics.py    GGA统计分析（纯逻辑，无GUI依赖）
├── worker.py        解析工作线程 (ParseWorker)
├── toolbar.py       matplotlib中文导航工具栏
└── ui/
    ├── main_window.py   主窗口 (NMEAParserGUI)
    └── dialogs.py       对话框（详情/导出选择/统计报告/可视化）

功能同原脚本：
1. 解析.dbg文件，区分NMEA0183报文和二进制数据
2. 分类显示各类NMEA报文
3. 支持导出为CSV（经度、纬度、高程等分列显示，可选择报文类型）
4. GGA报文统计分析（均值、中位数、CEP95、均方差等）
5. 可视化显示精度水平（含北东/高程精度箱型图）
6. 多线程处理，避免GUI界面卡死
"""

import sys

from PyQt5.QtWidgets import QApplication, QStyleFactory

from nmea_parser.ui.main_window import NMEAParserGUI


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    window = NMEAParserGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
