# -*- coding: utf-8 -*-
"""NMEA0183 Debug / 静态观测数据 解析器 —— 按模块拆分的包

原单文件脚本 nmea_debug_parser.py 拆分后的模块结构:

nmea_parser/
├── __init__.py         包入口
├── config.py           matplotlib中文字体 / TYPE_COLORS / CSV列定义
├── parser.py           NMEA0183 报文解析（纯逻辑，无GUI依赖）
├── statistics.py       GGA 报文统计分析（纯逻辑，无GUI依赖）
├── rinex.py            RINEX 观测文件 / 广播星历解析（纯逻辑，无GUI依赖）
├── rinex_statistics.py 静态观测数据质量分析（纯逻辑，无GUI依赖）
├── worker.py           解析工作线程 (ParseWorker)
├── rinex_worker.py     静态分析工作线程 (RinexWorker)
├── toolbar.py          matplotlib 中文导航工具栏
└── ui/
    ├── __init__.py
    ├── main_window.py  主窗口 (NMEAParserGUI)
    ├── dialogs.py      详情/导出选择/统计报告/可视化 对话框
    ├── satellite_dialog.py 卫星搜星与信噪比分析
    ├── quality_dialog.py   测量环境质量报告
    └── static_dialog.py    静态观测数据质量分析
"""

__version__ = '1.1.0'
