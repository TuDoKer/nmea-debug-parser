# -*- coding: utf-8 -*-
"""主窗口 (NMEAParserGUI)

负责: 文件打开 / 多线程解析 / 表格填充 / 筛选 / CSV导出。
统计报告与可视化等对话框已抽离到 dialogs.py。
"""

import csv
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QCheckBox, QScrollArea, QProgressBar, QStatusBar,
    QDialog, QPushButton, QLabel, QFileDialog, QMessageBox, QTabWidget,
    QTextEdit, QSplitter, QHeaderView, QGroupBox, QLineEdit, QComboBox,
    QAbstractItemView, QApplication, QListWidget, QListWidgetItem,
    QInputDialog,
)
from PyQt5.QtCore import Qt, QThread, QPoint
from PyQt5.QtGui import QFont, QBrush, QColor

# 先加载配置（配置 matplotlib 后端与中文字体，必须在任何 backend 导入前）
from ..config import (
    TYPE_COLORS, CSV_EXPORT_COLUMNS, last_dir, save_last_dir,
)
from ..statistics import GGAStatistics
from ..worker import ParseWorker
from ..rinex_worker import RinexWorker, EphemerisDownloadWorker
from .. import rinex_ephemeris as RXE
from .dialogs import (
    MessageDetailDialog, ExportTypeSelectionDialog,
    StatisticsDialog, VisualizationDialog,
)
from .satellite_dialog import SatelliteDialog
from .quality_dialog import QualityReportDialog
from .static_dialog import StaticAnalysisDialog

# 静态观测文件（RINEX）扩展名，用于自动识别打开的文件类型
RINEX_EXTS = ('.26o', '.25o', '.24o', '.rnx', '.obs', '.o')
# 广播星历（RINEX NAV）扩展名
RINEX_NAV_EXTS = ('.26p', '.25p', '.26n', '.25n', '.26b', '.26g', '.nav', '.rnx')

# 数据模式：决定「打开文件」接受的文件类型与显示的按钮组
MODE_DEBUG = 'debug'
MODE_STATIC = 'static'

# 异常 GGA 行高亮背景色（水平>5cm 或 高程>7.5cm）
OUTLIER_BG = QColor(255, 199, 199)
# 水平误差最大的 GGA 行高亮背景色（醒目橙色）
MAX_ERR_BG = QColor(255, 152, 0)


class NMEAParserGUI(QMainWindow):
    """NMEA Debug 数据解析器 GUI (PyQt5)"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMEA0183 Debug 数据解析器")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 700)

        # 数据存储
        self.parsed_data = None
        self.current_file = ''
        self.type_checkboxes = {}        # {msg_type: QCheckBox}
        self.type_counts = defaultdict(int)
        self.all_nmea_rows = []          # 缓存所有NMEA行数据用于筛选
        self.max_error_info = None       # 最大误差信息（水平/高程最大行）

        # 线程相关
        self.parse_thread = None
        self.parse_worker = None

        # 静态观测数据（RINEX）分析状态
        self.static_result = None        # StaticQuality.compute() 结果
        self.static_obs_file = ''        # 当前静态观测文件
        self.static_nav_file = ''        # 当前广播星历文件（可为空）
        self.rinex_thread = None
        self.rinex_worker = None
        self._pending_nav_dates = []     # 待下载星历的 DOY 列表
        self._nav_prompted = False       # 避免同一文件反复弹下载提示
        self._dialog_wants_eph = False   # 对话框内点了「获取星历」
        self.dl_thread = None
        self.dl_worker = None

        # 静态数据预览页控件（在 _build_preview_tab 中创建）
        self.preview_tables = {}

        self._build_ui()
        self._apply_style()

    def _apply_style(self):
        """应用全局样式"""
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QGroupBox {
                font-weight: bold; border: 1px solid #c0c0c0;
                border-radius: 4px; margin-top: 8px; padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px;
            }
            QTableWidget {
                gridline-color: #d0d0d0; alternate-background-color: #f9f9f9;
                selection-background-color: #4a90d9;
            }
            QHeaderView::section {
                background-color: #e0e0e0; padding: 4px;
                border: 1px solid #c0c0c0; font-weight: bold;
            }
            QPushButton {
                padding: 5px 15px; border-radius: 3px;
                background-color: #4a90d9; color: white;
                border: none; min-width: 80px;
            }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:pressed { background-color: #2a5f9e; }
            QPushButton:disabled { background-color: #b0b0b0; }
            QProgressBar {
                border: 1px solid #c0c0c0; border-radius: 3px;
                text-align: center; height: 20px;
            }
            QProgressBar::chunk { background-color: #4a90d9; border-radius: 2px; }
            QCheckBox { spacing: 6px; }
            QLineEdit { padding: 3px 6px; border: 1px solid #c0c0c0; border-radius: 3px; }
        """)

    def _build_ui(self):
        """构建用户界面"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # ===== 顶部工具栏 =====
        toolbar = QHBoxLayout()
        toolbar.setSpacing(5)

        # --- 数据模式选择：决定「打开文件」接受什么文件、显示哪些按钮 ---
        # 两种模式互斥，切换时只改变控件的可见性，不触碰任何数据与分析师流程,
        # 因此对既有功能零影响。
        toolbar.addWidget(QLabel("数据类型:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("Debug 文件 (.dbg)", MODE_DEBUG)
        self.cmb_mode.addItem("观测数据 (.26O / .rnx)", MODE_STATIC)
        self.cmb_mode.setToolTip(
            "选择「Debug 文件」：只显示 NMEA 相关功能；\n"
            "选择「观测数据」：只显示静态分析相关功能。")
        self.cmb_mode.setMinimumWidth(190)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self.cmb_mode)

        self.btn_open = QPushButton("📂 打开文件")
        self.btn_open.clicked.connect(self.open_file)
        self.btn_export = QPushButton("💾 导出CSV")
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_export.setEnabled(False)
        self.btn_stats = QPushButton("📊 GGA统计")
        self.btn_stats.clicked.connect(self.show_statistics)
        self.btn_stats.setEnabled(False)
        self.btn_viz = QPushButton("📈 可视化")
        self.btn_viz.clicked.connect(self.show_visualization)
        self.btn_viz.setEnabled(False)
        self.btn_sat = QPushButton("🛰 卫星分析")
        self.btn_sat.clicked.connect(self.show_satellite_analysis)
        self.btn_sat.setEnabled(False)
        self.btn_quality = QPushButton("📋 质量报告")
        self.btn_quality.clicked.connect(self.show_quality_report)
        self.btn_quality.setEnabled(False)

        # 精度评估：设备解算序列 vs 参考真值（已知点 / .pos / 另一台设备 / 自建）
        self.btn_accuracy = QPushButton("🎯 精度评估")
        self.btn_accuracy.clicked.connect(self.show_accuracy_dialog)
        self.btn_accuracy.setEnabled(False)
        self.btn_accuracy.setToolTip(
            "把设备 NMEA 的解算结果与参考真值比对，给出外符合精度\n"
            "（CEP50/95、RMS、DRMS/2DRMS 等）\n"
            "参考可以是：已知控制点 / RTKPOST 的 .pos / 另一台设备 / 自身均值")

        # --- 静态观测数据（RINEX）分析入口 ---
        # 与 NMEA 功能完全独立：打开静态文件不会影响 parsed_data 及既有按钮状态
        self.btn_static = QPushButton("🌐 静态数据分析")
        self.btn_static.clicked.connect(self.show_static_analysis)
        self.btn_static.setEnabled(False)
        self.btn_static_nav = QPushButton("🗺 加载星历")
        self.btn_static_nav.clicked.connect(self.load_static_ephemeris)
        self.btn_static_nav.setEnabled(False)

        # 手动在线下载广播星历：原来只有"分析后发现缺星历才被动询问"，
        # 想提前备好星历（或重下）没有入口
        self.btn_static_eph_dl = QPushButton("⬇ 下载星历")
        self.btn_static_eph_dl.clicked.connect(self.download_ephemeris)
        self.btn_static_eph_dl.setEnabled(False)
        self.btn_static_eph_dl.setToolTip(
            "从 IGS 服务器下载对应日期的广播星历（约 1~2 MB，需联网）。\n"
            "已打开观测文件时按文件头里的观测时段确定日期；\n"
            "尚未打开时手动输入日期（YYYY-MM-DD 或 年 年积日）。")

        # NMEA 侧按钮（切到静态模式时整体隐藏）
        self._debug_buttons = [
            self.btn_export, self.btn_stats, self.btn_viz,
            self.btn_sat, self.btn_quality, self.btn_accuracy,
        ]
        # 静态侧按钮（切到 Debug 模式时整体隐藏）
        self._static_buttons = [self.btn_static, self.btn_static_nav,
                                self.btn_static_eph_dl]

        toolbar.addWidget(self.btn_open)
        toolbar.addWidget(self.btn_export)
        toolbar.addWidget(self.btn_stats)
        toolbar.addWidget(self.btn_viz)
        toolbar.addWidget(self.btn_sat)
        toolbar.addWidget(self.btn_quality)
        toolbar.addWidget(self.btn_accuracy)
        toolbar.addWidget(self.btn_static)
        toolbar.addWidget(self.btn_static_nav)
        toolbar.addWidget(self.btn_static_eph_dl)
        toolbar.addStretch()
        self.lbl_file = QLabel("未选择文件")
        self.lbl_file.setStyleSheet("color: gray; font-style: italic;")
        toolbar.addWidget(self.lbl_file)

        main_layout.addLayout(toolbar)

        # 按初始模式设置可见性
        self._apply_mode_visibility()

        # ===== 进度条 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # ===== 主区域 Splitter =====
        splitter = QSplitter(Qt.Horizontal)

        # --- 左侧面板 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)

        # 报文类型筛选（仅 Debug/NMEA 模式有意义；静态模式换成下面的
        # 「显示内容筛选」—— RINEX 没有 NMEA 报文类型，沿用会误导）
        self.type_group = type_group = QGroupBox("报文类型筛选")
        type_layout = QVBoxLayout(type_group)

        btn_row = QHBoxLayout()
        btn_select_all = QPushButton("全选")
        btn_select_all.clicked.connect(self.select_all_types)
        btn_deselect_all = QPushButton("取消全选")
        btn_deselect_all.clicked.connect(self.deselect_all_types)
        btn_select_all.setStyleSheet("background-color: #5cb85c;")
        btn_deselect_all.setStyleSheet("background-color: #d9534f;")
        btn_row.addWidget(btn_select_all)
        btn_row.addWidget(btn_deselect_all)
        type_layout.addLayout(btn_row)

        # 滚动区域放置复选框
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_widget)
        type_layout.addWidget(self.scroll_area)

        left_layout.addWidget(type_group, stretch=3)

        # 统计信息
        self.stats_group = stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 9))
        self.stats_text.setMaximumHeight(250)
        self.stats_text.setPlainText("统计信息将在此显示...\n")
        stats_layout.addWidget(self.stats_text)
        left_layout.addWidget(stats_group, stretch=2)

        # --- 静态模式专用面板：目录 + 统计信息 ---
        # 与上面 NMEA 的两块**互斥显示**。若沿用 NMEA 面板，静态模式下
        # 用户会看到上一次 NMEA 解析的旧筛选/统计（或占位文字），
        # 看起来就像"导入静态数据后没有筛选和统计信息"。
        #
        # 为什么用「目录」而不是原来的勾选筛选：预览页 13 张子表纵向很长，
        # 用户的需求更多是"跳到我想看的那张"，而不是"把其它藏起来"。
        # 目录点一下就滚到位，还保留整体上下文；勾选式筛选会让用户
        # 忘记自己关掉了哪些表，反而更难找。
        self.static_type_group = QGroupBox("目录")
        stl = QVBoxLayout(self.static_type_group)
        self.preview_toc = QListWidget()
        self.preview_toc.setToolTip("点击条目跳到对应的子表")
        for _key, title, _cols in self.PREVIEW_TABLES:
            QListWidgetItem(title, self.preview_toc)
        self.preview_toc.currentRowChanged.connect(self._scroll_to_section)
        stl.addWidget(self.preview_toc)
        left_layout.addWidget(self.static_type_group, stretch=3)
        self.static_type_group.setVisible(False)

        self.static_stats_group = QGroupBox("统计信息")
        ssl = QVBoxLayout(self.static_stats_group)
        self.static_stats_text = QTextEdit()
        self.static_stats_text.setReadOnly(True)
        self.static_stats_text.setFont(QFont("Consolas", 9))
        self.static_stats_text.setMaximumHeight(250)
        self.static_stats_text.setPlainText("静态统计信息将在此显示...\n")
        ssl.addWidget(self.static_stats_text)
        left_layout.addWidget(self.static_stats_group, stretch=2)
        self.static_stats_group.setVisible(False)

        splitter.addWidget(left_panel)

        # --- 右侧面板 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(2, 2, 2, 2)

        self.notebook = QTabWidget()

        # NMEA报文页
        nmea_tab = QWidget()
        nmea_tab_layout = QVBoxLayout(nmea_tab)
        nmea_tab_layout.setContentsMargins(2, 2, 2, 2)

        # 搜索栏
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("搜索:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词筛选报文...")
        self.search_input.textChanged.connect(self.filter_nmea)
        search_row.addWidget(self.search_input, stretch=1)
        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self.reset_filter)
        search_row.addWidget(btn_reset)
        nmea_tab_layout.addLayout(search_row)

        # NMEA表格
        self.nmea_table = QTableWidget()
        self.nmea_table.setColumnCount(10)
        self.nmea_table.setHorizontalHeaderLabels(
            ['行号', '类型', 'UTC时间', '纬度(°)', '经度(°)', '高度(m)', 'Fix', '卫星数', 'HDOP', '原始报文']
        )
        self.nmea_table.setAlternatingRowColors(True)
        self.nmea_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.nmea_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.nmea_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.nmea_table.horizontalHeader().setSectionResizeMode(9, QHeaderView.Stretch)
        self.nmea_table.setColumnWidth(0, 50)
        self.nmea_table.setColumnWidth(1, 70)
        self.nmea_table.setColumnWidth(2, 85)
        self.nmea_table.setColumnWidth(3, 120)
        self.nmea_table.setColumnWidth(4, 120)
        self.nmea_table.setColumnWidth(5, 80)
        self.nmea_table.setColumnWidth(6, 40)
        self.nmea_table.setColumnWidth(7, 55)
        self.nmea_table.setColumnWidth(8, 55)
        self.nmea_table.verticalHeader().setDefaultSectionSize(24)
        self.nmea_table.doubleClicked.connect(self.on_nmea_double_click)
        nmea_tab_layout.addWidget(self.nmea_table)

        self.notebook.addTab(nmea_tab, "NMEA0183报文")

        # 二进制数据页
        bin_tab = QWidget()
        bin_tab_layout = QVBoxLayout(bin_tab)
        bin_tab_layout.setContentsMargins(2, 2, 2, 2)
        self.bin_table = QTableWidget()
        self.bin_table.setColumnCount(3)
        self.bin_table.setHorizontalHeaderLabels(['行号', '长度(字节)', 'Hex预览'])
        self.bin_table.setAlternatingRowColors(True)
        self.bin_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bin_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.bin_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.bin_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.bin_table.setColumnWidth(0, 60)
        self.bin_table.setColumnWidth(1, 80)
        self.bin_table.verticalHeader().setDefaultSectionSize(24)
        bin_tab_layout.addWidget(self.bin_table)
        self.notebook.addTab(bin_tab, "二进制数据")

        # 未知数据页
        unknown_tab = QWidget()
        unknown_tab_layout = QVBoxLayout(unknown_tab)
        unknown_tab_layout.setContentsMargins(2, 2, 2, 2)
        self.unknown_table = QTableWidget()
        self.unknown_table.setColumnCount(2)
        self.unknown_table.setHorizontalHeaderLabels(['行号', '内容'])
        self.unknown_table.setAlternatingRowColors(True)
        self.unknown_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.unknown_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.unknown_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.unknown_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.unknown_table.setColumnWidth(0, 60)
        self.unknown_table.verticalHeader().setDefaultSectionSize(24)
        unknown_tab_layout.addWidget(self.unknown_table)
        self.notebook.addTab(unknown_tab, "未知/其他")

        # 静态数据预览页（仅在静态模式可见）
        self.notebook.addTab(self._build_preview_tab(), "静态数据预览")

        right_layout.addWidget(self.notebook)
        splitter.addWidget(right_panel)

        splitter.setSizes([320, 1080])
        main_layout.addWidget(splitter, stretch=1)

        # ===== 状态栏 =====
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    # -------------------- 静态数据预览页 --------------------

    # 预览页表格定义: (键, 标题, 列名列表)
    PREVIEW_TABLES = [
        ('info',      '文件概况',   ['项目', '数值']),
        ('overall',   '总体指标',   ['指标', '数值', '评价']),
        ('warn',      '问题清单',   ['等级', '问题', '说明']),
        ('snr',       '信噪比（系统-频点）', ['系统-频点', '样本', '最小', 'P5', '中位', 'P95', '最大', '均值']),
        ('snr_sat',   '信噪比（逐卫星）',   ['卫星', '星座', '样本', '均值', '中位', '<30%', '<35%']),
        ('geo',       '几何（逐卫星）',     ['卫星', '平均仰角', '最低', '最高', '平均方位', '<15%']),
        ('resid',     'C/N0 仰角残差',      ['卫星', '平均仰角', '实测C/N0', '期望C/N0', '残差(dB)', '样本']),
        ('cont',      '连续性与完整性',     ['卫星', '星座', '历元', '完整率%', '中断次数', '最长中断', '判定']),
        ('mp',        '多路径 MP-RMS',      ['卫星', 'RMS(m)', '样本']),
        ('dop',       '几何强度 DOP',       ['指标', '均值', 'P50', 'P95', '最大']),
        ('slip',      '周跳 / 丢锁',        ['卫星', '跳变次数', '符合整周%', '周跳', '丢锁', '首次事件']),
        ('obs',       '原始观测量覆盖',     ['系统-频点', '卫星数', '采样点', '说明']),
        ('ts',        '时间序列覆盖',       ['序列', '卫星数', '说明']),
        ('mp_sys',    '多路径 MP（分系统）',
         ['系统', '卫星', 'RMS(m)', '说明']),
        ('mp_elev',   '多路径与仰角相关性',
         ['系统', '卫星数', '相关系数 r', '判定']),
        ('iono',      '电离层活动 ROT/IROT',
         ['系统', '卫星', 'ROT中位(TECU/min)', 'ROT最大', 'IROT中位', '仰角°']),
    ]

    def _build_preview_tab(self):
        """构建静态数据预览页

        用一组可折叠的子表 + 一个总览文本，把静态分析的详细信息
        直接铺在主界面上，不用打开对话框也能看。
        """
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 4)

        # 顶部提示条 + 打开详细对话框按钮
        head = QHBoxLayout()
        self.lbl_preview = QLabel("尚未分析静态观测数据。请切换到「观测数据」模式后点「📂 打开观测文件」。")
        self.lbl_preview.setStyleSheet("color: gray; font-style: italic;")
        head.addWidget(self.lbl_preview, stretch=1)
        self.btn_preview_detail = QPushButton("🔍 打开详细分析窗口")
        self.btn_preview_detail.setEnabled(False)
        self.btn_preview_detail.clicked.connect(self.show_static_analysis)
        head.addWidget(self.btn_preview_detail)
        v.addLayout(head)

        # 可滚动的子表区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(6)
        # 记下滚动区域与内容容器，供左侧「目录」跳转定位
        self.preview_scroll = scroll
        self.preview_inner = inner

        # 记住每个子表的 GroupBox 与顺序，供「目录」跳转
        self.preview_groups = {}
        self.preview_keys = []
        for key, title, cols in self.PREVIEW_TABLES:
            grp = QGroupBox(title)
            self.preview_groups[key] = grp
            self.preview_keys.append(key)
            gl = QVBoxLayout(grp)
            gl.setContentsMargins(4, 4, 4, 4)
            t = QTableWidget(0, len(cols))
            t.setHorizontalHeaderLabels(cols)
            t.setAlternatingRowColors(True)
            t.setEditTriggers(QAbstractItemView.NoEditTriggers)
            t.setSelectionBehavior(QAbstractItemView.SelectRows)
            hdr = t.horizontalHeader()
            # 最后一列（通常是「说明」「判定」）拉伸填满，其余按内容自适应
            for i in range(len(cols)):
                hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
            if cols[-1] in ('说明', '判定', '评价'):
                hdr.setSectionResizeMode(len(cols) - 1, QHeaderView.Stretch)
            t.verticalHeader().setDefaultSectionSize(22)
            # 限高：避免单个大表把其余表挤出视野，但太小会让用户看不到
            # 表里到底有什么，故给出较宽松的上限 + 内部滚动条。
            t.setMinimumHeight(140)
            if key in ('info', 'overall'):
                t.setMaximumHeight(300)
            elif key == 'warn':
                t.setMaximumHeight(240)
            else:
                t.setMaximumHeight(320)
            gl.addWidget(t)
            self.preview_tables[key] = t
            il.addWidget(grp)

        il.addStretch(1)
        scroll.setWidget(inner)
        v.addWidget(scroll, stretch=1)
        return page

    def populate_static_preview(self, result):
        """把静态分析结果填充到预览页各子表

        result 为 None 或含 error 时只清空并给提示，不抛异常。
        """
        if not getattr(self, 'preview_tables', None):
            return
        for t in self.preview_tables.values():
            t.setRowCount(0)

        if not result or result.get('error'):
            msg = (result or {}).get('error', '没有可用的分析结果。')
            self.lbl_preview.setText(f"⚠ {msg}")
            self.lbl_preview.setStyleSheet("color: #c0392b; font-weight: bold;")
            self.btn_preview_detail.setEnabled(False)
            return

        info = result.get('info') or {}
        nav_name = os.path.basename(self.static_nav_file) if self.static_nav_file else '未加载'
        self.lbl_preview.setText(
            f"✔ {os.path.basename(self.static_obs_file or '')}  ·  "
            f"{info.get('t_first_str', '—')} ~ {info.get('t_last_str', '—')}  ·  "
            f"{info.get('n_epochs', 0)} 历元 / {info.get('n_sats', 0)} 颗卫星  ·  "
            f"星历: {nav_name}")
        self.lbl_preview.setStyleSheet("color: #2a7a2a; font-weight: bold;")
        self.btn_preview_detail.setEnabled(True)

        self._fill_preview_info(result)
        self._fill_preview_overall(result)
        self._fill_preview_warnings(result)
        self._fill_preview_snr(result)
        self._fill_preview_snr_sat(result)
        self._fill_preview_geo(result)
        self._fill_preview_resid(result)
        self._fill_preview_cont(result)
        self._fill_preview_mp(result)
        self._fill_preview_dop(result)
        self._fill_preview_slips(result)
        self._fill_preview_obs(result)
        self._fill_preview_ts(result)
        self._fill_preview_mp_sys(result)
        self._fill_preview_mp_elev(result)
        self._fill_preview_iono(result)
        self._update_static_stats_text(result)

    # -------------------- 静态预览：目录跳转与统计 --------------------

    def _scroll_to_section(self, row):
        """左侧「目录」点击后滚动到对应子表

        用 mapTo 算出子表在内容容器里的绝对 y，再直接设滚动条 ——
        让目标**贴近视口顶部**。QScrollArea.ensureWidgetVisible() 只做
        "滚到能看见为止"，子表比视口高时往往只露出末尾几行，
        用户还得再往下滚，等于没跳。
        """
        keys = getattr(self, 'preview_keys', None)
        groups = getattr(self, 'preview_groups', None)
        sa = getattr(self, 'preview_scroll', None)
        inner = getattr(self, 'preview_inner', None)
        if not keys or not groups or sa is None or inner is None:
            return
        if not (0 <= row < len(keys)):
            return
        grp = groups.get(keys[row])
        if grp is None:
            return
        y = grp.mapTo(inner, QPoint(0, 0)).y()
        # 稍微留一点上边距，让标题不被滚动条/边框压住
        sa.verticalScrollBar().setValue(max(0, y - 8))

    def _update_static_stats_text(self, res):
        """把静态分析的关键数字写进左侧「统计信息」面板

        与 NMEA 侧的统计信息面板对等：都是一眼能看懂的摘要，
        详细数字仍在右侧子表里。
        """
        txt = getattr(self, 'static_stats_text', None)
        if txt is None:
            return
        if not res or res.get('error'):
            txt.setPlainText('（无统计数据）')
            return
        info = res.get('info') or {}
        t0, t1 = info.get('t_first'), info.get('t_last')
        dur_min = 0.0
        if t0 and t1:
            dur_min = ((t1[2] - t0[2]) * 86400 + (t1[3] - t0[3]) * 3600
                       + (t1[4] - t0[4]) * 60 + (t1[5] - t0[5])) / 60.0
        L = []
        L.append(f"文件    : {os.path.basename(self.static_obs_file or '')}")
        L.append(f"MARKER  : {info.get('marker') or '—'}")
        L.append(f"时段    : {info.get('t_first_str', '—')}")
        L.append(f"          {info.get('t_last_str', '—')}"
                 f"  ({dur_min:.1f} 分钟)")
        L.append(f"历元    : {info.get('n_epochs', 0)} 个"
                 f"  间隔 {info.get('modal_interval', '—')} s")
        L.append(f"记录    : {info.get('n_records', 0)} 条")
        L.append(f"卫星    : {info.get('n_sats', 0)} 颗"
                 f"  均值 {info.get('avg_sats_per_epoch', 0):.1f}/历元")
        sysc = info.get('sys_sat_count') or {}
        if sysc:
            L.append("系统    : " + '  '.join(f"{k}={v}" for k, v in sysc.items()))
        snr = (res.get('snr_band') or {}).get('overall') or {}
        if snr:
            L.append(f"C/N0     : 均值 {snr['mean']:.2f} dB-Hz"
                     f"  <30 占比 {snr['lt30']:.2f}%")
        cont = res.get('continuity') or {}
        if cont:
            L.append(f"完整率   : 均值 {cont.get('rate_mean', 0):.2f}%"
                     f"  ≥99% {cont.get('n_ge99', 0)}/{cont.get('n_total', 0)} 颗")
        mp = res.get('multipath') or {}
        if mp.get('available'):
            L.append(f"多路径   : MP-RMS 均值 {mp.get('mean_rms', 0):.3f} m"
                     f"  ({len(mp.get('rows') or [])} 颗)")
            if mp.get('excluded_sats'):
                L.append(f"           剔除 {len(mp['excluded_sats'])} 颗: "
                         + '、'.join(mp['excluded_sats']))
        elif mp.get('single_frequency'):
            L.append("多路径   : —（无 L1/L2 真双频）")
        ms = res.get('multipath_sys') or {}
        if ms.get('available'):
            parts = [f"{name} {blk['mean_rms']:.3f}"
                     for name, blk in ms['systems'].items()
                     if blk.get('available')]
            if parts:
                L.append('           分系统: ' + '  '.join(parts) + ' m')
        io_ = res.get('iono') or {}
        if io_.get('available'):
            parts = [f"{name} {(blk.get('level') or ('—',))[0]}"
                     for name, blk in io_['systems'].items()
                     if blk.get('available')]
            if parts:
                L.append('电离层   : ROT ' + '  '.join(parts)
                         + f"（跳过 {io_.get('n_skip_slip', 0)} 个周跳差分）")
        geo = res.get('geometry') or {}
        if geo.get('available'):
            L.append(f"仰角     : ≥30° {geo.get('ge30', 0):.1f}%"
                     f"  <15° {geo.get('lt15', 0):.1f}%")
        dp = res.get('dop_nsat') or {}
        if dp.get('available'):
            st = dp['stats']
            L.append(f"PDOP     : 均值 {st['pdop']['mean']:.2f}"
                     f"  P95 {st['pdop']['p95']:.2f}"
                     f"  最大 {st['pdop']['max']:.2f}")
        else:
            L.append(f"PDOP     : —（{dp.get('reason') or '需广播星历'}）")
        slip = res.get('cycle_slips') or {}
        n_slip = sum(len(v) if isinstance(v, list) else 0
                     for v in slip.values()) if isinstance(slip, dict) else 0
        if n_slip:
            L.append(f"周跳     : 检出 {n_slip} 处")
        n_warn = len(res.get('warnings') or [])
        L.append(f"问题     : {n_warn} 项" + ('' if n_warn else '（未发现）'))
        L.append(f"星历     : "
                 + (os.path.basename(self.static_nav_file)
                    if self.static_nav_file else '未加载'))
        txt.setPlainText('\n'.join(L))

    def _set_preview_rows(self, key, rows, colors=None):
        """批量写入某个子表；colors 为 {(行, 列): QColor}"""
        t = self.preview_tables.get(key)
        if t is None:
            return
        t.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem('' if val is None else str(val))
                if colors and (r, c) in colors:
                    item.setForeground(colors[(r, c)])
                t.setItem(r, c, item)

    def _fill_preview_info(self, res):
        info = res.get('info') or {}
        # 时长由首末时刻反推（info 不直接提供）
        t0, t1 = info.get('t_first'), info.get('t_last')
        dur_min = 0.0
        if t0 and t1:
            dur_min = ((t1[2] - t0[2]) * 86400 + (t1[3] - t0[3]) * 3600
                       + (t1[4] - t0[4]) * 60 + (t1[5] - t0[5])) / 60.0
        xyz = (res.get('info') or {}).get('xyz') or info.get('approx_xyz')
        obst = info.get('obstypes') or {}
        obst_txt = '; '.join(
            f"{k}: {', '.join(v)}" for k, v in obst.items()) or '—'
        rows = [
            ['文件', os.path.basename(self.static_obs_file or '')],
            ['MARKER', info.get('marker') or '—'],
            ['RINEX 版本', info.get('version')],
            ['时段', f"{info.get('t_first_str', '—')} ~ {info.get('t_last_str', '—')}"],
            ['时长', f"{dur_min:.2f} 分钟"],
            ['历元数', info.get('n_epochs')],
            ['记录总数', info.get('n_records')],
            ['卫星总数', info.get('n_sats')],
            ['平均卫星数', f"{info.get('avg_sats_per_epoch', 0):.2f} 颗/历元"],
            ['单历元卫星数',
             f"min {info.get('min_sats_per_epoch', '—')} / "
             f"max {info.get('max_sats_per_epoch', '—')}"],
            ['采样间隔', f"{info.get('modal_interval', '—')} s"
                        + (f"（标称 {info.get('declared_interval')} s）"
                           if info.get('declared_interval') else '')],
            ['系统构成', '  '.join(f"{k}={v}颗"
                                   for k, v in (info.get('sys_sat_count') or {}).items())
                         or '—'],
            ['观测码', obst_txt],
            ['近似坐标', (f"{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}")
                         if xyz else '—'],
            ['广播星历', os.path.basename(self.static_nav_file)
                         if self.static_nav_file else '未加载'],
        ]
        self._set_preview_rows('info', rows)

    def _fill_preview_overall(self, res):
        snr = (res.get('snr_band') or {}).get('overall') or {}
        cont = res.get('continuity') or {}
        mp = res.get('multipath') or {}
        geo = res.get('geometry') or {}
        rows, colors = [], {}
        i = 0

        def add(name, val, judge, good=True):
            nonlocal i
            rows.append([name, val, judge])
            if judge and not good:
                colors[(i, 2)] = QColor('#c0392b')
            i += 1

        if snr:
            add('C/N0 均值', f"{snr['mean']:.2f} dB-Hz",
                '优' if snr['mean'] >= 40 else ('良' if snr['mean'] >= 35 else '差'),
                snr['mean'] >= 35)
            add('C/N0 < 30 dB-Hz 占比', f"{snr['lt30']:.2f}%",
                '正常' if snr['lt30'] < 10 else '偏高', snr['lt30'] < 10)
        if cont:
            add('平均完整率', f"{cont.get('rate_mean', 0):.2f}%",
                '优' if cont.get('rate_mean', 0) >= 99 else
                ('良' if cont.get('rate_mean', 0) >= 95 else '差'),
                cont.get('rate_mean', 0) >= 95)
            add('≥99% 完整率卫星数',
                f"{cont.get('n_ge99', 0)}/{cont.get('n_total', 0)}",
                f"失锁 {len(cont.get('loss_of_lock') or [])} 颗",
                not (cont.get('loss_of_lock')))
        if mp.get('available'):
            add('多路径 MP-RMS 均值', f"{mp.get('mean_rms', 0):.3f} m",
                '优' if mp.get('mean_rms', 0) < 0.5 else
                ('良' if mp.get('mean_rms', 0) < 1.0 else '差'),
                mp.get('mean_rms', 0) < 1.0)
            add('最差卫星 MP-RMS',
                f"{mp['rows'][0]['label']} {mp['rows'][0]['rms']:.3f} m"
                if mp.get('rows') else '—', '—')
            if mp.get('excluded_sats'):
                add('剔除卫星（L2 非真双频）',
                    f"{len(mp['excluded_sats'])} 颗",
                    '、'.join(mp['excluded_sats']), True)
        elif mp.get('single_frequency'):
            add('多路径 MP', '—', '无 L1/L2 真双频，无法计算', False)
        if geo.get('available'):
            add('仰角 ≥30° 占比', f"{geo.get('ge30', 0):.1f}%", '—')
            add('仰角 <15° 占比', f"{geo.get('lt15', 0):.1f}%", '—')
        else:
            add('几何（仰角/方位）', '—', '需广播星历', False)
        add('问题总数', f"{len(res.get('warnings') or [])} 项",
            '见下方问题清单', not res.get('warnings'))
        self._set_preview_rows('overall', rows, colors)

    def _fill_preview_warnings(self, res):
        warns = res.get('warnings') or []
        lvl_color = {'严重': QColor('#d9534f'), '关注': QColor('#f0ad4e'),
                     '提示': QColor('#5bc0de')}
        rows, colors = [], {}
        for i, w in enumerate(warns):
            rows.append([w['level'], w['title'], w['detail']])
            colors[(i, 0)] = lvl_color.get(w['level'], QColor('#333333'))
        if not rows:
            rows = [['—', '未发现明显问题', '']]
        self._set_preview_rows('warn', rows, colors)

    def _fill_preview_snr(self, res):
        bands = (res.get('snr_band') or {}).get('bands') or []
        rows = [[b['label'], b['n'], f"{b['min']:.1f}", f"{b['p5']:.1f}",
                 f"{b['p50']:.1f}", f"{b['p95']:.1f}", f"{b['max']:.1f}",
                 f"{b['mean']:.2f}"] for b in bands]
        self._set_preview_rows('snr', rows)

    def _fill_preview_snr_sat(self, res):
        sats = (res.get('snr_band') or {}).get('sats') or []
        rows, colors = [], {}
        for i, s in enumerate(sats):
            rows.append([s['label'], s['sys'], s['n'], f"{s['mean']:.2f}",
                         f"{s['median']:.1f}", f"{s['lt30']:.1f}",
                         f"{s['lt35']:.1f}"])
            if s['lt30'] > 30:
                colors[(i, 5)] = QColor('#d9534f')
            elif s['lt30'] > 15:
                colors[(i, 5)] = QColor('#c77e00')
        self._set_preview_rows('snr_sat', rows, colors)

    def _fill_preview_geo(self, res):
        geo = res.get('geometry') or {}
        if not geo.get('available'):
            self._set_preview_rows('geo', [['—', '—', '—', '—', '—',
                                            '需广播星历']])
            return
        rows = [[p['label'], f"{p['el_mean']:.1f}", f"{p['el_min']:.1f}",
                 f"{p['el_max']:.1f}", f"{p['az_mean']:.0f}",
                 f"{p['lt15']:.1f}"] for p in (geo.get('per_sat') or [])]
        self._set_preview_rows('geo', rows)

    def _fill_preview_resid(self, res):
        rs = res.get('residual') or {}
        if not rs.get('available'):
            self._set_preview_rows('resid', [['—', '—', '—', '—', '—',
                                              '需广播星历']])
            return
        rows, colors = [], {}
        for i, r in enumerate(rs.get('sats') or []):
            rows.append([r['label'], f"{r['el_mean']:.1f}",
                         f"{r['real']:.1f}", f"{r['expect']:.1f}",
                         f"{r['residual']:+.1f}", r['n']])
            if r['residual'] < -2.5:
                colors[(i, 4)] = QColor('#c0392b')
            elif r['residual'] < -1.5:
                colors[(i, 4)] = QColor('#c77e00')
        self._set_preview_rows('resid', rows, colors)

    def _fill_preview_cont(self, res):
        rows = res.get('continuity') or {}
        rows_s, colors = [], {}
        for i, r in enumerate(rows.get('rows') or []):
            rows_s.append([r['label'], r['sys'], r['n_epochs'],
                           f"{r['rate']:.2f}", r['n_gaps'], r['max_gap'],
                           r['verdict']])
            if r['kind'] == 'loss_of_lock':
                colors[(i, 3)] = QColor('#d9534f')
                colors[(i, 6)] = QColor('#d9534f')
            elif r['rate'] < 95:
                colors[(i, 3)] = QColor('#c77e00')
        self._set_preview_rows('cont', rows_s, colors)

    def _fill_preview_mp(self, res):
        mp = res.get('multipath') or {}
        rows, colors = [], {}
        for i, r in enumerate(mp.get('rows') or []):
            rows.append([r['label'], f"{r['rms']:.3f}", r['n']])
            if r['rms'] > 1.5:
                colors[(i, 1)] = QColor('#c77e00')
        if not rows:
            note = ('无 L1/L2 真双频，无法计算'
                    if mp.get('single_frequency') else '缺 GPS 双频观测值')
            rows = [['—', '—', note]]
        self._set_preview_rows('mp', rows, colors)

    def _fill_preview_mp_sys(self, res):
        """多系统多路径：逐系统汇总 + 逐卫星

        每个系统先给一行汇总（含判定出的频率对），
        便于核对频率自动识别是否合理 —— 频率配错量级会整个错。
        """
        ms = res.get('multipath_sys') or {}
        rows, colors = [], {}
        if not ms.get('available'):
            rows = [['—', '—', '—', '无多系统多路径数据（各系统无双频观测值）']]
        else:
            for name, blk in ms['systems'].items():
                if not blk.get('available'):
                    rows.append([name, '—', '—',
                                 blk.get('reason') or '不可用'])
                    colors[(len(rows) - 1, 3)] = QColor('#5f5e5a')
                    continue
                fr = blk.get('freq') or {}
                rows.append([name, f"{len(blk['rows'])} 颗",
                             f"均值 {blk['mean_rms']:.3f}",
                             f"频率 {fr.get('f1', 0) / 1e6:.3f}/"
                             f"{fr.get('f2', 0) / 1e6:.3f} MHz"])
                for r in blk['rows']:
                    rows.append([name, r['label'], f"{r['rms']:.3f}", r['n']])
                    if r['rms'] > 1.5:
                        colors[(len(rows) - 1, 2)] = QColor('#c77e00')
        self._set_preview_rows('mp_sys', rows, colors)

    def _fill_preview_mp_elev(self, res):
        """多路径与仰角的相关性：低仰角大是物理，中高仰角大才是问题"""
        me = res.get('mp_elev') or {}
        rows = []
        if not me.get('available'):
            rows = [['—', '—', '—', '无 MP-仰角相关性数据（需多路径与星历）']]
        else:
            for name, blk in me['systems'].items():
                if not blk.get('available'):
                    rows.append([name, '—', '—',
                                 blk.get('reason') or '不可用'])
                    continue
                r = blk.get('r')
                rows.append([name, blk.get('n'),
                             '—' if r is None else f'{r:+.3f}',
                             blk.get('verdict') or '—'])
            ge = me.get('gps_epoch')
            if ge:
                rows.append(['GPS（逐历元）', ge['n'], f"{ge['r']:+.3f}",
                             ge['verdict']])
        self._set_preview_rows('mp_elev', rows)

    def _fill_preview_iono(self, res):
        """电离层活动 ROT/IROT：分系统逐卫星 + 汇总行

        ROT/IROT 大小反映电离层扰动脉冲的强度，与周跳率、RTK 固定
        时间直接相关 —— 环境评估里解释"固定慢"的关键维度。
        """
        io_ = res.get('iono') or {}
        rows = []
        if not io_.get('available'):
            reason = io_.get('reason') or '各系统无双频载波观测值'
            rows = [['—', '—', '—', '—', '—', reason]]
        else:
            for name, blk in io_['systems'].items():
                if not blk.get('available'):
                    rows.append([name, '—', '—', '—', '—',
                                 blk.get('reason') or '不可用'])
                    continue
                lvl = (blk.get('level') or ('—',))[0]
                rows.append([name, '汇总',
                             '—' if blk.get('rot_med_typ') is None
                             else f"{blk['rot_med_typ']:.3f}",
                             '—' if blk.get('rot_max_all') is None
                             else f"{blk['rot_max_all']:.2f}",
                             '—',
                             f"电离层活动: {lvl}（按典型值）"])
                for r in blk['rows']:
                    rows.append([
                        name, r['label'],
                        f"{r['rot_med']:.3f}", f"{r['rot_max']:.2f}",
                        '—' if r.get('irot_med') is None
                        else f"{r['irot_med']:.3f}",
                        '—' if r.get('el_mean') is None
                        else f"{r['el_mean']:.1f}",
                    ])
        self._set_preview_rows('iono', rows)

    def _fill_preview_dop(self, res):
        """DOP / 卫星数汇总（与 RTKPLOT 的 DOP-NSat 图同口径）"""
        dp = res.get('dop_nsat') or {}
        rows, colors = [], {}
        if not dp.get('available'):
            rows = [['—', '—', '—', '—', dp.get('reason') or '不可用']]
        else:
            st = dp['stats']
            for key, name in (('gdop', 'GDOP'), ('pdop', 'PDOP'),
                              ('hdop', 'HDOP'), ('vdop', 'VDOP')):
                s = st[key]
                rows.append([name, f"{s['mean']:.2f}", f"{s['p50']:.2f}",
                             f"{s['p95']:.2f}", f"{s['max']:.2f}"])
            s = st['nsat_used']
            rows.append(['参与解算卫星数', f"{s['mean']:.1f}",
                         f"{s['min']:.0f}", '—', f"{s['max']:.0f}"])
            rows.append(['PDOP>4 占比', f"{st['pdop_gt4']:.2f}%", '', '', ''])
            rows.append(['PDOP>6 占比', f"{st['pdop_gt6']:.2f}%", '', '', ''])
            # 星历只覆盖部分系统时会显著高估 DOP，必须标出来
            eph_sys = list(dp.get('eph_systems') or [])
            allsys = list((res.get('info') or {}).get('sys_sat_count') or {})
            miss = [x for x in allsys if x not in eph_sys]
            rows.append(['星历覆盖系统', '、'.join(eph_sys) or '—', '', '',
                         f"缺 {'、'.join(miss)}" if miss else '完整'])
            if st['pdop_gt6'] > 10:
                colors[(6, 1)] = QColor('#c77e00')
            if st['min_sat'] < 5:
                colors[(4, 1)] = QColor('#c77e00')
            if miss:
                colors[(7, 4)] = QColor('#c77e00')
        self._set_preview_rows('dop', rows, colors)

    def _fill_preview_slips(self, res):
        """周跳 / 丢锁清单（几何无关组合 GF 判据）

        「符合整周%」是判别力的关键：真周跳必然是 n·λ，该比例应显著高于
        随机水平（容差 0.10 周时随机约 36%）。比例低而跳变密集，
        说明是持续性跟踪异常而非零散周跳。
        """
        sl = res.get('cycle_slips') or {}
        rows, colors = [], {}
        if not sl.get('available'):
            rows = [['—', '—', '—', '—', '—', '无法检测（缺双频载波）']]
        else:
            for r in (sl.get('rows') or [])[:24]:
                if r['usable']:
                    if r.get('chronic'):
                        first = '持续性跟踪异常，周跳数不计入总数'
                    elif r['events']:
                        e0 = r['events'][0]
                        first = (f"t={e0['t']:.2f}min {e0['dgf']:+.3f}m"
                                 + ('（丢锁）' if e0['kind'] == 'loss' else ''))
                    else:
                        first = '—'
                    ir = r.get('int_ratio')
                    rows.append([r['label'], r.get('n_raw', 0),
                                 f"{ir:.0f}" if ir is not None else '—',
                                 r['n_slips'], r['n_loss'], first])
                    i = len(rows) - 1
                    if r['n_loss']:
                        colors[(i, 4)] = QColor('#d9534f')
                    if r['n_slips'] and not r.get('chronic'):
                        colors[(i, 3)] = QColor('#c77e00')
                    if r.get('chronic'):
                        for c in (0, 1, 2):
                            colors[(i, c)] = QColor('#d9534f')
                else:
                    rows.append([r['label'], r.get('n_raw', 0), '—',
                                 '—', '—', r['reason']])
            rows.append(['合计', sum(r.get('n_raw', 0)
                                    for r in sl.get('rows', [])), '',
                         sl.get('total', 0), sl.get('total_loss', 0),
                         f"可检测 {sl.get('n_sats_checked', 0)} 颗 / "
                         f"无法检测 {sl.get('n_sats_skipped', 0)} 颗"])
        self._set_preview_rows('slip', rows, colors)

    def _fill_preview_obs(self, res):
        """原始观测量覆盖情况"""
        series = res.get('obs_series') or {}
        rows = []
        for key in sorted(series):
            sats = series[key]
            first = sorted(sats)[0]
            _ts, ps, ls, ds = sats[first]
            # NaN != NaN，用它判断该序列是否真有有效值
            have = [nm for nm, arr in (('伪距', ps), ('载波', ls),
                                      ('多普勒', ds))
                    if any(v == v for v in arr)]
            rows.append([f"{key[0]}-{key[1]}", len(sats),
                         max(len(v[0]) for v in sats.values()),
                         '、'.join(have) or '无有效值'])
        if not rows:
            rows = [['—', 0, 0, '没有可识别的观测码']]
        self._set_preview_rows('obs', rows)

    def _fill_preview_ts(self, res):
        ts = res.get('timeseries') or {}
        dur = ts.get('duration_min') or 0
        rows = [
            ['信噪比 C/N0', f"{len(ts.get('snr') or {})} 组频点",
             '、'.join(f"{k[0]}-{k[1]}" for k in (ts.get('snr') or {}))],
            ['仰角', len(ts.get('elev') or {}),
             f"时长 {dur:.1f} 分钟" + ('' if ts.get('elev') else '（需广播星历）')],
            ['多路径 MP', len(ts.get('mp') or {}), '仅 GPS 真双频卫星'],
            ['完整性条带', (res.get('continuity') or {}).get('n_total', 0),
             '逐卫星观测时段与中断位置'],
        ]
        self._set_preview_rows('ts', rows)

    # -------------------- 数据模式切换 --------------------

    def current_mode(self):
        """当前选中的数据模式"""
        if getattr(self, 'cmb_mode', None) is None:
            return MODE_DEBUG
        return self.cmb_mode.currentData() or MODE_DEBUG

    def _on_mode_changed(self, _index):
        """下拉框切换时：更新按钮可见性 + 数据页可见性

        只改变界面控件的可见性，不动任何已解析的数据，
        因此来回切换不会影响已有分析结果。
        """
        self._apply_mode_visibility()
        self._apply_page_visibility()
        mode = self.current_mode()
        if mode == MODE_STATIC:
            have = '已有结果，可点「🌐 静态数据分析」查看' if self.static_result \
                else '请点「📂 打开文件」选择观测文件'
            self.status_bar.showMessage(f"已切换到静态观测数据模式 — {have}")
        else:
            self.status_bar.showMessage(
                "已切换到 Debug 文件模式 — 请点「📂 打开文件」选择 .dbg 文件")

    def _apply_mode_visibility(self):
        """按当前模式显示/隐藏两组互斥按钮与左右面板"""
        is_static = self.current_mode() == MODE_STATIC
        for b in getattr(self, '_debug_buttons', []):
            b.setVisible(not is_static)
        for b in getattr(self, '_static_buttons', []):
            b.setVisible(is_static)
        self.btn_open.setText(
            "📂 打开观测文件" if is_static else "📂 打开文件")
        # 左侧面板：NMEA 的报文筛选/统计 与 静态的内容筛选/统计 互斥
        # 用 getattr 防御 —— 本方法在 _build_ui 里被调用时，
        # 面板可能还没建到那一步
        for name in ('type_group', 'stats_group'):
            w = getattr(self, name, None)
            if w is not None:
                w.setVisible(not is_static)
        for name in ('static_type_group', 'static_stats_group'):
            w = getattr(self, name, None)
            if w is not None:
                w.setVisible(is_static)

    def _apply_page_visibility(self):
        """按当前模式显示/隐藏 notebook 中的数据页

        静态模式只显示「静态数据预览」页；Debug 模式显示原有三页。
        页签索引按标题查找，避免依赖固定顺序。
        """
        nb = getattr(self, 'notebook', None)
        if nb is None:
            return
        prev = self._find_tab("静态数据预览")
        if prev < 0:
            return
        is_static = self.current_mode() == MODE_STATIC
        for title in ('NMEA0183报文', '二进制数据', '未知/其他'):
            i = self._find_tab(title)
            if i >= 0:
                nb.setTabVisible(i, not is_static)
        nb.setTabVisible(prev, is_static)
        if is_static:
            nb.setCurrentIndex(prev)

    def _find_tab(self, title):
        """按标题查找页签索引，找不到返回 -1"""
        nb = getattr(self, 'notebook', None)
        if nb is None:
            return -1
        for i in range(nb.count()):
            if nb.tabText(i) == title:
                return i
        return -1

    # -------------------- 文件打开与多线程解析 --------------------

    def open_file(self):
        """打开并解析文件

        按顶部下拉框选择的模式分流：
          - Debug 模式：只接受 .dbg，走原有 NMEA 解析流程
          - 静态模式  ：只接受 RINEX 观测文件，转交静态分析
        无论哪种模式，另一侧的解析结果都原样保留，不会被清空。
        """
        is_static = self.current_mode() == MODE_STATIC
        if is_static:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "选择静态观测文件（RINEX）", last_dir('open_rinex'),
                "RINEX观测文件 (*.26O *.25O *.24O *.rnx *.obs);;"
                "所有文件 (*.*)"
            )
            if not filepath:
                return
            save_last_dir('open_rinex', filepath)
            # 选了非 RINEX 文件时给出明确提示，而不是静默走错流程
            if os.path.splitext(filepath)[1].lower() not in RINEX_EXTS:
                QMessageBox.warning(
                    self, "文件类型不符",
                    f"当前为「观测数据」模式，但选中的文件不是 RINEX 观测文件:\n"
                    f"{os.path.basename(filepath)}\n\n"
                    "如需分析 Debug 文件，请先把顶部数据类型切回「Debug 文件」。")
                return
            # 主界面已有预览页，导入后**先展示预览**，用户想深挖再点
            # 「🔍 打开详细分析窗口」。直接弹模态对话框会打断浏览节奏，
            # 而且大文件解析完立刻弹窗、窗口又在主界面之下，体验很差。
            self.open_static_file(filepath, skip_dialog=True)
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择 Debug 文件", last_dir('open_dbg'),
            "Debug文件 (*.dbg);;所有文件 (*.*)"
        )
        if not filepath:
            return
        save_last_dir('open_dbg', filepath)

        # 兼容：Debug 模式下若误选 RINEX 文件，仍按扩展名分流到静态分析，
        # 保证原有「自动识别」行为不丢失。
        if os.path.splitext(filepath)[1].lower() in RINEX_EXTS:
            self.open_static_file(filepath, skip_dialog=True)
            return

        self.current_file = filepath
        self.lbl_file.setText(os.path.basename(filepath))
        self.lbl_file.setStyleSheet("color: black; font-style: normal;")
        self.status_bar.showMessage(f"正在解析 {os.path.basename(filepath)}...")

        # 禁用按钮
        self.btn_open.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_stats.setEnabled(False)
        self.btn_viz.setEnabled(False)
        self.btn_sat.setEnabled(False)
        self.btn_quality.setEnabled(False)
        self.btn_accuracy.setEnabled(False)
        self.btn_static.setEnabled(False)
        self.btn_static_nav.setEnabled(False)
        self.btn_static_eph_dl.setEnabled(False)

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 若上一个线程还在收尾，先等它结束，避免替换引用时销毁仍在运行的线程
        if self.parse_thread is not None:
            try:
                if self.parse_thread.isRunning():
                    self.parse_thread.quit()
                    self.parse_thread.wait(3000)
            except RuntimeError:
                pass

        # 创建工作线程
        self.parse_thread = QThread()
        self.parse_worker = ParseWorker(filepath)
        self.parse_worker.moveToThread(self.parse_thread)

        # 连接信号
        self.parse_thread.started.connect(self.parse_worker.run)
        self.parse_worker.progress.connect(self.on_parse_progress)
        self.parse_worker.finished.connect(self.on_parse_finished)
        self.parse_worker.error.connect(self.on_parse_error)
        self.parse_worker.finished.connect(self.parse_thread.quit)
        self.parse_worker.error.connect(self.parse_thread.quit)

        # 线程完全停止后(thread.finished)再清理引用。
        # 注意: 不能在 worker.finished 里就置 None —— 此时线程仍在收尾，
        #       引用清零会直接销毁仍在运行的 QThread 导致崩溃。
        self.parse_thread.finished.connect(self._on_thread_finished)

        self.parse_thread.start()

    def on_parse_progress(self, percent, message):
        """解析进度回调"""
        self.progress_bar.setValue(percent)
        self.status_bar.showMessage(message)

    def on_parse_finished(self, result):
        """解析完成回调"""
        self.parsed_data = result
        self.progress_bar.setVisible(False)

        self.btn_open.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_stats.setEnabled(True)
        self.btn_viz.setEnabled(True)
        self.btn_sat.setEnabled(True)
        self.btn_quality.setEnabled(True)
        self.btn_accuracy.setEnabled(True)
        # 静态分析入口与 NMEA 解析结果无关，解析结束后恢复可用
        self.btn_static.setEnabled(True)
        self.btn_static_nav.setEnabled(True)
        self.btn_static_eph_dl.setEnabled(True)

        self.populate_ui()

        n_nmea = len(result['nmea_messages'])
        n_bin = len(result['binary_packets'])
        n_unk = len(result['unknown_lines'])
        self.status_bar.showMessage(
            f"解析完成: NMEA {n_nmea}条, 二进制 {n_bin}条, 未知 {n_unk}条"
        )
        # 注意: 线程仍在收尾，不能在这里清空 parse_thread/parse_worker 引用，
        #       清空会在线程停止前销毁 QThread 导致崩溃，由 _on_thread_finished 处理。

    def on_parse_error(self, error_msg):
        """解析错误回调"""
        self.progress_bar.setVisible(False)
        self.btn_open.setEnabled(True)
        self.btn_static.setEnabled(True)
        self.btn_static_nav.setEnabled(True)
        self.btn_static_eph_dl.setEnabled(True)
        QMessageBox.critical(self, "解析错误", error_msg)
        self.status_bar.showMessage("解析失败")
        # 线程仍在收尾，引用清空由 _on_thread_finished 处理

    def _on_thread_finished(self):
        """QThread 完全停止后清理引用。

        此时线程已安全结束，置 None 触发 Python 回收销毁 QThread 是安全的；
        同时避免 closeEvent 等后续访问悬空引用。
        """
        self.parse_thread = None
        self.parse_worker = None

    # -------------------- 填充UI数据 --------------------

    def populate_ui(self):
        """填充UI数据"""
        self._populate_nmea_table()
        self._populate_binary_table()
        self._populate_unknown_table()
        self._populate_type_checkboxes()
        self._update_stats_text()

    def _populate_nmea_table(self):
        """填充NMEA报文表格（批量填充，异常GGA行高亮）"""
        self.nmea_table.setRowCount(0)
        self.all_nmea_rows = []
        self.type_counts = defaultdict(int)

        nmea_messages = self.parsed_data['nmea_messages']

        # 一次性计算异常 GGA（水平>5cm 或 高程>7.5cm），用 id(msg) 集合 O(1) 查行
        # 同时找出水平/高程误差最大的样本（橙色高亮）
        outlier_ids = set()
        max_horiz_id = max_vert_id = None
        max_horiz_val = max_vert_val = 0.0
        max_horiz_line = max_vert_line = ''
        gga_msgs = [m for m in nmea_messages if m.get('type') == 'GGA']
        if len(gga_msgs) >= 2:
            se = GGAStatistics.compute_sample_errors(gga_msgs)
            outlier_ids = {id(m) for m, flag in zip(gga_msgs, se['outliers']) if flag}

            valid_idx = np.where(se['valid'])[0]
            if len(valid_idx) > 0:
                k = int(np.argmax(se['horiz_errors'][valid_idx]))
                max_horiz_id = id(gga_msgs[valid_idx[k]])
                max_horiz_val = float(se['horiz_errors'][valid_idx[k]])
                max_horiz_line = gga_msgs[valid_idx[k]].get('line_no', '')
                # 高程误差可能存在 NaN（无高度），先置 0 再取最大绝对值
                v_vals = np.abs(np.nan_to_num(se['vert_errors'][valid_idx]))
                j = int(np.argmax(v_vals))
                max_vert_id = id(gga_msgs[valid_idx[j]])
                max_vert_val = float(v_vals[j])
                max_vert_line = gga_msgs[valid_idx[j]].get('line_no', '')

        self.max_error_info = {
            'horiz_line': max_horiz_line, 'horiz_val': max_horiz_val,
            'vert_line': max_vert_line, 'vert_val': max_vert_val,
        }

        # 预置行数批量填充，避免逐行 insertRow（大文件提速明显）
        self.nmea_table.setRowCount(len(nmea_messages))
        max_row = -1
        for row, msg in enumerate(nmea_messages):
            msg_type = msg.get('type', 'UNKNOWN')
            self.type_counts[msg_type] += 1
            p = msg.get('parsed', {})
            is_outlier = msg_type == 'GGA' and id(msg) in outlier_ids
            is_max = msg_type == 'GGA' and id(msg) == max_horiz_id
            if is_max:
                max_row = row

            values = self._row_values(msg_type, p, msg)
            for col, item in enumerate(self._styled_items(values, msg_type, is_outlier, is_max)):
                self.nmea_table.setItem(row, col, item)

            self.all_nmea_rows.append((msg_type, values, is_outlier, is_max))

        # 滚动到水平误差最大的行，便于直接查看
        if max_row >= 0:
            self.nmea_table.scrollToItem(self.nmea_table.item(max_row, 0))

    @staticmethod
    def _row_values(msg_type, p, msg):
        """构造一行的10个显示值（字符串），供表格填充与筛选共用"""
        return [
            str(msg.get('line_no', '')),
            msg_type,
            p.get('time_str', ''),
            f"{p['latitude']:.8f}" if p.get('latitude') is not None else '',
            f"{p['longitude']:.8f}" if p.get('longitude') is not None else '',
            f"{p['altitude']:.3f}" if p.get('altitude') is not None else '',
            str(p.get('fix_quality', '')) if p.get('fix_quality') is not None else '',
            str(p.get('num_sats', '')) if p.get('num_sats') is not None else '',
            f"{p['hdop']:.2f}" if p.get('hdop') is not None else '',
            msg.get('raw', '')[:200]
        ]

    @staticmethod
    def _styled_items(values, msg_type, is_outlier, is_max=False):
        """根据值列表生成带样式的 QTableWidgetItem 列表

        优先级：最大误差行(橙) > 异常行(浅红) > 类型列着色。
        """
        items = []
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            if is_max:
                item.setBackground(QBrush(MAX_ERR_BG))
            elif is_outlier:
                item.setBackground(QBrush(OUTLIER_BG))
            elif col == 1 and msg_type in TYPE_COLORS:
                item.setBackground(QBrush(TYPE_COLORS[msg_type]))
            if col == 0:
                item.setTextAlignment(Qt.AlignCenter)
            items.append(item)
        return items

    def _populate_binary_table(self):
        """填充二进制数据表格"""
        self.bin_table.setRowCount(0)
        for pkt in self.parsed_data['binary_packets']:
            row = self.bin_table.rowCount()
            self.bin_table.insertRow(row)
            self.bin_table.setItem(row, 0, QTableWidgetItem(str(pkt.get('line_no', ''))))
            self.bin_table.setItem(row, 1, QTableWidgetItem(str(pkt.get('length', 0))))
            self.bin_table.setItem(row, 2, QTableWidgetItem(pkt.get('hex_preview', '')))

    def _populate_unknown_table(self):
        """填充未知数据表格"""
        self.unknown_table.setRowCount(0)
        for line in self.parsed_data['unknown_lines']:
            row = self.unknown_table.rowCount()
            self.unknown_table.insertRow(row)
            self.unknown_table.setItem(row, 0, QTableWidgetItem(str(line.get('line_no', ''))))
            self.unknown_table.setItem(row, 1, QTableWidgetItem(line.get('content', '')))

    def _populate_type_checkboxes(self):
        """创建报文类型复选框"""
        # 清除旧的复选框
        for cb in self.type_checkboxes.values():
            cb.deleteLater()
        self.type_checkboxes.clear()

        for msg_type in sorted(self.type_counts.keys()):
            cb = QCheckBox(f"{msg_type} ({self.type_counts[msg_type]})")
            cb.setChecked(True)
            cb.stateChanged.connect(self.filter_by_type)
            self.scroll_layout.addWidget(cb)
            self.type_checkboxes[msg_type] = cb

    def _update_stats_text(self):
        """更新统计文本"""
        n_nmea = sum(self.type_counts.values())
        gga_count = self.type_counts.get('GGA', 0)
        text = f"文件: {os.path.basename(self.current_file)}\n"
        text += f"NMEA报文总数: {n_nmea}\n"
        text += f"GGA报文数: {gga_count}\n\n各类型数量:\n"
        for t in sorted(self.type_counts.keys()):
            text += f"  {t}: {self.type_counts[t]}\n"

        # 误差最大的样本
        mi = getattr(self, 'max_error_info', None)
        if mi and mi.get('horiz_line'):
            text += f"\n水平误差最大: 行{mi['horiz_line']} = {mi['horiz_val'] * 100:.2f}cm\n"
            text += f"高程误差最大: 行{mi['vert_line']} = {mi['vert_val'] * 100:.2f}cm\n"

        self.stats_text.setPlainText(text)

    # -------------------- 筛选功能 --------------------

    def filter_by_type(self):
        """根据复选框筛选报文类型（两遍式批量填充，保持异常高亮）"""
        selected_types = {t for t, cb in self.type_checkboxes.items() if cb.isChecked()}
        keyword = self.search_input.text().strip().lower()

        # 第一遍：确定匹配行，批量填充避免逐行 insertRow
        matched = []
        for msg_type, values, is_outlier, is_max in self.all_nmea_rows:
            if msg_type not in selected_types:
                continue
            if keyword:
                text = ' '.join(values).lower()
                if keyword not in text:
                    continue
            matched.append((msg_type, values, is_outlier, is_max))

        self.nmea_table.setRowCount(0)
        self.nmea_table.setRowCount(len(matched))
        for row, (msg_type, values, is_outlier, is_max) in enumerate(matched):
            for col, item in enumerate(self._styled_items(values, msg_type, is_outlier, is_max)):
                self.nmea_table.setItem(row, col, item)

    def filter_nmea(self):
        """搜索框实时筛选"""
        self.filter_by_type()

    def reset_filter(self):
        """重置筛选"""
        self.search_input.clear()
        for cb in self.type_checkboxes.values():
            cb.setChecked(True)
        self.filter_by_type()

    def select_all_types(self):
        for cb in self.type_checkboxes.values():
            cb.setChecked(True)

    def deselect_all_types(self):
        for cb in self.type_checkboxes.values():
            cb.setChecked(False)

    def on_nmea_double_click(self, index):
        """双击NMEA行显示详细信息"""
        row = index.row()
        item = self.nmea_table.item(row, 9)
        if not item:
            return
        raw = item.text()

        dialog = MessageDetailDialog(raw, self)
        dialog.exec_()

    # -------------------- CSV导出 --------------------

    def export_csv(self):
        """导出选中的报文类型为CSV"""
        if not self.parsed_data:
            QMessageBox.warning(self, "提示", "请先打开一个文件")
            return

        dialog = ExportTypeSelectionDialog(self.type_counts, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        selected = dialog.selected_types()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一种报文类型")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件",
            os.path.join(last_dir('export_csv'),
                         f"nmea_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
            "CSV文件 (*.csv);;所有文件 (*.*)"
        )
        if not filepath:
            return
        save_last_dir('export_csv', filepath)

        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # 表头: 经度、纬度、高程等各占一列
                writer.writerow([name for name, _ in CSV_EXPORT_COLUMNS])
                count = 0
                for msg in self.parsed_data['nmea_messages']:
                    if msg.get('type') in selected:
                        # 合并外层字段(line_no/type/raw)与解析字段
                        merged = {
                            'line_no': msg.get('line_no', ''),
                            'type': msg.get('type', ''),
                            'raw': msg.get('raw', ''),
                        }
                        merged.update(msg.get('parsed', {}))
                        writer.writerow(
                            [getter(merged) for _, getter in CSV_EXPORT_COLUMNS]
                        )
                        count += 1
            QMessageBox.information(self, "完成", f"已导出 {count} 条记录到:\n{filepath}")
            self.status_bar.showMessage(f"CSV导出完成: {count} 条记录")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

    # -------------------- GGA统计分析 --------------------

    def show_statistics(self):
        """显示GGA报文统计分析窗口"""
        if not self.parsed_data:
            QMessageBox.warning(self, "提示", "请先打开一个文件")
            return

        gga_msgs = [m for m in self.parsed_data['nmea_messages'] if m.get('type') == 'GGA']

        dialog = StatisticsDialog(gga_msgs, os.path.basename(self.current_file), self)
        if dialog.valid:
            dialog.exec_()

    # -------------------- 可视化 --------------------

    def show_visualization(self):
        """显示GGA精度可视化窗口"""
        if not self.parsed_data:
            QMessageBox.warning(self, "提示", "请先打开一个文件")
            return

        gga_msgs = [m for m in self.parsed_data['nmea_messages'] if m.get('type') == 'GGA']

        dialog = VisualizationDialog(gga_msgs, self)
        if dialog.valid:
            dialog.exec_()

    # -------------------- 卫星分析 --------------------

    def show_satellite_analysis(self):
        """显示GSV搜星/信噪比分析窗口"""
        if not self.parsed_data:
            QMessageBox.warning(self, "提示", "请先打开一个文件")
            return

        gsv_msgs = [m for m in self.parsed_data['nmea_messages'] if m.get('type') == 'GSV']
        satsinfo_msgs = [m for m in self.parsed_data['nmea_messages']
                         if m.get('type') in ('SATSINFOA', 'SATSINFOB')]

        dialog = SatelliteDialog(gsv_msgs, os.path.basename(self.current_file), self,
                                 satsinfo_msgs=satsinfo_msgs)
        if dialog.valid:
            dialog.exec_()

    # -------------------- 质量报告 --------------------

    def show_quality_report(self):
        """显示测量环境质量报告窗口"""
        if not self.parsed_data:
            QMessageBox.warning(self, "提示", "请先打开一个文件")
            return

        dialog = QualityReportDialog(
            self.parsed_data['nmea_messages'],
            os.path.basename(self.current_file), self,
            parse_stats=self.parsed_data)
        if dialog.valid:
            dialog.exec_()

    def show_accuracy_dialog(self):
        """精度评估：设备 NMEA 解算结果 vs 参考真值

        与「质量报告」的分工：
          * 质量报告看**观测质量**（信号好不好、能不能解算好）
          * 精度评估看**解算结果对不对**（相对真值差多少）
        后者必须有参考，所以界面上要先选参考来源。
        """
        if not self.parsed_data:
            QMessageBox.warning(self, "提示", "请先打开一个文件")
            return
        from .. import timebase
        from ..solution import SolutionSeries
        from .accuracy_dialog import AccuracyDialog

        msgs = self.parsed_data.get('nmea_messages') or []
        if not msgs:
            QMessageBox.warning(self, "提示", "该文件里没有 NMEA 报文。")
            return
        # 时间轴 + 解算序列；两者都只读，不影响 parsed_data
        timebase.build_time_axis(msgs)
        series = SolutionSeries.from_messages(msgs)
        if len(series) == 0:
            QMessageBox.warning(self, "提示",
                                "没有解析出解算历元（文件里缺 GGA 报文？）")
            return
        dlg = AccuracyDialog(series, os.path.basename(self.current_file), self)
        if dlg.valid:
            dlg.exec_()

    # ==================== 静态观测数据（RINEX）分析 ====================
    #
    # 以下为独立功能块，与上方 NMEA 流程完全解耦：
    #   * open_file() 只做一处分流（.26O 等扩展名转交 open_static_file）
    #   * 静态分析的进度条/线程/按钮状态都不触碰 NMEA 侧的引用

    def open_static_file(self, filepath=None, skip_dialog=False):
        """打开并分析静态观测文件（RINEX）

        传入 filepath 时直接分析该文件（供 open_file 分流调用）；
        不传时弹出文件选择框。

        skip_dialog=True 时不自动弹出分析对话框 —— 供「打开观测文件」后
        直接在主界面预览页展示详细结果，用户想深挖再自己点按钮打开对话框。

        星历处理顺序:
          1. 已有手动指定的星历 -> 直接用
          2. 观测文件附近自动搜索 -> 命中则自动加载
          3. 都没有 -> 先出一版不含几何的分析，并主动询问是否在线下载
        """
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "选择静态观测文件（RINEX）", last_dir('open_rinex'),
                "RINEX观测文件 (*.26O *.25O *.24O *.rnx *.obs);;"
                "所有文件 (*.*)"
            )
            if not filepath:
                return
        save_last_dir('open_rinex', filepath)
        self.static_obs_file = filepath
        # 换文件后允许重新弹出星历下载提示
        if filepath != getattr(self, '_last_obs_for_prompt', None):
            self._nav_prompted = False
            self._last_obs_for_prompt = filepath

        self.lbl_file.setText(os.path.basename(filepath))
        self.lbl_file.setStyleSheet("color: black; font-style: normal;")

        # 若用户还没指定星历，先在本地找一遍
        if not self.static_nav_file or not os.path.exists(self.static_nav_file):
            self._try_auto_locate_ephemeris(filepath)

        self.status_bar.showMessage(
            f"正在分析静态观测文件 {os.path.basename(filepath)}...")

        # 解析期间禁用静态入口（不影响 NMEA 侧按钮）
        self._set_static_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # 上一个线程若仍在收尾，先等它结束，避免替换引用销毁运行中的 QThread
        if self.rinex_thread is not None:
            try:
                if self.rinex_thread.isRunning():
                    self.rinex_thread.quit()
                    self.rinex_thread.wait(3000)
            except RuntimeError:
                pass

        self.rinex_thread = QThread()
        self.rinex_worker = RinexWorker(filepath, self.static_nav_file or None)
        self.rinex_worker.moveToThread(self.rinex_thread)

        # 是否在分析完成后自动弹出详细分析对话框
        self._auto_popup_dialog = not skip_dialog

        self.rinex_thread.started.connect(self.rinex_worker.run)
        self.rinex_worker.progress.connect(self.on_parse_progress)
        self.rinex_worker.finished.connect(self.on_static_finished)
        self.rinex_worker.error.connect(self.on_static_error)
        self.rinex_worker.finished.connect(self.rinex_thread.quit)
        self.rinex_worker.error.connect(self.rinex_thread.quit)
        # 同 NMEA 侧：必须等 thread.finished 才能清引用，否则会销毁运行中的线程
        self.rinex_thread.finished.connect(self._on_rinex_thread_finished)

        self.rinex_thread.start()

    # -------------------- 星历自动定位 --------------------

    def _try_auto_locate_ephemeris(self, obs_path):
        """在观测文件附近自动搜索匹配日期的广播星历

        通过解析观测文件头拿到时段，推出需要的 DOY，再按文件名匹配。
        找到则设置 self.static_nav_file；返回命中的路径（没找到返回 None）。
        """
        dates = RXE.dates_from_obs_header(obs_path)
        if not dates:
            return None

        fallback_year = dates[0][0]
        hit = RXE.auto_find(obs_path, dates, fallback_year=fallback_year)
        found = hit.get('found') or []
        if not found:
            self._pending_nav_dates = dates
            return None

        # 按星历类型优先级挑选（GPS/混合 > Galileo > BDS > GLONASS）
        # auto_find 已按日期去重，同一日期通常只剩一个候选
        def rank(item):
            path = item[0]
            return (RXE.nav_rank(path), -item[2])

        path, (y, d), size = sorted(found, key=rank)[0]
        self.static_nav_file = path
        self._pending_nav_dates = dates
        self.status_bar.showMessage(
            f"已自动匹配星历: {os.path.basename(path)}")
        return path

    # -------------------- 星历在线下载 --------------------

    def _ephemeris_parent(self):
        """星历相关弹窗应该挂在哪个窗口上

        「📡 获取星历」按钮在 StaticAnalysisDialog 里，而那个对话框是
        **模态**的。若 QMessageBox 以主窗口为父，它会被压在模态对话框
        **下面** —— 用户点了按钮却看不到任何反应，只能去任务栏翻窗口。
        所以：有活动的模态窗口就挂在它上面，并主动 raise + activate。
        """
        w = QApplication.activeModalWidget()
        return w if w is not None else self

    def _raise_front(self, w):
        """把窗口提到最前并尝试激活（最小化时还原）"""
        if w is None:
            return
        w.setWindowState((w.windowState() & ~Qt.WindowMinimized)
                         | Qt.WindowActive)
        w.raise_()
        w.activateWindow()

    def _ask_download_ephemeris(self):
        """询问用户是否在线下载广播星历（GUI 线程）"""
        dates = getattr(self, '_pending_nav_dates', None)
        if not dates:
            return
        desc = '、'.join(f"{y}年第{d:03d}天" for y, d in dates)
        parent = self._ephemeris_parent()
        box = QMessageBox(QMessageBox.Question, "缺少广播星历",
                          "仰角/方位角、卫星天顶图与「逐卫星 C/N0 残差」都需要"
                          "广播星历才能计算。\n\n"
                          f"本地未找到匹配的星历文件（需要 {desc}）。\n\n"
                          "是否现在从 IGS 服务器自动下载？\n"
                          "（约 1~2 MB，需要联网）",
                          QMessageBox.Yes | QMessageBox.No, parent)
        self._raise_front(box)
        ans = box.exec_()
        if ans != QMessageBox.Yes:
            self.status_bar.showMessage(
                "未加载星历：仅提供 C/N0、完整率与多路径分析")
            return
        self._start_ephemeris_download(dates)

    def _start_ephemeris_download(self, dates):
        """启动后台下载线程"""
        self._set_static_busy(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("正在下载广播星历...")

        self.dl_thread = QThread()
        self.dl_worker = EphemerisDownloadWorker(dates)
        self.dl_worker.moveToThread(self.dl_thread)

        self.dl_thread.started.connect(self.dl_worker.run)
        self.dl_worker.progress.connect(self.on_parse_progress)
        self.dl_worker.finished.connect(self.on_download_finished)
        self.dl_worker.error.connect(self.on_download_error)
        self.dl_worker.finished.connect(self.dl_thread.quit)
        self.dl_worker.error.connect(self.dl_thread.quit)
        self.dl_thread.finished.connect(self._on_dl_thread_finished)
        self.dl_thread.start()

    def on_download_finished(self, paths):
        """星历下载完成 -> 立即用新星历重新分析"""
        self.progress_bar.setVisible(False)
        self._set_static_busy(False)
        if paths:
            self.static_nav_file = paths[0]
            self.status_bar.showMessage(
                f"星历已就绪: {os.path.basename(paths[0])}，正在重新分析...")
            if self.static_obs_file:
                self.open_static_file(self.static_obs_file)

    def on_download_error(self, msg):
        """星历下载失败 -> 提示并提供手动指定入口"""
        self.progress_bar.setVisible(False)
        self._set_static_busy(False)
        parent = self._ephemeris_parent()
        box = QMessageBox(QMessageBox.Warning, "星历下载失败",
                          f"{msg}\n\n"
                          "你可以：\n"
                          "  • 手动指定本地星历文件（点「🗺 加载星历」）\n"
                          "  • 或先跳过，仅查看 C/N0 / 完整率 / 多路径分析\n\n"
                          "是否现在手动选择星历文件？",
                          QMessageBox.Yes | QMessageBox.No, parent)
        self._raise_front(box)
        ans = box.exec_()
        if ans == QMessageBox.Yes:
            self.load_static_ephemeris()

    def _on_dialog_request_ephemeris(self, obs_path):
        """对话框里点了「📡 获取星历」时的处理（GUI 线程）

        对话框是模态的，这里只发信号、不直接弹窗；等 exec_() 返回后
        由 on_static_finished 统一走「询问 -> 下载 -> 重算」流程，
        避免嵌套模态窗口。
        """
        self._dialog_wants_eph = True
        self._nav_prompted = False
        # 顺手再自动搜一遍（用户可能刚把星历文件拷进来）
        if obs_path and os.path.exists(obs_path):
            if self._try_auto_locate_ephemeris(obs_path):
                self.status_bar.showMessage(
                    f"已找到星历: {os.path.basename(self.static_nav_file)}，"
                    "关闭本窗口后将自动重新分析")

    def _on_dl_thread_finished(self):
        self.dl_thread = None
        self.dl_worker = None

    def _set_static_busy(self, busy):
        """切换静态分析相关按钮的可用状态"""
        self.btn_static.setEnabled(not busy)
        self.btn_static_nav.setEnabled(not busy)
        self.btn_static_eph_dl.setEnabled(not busy)

    def on_static_finished(self, result):
        """静态分析完成回调"""
        self.static_result = result
        self.progress_bar.setVisible(False)
        self._set_static_busy(False)

        # 无论是否弹对话框，都把结果铺到主界面预览页
        self.populate_static_preview(result)

        info = result['info']
        n_warn = len(result.get('warnings') or [])
        self.status_bar.showMessage(
            f"静态分析完成: {info['n_epochs']} 历元, {info['n_sats']} 颗卫星, "
            f"检测到 {n_warn} 项问题"
        )

        # 线程仍在收尾，引用清空交给 _on_rinex_thread_finished

        # 没有星历 -> 几何相关分析缺失，主动询问是否在线下载并重算。
        # 放在对话框关闭之后，避免嵌套模态窗口。
        need_ask = (not result.get('has_ephemeris')
                    and (not self._nav_prompted or self._dialog_wants_eph))

        # 是否需要自动弹出详细分析对话框（主界面已能预览，故默认不弹）
        if not getattr(self, '_auto_popup_dialog', True):
            if need_ask:
                self._dialog_wants_eph = False
                self._nav_prompted = True
                self._ask_download_ephemeris()
            return

        nav_name = (os.path.basename(self.static_nav_file)
                    if self.static_nav_file else None)
        dialog = StaticAnalysisDialog(
            result, os.path.basename(self.static_obs_file), nav_name, self,
            obs_path=self.static_obs_file)
        if not result.get('has_ephemeris'):
            dialog.request_ephemeris.connect(
                lambda: self._on_dialog_request_ephemeris(self.static_obs_file))
        if dialog.valid:
            dialog.exec_()

        if need_ask:
            self._dialog_wants_eph = False
            self._nav_prompted = True
            self._ask_download_ephemeris()

    def on_static_error(self, error_msg):
        """静态分析出错回调"""
        self.progress_bar.setVisible(False)
        self._set_static_busy(False)
        QMessageBox.critical(self, "静态分析失败", error_msg)
        self.status_bar.showMessage("静态分析失败")

    def _on_rinex_thread_finished(self):
        """RINEX 线程完全停止后清理引用（原因同 NMEA 侧）"""
        self.rinex_thread = None
        self.rinex_worker = None

    def show_static_analysis(self):
        """重新打开上一次静态分析结果

        已分析过则直接展示；否则引导用户选择文件。
        """
        if not self.static_result:
            self.open_static_file()
            return

        nav_name = (os.path.basename(self.static_nav_file)
                    if self.static_nav_file else None)
        dialog = StaticAnalysisDialog(
            self.static_result,
            os.path.basename(self.static_obs_file or ''), nav_name, self,
            obs_path=self.static_obs_file)
        if not self.static_result.get('has_ephemeris'):
            dialog.request_ephemeris.connect(
                lambda: self._on_dialog_request_ephemeris(self.static_obs_file))
        if dialog.valid:
            dialog.exec_()

        # 用户在对话框里点了「获取星历」-> 询问下载
        if (not self.static_result.get('has_ephemeris')
                and self._dialog_wants_eph):
            self._dialog_wants_eph = False
            self._nav_prompted = False
            self._ask_download_ephemeris()

    def download_ephemeris(self):
        """手动在线下载广播星历

        日期来源：
          * 已打开观测文件 -> 读文件头的观测时段（轻量，不解析全文件）
          * 尚未打开       -> 手动输入（YYYY-MM-DD 或 年 年积日）

        下载完成后若已有观测文件会自动重新分析（on_download_finished）。
        """
        dates = []
        if self.static_obs_file and os.path.exists(self.static_obs_file):
            dates = RXE.dates_from_obs_header(self.static_obs_file)
            if not dates:
                QMessageBox.warning(
                    self, '无法确定日期',
                    '没能从观测文件头读出观测时段'
                    '（缺 TIME OF FIRST OBS）\n'
                    '请手动输入日期，或改用「🗺 加载星历」选择本地文件。')
        if not dates:
            txt, ok = QInputDialog.getText(
                self, '下载星历',
                '输入观测日期（YYYY-MM-DD 或 "年 年积日"，如 2026-09-16 '
                '或 2026 259）：')
            if not ok or not txt.strip():
                return
            dates = self._parse_nav_date_text(txt.strip())
            if not dates:
                QMessageBox.warning(self, '日期格式不对',
                                    '无法识别的日期: '
                                    f'{txt.strip()}\n'
                                    '支持 YYYY-MM-DD 或 "年 年积日"。')
                return

        desc = '、'.join(f'{y}年第{d:03d}天' for y, d in dates)
        box = QMessageBox(
            QMessageBox.Question, '下载广播星历',
            f'将从 IGS 服务器下载 {desc} 的广播星历。\n'
            '（约 1~2 MB/天，需要联网；已存在的同名文件会跳过）\n\n'
            '是否继续？',
            QMessageBox.Yes | QMessageBox.No, self._ephemeris_parent())
        self._raise_front(box)
        if box.exec_() != QMessageBox.Yes:
            return
        self._start_ephemeris_download(dates)

    @staticmethod
    def _parse_nav_date_text(text):
        """解析用户输入的日期 -> [(year, doy), ...]

        支持 YYYY-MM-DD、YYYY/MM/DD、YYYY 年积日 三种写法。
        识别失败返回 []。
        """
        import datetime as _dt
        t = text.replace('，', ' ').replace(',', ' ').strip()
        for sep in ('-', '/'):
            if sep in t:
                parts = t.split(sep)
                if len(parts) == 3 and all(p.strip().isdigit()
                                           for p in parts):
                    y, m, d = (int(p) for p in parts)
                    try:
                        d0 = _dt.date(y, m, d)
                    except ValueError:
                        return []
                    return [(y, d0.timetuple().tm_yday)]
                return []
        parts = t.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            y, doy = int(parts[0]), int(parts[1])
            try:
                d0 = _dt.date(y, 1, 1) + _dt.timedelta(days=doy - 1)
            except ValueError:
                return []
            if d0.year == y and d0.timetuple().tm_yday == doy:
                return [(y, doy)]
            return []
        return []

    def load_static_ephemeris(self):
        """为静态分析加载广播星历（RINEX NAV）

        有了星历才能计算卫星仰角/方位角，以及最关键的
        「C/N0 仰角基线残差」诊断。加载后会立即重新分析。
        """
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择广播星历文件（RINEX NAV）", last_dir('open_nav'),
            "RINEX星历文件 (*.26P *.25P *.26N *.25N *.26G *.26B *.nav *.rnx);;"
            "所有文件 (*.*)"
        )
        if not filepath:
            return
        save_last_dir('open_nav', filepath)
        self.static_nav_file = filepath
        self.status_bar.showMessage(
            f"已加载星历 {os.path.basename(filepath)}")

        # 磁盘上的实际文件（可能用户选的是 .gz）不做解压，直接提示
        if filepath.lower().endswith('.gz'):
            QMessageBox.information(
                self, "提示",
                "选中的是压缩文件（.gz）。请先解压出 .rnx / .26n 后再加载。")
            self.static_nav_file = ''
            return

        # 已有观测文件则立即重算，让仰角相关分析生效
        if self.static_obs_file:
            self.open_static_file(self.static_obs_file)
        else:
            QMessageBox.information(
                self, "已加载星历",
                f"已记录星历文件:\n{os.path.basename(filepath)}\n\n"
                "接下来请打开静态观测文件，分析时会自动使用该星历。")

    # -------------------- 窗口关闭 --------------------

    def closeEvent(self, event):
        """窗口关闭时清理线程"""
        try:
            if self.parse_thread and self.parse_thread.isRunning():
                if self.parse_worker:
                    self.parse_worker.cancel()
                self.parse_thread.quit()
                # 分块等待线程完全停止(最多10s)，避免销毁仍在运行的QThread
                for _ in range(100):
                    if not self.parse_thread.isRunning():
                        break
                    self.parse_thread.wait(100)
        except RuntimeError:
            # 线程对象可能已被销毁，忽略即可
            pass
        try:
            if self.rinex_thread and self.rinex_thread.isRunning():
                if self.rinex_worker:
                    self.rinex_worker.cancel()
                self.rinex_thread.quit()
                for _ in range(100):
                    if not self.rinex_thread.isRunning():
                        break
                    self.rinex_thread.wait(100)
        except RuntimeError:
            pass
        try:
            if self.dl_thread and self.dl_thread.isRunning():
                if self.dl_worker:
                    self.dl_worker.cancel()
                self.dl_thread.quit()
                for _ in range(30):
                    if not self.dl_thread.isRunning():
                        break
                    self.dl_thread.wait(100)
        except RuntimeError:
            pass
        event.accept()
