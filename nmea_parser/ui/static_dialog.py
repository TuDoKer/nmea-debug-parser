# -*- coding: utf-8 -*-
"""静态观测数据质量分析对话框

面向 base/基准站静态观测数据（RINEX 观测文件 + 可选广播星历）。

布局（2×3 共 6 个子图 + 右下文本摘要）:
    [0,0] C/N0 直方图 + 质量分档      [0,1] C/N0-仰角基线       [0,2] 逐星 C/N0 残差
    [1,0] 连续性与完整率             [1,1] 卫星天顶图(仰角配色) [1,2] 摘要文本

风格与 satellite_dialog / quality_dialog 保持一致：
Figure + add_gridspec、ChineseNavToolbar、valid 标志、_save_figure。
"""

import os
import math
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QFileDialog, QMessageBox, QTabWidget, QWidget, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView, QCheckBox,
    QLineEdit, QPlainTextEdit, QSpinBox, QSplitter,
)
from PyQt5.QtCore import (
    pyqtSignal, Qt, QThread, QObject, QTimer, QEventLoop,
)
from PyQt5.QtGui import QFont, QColor

# 导入中文工具栏会先触发 matplotlib.use('Qt5Agg')
from ..toolbar import ChineseNavToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..config import last_dir, save_last_dir

# 星座配色（与 satellite_dialog / quality_dialog 保持同一套）
CONSTELLATION_COLORS = {
    'GPS': '#1f77b4',
    '北斗': '#d62728',
    'GLONASS': '#2ca02c',
    'Galileo': '#ff7f0e',
    'QZSS': '#9467bd',
    'SBAS': '#8c564b',
    'IRNSS': '#e377c2',
}

# 问题等级配色
LEVEL_COLORS = {
    '严重': '#d9534f',
    '关注': '#f0ad4e',
    '提示': '#5bc0de',
}

SNR_POOR = 30.0      # 差/良分界 (dB-Hz)
SNR_GOOD = 40.0      # 良/优分界 (dB-Hz)


def _f(x, nd=2):
    """安全格式化：None -> '—'"""
    if x is None:
        return '—'
    try:
        return f'{x:.{nd}f}'
    except (TypeError, ValueError):
        return '—'


class _BaselineWorker(QObject):
    """相对定位解算工作线程对象（rnx2rtkp 最长可跑 30 分钟，必须离线跑）

    信号契约与 rinex_worker.RinexWorker 一致：
        log(str) / finished(dict) / failed(str)
    """

    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, rover, base, outdir, sys_str, elev, rtklib=None):
        super().__init__()
        self.rover = rover
        self.base = base
        self.outdir = outdir
        self.sys_str = sys_str
        self.elev = elev
        self.rtklib = rtklib or None
        self._cancelled = False

    def cancel(self):
        """请求终止：置真后 run_solve 的轮询在 0.3 s 内杀掉子进程"""
        self._cancelled = True

    def run(self):
        """在子线程中执行：找工具 -> 收集星历 -> 调 rnx2rtkp -> 解析统计"""
        try:
            from .. import baseline as rb

            res = rb.solve(
                self.rover, self.base, outdir=self.outdir,
                sys_str=self.sys_str, elev=self.elev,
                rtklib=self.rtklib, log=self.log.emit,
                cancelled=lambda: self._cancelled)
        except Exception as e:      # noqa: BLE001 - BaselineError/任意异常都要回传 GUI
            self.failed.emit(str(e))
        else:
            self.finished.emit(res)


class StaticAnalysisDialog(QDialog):
    """静态观测数据质量分析对话框

    参数:
        result: StaticQuality.compute() 返回的结果字典
        file_basename: 显示用的文件名
        nav_basename: 使用的星历文件名（可为 None）
        obs_path: 观测文件完整路径（「相对定位」页需要，可为 None）
    """

    # 请求外部获取星历时发出（主窗口连接后弹出下载/选择星历）
    request_ephemeris = pyqtSignal()

    def __init__(self, result, file_basename, nav_basename=None, parent=None,
                 obs_path=None):
        super().__init__(parent)
        self.setWindowTitle("静态观测数据质量分析")
        self.resize(1500, 1000)
        self.valid = False

        if not result or result.get('error'):
            QMessageBox.warning(
                self, "提示",
                (result or {}).get('error', '没有可用的分析结果。'))
            return

        self._res = result
        self._file_basename = file_basename
        self._nav_basename = nav_basename
        self._obs_path = obs_path
        self._fig = None
        # 相对定位解算线程（运行期间的关闭拦截见 reject()）
        self._bl_thread = None
        self._bl_worker = None
        self._bl_timer = None
        self._bl_t0 = None
        self._bl_result = None      # 最近一次成功解算结果（CSV 导出数据源）
        self.valid = True

        self._build_ui()
        self._draw_all()

    # ==================== UI ====================

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 顶部信息条
        top = QHBoxLayout()
        info = self._res['info']
        self._lbl_head = QLabel(
            f"文件: {self._file_basename}    "
            f"MARKER: {info.get('marker') or '—'}    "
            f"时段: {info['t_first_str']} ~ {info['t_last_str']}")
        self._lbl_head.setFont(QFont("Microsoft YaHei", 9))
        top.addWidget(self._lbl_head)
        top.addStretch(1)

        if not self._res.get('has_ephemeris'):
            hint = QLabel("⚠ 未提供广播星历：仰角/方位角与基线残差不可用")
            hint.setStyleSheet("color: #c77e00; font-weight: bold;")
            top.addWidget(hint)
            # 对话框内直接提供获取入口，避免用户只盯着工具栏找不到。
            btn_eph = QPushButton("📡 获取星历")
            btn_eph.setToolTip(
                "自动在观测文件附近搜索广播星历；\n"
                "找不到时可从 IGS 在线下载当日 BRDC 星历并重新分析。")
            btn_eph.setStyleSheet("font-weight: bold;")
            btn_eph.clicked.connect(self.request_ephemeris.emit)
            top.addWidget(btn_eph)
        layout.addLayout(top)

        # 图表 / 数据表 两个标签页
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        # ---- 图表页 ----
        chart_page = QWidget()
        chart_layout = QVBoxLayout(chart_page)
        chart_layout.setContentsMargins(2, 2, 2, 2)

        fig = self._fig = Figure(figsize=(16, 10), dpi=100)
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05])
        self._ax_hist = fig.add_subplot(gs[0, 0])
        self._ax_base = fig.add_subplot(gs[0, 1])
        self._ax_resid = fig.add_subplot(gs[0, 2])
        self._ax_cont = fig.add_subplot(gs[1, 0])
        self._ax_sky = fig.add_subplot(gs[1, 1], projection='polar')
        self._ax_sum = fig.add_subplot(gs[1, 2])

        self._canvas = FigureCanvas(fig)
        chart_layout.addWidget(self._canvas, stretch=1)

        toolbar = ChineseNavToolbar(self._canvas, self)
        chart_layout.addWidget(toolbar)
        self._tabs.addTab(chart_page, "质量图表")

        # ---- 时间序列页（RTKPLOT 风格）----
        self._tabs.addTab(self._build_timeseries(), "时间序列")

        # ---- 观测诊断页（DOP/卫星数 + 原始观测量 + 仰角散点）----
        self._tabs.addTab(self._build_diagnostics(), "观测诊断")

        # ---- 相对定位页：双站静态解算（rnx2rtkp，调用 baseline 模块）----
        self._tabs.addTab(self._build_baseline(), "相对定位")

        # ---- 数据表页 ----
        self._tabs.addTab(self._build_tables(), "详细数据")

        # 底部按钮
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("💾 保存图片")
        btn_report = QPushButton("📄 导出报告")
        self._btn_close = btn_close = QPushButton("关闭")
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_report)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        btn_save.clicked.connect(self._save_figure)
        btn_report.clicked.connect(self._export_report)
        btn_close.clicked.connect(self.accept)

    def _build_timeseries(self):
        """构建时间序列页（对齐 RTKPLOT 的三联图 + 静态完整性）

        上三格与 RTKPLOT 一致，横轴统一为「观测开始后的分钟数」：
            [0] 信噪比 C/N0 (dB-Hz)  逐卫星散点
            [1] 多路径 MP (m)        逐卫星曲线
            [2] 仰角 (°)             逐卫星曲线
        [3] 静态完整性：逐卫星「可见时段条带」，
            一眼看出哪些卫星中途断了、断在哪。
        """
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(2, 2, 2, 2)

        ts = self._res.get('timeseries') or {}
        if not ts or not (ts.get('snr') or ts.get('mp') or ts.get('elev')):
            v.addWidget(QLabel("没有可用于绘制时间序列的数据。"))
            self._ts_fig = None
            self._ts_ax = None
            return page

        # 顶部：频点选择（只影响 SNR 图）
        bar = QHBoxLayout()
        bar.addWidget(QLabel("信噪比频点:"))
        self._ts_band = QComboBox()
        for key in (ts.get('snr') or {}):
            self._ts_band.addItem(f"{key[0]}-{key[1]}", key)
        if self._ts_band.count() == 0:
            self._ts_band.addItem("(无)", None)
        bar.addWidget(self._ts_band)
        bar.addStretch(1)
        dur = ts.get('duration_min') or 0
        bar.addWidget(QLabel(f"时长 {dur:.1f} 分钟"))
        v.addLayout(bar)

        fig = self._ts_fig = Figure(figsize=(15, 10), dpi=100)
        gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1.0, 1.0, 0.85],
                              hspace=0.42)
        self._ts_ax_snr = fig.add_subplot(gs[0])
        self._ts_ax_mp = fig.add_subplot(gs[1], sharex=self._ts_ax_snr)
        self._ts_ax_el = fig.add_subplot(gs[2], sharex=self._ts_ax_snr)
        self._ts_ax_int = fig.add_subplot(gs[3], sharex=self._ts_ax_snr)
        self._ts_ax = (self._ts_ax_snr, self._ts_ax_mp,
                       self._ts_ax_el, self._ts_ax_int)

        canvas = FigureCanvas(fig)
        v.addWidget(canvas, stretch=1)
        v.addWidget(ChineseNavToolbar(canvas, self))

        self._ts_band.currentIndexChanged.connect(self._draw_timeseries)
        self._draw_timeseries()
        return page

    # ---- 时间序列绘图 ----

    # 卫星配色：按序号循环取色，同一颗星在三张图里颜色一致
    _SAT_PALETTE = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
        '#aec7e8', '#ff9896', '#98df8a', '#ffbb78', '#c5b0d5',
    ]

    @classmethod
    def _sat_color(cls, satid):
        try:
            idx = int(satid[1:]) % len(cls._SAT_PALETTE)
        except (ValueError, IndexError):
            idx = 0
        return cls._SAT_PALETTE[idx]

    def _draw_timeseries(self):
        if not getattr(self, '_ts_fig', None):
            return
        ts = self._res.get('timeseries') or {}
        ax_snr, ax_mp, ax_el, ax_int = self._ts_ax

        # ---------- [0] 信噪比 ----------
        ax_snr.clear()
        key = self._ts_band.currentData()
        groups = ts.get('snr') or {}
        series = groups.get(key) or {}
        if series:
            for satid, (t, val) in sorted(series.items(),
                                          key=lambda kv: kv[0]):
                ax_snr.plot(t, val, '.', ms=1.6,
                            color=self._sat_color(satid), alpha=0.65)
            ax_snr.set_ylabel('C/N0 (dB-Hz)', fontsize=10)
            ax_snr.set_title(
                f"信噪比 C/N0 — {key[0]}-{key[1]}（{len(series)} 颗卫星）",
                fontsize=11, loc='left')
            ax_snr.grid(alpha=0.25, linestyle=':')
            # 质量分档参考线
            ax_snr.axhline(SNR_POOR, color='#f0ad4e', lw=0.9, alpha=0.7)
            ax_snr.axhline(SNR_GOOD, color='#5cb85c', lw=0.9, alpha=0.7)
            ax_snr.text(0.002, SNR_POOR, ' 差/良 30', fontsize=7,
                        color='#c77e00', va='bottom',
                        transform=ax_snr.get_yaxis_transform())
            ax_snr.text(0.002, SNR_GOOD, ' 良/优 40', fontsize=7,
                        color='#4a8b3a', va='bottom',
                        transform=ax_snr.get_yaxis_transform())
        else:
            ax_snr.text(0.5, 0.5, '无信噪比数据', ha='center', va='center',
                        transform=ax_snr.transAxes)
            ax_snr.set_ylabel('C/N0 (dB-Hz)', fontsize=10)

        # ---------- [1] 多路径 ----------
        ax_mp.clear()
        mp = ts.get('mp') or {}
        if mp:
            allv = []
            for satid, (t, val) in sorted(mp.items(), key=lambda kv: kv[0]):
                ax_mp.plot(t, val, '-', lw=0.7,
                           color=self._sat_color(satid), alpha=0.8)
                allv.extend(val)
            ax_mp.set_ylabel('多路径 MP (m)', fontsize=10)
            ax_mp.set_title(
                f"多路径 MP（GPS 真双频 {len(mp)} 颗，已剔除仅 L1 的卫星）",
                fontsize=11, loc='left')
            ax_mp.grid(alpha=0.25, linestyle=':')
            ax_mp.axhline(0, color='#888888', lw=0.8)
            # ±0.5 m 是通常认为的良好水平，画出来作参照
            for y in (0.5, -0.5):
                ax_mp.axhline(y, color='#5cb85c', lw=0.8, ls='--', alpha=0.6)
            # 固定 ±10 m 显示范围。原来用 99.5 分位自适应，好处是不会被
            # 个别尖峰压扁，坏处是不同文件/不同批次的纵轴不一样，
            # 没法横向对比 —— 同一台接收机今天 ±0.8、明天 ±3.0，
            # 单看图根本分不清是变好了还是只是分位变了。
            # MP 通常认为良好的水平是 ±0.5 m，±10 m 已能容纳低仰角尖峰。
            if allv:
                lim = 10.0
                ax_mp.set_ylim(-lim, lim)
                clipped = sum(1 for v in allv if abs(v) > lim)
                if clipped:
                    ax_mp.text(0.998, 0.02,
                               f"⚠ {clipped} 点超出 ±{lim:.0f} m 未显示",
                               transform=ax_mp.transAxes, ha='right',
                               va='bottom', fontsize=7, color='#c77e00')
        else:
            mp_res = self._res.get('multipath') or {}
            msg = ('该文件不具备 L1/L2 真双频条件，无法计算多路径'
                   if mp_res.get('single_frequency')
                   else '无多路径数据（需 GPS 双频观测值）')
            ax_mp.text(0.5, 0.5, msg, ha='center', va='center',
                       transform=ax_mp.transAxes, fontsize=9, color='#c77e00')
            ax_mp.set_ylabel('多路径 MP (m)', fontsize=10)

        # ---------- [2] 仰角 ----------
        ax_el.clear()
        elev = ts.get('elev') or {}
        if elev:
            for satid, (t, val) in sorted(elev.items(), key=lambda kv: kv[0]):
                ax_el.plot(t, val, '-', lw=1.0,
                           color=self._sat_color(satid), alpha=0.85)
            ax_el.set_ylabel('仰角 (°)', fontsize=10)
            ax_el.set_title(
                f"卫星仰角（{len(elev)} 颗，需广播星历）",
                fontsize=11, loc='left')
            ax_el.grid(alpha=0.25, linestyle=':')
            # 15° 截止高度角参考线
            ax_el.axhline(15, color='#f0ad4e', lw=0.9, ls='--', alpha=0.7)
            ax_el.text(0.002, 15, ' 截止高度角 15°', fontsize=7,
                       color='#c77e00', va='bottom',
                       transform=ax_el.get_yaxis_transform())
            ax_el.set_ylim(bottom=0)
        else:
            ax_el.text(0.5, 0.5, '无仰角数据（未提供广播星历）',
                       ha='center', va='center',
                       transform=ax_el.transAxes, fontsize=9, color='#c77e00')
            ax_el.set_ylabel('仰角 (°)', fontsize=10)

        # ---------- [3] 静态完整性条带 ----------
        self._draw_integrity(ax_int)

        ax_int.set_xlabel('观测开始后的时间（分钟）', fontsize=10)
        for a in (ax_snr, ax_mp, ax_el):
            a.tick_params(labelbottom=False)

        self._ts_fig.subplots_adjust(left=0.075, right=0.985, top=0.955,
                                     bottom=0.075, hspace=0.42)
        try:
            self._ts_fig.canvas.draw_idle()
        except Exception:
            pass

    def _draw_integrity(self, ax):
        """静态完整性条带图

        每颗卫星一条水平带：整段底色表示「可见跨度」，
        其中绿色=有观测、红色=中断段。
        一眼看清哪些卫星中途失锁、断在第几分钟。

        为了能画出中断的**确切位置**，这里直接使用 continuity 行里
        预存的中断区间（gap_segs），而不是只靠 max_gap 猜。
        """
        ax.clear()
        cdat = self._res.get('continuity') or {}
        cont = cdat.get('rows') or []
        ts = self._res.get('timeseries') or {}
        dur = ts.get('duration_min') or 0
        dt_min = (ts.get('dt') or 1.0) / 60.0
        if not cont or dur <= 0:
            ax.text(0.5, 0.5, '无连续性数据', ha='center', va='center',
                    transform=ax.transAxes)
            return

        # 按完整率升序，问题最多的排在下面
        cont = sorted(cont, key=lambda r: r['rate'])
        n = len(cont)
        limit = 45
        shown = cont[:limit] if n > limit else cont
        note = f"（显示完整率最低的 {len(shown)}/{n} 颗）" if n > limit else ''

        loss = [r for r in shown if r['kind'] == 'loss_of_lock']
        for i, r in enumerate(shown):
            y = i
            t0 = r['first'] * dt_min
            t1 = r['last'] * dt_min
            span = max(t1 - t0, dur * 0.005)
            # 可见跨度：浅灰底
            ax.barh(y, span, left=t0, height=0.62,
                    color='#e9e9e9', edgecolor='none', zorder=1)
            # 中断段：红色（用预存的确切区间）
            for gs, ge in (r.get('gap_segs') or []):
                ax.barh(y, max((ge - gs) * dt_min, dur * 0.004),
                        left=gs * dt_min, height=0.62,
                        color='#d9534f', edgecolor='none', zorder=3)
            # 有观测的历元：绿色点阵（只在跨度内画，跳过中断段）
            gset = r.get('epoch_list')
            if gset:
                tt = [e * dt_min for e in gset]
                ax.plot(tt, [y] * len(tt), '|', ms=3.0, mew=0.55,
                        color='#3f9e3f', alpha=0.85, zorder=2)
            else:
                ax.barh(y, span, left=t0, height=0.62,
                        color='#8fc98f', edgecolor='none', zorder=2)

        # 卫星多时按步长抽稀标签，否则文字会糊成一团。
        # 必须用 set_yticks(positions, labels) 一次性设定，
        # 分两步调用会被 Axes 的自动刻度逻辑重新填满。
        step = max(1, int(np.ceil(len(shown) / 18.0)))
        pos = list(range(0, len(shown), step))
        ax.set_yticks(pos)
        ax.set_yticklabels([shown[i]['label'] for i in pos], fontsize=6.5)
        ax.set_ylim(-0.7, len(shown) - 0.3)
        ax.set_xlim(0, dur * 1.015)
        ax.grid(axis='x', alpha=0.22, linestyle=':')
        ax.set_title(
            f"静态完整性：逐卫星观测时段"
            f"（绿=有记录 红=中断 灰=可见跨度）{note}",
            fontsize=11, loc='left')
        ax.text(0.998, 1.015,
                f"完整率均值 {cdat.get('rate_mean', 0):.1f}%  |  "
                f"≥99% {cdat.get('n_ge99', 0)}/{cdat.get('n_total', 0)} 颗"
                + (f"  |  ⚠ {len(loss)} 颗真实失锁" if loss else '  |  无失锁'),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='#c0392b' if loss else '#4a8b3a')

    # -------------------- 相对定位（双站静态解算） --------------------

    def _build_baseline(self):
        """「相对定位」页：调用本机 RTKLIB rnx2rtkp 做双站静态相对定位

        当前分析的观测文件作 base（基准站），用户另选一台接收机同场次
        的观测文件作 rover（待定站），解出 ENU 基线并展示：
            - 文字报告：固定率 / 基线长度 / 内符合重复性（与命令行一致）
            - E/N/U 逐历元时序图：按解类型着色，固定解统计均值±σ
        解算逻辑复用 baseline.py（原 rinex_baseline.py 的 solve() 入口），
        结果 .pos 保存在 rover 观测文件所在目录。
        """
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(2, 2, 2, 2)

        # ---- 第一行：站设置 ----
        row1 = QHBoxLayout()
        base_name = (os.path.basename(self._obs_path)
                     if self._obs_path else self._file_basename)
        row1.addWidget(QLabel(f"基准站 base: {base_name}（当前文件）"))
        row1.addStretch(1)
        row1.addWidget(QLabel("第二站 rover:"))
        self._bl_rover = QLineEdit()
        self._bl_rover.setPlaceholderText(
            "另一台接收机同场次的 RINEX 观测文件（.26O / .rnx / .obs）")
        btn_pick = QPushButton("浏览…")
        btn_pick.clicked.connect(self._pick_rover)
        row1.addWidget(self._bl_rover, stretch=1)
        row1.addWidget(btn_pick)
        v.addLayout(row1)

        # ---- 第二行：解算参数 ----
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("参与系统:"))
        self._bl_sys = QComboBox()
        self._bl_sys.addItem("G,E,C（GPS+Galileo+BDS，默认）", "G,E,C")
        self._bl_sys.addItem("G,E,C,R（含 GLONASS，自动修复单位）", "G,E,C,R")
        self._bl_sys.addItem("G（仅 GPS）", "G")
        row2.addWidget(self._bl_sys)
        row2.addWidget(QLabel("截止高度角:"))
        self._bl_elev = QSpinBox()
        self._bl_elev.setRange(5, 30)
        self._bl_elev.setValue(15)
        self._bl_elev.setSuffix("°")
        self._bl_elev.setToolTip("低于该仰角的卫星不参与解算，默认 15°")
        row2.addWidget(self._bl_elev)
        row2.addWidget(QLabel("RTKLIB:"))
        self._bl_rtklib = QLineEdit()
        self._bl_rtklib.setPlaceholderText("留空自动查找 rnx2rtkp.exe")
        btn_rtk = QPushButton("浏览…")
        btn_rtk.clicked.connect(self._pick_rtklib)
        row2.addWidget(self._bl_rtklib, stretch=1)
        row2.addWidget(btn_rtk)
        self._bl_btn_start = QPushButton("▶ 开始解算")
        self._bl_btn_start.clicked.connect(self._start_baseline)
        row2.addWidget(self._bl_btn_start)
        self._bl_btn_cancel = QPushButton("⏹ 终止")
        self._bl_btn_cancel.setToolTip(
            "终止正在进行的解算（杀掉 rnx2rtkp 子进程）；\n"
            "解算期间点关闭窗口也会先问是否终止。")
        self._bl_btn_cancel.setEnabled(False)
        self._bl_btn_cancel.clicked.connect(self._cancel_baseline)
        row2.addWidget(self._bl_btn_cancel)
        v.addLayout(row2)

        # ---- 第三行：状态 + 固定率警告 + 明细导出 ----
        row3 = QHBoxLayout()
        self._bl_status = QLabel(
            "尚未解算。选择第二站观测文件后点「开始解算」。")
        self._bl_status.setFont(QFont("Microsoft YaHei", 9))
        row3.addWidget(self._bl_status, stretch=1)
        self._bl_warn = QLabel()
        self._bl_warn.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        self._bl_warn.setVisible(False)
        row3.addWidget(self._bl_warn)
        self._bl_btn_export = QPushButton("📤 导出ENU明细")
        self._bl_btn_export.setToolTip(
            "把逐历元 E/N/U 解算明细导出为 CSV（可用 Excel 打开）")
        self._bl_btn_export.setEnabled(False)
        self._bl_btn_export.clicked.connect(self._export_bl_csv)
        row3.addWidget(self._bl_btn_export)
        v.addLayout(row3)

        if not self._obs_path:
            self._bl_btn_start.setEnabled(False)
            self._bl_status.setText(
                "缺少当前观测文件的完整路径，无法作为基准站，请重新打开分析。")
            self._bl_status.setStyleSheet("color:#c0392b;")

        # ---- 上：E/N/U 时序图；下：日志与报告 ----
        self._bl_fig = Figure(figsize=(14, 8), dpi=100)
        gs = self._bl_fig.add_gridspec(3, 1, hspace=0.40)
        self._bl_ax_e = self._bl_fig.add_subplot(gs[0])
        self._bl_ax_n = self._bl_fig.add_subplot(gs[1], sharex=self._bl_ax_e)
        self._bl_ax_u = self._bl_fig.add_subplot(gs[2], sharex=self._bl_ax_e)
        for ax in (self._bl_ax_e, self._bl_ax_n, self._bl_ax_u):
            ax.grid(True, alpha=0.25, linestyle=':')

        fig_box = QWidget()
        fl = QVBoxLayout(fig_box)
        fl.setContentsMargins(0, 0, 0, 0)
        self._bl_canvas = FigureCanvas(self._bl_fig)
        fl.addWidget(self._bl_canvas, stretch=1)
        fl.addWidget(ChineseNavToolbar(self._bl_canvas, self))

        self._bl_report = QPlainTextEdit()
        self._bl_report.setReadOnly(True)
        self._bl_report.setFont(QFont("Consolas", 9))
        self._bl_report.setPlaceholderText(
            "解算过程日志与结果报告显示在这里…")

        split = QSplitter(Qt.Vertical)
        split.addWidget(fig_box)
        split.addWidget(self._bl_report)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        v.addWidget(split, stretch=1)
        return page

    # ---- 文件选择 ----

    def _pick_rover(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择第二站（rover）观测文件', last_dir('open_rinex'),
            "RINEX观测文件 (*.26O *.25O *.24O *.rnx *.obs);;"
            "所有文件 (*.*)")
        if path:
            self._bl_rover.setText(path)
            save_last_dir('open_rinex', path)

    def _pick_rtklib(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择 rnx2rtkp.exe', last_dir('rtklib'),
            "RTKLIB 解算器 (rnx2rtkp.exe);;所有文件 (*.*)")
        if path:
            self._bl_rtklib.setText(path)
            save_last_dir('rtklib', path)

    # ---- 解算流程 ----

    def _start_baseline(self):
        """启动后台解算线程（rnx2rtkp 可能跑数分钟，绝不能在 GUI 线程）"""
        if self._bl_thread is not None:
            return
        rover = (self._bl_rover.text() or '').strip()
        if not rover or not os.path.exists(rover):
            QMessageBox.warning(self, '提示',
                                '请先选择第二站（rover）的 RINEX 观测文件。')
            return
        if not self._obs_path or not os.path.exists(self._obs_path):
            QMessageBox.warning(self, '提示',
                                '当前观测文件路径不可用，无法作为基准站。')
            return
        if os.path.normcase(os.path.abspath(rover)) == \
                os.path.normcase(os.path.abspath(self._obs_path)):
            QMessageBox.warning(self, '提示',
                                '基准站和第二站不能是同一个观测文件。')
            return

        # 结果 .pos 与日志一起放在 rover 文件旁边，方便成对归档
        outdir = os.path.dirname(rover)
        self._bl_report.clear()
        self._bl_warn.setVisible(False)
        self._bl_set_busy(True)

        self._bl_t0 = datetime.now()
        self._bl_timer = QTimer(self)
        self._bl_timer.timeout.connect(self._on_bl_tick)
        self._bl_timer.start(500)

        self._bl_thread = QThread()
        self._bl_worker = _BaselineWorker(
            rover, self._obs_path, outdir,
            self._bl_sys.currentData(), self._bl_elev.value(),
            (self._bl_rtklib.text() or '').strip() or None)
        self._bl_worker.moveToThread(self._bl_thread)
        self._bl_thread.started.connect(self._bl_worker.run)
        self._bl_worker.log.connect(self._append_bl_log)
        self._bl_worker.finished.connect(self._on_bl_finished)
        self._bl_worker.failed.connect(self._on_bl_failed)
        self._bl_worker.finished.connect(self._bl_thread.quit)
        self._bl_worker.failed.connect(self._bl_thread.quit)
        # 必须等 thread.finished 才清引用，否则会销毁运行中的 QThread
        self._bl_thread.finished.connect(self._on_bl_thread_finished)
        self._bl_thread.start()

        self._bl_status.setText('解算中…')
        self._bl_status.setStyleSheet('')

    def _on_bl_tick(self):
        secs = int((datetime.now() - self._bl_t0).total_seconds())
        self._bl_status.setText(
            f'解算中… 已用时 {secs} s（rnx2rtkp 静态解算，最长 30 分钟）')

    def _append_bl_log(self, msg):
        self._bl_report.appendPlainText(str(msg))

    def _on_bl_finished(self, res):
        """解算成功：存结果 + 刷固定率警告 + 追报告文本 + 绘 ENU 时序图"""
        if self._bl_timer is not None:
            self._bl_timer.stop()
        self._bl_result = res                 # 先存结果，busy 复位时才能开导出
        self._bl_set_busy(False)
        self._refresh_bl_warning(res.get('rows') or [])
        self._bl_report.appendPlainText('')
        self._bl_report.appendPlainText(res.get('report', '').rstrip())
        self._draw_bl_enu(res)
        out = res.get('out_pos')
        self._bl_status.setText(
            f"解算完成：结果文件 {out}" if out else "解算完成。")
        self._bl_status.setStyleSheet('color:#4a8b3a; font-weight:bold;')

    def _on_bl_failed(self, msg):
        if self._bl_timer is not None:
            self._bl_timer.stop()
        self._bl_set_busy(False)
        cancelled = '已取消' in msg
        self._bl_report.appendPlainText('')
        self._bl_report.appendPlainText(
            f'已终止: {msg}' if cancelled else f'解算失败: {msg}')
        if cancelled:
            self._bl_status.setText(f'⏹ {msg}')
            self._bl_status.setStyleSheet('color:#c77e00; font-weight:bold;')
        else:
            self._bl_status.setText(f'解算失败: {msg}')
            self._bl_status.setStyleSheet('color:#c0392b; font-weight:bold;')

    def _on_bl_thread_finished(self):
        if self._bl_timer is not None:
            self._bl_timer.stop()
            self._bl_timer = None
        self._bl_thread = None
        self._bl_worker = None

    def _bl_set_busy(self, busy):
        """解算期间：开「⏹ 终止」、关「开始解算」；结束后按结果恢复导出"""
        self._bl_btn_start.setEnabled(
            not busy and bool(self._obs_path))
        self._bl_btn_cancel.setEnabled(busy)
        self._bl_btn_export.setEnabled(
            not busy and bool(self._bl_result))

    def _cancel_baseline(self):
        """⏹ 终止按钮：通知 worker 杀子进程（run_solve 轮询 0.3 s 内生效）"""
        if self._bl_worker is None:
            return
        self._bl_btn_cancel.setEnabled(False)
        self._bl_status.setText('正在终止…（等待 rnx2rtkp 退出）')
        self._bl_status.setStyleSheet('color:#c77e00; font-weight:bold;')
        self._bl_worker.cancel()

    # ---- 关闭拦截（窗口 X 与「关闭」按钮共用） ----

    def accept(self):
        self._guarded_close()

    def reject(self):
        self._guarded_close()

    def _guarded_close(self):
        """解算运行中关闭 -> 询问是否终止；等线程收尾后再关，不带线程销毁窗口"""
        if self._bl_thread is not None:
            ans = QMessageBox.question(
                self, '解算进行中',
                '相对定位正在解算，是否终止解算并关闭窗口？\n'
                '（选择「否」返回继续等待解算完成）',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
            if self._bl_worker is not None:
                self._bl_worker.cancel()
            if self._bl_thread.isRunning():
                # 嵌套事件循环：等待期间让 worker.failed -> thread.quit ->
                # thread.finished 这些排队槽正常执行（_on_bl_thread_finished
                # 的连接早于 loop.quit，会先触发并把 _bl_thread 清成 None）
                loop = QEventLoop()
                self._bl_thread.finished.connect(loop.quit)
                QTimer.singleShot(20000, loop.quit)   # 安全阀：20 s 兜底
                loop.exec_()
                if self._bl_thread is not None and self._bl_thread.isRunning():
                    QMessageBox.warning(
                        self, '提示',
                        '终止超时（子进程未退出），窗口保持打开。')
                    return
        super().reject()

    # ---- 固定率警告 / CSV 导出 ----

    def _refresh_bl_warning(self, rows):
        """固定率不达标时页面级标红（报告文本之外的醒目提示）"""
        n = len(rows)
        nf = sum(1 for r in rows if r['q'] == 1)
        if n == 0:
            txt = '⚠ 无有效历元'
        elif nf == 0:
            txt = f'⚠ 无固定解（0/{n}），仅参考、不可用于精度判定'
        elif nf < 0.9 * n:
            txt = (f'⚠ 固定率 {100.0 * nf / n:.1f}%（{nf}/{n}）< 90%，'
                   f'不可用于毫米级判定')
        else:
            txt = ''
        self._bl_warn.setText(txt)
        self._bl_warn.setStyleSheet('color:#c0392b;')
        self._bl_warn.setVisible(bool(txt))

    def _export_bl_csv(self):
        res = self._bl_result
        if not res or not res.get('rows'):
            QMessageBox.information(self, '提示', '还没有可导出的解算结果。')
            return
        out_pos = res.get('out_pos')
        default_dir = (os.path.dirname(str(out_pos))
                       if out_pos else last_dir('figure'))
        stem = (os.path.splitext(os.path.basename(str(out_pos)))[0]
                if out_pos else 'baseline')
        path, _ = QFileDialog.getSaveFileName(
            self, '导出 ENU 明细', os.path.join(default_dir, stem + '.csv'),
            'CSV 文件 (*.csv);;所有文件 (*.*)')
        if not path:
            return
        try:
            self._write_bl_csv(path)
        except OSError as e:
            QMessageBox.critical(self, '错误', f'保存失败:\n{e}')
            return
        save_last_dir('figure', path)
        QMessageBox.information(self, '完成', f'明细已保存到:\n{path}')

    def _write_bl_csv(self, path):
        """逐历元 E/N/U 明细 -> CSV

        utf-8-sig 带 BOM：Excel 直接双击打开不乱码（报告主要消费
        场景就是进 Excel）。列与 .pos 顺序对齐，另加相对分钟。
        """
        import csv
        rows = sorted((self._bl_result or {}).get('rows') or [],
                      key=lambda r: r['t'])
        if not rows:
            raise OSError('没有可导出的解算历元')
        t0 = rows[0]['t']
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['分钟', 'GPS周', 'tow(s)', 'E(m)', 'N(m)', 'U(m)',
                        'Q(解类型)', '卫星数', 'ratio'])
            for r in rows:
                w.writerow([f"{(r['t'] - t0) / 60.0:.2f}",
                            r.get('week', ''),
                            f"{r['t']:.4f}",
                            f"{r['e']:.4f}", f"{r['n']:.4f}",
                            f"{r['u']:.4f}",
                            r['q'], r['ns'], f"{r['ratio']:.1f}"])

    # ---- ENU 时序图 ----

    # 解类型配色（RTKLIB Q 值：1 固定 / 2 浮动 / 其余为单点等）
    _Q_COLORS = {1: '#2ca02c', 2: '#ff7f0e'}
    _Q_NAMES = {1: '固定解 Q=1', 2: '浮动解 Q=2'}

    def _draw_bl_enu(self, res):
        """E/N/U 逐历元时序：按解类型着色，固定解统计均值±σ"""
        rows = sorted(res.get('rows') or [], key=lambda r: r['t'])
        if not rows:
            return
        t0 = rows[0]['t']

        # 统计样本：优先只用固定解（与报告口径一致），无固定解退回全部解
        fix = [r for r in rows if r['q'] == 1]
        used = fix if fix else rows
        kind = (f'固定解 n={len(fix)}' if fix
                else f'无固定解，全部解 n={len(rows)}（仅参考）')

        # 按 Q 分组：1/2 单独成组，其余合并为「其他解」
        q_groups = {}
        for r in rows:
            g = r['q'] if r['q'] in self._Q_COLORS else 0
            q_groups.setdefault(g, []).append(r)

        axes = ((self._bl_ax_e, 'e', 'ΔE'), (self._bl_ax_n, 'n', 'ΔN'),
                (self._bl_ax_u, 'u', 'ΔU'))
        for ax, key, name in axes:
            ax.clear()
            ax.grid(True, alpha=0.25, linestyle=':')
            for g, grp in sorted(q_groups.items()):
                gx = [(r['t'] - t0) / 60.0 for r in grp]
                gy = [r[key] * 1000.0 for r in grp]     # m -> mm
                color = self._Q_COLORS.get(g, '#d9534f')
                label = self._Q_NAMES.get(g, '其他解')
                ax.plot(gx, gy, '.', ms=2.2, color=color, label=label,
                        alpha=0.75)

            vals = [r[key] * 1000.0 for r in used]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ax.axhline(mean, color='#185FA5', lw=1.0, ls='--',
                       label=f'均值 {mean:+.1f} mm')
            ax.set_ylabel(f'{name} (mm)', fontsize=9)
            ax.set_title(f'{name} 逐历元基线分量（{kind}）　'
                         f'mean {mean:+.2f} ± {std:.2f} mm',
                         fontsize=10, loc='left')

        self._bl_ax_e.legend(fontsize=8, loc='best', ncol=3, framealpha=0.85)
        self._bl_ax_u.set_xlabel('开始解算后的时间 (min)', fontsize=9)
        self._bl_fig.subplots_adjust(left=0.08, right=0.985, top=0.955,
                                     bottom=0.08, hspace=0.40)
        try:
            self._bl_canvas.draw_idle()
        except Exception:
            pass

    def _build_tables(self):
        """构建详细数据表格页（逐卫星指标 + 问题清单）"""
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(2, 2, 2, 2)

        # --- 问题清单 ---
        warns = self._res.get('warnings') or []
        if warns:
            lbl = QLabel(f"自动检测到 {len(warns)} 项问题（按严重程度排序）")
            lbl.setStyleSheet("font-weight: bold;")
            v.addWidget(lbl)
            wt = QTableWidget(len(warns), 3)
            wt.setHorizontalHeaderLabels(['等级', '问题', '说明'])
            wt.setAlternatingRowColors(True)
            wt.setEditTriggers(QAbstractItemView.NoEditTriggers)
            wt.setSelectionBehavior(QAbstractItemView.SelectRows)
            wt.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            wt.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
            wt.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
            wt.setColumnWidth(1, 240)
            wt.verticalHeader().setDefaultSectionSize(26)
            for r, w in enumerate(warns):
                it0 = QTableWidgetItem(w['level'])
                it0.setForeground(QColor(LEVEL_COLORS.get(w['level'], '#333333')))
                it0.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
                wt.setItem(r, 0, it0)
                wt.setItem(r, 1, QTableWidgetItem(w['title']))
                wt.setItem(r, 2, QTableWidgetItem(w['detail']))
            wt.setMaximumHeight(220)
            v.addWidget(wt)

        # --- 逐卫星明细 ---
        lbl2 = QLabel("逐卫星明细（按完整率升序；C/N0 残差越负说明信号越差）")
        lbl2.setStyleSheet("font-weight: bold;")
        v.addWidget(lbl2)

        cont = {r['sat']: r for r in (self._res.get('continuity') or {}).get('rows', [])}
        snr = {r['sat']: r for r in (self._res.get('snr_band') or {}).get('sats', [])}
        resid = {r['sat']: r for r in (self._res.get('residual') or {}).get('sats', [])}
        geo = {r['sat']: r for r in (self._res.get('geometry') or {}).get('per_sat', [])}
        mp = {r['sat']: r for r in (self._res.get('multipath') or {}).get('rows', [])}
        # 被剔除的卫星集合（用于在多路径列标注「未计算」）
        excl = set()
        mpres = self._res.get('multipath') or {}
        for lbl in (mpres.get('excluded_sats') or []):
            for s in cont:
                if cont[s]['label'] == lbl:
                    excl.add(s)

        sats = sorted(cont.keys(), key=lambda s: cont[s]['rate'])
        cols = ['卫星', '星座', '观测历元', '完整率%', '最长中断', '中断次数',
                '平均仰角°', 'C/N0均值', '<30%', '残差dB', 'MP-RMS(m)', '判定']
        t2 = QTableWidget(len(sats), len(cols))
        t2.setHorizontalHeaderLabels(cols)
        t2.setAlternatingRowColors(True)
        t2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t2.setSelectionBehavior(QAbstractItemView.SelectRows)
        hdr = t2.horizontalHeader()
        for i in range(len(cols)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(len(cols) - 1, QHeaderView.Stretch)
        t2.verticalHeader().setDefaultSectionSize(24)

        for r, sat in enumerate(sats):
            c = cont[sat]
            n = snr.get(sat) or {}
            rs = resid.get(sat) or {}
            g = geo.get(sat) or {}
            m = mp.get(sat) or {}
            if sat in excl:
                mp_txt = '未计算'
            elif m.get('rms') is not None:
                mp_txt = _f(m['rms'], 3)
            else:
                mp_txt = '—'
            vals = [
                c['label'], c['sys'], str(c['n_epochs']), _f(c['rate'], 2),
                str(c['max_gap']), str(c['n_gaps']),
                _f(g.get('el_mean'), 1), _f(n.get('mean'), 2),
                _f(n.get('lt30'), 1), _f(rs.get('residual'), 2),
                mp_txt,
                c['verdict'],
            ]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                # 完整率偏低 / 残差显著为负 的行标红提示
                if col == 3 and c['rate'] < 95:
                    item.setForeground(QColor('#d9534f'))
                if col == 9 and rs.get('residual') is not None and rs['residual'] < -2.5:
                    item.setForeground(QColor('#d9534f'))
                # 多路径偏大标橙
                if col == 10 and m.get('rms') is not None and m['rms'] > 1.5:
                    item.setForeground(QColor('#c77e00'))
                t2.setItem(r, col, item)
        v.addWidget(t2, stretch=1)
        return page

    # ==================== 绘图 ====================

    def _draw_all(self):
        self._draw_snr_hist()
        self._draw_baseline()
        self._draw_residual()
        self._draw_continuity()
        self._draw_skyplot()
        self._draw_summary()
        self._fig.tight_layout(pad=1.8, h_pad=3.2, w_pad=2.4)
        self._canvas.draw_idle()

    def _draw_snr_hist(self):
        """C/N0 直方图 + 质量分档线"""
        ax = self._ax_hist
        ax.clear()
        ov = (self._res.get('snr_band') or {}).get('overall')
        if not ov:
            ax.text(0.5, 0.5, '无 S（信噪比）观测值', ha='center', va='center')
            ax.set_title('C/N0 分布', fontsize=11)
            return

        # 用各频点样本近似整体分布：优先取样本最多的频点
        bands = (self._res.get('snr_band') or {}).get('bands', [])
        src = max(bands, key=lambda b: b['n']) if bands else None
        if src:
            # 频点级统计只有汇总值，重绘直方图需原始样本；此处退化为
            # 用 min/p5/p50/p95/max 勾勒分布轮廓（比空图直观）
            xs = [src['min'], src['p5'], src['p50'], src['p95'], src['max']]
            ys = [0.06, 0.35, 1.0, 0.35, 0.06]
            ax.fill_between(xs, ys, color='#4a90d9', alpha=0.35)
            ax.plot(xs, ys, color='#4a90d9', linewidth=2)
            ax.set_title(f"C/N0 分布轮廓（{src['label']}，n={src['n']}）", fontsize=11)
        else:
            ax.set_title('C/N0 分布', fontsize=11)

        ax.axvline(SNR_POOR, color='#d9534f', ls='--', lw=1.2,
                   label=f'差/良分界 {SNR_POOR:.0f}')
        ax.axvline(SNR_GOOD, color='#5cb85c', ls='--', lw=1.2,
                   label=f'良/优分界 {SNR_GOOD:.0f}')
        ax.set_xlabel('C/N0 (dB-Hz)')
        ax.set_ylabel('相对频次')
        ax.set_xlim(15, 58)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    def _draw_baseline(self):
        """C/N0 随仰角变化的基线"""
        ax = self._ax_base
        ax.clear()
        r = self._res.get('residual') or {}
        if not r.get('available'):
            ax.text(0.5, 0.5, '需要广播星历\n才能计算仰角基线',
                    ha='center', va='center', fontsize=10)
            ax.set_title('C/N0 - 仰角基线', fontsize=11)
            return

        rows = r.get('baseline') or []
        if not rows:
            ax.text(0.5, 0.5, '基线样本不足', ha='center', va='center')
            return

        bands = [x['band'] for x in rows]
        means = [x['mean'] for x in rows]
        ns = [x['n'] for x in rows]
        ax.plot(bands, means, 'o-', color='#1f77b4', linewidth=2, markersize=6)
        for b, m, n in zip(bands, means, ns):
            ax.annotate(f"{m:.1f}", (b, m), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=7)
        ax.set_xlabel('仰角 (°)')
        ax.set_ylabel('C/N0 均值 (dB-Hz)')
        ax.set_title('C/N0 - 仰角基线（全站基准）', fontsize=11)
        ax.grid(True, alpha=0.3)

    def _draw_residual(self):
        """逐卫星 C/N0 残差（实测 - 同仰角期望）"""
        ax = self._ax_resid
        ax.clear()
        r = self._res.get('residual') or {}
        if not r.get('available'):
            ax.text(0.5, 0.5, '需要广播星历', ha='center', va='center')
            ax.set_title('逐卫星 C/N0 残差', fontsize=11)
            return

        rows = r.get('sats') or []
        if not rows:
            ax.text(0.5, 0.5, '样本不足', ha='center', va='center')
            return

        rows = rows[:15]           # 残差最小的 15 颗
        labels = [x['label'] for x in rows][::-1]
        vals = [x['residual'] for x in rows][::-1]
        colors = ['#d9534f' if v < -4 else ('#f0ad4e' if v < -2.5 else '#5cb85c')
                  for v in vals]

        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, alpha=0.88, edgecolor='white')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.axvline(0, color='#333333', lw=1)
        ax.axvline(-2.5, color='#f0ad4e', ls='--', lw=1, alpha=0.8)
        ax.axvline(-4.0, color='#d9534f', ls='--', lw=1, alpha=0.8)
        ax.set_xlabel('残差 (dB)  ← 越负信号越差')
        ax.set_title('逐卫星 C/N0 残差（剔除仰角影响）', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')

    def _draw_continuity(self):
        """连续性：完整率最低的若干卫星 + 中断归因配色"""
        ax = self._ax_cont
        ax.clear()
        rows = (self._res.get('continuity') or {}).get('rows') or []
        if not rows:
            ax.text(0.5, 0.5, '无连续性数据', ha='center', va='center')
            return

        worst = rows[:15][::-1]     # rows 已按完整率升序
        labels = [x['label'] for x in worst]
        rates = [x['rate'] for x in worst]
        colors = []
        for x in worst:
            if x['kind'] == 'loss_of_lock':
                colors.append('#d9534f')
            elif x['kind'] in ('rise_cut', 'set_cut', 'obscured'):
                colors.append('#f0ad4e')
            else:
                colors.append('#5cb85c')

        y = np.arange(len(labels))
        ax.barh(y, rates, color=colors, alpha=0.88, edgecolor='white')
        ax.axvline(99, color='#333333', ls='--', lw=1, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel('完整率 (%)')
        ax.set_xlim(0, 105)
        ax.set_title('完整率最低的卫星（红=真实失锁）', fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')

    def _draw_skyplot(self):
        """卫星天顶图：位置=平均方位/仰角，颜色=平均 C/N0"""
        ax = self._ax_sky
        ax.clear()
        geo = self._res.get('geometry') or {}
        if not geo.get('available'):
            ax.set_title('卫星天顶图（需要星历）', fontsize=10)
            return

        per_sat = geo.get('per_sat') or []
        snr_map = {r['sat']: r['mean']
                   for r in (self._res.get('snr_band') or {}).get('sats', [])}
        pts = [s for s in per_sat if s['sat'] in snr_map]
        if not pts:
            ax.set_title('卫星天顶图（无匹配数据）', fontsize=10)
            return

        theta = np.radians([90 - s['az_mean'] for s in pts])   # 北在顶部
        r = [90 - s['el_mean'] for s in pts]
        c = [snr_map[s['sat']] for s in pts]

        sc = ax.scatter(theta, r, c=c, cmap='viridis', s=90,
                        edgecolors='black', linewidths=0.7, zorder=3)
        for s, th, rr in zip(pts, theta, r):
            ax.annotate(s['sat'][1:], (th, rr), xytext=(4, 4),
                        textcoords='offset points', fontsize=6.5,
                        bbox=dict(boxstyle='round,pad=0.10', fc='white',
                                  alpha=0.72, ec='none'))

        ax.set_ylim(0, 90)
        ax.set_yticks([30, 60, 90])
        ax.set_yticklabels(['60°', '30°', '0°'], fontsize=8)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(['东', '北', '西', '南'], fontsize=9)
        ax.set_title('卫星天顶图（按平均 C/N0 配色）', fontsize=11)
        self._fig.colorbar(sc, ax=ax, label='C/N0 (dB-Hz)', pad=0.12, shrink=0.75)

    def _draw_summary(self):
        """右下文本摘要"""
        ax = self._ax_sum
        ax.clear()
        ax.axis('off')

        info = self._res['info']
        ov = (self._res.get('snr_band') or {}).get('overall') or {}
        cont = self._res.get('continuity') or {}
        mp = self._res.get('multipath') or {}
        geo = self._res.get('geometry') or {}
        warns = self._res.get('warnings') or []

        lines = [
            "━━━ 文件概况 ━━━",
            f"  文件: {self._file_basename}",
            f"  MARKER: {info.get('marker') or '—'}   RINEX {self._res['info'].get('version')}",
            f"  时段: {info['t_first_str']}",
            f"        ~ {info['t_last_str']}",
            f"  历元: {info['n_epochs']}   记录: {info['n_records']}",
            f"  间隔: 声明{info.get('declared_interval')}s / 实测{info.get('modal_interval')}s",
            f"  卫星: {info['n_sats']} 颗   平均{info['avg_sats_per_epoch']:.1f}颗/历元",
            "",
            "━━━ 信号与几何 ━━━",
        ]
        if ov:
            lines += [
                f"  C/N0: 均{_f(ov['mean'], 1)} 中位{_f(ov['median'], 0)} "
                f"({_f(ov['min'], 0)}~{_f(ov['max'], 0)})",
                f"  分档: <25={_f(ov['lt25'], 1)}%  <30={_f(ov['lt30'], 1)}%  "
                f"<35={_f(ov['lt35'], 1)}%  ≥40={_f(ov['ge40'], 1)}%",
            ]
        else:
            lines.append("  无 S 观测值")
        if geo.get('available'):
            lines.append(f"  仰角: <15°={_f(geo['lt15'], 2)}%  "
                         f"≥30°={_f(geo['ge30'], 1)}%")
        lines += [
            "",
            "━━━ 连续性与多路径 ━━━",
            f"  平均完整率: {_f(cont.get('rate_mean'), 2)}%",
            f"  ≥99% 的卫星: {cont.get('n_ge99')}/{cont.get('n_total')} 颗",
            f"  真实失锁卫星: {len(cont.get('loss_of_lock') or [])} 颗",
        ]
        if mp.get('available'):
            lines.append(f"  多路径 MP-RMS: 均{_f(mp['mean_rms'], 3)} m")
            rows = mp.get('rows') or []
            if rows:
                lines.append(f"    最差 {rows[0]['label']}: {_f(rows[0]['rms'], 3)} m")
            if mp.get('excluded_sats'):
                lines.append(f"    剔除 {len(mp['excluded_sats'])} 颗（L2 非真双频）")
        elif mp.get('single_frequency'):
            lines.append("  多路径: 无 L1/L2 真双频，无法计算")
        else:
            lines.append("  多路径: 缺 GPS 双频观测值")

        lines.append("")
        lines.append(f"━━━ 问题清单 ({len(warns)}) ━━━")
        if warns:
            for w in warns[:6]:
                lines.append(f"  [{w['level']}] {w['title']}")
            if len(warns) > 6:
                lines.append(f"  ... 另有 {len(warns) - 6} 项，见「详细数据」页")
        else:
            lines.append("  未发现明显问题")

        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
                va='top', ha='left', fontsize=8.2,
                bbox=dict(boxstyle='round,pad=0.5', fc='#f8f8f8', ec='#cccccc'))

    # ==================== 观测诊断页 ====================

    def _build_diagnostics(self):
        """构建「观测诊断」页（四格）

            [0] DOP / 可见卫星数 时序   —— 几何强度，对齐 RTKPLOT 的 DOP/NSat
            [1] 原始观测量时序          —— 伪距/载波/多普勒，可切频点，标出周跳
            [2] C/N0 – 仰角 散点        —— 对齐 RTKPLOT 的 SNR-EL
            [3] MP   – 仰角 散点        —— 对齐 RTKPLOT 的 MP-EL
        """
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(2, 2, 2, 2)

        res = self._res
        dop = res.get('dop_nsat') or {}
        series = res.get('obs_series') or {}
        slips = res.get('cycle_slips') or {}

        self._dx_fig = None
        self._dx_ax_dop2 = None
        if not (dop.get('available') or series):
            v.addWidget(QLabel(
                "没有可用于观测诊断的数据。\n"
                "本页需要观测文件；其中 DOP、卫星数、仰角散点还需要广播星历。"))
            return page

        # ---- 顶部控制条 ----
        bar = QHBoxLayout()

        # 星历覆盖范围提示：只有 GPS 星历时 DOP/仰角只能算 GPS，
        # 不说清楚用户会把结果误当成全星座的
        eph_sys = dop.get('eph_systems') or []
        if dop.get('available') and eph_sys:
            allsys = list((res.get('info') or {}).get('sys_sat_count') or {})
            miss = [s for s in allsys if s not in eph_sys]
            if miss:
                tip = QLabel(f"⚠ 星历仅覆盖 {'、'.join(eph_sys)}："
                             f"DOP/卫星数/仰角散点不含 {'、'.join(miss)}")
                tip.setStyleSheet("color: #c77e00; font-weight: bold;")
                bar.addWidget(tip)
                bar.addStretch(1)

        self._dx_quant = self._dx_band = self._dx_mean0 = None
        if series:
            bar.addWidget(QLabel("观测量:"))
            self._dx_quant = QComboBox()
            self._dx_quant.addItem("伪距 C/P", 'p')
            self._dx_quant.addItem("载波 L", 'l')
            self._dx_quant.addItem("多普勒 D", 'd')
            bar.addWidget(self._dx_quant)

            bar.addWidget(QLabel("频点:"))
            self._dx_band = QComboBox()
            for k in sorted(series):
                self._dx_band.addItem(f"{k[0]}-{k[1]}", k)
            bar.addWidget(self._dx_band)

            self._dx_mean0 = QCheckBox("以首值归零")
            # 默认**不**归零：让用户先看到观测值的真实量级与形态，
            # 需要横向比较各卫星曲线时再自己勾选。
            # 之前默认勾选，第一次打开看到的是"从 0 开始的相对曲线"，
            # 容易误以为那就是原始观测值。
            self._dx_mean0.setChecked(False)
            self._dx_mean0.setToolTip(
                "不勾选：显示观测值的绝对量（看真实量级、整体形态与跳变）。\n"
                "勾选：各卫星减去自己的首个值，曲线以 0 为基准，\n"
                "便于看变化过程（升降/周跳/异常跳变），截取短时段或\n"
                "只看某颗卫星时尤其有用。\n"
                "注意：整段观测里单颗卫星的距离变化可达 ±5000 km，\n"
                "所以归零不一定让纵轴变小；查看多普勒时本项自动失效\n"
                "（其绝对值本身就是信息）。")
            bar.addWidget(self._dx_mean0)

        self._dx_lbl = QLabel()
        self._dx_lbl.setFont(QFont("Microsoft YaHei", 9))
        if slips.get('available'):
            n_raw = sum(r.get('n_raw', 0) for r in (slips.get('rows') or []))
            self._dx_lbl.setText(
                f"　载波跳变 {n_raw} 次 → 整周周跳 {slips.get('total', 0)} 次 · "
                f"丢锁 {slips.get('total_loss', 0)} 次"
                f"（GF 判据，{slips.get('n_sats_checked', 0)} 颗卫星）")
        bar.addWidget(self._dx_lbl)
        bar.addStretch(1)
        v.addLayout(bar)

        fig = self._dx_fig = Figure(figsize=(15, 11), dpi=100)
        gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.05, 0.95],
                              hspace=0.45, wspace=0.22)
        self._dx_ax_dop = fig.add_subplot(gs[0, :])
        self._dx_ax_obs = fig.add_subplot(gs[1, :])
        self._dx_ax_snr = fig.add_subplot(gs[2, 0])
        self._dx_ax_mp = fig.add_subplot(gs[2, 1])

        canvas = FigureCanvas(fig)
        v.addWidget(canvas, stretch=1)
        v.addWidget(ChineseNavToolbar(canvas, self))

        for w in (self._dx_quant, self._dx_band):
            if w is not None:
                w.currentIndexChanged.connect(self._draw_diagnostics)
        if self._dx_mean0 is not None:
            self._dx_mean0.stateChanged.connect(self._draw_diagnostics)

        self._draw_diagnostics()
        return page

    # （SPP 单点定位的界面显示已按需求移除，解算代码见 spp.py；
    #  恢复显示可参考 git 历史中的 _build_height。）

    def _draw_diagnostics(self):
        """绘制观测诊断页（切换观测量/频点/归零时重绘）"""
        if getattr(self, '_dx_fig', None) is None:
            return
        res = self._res
        dop = res.get('dop_nsat') or {}
        series = res.get('obs_series') or {}
        sc = res.get('elev_scatter') or {}
        slips = res.get('cycle_slips') or {}

        # twinx 生成的是独立 Axes，clear() 不会清掉它，重复重绘会越叠越多
        if self._dx_ax_dop2 is not None:
            self._dx_ax_dop2.remove()
            self._dx_ax_dop2 = None

        for ax in (self._dx_ax_dop, self._dx_ax_obs,
                   self._dx_ax_snr, self._dx_ax_mp):
            ax.clear()
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

        # ---------- [0] DOP / 卫星数 ----------
        ax = self._dx_ax_dop
        if dop.get('available'):
            t = dop['times']
            for key, color, lbl in (('pdop', '#d62728', 'PDOP'),
                                    ('gdop', '#9467bd', 'GDOP'),
                                    ('hdop', '#1f77b4', 'HDOP'),
                                    ('vdop', '#2ca02c', 'VDOP')):
                ax.plot(t, dop[key], lw=0.9, color=color, label=lbl)
            ax.set_xlabel('观测开始后的分钟数', fontsize=9)
            ax.set_ylabel('DOP', fontsize=9)

            ax2 = self._dx_ax_dop2 = ax.twinx()
            ax2.plot(t, dop['nsat_used'], lw=1.5, color='#ff7f0e',
                     alpha=0.8, label='参与解算卫星数')
            ax2.set_ylabel('参与解算卫星数', fontsize=9, color='#c77e00')
            ax2.tick_params(axis='y', labelcolor='#c77e00')
            ax2.set_ylim(bottom=0)

            # 卫星数画在右轴，matplotlib 不会自动并进左图图例，手工补一个
            h, lb = ax.get_legend_handles_labels()
            h.append(Line2D([], [], color='#ff7f0e', lw=1.5))
            lb.append('参与解算卫星数(右轴)')
            ax.legend(h, lb, fontsize=8, loc='upper left', ncol=5,
                      framealpha=0.85)

            st = dop['stats']
            sys_txt = '、'.join(dop.get('used_systems') or []) or '—'
            ax.set_title(
                f"DOP / 可见卫星数（仰角掩蔽 {dop['mask']:.0f}°，"
                f"星历仅覆盖 {sys_txt}）　"
                f"PDOP 均值 {st['pdop']['mean']:.2f} / 最大 {st['pdop']['max']:.2f}，"
                f"卫星数 {st['nsat_used']['min']:.0f} ~ {st['nsat_used']['max']:.0f} 颗",
                fontsize=10)
        else:
            ax.set_title(f"DOP / 卫星数不可用：{dop.get('reason', '')}",
                         fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # ---------- [1] 原始观测量 ----------
        ax = self._dx_ax_obs
        quant = self._dx_quant.currentData() if self._dx_quant else 'p'
        key = self._dx_band.currentData() if self._dx_band else None
        sats = (series.get(key) or {}) if key else {}
        if not sats:
            ax.set_title('原始观测量：无数据', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            qi = {'p': 1, 'l': 2, 'd': 3}[quant]
            unit = {'p': 'm', 'l': '周', 'd': 'Hz'}[quant]
            qname = {'p': '伪距', 'l': '载波', 'd': '多普勒'}[quant]
            # 归零只对伪距/载波有意义（多普勒的绝对值本身是信息）
            mean0 = bool(self._dx_mean0 is not None
                         and self._dx_mean0.isChecked() and quant in ('p', 'l'))
            slip_by_sat = {r['sat']: r for r in (slips.get('rows') or [])}

            n_show = n_mark = 0
            for satid in sorted(sats):
                ts, ps, ls, ds = sats[satid]
                arr = np.array((ps, ls, ds)[qi - 1], dtype=float)
                good = ~np.isnan(arr)
                if not good.any():
                    continue
                tg, yg = np.array(ts)[good], arr[good]
                if mean0:
                    yg = yg - yg[0]
                ax.plot(tg, yg, lw=0.8, color=self._sat_color(satid),
                        label=satid, alpha=0.9)
                n_show += 1

                r = slip_by_sat.get(satid)
                if r and r.get('events') and quant == 'l':
                    for e in r['events']:
                        ax.plot([e['t']], [float(np.interp(e['t'], tg, yg))],
                                marker='o', ms=4.2, ls='none', mec='white',
                                mew=0.5, zorder=6,
                                mfc='#d62728' if e['kind'] == 'loss'
                                else '#ff7f0e')
                        n_mark += 1

            ax.set_xlabel('观测开始后的分钟数', fontsize=9)
            ax.set_ylabel(f'{qname} ({unit})', fontsize=9)
            if n_show <= 16:
                ax.legend(fontsize=7, loc='best', ncol=4, framealpha=0.8)
            ax.set_title(
                f"原始观测量 — {key[0]}-{key[1]} {qname}"
                f"{'（已按首值归零）' if mean0 else ''}　"
                f"{n_show} 颗卫星，标记 {n_mark} 个周跳/丢锁",
                fontsize=10)

        # ---------- [2] C/N0 – 仰角 散点 ----------
        ax = self._dx_ax_snr
        snr_sc = (sc.get('snr') or {}).get(key) or {}
        empty_msg = ('该频点没有可配对的仰角数据。\n'
                     '（仰角需要对应系统的广播星历）'
                     if (key and sc.get('snr')) else None)
        self._scatter_by_elevation(ax, snr_sc, key, 'C/N0 (dB-Hz)',
                                   'C/N0 – 仰角', empty_msg=empty_msg)

        # ---------- [3] MP – 仰角 散点 ----------
        # 注意：MP 只用 GPS 的 L1/L2 无电离层组合，与上面选的频点无关，
        # 所以标题必须写死说明来源，不能跟着频点下拉框走，否则会误标。
        ax = self._dx_ax_mp
        self._scatter_by_elevation(
            ax, sc.get('mp') or {}, None, '|多路径 MP| (m)',
            '多路径 – 仰角（GPS L1/L2 组合）', ylim_top=8.0, abs_y=True)

        # 必须显式请求重绘：matplotlib 的 Qt 画布不会因为 Axes 被改动而
        # 自动刷新，少了这一句的表现就是「下拉框选了没反应」——
        # 数据其实已经重算，只是画面停在旧图上。
        self._dx_fig.canvas.draw_idle()

    def _scatter_by_elevation(self, ax, sat_pts, key, ylabel, title,
                              ylim_top=None, abs_y=False, empty_msg=None):
        """把 {卫星: [(仰角, 值)]} 画成散点 + 5° 分箱中位线

        abs_y=True 时取绝对值：多路径 MP 是围绕 0 的带符号量，
        直接画会一半落在负半轴，而纵轴从 0 起，负值全被裁掉。
        多路径的「大小」才是要看的，故取幅值。
        """
        pts_all = {s: v for s, v in (sat_pts or {}).items() if v}
        if not pts_all:
            ax.set_title(f'{title}：{empty_msg or "无数据（需要星历）"}',
                         fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        xs_all, ys_all = [], []
        for satid in sorted(pts_all):
            arr = np.array(pts_all[satid], dtype=float)
            xs, ys = arr[:, 0], np.abs(arr[:, 1]) if abs_y else arr[:, 1]
            ax.plot(xs, ys, ls='none', marker='.', ms=2.4, alpha=0.5,
                    color=self._sat_color(satid))
            xs_all.append(xs)
            ys_all.append(ys)

        X = np.concatenate(xs_all)
        Y = np.concatenate(ys_all)
        edges = np.arange(0.0, 95.0, 5.0)
        cx, cy = [], []
        for i in range(len(edges) - 1):
            m = (X >= edges[i]) & (X < edges[i + 1])
            if m.sum() >= 10:
                cx.append((edges[i] + edges[i + 1]) / 2.0)
                cy.append(float(np.median(Y[m])))
        if cx:
            ax.plot(cx, cy, lw=2.0, color='#333333', marker='o', ms=3.5,
                    alpha=0.85, label='5° 分箱中位')

        if ylim_top is not None:
            # 低仰角离群点会把其余点压扁，按分位裁剪并注明裁掉了多少
            top = min(max(ylim_top, 1.0),
                      max(float(np.percentile(Y, 99.5)), 1.0) * 1.1)
            over = int((Y > top).sum())
            ax.set_ylim(0, top)
            if over:
                ax.text(0.98, 0.96, f'⚠ {over} 点超过 {top:.1f} m 未显示',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, color='#c77e00')

        ax.set_xlim(0, 90)
        ax.set_xlabel('仰角 (°)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        band_txt = f'（{key[0]}-{key[1]}）' if key else ''
        ax.set_title(f'{title} {band_txt}　{len(pts_all)} 颗卫星，'
                     f'{len(X)} 个点', fontsize=10)
        if cx:
            ax.legend(fontsize=8, loc='lower right', framealpha=0.85)

    # ==================== 保存 / 导出 ====================

    def _save_figure(self):
        # 保存当前所在标签页的图，否则用户会拿到一张不是眼前的图。
        # 按标签标题取，不依赖固定索引 —— 以后增删页签都不会错位。
        title = self._tabs.tabText(self._tabs.currentIndex())
        mapping = {
            '质量图表': (getattr(self, '_fig', None), 'quality'),
            '时间序列': (getattr(self, '_ts_fig', None), 'timeseries'),
            '观测诊断': (getattr(self, '_dx_fig', None), 'diagnostics'),
            '相对定位': (getattr(self, '_bl_fig', None), 'baseline'),
        }
        fig, tag = mapping.get(title, (None, None))
        if fig is None:
            fig, tag = getattr(self, '_fig', None), 'quality'
        if fig is None:
            QMessageBox.information(self, "提示", "当前页面没有可保存的图。")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存图片",
            os.path.join(
                last_dir('figure'),
                f"static_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
            "PNG图片 (*.png);;PDF文档 (*.pdf);;SVG矢量图 (*.svg)"
        )
        if filepath:
            save_last_dir('figure', filepath)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "完成", f"图片已保存到:\n{filepath}")

    def _export_report(self):
        """把结果导出为纯文本报告（可直接交付/归档）"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出分析报告",
            os.path.join(
                last_dir('static_report'),
                f"static_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"),
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.build_report_text())
            save_last_dir('static_report', filepath)
            QMessageBox.information(self, "完成", f"报告已保存到:\n{filepath}")
        except OSError as e:
            QMessageBox.critical(self, "错误", f"保存失败:\n{e}")

    def build_report_text(self):
        """生成纯文本报告内容"""
        res = self._res
        info = res['info']
        out = []
        add = out.append

        add("=" * 78)
        add("静态观测数据质量分析报告")
        add("=" * 78)
        add(f"  文件      : {self._file_basename}")
        if self._nav_basename:
            add(f"  星历      : {self._nav_basename}")
        add(f"  MARKER    : {info.get('marker') or '—'}")
        add(f"  RINEX 版本: {info.get('version')}")
        add(f"  近似坐标  : {info.get('xyz')}")
        ant = info.get('ant_delta') or (0, 0, 0)
        add(f"  天线高 H  : {ant[0]} m")
        add(f"  时段(UTC) : {info['t_first_str']} ~ {info['t_last_str']}")
        add(f"  历元数    : {info['n_epochs']}   记录总数: {info['n_records']}")
        add(f"  采样间隔  : 声明 {info.get('declared_interval')} s / "
            f"实测众数 {info.get('modal_interval')} s")
        add(f"  卫星总数  : {info['n_sats']} 颗   平均 {info['avg_sats_per_epoch']:.2f} 颗/历元")
        add(f"  单历元卫星: {info['min_sats_per_epoch']} ~ {info['max_sats_per_epoch']} 颗")
        add("")
        add("  ── 系统构成 ──")
        for k, v in sorted(info['sys_share'].items(), key=lambda x: -x[1]):
            add(f"    {k:<10} {info['sys_sat_count'].get(k, 0):>3} 颗卫星  "
                f"{info['sys_rec_count'].get(k, 0):>8} 条记录  {v:>6.2f}%")

        # 观测值缺失
        add("")
        add("  ── 观测值缺失/零值统计 ──")
        add(f"    {'系统':<10}{'观测码':<8}{'有效':>10}{'零值':>10}{'缺失率':>10}")
        for r in (res.get('missing') or {}).get('rows', []):
            add(f"    {r['sys']:<10}{r['code']:<8}{r['total']:>10}"
                f"{r['zero']:>10}{r['rate']:>9.2f}%")

        # 信噪比
        add("")
        add("  ── 信噪比 C/N0 (dB-Hz) ──")
        add(f"    {'系统-频点':<14}{'样本':>9}{'最小':>7}{'P5':>7}{'中位':>7}"
            f"{'P95':>7}{'最大':>7}{'均值':>8}")
        for b in (res.get('snr_band') or {}).get('bands', []):
            add(f"    {b['label']:<14}{b['n']:>9}{b['min']:>7.1f}{b['p5']:>7.0f}"
                f"{b['p50']:>7.0f}{b['p95']:>7.0f}{b['max']:>7.1f}{b['mean']:>8.2f}")
        ov = (res.get('snr_band') or {}).get('overall')
        if ov:
            add(f"    合并: 均值={ov['mean']:.2f} 中位={ov['median']:.0f}  "
                f"<30={ov['lt30']:.2f}%  ≥40={ov['ge40']:.2f}%  "
                f"量化上限={ov['ceiling']:.0f}(顶格{ov['at_ceiling']:.1f}%)")

        # 几何
        geo = res.get('geometry') or {}
        if geo.get('available'):
            add("")
            add("  ── 仰角分布 ──")
            total = geo['total']
            for k in sorted(geo['elev_hist'], key=lambda x: int(x)):
                v = geo['elev_hist'][k]
                add(f"    {int(k):3d}~{int(k) + 5:3d}° : {v:>7} "
                    f"{v / total * 100:>6.2f}%  {'#' * int(v / total * 90)}")
            add(f"    <15°={geo['lt15']:.2f}%  ≥30°={geo['ge30']:.2f}%")

            add("")
            add("  ── 低仰角问题卫星（平均仰角最低 12 颗）──")
            add(f"    {'卫星':<14}{'样本':>7}{'最低':>8}{'最高':>8}{'平均':>8}{'<15°占比':>10}")
            for s in (geo.get('per_sat') or [])[:12]:
                add(f"    {s['label']:<14}{s['n']:>7}{s['el_min']:>8.1f}"
                    f"{s['el_max']:>8.1f}{s['el_mean']:>8.1f}{s['lt15']:>9.1f}%")

        # 残差
        rd = res.get('residual') or {}
        if rd.get('available'):
            add("")
            add("  ── C/N0 仰角基线 ──")
            for b in rd.get('baseline', []):
                add(f"    {b['band']:>3}-{b['band'] + 5:>3}°  n={b['n']:>7}  "
                    f"均值 {b['mean']:.2f} dB-Hz")
            add("")
            add("  ── 卫星 C/N0 残差（负值 = 信号弱于同仰角基线）──")
            add(f"    {'卫星':<14}{'样本':>7}{'平均仰角':>10}{'实测':>9}"
                f"{'期望':>9}{'残差':>8}")
            for s in (rd.get('sats') or [])[:20]:
                flag = '  <== 明显偏低' if s['residual'] < -4 else (
                    '  <-- 偏低' if s['residual'] < -2.5 else '')
                add(f"    {s['label']:<14}{s['n']:>7}{s['el_mean']:>10.1f}"
                    f"{s['real']:>9.2f}{s['expect']:>9.2f}{s['residual']:>8.2f}{flag}")

        # 连续性
        cont = res.get('continuity') or {}
        add("")
        add("  ── 连续性 / 完整率（最差 20 颗）──")
        add(f"    {'卫星':<14}{'历元':>8}{'跨度':>8}{'完整率':>9}"
            f"{'中断':>6}{'最长':>8}  判定")
        for r in (cont.get('rows') or [])[:20]:
            add(f"    {r['label']:<14}{r['n_epochs']:>8}{r['span']:>8}"
                f"{r['rate']:>8.2f}%{r['n_gaps']:>6}{r['max_gap']:>8}  {r['verdict']}")
        add(f"    平均完整率={cont.get('rate_mean', 0):.2f}%  "
            f"≥99% 的 {cont.get('n_ge99')}/{cont.get('n_total')} 颗")
        add("")
        add("    按系统汇总:")
        for s, v in (cont.get('sys_summary') or {}).items():
            add(f"      {s:<10} {v['n_sats']:>3} 颗  平均完整率 "
                f"{v['rate_mean']:>6.2f}%  最低 {v['rate_min']:.2f}%")

        # 多路径
        mp = res.get('multipath') or {}
        if mp.get('available'):
            add("")
            add(f"  ── 多路径 MP-RMS (GPS, {mp['codes']}) ──")
            add(f"    {'卫星':<14}{'RMS(m)':>10}{'样本':>10}")
            for r in (mp.get('rows') or [])[:12]:
                add(f"    {r['label']:<14}{r['rms']:>10.3f}{r['n']:>10}")
            add(f"    GPS 平均 RMS = {mp['mean_rms']:.3f} m")
            if mp.get('excluded_sats'):
                add("")
                add(f"    另有 {len(mp['excluded_sats'])} 颗卫星按保守判据"
                    f"未参与多路径计算: {'、'.join(mp['excluded_sats'])}")
                add("    （判据：双频伪距差 |C1-P2| 中位数 < 0.7 m，"
                    "远低于其余卫星的 1.0 ~ 4.1 m。该判据基于伪距、"
                    "受接收机硬件偏差影响，偏保守；这几颗星的载波"
                    "几何无关组合其实是自洽的，是否真跟踪到第二频点"
                    "请结合「观测诊断」页的原始观测量一并判断。）")
        elif mp.get('single_frequency'):
            add("")
            add("  ── 多路径 ──")
            add(f"    {mp.get('reason', '无法计算')}")

        # 几何强度 DOP / 卫星数
        dn = res.get('dop_nsat') or {}
        if dn.get('available'):
            st = dn['stats']
            add("")
            add("  ── 几何强度 DOP / 卫星数 ──")
            add(f"    仰角掩蔽 {dn['mask']:.0f}°   采样 {dn['n']} 个历元   "
                f"星历覆盖 {'、'.join(dn.get('eph_systems') or [])}")
            add(f"    {'指标':<8}{'均值':>9}{'最小':>9}{'P50':>9}{'P95':>9}{'最大':>9}")
            for k, nm in (('gdop', 'GDOP'), ('pdop', 'PDOP'),
                          ('hdop', 'HDOP'), ('vdop', 'VDOP')):
                s = st[k]
                add(f"    {nm:<8}{s['mean']:>9.2f}{s['min']:>9.2f}"
                    f"{s['p50']:>9.2f}{s['p95']:>9.2f}{s['max']:>9.2f}")
            s = st['nsat_used']
            add(f"    参与解算卫星数  均值 {s['mean']:.1f}  "
                f"最少 {s['min']:.0f}  最多 {s['max']:.0f}")
            add(f"    PDOP>4 占比 {st['pdop_gt4']:.2f}%   "
                f"PDOP>6 占比 {st['pdop_gt6']:.2f}%   "
                f"卫星<5 历元占比 {st['nsat_lt5']:.2f}%")
            miss = [x for x in (info.get('sys_sat_count') or {})
                    if x not in (dn.get('eph_systems') or [])]
            if miss:
                add(f"    注意: 星历未覆盖 {'、'.join(miss)}，"
                    f"上述几何指标不含这些系统，数值偏大/偏少。")
        else:
            add("")
            add("  ── 几何强度 DOP / 卫星数 ──")
            add(f"    不可用: {dn.get('reason', '缺少星历或测站坐标')}")

        # 原始观测量 + 周跳
        sl = res.get('cycle_slips') or {}
        if sl.get('available'):
            add("")
            add("  ── 周跳 / 丢锁（几何无关组合 GF = λ₁L₁ − λ₂L₂）──")
            add(f"    判据: 相邻历元 |ΔGF| > {sl['threshold']} m，"
                f"且要求与整周相符（偏差 ≤ {sl.get('fit_tol', 0.10)} 周）")
            add(f"    幅值 > {sl.get('loss_threshold', 1.0)} m 判为丢锁 / 大跳变")
            add(f"    结果: 整周周跳 {sl['total']} 次（已排除持续性异常卫星），"
                f"丢锁/大跳变 {sl.get('total_loss', 0)} 次；"
                f"可检测 {sl['n_sats_checked']} 颗，"
                f"无法检测 {sl['n_sats_skipped']} 颗")
            add("")
            add(f"    {'卫星':<14}{'样本':>7}{'跳变':>7}{'符合整周%':>11}"
                f"{'周跳':>6}{'丢锁':>6}  说明")
            for r in (sl.get('rows') or [])[:20]:
                ir = r.get('int_ratio')
                irs = f"{ir:.0f}" if ir is not None else '—'
                if r['usable']:
                    if r.get('chronic'):
                        note = ('*** 跳变密集且符合整周比例接近随机，'
                                '判为持续性跟踪异常，周跳数不计入总数')
                    elif r['events']:
                        e0 = r['events'][0]
                        note = f"首次 t={e0['t']:.2f} min {e0['dgf']:+.3f} m"
                    else:
                        note = ''
                    add(f"    {r['label']:<14}{r['n']:>7}{r.get('n_raw', 0):>7}"
                        f"{irs:>11}{r['n_slips']:>6}{r['n_loss']:>6}  {note}")
                else:
                    add(f"    {r['label']:<14}{r['n']:>7}{r.get('n_raw', 0):>7}"
                        f"{irs:>11}{'—':>6}{'—':>6}  {r['reason']}")
            add("")
            add("    判读方法: 「符合整周%」是判别力的关键 —— 真周跳必然是 n·λ，")
            add("    该比例应显著高于随机水平（容差 0.10 周时随机约 36%）。")
            add("    若跳变次数很多而该比例接近 36%，说明不是零散周跳，")
            add("    而是接收机在一段时间里反复失锁、输出非整周相位不连续。")

            hot = [r for r in (sl.get('rows') or [])
                   if r['usable'] and r['events']]
            if hot:
                add("")
                add("    周跳 / 丢锁事件明细（每颗星最多列 12 个）:")
                for r in hot[:8]:
                    add(f"      {r['label']}:")
                    for e in r['events'][:12]:
                        if e['kind'] == 'loss':
                            add(f"        t={e['t']:8.2f} min   "
                                f"ΔGF={e['dgf']:>+10.3f} m   [丢锁/大跳变]")
                        else:
                            add(f"        t={e['t']:8.2f} min   "
                                f"ΔGF={e['dgf']:>+10.3f} m   "
                                f"约 {e['cycles']:+d} 周")

        # 问题清单
        add("")
        add("=" * 78)
        add(f"问题清单（{len(res.get('warnings') or [])} 项，按严重程度排序）")
        add("=" * 78)
        for i, w in enumerate(res.get('warnings') or [], 1):
            add(f"  {i:>2}. [{w['level']}] {w['title']}")
            add(f"      {w['detail']}")
        if not res.get('warnings'):
            add("  未发现明显问题。")

        add("")
        add("说明: C/N0 残差 = 实测均值 - 按该星自身仰角分布查基线得到的期望值，")
        add("      残差显著为负说明剔除几何因素后该星信号确实偏弱。")
        add("      完整率 = 观测历元数 / (末历元 - 首历元 + 1)；")
        add("      最长中断 > 100 历元判定为真实失锁，否则多为升落期低仰角遮挡。")
        add("      DOP 为单位权纯几何结果，与 RTKPLOT 的 DOP/NSat 图同口径，")
        add("      仅统计仰角高于掩蔽角且星历可用的卫星。")
        add("      周跳由几何无关组合 GF = λ₁L₁ - λ₂L₂ 判定：该组合抵消了几何")
        add("      距离、接收机钟差与对流层，只剩电离层与模糊度常数，")
        add("      对周跳极敏感（噪声 0.4~4 mm，1 周周跳即 0.19~0.24 m）。")
        return '\n'.join(out)
