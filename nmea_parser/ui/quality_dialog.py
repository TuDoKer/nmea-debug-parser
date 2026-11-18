# -*- coding: utf-8 -*-
"""测量环境质量报告对话框

集中展示定位质量、几何质量、信号质量、误差椭圆、报文完整性等板块。
数据来源：GGA / GSA / GST / GSV / RMC / SATSINFO。
布局：左侧6行2列共11子图 + 右侧文本摘要面板。
"""

import os
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QMessageBox, QTabWidget, QWidget, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from .. import timebase
from .. import solution as solution_mod
from ..toolbar import ChineseNavToolbar
from ..statistics import QualityStatistics, GSVStatistics
from ..config import last_dir, save_last_dir
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 星座颜色（与 satellite_dialog 一致）
CONSTELLATION_COLORS = {
    'GPS': '#1f77b4',
    '北斗': '#d62728',
    'GLONASS': '#2ca02c',
    'Galileo': '#ff7f0e',
    'QZSS': '#9467bd',
    'SBAS': '#8c564b',
    'IRNSS': '#e377c2',
}

# 解析入口的拒收原因 → 显示名（与 parser._reject_reason 的键一一对应）
REJECT_NAMES = {
    'too_short': '过短',
    'too_long': '超长',
    'no_comma': '无逗号',
    'unrepairable': '不可修复',
    'bad_checksum_format': '校验和格式异常',
    'truncated': '半句（无校验和终止符）',
    'unsupported_type': '类型不支持',
    'other': '其他',
}


class QualityReportDialog(QDialog):
    """测量环境质量报告（11子图 + 文本摘要）"""

    def __init__(self, all_nmea_messages, file_basename, parent=None,
                 parse_stats=None):
        super().__init__(parent)
        self.setWindowTitle("测量环境质量报告")
        self.resize(1500, 1050)
        self.valid = False

        # parse_stats 是 parse_file 的完整返回值；这里只取拒收计数与样本，
        # 让完整性板块能算上"被入口丢掉的那些"，并能翻到具体是哪几行
        parse_stats = parse_stats or {}
        self._stats = QualityStatistics.compute_quality(
            all_nmea_messages,
            {'rejected': parse_stats.get('rejected'),
             'rejected_samples': parse_stats.get('rejected_samples')})
        self._file_basename = file_basename

        # 至少有一个板块有数据。
        # 不能按"值是否为 None"判断：time_axis 和 rtk 永远返回 dict，
        # 空文件时也算"有数据"，警告框就永远不会弹出来。
        # 只有真正解析出报文、或入口确实拒收过句子，才算有东西可看。
        counts = self._stats.get('msg_counts') or {}
        integrity = self._stats.get('integrity') or {}
        has_data = (sum(counts.values()) > 0
                    or integrity.get('rejected_total', 0) > 0)
        if not has_data:
            QMessageBox.warning(
                self, "提示",
                "数据中没有可用于质量分析的报文"
                "（GGA/GSA/GST/GSV/RMC/SATSINFO/REFSTATION）。\n"
                "请检查采集配置后重新采集。")
            return

        self.valid = True
        self._build_ui()
        self._draw_all()

    # -------------------- UI --------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, stretch=1)

        # ---- 质量图表页 ----
        page1 = QWidget()
        v1 = QVBoxLayout(page1)
        v1.setContentsMargins(2, 2, 2, 2)

        fig = self._fig = Figure(figsize=(16, 10), dpi=100)
        # 6行3列：左2列放11子图，右1列文本摘要
        gs = fig.add_gridspec(6, 3, width_ratios=[1.1, 1.1, 0.65])
        self._ax_fix = fig.add_subplot(gs[0, 0])
        self._ax_age = fig.add_subplot(gs[0, 1])
        self._ax_dop = fig.add_subplot(gs[1, 0])
        self._ax_lock = fig.add_subplot(gs[1, 1])
        self._ax_snr = fig.add_subplot(gs[2, 0])
        self._ax_const = fig.add_subplot(gs[2, 1])
        self._ax_rms = fig.add_subplot(gs[3, 0])
        self._ax_ellipse = fig.add_subplot(gs[3, 1])
        self._ax_cpu = fig.add_subplot(gs[4, 0])
        self._ax_rmc = fig.add_subplot(gs[4, 1])
        self._ax_integrity = fig.add_subplot(gs[5, 0:2])
        self._ax_summary = fig.add_subplot(gs[:, 2])

        self._canvas = FigureCanvas(fig)
        v1.addWidget(self._canvas, stretch=1)
        v1.addWidget(ChineseNavToolbar(self._canvas, self))
        self._tabs.addTab(page1, "质量图表")

        # ---- RTK 时效页 ----
        self._tabs.addTab(self._build_rtk_page(), "RTK 时效")

        # ---- 拒收样本页 ----
        self._tabs.addTab(self._build_reject_page(), "拒收样本")

        # 按钮
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("💾 保存图片")
        btn_close = QPushButton("关闭")
        btn_layout.addWidget(btn_save)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        btn_save.clicked.connect(self._save_figure)
        btn_close.clicked.connect(self.accept)

    # -------------------- RTK 时效页 --------------------

    def _build_rtk_page(self):
        """构建「RTK 时效」页

            [0, :] 解算状态时序条带 —— 一眼看出什么时候掉了固定、掉多久
            [1, 0] 解算类型占比     —— 固定率单列，不再与单点/差分混算
            [1, 1] 分档固定率       —— 固定率随卫星数/HDOP/龄期的变化
            [2, :] 文本摘要         —— TTFF / 失锁恢复 / 固定段
        """
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(2, 2, 2, 2)

        self._rtk_fig = None
        rtk = (self._stats or {}).get('rtk') or {}
        if not rtk.get('available'):
            v.addWidget(QLabel(
                f"没有可用于 RTK 时效分析的数据。\n{rtk.get('reason', '')}"))
            return page

        fig = self._rtk_fig = Figure(figsize=(15, 10), dpi=100)
        gs = fig.add_gridspec(3, 2, height_ratios=[0.75, 1.0, 1.05],
                              hspace=0.42, wspace=0.22)
        self._ax_rtk_state = fig.add_subplot(gs[0, :])
        self._ax_rtk_share = fig.add_subplot(gs[1, 0])
        self._ax_rtk_bins = fig.add_subplot(gs[1, 1])
        self._ax_rtk_text = fig.add_subplot(gs[2, :])

        canvas = FigureCanvas(fig)
        v.addWidget(canvas, stretch=1)
        v.addWidget(ChineseNavToolbar(canvas, self))
        return page

    # -------------------- 拒收样本页 --------------------

    def _build_reject_page(self):
        """构建「拒收样本」页

        完整性板块只报"各原因多少条"，仍然不知道丢的是哪几行。
        排查干扰段最需要原文：只有看到句子在哪一列被打断，才能判断
        是帧长配错、串口丢字节，还是报文本身就不合法。
        样本数量在解析器侧按原因封顶，这里只负责列出来。
        """
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(6, 6, 6, 6)

        ig = self._stats.get('integrity') or {}
        counts = ig.get('rejected') or {}
        samples = ig.get('rejected_samples') or {}
        total = ig.get('rejected_total') or 0

        self._reject_table = None
        if not total:
            v.addWidget(QLabel("没有被入口拒收的报文。"))
            return page

        dist = '  '.join(f"{REJECT_NAMES.get(k, k)}×{n}" for k, n in
                         sorted(counts.items(), key=lambda kv: -kv[1]))
        v.addWidget(QLabel(f"入口拒收 {total} 条：{dist}"))

        rows = [(reason, it) for reason, items in samples.items()
                for it in items]
        if not rows:
            v.addWidget(QLabel("没有保留样本（仅统计了拒收条数）。"))
            return page
        rows.sort(key=lambda x: (-counts.get(x[0], 0), x[0],
                                 x[1].get('line_no') or 0))
        v.addWidget(QLabel(
            f"每种原因最多保留 {max(len(x) for x in samples.values())} 条样本，"
            "用于定位原文。"))

        table = self._reject_table = QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(
            ["行号", "原因", "原文（不可打印字符以 · 代替）"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        for r, (reason, it) in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(str(it.get('line_no'))))
            table.setItem(r, 1, QTableWidgetItem(
                f"{REJECT_NAMES.get(reason, reason)}"
                f"（共 {counts.get(reason, 0)} 条）"))
            table.setItem(r, 2, QTableWidgetItem(it.get('content', '')))
        hh = table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        v.addWidget(table, stretch=1)
        return page

    # -------------------- 绘图 --------------------

    # -------------------- 绘图辅助 --------------------

    def _xy(self, sub):
        """取绘图横轴：优先绝对 UTC 时刻，缺失时退回文件行号

        返回 (x, use_time, xlabel)。

        行号作为横轴有三个致命问题：定位不到时刻、换算不出秒、
        无法跨文件对齐 —— RTK 测试的比对与批量分析都依赖统一时间。
        报文没有日期字段（无 RMC/ZDA）时无法组成绝对时刻，
        此时退回行号并在标题里注明，而不是画一条假的"时间"轴。
        """
        tu = (sub or {}).get('times_utc')
        if tu:
            filled, last = [], None
            for t in tu:
                if t is not None:
                    last = t
                filled.append(last)
            if all(t is not None for t in filled):
                return filled, True, 'UTC 时间'
            got = sum(1 for t in filled if t is not None)
            if got >= len(filled) * 0.5:
                # 超过一半有时刻：用最近已知时刻补齐空缺，仍按时间画
                filled = [t if t is not None else filled[0] for t in filled]
                return filled, True, 'UTC 时间'
        return (sub or {}).get('times'), False, '文件行号'

    def _xlabel(self, ax, use_time, xl):
        if use_time:
            timebase.style_time_axis(ax)
        else:
            ax.set_xlabel(xl)
        return use_time

    def _draw_all(self):
        self._draw_fix()
        self._draw_age()
        self._draw_dop()
        self._draw_lock()
        self._draw_snr()
        self._draw_constellation()
        self._draw_rms()
        self._draw_ellipse()
        self._draw_cpu()
        self._draw_rmc()
        self._draw_integrity()
        self._draw_summary()
        self._fig.tight_layout(pad=1.5, h_pad=3.0, w_pad=2.0)
        self._canvas.draw_idle()
        self._draw_rtk()

    # -------------------- RTK 时效页绘制 --------------------

    def _draw_rtk(self):
        """绘制 RTK 时效页

        必须有 draw_idle：matplotlib 的 Qt 画布不会因为 Axes 被改动而
        自动刷新，漏掉会表现为「切了页签图是空的」。
        """
        if getattr(self, '_rtk_fig', None) is None:
            return
        import matplotlib.dates as mdates

        rtk = self._stats.get('rtk') or {}
        for ax in (self._ax_rtk_state, self._ax_rtk_share,
                   self._ax_rtk_bins, self._ax_rtk_text):
            ax.clear()

        # ---- [0] 解算状态时序条带 ----
        ax = self._ax_rtk_state
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        mi = rtk.get('median_interval_s') or 1.0
        minw = max(mi / 86400.0, 1e-7)      # 单历元片段也要看得见

        slots = [('固定段', rtk.get('segments'), '#2ca02c'),
                 ('抖动', rtk.get('jitters'), '#ff7f0e'),
                 ('真失锁', rtk.get('losses'), '#d62728'),
                 ('捕获期', rtk.get('warmups'), '#9a9a9a')]

        def span_of(x):
            """取片段端点（segments 与 episodes 统一用 t_start / t_end）"""
            a = x.get('t_start')
            b = x.get('t_end') or x.get('t_last') or a
            return a, b

        y = 0
        shown = []
        for name, items, color in slots:
            if not items:
                continue
            spans = []
            for x in items:
                a, b = span_of(x)
                a = mdates.date2num(a)
                b = mdates.date2num(b)
                spans.append((a, max(b - a, minw)))
            ax.broken_barh(spans, (y + 0.1, 0.8), facecolors=color, alpha=0.9)
            shown.append((name, y + 0.5))
            y += 1
        if shown:
            ax.set_yticks([p for _, p in shown])
            ax.set_yticklabels([n for n, _ in shown], fontsize=9)
            ax.set_ylim(-0.2, y + 0.2)
        timebase.style_time_axis(ax)
        ax.set_title(
            f"解算状态时序　{rtk['n_epochs']} 个历元 / "
            f"{timebase.fmt_duration(rtk['duration_s'])}　"
            f"固定率 {rtk['fix_rate']:.2f}%"
            f"（{rtk['n_fix']}/{rtk['n_epochs']}）", fontsize=11)

        # ---- [1] 解算类型占比 ----
        ax = self._ax_rtk_share
        ax.grid(True, axis='x', alpha=0.25, linestyle='--', linewidth=0.5)
        items = sorted(rtk['counts'].items(), key=lambda kv: -kv[1])
        names = [solution_mod.q_name(k) for k, _ in items]
        vals = [v for _, v in items]
        cmap = {4: '#2ca02c', 5: '#ff7f0e', 2: '#1f77b4', 1: '#d62728',
                0: '#8c8c8c'}
        colors = [cmap.get(k, '#7f7f7f') for k, _ in items]
        yy = np.arange(len(items))
        ax.barh(yy, vals, color=colors, alpha=0.9)
        ax.set_yticks(yy)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        for i, v in zip(yy, vals):
            ax.text(v + max(vals) * 0.01, i, f'{v}  ({v / rtk["n_epochs"] * 100:.1f}%)',
                    va='center', fontsize=8)
        ax.set_xlim(0, max(vals) * 1.25 if vals else 1)
        ax.set_xlabel('历元数', fontsize=9)
        ax.set_title('解算类型占比（固定率 = RTK 固定 / 全部历元）', fontsize=11)

        # ---- [2] 分档固定率 ----
        ax = self._ax_rtk_bins
        ax.grid(True, axis='x', alpha=0.25, linestyle='--', linewidth=0.5)
        rows, labels = [], []
        for key, title in (('by_nsat', '卫星数'), ('by_hdop', 'HDOP'),
                           ('by_age', '龄期(s)')):
            for r in rtk.get(key) or []:
                rows.append((f"{title} {r['label']}", r['fix_rate'], r['n']))
        if rows:
            yy = np.arange(len(rows))
            vals = [r[1] for r in rows]
            cols = ['#d62728' if v < 50 else '#ff7f0e' if v < 90 else '#2ca02c'
                    for v in vals]
            ax.barh(yy, vals, color=cols, alpha=0.9)
            ax.set_yticks(yy)
            ax.set_yticklabels([r[0] for r in rows], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            ax.axvline(100, color='gray', lw=0.5, ls=':')
            for i, r in zip(yy, rows):
                ax.text(r[1] + 1.5, i, f"{r[1]:.0f}% (n={r[2]})",
                        va='center', fontsize=8)
            ax.set_xlabel('固定率 (%)', fontsize=9)
        else:
            ax.text(0.5, 0.5, '无分档数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='gray')
        ax.set_title('分档固定率（什么条件下会掉固定）', fontsize=11)

        # ---- [3] 文本摘要 ----
        ax = self._ax_rtk_text
        ax.axis('off')
        self._render_rtk_text(ax, rtk)

        # 不能用 tight_layout：broken_barh 与日期轴会让它报警且布局错乱
        # （共享轴/特殊 Axes 都不兼容），一律用 subplots_adjust 显式给边距。
        self._rtk_fig.subplots_adjust(left=0.075, right=0.985, top=0.945,
                                      bottom=0.06, hspace=0.42, wspace=0.24)
        self._rtk_fig.canvas.draw_idle()

    def _render_rtk_text(self, ax, rtk):
        """把 RTK 时效指标排成文本摘要"""
        L = []
        L.append('── 时段 ──')
        L.append(f"  {timebase.fmt_dt(rtk['t_start'], True)} ~ "
                 f"{timebase.fmt_dt(rtk['t_end'], True)}"
                 f"   时长 {timebase.fmt_duration(rtk['duration_s'])}"
                 f"   历元 {rtk['n_epochs']}"
                 f"   间隔中位 {rtk['median_interval_s']} s")
        note = rtk.get('series_note') or {}
        if note.get('n_dup_gga'):
            L.append(f"  注：GGA 原始 {note['n_gga_raw']} 条，"
                     f"按绝对时刻去重后为 {rtk['n_epochs']} 个历元"
                     f"（丢弃重复 {note['n_dup_gga']} 条）")
        L.append('')
        L.append('── 时效指标 ──')
        L.append(f"  固定率（占全部历元） {rtk['fix_rate']:6.2f}%"
                 f"   （{rtk['n_fix']}/{rtk['n_epochs']}）")
        L.append(f"  固定率（占有解历元） {rtk['fix_rate_valid']:6.2f}%"
                 f"   有解率 {rtk['valid_rate']:6.2f}%")
        t = rtk.get('ttff')
        if t:
            L.append(f"  首次固定 TTFF        {timebase.fmt_duration(t['seconds'])}"
                     f"   （{timebase.fmt_dt(t['t'], True)}）")
            if t.get('from_valid_s') is not None:
                L.append(f"    从首个有解算起      "
                         f"{timebase.fmt_duration(t['from_valid_s'])}")
        else:
            L.append('  首次固定 TTFF        全程未固定')
        L.append(f"  状态切换             {rtk['switches']} 次")
        L.append(f"  失锁（真）           {rtk['n_losses']} 次"
                 f"（未恢复 {rtk['n_losses_unrecovered']} 次）"
                 f"   抖动 {rtk['n_jitter']} 次   捕获期 {rtk['n_warmup']} 段")
        r = rtk.get('recovery')
        if r:
            L.append(f"  恢复耗时             中位 "
                     f"{timebase.fmt_duration(r['median_s'])}"
                     f"   P95 {timebase.fmt_duration(r['p95_s'])}"
                     f"   最长 {timebase.fmt_duration(r['max_s'])}")
        sg = rtk.get('seg_stats')
        if sg:
            L.append(f"  固定段               {sg['n']} 段"
                     f"   最长 {timebase.fmt_duration(sg['longest_s'])}"
                     f"   中位 {timebase.fmt_duration(sg['median_s'])}"
                     f"   累计 {timebase.fmt_duration(sg['total_s'])}")
        L.append('')
        L.append('── 说明 ──')
        L.append(f"  · 固定率 = fix_quality==4 的历元占比，与「有解率」"
                 f"（>0，含单点/差分/浮点）不是一回事")
        L.append(f"  · 首次固定之前的时段属捕获期，不计入失锁")
        L.append(f"  · 掉出固定短于 {rtk.get('recovery_min_s', 1.0):.0f}s "
                 f"记为抖动，不计入失锁次数，但仍列出")
        ax.text(0.005, 0.985, '\n'.join(L), transform=ax.transAxes,
                va='top', ha='left', fontsize=8.8)

    def _draw_fix(self):
        """Fix 质量时序"""
        ax = self._ax_fix
        ax.clear()
        fq = self._stats.get('fix_quality')
        if not fq:
            ax.text(0.5, 0.5, '无 GGA / RMC 报文', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('定位解质量时序', fontsize=11)
            return

        times, use_time, xl = self._xy(fq)
        vals = fq['values']
        ax.scatter(times, vals, c=vals, cmap='RdYlGn', s=12, alpha=0.8,
                   edgecolors='none')
        ax.step(times, vals, where='mid', linewidth=0.5, alpha=0.5)
        self._xlabel(ax, use_time, xl)
        ax.set_ylabel('Fix Quality')
        ax.set_yticks(sorted(fq['quality_distribution'].keys()))
        labels = [QualityStatistics.FIX_NAMES.get(k, str(k))
                  for k in sorted(fq['quality_distribution'].keys())]
        ax.set_yticklabels(labels, fontsize=8)
        # 固定率与有解率是两件事：前者才是 RTK 能力
        ttl = (f'定位解质量时序  固定率 {fq["fix_rate"]:.1f}%'
               if 'fix_rate' in fq else
               f'定位解质量时序  成功率 {fq["success_rate"] * 100:.1f}%')
        if use_time:
            ttl += ''
        else:
            ttl += '（无日期字段，横轴为行号）'
        ax.set_title(f'{ttl}，切换{fq["switch_count"]}次', fontsize=11)
        ax.grid(True, alpha=0.3)

    def _draw_age(self):
        """差分龄期时序"""
        ax = self._ax_age
        ax.clear()
        age = self._stats.get('age')
        if not age:
            ax.text(0.5, 0.5, '无差分龄期数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('差分龄期时序', fontsize=11)
            return

        times, use_time, xl = self._xy(age)
        vals = age['values']
        ax.plot(times, vals, '-', linewidth=0.8, color='#4a90d9', alpha=0.8)
        ax.scatter(times, vals, s=8, color='#4a90d9')
        ax.axhline(y=age['mean'], color='orange', linestyle='-',
                   label=f'均值 {age["mean"]:.1f}s')
        warn_s = age.get('age_warn_s', QualityStatistics.AGE_WARN_S)
        ax.axhline(y=warn_s, color='red', linestyle='--',
                   label=f'告警 {warn_s:.0f}s')
        self._xlabel(ax, use_time, xl)
        ax.set_ylabel('龄期 (s)')
        over = age.get('over_warn_count', age.get('over_60s_count', 0))
        ax.set_title(f'差分龄期时序 (最大{age["max"]:.1f}s, 超{warn_s:.0f}s {over}次)',
                     fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_dop(self):
        """DOP 时序（PDOP/HDOP/VDOP）"""
        ax = self._ax_dop
        ax.clear()
        dop = self._stats.get('dop')
        if not dop:
            ax.text(0.5, 0.5, '无 GSA 报文', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('DOP 时序', fontsize=11)
            return

        use_time = False
        xl = '文件行号'
        for name, key, color in [('PDOP', 'pdop', '#d62728'),
                                 ('HDOP', 'hdop', '#1f77b4'),
                                 ('VDOP', 'vdop', '#2ca02c')]:
            d = dop.get(key)
            if not d:
                continue
            times, use_time, xl = self._xy(d)
            vals = d['values']
            ax.plot(times, vals, '-', linewidth=0.7, alpha=0.7, color=color,
                    label=f'{name} (均{d["mean"]:.2f})')

        self._xlabel(ax, use_time, xl)
        ax.set_ylabel('DOP')
        ratio = dop.get('hdop_vdop_ratio')
        ratio_txt = f', H/V比 {ratio:.2f}' if ratio else ''
        ax.set_title(f'DOP 时序{ratio_txt}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_lock(self):
        """锁定率：参与解算 vs 在视卫星数"""
        ax = self._ax_lock
        ax.clear()
        lr = self._stats.get('lock_rate') or {}
        used = lr.get('sats_used_mean')
        view = lr.get('sats_in_view_mean')
        rate = lr.get('lock_rate')

        if used is None and view is None:
            ax.text(0.5, 0.5, '无 GSA / GSV 报文', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('锁定率', fontsize=11)
            return

        labels = []
        values = []
        if used is not None:
            labels.append(f'参与解算\n{used:.1f}')
            values.append(used)
        if view is not None:
            labels.append(f'在视卫星\n{view:.1f}')
            values.append(view)
        x = np.arange(len(labels))
        colors = ['#4a90d9', '#5cb85c'][:len(labels)]
        ax.bar(x, values, width=0.5, color=colors, alpha=0.85, edgecolor='white')
        for xi, vi in zip(x, values):
            ax.text(xi, vi + 0.5, f'{vi:.1f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('卫星数')
        rate_txt = f'{rate*100:.1f}%' if rate else '—'
        ax.set_title(f'锁定率 ({rate_txt})', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    def _draw_snr(self):
        """SNR 分布直方图"""
        ax = self._ax_snr
        ax.clear()
        sig = self._stats.get('signal')
        if not sig:
            ax.text(0.5, 0.5, '无 GSV / SATSINFO 报文', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('SNR 分布直方图', fontsize=11)
            return

        snrs = [s['snr'] for s in sig['samples'] if s['snr'] is not None]
        if not snrs:
            ax.text(0.5, 0.5, '无有效 SNR 数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('SNR 分布直方图', fontsize=11)
            return

        ax.hist(snrs, bins=20, color='#4a90d9', alpha=0.8, edgecolor='white')
        ax.axvline(GSVStatistics.SNR_POOR, color='#d9534f', ls='--', lw=1.2,
                   label=f'差/良分界 {GSVStatistics.SNR_POOR}')
        ax.axvline(GSVStatistics.SNR_GOOD, color='#5cb85c', ls='--', lw=1.2,
                   label=f'良/优分界 {GSVStatistics.SNR_GOOD}')
        ax.set_xlabel('SNR (dBHz)')
        ax.set_ylabel('观测次数')
        src = sig.get('source', '')
        ax.set_title(f'SNR 分布直方图 ({src})', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_constellation(self):
        """各星座卫星数与平均 SNR 柱状图"""
        ax = self._ax_const
        ax.clear()
        sig = self._stats.get('signal')
        if not sig:
            ax.text(0.5, 0.5, '无星座数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('各星座卫星数与平均SNR', fontsize=11)
            return

        per_const = sig['per_constellation']
        if not per_const:
            ax.set_title('各星座卫星数与平均SNR (无数据)', fontsize=11)
            return

        names = list(per_const.keys())
        sats_cnt = [per_const[c]['sats'] for c in names]
        snr_mean = [per_const[c]['snr_mean'] or 0 for c in names]
        colors = [CONSTELLATION_COLORS.get(c, '#999999') for c in names]

        x = np.arange(len(names))
        ax.bar(x - 0.18, sats_cnt, width=0.36, color=colors, alpha=0.85,
               label='去重卫星数')
        ax.set_ylabel('去重卫星数')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax2 = ax.twinx()
        ax2.bar(x + 0.18, snr_mean, width=0.36, color='#f0ad4e', alpha=0.9,
                label='平均SNR(dBHz)')
        ax2.set_ylabel('平均 SNR (dBHz)', color='#c77e00')
        ax2.set_ylim(0, max(max(snr_mean) * 1.3, 10))
        ax.set_title('各星座卫星数与平均SNR', fontsize=11)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    def _draw_rms(self):
        """GST RMS 时序"""
        ax = self._ax_rms
        ax.clear()
        gst = self._stats.get('gst')
        if not gst:
            ax.text(0.5, 0.5, '无 GST 报文', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('位置 RMS 时序', fontsize=11)
            return

        rms = gst['rms']
        times, use_time, xl = self._xy(rms)
        vals = rms['values']
        ax.plot(times, vals, '-', linewidth=0.8, color='#d62728', alpha=0.8)
        ax.fill_between(times, 0, vals, alpha=0.15, color='#d62728')
        ax.axhline(y=rms['mean'], color='orange', linestyle='-',
                   label=f'均值 {rms["mean"]:.4f}m')
        self._xlabel(ax, use_time, xl)
        ax.set_ylabel('RMS (m)')
        ax.set_title(f'位置 RMS 时序 (最小{rms["min"]:.4f}, 最大{rms["max"]:.4f})',
                     fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_ellipse(self):
        """误差椭圆面积与长短轴比"""
        ax = self._ax_ellipse
        ax.clear()
        gst = self._stats.get('gst')
        if not gst or gst.get('ellipse_area_mean') is None:
            ax.text(0.5, 0.5, '无误差椭圆数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('误差椭圆指标', fontsize=11)
            return

        area = gst['ellipse_area_mean']
        ratio = gst.get('axis_ratio_mean')
        lat_e = gst.get('lat_error_mean')
        lon_e = gst.get('lon_error_mean')
        alt_e = gst.get('alt_error_mean')

        labels = ['椭圆面积\n(m²)', '长短轴比', '纬度误差\n(m)', '经度误差\n(m)',
                  '高程误差\n(m)']
        values = [area, ratio or 0, lat_e or 0, lon_e or 0, alt_e or 0]
        colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd']
        x = np.arange(len(labels))
        ax.bar(x, values, width=0.55, color=colors, alpha=0.85, edgecolor='white')
        for xi, vi, lbl in zip(x, values, labels):
            unit = 'm²' if '面积' in lbl else ''
            ax.text(xi, vi + max(values) * 0.02, f'{vi:.4f}{unit}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title('误差椭圆与各方向误差 (均值)', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    def _draw_cpu(self):
        """接收机板卡 CPU 利用率时序"""
        ax = self._ax_cpu
        ax.clear()
        cpu = self._stats.get('cpu')
        if not cpu:
            ax.text(0.5, 0.5, '无 REFSTATION / CPU 数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('板卡 CPU 利用率', fontsize=11)
            return

        x, use_time, xl = self._xy(cpu)
        vals = cpu['values']
        ax.plot(x, vals, '-', linewidth=0.8, color='#7f3c7f', alpha=0.8)
        ax.fill_between(x, 0, vals, alpha=0.12, color='#7f3c7f')
        ax.axhline(y=cpu['mean'], color='orange', linestyle='-',
                   label=f'均值 {cpu["mean"]:.1f}%')
        ax.axhline(y=90, color='red', linestyle='--', lw=1.2, label='高负载 90%')
        ax.set_ylabel('CPU 利用率 (%)')
        self._xlabel(ax, use_time, xl)
        src = []
        for name, n in (('REFSTATION', cpu['n_ref']), ('SATSINFO', cpu['n_satsinfo'])):
            if n:
                src.append(f'{name}{n}')
        src_txt = '/'.join(src)
        ax.set_title(f'板卡 CPU 利用率 (最大{cpu["max"]:.0f}%{("，来源 " + src_txt) if src_txt else ""})',
                     fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(cpu['max'] * 1.2, 100))

    def _draw_rmc(self):
        """RMC 运动动态：速度时序 + 动静态标注"""
        ax = self._ax_rmc
        ax.clear()
        rm = self._stats.get('rmc_motion')
        if not rm:
            ax.text(0.5, 0.5, '无 RMC 速度数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('RMC 运动动态', fontsize=11)
            return

        x, use_time, xl = self._xy(rm)
        speeds = rm['speeds']
        ax.plot(x, speeds, '-', linewidth=0.8, color='#17a2b8', alpha=0.85)
        ax.fill_between(x, 0, speeds, alpha=0.12, color='#17a2b8')
        ax.axhline(y=QualityStatistics.MOVING_KNOTS, color='red', ls='--', lw=1.2,
                   label=f'运动阈值 {QualityStatistics.MOVING_KNOTS:.1f} 节')
        ax.set_ylabel('速度 (节)')
        self._xlabel(ax, use_time, xl)
        state = rm['motion_state']
        mv = rm['moving_fraction'] * 100
        ax.set_title(f'RMC 运动动态 ({state}, 运动占比{mv:.0f}%)', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(rm['speed_max'] * 1.3, QualityStatistics.MOVING_KNOTS * 2))

    def _draw_integrity(self):
        """报文完整性：逐类型的校验通过 / 无法判定 / 失败

        这一块回答的是「采到的数据本身可不可信」。校验失败的句子原先
        和正常句子无法区分，一起进了统计 —— 单列出来比多画一张 SNR
        图更有诊断价值，也是判断链路是否受干扰的直接证据。
        """
        ax = self._ax_integrity
        ax.clear()
        ig = self._stats.get('integrity')
        if not ig or not ig.get('total'):
            ax.text(0.5, 0.5, '无报文', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('报文完整性（校验和）', fontsize=11)
            return

        rows = [(t, r) for t, r in ig['by_type'].items() if r['total']]
        # 失败多的排前面：一眼看到问题类型，而不是被 GGA 的数量淹没
        rows.sort(key=lambda kv: (-kv[1]['bad'], -kv[1]['total']))
        rows = rows[:10]

        ok = np.array([r['ok'] for _, r in rows], dtype=float)
        unk = np.array([r['unknown'] for _, r in rows], dtype=float)
        bad = np.array([r['bad'] for _, r in rows], dtype=float)
        yy = np.arange(len(rows))

        ax.barh(yy, ok, color='#2ca02c', alpha=0.85, label='校验通过')
        ax.barh(yy, unk, left=ok, color='#9a9a9a', alpha=0.7, label='无法判定')
        ax.barh(yy, bad, left=ok + unk, color='#d62728', alpha=0.9,
                label='校验失败')
        ax.set_yticks(yy)
        ax.set_yticklabels([t for t, _ in rows], fontsize=8)
        ax.invert_yaxis()

        for i, (t, r) in enumerate(rows):
            txt = f"{r['total']} 条"
            if r['bad']:
                txt += f"  失败 {r['bad']}"
            ax.text(r['total'] * 1.01, i, txt, va='center', fontsize=7.5)

        ax.set_xlim(0, max(r['total'] for _, r in rows) * 1.3)
        ax.set_xlabel('报文条数', fontsize=9)
        ax.grid(True, axis='x', alpha=0.25, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
        ax.set_title(
            f"报文完整性（校验和）  通过 {ig['ok_rate'] * 100:.2f}%"
            f"   失败 {ig['n_bad']} 条   无法判定 {ig['n_unknown']} 条"
            f"   修复 {ig['n_repaired']} 条   入口拒收 {ig['rejected_total']} 条",
            fontsize=11)

    def _draw_summary(self):
        """文本摘要面板"""
        ax = self._ax_summary
        ax.clear()
        ax.axis('off')

        def f(x, nd=2):
            return f'{x:.{nd}f}' if x is not None else '—'

        mc = self._stats['msg_counts']
        lines = [
            f"文件: {self._file_basename}",
            f"━━━ 报文统计 ━━━",
        ]
        for t in ('GGA', 'GSA', 'GST', 'GSV', 'RMC', 'SATSINFO'):
            lines.append(f"  {t}: {mc.get(t, 0)} 条")

        # 报文完整性
        ig = self._stats.get('integrity')
        lines.append("")
        lines.append("━━━ 报文完整性 ━━━")
        if ig:
            if ig['total']:
                lines.append(f"  校验通过: {ig['n_ok']}/{ig['total']} "
                             f"({ig['ok_rate']*100:.2f}%)")
                lines.append(f"  校验失败: {ig['n_bad']} 条 "
                             f"({ig['bad_rate']*100:.2f}%)")
                lines.append(f"  无法判定 {ig['n_unknown']} 条 | "
                             f"修复 {ig['n_repaired']} 条")
            else:
                # 全部在入口就被拒了，0/0 的百分比没有意义
                lines.append("  进入统计的报文: 0 条")
            if ig.get('rejected_total'):
                rej = sorted(ig['rejected'].items(), key=lambda kv: -kv[1])
                lines.append(f"  入口拒收: {ig['rejected_total']} 条（"
                             + ' '.join(f"{REJECT_NAMES.get(k, k)}×{v}"
                                        for k, v in rej[:5]) + '）')
            bad_types = [f"{t}×{r['bad']}" for t, r in ig['by_type'].items()
                         if r['bad']]
            if bad_types:
                lines.append(f"  失败分布: {' '.join(bad_types[:6])}")
        else:
            lines.append("  无报文")

        # 定位质量
        fq = self._stats.get('fix_quality')
        lines.append("")
        lines.append("━━━ 定位质量 ━━━")
        if fq:
            lines.append(f"  定位成功率: {fq['success_rate']*100:.1f}%")
            lines.append(f"  切换次数: {fq['switch_count']}")
            for k in sorted(fq['quality_distribution'].keys()):
                name = QualityStatistics.FIX_NAMES.get(k, str(k))
                lines.append(f"    {name}: {fq['quality_distribution'][k]} 次")
        else:
            lines.append("  无数据")

        # 几何质量
        dop = self._stats.get('dop')
        lines.append("")
        lines.append("━━━ 几何质量 ━━━")
        if dop:
            if dop.get('pdop'):
                lines.append(f"  PDOP: 均{f(dop['pdop']['mean'])} "
                             f"({f(dop['pdop']['min'])}~{f(dop['pdop']['max'])})")
            if dop.get('hdop'):
                lines.append(f"  HDOP: 均{f(dop['hdop']['mean'])}")
            if dop.get('vdop'):
                lines.append(f"  VDOP: 均{f(dop['vdop']['mean'])}")
            if dop.get('hdop_vdop_ratio'):
                lines.append(f"  H/V比: {f(dop['hdop_vdop_ratio'])}")
        else:
            lines.append("  无 GSA 数据")

        lr = self._stats.get('lock_rate') or {}
        if lr.get('lock_rate') is not None:
            lines.append(f"  锁定率: {lr['lock_rate']*100:.1f}% "
                         f"(解算{f(lr['sats_used_mean'],0)}/在视{f(lr['sats_in_view_mean'],0)})")

        # 差分龄期
        age = self._stats.get('age')
        lines.append("")
        lines.append("━━━ 差分龄期 ━━━")
        if age:
            lines.append(f"  均值: {f(age['mean'])}s | 最大: {f(age['max'])}s")
            lines.append(f"  超60s次数: {age['over_60s_count']}")
        else:
            lines.append("  无龄期数据")

        # 信号质量
        sig = self._stats.get('signal')
        lines.append("")
        lines.append("━━━ 信号质量 ━━━")
        if sig:
            snr = sig['snr']
            lines.append(f"  数据源: {sig.get('source', '')}")
            lines.append(f"  SNR(dBHz): 均{f(snr['mean'])} "
                         f"({f(snr['min'])}~{f(snr['max'])})")
            q = sig['quality']
            lines.append(f"  质量: 优{q['good']} 良{q['mid']} 差{q['poor']}")
            lines.append(f"  去重卫星: {sig['n_unique_sats']} 颗")
        else:
            lines.append("  无 GSV/SATSINFO 数据")

        # 误差椭圆
        gst = self._stats.get('gst')
        lines.append("")
        lines.append("━━━ 误差椭圆 (GST) ━━━")
        if gst:
            lines.append(f"  RMS均值: {f(gst['rms']['mean'], 4)}m")
            lines.append(f"  椭圆面积: {f(gst.get('ellipse_area_mean'), 4)}m²")
            lines.append(f"  长短轴比: {f(gst.get('axis_ratio_mean'), 2)}")
            lines.append(f"  纬向/经向/高程: "
                         f"{f(gst.get('lat_error_mean'), 4)}/"
                         f"{f(gst.get('lon_error_mean'), 4)}/"
                         f"{f(gst.get('alt_error_mean'), 4)}m")
        else:
            lines.append("  无 GST 数据")

        # 接收机 CPU 健康
        cpu = self._stats.get('cpu')
        lines.append("")
        lines.append("━━━ 板卡 CPU 健康 ━━━")
        if cpu:
            lines.append(f"  CPU利用率: 均{f(cpu['mean'],1)}% | 最大{f(cpu['max'],0)}%")
            srcs = []
            if cpu['n_ref']:
                srcs.append(f"REFSTATION×{cpu['n_ref']}")
            if cpu['n_satsinfo']:
                srcs.append(f"SATSINFO idle×{cpu['n_satsinfo']}")
            lines.append(f"  来源: {'+'.join(srcs) if srcs else '—'}")
        else:
            lines.append("  无 REFSTATION/SATSINFO CPU 数据")

        # RMC 运动动态
        rm = self._stats.get('rmc_motion')
        lines.append("")
        lines.append("━━━ RMC 运动动态 ━━━")
        if rm:
            lines.append(f"  状态: {rm['motion_state']}")
            lines.append(f"  速度: 均{f(rm['speed_mean'],3)}节 | 最大{f(rm['speed_max'],3)}节")
            lines.append(f"  运动占比: {rm['moving_fraction']*100:.0f}% "
                         f"({rm['moving_count']}/{rm['n_samples']})")
        else:
            lines.append("  无 RMC 速度数据")

        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
                va='top', ha='left', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.5', fc='#f8f8f8', ec='#cccccc'))

    # -------------------- 保存 --------------------

    def _save_figure(self):
        # 保存当前标签页的图（按标题取，不依赖固定索引）
        title = self._tabs.tabText(self._tabs.currentIndex())
        m = {'质量图表': (getattr(self, '_fig', None), 'quality'),
             'RTK 时效': (getattr(self, '_rtk_fig', None), 'rtk')}
        fig, tag = m.get(title, (None, None))
        if fig is None:
            fig, tag = getattr(self, '_fig', None), 'quality'
        if fig is None:
            QMessageBox.information(self, "提示", "当前页面没有可保存的图。")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存图片",
            os.path.join(last_dir('figure'),
                         f"quality_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
            "PNG图片 (*.png);;PDF文档 (*.pdf);;SVG矢量图 (*.svg)"
        )
        if filepath:
            save_last_dir('figure', filepath)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "完成", f"图片已保存到:\n{filepath}")
