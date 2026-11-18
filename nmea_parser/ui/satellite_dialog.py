# -*- coding: utf-8 -*-
"""卫星搜星与信噪比分析对话框（基于 GSV 报文）

呈现：卫星分布图(天顶图) / SNR直方图 / 搜星数时序 / 星座统计柱状图 /
逐卫星SNR时序（顶部下拉框选定某星后刷新）。
"""

import os
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QFileDialog, QMessageBox,
)

# 导入中文工具栏会先触发 matplotlib.use('Qt5Agg')
from ..toolbar import ChineseNavToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..statistics import GSVStatistics, SATSINFOStatistics
from ..config import last_dir, save_last_dir

# 星座颜色（天顶图均值点描边用，柱状图用）
CONSTELLATION_COLORS = {
    'GPS': '#1f77b4',
    '北斗': '#d62728',
    'GLONASS': '#2ca02c',
    'Galileo': '#ff7f0e',
    'QZSS': '#9467bd',
    'IRNSS': '#8c564b',
}


class SatelliteDialog(QDialog):
    """卫星搜星/信噪比分析对话框

    数据源优先使用扩展 SATSINFO 报文（多星座、逐星多频点 SNR），
    若无则回退到标准 GSV 报文。
    """

    def __init__(self, gsv_msgs, file_basename, parent=None, satsinfo_msgs=None):
        super().__init__(parent)
        self.setWindowTitle("卫星搜星与信噪比分析")
        self.resize(1400, 1000)
        self.valid = False

        # 优先使用 SATSINFO 数据
        satsinfo_msgs = satsinfo_msgs or []
        if satsinfo_msgs:
            self._source = 'SATSINFO'
            self._source_detail = (
                f"{len(satsinfo_msgs)} 条 SATSINFO" if satsinfo_msgs else '')
            self._stats = SATSINFOStatistics.compute_stats(satsinfo_msgs)
        elif gsv_msgs:
            self._source = 'GSV'
            self._source_detail = f"{len(gsv_msgs)} 条 GSV"
            self._stats = GSVStatistics.compute_stats(gsv_msgs)
        else:
            QMessageBox.warning(
                self, "提示",
                "数据中没有 GSV 或 SATSINFO 报文，无法分析搜星/信噪比。\n"
                "请在接收机或采集软件中开启 GSV / SATSINFO 输出后重新采集。")
            return

        self._per_sat = self._stats['per_satellite']
        self._file_basename = file_basename
        self.valid = True

        self._build_ui()
        self._draw_all()

    # -------------------- UI --------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 顶部：卫星选择
        top = QHBoxLayout()
        top.addWidget(QLabel("逐卫星SNR曲线 卫星选择:"))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(320)
        for key in sorted(self._per_sat.keys(),
                          key=lambda k: (self._per_sat[k]['constellation'],
                                         int(self._per_sat[k]['prn'])
                                         if self._per_sat[k]['prn'].isdigit()
                                         else self._per_sat[k]['prn'])):
            v = self._per_sat[key]
            freq_txt = ""
            if v.get('freqs'):
                freq_txt = f", {len(v['freqs'])}频点"
            if v['snr_mean'] is not None:
                label = f"{key}  (均值 {v['snr_mean']:.1f} dBHz, 观测{v['n_obs']}次{freq_txt})"
            else:
                label = f"{key}  (未锁定, 观测{v['n_obs']}次{freq_txt})"
            self._combo.addItem(label, userData=key)
        self._combo.currentIndexChanged.connect(self._update_sat_curve)
        top.addWidget(self._combo, stretch=1)
        top.addStretch(1)
        layout.addLayout(top)

        # 图形：天顶图占左侧整列，右侧 2x2 放 SNR直方/星座柱状/逐星SNR/统计摘要
        fig = self._fig = Figure(figsize=(15, 10), dpi=100)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
        self._ax_sky = fig.add_subplot(gs[:, 0], projection='polar')
        self._ax_hist = fig.add_subplot(gs[0, 1])
        self._ax_const = fig.add_subplot(gs[1, 1])
        self._ax_sat = fig.add_subplot(gs[0, 2])
        self._ax_sum = fig.add_subplot(gs[1, 2])

        self._canvas = FigureCanvas(fig)
        layout.addWidget(self._canvas, stretch=1)

        toolbar = ChineseNavToolbar(self._canvas, self)
        layout.addWidget(toolbar)

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

    # -------------------- 绘图 --------------------

    def _draw_all(self):
        self._draw_skyplot()
        self._draw_hist()
        self._draw_constellation()
        self._update_sat_curve()   # 也负责首颗卫星
        self._draw_summary()
        self._fig.tight_layout(pad=2.0, h_pad=3.5, w_pad=3.0)
        self._canvas.draw_idle()

    def _draw_skyplot(self):
        """卫星分布图（极坐标天顶图）

        方位角为角度(北在顶部)、90-仰角为半径。
        每颗卫星的均值位置按平均SNR上色（主视觉），并叠加浅色轨迹背景。
        """
        stats = self._stats
        samples = stats['samples']
        ax = self._ax_sky
        ax.clear()

        azs = [s['azimuth'] for s in samples if s['azimuth'] is not None]
        if not azs:
            ax.set_title('卫星分布图（无有效方位/仰角数据）')
            return

        # 背景：全部观测点（细点、低透明度，按瞬时SNR配色，仅作轨迹）
        els = [s['elevation'] for s in samples if s['azimuth'] is not None]
        snrs_all = [s['snr'] if s['snr'] is not None else np.nan
                    for s in samples if s['azimuth'] is not None]
        theta = np.radians(90 - np.array(azs))   # 北在顶部
        r = 90 - np.array(els)                    # 天顶=0，地平=90
        ax.scatter(theta, r, c=snrs_all, cmap='viridis', s=6,
                   alpha=0.25, edgecolors='none')

        # 前景：每颗卫星均值位置，按平均SNR上色 + PRN标注
        mean_th, mean_r, mean_snr, labels = [], [], [], []
        for key, v in self._per_sat.items():
            azs_k = [s['azimuth'] for s in samples
                     if s['azimuth'] is not None
                     and f"{s['constellation']}-{s['prn']}" == key]
            els_k = [s['elevation'] for s in samples
                     if s['azimuth'] is not None
                     and f"{s['constellation']}-{s['prn']}" == key]
            if not azs_k:
                continue
            # 方位角圆周均值（处理0/360环绕）
            azm = (np.degrees(np.arctan2(
                np.mean(np.sin(np.radians(azs_k))),
                np.mean(np.cos(np.radians(azs_k))))) % 360)
            elm = float(np.mean(els_k))
            mean_th.append(np.radians(90 - azm))
            mean_r.append(90 - elm)
            mean_snr.append(v['snr_mean'] if v['snr_mean'] is not None else np.nan)
            labels.append((key, v['prn']))

        sc_mean = ax.scatter(mean_th, mean_r, c=mean_snr, cmap='viridis',
                             s=95, edgecolors='black', linewidths=0.8, zorder=3)
        for (key, prn), th, rr in zip(labels, mean_th, mean_r):
            ax.annotate(prn, (th, rr), xytext=(4, 4),
                        textcoords='offset points', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                  alpha=0.75, ec='none'))

        # 轴配置：北在顶部，东/西/南标注；高程环带 60/30/0 度
        ax.set_ylim(0, 90)
        ax.set_yticks([30, 60, 90])
        ax.set_yticklabels(['60°', '30°', '0°'], fontsize=8)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(['东', '北', '西', '南'], fontsize=9)
        src = 'SATSINFO' if self._source == 'SATSINFO' else 'GSV'
        ax.set_title(f'卫星分布图（{src}，天顶图，按SNR配色）', fontsize=11)
        self._fig.colorbar(sc_mean, ax=ax, label='SNR (dBHz)', pad=0.10, shrink=0.8)

    def _draw_hist(self):
        """SNR 直方图 + 质量分档界线"""
        stats = self._stats
        snrs = [s['snr'] for s in stats['samples'] if s['snr'] is not None]
        ax = self._ax_hist
        ax.clear()
        if not snrs:
            ax.set_title('SNR 直方图（无有效SNR）')
            return
        ax.hist(snrs, bins=20, color='#4a90d9', alpha=0.8, edgecolor='white')
        ax.axvline(GSVStatistics.SNR_POOR, color='#d9534f', ls='--', lw=1.2,
                   label=f"差/良分界 {GSVStatistics.SNR_POOR}")
        ax.axvline(GSVStatistics.SNR_GOOD, color='#5cb85c', ls='--', lw=1.2,
                   label=f"良/优分界 {GSVStatistics.SNR_GOOD}")
        ax.set_xlabel('SNR (dBHz)')
        ax.set_ylabel('观测次数')
        ax.set_title('SNR 分布直方图', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _draw_constellation(self):
        """各星座卫星数与平均SNR 柱状图（双Y轴）"""
        per_const = self._stats['per_constellation']
        ax = self._ax_const
        ax.clear()
        if not per_const:
            ax.set_title('星座统计（无数据）')
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
        ax.set_xticklabels(names)
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

    def _update_sat_curve(self):
        """刷新逐卫星 SNR（跟随下拉框选择）

        SATSINFO 数据源：显示该星各频点的平均 SNR 柱状图；
        GSV 数据源：显示 SNR 随报文序号的时序曲线。
        """
        key = self._combo.currentData()
        ax = self._ax_sat
        ax.clear()
        if not key or key not in self._per_sat:
            ax.set_title('逐卫星SNR')
            self._canvas.draw_idle()
            return
        v = self._per_sat[key]

        # SATSINFO：多频点柱状图
        if v.get('freqs'):
            names = [f['name'] for f in v['freqs']]
            means = [f['snr_mean'] if f['snr_mean'] is not None else 0
                     for f in v['freqs']]
            x = np.arange(len(names))
            ax.bar(x, means, color='#4a90d9', alpha=0.85, edgecolor='white')
            for xi, fi in zip(x, v['freqs']):
                val = fi['snr_mean']
                if val is not None:
                    ax.text(xi, val + 1, f"{val:.0f}", ha='center',
                            va='bottom', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=8, rotation=20, ha='right')
            mean_str = f"{v['snr_mean']:.1f} dBHz" if v['snr_mean'] is not None else '无SNR'
            ax.set_ylabel('SNR (dBHz)')
            ax.set_ylim(0, 60)
            ax.set_title(f"{key} 各频点平均SNR (最佳均值 {mean_str})", fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            self._canvas.draw_idle()
            return

        # GSV：SNR 时序曲线
        lines = np.asarray(v['lines'], dtype=float)
        snrs = np.asarray(v['snrs'], dtype=float)
        ax.plot(lines, snrs, '-', linewidth=0.9, color='#4a90d9', alpha=0.85)
        ax.scatter(lines, snrs, s=14, color='#4a90d9')
        mean_str = f"{v['snr_mean']:.1f} dBHz" if v['snr_mean'] is not None else '无SNR'
        ax.set_xlabel('报文序号')
        ax.set_ylabel('SNR (dBHz)')
        ax.set_title(f"{key} 信噪比时序 (均值 {mean_str})", fontsize=11)
        ax.set_ylim(0, 60)
        ax.grid(True, alpha=0.3)
        # 必须刷新画布，否则下拉切换后画面不更新
        self._canvas.draw_idle()

    def _draw_summary(self):
        """统计摘要文本"""
        st = self._stats['snr']
        q = self._stats['quality']
        stats = self._stats
        ax = self._ax_sum
        ax.clear()
        ax.axis('off')

        def f(x, nd=1):
            return f'{x:.{nd}f}' if x is not None else '—'

        lines = [
            f"文件: {self._file_basename}",
            f"数据源: {self._source} ({self._source_detail})",
            f"{self._source}消息: {stats['n_gsv']} 条",
            f"卫星观测: {stats['n_samples']} 次 | 去重卫星: {stats['n_unique_sats']} 颗",
            f"未锁定(SNR空): {stats['n_blank_snr']} 次"
            + (f" ({stats['n_blank_snr'] / stats['n_samples'] * 100:.1f}%)"
               if stats['n_samples'] else ""),
            f"SNR(dBHz): 最小 {f(st['min'])} | 均值 {f(st['mean'])} | "
            f"中位 {f(st['median'])} | 最大 {f(st['max'])} | std {f(st['std'])}",
            f"质量分档: 优(≥{GSVStatistics.SNR_GOOD}) {q['good']} | "
            f"良({GSVStatistics.SNR_POOR}~{GSVStatistics.SNR_GOOD}) {q['mid']} | "
            f"差(<{GSVStatistics.SNR_POOR}) {q['poor']}",
            "",
            "各星座平均SNR:",
        ]
        for name, c in stats['per_constellation'].items():
            lines.append(
                f"  {name}: {c['sats']}颗, 观测{c['obs']}次, 均SNR {f(c['snr_mean'])}dBHz")
        lines.append("")
        if self._source == 'SATSINFO':
            lines.append("注: 卫星SNR为其各频点最佳值; 逐星面板显示各频点SNR柱状图")
        else:
            lines.append("注: SNR为空=未锁定; 报文序号≈时间(线性)")
        ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
                va='top', ha='left', fontsize=9)

    # -------------------- 保存 --------------------

    def _save_figure(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存图片",
            os.path.join(last_dir('figure'),
                         f"satellite_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
            "PNG图片 (*.png);;PDF文档 (*.pdf);;SVG矢量图 (*.svg)"
        )
        if filepath:
            save_last_dir('figure', filepath)
            self._fig.savefig(filepath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "完成", f"图片已保存到:\n{filepath}")
