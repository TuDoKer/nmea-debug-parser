# -*- coding: utf-8 -*-
"""各类对话框：报文详情 / 导出类型选择 / GGA统计报告 / GGA精度可视化"""

import csv
import os
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
    QCheckBox, QScrollArea, QWidget, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

# 导入中文工具栏会先触发 matplotlib.use('Qt5Agg')，再导入 backend 组件
from ..toolbar import ChineseNavToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..statistics import GGAStatistics
from ..config import last_dir, save_last_dir


class MessageDetailDialog(QDialog):
    """双击 NMEA 行 → 显示完整原始报文"""

    def __init__(self, raw_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("报文详情")
        self.resize(800, 200)
        layout = QVBoxLayout(self)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Consolas", 10))
        text_edit.setPlainText(raw_text)
        layout.addWidget(text_edit)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)


class ExportTypeSelectionDialog(QDialog):
    """导出 CSV 前选择要导出的报文类型"""

    def __init__(self, type_counts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择要导出的报文类型")
        self.resize(320, 450)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("请选择要导出的报文类型:"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setAlignment(Qt.AlignTop)

        self._checkboxes = {}
        for msg_type in sorted(type_counts.keys()):
            cb = QCheckBox(f"{msg_type} ({type_counts[msg_type]})")
            cb.setChecked(True)
            scroll_layout.addWidget(cb)
            self._checkboxes[msg_type] = cb

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        btn_layout = QHBoxLayout()
        btn_export = QPushButton("导出")
        btn_export.setStyleSheet("background-color: #5cb85c;")
        btn_cancel = QPushButton("取消")
        btn_cancel.setStyleSheet("background-color: #d9534f;")
        btn_export.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_export)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def selected_types(self):
        """返回用户勾选的报文类型列表"""
        return [t for t, cb in self._checkboxes.items() if cb.isChecked()]


class StatisticsDialog(QDialog):
    """GGA 报文统计分析对话框（表格展示）"""

    def __init__(self, gga_msgs, file_basename, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GGA报文统计分析")
        self.resize(820, 850)
        self.valid = False
        self._report = ""

        # 校验 GGA 数据量
        if len(gga_msgs) < 2:
            QMessageBox.warning(self, "提示",
                                f"有效GGA报文不足（当前{len(gga_msgs)}条，至少需要2条）")
            return

        stats = GGAStatistics.compute_stats(gga_msgs)
        if 'error' in stats:
            QMessageBox.warning(self, "提示", stats['error'])
            return

        self.valid = True
        self._report = self._build_stats_report(stats, file_basename)
        self._stats = stats
        self._file_basename = file_basename

        layout = QVBoxLayout(self)
        table = self._build_table(stats, file_basename)
        layout.addWidget(table)

        btn_layout = QHBoxLayout()
        btn_save = QPushButton("💾 保存报告")
        btn_csv = QPushButton("📋 导出CSV")
        btn_close = QPushButton("关闭")
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_csv)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        btn_save.clicked.connect(self._save_report)
        # 用 lambda 丢弃 clicked 的 checked 参数，避免它被当成 filepath 导致静默返回
        btn_csv.clicked.connect(lambda: self._export_csv())
        btn_close.clicked.connect(self.accept)

    @staticmethod
    def _collect_rows(stats, file_basename):
        """收集统计表格行：('section'|'data', label, value)，界面与CSV导出共用"""
        h = stats['horizontal']
        v = stats['vertical']
        o = stats['other']
        ex = stats.get('outliers')
        ref = stats['reference']

        rows = []  # ('section'|'data', label, value)

        def add_section(text):
            rows.append(('section', text, ''))

        def add(label, value):
            rows.append(('data', label, value))

        add_section('基础信息')
        add('数据文件', file_basename)
        add('GGA报文总数', str(h['count']))
        add('参考点纬度', f"{ref['lat']:.8f}°")
        add('参考点经度', f"{ref['lon']:.8f}°")
        add('参考点高程', f"{ref['alt']:.3f} m" if ref['alt'] is not None else '无')

        add_section('经纬度统计')
        add('纬度均值', f"{h['lat_mean']:.8f}°")
        add('纬度中位数', f"{h['lat_median']:.8f}°")
        add('纬度标准差', f"{h['lat_std']:.8f}°")
        add('纬度范围', f"{h['lat_min']:.8f} ~ {h['lat_max']:.8f}°")
        add('经度均值', f"{h['lon_mean']:.8f}°")
        add('经度中位数', f"{h['lon_median']:.8f}°")
        add('经度标准差', f"{h['lon_std']:.8f}°")
        add('经度范围', f"{h['lon_min']:.8f} ~ {h['lon_max']:.8f}°")

        add_section('ENU方向统计')
        add('东向(E)均值', f"{h['east_mean']:.4f} m")
        add('东向(E)中位数', f"{h['east_median']:.4f} m")
        add('东向(E)标准差', f"{h['east_std']:.4f} m")
        add('东向(E)RMS', f"{h['east_rms']:.4f} m")
        add('北向(N)均值', f"{h['north_mean']:.4f} m")
        add('北向(N)中位数', f"{h['north_median']:.4f} m")
        add('北向(N)标准差', f"{h['north_std']:.4f} m")
        add('北向(N)RMS', f"{h['north_rms']:.4f} m")

        add_section('水平误差统计')
        add('水平误差均值', f"{h['horiz_error_mean']:.4f} m")
        add('水平误差中位数', f"{h['horiz_error_median']:.4f} m")
        add('水平误差标准差', f"{h['horiz_error_std']:.4f} m")
        add('水平误差RMS', f"{h['horiz_error_rms']:.4f} m")
        add('水平误差最小值', f"{h['horiz_error_min']:.4f} m")
        add('水平误差最大值', f"{h['horiz_error_max']:.4f} m")
        add('CEP50', f"{h['cep50']:.4f} m")
        add('CEP68', f"{h['cep68']:.4f} m")
        add('CEP95', f"{h['cep95']:.4f} m")
        add('CEP99', f"{h['cep99']:.4f} m")
        add('DRMS', f"{h['drms']:.4f} m")
        add('2DRMS', f"{h['2drms']:.4f} m")

        add_section('垂直方向统计')
        if 'error' not in v:
            add('高度均值', f"{v['alt_mean']:.3f} m")
            add('高度中位数', f"{v['alt_median']:.3f} m")
            add('高度标准差', f"{v['alt_std']:.3f} m")
            add('高度范围', f"{v['alt_min']:.3f} ~ {v['alt_max']:.3f} m")
            add('垂直误差均值', f"{v['vert_error_mean']:.3f} m")
            add('垂直误差中位数', f"{v['vert_error_median']:.3f} m")
            add('垂直误差标准差', f"{v['vert_error_std']:.3f} m")
            add('垂直误差RMS', f"{v['vert_error_rms']:.3f} m")
            add('垂直误差最大值', f"{v['vert_error_max']:.3f} m")
            add('VCEP68', f"{v['vert_cep68']:.3f} m")
            add('VCEP95', f"{v['vert_cep95']:.3f} m")
            add('VCEP99', f"{v['vert_cep99']:.3f} m")
        else:
            add('高度数据', v['error'])

        if o:
            add_section('其他信息')
            if 'hdop_mean' in o:
                add('HDOP均值', f"{o['hdop_mean']:.2f}")
                add('HDOP中位数', f"{o['hdop_median']:.2f}")
                add('HDOP范围', f"{o['hdop_min']:.2f} ~ {o['hdop_max']:.2f}")
            if 'sats_mean' in o:
                add('卫星数均值', f"{o['sats_mean']:.1f}")
                add('卫星数中位数', f"{o['sats_median']:.1f}")
                add('卫星数范围', f"{o['sats_min']:.0f} ~ {o['sats_max']:.0f}")

        if ex:
            add_section('异常值统计')
            add('判定阈值',
                f"水平距离 > {ex['threshold_h'] * 100:.1f}cm 或 "
                f"高程误差 > {ex['threshold_v'] * 100:.1f}cm")
            add('有效样本数', str(ex['n_valid']))
            add('异常样本数', str(ex['n_outliers']))
            add('异常率', f"{ex['rate'] * 100:.2f}%")

        return rows

    @staticmethod
    def _build_table(stats, file_basename):
        """构建统计汇总表（统计项 / 数值），分节展示"""
        rows = StatisticsDialog._collect_rows(stats, file_basename)

        table = QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(['统计项', '数值'])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.verticalHeader().setDefaultSectionSize(24)

        section_font = QFont()
        section_font.setBold(True)
        section_bg = QColor(215, 228, 242)

        for r, (kind, label, value) in enumerate(rows):
            if kind == 'section':
                item = QTableWidgetItem(label)
                item.setFont(section_font)
                item.setBackground(section_bg)
                table.setItem(r, 0, item)
                table.setSpan(r, 0, 1, 2)
            else:
                table.setItem(r, 0, QTableWidgetItem(label))
                table.setItem(r, 1, QTableWidgetItem(value))

        return table

    def _save_report(self):
        """保存统计报告为文本文件"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存报告",
            os.path.join(last_dir('report'),
                         f"gga_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"),
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if filepath:
            save_last_dir('report', filepath)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self._report)
            QMessageBox.information(self, "完成", f"报告已保存到:\n{filepath}")

    def _export_csv(self, filepath=None):
        """导出统计汇总表为 CSV（与界面表格一致，含分节，utf-8-sig 供 Excel 使用）"""
        if filepath is None:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "导出统计CSV",
                os.path.join(last_dir('stats_csv'),
                             f"gga_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
                "CSV文件 (*.csv);;所有文件 (*.*)"
            )
        if not filepath:
            return
        save_last_dir('stats_csv', filepath)
        try:
            rows = self._collect_rows(self._stats, self._file_basename)
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['统计项', '数值'])
                for kind, label, value in rows:
                    writer.writerow([label, value if kind == 'data' else ''])
            QMessageBox.information(self, "完成", f"已导出统计CSV到:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")

    @staticmethod
    def _build_stats_report(stats, file_basename):
        """构建统计报告文本"""
        lines = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("           GGA 报文精度统计分析报告")
        lines.append(sep)
        lines.append(f"\n数据文件: {file_basename}")
        lines.append(f"GGA报文总数: {stats['horizontal']['count']}")
        ref = stats['reference']
        lines.append(f"参考点: 纬度={ref['lat']:.8f}°, 经度={ref['lon']:.8f}°")
        if ref['alt'] is not None:
            lines.append(f"        高度={ref['alt']:.3f}m")

        h = stats['horizontal']
        lines.append("\n" + "-" * 60)
        lines.append("【水平方向统计】")
        lines.append("-" * 60)
        lines.append("\n--- 经纬度统计 ---")
        lines.append(f"  纬度均值:   {h['lat_mean']:.8f}°")
        lines.append(f"  纬度中位数: {h['lat_median']:.8f}°")
        lines.append(f"  纬度标准差: {h['lat_std']:.8f}°")
        lines.append(f"  纬度范围:   {h['lat_range']:.8f}° ({h['lat_min']:.8f} ~ {h['lat_max']:.8f})")
        lines.append(f"\n  经度均值:   {h['lon_mean']:.8f}°")
        lines.append(f"  经度中位数: {h['lon_median']:.8f}°")
        lines.append(f"  经度标准差: {h['lon_std']:.8f}°")
        lines.append(f"  经度范围:   {h['lon_range']:.8f}° ({h['lon_min']:.8f} ~ {h['lon_max']:.8f})")

        lines.append("\n--- ENU方向统计 ---")
        lines.append(f"  东向(E)均值:     {h['east_mean']:.4f} m")
        lines.append(f"  东向(E)中位数:   {h['east_median']:.4f} m")
        lines.append(f"  东向(E)标准差:   {h['east_std']:.4f} m")
        lines.append(f"  东向(E)RMS:      {h['east_rms']:.4f} m")
        lines.append(f"\n  北向(N)均值:     {h['north_mean']:.4f} m")
        lines.append(f"  北向(N)中位数:   {h['north_median']:.4f} m")
        lines.append(f"  北向(N)标准差:   {h['north_std']:.4f} m")
        lines.append(f"  北向(N)RMS:      {h['north_rms']:.4f} m")

        lines.append("\n--- 水平误差统计 ---")
        lines.append(f"  水平误差均值:    {h['horiz_error_mean']:.4f} m")
        lines.append(f"  水平误差中位数:  {h['horiz_error_median']:.4f} m")
        lines.append(f"  水平误差标准差:  {h['horiz_error_std']:.4f} m")
        lines.append(f"  水平误差RMS:     {h['horiz_error_rms']:.4f} m")
        lines.append(f"  水平误差最大值:  {h['horiz_error_max']:.4f} m")
        lines.append(f"  水平误差最小值:  {h['horiz_error_min']:.4f} m")
        lines.append(f"\n  CEP50 (中位数):  {h['cep50']:.4f} m")
        lines.append(f"  CEP68:           {h['cep68']:.4f} m")
        lines.append(f"  CEP95:           {h['cep95']:.4f} m")
        lines.append(f"  CEP99:           {h['cep99']:.4f} m")
        lines.append(f"  DRMS:            {h['drms']:.4f} m")
        lines.append(f"  2DRMS:           {h['2drms']:.4f} m")

        v = stats['vertical']
        lines.append("\n" + "-" * 60)
        lines.append("【垂直方向统计】")
        lines.append("-" * 60)
        if 'error' not in v:
            lines.append(f"\n  高度均值:        {v['alt_mean']:.3f} m")
            lines.append(f"  高度中位数:      {v['alt_median']:.3f} m")
            lines.append(f"  高度标准差:      {v['alt_std']:.3f} m")
            lines.append(f"  高度范围:        {v['alt_range']:.3f} m ({v['alt_min']:.3f} ~ {v['alt_max']:.3f})")
            lines.append(f"\n  垂直误差均值:    {v['vert_error_mean']:.3f} m")
            lines.append(f"  垂直误差中位数:  {v['vert_error_median']:.3f} m")
            lines.append(f"  垂直误差标准差:  {v['vert_error_std']:.3f} m")
            lines.append(f"  垂直误差RMS:     {v['vert_error_rms']:.3f} m")
            lines.append(f"  垂直误差最大值:  {v['vert_error_max']:.3f} m")
            lines.append(f"\n  VCEP68:          {v['vert_cep68']:.3f} m")
            lines.append(f"  VCEP95:          {v['vert_cep95']:.3f} m")
            lines.append(f"  VCEP99:          {v['vert_cep99']:.3f} m")
        else:
            lines.append(f"\n  {v['error']}")

        o = stats['other']
        if o:
            lines.append("\n" + "-" * 60)
            lines.append("【其他信息】")
            lines.append("-" * 60)
            if 'hdop_mean' in o:
                lines.append(f"\n  HDOP均值: {o['hdop_mean']:.2f}, 中位数: {o['hdop_median']:.2f}, 范围: {o['hdop_min']:.2f}~{o['hdop_max']:.2f}")
            if 'sats_mean' in o:
                lines.append(f"  卫星数均值: {o['sats_mean']:.1f}, 中位数: {o['sats_median']:.1f}, 范围: {o['sats_min']:.0f}~{o['sats_max']:.0f}")

        ex = stats.get('outliers')
        if ex:
            lines.append("\n" + "-" * 60)
            lines.append("【异常值统计】")
            lines.append("-" * 60)
            lines.append(f"  判定阈值: 水平距离 > {ex['threshold_h'] * 100:.1f}cm 或 "
                         f"高程误差 > {ex['threshold_v'] * 100:.1f}cm")
            lines.append(f"  有效样本数: {ex['n_valid']}")
            lines.append(f"  异常样本数: {ex['n_outliers']} ({ex['rate'] * 100:.2f}%)")

        lines.append("\n" + sep)
        lines.append("注: CEP = Circular Error Probable (圆概率误差)")
        lines.append("    DRMS = Distance Root Mean Square")
        lines.append("    参考点为经纬度中位数，ENU为本地坐标系")
        lines.append(sep)

        return '\n'.join(lines)


class VisualizationDialog(QDialog):
    """GGA 精度可视化对话框（水平散点 / 北东高程箱型图 / 时间序列）"""

    def __init__(self, gga_msgs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GGA精度可视化")
        self.resize(1300, 950)
        self.valid = False

        if len(gga_msgs) < 2:
            QMessageBox.warning(self, "提示",
                                f"有效GGA报文不足（当前{len(gga_msgs)}条，至少需要2条）")
            return

        stats = GGAStatistics.compute_stats(gga_msgs)
        if 'error' in stats:
            QMessageBox.warning(self, "提示", stats['error'])
            return

        self.valid = True
        self._fig = self._build_figure(stats)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        canvas = FigureCanvas(self._fig)
        layout.addWidget(canvas, stretch=1)

        toolbar = ChineseNavToolbar(canvas, self)
        layout.addWidget(toolbar)

        # 保存按钮
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("💾 保存图片")
        btn_close = QPushButton("关闭")
        btn_layout.addWidget(btn_save)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        btn_save.clicked.connect(self._save_figure)
        btn_close.clicked.connect(self.accept)

    def _save_figure(self):
        """保存可视化图片（PNG/PDF/SVG）"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存图片",
            os.path.join(last_dir('figure'),
                         f"gga_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
            "PNG图片 (*.png);;PDF文档 (*.pdf);;SVG矢量图 (*.svg)"
        )
        if filepath:
            save_last_dir('figure', filepath)
            self._fig.savefig(filepath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "完成", f"图片已保存到:\n{filepath}")

    @staticmethod
    def _build_figure(stats):
        """构建4子图 Figure，返回 matplotlib Figure"""
        raw = stats['raw']
        h = stats['horizontal']

        fig = Figure(figsize=(14, 12), dpi=100)

        # 子图1: 水平散点图（ENU坐标），异常点红色突出
        ax1 = fig.add_subplot(2, 2, 1)
        e = raw['eastings']
        n = raw['northings']
        h_err = raw['horiz_errors']
        outliers = raw.get('outliers')
        if outliers is not None and outliers.any():
            normal = ~outliers
            scatter = ax1.scatter(e[normal], n[normal],
                                  c=h_err[normal], cmap='viridis',
                                  s=15, alpha=0.6, edgecolors='none')
            ax1.scatter(e[outliers], n[outliers],
                        c='red', s=45, alpha=0.95, marker='o',
                        edgecolors='black', linewidths=0.5,
                        label=f'异常值 ({int(outliers.sum())}点)')
        else:
            scatter = ax1.scatter(e, n,
                                  c=h_err, cmap='viridis',
                                  s=15, alpha=0.6, edgecolors='none')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

        theta = np.linspace(0, 2 * np.pi, 100)
        for cep_val, style, label in [
            (h['cep50'], 'g--', f'CEP50={h["cep50"]:.3f}m'),
            (h['cep95'], 'r-', f'CEP95={h["cep95"]:.3f}m')
        ]:
            ax1.plot(cep_val * np.cos(theta), cep_val * np.sin(theta),
                     style, linewidth=1.5, label=label)

        ax1.set_xlabel('东向 (m)', fontsize=10)
        ax1.set_ylabel('北向 (m)', fontsize=10)
        ax1.set_title(f'水平散点图 (ENU) - 共{h["count"]}点', fontsize=11)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        fig.colorbar(scatter, ax=ax1, label='误差 (m)')

        # 子图2: 北东水平与高程精度箱型图
        ax2 = fig.add_subplot(2, 2, 2)
        # 数据: 北向误差、东向误差、高程误差
        box_data = []
        box_labels = []
        if len(raw['northings']) > 0:
            box_data.append(raw['northings'])
            box_labels.append('北向误差')
        if len(raw['eastings']) > 0:
            box_data.append(raw['eastings'])
            box_labels.append('东向误差')
        if len(raw['vert_errors']) > 0:
            box_data.append(raw['vert_errors'])
            box_labels.append('高程误差')

        # whis=[0,100]: 上下边缘(须)对应最大最小值
        # 箱体上下线=Q3/Q1，箱内横线=中位数
        bp = ax2.boxplot(
            box_data, whis=[0, 100], showfliers=False,
            patch_artist=True, widths=0.5,
            medianprops=dict(color='orange', linewidth=2),
            boxprops=dict(edgecolor='#333333', linewidth=1.2),
            whiskerprops=dict(color='#333333', linewidth=1.2),
            capprops=dict(color='#333333', linewidth=1.2),
        )
        # 各方向箱体配色
        box_colors = ['#4a90d9', '#5cb85c', '#d9534f']
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        # 直接在图上标注各箱体五数概括（最小/Q1/Q3/最大；中位数由箱内橙线体现，不标注文字），
        # 标在箱体右侧对应高度上，避免角注遮挡图形
        for i, data in enumerate(box_data, 1):
            q1, _, q3 = np.percentile(data, [25, 50, 75])
            dmin, dmax = float(np.min(data)), float(np.max(data))
            x_ann = i + 0.10  # 箱体右侧
            ax2.text(x_ann, dmin, f'{dmin:+.3f}', fontsize=8,
                     va='bottom', ha='left', color='#333333')
            ax2.text(x_ann, q1, f'Q1 {q1:+.3f}', fontsize=8,
                     va='center', ha='left', color='#333333')
            ax2.text(x_ann, q3, f'Q3 {q3:+.3f}', fontsize=8,
                     va='center', ha='left', color='#333333')
            ax2.text(x_ann, dmax, f'{dmax:+.3f}', fontsize=8,
                     va='top', ha='left', color='#333333')

        ax2.set_xticks(range(1, len(box_labels) + 1))
        ax2.set_xticklabels(box_labels, fontsize=10)
        ax2.set_ylabel('误差 (m)', fontsize=10)
        ax2.set_title('北东水平与高程精度箱型图', fontsize=11)
        ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
        ax2.grid(True, alpha=0.3, axis='y')

        # 子图3: 时间序列 - 水平误差
        ax3 = fig.add_subplot(2, 2, 3)
        times = raw['times']
        ax3.plot(times, raw['horiz_errors'], 'b-', linewidth=0.5, alpha=0.7)
        ax3.axhline(y=h['horiz_error_mean'], color='orange', linestyle='-',
                     label=f'均值={h["horiz_error_mean"]:.3f}m')
        ax3.axhline(y=h['cep95'], color='red', linestyle='--',
                     label=f'CEP95={h["cep95"]:.3f}m')
        ax3.fill_between(times, 0, raw['horiz_errors'], alpha=0.2)
        ax3.set_xlabel('样本序号', fontsize=10)
        ax3.set_ylabel('水平误差 (m)', fontsize=10)
        ax3.set_title('水平误差时间序列', fontsize=11)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 子图4: 垂直误差
        ax4 = fig.add_subplot(2, 2, 4)
        if len(raw['vert_errors']) > 0:
            v = stats['vertical']
            n_vert = len(raw['vert_errors'])
            ax4.plot(times[:n_vert], raw['vert_errors'], 'g-', linewidth=0.5, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax4.axhline(y=v['vert_error_mean'], color='orange', linestyle='-',
                         label=f'均值={v["vert_error_mean"]:.3f}m')
            ax4.axhline(y=v['vert_cep95'], color='red', linestyle='--',
                         label=f'VCEP95={v["vert_cep95"]:.3f}m')
            ax4.axhline(y=-v['vert_cep95'], color='red', linestyle='--')
            ax4.fill_between(times[:n_vert], 0, raw['vert_errors'], alpha=0.2, color='green')
            ax4.set_xlabel('样本序号', fontsize=10)
            ax4.set_ylabel('垂直误差 (m)', fontsize=10)
            ax4.set_title('垂直误差时间序列', fontsize=11)
            ax4.legend(loc='upper right', fontsize=8)
        else:
            ax4.text(0.5, 0.5, '无高度数据', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=14)
            ax4.set_title('垂直误差时间序列', fontsize=11)
        ax4.grid(True, alpha=0.3)

        # 加大行间(h_pad)/列间(w_pad)间距，避免上一行X轴标签与下一行标题重叠
        fig.tight_layout(pad=2.0, h_pad=4.0, w_pad=3.0)

        return fig
