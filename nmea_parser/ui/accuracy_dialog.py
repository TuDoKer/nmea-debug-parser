# -*- coding: utf-8 -*-
"""精度评估对话框

把「设备解算序列」与「参考真值」比对，给出外符合/内符合精度。

四种参考来源在本界面上各有自己的参数区（QStackedWidget 切换）：
  已知点 / 后处理 .pos / 另一台设备 / 自建参考

界面刻意把**口径**摆在显眼位置：
  * 参考是"绝对真值"还是"相对参考"会直接写在顶部与报告里 ——
    自建参考与设备互比只能说明一致性/重复性，绝不能当精度报出去。
  * 粗差剔除只作对照展示，主结果永远包含全部历元。
"""

import os
from datetime import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QFileDialog, QMessageBox, QTabWidget, QWidget, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView, QStackedWidget,
    QLineEdit, QFormLayout, QGroupBox, QCheckBox, QDoubleSpinBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from .. import accuracy, solfile, timebase, track
from ..reference import (KnownPointRef, PeerDeviceRef, PosFileRef,
                         SelfMeanRef, normalize_require)
from ..solution import SolutionSeries
from ..config import last_dir, save_last_dir
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ..toolbar import ChineseNavToolbar

REF_TYPES = [
    ('known', '已知控制点'),
    ('pos', '后处理 .pos'),
    ('peer', '另一台设备'),
    ('self', '自建参考（自身均值）'),
]

REQ_CHOICES = [
    ('仅固定解 (Q=4)', 'fix'),
    ('固定 + 浮动 (Q=4,5)', 'fix_float'),
    ('所有有效解 (Q=1,2,4,5)', 'valid'),
    ('不限', None),
]


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class AccuracyDialog(QDialog):
    """精度评估

    参数
    ----
    device_series : SolutionSeries|None
        设备解算序列；为 None 时由界面上选择 .dbg 现场解析
    device_label : str
    parent : QWidget
    """

    def __init__(self, device_series=None, device_label='', parent=None):
        super().__init__(parent)
        self.setWindowTitle('RTK 精度评估')
        self.resize(1480, 1000)
        self.valid = False

        self._series = device_series
        self._device_label = device_label
        self._result = None
        self._pairs = None
        self._fig = None

        self._build_ui()
        if device_series is not None and len(device_series):
            self._lbl_dev.setText(
                f'设备解算序列：{device_label or "（已载入）"}　'
                f'{len(device_series)} 个历元　'
                f'{device_series.t[0]:%Y-%m-%d %H:%M:%S} ~ '
                f'{device_series.t[-1]:%H:%M:%S}')
            self.valid = True
        else:
            self.valid = True

    # ==================== UI ====================

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ---- 设备侧 ----
        dev_box = QGroupBox('设备解算序列')
        dv = QHBoxLayout(dev_box)
        self._lbl_dev = QLabel('尚未载入 .dbg 文件')
        self._lbl_dev.setFont(QFont('Microsoft YaHei', 9))
        dv.addWidget(self._lbl_dev, stretch=1)
        btn_dev = QPushButton('📂 选择 .dbg')
        btn_dev.clicked.connect(self._pick_device)
        dv.addWidget(btn_dev)
        root.addWidget(dev_box)

        # ---- 参考来源 ----
        ref_box = QGroupBox('参考真值来源')
        rv = QVBoxLayout(ref_box)
        row = QHBoxLayout()
        row.addWidget(QLabel('类型:'))
        self.cmb_ref = QComboBox()
        for key, name in REF_TYPES:
            self.cmb_ref.addItem(name, key)
        self.cmb_ref.currentIndexChanged.connect(self._on_ref_changed)
        row.addWidget(self.cmb_ref)
        row.addStretch(1)
        self._lbl_ref_hint = QLabel()
        self._lbl_ref_hint.setStyleSheet('color:#c77e00;')
        row.addWidget(self._lbl_ref_hint)
        rv.addLayout(row)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._page_known())
        self.stack.addWidget(self._page_pos())
        self.stack.addWidget(self._page_peer())
        self.stack.addWidget(self._page_self())
        rv.addWidget(self.stack)
        root.addWidget(ref_box)

        # ---- 公共选项 ----
        opt = QHBoxLayout()
        opt.addWidget(QLabel('设备解算质量要求:'))
        self.cmb_req = QComboBox()
        for name, key in REQ_CHOICES:
            self.cmb_req.addItem(name, key)
        self.cmb_req.setToolTip(
            'Q 不是有序量：4 固定、5 浮动。\n'
            '「仅固定解」只取 Q==4，不会把浮动解算进来。')
        opt.addWidget(self.cmb_req)
        opt.addWidget(QLabel('时间对齐容差:'))
        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setRange(0.05, 60.0)
        self.spin_dt.setSingleStep(0.5)
        self.spin_dt.setValue(1.5)
        self.spin_dt.setSuffix(' s')
        opt.addWidget(self.spin_dt)
        opt.addWidget(QLabel('粗差剔除:'))
        self.cmb_reject = QComboBox()
        self.cmb_reject.addItem('不剔除（推荐）', 'none')
        self.cmb_reject.addItem('3σ (MAD)', 'sigma')
        self.cmb_reject.addItem('1.5×IQR', 'iqr')
        opt.addWidget(self.cmb_reject)
        opt.addStretch(1)
        self.btn_run = QPushButton('▶ 开始评估')
        self.btn_run.setStyleSheet('font-weight:bold;')
        self.btn_run.clicked.connect(self._run)
        opt.addWidget(self.btn_run)
        root.addLayout(opt)

        # ---- 结果页 ----
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)
        self.tabs.addTab(self._build_metrics_page(), '精度指标')
        self.tabs.addTab(self._build_fig_page(), '误差图表')
        self.tabs.addTab(self._build_track_page(), '动态轨迹')

        bar = QHBoxLayout()
        self.btn_save = QPushButton('💾 保存图片')
        self.btn_report = QPushButton('📄 导出报告')
        self.btn_pos = QPushButton('📤 导出 .pos')
        self.btn_pos.setToolTip(
            '把设备解算序列写成 RTKPOST 兼容的 .pos，\n'
            '可交给 RTKPOST 复算或与其他软件比对。\n'
            '时间按 GPST 写（RTKPOST 惯例），头部注明。')
        btn_close = QPushButton('关闭')
        bar.addWidget(self.btn_save)
        bar.addWidget(self.btn_report)
        bar.addWidget(self.btn_pos)
        bar.addStretch(1)
        bar.addWidget(btn_close)
        root.addLayout(bar)
        self.btn_save.clicked.connect(self._save_figure)
        self.btn_report.clicked.connect(self._export_report)
        self.btn_pos.clicked.connect(self._export_pos)
        btn_close.clicked.connect(self.accept)
        self.btn_save.setEnabled(False)
        self.btn_report.setEnabled(False)
        self.btn_pos.setEnabled(False)

        self._on_ref_changed()

    def _page_known(self):
        p = QWidget()
        f = QFormLayout(p)
        self.cmb_known_mode = QComboBox()
        self.cmb_known_mode.addItem('经纬高（度 / 米）', 'blh')
        self.cmb_known_mode.addItem('ECEF 直角坐标（米）', 'ecef')
        self.cmb_known_mode.addItem('高斯平面坐标（X 北 / Y 东）', 'gauss')
        self.cmb_known_mode.currentIndexChanged.connect(self._on_known_mode)
        f.addRow('输入方式:', self.cmb_known_mode)
        self.ed_a = QLineEdit('0.000000000')
        self.ed_b = QLineEdit('0.000000000')
        self.ed_c = QLineEdit('15.7207')
        self.lbl_a = QLabel('纬度(°)')
        self.lbl_b = QLabel('经度(°)')
        self.lbl_c = QLabel('大地高(m)')
        f.addRow(self.lbl_a, self.ed_a)
        f.addRow(self.lbl_b, self.ed_b)
        f.addRow(self.lbl_c, self.ed_c)

        # 高斯平面坐标专用：带号与带宽
        self.ed_zone = QLineEdit('40')
        self.cmb_zone_width = QComboBox()
        self.cmb_zone_width.addItem('3° 带', 3)
        self.cmb_zone_width.addItem('6° 带', 6)
        self.row_gauss = QWidget()
        rl = QHBoxLayout(self.row_gauss)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel('带号'))
        rl.addWidget(self.ed_zone)
        rl.addWidget(self.cmb_zone_width)
        rl.addStretch(1)
        self.lbl_gauss = QLabel('带号 / 带宽:')
        f.addRow(self.lbl_gauss, self.row_gauss)

        self.ed_ant = QLineEdit('0.0')
        f.addRow('天线高改正(m):', self.ed_ant)
        self.ed_name = QLineEdit('已知点')
        f.addRow('点名:', self.ed_name)
        note = QLabel('！中国已知点常给 CGCS2000 高斯平面坐标（如 '
                      'X=3453657.5054  Y=40620397.4442，40 带 3°）。\n'
                      '  带号 40 对应中央子午线 120°；注意 6° 带用的是国内'
                      '编号（int(经度/6)+1），与 UTM 编号不同。\n'
                      '  东坐标含带号前缀时可直接填（本工具会去掉前缀）。\n'
                      '  天线高为「相位中心 − 标石面」；留 0 表示已知点即相位中心。')
        note.setStyleSheet('color:#5f5e5a;')
        f.addRow(note)
        return p

    def _page_pos(self):
        p = QWidget()
        f = QFormLayout(p)
        self.ed_pos = QLineEdit()
        self.ed_pos.setPlaceholderText('RTKPOST / rnx2rtkp 输出的 .pos 或 .llh')
        btn = QPushButton('浏览…')
        btn.clicked.connect(lambda: self._pick_file(self.ed_pos, 'pos'))
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.ed_pos, stretch=1)
        rl.addWidget(btn)
        f.addRow('文件:', row)
        self.chk_gpst = QCheckBox('时间标签是 GPST（RTKPOST 默认）')
        self.chk_gpst.setChecked(True)
        self.chk_gpst.setToolTip(
            'GPST 与 UTC 差 18 秒闰秒。勾选则读入时自动减 18s 与设备对齐；\n'
            '若你的 .pos 已是 UTC 则取消勾选。')
        f.addRow(self.chk_gpst)
        note = QLabel('参考只取**固定解**（Q=4）：拿浮动解当真值会把参考自身的'
                      '误差算到设备头上。')
        note.setStyleSheet('color:#5f5e5a;')
        f.addRow(note)
        return p

    def _page_peer(self):
        p = QWidget()
        f = QFormLayout(p)
        self.ed_peer = QLineEdit()
        self.ed_peer.setPlaceholderText('另一台设备的 .dbg')
        btn = QPushButton('浏览…')
        btn.clicked.connect(lambda: self._pick_file(self.ed_peer, 'dbg'))
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.ed_peer, stretch=1)
        rl.addWidget(btn)
        f.addRow('文件:', row)
        self.ed_peer_label = QLineEdit('另一台设备')
        f.addRow('标签:', self.ed_peer_label)
        note = QLabel('⚠ 两台设备可能同时有偏，只能评估**一致性/相对精度**，'
                      '不能当作绝对精度。')
        note.setStyleSheet('color:#c77e00;')
        f.addRow(note)
        return p

    def _page_self(self):
        p = QWidget()
        f = QFormLayout(p)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(1.0, 6.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(3.0)
        f.addRow('粗差剔除倍数:', self.spin_sigma)
        note = QLabel('⚠ 用设备自身长时间观测的稳健均值作参考，'
                      '衡量的是**重复性**，不含任何绝对基准 ——\n'
                      '设备可能每次都落在同一个错的地方。'
                      '要评估精度请用已知点或后处理结果。')
        note.setStyleSheet('color:#c77e00; font-weight:bold;')
        f.addRow(note)
        return p

    def _build_metrics_page(self):
        p = QWidget()
        v = QVBoxLayout(p)
        self.lbl_refinfo = QLabel('尚未评估')
        self.lbl_refinfo.setWordWrap(True)
        self.lbl_refinfo.setFont(QFont('Microsoft YaHei', 9))
        v.addWidget(self.lbl_refinfo)
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(['指标', '数值', '说明'])
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hdr = self.tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        v.addWidget(self.tbl, stretch=1)
        return p

    def _build_fig_page(self):
        p = QWidget()
        v = QVBoxLayout(p)
        v.setContentsMargins(2, 2, 2, 2)
        fig = self._fig = Figure(figsize=(15, 10), dpi=100)
        gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95],
                              hspace=0.42, wspace=0.28)
        self.ax_e = fig.add_subplot(gs[0, 0])
        self.ax_n = fig.add_subplot(gs[0, 1])
        self.ax_u = fig.add_subplot(gs[0, 2])
        self.ax_sc = fig.add_subplot(gs[1, 0:2])
        self.ax_cdf = fig.add_subplot(gs[1, 2])
        self.ax_bins = fig.add_subplot(gs[2, 0:2])
        self.ax_time = fig.add_subplot(gs[2, 2])
        canvas = FigureCanvas(fig)
        v.addWidget(canvas, stretch=1)
        v.addWidget(ChineseNavToolbar(canvas, self))
        return p

    # ==================== 交互 ====================

    @staticmethod
    def _busy():
        """忙碌光标上下文

        解析一个 .dbg 要一两秒，界面在这期间完全无响应也没有任何提示，
        用户很容易以为卡死了又点一次。用光标 + 阻塞输入是最轻的反馈；
        真要接大文件再升级成后台线程 + 进度条。
        """
        import contextlib
        from PyQt5.QtWidgets import QApplication

        @contextlib.contextmanager
        def _ctx():
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                yield
            finally:
                QApplication.restoreOverrideCursor()
        return _ctx()

    def _on_ref_changed(self, *_):
        i = self.cmb_ref.currentIndex()
        self.stack.setCurrentIndex(max(0, i))
        self._lbl_ref_hint.setText(
            '' if i in (0, 1) else
            '⚠ 相对参考：只能评估一致性，不能当精度')

    def _on_known_mode(self, *_):
        mode = self.cmb_known_mode.currentData()
        is_gauss = mode == 'gauss'
        # 高斯模式的「带号/带宽」行只在需要时出现
        self.row_gauss.setVisible(is_gauss)
        self.lbl_gauss.setVisible(is_gauss)
        if mode == 'ecef':
            self.lbl_a.setText('X(m)')
            self.lbl_b.setText('Y(m)')
            self.lbl_c.setText('Z(m)')
            self.ed_a.setText('-2833913.2547')
            self.ed_b.setText('4667714.8755')
            self.ed_c.setText('3284735.9227')
        elif is_gauss:
            self.lbl_a.setText('X 北坐标(m)')
            self.lbl_b.setText('Y 东坐标(m)')
            self.lbl_c.setText('大地高(m)')
            self.ed_a.setText('3453657.5054')
            self.ed_b.setText('40620397.4442')
            self.ed_c.setText('15.7207')
            self.ed_zone.setText('40')
            self.cmb_zone_width.setCurrentIndex(0)
        else:
            self.lbl_a.setText('纬度(°)')
            self.lbl_b.setText('经度(°)')
            self.lbl_c.setText('大地高(m)')
            self.ed_a.setText('0.000000000')
            self.ed_b.setText('0.000000000')
            self.ed_c.setText('15.7207')

    def _pick_file(self, edit, kind):
        if kind == 'pos':
            path, _ = QFileDialog.getOpenFileName(
                self, '选择后处理结果', last_dir('open_pos'),
                '解算结果 (*.pos *.llh *.stat);;所有文件 (*.*)')
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, '选择设备日志', last_dir('open_dbg'),
                'Debug文件 (*.dbg);;所有文件 (*.*)')
        if path:
            edit.setText(path)
            save_last_dir('open_dbg' if kind == 'dbg' else 'open_pos', path)

    def _pick_device(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择设备 .dbg', last_dir('open_dbg'),
            'Debug文件 (*.dbg);;所有文件 (*.*)')
        if not path:
            return
        save_last_dir('open_dbg', path)
        try:
            self._load_device(path)
        except Exception as e:                                  # noqa: BLE001
            QMessageBox.critical(self, '解析失败', f'{type(e).__name__}: {e}')

    def _load_device(self, path):
        with self._busy():
            s = self._load_series_from(path)
        self._series = s
        self._device_label = os.path.basename(path)
        self._lbl_dev.setText(
            f'设备解算序列：{self._device_label}　{len(s)} 个历元　'
            f'{s.t[0]:%Y-%m-%d %H:%M:%S} ~ {s.t[-1]:%H:%M:%S}　'
            f'（GGA 原始 {s.n_gga_raw} 条，去重 {s.n_dup_gga} 条）')
        # 轨迹页在「无序列」时是占位标签，载入后必须重建才能出图
        if getattr(self, '_track_fig', None) is None:
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == '动态轨迹':
                    self.tabs.removeTab(i)
                    self.tabs.insertTab(i, self._build_track_page(),
                                        '动态轨迹')
                    break

    def _collect_reference(self):
        """按界面选择构建参考源；失败抛 ValueError"""
        key = self.cmb_ref.currentData()
        if key == 'known':
            mode = self.cmb_known_mode.currentData()
            a = _f(self.ed_a.text())
            b = _f(self.ed_b.text())
            c = _f(self.ed_c.text())
            if None in (a, b, c):
                raise ValueError('已知点坐标要填满三个数值')
            ant = _f(self.ed_ant.text(), 0.0) or 0.0
            name = self.ed_name.text().strip() or '已知点'
            if mode == 'ecef':
                return KnownPointRef(xyz=(a, b, c), name=name,
                                     antenna_h=ant,
                                     source_note='ECEF 输入')
            if mode == 'gauss':
                zone = _f(self.ed_zone.text())
                if zone is None or int(zone) <= 0:
                    raise ValueError(
                        '高斯平面坐标必须给有效带号（如上海的 40 带）')
                width = self.cmb_zone_width.currentData()
                return KnownPointRef(gauss=(a, b, int(zone), width), h=c,
                                     name=name, antenna_h=ant)
            return KnownPointRef(lat=a, lon=b, h=c, name=name,
                                 antenna_h=ant)
        if key == 'pos':
            path = self.ed_pos.text().strip()
            if not path or not os.path.exists(path):
                raise ValueError('请选择存在的 .pos 文件')
            series, meta = solfile.read_pos(
                path, time_is_gpst=self.chk_gpst.isChecked())
            if len(series) == 0:
                raise ValueError('该 .pos 没有解析到历元')
            return PosFileRef(series, filename=os.path.basename(path),
                              max_dt=self.spin_dt.value())
        if key == 'peer':
            path = self.ed_peer.text().strip()
            if not path or not os.path.exists(path):
                raise ValueError('请选择另一台设备的 .dbg 文件')
            from ..parser import NMEAParser
            msgs = NMEAParser.parse_file(path)['nmea_messages']
            timebase.build_time_axis(msgs)
            s = SolutionSeries.from_messages(msgs)
            if len(s) == 0:
                raise ValueError('对比设备文件没有解析出解算历元')
            return PeerDeviceRef(s, label=self.ed_peer_label.text().strip()
                                 or '另一台设备',
                                 max_dt=self.spin_dt.value())
        if key == 'self':
            if self._series is None:
                raise ValueError('请先载入设备 .dbg')
            return SelfMeanRef(self._series, require_q='fix',
                               sigma_k=self.spin_sigma.value())
        raise ValueError(f'未知的参考类型 {key}')

    def _run(self):
        if self._series is None or len(self._series) == 0:
            QMessageBox.warning(self, '提示', '请先载入设备 .dbg 文件。')
            return
        try:
            with self._busy():
                ref = self._collect_reference()
        except Exception as e:                                  # noqa: BLE001
            QMessageBox.warning(self, '参考设置有问题', str(e))
            return

        req = self.cmb_req.currentData()
        max_dt = self.spin_dt.value()
        try:
            pairs, ast = accuracy.align(self._series, ref, max_dt=max_dt,
                                        require=req)
        except Exception as e:                                  # noqa: BLE001
            import traceback
            QMessageBox.critical(
                self, '对齐失败',
                f'{type(e).__name__}: {e}\n\n'
                f'--- 调用栈（供排查）---\n'
                f'{traceback.format_exc()[-1200:]}')
            return

        if not pairs:
            QMessageBox.warning(
                self, '没有可比对的历元',
                f'共 {ast["n_dev"]} 个设备历元：\n'
                f'  被质量要求过滤 {ast["n_filtered"]} 个\n'
                f'  历元本身无坐标 {ast.get("n_nopos", 0)} 个'
                f'（无定位输出，如 GGA 质量 0）\n'
                f'  找不到对应参考 {ast["n_no_ref"]} 个\n\n'
                f'参考说明：{ref.info()}\n\n'
                f'常见原因：参考与设备时段不重叠、参考无固定解、'
                f'时间容差过小（当前 {max_dt:g}s）。')
            return

        self._result = accuracy.accuracy_stats(
            pairs, reject=self.cmb_reject.currentData(),
            reject_k=3.0, ref_absolute=ref.is_absolute(),
            ref_kind=ref.kind, ref_info=ref.info(), align_stats=ast)
        self._ref = ref
        self._pairs = pairs
        self._fill_metrics(ref, ast)
        self._draw_all(ref)
        self.btn_save.setEnabled(True)
        self.btn_report.setEnabled(True)
        self.btn_pos.setEnabled(True)

    # ==================== 结果展示 ====================

    def _fill_metrics(self, ref, ast):
        st = self._result
        warn = ('' if st['ref_absolute'] else
                '　⚠ 这是相对参考，只有一致性/重复性，不是绝对精度')
        self.lbl_refinfo.setText(
            f"<b>参考：</b>{ref.info()}<br>"
            f"<b>对齐：</b>设备 {ast['n_dev']} 个历元 → 成功对齐 "
            f"{ast['n_aligned']} 个（过滤 {ast['n_filtered']}，"
            f"无参考 {ast['n_no_ref']}"
            + (f"，<b>无坐标 {ast.get('n_nopos', 0)}</b>"
               if ast.get('n_nopos') else '')
            + f"）；"
            f"时间差 均值 {ast['dt_mean']:.2f}s / 最大 {ast['dt_max']:.2f}s；"
            f"质量要求 "
            f"{'不限' if ast['require'] is None else ast['require']}"
            f"<span style='color:#c77e00'>{warn}</span>")

        rows = []

        def add(name, val, note=''):
            rows.append((name, val, note))

        h = st['horiz']
        add('── 水平 ──', '', '')
        add('水平误差 RMS', f"{h['rms']:.4f} m", '≈68% 落在该半径内')
        add('CEP50 / 中位', f"{h['cep50']:.4f} m", '圆概率误差 50%')
        add('CEP68', f"{h['cep68']:.4f} m", '')
        add('CEP95', f"{h['cep95']:.4f} m", '95% 分位，工程上常用')
        add('CEP99', f"{h['cep99']:.4f} m", '')
        add('最大水平误差', f"{h['max']:.4f} m", '')
        add('DRMS', f"{h['drms']:.4f} m", '√(RMS_E²+RMS_N²)')
        add('2DRMS', f"{h['2drms']:.4f} m", '约 95% 置信')
        add('3DRMS', f"{h['3drms']:.4f} m", '约 99.7% 置信')

        v = st['vert']
        add('── 垂直 ──', '', '')
        add('高程偏差(均值)', f"{v['bias']:+.4f} m", '系统性偏移，应接近 0')
        add('高程 RMS', f"{v['rms']:.4f} m", '')
        add('|高程误差| 中位', f"{v['median_abs']:.4f} m", '')
        add('|高程误差| P95', f"{v['p95_abs']:.4f} m", '')
        add('|高程误差| 最大', f"{v['max_abs']:.4f} m", '')

        s3 = st['sp3d']
        add('── 三维 ──', '', '')
        add('3D RMS', f"{s3['rms']:.4f} m", '')
        add('SEP(中位)', f"{s3['sep']:.4f} m", '球概率误差')
        add('3D P95', f"{s3['p95']:.4f} m", '')
        add('3D 最大', f"{s3['max']:.4f} m", '')

        add('── 分量 ──', '', '')
        for key, nm in (('e', '东(E)'), ('n_ax', '北(N)'), ('u', '高(U)')):
            d = st[key]
            add(f'{nm} 均值', f"{d['mean']:+.4f} m", '偏差')
            add(f'{nm} 标准差', f"{d['std']:.4f} m", '')
            add(f'{nm} RMS', f"{d['rms']:.4f} m", '')

        add('── 超限统计 ──', '', '')
        for r in st['over']:
            add(f"水平 > {r['th'] * 100:.0f} cm",
                f"{r['n_horiz']} 次 ({r['pct_horiz']:.2f}%)",
                f"垂直 > 同值 {r['n_vert']} 次")

        rj = st.get('rejected')
        add('── 粗差剔除（对照）──', '', '')
        if rj:
            add('剔除数量', f"{rj['n_rejected']} 个 ({rj['reject_pct']:.1f}%)",
                rj['rule'])
            add('剔除后水平 RMS', f"{rj['horiz']['rms']:.4f} m",
                '与上面的主结果对比看影响')
            add('剔除后 CEP95', f"{rj['horiz']['cep95']:.4f} m", '')
        else:
            add('剔除', '未启用', '主结果包含全部历元')

        self.tbl.setRowCount(len(rows))
        for i, (a, b, c) in enumerate(rows):
            for col, txt in enumerate((a, b, c)):
                it = QTableWidgetItem(str(txt))
                if a.startswith('──'):
                    it.setForeground(QColor('#185FA5'))
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                if col == 1 and b:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl.setItem(i, col, it)

    def _draw_all(self, ref):
        st = self._result
        pairs = self._pairs
        for ax in (self.ax_e, self.ax_n, self.ax_u, self.ax_sc,
                   self.ax_cdf, self.ax_bins, self.ax_time):
            ax.clear()
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

        ts = [p.t for p in pairs]
        self._plot_series(self.ax_e, ts, [p.e for p in pairs], '东向偏差 dE (m)',
                          st['e'])
        self._plot_series(self.ax_n, ts, [p.n for p in pairs], '北向偏差 dN (m)',
                          st['n_ax'])
        self._plot_series(self.ax_u, ts, [p.u for p in pairs], '高程偏差 dU (m)',
                          st['u'])

        # 散点 + 误差圆
        ax = self.ax_sc
        e = np.array([p.e for p in pairs])
        n = np.array([p.n for p in pairs])
        ax.scatter(e, n, s=8, alpha=0.45, color='#185FA5', edgecolors='none')
        rms = st['horiz']['rms']
        for k, col in ((1, '#2ca02c'), (2, '#ff7f0e'), (3, '#d62728')):
            ax.add_patch(plt_circle((0, 0), k * rms, col))
        lim = max(3.3 * rms, float(np.max(np.abs(np.concatenate([e, n])))) * 1.1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal', adjustable='box')
        ax.axhline(0, color='gray', lw=0.6)
        ax.axvline(0, color='gray', lw=0.6)
        ax.set_xlabel('东 (m)', fontsize=9)
        ax.set_ylabel('北 (m)', fontsize=9)
        ax.set_title(f"误差散点　RMS={rms:.4f} m　"
                     f"（圆为 1σ/2σ/3σ 半径）", fontsize=11)

        # 水平误差 CDF
        ax = self.ax_cdf
        hs = np.sort([p.horiz for p in pairs])
        cdf = np.arange(1, len(hs) + 1) / len(hs) * 100
        ax.plot(hs * 100, cdf, lw=1.6, color='#185FA5')
        for q, col in ((50, '#2ca02c'), (95, '#d62728')):
            xv = st['horiz'][f'cep{q}'] * 100
            ax.axvline(xv, color=col, ls='--', lw=1.0)
            ax.text(xv, 5, f'CEP{q}\n{xv:.1f}cm', fontsize=7.5, color=col,
                    ha='left')
        ax.set_xlabel('水平误差 (cm)', fontsize=9)
        ax.set_ylabel('累积占比 (%)', fontsize=9)
        ax.set_ylim(0, 102)
        ax.set_title('水平误差累积分布', fontsize=11)

        # 分档
        ax = self.ax_bins
        rows = []
        for key, nm in (('nsat', '卫星数'), ('hdop', 'HDOP'), ('age', '龄期(s)')):
            for r in accuracy.bin_by_condition(pairs, key):
                rows.append((f"{nm} {r['label']}", r['horiz_rms'] * 100, r['n']))
        if rows:
            yy = np.arange(len(rows))
            ax.barh(yy, [r[1] for r in rows], color='#85B7EB', alpha=0.95)
            ax.set_yticks(yy)
            ax.set_yticklabels([r[0] for r in rows], fontsize=8)
            ax.invert_yaxis()
            for i, r in zip(yy, rows):
                ax.text(r[1] * 1.02, i, f"{r[1]:.2f}cm (n={r[2]})",
                        va='center', fontsize=7.5)
            ax.set_xlabel('水平误差 RMS (cm)', fontsize=9)
        else:
            ax.text(0.5, 0.5, '样本不足，无法分档', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=11)
        ax.set_title('分档精度（什么条件下精度会掉）', fontsize=11)

        # 时间对齐情况
        ax = self.ax_time
        dts = [p.dt for p in pairs]
        if dts:
            ax.hist(dts, bins=min(30, max(5, len(set(dts)))), color='#97C459',
                    alpha=0.9)
            ax.set_xlabel('设备与参考的时间差 (s)', fontsize=9)
            ax.set_ylabel('历元数', fontsize=9)
            ax.set_title(f'时间对齐（最大 {max(dts):.2f}s）', fontsize=11)

        self._fig.subplots_adjust(left=0.06, right=0.985, top=0.945,
                                  bottom=0.06, hspace=0.48, wspace=0.30)
        self._fig.canvas.draw_idle()

    def _plot_series(self, ax, ts, vals, ylabel, d):
        ax.plot(ts, vals, lw=0.7, color='#185FA5', alpha=0.85)
        ax.axhline(0, color='gray', lw=0.6)
        ax.axhline(d['mean'], color='#d62728', lw=1.0, ls='--',
                   label=f"均值 {d['mean'] * 100:+.2f} cm")
        ax.axhline(d['mean'] + d['std'], color='#ff7f0e', lw=0.9, ls=':',
                   label=f"±1σ {d['std'] * 100:.2f} cm")
        ax.axhline(d['mean'] - d['std'], color='#ff7f0e', lw=0.9, ls=':')
        timebase.style_time_axis(ax, 'UTC 时间')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7.5, loc='best')
        ax.set_title(f"{ylabel.split(' ')[0]}  RMS={d['rms'] * 100:.2f} cm",
                     fontsize=10)

    # ==================== 导出 ====================

    def _build_track_page(self):
        """构建「动态轨迹」页

        静态测试看的是"一个点上重复得准不准"，动态测试看的是"沿轨迹走得
        准不准"，两者指标完全不同，所以单独一页。

        内容：轨迹图（ENU，等比例）、速度/航向时序、动静分段条带、统计。
        可以在下面选另一条 .dbg 比较**重复线重合度**（横向偏差）。
        """
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(2, 2, 2, 2)

        self._track_fig = None
        if self._series is None or len(self._series) == 0:
            v.addWidget(QLabel('尚未载入设备 .dbg 文件，无法分析轨迹。'))
            return page

        bar = QHBoxLayout()
        bar.addWidget(QLabel('静止判定阈值:'))
        self.spin_static = QDoubleSpinBox()
        self.spin_static.setRange(0.01, 5.0)
        self.spin_static.setSingleStep(0.05)
        self.spin_static.setValue(0.05)
        self.spin_static.setSuffix(' m/s')
        self.spin_static.setToolTip(
            '速度低于该值判为静止。默认 0.05 m/s 能挡住 RTK 静态漂移\n'
            '造成的伪运动，又不至于把慢速推行判成静止。')
        bar.addWidget(self.spin_static)
        bar.addSpacing(16)
        bar.addWidget(QLabel('对比轨迹:'))
        self.ed_cmp_track = QLineEdit()
        self.ed_cmp_track.setPlaceholderText('可选：另一条 .dbg，用于重复线重合度')
        bar.addWidget(self.ed_cmp_track, stretch=1)
        btn_cmp = QPushButton('浏览…')
        btn_cmp.clicked.connect(
            lambda: self._pick_file(self.ed_cmp_track, 'dbg'))
        bar.addWidget(btn_cmp)
        btn_go = QPushButton('▶ 分析轨迹')
        btn_go.setStyleSheet('font-weight:bold;')
        btn_go.clicked.connect(self._run_track)
        bar.addWidget(btn_go)
        v.addLayout(bar)

        fig = self._track_fig = Figure(figsize=(15, 9.5), dpi=100)
        # 高度比按内容给：轨迹图最高；文字行只需要放 4~5 行，给最小权重，
        # 否则会在文字与条带之间留一大片空白
        gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.35],
                              height_ratios=[1.7, 0.62, 0.45],
                              hspace=0.42, wspace=0.22)
        self._ax_track = fig.add_subplot(gs[0, 0])
        self._ax_spd = fig.add_subplot(gs[0, 1])
        self._ax_alt = fig.add_subplot(gs[1, :])
        self._ax_seg = fig.add_subplot(gs[2, :])

        canvas = FigureCanvas(fig)
        v.addWidget(canvas, stretch=1)
        v.addWidget(ChineseNavToolbar(canvas, self))
        # 载入后自动画一次，用户不必先点按钮
        self._run_track()
        return page

    def _load_series_from(self, path):
        from ..parser import NMEAParser
        msgs = NMEAParser.parse_file(path)['nmea_messages']
        if not msgs:
            raise ValueError('该文件里没有 NMEA 报文')
        timebase.build_time_axis(msgs)
        s = SolutionSeries.from_messages(msgs)
        if len(s) == 0:
            raise ValueError('没有解析出解算历元（缺 GGA 报文？）')
        return s

    def _run_track(self):
        if getattr(self, '_track_fig', None) is None or self._series is None:
            return
        req = normalize_require(self.cmb_req.currentData())
        thresh = self.spin_static.value()
        pts = track.build_track(self._series, require=req,
                                static_thresh=thresh)
        if not pts:
            self._track_note('没有可用的轨迹点（可能全被质量要求过滤）。')
            return
        self._track_pts = pts
        cmp_pts = None
        cmp_txt = ''
        path = (self.ed_cmp_track.text() or '').strip()
        if path and os.path.exists(path):
            try:
                s2 = self._load_series_from(path)
                # 对比轨迹必须用**与设备轨迹相同的 ENU 原点**，
                # 否则各自的偏移会被原点差异抵消，重合度恒为 0
                cmp_pts = track.build_track(
                    s2, ref=(pts[0].lat, pts[0].lon, pts[0].h),
                    require=req, static_thresh=thresh)
            except Exception as e:                              # noqa: BLE001
                cmp_txt = f'（对比轨迹解析失败：{type(e).__name__}: {e}）'
        self._draw_track(pts, cmp_pts, cmp_txt)

    def _track_note(self, msg):
        for ax in (self._ax_track, self._ax_spd, self._ax_alt, self._ax_seg):
            ax.clear()
            ax.axis('off')
        self._ax_track.text(0.5, 0.5, msg, ha='center', va='center',
                            transform=self._ax_track.transAxes, color='gray')
        self._track_fig.canvas.draw_idle()

    def _draw_track(self, pts, cmp_pts=None, cmp_txt=''):
        """绘制轨迹页

        必须显式 draw_idle：matplotlib 的 Qt 画布不会因为 Axes 被改动而
        自动刷新，漏掉会表现为「切到这一页是空的」。
        """
        import matplotlib.dates as mdates

        st = track.track_stats(pts)
        segs = track.split_segments(pts)
        for ax in (self._ax_track, self._ax_spd, self._ax_alt, self._ax_seg):
            ax.clear()
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

        # ---- 轨迹图（等比例）----
        ax = self._ax_track
        es = [p.e for p in pts]
        ns = [p.n for p in pts]
        ax.plot(es, ns, lw=0.8, color='#185FA5', alpha=0.85, label='设备轨迹')
        if cmp_pts:
            ax.plot([p.e for p in cmp_pts], [p.n for p in cmp_pts],
                    lw=0.8, color='#ff7f0e', alpha=0.8, label='对比轨迹')
            ax.legend(fontsize=8)
        ax.plot([es[0]], [ns[0]], 'o', ms=5, color='#2ca02c')
        ax.text(es[0], ns[0], ' 起点', fontsize=8, color='#2ca02c',
                va='bottom')
        # 等比例才能看出真实形状；不做等比例会把 1 m 的抖动画成 10 m
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('东 (m)', fontsize=9)
        ax.set_ylabel('北 (m)', fontsize=9)
        ax.set_title(f"轨迹　跨度 {st['span_e']:.2f} × {st['span_n']:.2f} m"
                     f"　路径 {st['length_m']:.2f} m", fontsize=11)

        # ---- 速度时序 ----
        ax = self._ax_spd
        ts = [p.t for p in pts]
        sp = [p.speed if p.speed is not None else float('nan') for p in pts]
        ax.plot(ts, sp, lw=0.8, color='#d62728', alpha=0.85)
        ax.axhline(self.spin_static.value(), color='gray', ls='--', lw=1.0,
                   label=f'静止阈值 {self.spin_static.value():.2f} m/s')
        timebase.style_time_axis(ax, 'UTC 时间')
        ax.set_ylabel('速度 (m/s)', fontsize=9)
        ax.legend(fontsize=7.5)
        ax.set_title(f"速度　中位 {st['speed_median']:.3f} / "
                     f"P95 {st['speed_p95']:.3f} / 最大 "
                     f"{st['speed_max']:.3f} m/s", fontsize=10)

        # ---- 动静分段条带 ----
        ax = self._ax_seg
        y = 0
        labels = []
        for kind, color, name in (('moving', '#ff7f0e', '运动'),
                                  ('static', '#2ca02c', '静止')):
            items = [s for s in segs if s['kind'] == kind]
            if not items:
                continue
            spans = [(mdates.date2num(s['t_start']),
                      max(mdates.date2num(s['t_end'])
                          - mdates.date2num(s['t_start']), 1e-7))
                     for s in items]
            ax.broken_barh(spans, (y + 0.15, 0.7), facecolors=color,
                           alpha=0.9)
            labels.append((name, y + 0.5))
            y += 1
        if labels:
            ax.set_yticks([v for _, v in labels])
            ax.set_yticklabels([n for n, _ in labels], fontsize=9)
            ax.set_ylim(0, y)
        timebase.style_time_axis(ax)
        ax.set_title(f"动静分段（共 {len(segs)} 段）　静止占比 "
                     f"{st['static_pct']:.1f}%", fontsize=10)

        # ---- 统计文本（把对比结果并进同一块，避免两段文字重叠）----
        ax = self._ax_alt
        ax.axis('off')
        L = [f"轨迹点 {st['n']}　时长 {timebase.fmt_duration(st['duration_s'])}　"
             f"路径长度 {st['length_m']:.3f} m",
             f"包络：东 {st['span_e']:.3f} m　北 {st['span_n']:.3f} m　"
             f"高 {st['span_u']:.3f} m",
             f"运动 {st['n_moved']} 点 ({st['moving_pct']:.1f}%)　"
             f"静止 {st['n'] - st['n_moved']} 点 ({st['static_pct']:.1f}%)"]
        if cmp_pts:
            ct = track.cross_track_error(pts, cmp_pts)
            if ct.get('available'):
                L.append('')
                L.append(
                    f"重复线重合度（以设备轨迹为基准）："
                    f"中位 {ct['median'] * 100:.2f} cm　"
                    f"RMS {ct['rms'] * 100:.2f} cm　"
                    f"P95 {ct['p95'] * 100:.2f} cm　"
                    f"最大 {ct['max'] * 100:.2f} cm"
                    f"（{ct['n']} 点，剔除 {ct['n_far']} 个离群）")
            else:
                L.append(f"重复线比较不可用：{ct.get('reason', '')}")
        else:
            L.append('')
            L.append('未选择对比轨迹。要评估重复线重合度（同一条线路跑两遍的'
                     '横向偏差），请在上方选另一条 .dbg。')
        if cmp_txt:
            L.append(cmp_txt)
        ax.text(0.01, 0.95, '\n'.join(L), transform=ax.transAxes,
                va='top', ha='left', fontsize=9.5)

        self._track_fig.subplots_adjust(left=0.07, right=0.98, top=0.94,
                                        bottom=0.07, hspace=0.5, wspace=0.2)
        self._track_fig.canvas.draw_idle()

    def _save_figure(self):
        # 保存当前标签页的图（按标题取，不依赖固定索引）
        title = self.tabs.tabText(self.tabs.currentIndex())
        figs = {'精度指标': getattr(self, '_fig', None),
                '误差图表': getattr(self, '_fig', None),
                '动态轨迹': getattr(self, '_track_fig', None)}
        fig = figs.get(title)
        if fig is None:
            QMessageBox.information(self, '提示',
                                    f'「{title}」页没有可保存的图。')
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, '保存图片',
            os.path.join(last_dir('figure'),
                         f"accuracy_{datetime.now():%Y%m%d_%H%M%S}.png"),
            'PNG图片 (*.png);;PDF文档 (*.pdf);;SVG矢量图 (*.svg)')
        if filepath:
            save_last_dir('figure', filepath)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, '完成', f'图片已保存到:\n{filepath}')

    def _export_report(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, '导出精度评估报告',
            os.path.join(last_dir('static_report'),
                         f"accuracy_{datetime.now():%Y%m%d_%H%M%S}.txt"),
            '文本文件 (*.txt);;所有文件 (*.*)')
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.build_report_text())
            save_last_dir('static_report', filepath)
            QMessageBox.information(self, '完成', f'报告已保存到:\n{filepath}')
        except OSError as e:
            QMessageBox.critical(self, '错误', f'保存失败:\n{e}')

    def _export_pos(self):
        """把设备解算序列导出成 RTKPOST 兼容的 .pos

        用途：交给 RTKPOST 复算、与其他软件比对，或作为下一轮评估的参考。
        """
        if self._series is None or len(self._series) == 0:
            QMessageBox.information(self, '提示', '还没有可导出的解算序列。')
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, '导出 .pos',
            os.path.join(last_dir('open_pos'),
                         f"solution_{datetime.now():%Y%m%d_%H%M%S}.pos"),
            '解算结果 (*.pos);;所有文件 (*.*)')
        if not filepath:
            return
        try:
            with self._busy():
                solfile.write_pos(
                    filepath, self._series,
                    program='RTK 接收机测试工具',
                    note=f'来自 {self._device_label}')
            save_last_dir('open_pos', filepath)
            QMessageBox.information(
                self, '完成',
                f'已导出 {len(self._series)} 个历元到:\n{filepath}\n\n'
                f'时间按 GPST 写（RTKPOST 惯例），头部有注明。')
        except OSError as e:
            QMessageBox.critical(self, '错误', f'保存失败:\n{e}')

    def build_report_text(self):
        if not self._result:
            return '（尚未评估）'
        st = self._result
        out = []
        a = out.append
        a('=' * 78)
        a('RTK 精度评估报告')
        a('=' * 78)
        a(f"  设备        : {self._device_label}")
        a(f"  参考        : {self._ref.name}")
        a(f"  参考性质    : {'绝对真值' if st['ref_absolute'] else '相对参考'}")
        if not st['ref_absolute']:
            a('  ⚠ 注意      : 相对参考只能说明一致性/重复性，')
            a('                不能作为绝对精度对外声明。')
        a(f"  参考说明    : {st['ref_info']}")
        al = st['align']
        a(f"  对齐        : 设备 {al['n_dev']} → 对齐 {al['n_aligned']}"
          f"（过滤 {al['n_filtered']}，无参考 {al['n_no_ref']}"
          + (f"，无坐标 {al.get('n_nopos', 0)}" if al.get('n_nopos') else '')
          + "）")
        a(f"  时间容差    : {al['max_dt']:g} s"
          f"（实际 均值 {al['dt_mean']:.3f} / 最大 {al['dt_max']:.3f}）")
        a(f"  质量要求    : "
          f"{'不限' if al['require'] is None else al['require']}")
        a('')
        h = st['horiz']
        v = st['vert']
        s3 = st['sp3d']
        a('  ── 水平精度 ──')
        a(f"    RMS {h['rms']:.4f} m    CEP50 {h['cep50']:.4f} m    "
          f"CEP68 {h['cep68']:.4f} m")
        a(f"    CEP95 {h['cep95']:.4f} m   CEP99 {h['cep99']:.4f} m    "
          f"最大 {h['max']:.4f} m")
        a(f"    DRMS {h['drms']:.4f} m    2DRMS {h['2drms']:.4f} m    "
          f"3DRMS {h['3drms']:.4f} m")
        a('  ── 垂直精度 ──')
        a(f"    偏差 {v['bias']:+.4f} m   RMS {v['rms']:.4f} m   "
          f"|误差| 中位 {v['median_abs']:.4f} m")
        a(f"    |误差| P95 {v['p95_abs']:.4f} m   最大 {v['max_abs']:.4f} m")
        a('  ── 三维 ──')
        a(f"    3D RMS {s3['rms']:.4f} m   SEP {s3['sep']:.4f} m   "
          f"P95 {s3['p95']:.4f} m   最大 {s3['max']:.4f} m")
        a('  ── 分量 ──')
        for key, nm in (('e', '东'), ('n_ax', '北'), ('u', '高')):
            d = st[key]
            a(f"    {nm}: 均值 {d['mean']:+.4f}  标准差 {d['std']:.4f}  "
              f"RMS {d['rms']:.4f}  m")
        a('  ── 超限统计 ──')
        for r in st['over']:
            a(f"    水平 > {r['th'] * 100:5.1f} cm : {r['n_horiz']:5d} 次 "
              f"({r['pct_horiz']:5.2f}%)   垂直 > 同值 {r['n_vert']} 次")
        rj = st.get('rejected')
        a('  ── 粗差剔除（对照）──')
        if rj:
            a(f"    规则 {rj['rule']}")
            a(f"    剔除 {rj['n_rejected']} 个（{rj['reject_pct']:.2f}%）后："
              f"水平 RMS {rj['horiz']['rms']:.4f} m，"
              f"CEP95 {rj['horiz']['cep95']:.4f} m")
            a('    主结果未剔除粗差，此处仅作影响评估。')
        else:
            a('    未启用剔除，主结果包含全部历元。')
        a('')
        a('  ── 分档精度 ──')
        for key, nm in (('nsat', '卫星数'), ('hdop', 'HDOP'),
                        ('age', '差分龄期(s)'), ('q', '解算质量')):
            rows = accuracy.bin_by_condition(self._pairs, key)
            if not rows:
                continue
            a(f"    {nm}:")
            for r in rows:
                a(f"      {r['label']:>10}  n={r['n']:5d}   "
                  f"水平RMS {r['horiz_rms'] * 100:7.2f} cm   "
                  f"CEP95 {r['horiz_cep95'] * 100:7.2f} cm   "
                  f"高程RMS {r['vert_rms'] * 100:7.2f} cm   "
                  f"高程偏差 {r['vert_bias'] * 100:+7.2f} cm")
        a('')
        a('说明:')
        a('  · 水平误差是径向量 √(dE²+dN²)；CEP50/95 是它的经验分位，')
        a('    不是 1D 标准差乘系数。')
        a('  · DRMS=√(RMS_E²+RMS_N²)；2DRMS 约对应 95% 置信。')
        a('  · 高程偏差是系统性偏移，应接近 0；RMS 才反映离散度。')
        a('  · 外符合精度（相对真值）与内符合精度（相对自身均值）是两件事，')
        a('    后者可能很好而前者很差。')
        return '\n'.join(out)


def plt_circle(center, r, color):
    """构造一个空心圆 patch（画误差圆用）"""
    from matplotlib.patches import Circle
    return Circle(center, r, fill=False, edgecolor=color, lw=1.0,
                  alpha=0.9, linestyle='--')
