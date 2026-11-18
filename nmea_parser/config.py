# -*- coding: utf-8 -*-
"""全局配置：matplotlib 中文字体、报文类型颜色、CSV 导出列定义"""

import os
import warnings

import matplotlib
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QColor

# Matplotlib 嵌入 PyQt5 —— 必须在导入任何 matplotlib backend 之前设置
matplotlib.use('Qt5Agg')

# 配置 matplotlib 中文字体，避免图表中文显示为方块
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'sans-serif'
]
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 抑制缺失字形警告(兜底)：避免反复重绘时刷屏拖慢界面
warnings.filterwarnings('ignore',
                        message=r'Glyph .* missing from font')


# ==================== 报文类型颜色映射 ====================
# 用于表格中按类型给"类型列"着色
TYPE_COLORS = {
    'GGA': QColor(232, 245, 233),
    'GSA': QColor(227, 242, 253),
    'GST': QColor(255, 243, 224),
    'RMC': QColor(243, 229, 245),
    'REFSTATIONA': QColor(224, 247, 250),
}


# ==================== CSV 导出列定义 ====================
# 导出CSV的列定义: (表头, 取值函数)
# 取值函数接收合并后的字典(含 line_no/type/raw 及 parsed 字段)，返回该列的值
CSV_EXPORT_COLUMNS = [
    ("行号",            lambda m: m.get('line_no', '')),
    ("类型",            lambda m: m.get('type', '')),
    ("UTC时间",         lambda m: m.get('utc_time', '')),
    ("时间(HH:MM:SS)",  lambda m: m.get('time_str', '')),
    ("纬度(°)",         lambda m: m.get('latitude', '')),
    ("经度(°)",         lambda m: m.get('longitude', '')),
    ("高程(m)",         lambda m: m.get('altitude', '')),
    ("定位质量",         lambda m: m.get('fix_quality', '')),
    ("卫星数",           lambda m: m.get('num_sats', '')),
    ("HDOP",            lambda m: m.get('hdop', '')),
    ("高程异常(m)",      lambda m: m.get('geoid_sep', '')),
    ("差分龄期(s)",      lambda m: m.get('age', '')),
    ("基准站ID",         lambda m: m.get('station_id', '')),
    ("GSA定位模式",      lambda m: m.get('mode', '')),
    ("GSA定位类型",      lambda m: m.get('fix_type', '')),
    ("GSA使用卫星",      lambda m: ','.join(m['sats_used']) if isinstance(m.get('sats_used'), list) else ''),
    ("PDOP",            lambda m: m.get('pdop', '')),
    ("VDOP",            lambda m: m.get('vdop', '')),
    ("GST RMS(m)",      lambda m: m.get('rms', '')),
    ("GST长半轴(m)",     lambda m: m.get('semi_major', '')),
    ("GST短半轴(m)",     lambda m: m.get('semi_minor', '')),
    ("GST误差方向(°)",   lambda m: m.get('orientation', '')),
    ("纬度误差(m)",      lambda m: m.get('lat_error', '')),
    ("经度误差(m)",      lambda m: m.get('lon_error', '')),
    ("高程误差(m)",      lambda m: m.get('alt_error', '')),
    ("RMC状态",          lambda m: m.get('status', '')),
    ("RMC速度(节)",      lambda m: m.get('speed_knots', '')),
    ("航向(°)",          lambda m: m.get('course', '')),
    ("RMC日期",          lambda m: m.get('date', '')),
    ("磁偏角(°)",        lambda m: m.get('mag_var', '')),
    ("基准站名称",        lambda m: m.get('station_name', '')),
    ("端口",             lambda m: m.get('port', '')),
    ("标志",             lambda m: m.get('flags', '')),
    ("CPU负载",          lambda m: m.get('cpu_load', '')),
    ("GPS周",            lambda m: m.get('gps_week', '')),
    ("GPS毫秒",          lambda m: m.get('gps_ms', '')),
    ("GSV总帧数",        lambda m: m.get('num_msgs', '')),
    ("GSV帧序号",        lambda m: m.get('msg_idx', '')),
    ("GSV在视卫星数",    lambda m: m.get('sats_in_view', '')),
    ("原始报文",          lambda m: m.get('raw', '')),
]


# ==================== 记住上次文件路径 ====================
# 导入/导出对话框记住上次选中的目录，下次默认打开该目录。
# 用 QSettings 持久化(Windows 注册表)，跨重启保留；用户手动改选后自动更新。
_SETTINGS = QSettings("RTK-Tools", "NMEADebugParser")


def last_dir(key):
    """返回某操作上次使用的目录（可能为空字符串）"""
    return str(_SETTINGS.value(f"last_dir/{key}", "") or "")


def save_last_dir(key, filepath):
    """记录某操作最后选中的文件所在目录"""
    if filepath:
        _SETTINGS.setValue(f"last_dir/{key}", os.path.dirname(filepath))


# ==================== 已知控制点（静态坐标单点比对） ====================
# 静态分析里把 RINEX 头的近似坐标与已知控制点比对，得到平面/高程偏差。
# 已知点通常一个工地固定不变，持久化后打开文件即自动比对，不必重输。

_KP_KEYS = ('lat', 'lon', 'h', 'antenna_h', 'name', 'mode',
            'x', 'y', 'z')


def known_point():
    """读取持久化的已知控制点；没有设置过返回 None"""
    if _SETTINGS.value("known_point/exists", "") != "1":
        return None
    out = {}
    for k in _KP_KEYS:
        v = _SETTINGS.value(f"known_point/{k}", "")
        if v != "":
            try:
                out[k] = float(v) if k != 'name' and k != 'mode' else str(v)
            except (TypeError, ValueError):
                out[k] = v
    out.setdefault('mode', 'blh')
    out.setdefault('name', '已知点')
    out.setdefault('antenna_h', 0.0)
    return out


def save_known_point(**kw):
    """持久化已知控制点（只存有效字段，键见 _KP_KEYS）"""
    for k in _KP_KEYS:
        v = kw.get(k)
        _SETTINGS.setValue(f"known_point/{k}",
                           "" if v is None else str(v))
    _SETTINGS.setValue("known_point/exists", "1")


def clear_known_point():
    """清除持久化的已知控制点"""
    _SETTINGS.remove("known_point")
