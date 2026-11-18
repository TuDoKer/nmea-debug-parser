# -*- coding: utf-8 -*-
"""后处理解算结果文件解析（RTKPOST / rnx2rtkp 的 .pos / .llh）

RTK 测试的"真值"从哪来
---------------------
我们**不自研 RTK 解算**（那是另一个量级的工程）。真值走两条路：
  1. 已知控制点（架在已知点上直接比）
  2. 第三方后处理（RTKPOST / rnx2rtkp 的 .pos 结果当参考）

本模块负责第 2 条：把 .pos 读成与设备侧同构的 `SolutionSeries`，
这样后面对比、统计、绘图全部可以走同一套代码。

RTKPOST 的 .pos 格式
-------------------
头部是 ``%`` 开头的注释行，然后是数据行::

    %  GPST                  latitude(deg) longitude(deg)  height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  sdne(m)  sdeu(m)  sdun(m) age(s)  ratio
     2026/09/16 01:17:27.00   30.000000000   120.000000000   10.0000   4  27   0.0050   0.0040   0.0090   0.0010   0.0020  0.0030   1.00   12.5

要点与坑：
  * 时间标签是 **GPST**（不是 UTC），与设备 NMEA 的时间有 18 s 闰秒差 ——
    直接按时间比对会整体错位 18 秒。比对时必须先统一（见 `gpst_offset`）。
  * 加 ``-ecef`` 选项时输出的是 X/Y/Z（可达 1e6 量级）而不是经纬度，
    必须靠量级自动判别，否则会把 ECEF 当经纬度算出荒谬结果。
  * 列数是**可变的**：老版本没有 ratio 列，``-dp`` 等选项也会改变尾列。
    解析要按"前 6 列必有、后面能读多少读多少"来做，不要按固定索引硬取。
  * Q 的含义与 NMEA 一致（1 单点 / 2 差分 / 4 固定 / 5 浮动）。
"""

import datetime
import os

from . import geodesy
from .solution import SolutionSeries, Q_FIX, Q_FLOAT

# .pos 时间标签默认是 GPST；转 UTC 要减掉的秒数
GPS_UTC_LEAP = 18


class PosFileError(ValueError):
    """解析失败"""


def _parse_time(parts, start_idx=0):
    """从 ``YYYY/MM/DD hh:mm:ss.ss`` 解析出 naive datetime

    返回 (datetime, 用掉的字段数)。日期分隔符可能是 / 或 -，
    也见过 ``YYYYMMDD`` 连写的形式。
    """
    if start_idx + 1 >= len(parts):
        return None, 0
    d, t = parts[start_idx], parts[start_idx + 1]
    for sep in ('/', '-'):
        if sep in d:
            y, mo, dd = d.split(sep)
            break
    else:
        if len(d) == 8 and d.isdigit():
            y, mo, dd = d[:4], d[4:6], d[6:]
        else:
            return None, 0
    try:
        hh, mi, ss = t.split(':')
        sec = float(ss)
        si = int(sec)
        micro = int(round((sec - si) * 1e6))
        if micro >= 1000000:
            si += 1
            micro -= 1000000
        return datetime.datetime(int(y), int(mo), int(dd), int(hh), int(mi),
                                 min(si, 59), micro), 2
    except (ValueError, IndexError):
        return None, 0


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ==================== Q 值口径换算 ====================
# .pos 是 RTKLIB（RTKPOST/rnx2rtkp）定义的格式，它的 Q 列口径
# 与 NMEA 的 fix_quality **不一样**，必须换算，否则：
#   * RTKPOST 的「固定解」Q=1 会被当成 NMEA 的「单点」，
#     PosFileRef 的「仅固定解」过滤（Q∈{4}）会把全部历元滤光，
#     精度评估直接得不到结果；
#   * 反过来我们导出的 .pos 若写 NMEA 口径，RTKLIB 系工具
#     也会误读解算状态。
# 映射依据 RTKLIB 手册（Q: 1=fix 2=float 3=SBAS 4=DGPS 5=single
# 6=PPP）与 NMEA fix_quality（4=RTK固定 5=RTK浮动 9=SBAS 2=差分
# 1=单点 6=估算）。
_Q_RTKLIB2IN = {1: 4, 2: 5, 3: 9, 4: 2, 5: 1, 6: 6}
_Q_IN2RTKLIB = {v: k for k, v in _Q_RTKLIB2IN.items()}


def read_pos(filepath, time_is_gpst=True):
    """读 RTKPOST 的 .pos / .llh，返回 (SolutionSeries, meta)

    参数
    ----
    time_is_gpst : bool
        .pos 的时间标签默认是 GPST；为 True 时统一转成 UTC 再放进序列，
        这样与设备 NMEA（UTC）能直接按时间比对。
        如果你的 .pos 已经是 UTC（少见），传 False。

    meta 里包含头部信息、是否为 ECEF、时间系统、有效历元数等，
    供报告展示与排错。
    """
    if not os.path.exists(filepath):
        raise PosFileError(f'文件不存在: {filepath}')

    series = SolutionSeries()
    meta = {
        'path': filepath,
        'header': {},
        'is_ecef': False,
        'time_system': 'GPST' if time_is_gpst else 'UTC',
        'n_lines': 0,
        'n_skipped': 0,
        'warnings': [],
    }

    with open(filepath, 'r', errors='replace') as f:
        lines = f.readlines()

    raw_rows = []
    for ln in lines:
        meta['n_lines'] += 1
        s = ln.strip()
        if not s:
            continue
        if s.startswith('%'):
            # 头部注释：形如 "% key : value"，也可能只是列名行
            if ':' in s:
                k, v = s[1:].split(':', 1)
                meta['header'][k.strip()] = v.strip()
            continue
        parts = s.split()
        if len(parts) < 6:
            meta['n_skipped'] += 1
            continue
        t, used = _parse_time(parts, 0)
        if t is None:
            meta['n_skipped'] += 1
            continue
        rest = parts[used:]
        if len(rest) < 4:
            meta['n_skipped'] += 1
            continue
        a, b, c = _f(rest[0]), _f(rest[1]), _f(rest[2])
        q = _f(rest[3])
        if None in (a, b, c) or q is None:
            meta['n_skipped'] += 1
            continue
        tail = [_f(x) for x in rest[4:]]
        raw_rows.append((t, a, b, c, q, tail))

    if not raw_rows:
        raise PosFileError(
            f'没有解析到任何数据行（读了 {meta["n_lines"]} 行）。\n'
            f'请确认这是 RTKPOST/rnx2rtkp 输出的 .pos/.llh 文件。')

    # ---- ECEF 还是经纬度：靠量级判别 ----
    # 经纬度: |lat|<=90, |lon|<=180, h 一般 <1e4
    # ECEF  : |x|,|y| <= 6.4e6，|z| <= 6.4e6，三个都可能上万
    maxabs = max(max(abs(r[1]), abs(r[2]), abs(r[3])) for r in raw_rows)
    is_ecef = maxabs > 1.0e5
    meta['is_ecef'] = is_ecef
    if is_ecef:
        meta['warnings'].append(
            '该 .pos 输出的是 ECEF 直角坐标（RTKPOST 的 -ecef 选项），'
            '已自动换算为经纬高。')

    for t, a, b, c, q, tail in raw_rows:
        if is_ecef:
            lat, lon, h = geodesy.ecef_to_blh(a, b, c)
        else:
            lat, lon, h = a, b, c
        if time_is_gpst:
            t = t - datetime.timedelta(seconds=GPS_UTC_LEAP)
        series.t.append(t)
        series.lat.append(lat)
        series.lon.append(lon)
        series.h.append(h)
        q_raw = int(q) if q is not None else None
        # RTKLIB 口径 -> 内部口径（4=固定 5=浮动）。.pos 落盘的一律是
        # RTKLIB 口径（含本工具 write_pos 的导出，见 _Q_IN2RTKLIB）
        series.q.append(_Q_RTKLIB2IN.get(q_raw, q_raw))
        # 尾列顺序（能读多少读多少）: ns sdn sde sdu sdne sdeu sdun age ratio
        def tk(i):
            return tail[i] if i < len(tail) else None
        series.nsat.append(int(tk(0)) if tk(0) is not None else None)
        series.sd_n.append(tk(1))
        series.sd_e.append(tk(2))
        series.sd_u.append(tk(3))
        series.age.append(tk(7))
        series.station.append('')

    # 按时间排序
    order = sorted(range(len(series.t)), key=lambda i: series.t[i])
    if order != list(range(len(series.t))):
        for name in ('t', 'lat', 'lon', 'h', 'q', 'nsat',
                     'sd_e', 'sd_n', 'sd_u', 'age', 'station'):
            arr = getattr(series, name)
            setattr(series, name, [arr[i] for i in order])

    # 统一补齐 SolutionSeries 里本格式没有的字段
    n = len(series.t)
    series.hdop = [None] * n
    series.pdop = [None] * n
    series.vdop = [None] * n

    meta['q_converted'] = 'RTKLIB->NMEA口径(1->4固定,2->5浮动)'
    meta['n_epochs'] = n
    meta['n_fix'] = sum(1 for x in series.q if x == Q_FIX)
    meta['n_float'] = sum(1 for x in series.q if x == Q_FLOAT)
    if n:
        meta['t_start'] = series.t[0]
        meta['t_end'] = series.t[-1]
        meta['duration_s'] = (series.t[-1] - series.t[0]).total_seconds()
    return series, meta


def write_pos(filepath, series, program='本项目', note='',
              time_is_gpst=True):
    """把 SolutionSeries 写成 RTKPOST 兼容的 .pos

    用途：没有真实 .pos 时，可以把设备解算序列导出成 .pos 作为
    "参考"来打通链路自检；也方便把结果交给别的软件复算。

    ``time_is_gpst`` 默认 **True** —— 与 ``read_pos`` 的默认一致，
    否则 write→read 一圈会凭空差 18 秒（GPS 闰秒）。GPST 也是
    RTKPOST 的惯例。头部会写明时间系统，便于人工核对。
    """
    lines = []
    n_skipped = 0
    lines.append(f'% program   : {program}')
    if note:
        lines.append(f'% note      : {note}')
    lines.append(f'% time sys  : {"GPST" if time_is_gpst else "UTC"}')
    if series.t:
        lines.append(f'% obs start : {series.t[0]:%Y/%m/%d %H:%M:%S.%f}'
                     f'{" GPST" if time_is_gpst else " UTC"}')
        lines.append(f'% obs end   : {series.t[-1]:%Y/%m/%d %H:%M:%S.%f}'
                     f'{" GPST" if time_is_gpst else " UTC"}')
    lines.append('%  time                  latitude(deg) longitude(deg) '
                 ' height(m)   Q  ns   sdn(m)   sde(m)   sdu(m)  '
                 'sdne(m)  sdeu(m)  sdun(m) age(s)  ratio')
    for i in range(len(series.t)):
        dt = series.t[i]
        if time_is_gpst:
            dt = dt + datetime.timedelta(seconds=GPS_UTC_LEAP)
        def g(attr, idx=i, nd=None, w=9):
            v = getattr(series, attr)[idx]
            if v is None:
                return ' ' * w
            return f'{v:{w}.{nd}f}' if nd is not None else f'{v:{w}}'
        # lat/lon/h 可能是 None（无定位输出的历元，见 accuracy.align 的
        # 说明）。直接格式化 None 会 TypeError —— 写成 0 会误导下游，
        # 这里跳过该历元不写（.pos 本来就只该含有效解）。
        if (series.lat[i] is None or series.lon[i] is None
                or series.h[i] is None):
            n_skipped += 1
            continue
        lines.append(
            f'{dt:%Y/%m/%d %H:%M:%S}.{dt.microsecond // 10000:02d} '
            f'{series.lat[i]:15.9f} {series.lon[i]:15.9f} '
            f'{series.h[i]:10.4f} '
            f'{_Q_IN2RTKLIB.get(series.q[i], series.q[i]) if series.q[i] is not None else 0:3d} '
            f'{series.nsat[i] if series.nsat[i] is not None else 0:3d} '
            f'{getattr(series, "sd_n")[i] if series.sd_n[i] is not None else 0:9.4f} '
            f'{series.sd_e[i] if series.sd_e[i] is not None else 0:9.4f} '
            f'{series.sd_u[i] if series.sd_u[i] is not None else 0:9.4f} '
            f'{0.0:9.4f} {0.0:9.4f} {0.0:9.4f} '
            f'{series.age[i] if series.age[i] is not None else 0:6.2f} '
            f'{0.0:7.1f}')
    if n_skipped:
        lines.append(f'% note      : 跳过 {n_skipped} 个无坐标历元'
                     f'（无定位输出）')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return filepath
