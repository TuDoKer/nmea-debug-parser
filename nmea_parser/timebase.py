# -*- coding: utf-8 -*-
"""统一时间轴：把 NMEA 的时刻/日期字段还原成绝对 UTC 时刻

为什么需要这个模块
------------------
原先所有质量统计的横轴用的是**文件行号**（statistics.py 里
`times.append(m.get('line_no', 0))`）。行号有三个致命问题：

  1. 无法定位到具体时刻 —— 说不出「固定是在 10:23:15 成功的」
  2. 无法换算成秒 —— 算不出「失锁后多久恢复」
  3. 无法跨文件对齐 —— 多设备比对、与后处理结果比对全都做不了

RTK 测试的比对、轨迹、批量分析全部依赖统一时间轴，所以这一层必须先做。

NMEA 的时间来源
---------------
  * GGA / RMC / GST 的第 1 个字段是 UTC 时刻 ``hhmmss.ss``
    （GSA **没有**时刻字段，它的第 1 字段是模式 M/A）
  * RMC 第 10 个字段是 UTC 日期 ``ddmmyy``
  * ZDA 同时带时刻与日期（部分固件会输出）

实测本批接收机的历元顺序为::

    GGA -> GSA x4 -> GST -> RMC

GGA 在最前且带时刻，所以「承前」策略（自己没有时刻的报文沿用最近一次
已知时刻）能把 GSA 正确归到同一个历元。

本模块统一使用**无时区的 UTC datetime**（naive UTC）。不用 `aware`
是为了和 matplotlib 的日期轴配合更省事；所有涉及「换算成绝对时间戳」
的地方一律走 `dt_to_unix()`，它用 `calendar.timegm` 而不是
`datetime.timestamp()` —— 后者会按本地时区解释 naive 时间，是经典坑。
"""

import calendar
import datetime

# GPST - UTC 的整数秒差（闰秒累计）。2017-01-01 起为 18 s。
# 本阶段 NMEA 内部只用 UTC，常量先备着，供后续与 .pos（通常为 GPST）比对时使用。
GPST_UTC_LEAP = 18


def dt_to_unix(dt):
    """naive UTC datetime -> Unix 时间戳（秒）

    刻意不用 `dt.timestamp()`：那是按**本地时区**解释 naive 时间的，
    在东八区会整体偏 8 小时。
    """
    if dt is None:
        return None
    return calendar.timegm(dt.timetuple()) + dt.microsecond / 1e6


def unix_to_dt(ts):
    """Unix 时间戳（秒）-> naive UTC datetime"""
    if ts is None:
        return None
    return datetime.datetime.utcfromtimestamp(ts)


def gpst_to_utc(dt):
    """GPS 时 -> UTC（减去闰秒累计）"""
    if dt is None:
        return None
    return dt - datetime.timedelta(seconds=GPST_UTC_LEAP)


def utc_to_gpst(dt):
    """UTC -> GPS 时（加上闰秒累计）"""
    if dt is None:
        return None
    return dt + datetime.timedelta(seconds=GPST_UTC_LEAP)


def parse_nmea_tod(text):
    """解析 NMEA 的 UTC 时刻字段 ``hhmmss.ss`` -> (h, m, s_float)

    兼容位数不足、带/不带小数、以及空串。解析不出来返回 None。
    """
    if not text:
        return None
    s = str(text).strip()
    if len(s) < 6 or not s[:6].isdigit():
        return None
    try:
        hh = int(s[0:2])
        mm = int(s[2:4])
        ss = float(s[4:])
    except (ValueError, IndexError):
        return None
    if hh > 23 or mm > 59 or not (0.0 <= ss < 61.0):
        return None
    return hh, mm, ss


def parse_nmea_date(text):
    """解析 NMEA 的日期字段 ``ddmmyy`` -> datetime.date

    RMC 用的是 2 位年，按 POSIX 惯例映射到 1980~2079
    （00-79 -> 2000-2079，80-99 -> 1980-1999）。
    """
    if not text:
        return None
    s = str(text).strip()
    if len(s) < 6 or not s[:6].isdigit():
        return None
    try:
        dd, mm, yy = int(s[0:2]), int(s[2:4]), int(s[4:6])
        year = 2000 + yy if yy < 80 else 1900 + yy
        return datetime.date(year, mm, dd)
    except ValueError:
        return None


# 判定跨日的阈值：当前时刻比上一次「早」超过这么多小时才认为进位了一天。
# 用 12 小时而不是「任意变早就进位」，是为了容忍同一历元内报文顺序错乱
# 或秒级回退，避免把正常抖动误判成跨日。
_ROLLOVER_HOURS = 12.0


def build_time_axis(messages):
    """为报文列表构建绝对 UTC 时间轴（原地给每条报文加 `utc_dt`）

    参数
    ----
    messages : list[dict]
        NMEAParser 的输出，需按文件顺序排列，每条含 `parsed` 与 `line_no`。

    返回
    ----
    dict:
        available     : bool         是否拿到了可用的绝对时间
        reason        : str          不可用时的原因说明
        source        : str          日期来源 'rmc' / 'zda'
        n_total       : int          报文总数
        n_timed       : int          自身带时刻的报文数
        n_inherited   : int          靠「承前」拿到时刻的报文数
        n_missing     : int          仍无时刻的报文数
        t_start       : datetime|None
        t_end         : datetime|None
        day_rollovers : int          跨日次数
        backward_steps: int          时刻真实倒退的次数（接收机时标异常）
        backward_at   : list         倒退发生的位置 [(前, 后), ...]（最多 10 个）
    """
    out = {
        'available': False, 'reason': '', 'source': '',
        'n_total': len(messages or []), 'n_timed': 0, 'n_inherited': 0,
        'n_missing': 0, 't_start': None, 't_end': None, 'day_rollovers': 0,
        'backward_steps': 0, 'backward_at': [],
    }
    if not messages:
        out['reason'] = '没有报文'
        return out

    # 先找日期：优先 ZDA（自带日期），否则 RMC
    base_date = None
    source = ''
    for m in messages:
        p = m.get('parsed') or {}
        if m.get('type') == 'ZDA':
            d = parse_nmea_date(p.get('date') or '')
            if d:
                base_date, source = d, 'zda'
                break
        if m.get('type') == 'RMC':
            d = parse_nmea_date(p.get('date') or '')
            if d:
                base_date, source = d, 'rmc'
                break

    if base_date is None:
        out['reason'] = ('报文里没有 RMC/ZDA 的日期字段，'
                         '无法组成绝对时刻（横轴将退回行号）')
        return out

    cur_date = base_date
    prev_tod = None
    last_dt = None
    last_own_dt = None
    rollovers = 0
    backward = 0
    backward_at = []

    for m in messages:
        p = m.get('parsed') or {}
        mtype = m.get('type')

        # RMC/ZDA 的日期是权威的：遇到就用它校正当前日期
        if mtype in ('RMC', 'ZDA'):
            d = parse_nmea_date(p.get('date') or '')
            if d and d != cur_date:
                cur_date = d

        tod = parse_nmea_tod(p.get('utc_time'))
        if tod is None:
            # 自己没有时刻（典型是 GSA）：承前
            if last_dt is not None:
                m['utc_dt'] = last_dt
                out['n_inherited'] += 1
            else:
                m['utc_dt'] = None
                out['n_missing'] += 1
            continue

        hh, mm, ss = tod
        # 跨日判定：时刻比上一次早得离谱，说明已经过了午夜
        if prev_tod is not None and (prev_tod - (hh * 3600 + mm * 60 + ss)
                                     ) > _ROLLOVER_HOURS * 3600:
            cur_date += datetime.timedelta(days=1)
            rollovers += 1
        prev_tod = hh * 3600 + mm * 60 + ss

        sec_int = int(ss)
        micro = int(round((ss - sec_int) * 1e6))
        if micro >= 1000000:            # 四舍五入到下一秒
            sec_int += 1
            micro = 0
        try:
            dt = datetime.datetime(cur_date.year, cur_date.month,
                                   cur_date.day, hh, mm,
                                   min(sec_int, 59), micro)
        except ValueError:
            m['utc_dt'] = last_dt
            out['n_missing'] += 1
            continue
        if sec_int >= 60:               # 闰秒，归到 59.999999
            dt = dt.replace(second=59, microsecond=999999)

        m['utc_dt'] = dt
        # 时刻真实倒退：只在「自带时刻」的报文之间比较（承前的报文不参与，
        # 否则同一时刻内的多条报文会被误判）。实测本批接收机每个文件各有
        # 1 处 1 秒倒退，属接收机时标异常，值得单独报告。
        if last_own_dt is not None and dt < last_own_dt:
            backward += 1
            if len(backward_at) < 10:
                backward_at.append((last_own_dt, dt))
        last_own_dt = dt
        last_dt = dt

    timed = [m['utc_dt'] for m in messages if m.get('utc_dt') is not None]
    if not timed:
        out['reason'] = '所有报文都没能解析出时刻'
        return out

    out.update({
        'available': True, 'source': source,
        'n_timed': sum(1 for m in messages
                       if m.get('utc_dt') is not None) - out['n_inherited'],
        't_start': min(timed), 't_end': max(timed),
        'day_rollovers': rollovers,
        'backward_steps': backward, 'backward_at': backward_at,
    })
    return out


def dedupe_by_time(messages, types=None):
    """按 (报文类型, 绝对时刻) 去重，返回 (保留的报文, 被丢弃的报文)

    为什么必须去重
    --------------
    实测本批接收机**每个历元会发两条 GGA**，内容相同但格式不同::

        $GPGGA,093300.00,3109.9646858,N,...,4,27,0.6,17.2497,M,0.000,M,1.0,0211*7E
        $GPGGA,093300   ,3109.96468578,N,...,4,027,0.6,17.250,M,0.00,M,1,0211*70

    时间字段一条带两位小数一条不带、卫星数一条 `27` 一条补零成 `027`、
    高程一条 17.2497 一条 17.250 —— 数值一致（差异在舍入量级，亚毫米），
    是同一历元的两种输出格式。若不去重，历元数会被算成近两倍
    （实测 I5：1087 条 GGA 实际只有 907 个历元，正好等于 RMC/GST 的条数）。

    去重键用**绝对时刻**而不是时间字符串 —— 后者 `'093300'` 与
    `'093300.00'` 不相等，按字符串去重会全部漏掉。

    保留每组第一条（先出现的），丢弃的都是重复历元。
    """
    keep, drop = [], []
    seen = set()
    for m in messages:
        if types is not None and m.get('type') not in types:
            keep.append(m)
            continue
        dt = m.get('utc_dt')
        if dt is None:                      # 没有时刻的无法去重，原样保留
            keep.append(m)
            continue
        key = (m.get('type'), dt)
        if key in seen:
            drop.append(m)
        else:
            seen.add(key)
            keep.append(m)
    return keep, drop


def times_of(messages):
    """取一批报文的绝对时刻列表（无时刻的丢 None 占位，长度对齐）"""
    return [m.get('utc_dt') for m in messages]


def valid_times(messages):
    """取一批报文里**非空**的绝对时刻，用于统计与绘图"""
    return [m['utc_dt'] for m in messages if m.get('utc_dt') is not None]


def has_time(messages):
    """这批报文是否带可用的绝对时刻"""
    return any(m.get('utc_dt') is not None for m in messages)


def record_time_range(messages):
    """从一批报文里取 (起, 止) 时刻；没有则返回 (None, None)

    给质量统计各分项用：只统计带时刻的报文，得到该分项的真实时间跨度。
    """
    t = valid_times(messages)
    if not t:
        return None, None
    return min(t), max(t)


def style_time_axis(ax, label='UTC 时间'):
    """把某个 Axes 的横轴配成 UTC 时间轴

    用 ConciseDateFormatter：短跨度只显示时分秒，跨天时自动补上日期，
    不需要人为判断该用哪种格式。
    没有 matplotlib 时静默跳过（本模块在纯逻辑场景下也会被 import）。
    """
    try:
        import matplotlib.dates as mdates
    except ImportError:
        return False
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel(label, fontsize=9)
    return True


def fmt_dt(dt, with_date=False):
    """格式化时刻用于显示；None -> '—'"""
    if dt is None:
        return '—'
    if with_date:
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    return dt.strftime('%H:%M:%S')


def fmt_duration(seconds):
    """把秒数格式化成人能读的形式"""
    if seconds is None:
        return '—'
    s = float(seconds)
    if s < 0:
        return '—'
    if s < 60:
        return f'{s:.1f} s'
    if s < 3600:
        return f'{int(s // 60)} 分 {s % 60:.0f} 秒'
    return f'{int(s // 3600)} 时 {int((s % 3600) // 60)} 分'
