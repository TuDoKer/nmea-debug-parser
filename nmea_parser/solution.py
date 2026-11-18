# -*- coding: utf-8 -*-
"""解算记录序列（SolutionSeries）与 RTK 时效状态机

为什么单独一个模块
------------------
`statistics.py` 关心的是「观测值好不好」，
RTK 测试关心的是「解算结果固定快不快、稳不稳」。

后者需要一条按时间排列的**解算记录**（时刻 + 坐标 + 解算状态 + 环境量），
而这几个字段现在散在 GGA / GSA / GST 三套结构里，没法整体处理 ——
例如「固定率」要把 GGA 的 fix_quality 按时间排好才能算，
「失锁恢复耗时」更是需要按时间扫描状态机。

本模块把三套结构整成一条列式序列，再在其上做 RTK 时效分析。

NMEA 的 GGA fix_quality 取值
----------------------------
  0 = 无效    1 = 单点    2 = 差分(DGPS)   3 = PPS
  4 = RTK 固定  5 = RTK 浮动  6 = 估算   7 = 手动   8 = 仿真

RTK 测试里真正关心的是 **4（固定）**。原先的「成功率」口径是
`fix_quality > 0`，把单点/差分/浮点/固定混在一起算 —— 那个数字
反映不出 RTK 能力，本模块把它拆开单列。
"""

import collections

from . import timebase

Q_NAMES = {
    0: '无效', 1: '单点', 2: '差分', 3: 'PPS',
    4: 'RTK固定', 5: 'RTK浮动', 6: '估算', 7: '手动', 8: '仿真',
    # NMEA 标准只定义到 8；部分接收机用 9 表示 SBAS（星基增强）。
    # 9 属**厂商扩展**，不是标准值，标注出来避免误导。
    9: 'SBAS(扩展)',
}
Q_FIX = 4          # RTK 固定
Q_FLOAT = 5        # RTK 浮动

# 本表是 Q 值的**唯一来源**。以前 statistics.py 里还有一份 FIX_NAMES，
# 两处对 5/6/8 的命名不一致（"RTK浮点" vs "RTK浮动"、"DR" vs "估算"、
# "模拟" vs "仿真"），同一个数值在不同对话框里显示不同标签。
# 现在 statistics.FIX_NAMES 直接引用本表，别再另起一份。

# 分档固定率用的区间（左闭右开，最后一档用大数兜住）
NSAT_BINS = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 999)]
HDOP_BINS = [(0.0, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 999.0)]
AGE_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 999.0)]


def q_name(q):
    """解算质量的中文名"""
    if q is None:
        return '—'
    return Q_NAMES.get(int(q), f'未知({q})')


class SolutionSeries:
    """一次观测的解算记录序列（列式存储）

    所有列表按历元一一对应，缺失字段填 None。历元按时间升序。
    """

    __slots__ = ('t', 'lat', 'lon', 'h', 'q', 'nsat', 'hdop', 'pdop',
                 'vdop', 'age', 'sd_e', 'sd_n', 'sd_u', 'station',
                 'time_axis', 'n_gga_raw', 'n_dup_gga', 'n_unrecoverable')

    def __init__(self):
        self.t = []
        self.lat = []
        self.lon = []
        self.h = []
        self.q = []
        self.nsat = []
        self.hdop = []
        self.pdop = []
        self.vdop = []
        self.age = []
        self.sd_e = []
        self.sd_n = []
        self.sd_u = []
        self.station = []
        # 构建过程的信息（供报告说明数据质量）
        self.time_axis = None
        self.n_gga_raw = 0          # 去重前的 GGA 条数
        self.n_dup_gga = 0          # 被判为重复历元而丢弃的 GGA 条数
        self.n_unrecoverable = 0    # 没有时刻、无法定位到历元的 GGA 条数

    def __len__(self):
        return len(self.t)

    @property
    def n(self):
        return len(self.t)

    def t_start(self):
        return self.t[0] if self.t else None

    def t_end(self):
        return self.t[-1] if self.t else None

    def duration_s(self):
        if len(self.t) < 2:
            return 0.0
        return (self.t[-1] - self.t[0]).total_seconds()

    def median_interval_s(self):
        """历元间隔中位数（秒），用于判断采样率是否规整"""
        if len(self.t) < 3:
            return None
        gaps = sorted((self.t[i] - self.t[i - 1]).total_seconds()
                      for i in range(1, len(self.t)))
        return gaps[len(gaps) // 2]

    # ---------------- 构建 ----------------

    @classmethod
    def from_messages(cls, messages, time_axis=None, apply_dedup=True):
        """从 NMEA 报文列表构建解算序列

        参数
        ----
        messages : list[dict]
            解析器的 nmea_messages（需已按文件顺序排列）
        time_axis : dict|None
            已构建的时间轴；为 None 时本方法自己调用 timebase 构建
        apply_dedup : bool
            是否按绝对时刻去重。**默认必须开** —— 实测本批接收机
            每个历元会发两条格式不同的 GGA（一条 `093300.00`/`27`，
            一条 `093300`/`027`，数值差在舍入量级），不去重会把
            历元数算成近两倍。

        解算序列以 **GGA** 为骨架（每个历元一条，且自带时刻、坐标、
        解算质量、卫星数、HDOP、龄期）；GST 按时刻补精度因子
        （σE/σN/σU），GSA 按时刻补 PDOP/VDOP。
        """
        s = cls()
        if not messages:
            return s

        if time_axis is None:
            time_axis = timebase.build_time_axis(messages)
        s.time_axis = time_axis

        gga = [m for m in messages if m.get('type') == 'GGA']
        s.n_gga_raw = len(gga)
        if apply_dedup:
            gga, dropped = timebase.dedupe_by_time(gga)
            s.n_dup_gga = len(dropped)

        # 按时刻索引 GST / GSA，供 GGA 骨架取用
        gst_at = collections.defaultdict(list)
        gsa_at = collections.defaultdict(list)
        for m in messages:
            dt = m.get('utc_dt')
            if dt is None:
                continue
            if m.get('type') == 'GST':
                gst_at[dt].append(m.get('parsed') or {})
            elif m.get('type') == 'GSA':
                gsa_at[dt].append(m.get('parsed') or {})

        for m in gga:
            p = m.get('parsed') or {}
            dt = m.get('utc_dt')
            if dt is None:
                s.n_unrecoverable += 1
                continue
            s.t.append(dt)
            s.lat.append(p.get('latitude'))
            s.lon.append(p.get('longitude'))
            s.h.append(p.get('altitude'))
            s.q.append(p.get('fix_quality'))
            s.nsat.append(p.get('num_sats'))
            s.hdop.append(p.get('hdop'))
            s.age.append(p.get('age'))
            s.station.append(p.get('station_id'))

            # GST: 取该时刻第一条（每历元一条）
            g = gst_at.get(dt) or [{}]
            g0 = g[0]
            s.sd_e.append(g0.get('lat_error'))
            s.sd_n.append(g0.get('lon_error'))
            s.sd_u.append(g0.get('alt_error'))

            # GSA: 该时刻有多条（每星座一条），
            # PDOP/VDOP 取各星座的**最大值** —— 保守口径，宁可报差的那颗
            gs = gsa_at.get(dt) or []
            pd = [x.get('pdop') for x in gs if x.get('pdop') is not None]
            vd = [x.get('vdop') for x in gs if x.get('vdop') is not None]
            s.pdop.append(max(pd) if pd else None)
            s.vdop.append(max(vd) if vd else None)

        # 按时间排序（报文顺序通常已升序，但不保证）
        order = sorted(range(len(s.t)), key=lambda i: s.t[i])
        if order != list(range(len(s.t))):
            for name in ('t', 'lat', 'lon', 'h', 'q', 'nsat', 'hdop',
                         'pdop', 'vdop', 'age', 'sd_e', 'sd_n', 'sd_u',
                         'station'):
                arr = getattr(s, name)
                setattr(s, name, [arr[i] for i in order])
        return s

    # ---------------- 便捷统计 ----------------

    def q_counts(self):
        return collections.Counter(x for x in self.q if x is not None)

    def valid_mask(self):
        """有解（fix_quality > 0）的历元掩码"""
        return [(x is not None and x > 0) for x in self.q]

    def fix_mask(self):
        """RTK 固定的历元掩码"""
        return [x == Q_FIX for x in self.q]

    def positions(self):
        """(lat, lon, h) 三元组列表，缺测为 None"""
        return list(zip(self.lat, self.lon, self.h))


# ==================== RTK 时效状态机 ====================


def _bin_fix_rate(values, masks, bins, unit=''):
    """按 value 分档统计固定率

    values 与 masks 等长；只统计 value 非 None 的历元。
    """
    rows = []
    for lo, hi in bins:
        idx = [i for i, v in enumerate(values)
               if v is not None and lo <= v < hi]
        if not idx:
            continue
        n = len(idx)
        nf = sum(1 for i in idx if masks[i])
        label = (f'{lo:g}~{hi:g}' if hi < 900 else f'≥{lo:g}')
        rows.append({
            'label': label, 'lo': lo, 'hi': hi,
            'n': n, 'n_fix': nf, 'fix_rate': nf / n * 100.0,
        })
    return rows


def analyse_rtk_state(series, recovery_min_s=1.0):
    """RTK 时效与状态分析

    参数
    ----
    series : SolutionSeries
    recovery_min_s : float
        失锁最短时长（秒）。短于此的丢固定视为抖动，不计入失锁次数
        （但仍计入固定率），避免把 1 秒的状态抖动报成一堆失锁。

    返回
    ----
    dict:
        counts / shares     各解算类型的历元数与占比
        fix_rate            固定率（占全部历元，%）
        fix_rate_valid      固定率（占有解历元，%）
        ttff                首次固定：{'seconds','t','from','from_kind'}
        switches            解算状态切换次数
        losses              真失锁事件（超出 recovery_min_s 或未恢复）
        jitters             秒级抖动（非固定时段短于 recovery_min_s）
        warmups             捕获期（首次固定之前的时段，TTFF 衡量它）
        recovery            恢复耗时统计
        segments            固定段列表
        seg_stats           固定段统计（段数/最长/均值/中位）
        by_nsat/by_hdop/by_age  分档固定率
        timeline            供绘图的状态时序
    """
    n = len(series.t)
    if n == 0:
        return {'available': False, 'reason': '没有解算历元'}

    q = series.q
    masks = series.fix_mask()
    valids = series.valid_mask()
    n_fix = sum(1 for x in masks if x)
    n_valid = sum(1 for x in valids if x)

    counts = series.q_counts()
    shares = {k: v / n * 100.0 for k, v in counts.items()}

    # ---- 首次固定时间 ----
    ttff = None
    for i, m in enumerate(masks):
        if m:
            from_start = (series.t[i] - series.t[0]).total_seconds()
            first_valid = next((j for j, v in enumerate(valids) if v), None)
            from_valid = (None if first_valid is None
                          else (series.t[i] - series.t[first_valid]
                                ).total_seconds())
            ttff = {
                'seconds': from_start,
                'from_valid_s': from_valid,
                't': series.t[i],
                'from': series.t[0],
                'from_kind': '观测开始',
            }
            break

    # ---- 状态切换 ----
    switches = sum(1 for i in range(1, n) if q[i] != q[i - 1])

    # ---- 非固定片段：分成「捕获期」「抖动」「真失锁」三类 ----
    # 分类的意义在于不能张冠李戴：
    #   * 首次固定之前的时段是**捕获期**（warmup），它正是 TTFF 要衡量的
    #     对象，把它算成"失锁"是错的 —— 全程未固定时更不该报出 N 次失锁。
    #   * 一个历元的浮点确实只是**抖动**，但发生了就该让用户看见，
    #     只是不该混进"失锁次数"这个指标里；简单丢掉等于隐藏信息。
    first_fix_idx = next((k for k, m in enumerate(masks) if m), None)

    episodes = []
    i = 0
    while i < n:
        if masks[i]:
            i += 1
            continue
        start = i
        while i < n and not masks[i]:
            i += 1
        end = i - 1
        recovered = i < n
        span = (series.t[end] - series.t[start]).total_seconds()
        # 恢复耗时：从掉出固定到重新固定。未恢复时退化为片段自身跨度
        recovery_s = ((series.t[i] - series.t[start]).total_seconds()
                      if recovered else span)
        if first_fix_idx is None or start < first_fix_idx:
            kind = 'warmup'
        elif not recovered or recovery_s > recovery_min_s:
            kind = 'loss'
        else:
            kind = 'jitter'
        episodes.append({
            # 统一用 t_start / t_end 两个键（与 segments 一致），
            # 绘图/报告直接取用，避免两处键名不同而读错导致画错
            't_start': series.t[start],
            't_end': series.t[end],
            't_last': series.t[end],
            'n_epochs': end - start + 1,
            'span_s': span,
            'recovered': recovered,
            'recovery_s': recovery_s,
            'q_at_start': q[start],
            'kind': kind,
        })
    losses = [x for x in episodes if x['kind'] == 'loss']
    jitters = [x for x in episodes if x['kind'] == 'jitter']
    warmups = [x for x in episodes if x['kind'] == 'warmup']

    recov = sorted(x['recovery_s'] for x in losses
                   if x['recovered'] and x['recovery_s'] is not None)
    recovery = None
    if recov:
        recovery = {
            'n': len(recov), 'min_s': recov[0], 'max_s': recov[-1],
            'median_s': recov[len(recov) // 2],
            'p95_s': recov[min(len(recov) - 1, int(len(recov) * 0.95))],
        }

    # ---- 固定段 ----
    segments = []
    i = 0
    while i < n:
        if not masks[i]:
            i += 1
            continue
        start = i
        while i < n and masks[i]:
            i += 1
        dur = (series.t[i - 1] - series.t[start]).total_seconds()
        segments.append({
            't_start': series.t[start], 't_end': series.t[i - 1],
            'duration_s': dur, 'n_epochs': i - start,
        })

    seg_stats = None
    if segments:
        durs = sorted(x['duration_s'] for x in segments)
        seg_stats = {
            'n': len(segments), 'longest_s': durs[-1],
            'shortest_s': durs[0],
            'mean_s': sum(durs) / len(durs),
            'median_s': durs[len(durs) // 2],
            'total_s': sum(durs),
        }

    return {
        'available': True,
        'n_epochs': n,
        't_start': series.t[0], 't_end': series.t[-1],
        'duration_s': series.duration_s(),
        'median_interval_s': series.median_interval_s(),
        'counts': dict(counts),
        'shares': shares,
        'n_fix': n_fix,
        'n_valid': n_valid,
        'fix_rate': n_fix / n * 100.0,
        'fix_rate_valid': (n_fix / n_valid * 100.0) if n_valid else 0.0,
        'float_rate': shares.get(Q_FLOAT, 0.0),
        'single_rate': shares.get(1, 0.0),
        'dgps_rate': shares.get(2, 0.0),
        'valid_rate': n_valid / n * 100.0,
        'ttff': ttff,
        'switches': switches,
        'losses': losses,
        'n_losses': len(losses),
        'n_losses_unrecovered': sum(1 for x in losses if not x['recovered']),
        'jitters': jitters,
        'n_jitter': len(jitters),
        'warmups': warmups,
        'n_warmup': len(warmups),
        'n_episodes': len(episodes),
        'recovery': recovery,
        'segments': segments,
        'seg_stats': seg_stats,
        'by_nsat': _bin_fix_rate(series.nsat, masks, NSAT_BINS),
        'by_hdop': _bin_fix_rate(series.hdop, masks, HDOP_BINS),
        'by_age': _bin_fix_rate(series.age, masks, AGE_BINS),
        'timeline': {'t': series.t, 'q': list(q)},
        'recovery_min_s': recovery_min_s,
    }
