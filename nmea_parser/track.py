# -*- coding: utf-8 -*-
"""动态轨迹分析：位移、速度航向、动静区分、重复线重合度

静态测试看的是"一个点上重复得准不准"，动态测试看的是
"沿轨迹走得准不准"。两者用的是不同的指标：

  * 静态：以真值为原点的**水平/垂直误差**分布
  * 动态：沿轨迹的**横向偏差（cross-track）**与纵向偏差，
    以及往返重复线的**重合度**

为什么横向/纵向要分开：轨迹上的误差方向不是等价的 ——
横向偏差直接反映在成图/放样上，纵向偏差往往被运动学吸收掉。
只报一个"水平误差"会把两者混在一起。
"""

import math

from . import geodesy
from .numeric import quantile as _pct
from .reference import normalize_require, require_ok


class TrackPoint:
    __slots__ = ('t', 'e', 'n', 'u', 'lat', 'lon', 'h', 'q',
                 'speed', 'course', 'moved')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def build_track(series, ref=None, require=None, static_thresh=0.05,
                speed_window=3):
    """把解算序列转成轨迹（ENU 平面点 + 速度 + 动静标记）

    参数
    ----
    ref : (lat, lon, h) | None
        ENU 原点。None 时用首个有效点作原点。
        **多条轨迹互比时必须显式给同一个原点** —— 各自取首点会把横向偏移
        抵消掉，比较结果恒为 0（本项目踩过）。
    require : None | str | 集合
        保留哪些解算质量。用集合而不是阈值 —— **Q 不是有序量**
        （4 固定、5 浮动，而 5 > 4），``q >= 4`` 会把浮动解也算进来。
        可写 ``'fix'`` / ``'fix_float'`` / ``'valid'`` / ``(4,)`` / ``None``。
    static_thresh : float
        判定"静止"的速度阈值（m/s）。默认 0.05 m/s ≈ 5 cm/s，
        能挡住 RTK 静态漂移造成的伪运动。
    speed_window : int
        速度用前后若干点的中心差分算，抑制定位噪声被放大成假速度。

    返回 list[TrackPoint]
    """
    n = len(series.t)
    if n == 0:
        return []

    req = normalize_require(require)
    idx = [i for i in range(n)
           if series.lat[i] is not None and series.lon[i] is not None
           and require_ok(req, series.q[i])]
    if not idx:
        return []

    if ref is None:
        i0 = idx[0]
        ref = (series.lat[i0], series.lon[i0],
               series.h[i0] if series.h[i0] is not None else 0.0)
    to_enu = geodesy.make_local_frame(*ref)

    pts = []
    for i in idx:
        e, nn, u = to_enu(series.lat[i], series.lon[i],
                          series.h[i] if series.h[i] is not None else 0.0)
        pts.append(TrackPoint(t=series.t[i], e=e, n=nn, u=u,
                              lat=series.lat[i], lon=series.lon[i],
                              h=series.h[i], q=series.q[i]))

    # 速度与航向：中心差分，窗口至少 ±1
    half = max(1, int(speed_window) // 2)
    for k, p in enumerate(pts):
        a = max(0, k - half)
        b = min(len(pts) - 1, k + half)
        dts = (pts[b].t - pts[a].t).total_seconds()
        if dts <= 0:
            p.speed = None
            p.course = None
            p.moved = False
            continue
        de = pts[b].e - pts[a].e
        dn = pts[b].n - pts[a].n
        # 窗口拉长后要按窗口内的平均速度算，不能直接用端点差除以窗长
        p.speed = math.hypot(de, dn) / dts
        p.course = math.degrees(math.atan2(de, dn)) % 360.0
        p.moved = p.speed > static_thresh

    # 端点没有窗口就退回单侧差分，避免首尾速度为空
    for k in (0, len(pts) - 1):
        if pts[k].speed is not None:
            continue
        j = 1 if k == 0 else len(pts) - 2
        if not (0 <= j < len(pts)):
            pts[k].speed = 0.0
            pts[k].course = 0.0
            pts[k].moved = False
            continue
        dts = abs((pts[k].t - pts[j].t).total_seconds())
        if dts <= 0:
            pts[k].speed = 0.0
            pts[k].course = 0.0
        else:
            de = pts[k].e - pts[j].e
            dn = pts[k].n - pts[j].n
            pts[k].speed = math.hypot(de, dn) / dts
            pts[k].course = math.degrees(math.atan2(de, dn)) % 360.0
        pts[k].moved = pts[k].speed > static_thresh
    return pts


def track_stats(pts):
    """轨迹总体统计：路径长度、包络、速度、动静占比"""
    if not pts:
        return {'available': False, 'reason': '没有轨迹点'}

    length = 0.0
    for k in range(1, len(pts)):
        length += math.hypot(pts[k].e - pts[k - 1].e, pts[k].n - pts[k - 1].n)
    es = [p.e for p in pts]
    ns = [p.n for p in pts]
    us = [p.u for p in pts]
    speeds = sorted(p.speed for p in pts if p.speed is not None)
    n_moved = sum(1 for p in pts if p.moved)
    dur = (pts[-1].t - pts[0].t).total_seconds()

    return {
        'available': True,
        'n': len(pts),
        'duration_s': dur,
        'length_m': length,
        'span_e': max(es) - min(es),
        'span_n': max(ns) - min(ns),
        'span_u': max(us) - min(us),
        'speed_mean': (sum(speeds) / len(speeds)) if speeds else None,
        'speed_median': _pct(speeds, 0.5),
        'speed_p95': _pct(speeds, 0.95),
        'speed_max': speeds[-1] if speeds else None,
        'n_moved': n_moved,
        'moving_pct': n_moved / len(pts) * 100.0,
        'static_pct': (len(pts) - n_moved) / len(pts) * 100.0,
    }


def split_segments(pts, min_len=5):
    """把轨迹按动静切成若干连续段

    返回 [{'kind':'static'|'moving', 'i0','i1','n','length_m',
           't_start','t_end','speed_mean'}, ...]
    段太短（< min_len 个点）的并入相邻段，避免噪声造成频繁抖动。
    """
    if not pts:
        return []
    raw = []
    i0 = 0
    for k in range(1, len(pts) + 1):
        if k == len(pts) or pts[k].moved != pts[i0].moved:
            raw.append([i0, k - 1])
            i0 = k
    # 合并过短的段
    merged = []
    for seg in raw:
        if merged and (seg[1] - seg[0] + 1) < min_len:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    out = []
    for a, b in merged:
        sub = pts[a:b + 1]
        length = sum(math.hypot(sub[j].e - sub[j - 1].e,
                                sub[j].n - sub[j - 1].n)
                     for j in range(1, len(sub)))
        sp = [p.speed for p in sub if p.speed is not None]
        out.append({
            'kind': 'moving' if sub[0].moved else 'static',
            'i0': a, 'i1': b, 'n': len(sub),
            'length_m': length,
            't_start': sub[0].t, 't_end': sub[-1].t,
            'speed_mean': (sum(sp) / len(sp)) if sp else None,
        })
    return out


def path_length(pts):
    return sum(math.hypot(pts[k].e - pts[k - 1].e, pts[k].n - pts[k - 1].n)
               for k in range(1, len(pts)))


def _point_to_polyline(p, poly):
    """点到折线的最短距离（局部平面内），返回 (距离, 最近段索引)

    用点到**线段**的距离而不是到顶点 —— 采点稀疏时用顶点距离会系统性偏大。
    """
    best_d = float('inf')
    best_i = -1
    for k in range(len(poly) - 1):
        ax, ay = poly[k]
        bx, by = poly[k + 1]
        vx, vy = bx - ax, by - ay
        wx, wy = p[0] - ax, p[1] - ay
        L2 = vx * vx + vy * vy
        if L2 <= 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / L2))
        dx = wx - t * vx
        dy = wy - t * vy
        d = math.hypot(dx, dy)
        if d < best_d:
            best_d = d
            best_i = k
    if len(poly) == 1:
        return math.hypot(p[0] - poly[0][0], p[1] - poly[0][1]), 0
    return best_d, best_i


def cross_track_error(path_ref, path_test, max_dist=50.0):
    """重复线重合度：测试轨迹相对参考轨迹的横向偏差

    逐点算测试轨迹到参考折线的最近距离。返回统计 + 逐点序列。

    注意方向性：结果不对称（A→B 与 B→A 数值可能不同），
    报告里要说明"以谁为基准"。这里以 ``path_ref`` 为基准。

    ``max_dist`` 用于剔除明显离群点（比如轨迹首尾错位、坐标跳变），
    超限的点计入 ``n_far`` 而不混进统计。
    """
    if len(path_ref) < 2 or len(path_test) < 2:
        return {'available': False, 'reason': '两条轨迹都至少需要 2 个点'}
    poly = [(p.e, p.n) for p in path_ref]
    ds = []
    n_far = 0
    detail = []
    for p in path_test:
        d, k = _point_to_polyline((p.e, p.n), poly)
        detail.append((p.t, d, k))
        if d > max_dist:
            n_far += 1
        else:
            ds.append(d)
    if not ds:
        return {'available': False, 'reason': '没有可用的重合度样本',
                'n_far': n_far}
    ss = sorted(ds)

    return {
        'available': True,
        'n': len(ds), 'n_far': n_far,
        'mean': sum(ds) / len(ds),
        'median': _pct(ss, 0.5),
        'rms': math.sqrt(sum(x * x for x in ds) / len(ds)),
        'p95': _pct(ss, 0.95),
        'max': ss[-1],
        'min': ss[0],
        'detail': detail,
        'ref_len_m': path_length(path_ref),
        'test_len_m': path_length(path_test),
    }


def repeat_run_stats(runs):
    """多次重复线：两两比较重合度，给出汇总

    runs: [(label, pts), ...]
    以第一条为基准，也可给出任意两两组合。返回汇总表 + 逐对明细。
    """
    if len(runs) < 2:
        return {'available': False, 'reason': '至少需要两条轨迹'}
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            la, pa = runs[i]
            lb, pb = runs[j]
            r = cross_track_error(pa, pb)
            if r.get('available'):
                pairs.append({'a': la, 'b': lb, 'median': r['median'],
                              'mean': r['mean'], 'rms': r['rms'],
                              'p95': r['p95'], 'max': r['max'],
                              'n': r['n'],
                              'len_a': r['ref_len_m'], 'len_b': r['test_len_m']})
    if not pairs:
        return {'available': False, 'reason': '没有可比较的轨迹对'}
    meds = sorted(p['median'] for p in pairs)
    return {
        'available': True,
        'npairs': len(pairs),
        'pairs': pairs,
        'median_of_medians': meds[len(meds) // 2],
        'worst_median': meds[-1],
        'best_median': meds[0],
    }
