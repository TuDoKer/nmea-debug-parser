# -*- coding: utf-8 -*-
"""精度评估引擎：时间对齐 + 外符合/内符合精度 + 基线互比

关键概念（报告里必须分清，否则数字会被误读）
------------------------------------------
* **外符合精度（accuracy）**：相对**真值**的偏差。只有已知点、第三方
  后处理结果才谈得上。这才是 RTK 测试要的"精度"。
* **内符合精度（repeatability）**：相对**自身均值**的离散度。
  设备重复性再好，也可能整体偏了一大截。
* **相对一致性**：两台设备之间的偏差（基线互比）。两台可能同时有偏。

三者数值可能差很多，混在一起报就是耍流氓。本模块对每份结果都标注
``ref_absolute``，界面/报告据此说明口径。

ENU 原点的取法
-------------
**每个历元都以该时刻的参考点为原点**。对已知点（坐标恒定）就等价于
以已知点为原点；对移动参考（另一台设备）则得到相对该参考的偏差。
两种情形统一，不用分支。

误差指标口径
-----------
* 水平误差 = √(dE² + dN²)，是**径向量**（非 1D 分量）
* CEP50/68/95/99 = 水平误差的**经验分位**（不是 1D σ 乘系数）
* DRMS = √(mean(dE²) + mean(dN²))，2DRMS = 2×DRMS
* 高程用 |dU| 的分位与 RMS
* 3D 误差 = √(dE² + dN² + dU²)
"""

import math

from . import geodesy
from .numeric import mean, pstdev, quantile, rms
from .reference import SelfMeanRef, normalize_require, require_ok

# 默认的精度评价阈值（米），可按检验标准覆盖
DEFAULT_THRESHOLDS = [0.02, 0.05, 0.10, 0.20, 0.50]

# 分位数/均值/RMS 统一从 numeric 取，本文件不再自己实现一份。
# 别名成 _pct 是为了不改动大量既有调用点。
_pct = quantile
_rms = rms
_mean = mean


def _std(vals):
    """总体标准差；单点返回 0（不是 None，便于直接填进统计表）"""
    return pstdev(vals) if len(vals) > 1 else 0.0


# ==================== 时间对齐 ====================


class Pair:
    """一对"设备观测 / 参考"比对记录"""

    __slots__ = ('t', 'dev_lat', 'dev_lon', 'dev_h', 'ref_lat', 'ref_lon',
                 'ref_h', 'e', 'n', 'u', 'horiz', 'vert', 'sp3d',
                 'q', 'nsat', 'hdop', 'age', 'dt', 'ref_interpolated')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def align(device, ref_source, max_dt=None, require=None, t_from=None,
          t_to=None):
    """把设备解算序列与参考源按时间对齐

    参数
    ----
    device : SolutionSeries
    ref_source : ReferenceSource
    max_dt : float|None
        设备历元与参考**采样点**允许的最大时间差（秒）。
        None 时用参考源自己的设置。
    require : None | str | 集合
        只保留设备解算质量符合该要求的历元。
        用集合而不是阈值 —— **Q 不是有序量**（4 固定、5 浮动，而 5>4），
        ``q >= 4`` 会把浮动解也算进来。可直接写 ``'fix'`` / ``'fix_float'``
        / ``'valid'`` / ``(4,)`` / ``None``（不限）。
    t_from / t_to : datetime|None
        只统计该时段。

    返回
    ----
    (pairs, stats)
    """
    if max_dt is None:
        max_dt = getattr(ref_source, 'max_dt', 1.5)
    req = normalize_require(require)

    pairs = []
    n_dev = len(device.t)
    n_no_ref = n_filtered = n_nopos = 0
    dts = []

    for i in range(n_dev):
        t = device.t[i]
        q = device.q[i]
        if not require_ok(req, q):
            n_filtered += 1
            continue
        if t_from is not None and t < t_from:
            n_filtered += 1
            continue
        if t_to is not None and t > t_to:
            n_filtered += 1
            continue
        # **历元本身没有完整坐标**：无定位输出（GGA 定位质量 0 / 字段为空），
        # 或有经纬度但没有高度。
        # 这种历元留在序列里是正常的 —— 固定率统计仍要把它算进分母 ——
        # 但做精度比对必须**整条跳过**：
        #   * lat/lon 为 None 会让 blh_to_ecef(None) 直接 TypeError，
        #     界面只弹一句"对齐失败"，用户完全看不出原因；
        #   * h 为 None 更隐蔽 —— 若当 0 处理，高程误差会凭空多出
        #     一整个椭球高（实测约 18 m），悄悄污染垂直精度。
        #     宁可少一个历元，也不能让静默替换扭曲统计。
        if (device.lat[i] is None or device.lon[i] is None
                or device.h[i] is None):
            n_nopos += 1
            continue
        rp = ref_source.at(t)
        if rp is None:
            n_no_ref += 1
            continue
        # 时间差按参考的**原始采样时刻** src_t 算。SeriesRef 最近邻取数时
        # 参考值只在采样点上成立；若按请求时刻算，dt 恒为 0，容差失效。
        dt = abs((rp.src_t - t).total_seconds())
        if dt > max_dt:
            n_no_ref += 1
            continue
        # 参考点坐标缺失同样无法比对（如对比设备在该历元无解）
        if rp.lat is None or rp.lon is None or rp.h is None:
            n_no_ref += 1
            continue
        dts.append(dt)

        # 以该时刻的参考点为 ENU 原点
        e, n, u = geodesy.blh_diff_to_enu(
            device.lat[i], device.lon[i],
            device.h[i] if device.h[i] is not None else 0.0,
            rp.lat, rp.lon, rp.h)
        horiz = math.hypot(e, n)
        vert = u
        pairs.append(Pair(
            t=t, dev_lat=device.lat[i], dev_lon=device.lon[i],
            dev_h=device.h[i], ref_lat=rp.lat, ref_lon=rp.lon, ref_h=rp.h,
            e=e, n=n, u=u, horiz=horiz, vert=vert,
            sp3d=math.sqrt(e * e + n * n + u * u),
            q=q, nsat=device.nsat[i], hdop=device.hdop[i],
            age=device.age[i], dt=dt,
            ref_interpolated=getattr(rp, 'interpolated', False)))

    stats = {
        'n_dev': n_dev,
        'n_aligned': len(pairs),
        'n_no_ref': n_no_ref,
        'n_filtered': n_filtered,
        'n_nopos': n_nopos,
        'max_dt': max_dt,
        'require': (sorted(req) if req is not None else None),
        'dt_mean': _mean(dts),
        'dt_max': max(dts) if dts else None,
    }
    return pairs, stats


# ==================== 精度统计 ====================


def accuracy_stats(pairs, thresholds=None, reject='none',
                   reject_k=3.0, ref_absolute=True, ref_kind='',
                   ref_info='', align_stats=None):
    """由比对记录算精度指标

    参数
    ----
    reject : 'none' | 'sigma' | 'iqr'
        粗差剔除策略。**默认不剔除**并在报告里同时给出剔除后的结果 ——
        先剔粗差再报精度是自欺欺人，必须让用户看到剔了多少、影响多大。
    reject_k : float
        sigma 法用 MAD 估计 σ 再乘该系数；iqr 法用 Q3 + k×IQR。

    返回 dict，含 'rejected'（剔除后的对照）与 'thresholds' 超限统计。
    """
    if not pairs:
        return {'available': False, 'reason': '没有可对齐的历元'}

    thr = list(thresholds or DEFAULT_THRESHOLDS)

    def core(ps):
        es = [p.e for p in ps]
        ns = [p.n for p in ps]
        us = [p.u for p in ps]
        hs = sorted(p.horiz for p in ps)
        vs = sorted(abs(p.vert) for p in ps)
        ds = sorted(p.sp3d for p in ps)
        return {
            'n': len(ps),
            'e': {'mean': _mean(es), 'std': _std(es), 'rms': _rms(es),
                  'min': min(es), 'max': max(es),
                  'median': _pct(sorted(es), 0.5)},
            'n_ax': {'mean': _mean(ns), 'std': _std(ns), 'rms': _rms(ns),
                     'min': min(ns), 'max': max(ns),
                     'median': _pct(sorted(ns), 0.5)},
            'u': {'mean': _mean(us), 'std': _std(us), 'rms': _rms(us),
                  'min': min(us), 'max': max(us),
                  'median': _pct(sorted(us), 0.5)},
            'horiz': {
                'mean': _mean(hs), 'median': _pct(hs, 0.5),
                'rms': _rms(hs), 'min': hs[0], 'max': hs[-1],
                'cep50': _pct(hs, 0.50), 'cep68': _pct(hs, 0.68),
                'cep95': _pct(hs, 0.95), 'cep99': _pct(hs, 0.99),
                'drms': math.sqrt(_mean([x * x for x in es])
                                  + _mean([x * x for x in ns])),
            },
            'vert': {
                'mean_abs': _mean(vs), 'median_abs': _pct(vs, 0.5),
                'rms': _rms(us), 'p95_abs': _pct(vs, 0.95),
                'max_abs': vs[-1], 'bias': _mean(us),
            },
            'sp3d': {
                'mean': _mean(ds), 'median': _pct(ds, 0.5),
                'rms': _rms(ds), 'sep': _pct(ds, 0.50),
                'p95': _pct(ds, 0.95), 'max': ds[-1],
            },
        }

    base = core(pairs)
    base['horiz']['2drms'] = base['horiz']['drms'] * 2.0
    base['horiz']['3drms'] = base['horiz']['drms'] * 3.0

    result = {
        'available': True,
        'ref_absolute': ref_absolute,
        'ref_kind': ref_kind,
        'ref_info': ref_info,
        'align': align_stats or {},
        'thresholds': thr,
        'over': _exceedance(pairs, thr),
        'reject': reject,
        'rejected': None,
    }
    result.update(base)

    # ---- 粗差剔除（作为对照，不替代主结果）----
    if reject in ('sigma', 'iqr') and len(pairs) > 8:
        hs = sorted(p.horiz for p in pairs)
        if reject == 'sigma':
            med = _pct(hs, 0.5)
            mad = _pct(sorted(abs(x - med) for x in hs), 0.5)
            sigma = max(1.4826 * mad, 1e-9)
            keep = [p for p in pairs
                    if abs(p.horiz - med) <= reject_k * sigma]
            rule = f'中位 {med:.4f} m ± {reject_k:g}×1.4826×MAD({mad:.4f} m)'
        else:
            q1 = _pct(hs, 0.25)
            q3 = _pct(hs, 0.75)
            lim = q3 + reject_k * (q3 - q1)
            keep = [p for p in pairs if p.horiz <= lim]
            rule = f'≤ Q3 + {reject_k:g}×IQR = {lim:.4f} m'
        if 0 < len(keep) < len(pairs):
            sub = core(keep)
            sub['horiz']['2drms'] = sub['horiz']['drms'] * 2.0
            sub['horiz']['3drms'] = sub['horiz']['drms'] * 3.0
            sub['n_rejected'] = len(pairs) - len(keep)
            sub['reject_pct'] = (len(pairs) - len(keep)) / len(pairs) * 100
            sub['rule'] = rule
            result['rejected'] = sub

    return result


def _exceedance(pairs, thresholds):
    """各阈值下的超限历元数与比例"""
    rows = []
    n = len(pairs)
    for th in thresholds:
        nh = sum(1 for p in pairs if p.horiz > th)
        nv = sum(1 for p in pairs if abs(p.vert) > th)
        rows.append({
            'th': th, 'n_horiz': nh, 'n_vert': nv,
            'pct_horiz': nh / n * 100 if n else 0.0,
            'pct_vert': nv / n * 100 if n else 0.0,
        })
    return rows


def repeatability(series, require='fix', reject='sigma', reject_k=3.0):
    """内符合精度（重复性）：相对自身稳健均值的离散度

    用于自建参考场景。**不要把它当成精度报出去** —— 它衡量的是
    "每次是否落在同一个地方"，不含任何绝对基准。
    """
    try:
        smr = SelfMeanRef(series, require_q=require,
                          sigma_k=reject_k if reject == 'sigma' else 3.0)
    except ValueError as e:
        return {'available': False, 'reason': str(e)}
    req = normalize_require(require)
    to_enu = geodesy.make_local_frame(smr.lat, smr.lon, smr.h)
    # 用与稳健均值**同一套**历元：均值剔了粗差而离散度没剔，
    # 会让个别粗差把 RMS 抬高一两个数量级，两个数字口径不一致
    keep = set(smr.keep_epochs)
    pairs = []
    for i in range(len(series.t)):
        if i not in keep:
            continue
        if not require_ok(req, series.q[i]):
            continue
        if series.lat[i] is None or series.lon[i] is None:
            continue
        e, n, u = to_enu(series.lat[i], series.lon[i],
                         series.h[i] if series.h[i] is not None else 0.0)
        pairs.append(Pair(t=series.t[i], e=e, n=n, u=u,
                          horiz=math.hypot(e, n), vert=u,
                          sp3d=math.sqrt(e * e + n * n + u * u),
                          q=series.q[i], nsat=series.nsat[i],
                          hdop=series.hdop[i], age=series.age[i], dt=0.0))
    res = accuracy_stats(pairs, ref_absolute=False,
                         ref_kind='self_mean',
                         ref_info=smr.info(),
                         reject=reject, reject_k=reject_k)
    res['self_mean'] = {
        'lat': smr.lat, 'lon': smr.lon, 'h': smr.h,
        'n_used': smr.n_used, 'n_rejected': smr.rejects,
        'n_iter': smr.n_iter, 'converged': smr.converged,
    }
    res['note'] = ('已采用与稳健均值相同的历元子集，'
                   '粗差 %d 个已排除在重复性统计之外' % smr.rejects)
    return res


# ==================== 基线互比 ====================


def baseline_stats(series_a, series_b, max_dt=1.5, label_a='A', label_b='B',
                   require_a='fix', require_b='fix'):
    """两台设备的基线互比

    返回：
      * 基线长度时序与统计（均值/最小/最大/标准差）
      * 相对 ENU 偏差
      * 相对精度指标（与 accuracy_stats 同构）

    **方向约定：报的是 A 相对 B（即 a − b）**。写报告时要说清"以谁为基准"，
    否则符号会让人误解成设备偏了方向。
    **这是相对参考**，两台设备可能同时有偏，只能评估一致性。
    默认只比较两台都处于固定解的历元。
    """
    ref = _as_ref(series_b, label_b, require_b, max_dt)
    pairs, ast = align(series_a, ref, max_dt=max_dt, require=require_a)

    # 基线长度：逐历元算 A → B 的大地线距离
    baselines = []
    for p in pairs:
        baselines.append(geodesy.distance(p.dev_lat, p.dev_lon,
                                          p.ref_lat, p.ref_lon))
    res = accuracy_stats(pairs, ref_absolute=False, ref_kind='peer_device',
                         ref_info=f'{label_a} vs {label_b}（相对参考）',
                         align_stats=ast)
    res['label_a'] = label_a
    res['label_b'] = label_b
    if baselines:
        bs = sorted(baselines)
        res['baseline'] = {
            'mean': _mean(baselines), 'min': bs[0], 'max': bs[-1],
            'median': _pct(bs, 0.5), 'std': _std(baselines),
            'range': bs[-1] - bs[0],
            'n': len(bs),
        }
    else:
        res['baseline'] = None
    return res


def _as_ref(series, label, require_q, max_dt):
    from .reference import PeerDeviceRef
    return PeerDeviceRef(series, label=label, max_dt=max_dt,
                         require_q=require_q)


def aggregate_by_baseline(results):
    """把多份基线互比结果按基线长度归档，看精度随基线的变化

    results: [{'baseline': {...}, ...}, ...]（baseline_stats 的返回值）
    返回 [(基线代表值, 水平 RMS, 垂直 RMS, 历元数), ...] 并按基线排序。
    """
    rows = []
    for r in results:
        b = r.get('baseline')
        if not b or not r.get('available'):
            continue
        rows.append({
            'baseline_m': b['mean'],
            'horiz_rms': r['horiz']['rms'],
            'vert_rms': r['vert']['rms'],
            'cep95': r['horiz']['cep95'],
            'n': r['n'],
            'label_a': r.get('label_a', ''),
            'label_b': r.get('label_b', ''),
        })
    rows.sort(key=lambda x: x['baseline_m'])
    return rows


# ==================== 按条件分档 ====================

BIN_SETS = {
    'nsat': [(0, 15), (15, 20), (20, 25), (25, 30), (30, 999)],
    'hdop': [(0.0, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 999.0)],
    'age': [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 999.0)],
    'q': [(0, 2), (2, 4), (4, 5), (5, 6)],
}


def bin_by_condition(pairs, key='nsat', bins=None):
    """按卫星数/HDOP/龄期/解算质量分档看精度

    “什么条件下精度会掉”比一个总 RMS 有用得多 ——
    总 RMS 好的设备可能只是大部分时间条件好。
    """
    if not pairs:
        return []
    bins = bins or BIN_SETS.get(key)
    if not bins:
        return []
    rows = []
    for lo, hi in bins:
        idx = []
        for p in pairs:
            v = {'nsat': p.nsat, 'hdop': p.hdop, 'age': p.age,
                 'q': p.q}.get(key)
            if v is None:
                continue
            if lo <= v < hi:
                idx.append(p)
        if len(idx) < 2:
            continue
        hs = sorted(p.horiz for p in idx)
        us = [p.u for p in idx]
        label = f'{lo:g}~{hi:g}' if hi < 900 else f'≥{lo:g}'
        rows.append({
            'key': key, 'label': label, 'lo': lo, 'hi': hi,
            'n': len(idx),
            'horiz_rms': _rms([p.horiz for p in idx]),
            'horiz_cep95': _pct(hs, 0.95),
            'vert_rms': _rms(us),
            'vert_bias': _mean(us),
        })
    return rows
