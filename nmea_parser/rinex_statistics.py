# -*- coding: utf-8 -*-
"""静态（基准站）观测数据质量分析（纯逻辑，无 GUI 依赖）

在 rinex.py 解析结果之上计算:
  * 文件概况: 时段 / 历元数 / 采样间隔 / 卫星数 / 系统构成
  * 观测值缺失(零值)率 —— 逐观测码
  * C/N0（信噪比）统计 —— 按系统-频点、按卫星
  * 仰角/方位角分布 —— 需要广播星历
  * C/N0-仰角基线残差 —— 剔除几何因素后找出真正信号偏弱的卫星
  * 卫星连续性/完整率 + 中断归因 —— 区分"升落遮挡"与"真实失锁"
  * 多路径 MP (GPS L1/L2 无电离层组合) RMS

核心方法说明:
  「仰角分层基线残差」先按 5° 仰角分段统计全站 C/N0 基线，
  再用每颗卫星自己的仰角分布去查基线得到"期望 C/N0"，
  实测均值 - 期望均值的差值即为残差。残差显著为负 = 该卫星信号质量有问题
  （而不是仅仅因为它飞得低）。这比单看平均 C/N0 更可靠。
"""

import collections
import math
import statistics

from .numeric import quantile as _quantile
from .rinex import (
    C_LIGHT, F_B1I, F_B2I, F_B3I, F_E1, F_E5A, F_E5B, F_L1, F_L2, band_of,
    day0_timestamp, doy_of, is_doppler,
    is_phase, is_pseudorange, is_snr, sat_label, sys_name,
)

# 各系统各频点的标称载波频率（Hz），用于几何无关组合
# 注：仅用于周跳判据 GF，对频率的少许偏差不敏感（GLONASS 是 FDMA，
#     卫星间 L1 有 ±0.35% 的差异，标称中心频率即可）。
_BAND_FREQ = {
    'G': {'L1': 1575.42e6, 'L2': 1227.60e6, 'L5': 1176.45e6},
    'R': {'L1': 1602.00e6, 'L2': 1246.00e6, 'L3': 1202.025e6},
    'E': {'L1': 1575.42e6, 'L5': 1176.45e6, 'L6': 1278.75e6,
          'L7': 1207.14e6, 'L8': 1191.795e6},
    'C': {'L1': 1561.098e6, 'L2': 1207.14e6, 'L5': 1176.45e6,
          'L6': 1268.52e6, 'L7': 1207.14e6},
    'J': {'L1': 1575.42e6, 'L2': 1227.60e6, 'L5': 1176.45e6,
          'L6': 1278.75e6},
    'S': {'L1': 1575.42e6, 'L5': 1176.45e6},
    'I': {'L5': 1176.45e6},
}

# 几何无关组合优先使用的频点对（按优先级）
_GF_BAND_PAIRS = [('L1', 'L2'), ('L1', 'L5'), ('L2', 'L5'), ('L1', 'L7'),
                  ('L1', 'L6'), ('L7', 'L5'), ('L6', 'L5')]

# 多系统多路径 MP / 电离层 ROT 使用的 频率对 与 观测码优先级
# 观测码按优先级排列：接收机导出的码名五花八门（老式 C1/P2 与
# RINEX 3 的 C2I/C7I 混用），按前缀逐个尝试，取第一个存在的。
# GLONASS（R）不在此列：FDMA 每颗卫星频率不同，需要 RINEX 头里的
# GLONASS SLOT / FRQ # 频道号才能确定频率比，没有它硬算等于把
# 频率比当常数用，结果是垃圾 —— 明确跳过而不是给错误数字。
_MP_SYSTEM_PREFS = {
    'G': {'f1': [F_L1], 'f2': [F_L2],
          'c1': ['C1', 'P1'], 'l1': ['L1'],
          'c2': ['P2', 'C2'], 'l2': ['L2']},
    # Galileo 的第二频点写进 L2 列的可能是 E5b 也可能是 E5a
    # （部分接收机是 E5b=1207.14，不是 E5a）—— 不能猜，见
    # _resolve_system_freq：按 GF 跨度自动判定
    'E': {'f1': [F_E1], 'f2': [F_E5B, F_E5A],
          'c1': ['C1'], 'l1': ['L1'],
          'c2': ['P2', 'C5', 'C7'], 'l2': ['L2', 'L5', 'L7']},
    'C': {'f1': [F_B1I, F_E1], 'f2': [F_B2I, F_B3I],
          'c1': ['C2', 'C1'], 'l1': ['L2', 'L1'],
          'c2': ['C7', 'C6'], 'l2': ['L7', 'L6']},
}


# 频率对自动判定用的 GF 跨度上限（米）。真双频实测 < 62 m，
# 频率配错 > 6000 m，取 200 m 有足够余量
_MP_GF_SPAN_MAX = 200.0


def _iono_level(rot_med):
    """电离层活动分级（TECU/min，经验值）

    安静 < 1 <= 中扰 < 5 <= 扰动。阈值是工程经验值（IGS 文献里
    常见的量级划分），不是严格标准 —— 用途是给用户一个
    "今天环境好不好"的直觉判断。
    """
    if rot_med is None:
        return ('—', 'gray')
    if rot_med < 1.0:
        return ('安静', '#2a7a2a')
    if rot_med < 5.0:
        return ('中等', '#c77e00')
    return ('扰动', '#c0392b')


def known_point_offset(xyz, lat, lon, h=0.0, antenna_h=0.0):
    """RINEX 头近似坐标 vs 已知控制点 的平面/高程偏差（单点比对）

    为什么只有单点：RINEX 观测文件里**没有逐历元坐标**（整个格式只有
    文件头一个近似坐标），做不出误差时序、箱型图或散点图 ——
    那类分析需要坐标序列，来源是 NMEA 日志（GGA）或 RTKPOST 的 .pos。

    参数
    ----
    xyz : RINEX 头的 APPROX POSITION XYZ（ECEF，米）
    lat/lon/h : 已知控制点（十进制度 / 椭球高米）
    antenna_h : 天线高改正（相位中心 − 标石面，正值表示相位中心更高）。
        RINEX 头的坐标是**天线**的位置，而控制点成果通常指标石面 ——
        不加天线高直接比，高程会凭空差一个天线高
        （常见 1~2 m，正是"高程怎么差这么多"的经典原因）。

    返回 dict：e/n/u（ENU 偏差，米）、horiz（平面误差）、vert（高程偏差）。
    """
    from . import geodesy
    if not xyz:
        raise ValueError('没有近似坐标（APPROX POSITION XYZ）')
    kx, ky, kz = geodesy.blh_to_ecef(lat, lon, (h or 0.0) + (antenna_h or 0.0))
    e, n, u = geodesy.ecef_delta_to_enu(xyz[0] - kx, xyz[1] - ky,
                                        xyz[2] - kz, lat, lon)
    return {
        'e': e, 'n': n, 'u': u,
        'horiz': math.hypot(e, n),
        'vert': u,
    }

_AL = F_L1 ** 2 / (F_L1 ** 2 - F_L2 ** 2)
_BE = -F_L2 ** 2 / (F_L1 ** 2 - F_L2 ** 2)

# 多路径/残差统计的最小样本量（样本太少统计不可信，直接忽略）
MIN_SAMPLES_SAT = 100
MIN_SAMPLES_BAND = 50
MIN_SAMPLES_RESIDUAL = 200

# 判定"两频点载波列是否自洽"的 GF 跨度阈值（米）
#
# 几何无关组合 GF = λ₁·L₁ − λ₂·L₂。若接收机没跟踪第二频点、直接把第一频点的
# 值填进第二频点列，GF 会退化成 L₁·(λ₁−λ₂)，而 L₁ ≈ 1e8 周 —— 跨度是
# **百万米级**；真双频时 GF 只剩电离层延迟差与模糊度常数，跨度是米级
# （实测四个测站为 0.16 ~ 61.6 m）。两者相差六个数量级，
# 取 1000 m 作阈值既安全又宽松。
GF_DUPLICATE_SPAN_M = 1000.0

# 计算 DOP 时使用的仰角掩蔽角（低于此仰角不参与解算，与 RTKPLOT 默认口径一致）
ELEV_MASK_DEG = 5.0

# 周跳判据：几何无关组合 GF 的历元间跳变门限（米）
# GF 噪声实测只有 0.4 ~ 4 mm，而一个 1 周周跳会造成 λ ≈ 0.19 / 0.24 m 的跳变，
# 两者相差两个数量级，取 5 cm 既有极高灵敏度又留足余量。
SLIP_GF_THRESHOLD = 0.05
# 判定"该卫星 GF 是否干净可用"的门限：中位跳变超过它说明 L1/L2 列本身有问题
SLIP_GF_CLEAN = 0.02
# 周跳必须与"整周"相符：GF 跳变应等于 n·λ1 或 -n·λ2。
# 残差超过该值（单位：周）说明不是整周跳变，多半是电离层扰动或噪声，
# 直接丢弃而不是硬凑一个整数周数。
# 取 0.10 周：GF 实测噪声只有 0.0018 m ≈ 0.01 周，0.10 已是 10 倍余量；
# 再放宽会显著混入非整周的电离层/多路径扰动。
# （随机值的通过率参考：容差 t 下 min 二均匀量的通过率为 1-(1-2t)²，
#   即 t=0.10 时随机值约 36% 通过 —— 所以判别力主要靠下面
#   「符合整周比例」这个统计量，而不是单个事件。）
SLIP_FIT_TOL = 0.10
# 超过此幅值（米）的 GF 跳变判为「丢锁/大跳变」而非普通周跳
SLIP_LOSS_THRESHOLD = 1.0
# 持续性跟踪异常判据：跳变次数够多，且「符合整周」的比例接近随机水平。
# 真周跳必然 n·λ，该比例应明显高于随机（容差 0.10 周时随机约 36%）；
# 若接近随机，说明该卫星在这段时间反复丢锁/输出非整周相位不连续，
# 此时它那些"碰巧符合整周"的跳变只是噪声过筛，不能计入周跳总数。
SLIP_CHRONIC_MIN = 50        # 最少跳变次数
SLIP_CHRONIC_RATIO = 50.0    # 符合整周比例上限（%）


def _inv4(m):
    """4x4 矩阵求逆（高斯-约当消元）。奇异时返回 None。

    只用于 DOP 计算里的 (GᵀG)⁻¹，规模固定为 4，无需通用实现。
    """
    n = 4
    a = [[float(m[i][j]) for j in range(n)] + [1.0 if i == j else 0.0
                                               for j in range(n)]
         for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
        d = a[col][col]
        a[col] = [x / d for x in a[col]]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col]
            if f:
                a[r] = [a[r][k] - f * a[col][k] for k in range(2 * n)]
    return [[a[i][n + j] for j in range(n)] for i in range(n)]


def _fmt_time(t):
    if not t:
        return '—'
    y, mo, d, hh, mi, ss = t
    return f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mi:02d}:{int(ss):02d}"


class StaticQuality:
    """静态观测数据质量分析器"""

    def __init__(self, rinex_obs, ephemeris=None):
        self.rnx = rinex_obs
        self.eph = ephemeris
        self.result = {}

    # ==================== 主入口 ====================

    def compute(self, progress_callback=None):
        """执行全部分析。progress_callback(percent:int, message:str)"""
        rnx = self.rnx
        epochs = rnx.epochs

        if not epochs:
            self.result = {'error': '观测文件中没有解析到任何历元数据。\n'
                                    '请确认文件是完整的 RINEX 观测文件（.26O/.rnx 等）。'}
            return self.result

        def report(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        report(5, '统计文件概况...')
        info = self._file_info(epochs)

        report(20, '统计观测值缺失情况...')
        missing = self._missing_stats(epochs)

        report(35, '统计信噪比 C/N0...')
        snr = self._snr_stats(epochs)

        report(55, '计算卫星仰角/方位角...')
        geometry = self._geometry_stats(epochs)

        report(75, '计算 C/N0-仰角基线残差...')
        residual = self._residual_stats(geometry)

        report(85, '统计连续性与中断...')
        continuity = self._continuity_stats(epochs)

        report(93, '计算多路径 MP...')
        multipath = self._multipath_stats(epochs)

        report(94, '计算多系统多路径 MP...')
        multipath_sys = self._multipath_sys_stats(epochs)

        report(96, '构建时间序列...')
        timeseries = self._timeseries_stats(epochs)

        report(97, '计算 DOP / 可见卫星数 / 仰角散点...')
        diag = self._diagnostics_stats(epochs)

        report(98, '提取原始观测量并检测周跳...')
        obsd = self._obs_series_stats(epochs)

        report(98, '单点定位解算高程时序（SPP）...')
        from .spp import compute_height_series
        height_series = compute_height_series(
            rnx, self.eph, approx_xyz=self._observer_xyz())

        report(99, '计算 MP-仰角相关与电离层 ROT...')
        mp_elev = self._mp_elev_corr(multipath_sys, geometry,
                                     diag['scatter'].get('mp') or {})
        iono = self._iono_stats(epochs, geometry)

        report(100, '汇总...')
        self.result = {
            'info': info,
            'missing': missing,
            'snr': snr,
            'snr_band': snr,
            'snr_sat': snr,
            'geometry': geometry,
            'residual': residual,
            'continuity': continuity,
            'multipath': multipath,
            'multipath_sys': multipath_sys,
            'mp_elev': mp_elev,
            'iono': iono,
            'timeseries': timeseries,
            'dop_nsat': diag['dop'],
            'elev_scatter': diag['scatter'],
            'obs_series': obsd['series'],
            'cycle_slips': obsd['slips'],
            'height_series': height_series,
            'observer': self._observer_xyz(),
            'has_ephemeris': bool(self.eph and self.eph.epochs),
        }
        self.result['warnings'] = build_warnings(self.result)
        report(100, '分析完成')
        return self.result

    # ==================== 概况 ====================

    def _observer_xyz(self):
        """测站 ECEF 坐标: 优先头文件近似坐标"""
        rnx = self.rnx
        if rnx.approx_xyz:
            return tuple(rnx.approx_xyz)
        return None

    def _file_info(self, epochs):
        rnx = self.rnx
        n_ep = len(epochs)
        t0, t1 = rnx.time_span()

        # 采样间隔分布
        secs = [e[0] for e in epochs]
        diffs = [round(secs[i] - secs[i - 1]) for i in range(1, len(secs))]
        interval_hist = collections.Counter(d for d in diffs if d > 0)
        modal_interval = None
        if interval_hist:
            modal_interval = interval_hist.most_common(1)[0][0]

        # 卫星集合与系统构成
        per_epoch_count = []
        sat_records = collections.Counter()
        for _, satdict in epochs:
            per_epoch_count.append(len(satdict))
            for satid in satdict:
                sat_records[satid] += 1

        sys_sats = collections.Counter()
        for satid in sat_records:
            sys_sats[satid[0]] += 1
        sys_records = collections.Counter()
        for satid, n in sat_records.items():
            sys_records[satid[0]] += n

        n_rec = sum(sat_records.values())

        return {
            'marker': rnx.marker,
            'version': rnx.version,
            'file': rnx.filepath,
            'xyz': rnx.approx_xyz,
            'ant_delta': rnx.antenna_delta,
            't_first': t0,
            't_last': t1,
            't_first_str': _fmt_time(t0),
            't_last_str': _fmt_time(t1),
            'declared_interval': rnx.interval,
            'modal_interval': modal_interval,
            'interval_hist': dict(sorted(interval_hist.items())[:8]),
            'n_epochs': n_ep,
            'n_records': n_rec,
            'n_sats': len(sat_records),
            'avg_sats_per_epoch': n_rec / n_ep if n_ep else 0,
            'min_sats_per_epoch': min(per_epoch_count) if per_epoch_count else 0,
            'max_sats_per_epoch': max(per_epoch_count) if per_epoch_count else 0,
            'per_epoch_counts': per_epoch_count,
            'epoch_secs': secs,
            'sys_sat_count': {sys_name(k): v for k, v in sys_sats.items()},
            'sys_rec_count': {sys_name(k): v for k, v in sys_records.items()},
            'sys_share': {sys_name(k): v / n_rec * 100 for k, v in sys_records.items()},
            'obstypes': {k or 'RINEX2': list(v) for k, v in rnx.obstypes.items()},
        }

    # ==================== 观测值缺失 ====================

    def _missing_stats(self, epochs):
        """逐观测码统计零值/缺失比例。

        RINEX 规范中，缺失的观测值写 0.0。对某系统某观测码:
            零值率 = 该码为 0 或缺失的次数 / 该系统出现的历元次数
        高零值率通常说明接收机没有该频点能力，或该频点被关闭。
        """
        code_num = collections.Counter()
        code_zero = collections.Counter()
        sys_epoch_seen = collections.Counter()
        code_seen = collections.defaultdict(set)

        for _, satdict in epochs:
            for satid, vals in satdict.items():
                sysc = satid[0]
                sys_epoch_seen[sysc] += 1
                codes = self.rnx.obs_codes(sysc)
                for code in codes:
                    code_num[(sysc, code)] += 1
                    if vals.get(code, 0.0) == 0.0:
                        code_zero[(sysc, code)] += 1
                    else:
                        code_seen[sysc].add(code)

        rows = []
        for (sysc, code), total in code_num.items():
            zero = code_zero[(sysc, code)]
            rows.append({
                'sys': sys_name(sysc),
                'code': code,
                'total': total,
                'zero': zero,
                'rate': zero / total * 100 if total else 0.0,
            })
        rows.sort(key=lambda r: (-r['rate'], r['sys'], r['code']))

        # 该系统的接收机是否根本不输出某频点（几乎全零）
        dead = [r for r in rows if r['rate'] > 95 and r['total'] > 100]
        return {'rows': rows, 'dead_codes': dead}

    # ==================== 信噪比 ====================

    def _snr_stats(self, epochs):
        """按 系统-频点 与 逐卫星 统计 S 观测值（C/N0, dB-Hz）"""
        band_vals = collections.defaultdict(list)
        sat_vals = collections.defaultdict(list)
        sat_sysband = collections.defaultdict(lambda: collections.defaultdict(list))

        # 每个系统缓存观测码 -> 归纳频点
        # 注意: RINEX 2.10 的所有系统共用同一张观测码表，存在 obstypes[''] 下，
        #       因此按卫星前缀查不到时需要回退到 ''。
        sys_snr_codes = {}
        for sysc in set(list(self.rnx.obstypes.keys()) + ['']):
            codes = self.rnx.obs_codes(sysc)
            sys_snr_codes[sysc] = [(c, band_of(c)) for c in codes if is_snr(c)]

        for _, satdict in epochs:
            for satid, vals in satdict.items():
                sysc = satid[0]
                codes = sys_snr_codes.get(sysc) or sys_snr_codes.get('', [])
                for code, band in codes:
                    v = vals.get(code)
                    if v and v > 0:
                        band_vals[(sysc, band)].append(v)
                        sat_vals[satid].append(v)
                        sat_sysband[satid][band].append(v)

        band_rows = []
        for (sysc, band), arr in band_vals.items():
            if len(arr) < MIN_SAMPLES_BAND:
                continue
            a = sorted(arr)
            band_rows.append({
                'sys': sys_name(sysc),
                'band': band,
                'label': f"{sys_name(sysc)}-{band}",
                'n': len(a),
                'min': a[0], 'max': a[-1],
                'p5': _quantile(a, 0.05), 'p50': _quantile(a, 0.50),
                'p95': _quantile(a, 0.95),
                'mean': statistics.mean(a),
            })
        band_rows.sort(key=lambda r: r['mean'])

        sat_rows = []
        for satid, arr in sat_vals.items():
            if len(arr) < MIN_SAMPLES_SAT:
                continue
            a = sorted(arr)
            sat_rows.append({
                'sat': satid,
                'label': sat_label(satid),
                'sys': sys_name(satid[0]),
                'n': len(a),
                'mean': statistics.mean(a),
                'median': _quantile(a, 0.50),
                'lt25': sum(1 for x in a if x < 25) / len(a) * 100,
                'lt30': sum(1 for x in a if x < 30) / len(a) * 100,
                'lt35': sum(1 for x in a if x < 35) / len(a) * 100,
                'per_band': {b: statistics.mean(v)
                             for b, v in sat_sysband[satid].items() if v},
            })
        sat_rows.sort(key=lambda r: r['mean'])

        allv = [x for arr in band_vals.values() for x in arr]
        overall = None
        if allv:
            a = sorted(allv)
            overall = {
                'n': len(a),
                'mean': statistics.mean(a),
                'median': _quantile(a, 0.50),
                'min': a[0], 'max': a[-1],
                'lt25': sum(1 for x in a if x < 25) / len(a) * 100,
                'lt30': sum(1 for x in a if x < 30) / len(a) * 100,
                'lt35': sum(1 for x in a if x < 35) / len(a) * 100,
                'ge40': sum(1 for x in a if x >= 40) / len(a) * 100,
                # 接收机量化上限：若大量样本顶格在最大值，说明信噪比被硬件截断
                'at_ceiling': sum(1 for x in a if x >= a[-1]) / len(a) * 100,
                'ceiling': a[-1],
            }

        return {'bands': band_rows, 'sats': sat_rows, 'overall': overall}

    # ==================== 几何（仰角/方位角） ====================

    def _geometry_stats(self, epochs):
        """用广播星历计算每颗卫星的仰角/方位角分布

        为了控制耗时，按 step 抽稀历元（默认每 30 个历元算一次）。
        没有星历时返回 available=False。
        """
        if not self.eph or not self.eph.epochs:
            return {'available': False}

        xyz = self._observer_xyz()
        if not xyz:
            return {'available': False, 'reason': '头文件缺少近似坐标'}

        from .rinex import LocalFrame
        frame = LocalFrame(*xyz)

        t0, _ = self.rnx.time_span()
        if not t0:
            return {'available': False, 'reason': '无法确定观测日期'}
        day0 = day0_timestamp(t0[0], doy_of(t0[0], t0[1], t0[2]))

        step = max(1, len(epochs) // 400)     # 最多算约 400 个历元

        elev_hist = collections.Counter()
        azim_hist = collections.Counter()
        sat_elev = collections.defaultdict(list)
        sat_az = collections.defaultdict(list)
        n_ok = n_fail = 0

        for ei in range(0, len(epochs), step):
            secs, satdict = epochs[ei]
            t = day0 + secs
            for satid in satdict:
                if satid[0] == 'R':           # GLONASS 星历未解析
                    continue
                d = self.eph.best(satid, t)
                if d is None:
                    n_fail += 1
                    continue
                try:
                    sx, sy, sz = self.eph.sat_ecef(d, t)
                except (ValueError, TypeError, ZeroDivisionError):
                    n_fail += 1
                    continue
                r = frame.az_el(sx, sy, sz)
                if r is None:
                    n_fail += 1
                    continue
                az, el, _ = r
                n_ok += 1
                elev_hist[int(math.floor(el / 5)) * 5] += 1
                azim_hist[int(az // 30) * 30] += 1
                sat_elev[satid].append(el)
                sat_az[satid].append(az)

        total = sum(elev_hist.values())
        if total == 0:
            return {'available': False, 'reason': '星历与观测时间不匹配'}

        per_sat = []
        for satid, els in sat_elev.items():
            if len(els) < 20:
                continue
            azs = sat_az[satid]
            # 方位角圆周均值（处理 0/360 环绕）
            azm = math.degrees(math.atan2(
                sum(math.sin(math.radians(a)) for a in azs) / len(azs),
                sum(math.cos(math.radians(a)) for a in azs) / len(azs))) % 360
            per_sat.append({
                'sat': satid,
                'label': sat_label(satid),
                'sys': sys_name(satid[0]),
                'n': len(els),
                'el_min': min(els), 'el_max': max(els),
                'el_mean': statistics.mean(els),
                'az_mean': azm,
                'lt15': sum(1 for e in els if e < 15) / len(els) * 100,
            })
        per_sat.sort(key=lambda r: r['el_mean'])

        def pct_lt(limit):
            return sum(v for k, v in elev_hist.items() if k < limit) / total * 100

        return {
            'available': True,
            'elev_hist': {k: v for k, v in sorted(elev_hist.items())},
            'azim_hist': {k: v for k, v in sorted(azim_hist.items())},
            'per_sat': per_sat,
            'n_computed': n_ok, 'n_failed': n_fail,
            'total': total,
            'lt10': pct_lt(10), 'lt15': pct_lt(15),
            'ge30': sum(v for k, v in elev_hist.items() if k >= 30) / total * 100,
            'el_range': (min(elev_hist), max(elev_hist) + 5),
            'elev_step': step,
        }

    # ==================== C/N0 仰角基线残差 ====================

    def _residual_stats(self, geometry):
        """仰角分层基线残差 —— 本工具的核心诊断方法

        步骤:
          1. 按 5° 仰角分段，汇集所有卫星的 C/N0 样本 -> 得到"该仰角下的正常水平"
          2. 对每颗卫星，用其自身仰角分布去基线查表并求期望 C/N0
          3. 残差 = 实测均值 - 期望均值
        残差显著为负 => 排除几何因素后该星确实偏弱。
        """
        if not geometry.get('available'):
            return {'available': False}

        if not self.eph or not self.eph.epochs:
            return {'available': False}

        xyz = self._observer_xyz()
        if not xyz:
            return {'available': False}
        from .rinex import LocalFrame
        frame = LocalFrame(*xyz)

        t0, _ = self.rnx.time_span()
        if not t0:
            return {'available': False}
        day0 = day0_timestamp(t0[0], doy_of(t0[0], t0[1], t0[2]))

        epochs = self.rnx.epochs
        step = max(1, len(epochs) // 400)

        # 每个系统选一个 S 观测码（与 snr_stats 一致，第一个 S 码）
        # RINEX 2.10 所有系统共用 obstypes['']，需要回退
        sys_snr_code = {}
        for sysc in set(list(self.rnx.obstypes.keys()) + ['']):
            for c in self.rnx.obs_codes(sysc):
                if is_snr(c):
                    sys_snr_code[sysc] = c
                    break

        el_band_vals = collections.defaultdict(list)
        sat_pairs = collections.defaultdict(list)      # satid -> [(el, snr)]

        for ei in range(0, len(epochs), step):
            secs, satdict = epochs[ei]
            t = day0 + secs
            for satid, vals in satdict.items():
                if satid[0] == 'R':
                    continue
                code = sys_snr_code.get(satid[0]) or sys_snr_code.get('')
                if not code:
                    continue
                snr = vals.get(code)
                if not snr or snr <= 0:
                    continue
                d = self.eph.best(satid, t)
                if d is None:
                    continue
                try:
                    sx, sy, sz = self.eph.sat_ecef(d, t)
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
                r = frame.az_el(sx, sy, sz)
                if r is None:
                    continue
                el = r[1]
                el_band_vals[int(math.floor(el / 5)) * 5].append(snr)
                sat_pairs[satid].append((el, snr))

        # 基线
        baseline = {}
        for band, arr in el_band_vals.items():
            if len(arr) >= MIN_SAMPLES_BAND:
                baseline[band] = statistics.mean(arr)

        if not baseline:
            return {'available': False}

        baseline_rows = [{'band': k, 'n': len(el_band_vals[k]), 'mean': v}
                         for k, v in sorted(baseline.items())]

        # 残差
        resid_rows = []
        for satid, pairs in sat_pairs.items():
            if len(pairs) < MIN_SAMPLES_RESIDUAL:
                continue
            els = [p[0] for p in pairs]
            snrs = [p[1] for p in pairs]
            expect = []
            for e in els:
                b = int(math.floor(e / 5)) * 5
                cands = [k for k in baseline if abs(k - b) <= 10]
                if cands:
                    expect.append(baseline[min(cands, key=lambda k: abs(k - b))])
            if not expect:
                continue
            real = statistics.mean(snrs)
            exp = statistics.mean(expect)
            resid_rows.append({
                'sat': satid,
                'label': sat_label(satid),
                'sys': sys_name(satid[0]),
                'n': len(pairs),
                'el_mean': statistics.mean(els),
                'real': real,
                'expect': exp,
                'residual': real - exp,
            })
        resid_rows.sort(key=lambda r: r['residual'])

        return {
            'available': True,
            'baseline': baseline_rows,
            'sats': resid_rows,
            'elev_step': step,
        }

    # ==================== 连续性 / 中断归因 ====================

    def _continuity_stats(self, epochs):
        """逐卫星连续性: 完整率 + 中断归因

        完整率 = 观测历元数 / (末历元 - 首历元 + 1)
        高完整率但首末有长空白 => 正常升落（被 15° 截止高度角截断）
        中间出现长中断 => 真实失锁
        """
        n_ep = len(epochs)
        occ = collections.defaultdict(list)
        for ei, (_, satdict) in enumerate(epochs):
            for satid in satdict:
                occ[satid].append(ei)

        rows = []
        for satid, epl in occ.items():
            epl = sorted(set(epl))
            first, last = epl[0], epl[-1]
            span = last - first + 1
            gaps = []
            gap_segs = []          # [(起始历元, 结束历元), ...] 供完整性图使用
            for i in range(1, len(epl)):
                d = epl[i] - epl[i - 1]
                if d > 1:
                    gaps.append(d - 1)
                    gap_segs.append((epl[i - 1] + 1, epl[i] - 1))
            max_gap = max(gaps) if gaps else 0
            rate = len(epl) / span * 100 if span else 0.0
            head = first
            tail = n_ep - 1 - last

            # 归因
            if max_gap > 100:
                verdict = f"真实失锁: 存在 {max_gap} 历元（约 {max_gap / 60:.1f} 分钟）连续中断"
                kind = 'loss_of_lock'
            elif head > 30 and tail > 30:
                verdict = f"时段中段被遮挡: 前后各缺 {head}/{tail} 历元"
                kind = 'obscured'
            elif head > 30:
                verdict = f"升空期被遮挡: 前 {head} 历元未观测（低仰角/城市遮挡）"
                kind = 'rise_cut'
            elif tail > 30:
                verdict = f"降空期被遮挡: 末 {tail} 历元未观测"
                kind = 'set_cut'
            else:
                verdict = "断续小中断（典型低仰角/多路径）"
                kind = 'scattered'

            rows.append({
                'sat': satid,
                'label': sat_label(satid),
                'sys': sys_name(satid[0]),
                'n_epochs': len(epl),
                'first': first, 'last': last, 'span': span,
                'rate': rate,
                'n_gaps': len(gaps),
                'max_gap': max_gap,
                'head_gap': head, 'tail_gap': tail,
                'verdict': verdict,
                'kind': kind,
                # 供完整性条带图使用：中断区间的确切位置
                'gap_segs': gap_segs,
                'epoch_list': epl,
            })

        rows.sort(key=lambda r: r['rate'])
        rates = [r['rate'] for r in rows]

        # 按系统汇总
        sys_summary = {}
        by_sys = collections.defaultdict(list)
        for r in rows:
            by_sys[r['sys']].append(r)
        for s, sub in by_sys.items():
            sys_summary[s] = {
                'n_sats': len(sub),
                'rate_mean': statistics.mean(x['rate'] for x in sub),
                'rate_min': min(x['rate'] for x in sub),
            }

        return {
            'rows': rows,
            'rate_mean': statistics.mean(rates) if rates else 0.0,
            'n_ge99': sum(1 for r in rates if r >= 99),
            'n_total': len(rates),
            'sys_summary': sys_summary,
            'loss_of_lock': [r for r in rows if r['kind'] == 'loss_of_lock'],
        }

    # ==================== 时间序列（供 RTKPLOT 式图表） ====================

    def _timeseries_stats(self, epochs):
        """构建逐卫星的时间序列，用于绘制 RTKPLOT 式图表

        返回三组序列（横轴都是「历元序号」的分钟值，便于多图对齐）：
          snr    : {(sys, band): {satid: (t_list, v_list)}}
          elev   : {satid: (t_list, v_list)}   需要星历，否则空
          mp     : {satid: (t_list, v_list)}   仅 GPS 真双频卫星

        只在有值时才记点，缺测处自然断开，这样中断能直接在图上看见。
        """
        n_ep = len(epochs)
        # 采样间隔（秒），用于把历元序号换算成分钟
        dt = 1.0
        if n_ep >= 2:
            d = epochs[1][0] - epochs[0][0]
            if d > 0:
                dt = d

        # ---- SNR 时间序列（按 系统-频点 分组，逐卫星一条线）----
        sys_snr_codes = {}
        for sysc in set(list(self.rnx.obstypes.keys()) + ['']):
            codes = self.rnx.obs_codes(sysc)
            sys_snr_codes[sysc] = [(c, band_of(c)) for c in codes if is_snr(c)]

        snr = collections.defaultdict(lambda: collections.defaultdict(
            lambda: ([], [])))
        for ei, (_, satdict) in enumerate(epochs):
            tmin = ei * dt / 60.0
            for satid, vals in satdict.items():
                sysc = satid[0]
                codes = sys_snr_codes.get(sysc) or sys_snr_codes.get('', [])
                for code, band in codes:
                    v = vals.get(code)
                    if v and v > 0:
                        key = (sys_name(sysc), band)
                        ts, vs = snr[key][satid]
                        ts.append(tmin)
                        vs.append(v)

        # ---- 仰角时间序列 ----
        elev = {}
        if self.eph and self.eph.epochs:
            xyz = self._observer_xyz()
            t0, _ = self.rnx.time_span()
            if xyz and t0:
                from .rinex import LocalFrame
                frame = LocalFrame(*xyz)
                day0 = day0_timestamp(t0[0], doy_of(t0[0], t0[1], t0[2]))
                # 控制耗时：最多取 800 个点/卫星
                step = max(1, n_ep // 800)
                acc = collections.defaultdict(lambda: ([], []))
                for ei in range(0, n_ep, step):
                    secs, satdict = epochs[ei]
                    tmin = ei * dt / 60.0
                    t = day0 + secs
                    for satid in satdict:
                        if satid[0] == 'R':
                            continue
                        d = self.eph.best(satid, t)
                        if d is None:
                            continue
                        try:
                            sx, sy, sz = self.eph.sat_ecef(d, t)
                        except (ValueError, TypeError, ZeroDivisionError):
                            continue
                        r = frame.az_el(sx, sy, sz)
                        if r is None:
                            continue
                        ts, vs = acc[satid]
                        ts.append(tmin)
                        vs.append(r[1])
                elev = {k: tuple(v) for k, v in acc.items() if len(v[0]) >= 5}

        # ---- 多路径时间序列（复用 _mp_series，与 RMS 统计同一套判定）----
        mp = {}
        c1 = self.rnx.pick('G', ['C1', 'P1'])
        l1 = self.rnx.pick('G', ['L1'])
        c2 = self.rnx.pick('G', ['P2', 'C2'])
        l2 = self.rnx.pick('G', ['L2'])
        if all((c1, l1, c2, l2)):
            dual, _ = self._split_by_frequency(epochs, c1, l1, c2, l2)
            series, _slips = self._mp_series(epochs, dual, c1, l1, c2, l2,
                                             with_time=True)
            mp = {k: ([p[0] for p in v], [p[1] for p in v])
                  for k, v in series.items() if len(v) >= 10}

        return {
            'snr': {k: dict(v) for k, v in snr.items()},
            'elev': elev,
            'mp': mp,
            'duration_min': n_ep * dt / 60.0,
            'dt': dt,
        }

    # ==================== 多路径 ====================

    def _mp_series(self, epochs, dual, c1, l1, c2, l2, with_time=False,
                   f1=F_L1, f2=F_L2):
        """计算逐卫星多路径序列

        返回 {satid: [(t_min, mp), ...]} 或 {satid: [mp, ...]}。

        关于周跳处理：**不做**基于相位增量的周跳剔除。
        本批接收机的相位增量中位数高达数百米且离散度极大
        （实测 L1 增量中位 -427 ~ +658 m，MAD 55 ~ 124 m），
        说明每个历元都带接收机钟跳，根本不存在"近似常数增量"这个前提，
        任何基于增量的周跳判据都会把绝大多数正常历元误杀。
        改用组合值的幅值门限 |m| < 50 m 过滤明显异常点即可。
        """
        dt = 1.0
        if len(epochs) >= 2:
            d = epochs[1][0] - epochs[0][0]
            if d > 0:
                dt = d

        # 系数按实际频率对计算：多系统 MP 用的不是 GPS 的 L1/L2，
        # 模块级 _AL/_BE 是 GPS 专用的，直接复用会把 BDS/Galileo 算错
        al = f1 ** 2 / (f1 ** 2 - f2 ** 2)
        be = -f2 ** 2 / (f1 ** 2 - f2 ** 2)
        lam1 = C_LIGHT / f1
        lam2 = C_LIGHT / f2

        out = collections.defaultdict(list)
        prev = {}
        for ei, (_, satdict) in enumerate(epochs):
            tmin = ei * dt / 60.0
            for satid, vals in satdict.items():
                if satid not in dual:
                    continue
                need = (vals.get(c1), vals.get(l1),
                        vals.get(c2), vals.get(l2))
                if not all(need):
                    prev.pop(satid, None)
                    continue
                if satid in prev:
                    pc1, pl1, pc2, pl2 = prev[satid]
                    a = al * (need[0] - pc1) + be * (need[2] - pc2)
                    b = (al * (need[1] - pl1) * lam1
                         + be * (need[3] - pl2) * lam2)
                    m = a - b
                    if -50 < m < 50:
                        out[satid].append((tmin, m) if with_time else m)
                prev[satid] = need
        return out, {}

    def _multipath_stats(self, epochs):
        """GPS 多路径 MP（L1/L2 无电离层组合）

        MP = [α·(P1-P2) + β·(L1·λ1 - L2·λ2)] 的历元间差分，
        剔除几何项后保留多路径 + 噪声，取 RMS 衡量。
        仅 GPS 有标准 L1/L2 双频对；其他系统需按频点组合另行处理。

        关键前提：**该卫星必须真的有 L2 信号**。
        接收机导出的文件常带 L2/S2/D2/P2 四列，但并非所有卫星都真的
        在 L2 上有数据——没有时这些列会被填成 L1 的重复值，
        此时 C1≈P2、L1≈L2，双频组合会算出几十米量级的假多路径。
        本方法逐卫星检查 L1/L2 是否来自不同频点，把不满足的卫星剔除，
        而不是把垃圾值混进统计结果。
        """
        c1 = self.rnx.pick('G', ['C1', 'P1'])
        l1 = self.rnx.pick('G', ['L1'])
        c2 = self.rnx.pick('G', ['P2', 'C2'])
        l2 = self.rnx.pick('G', ['L2'])
        if not all((c1, l1, c2, l2)):
            return {'available': False, 'reason': '缺少 GPS 双频伪距/相位观测值'}

        # ---- 逐卫星双频有效性判定 ----
        dual, single = self._split_by_frequency(epochs, c1, l1, c2, l2)
        if not dual:
            return {
                'available': False,
                'single_frequency': True,
                'reason': ('该观测文件不具备 L1/L2 真双频条件：'
                           '所有 GPS 卫星的 P2/L2 列都与 C1/L1 重合'
                           '（接收机对未跟踪 L2 的卫星用 L1 值填充），'
                           '多路径组合无法计算。'),
            }

        mp_vals, slip_count = self._mp_series(epochs, dual, c1, l1, c2, l2)

        rows = []
        for satid, arr in mp_vals.items():
            if len(arr) < MIN_SAMPLES_SAT:
                continue
            rms = math.sqrt(sum(x * x for x in arr) / len(arr))
            rows.append({
                'sat': satid,
                'label': sat_label(satid),
                'rms': rms,
                'n': len(arr),
                'n_slip': slip_count.get(satid, 0),
                'series': arr,
            })
        rows.sort(key=lambda r: -r['rms'])

        mean_rms = statistics.mean(r['rms'] for r in rows) if rows else None
        max_rms = max((r['rms'] for r in rows), default=None)
        return {
            'available': bool(rows),
            'rows': rows,
            'mean_rms': mean_rms,
            'max_rms': max_rms,
            'codes': {'c1': c1, 'l1': l1, 'c2': c2, 'l2': l2},
            'excluded_sats': [sat_label(s) for s in sorted(single)],
            'total_slips': sum(slip_count.values()),
        }

    # ============ 多系统 MP / MP-仰角相关 / 电离层 ROT ============

    def known_point_offset(self, lat, lon, h=0.0, antenna_h=0.0):
        """RINEX 头近似坐标 vs 已知控制点 的平面/高程偏差（单点比对）

        见模块级 :func:`known_point_offset` 的说明。
        """
        if not self.rnx.approx_xyz:
            raise ValueError('RINEX 头里没有近似坐标（APPROX POSITION XYZ）')
        return known_point_offset(self.rnx.approx_xyz, lat, lon, h, antenna_h)



    def _resolve_system_freq(self, epochs, sysc, pref):
        """确定某系统实际使用的 (f1, f2) 频率对

        接收机把第二频点写进哪个列、对应哪个频率，各厂家习惯不同：
        部分接收机的 Galileo 把 **E5b(1207.14)** 写进 L2 列而不是 E5a。
        频率比 γ 一旦用错，MP 组合的系数整体偏离，量级直接错。

        做法：对每个候选 (f1, f2) 组合，抽样计算全部卫星的 GF 跨度
        中位数，取**最小者**。真双频的 GF 是米级（实测 0.06 ~ 62 m），
        频率配错时 λ1/λ2 比例失配，GF 跨度变成千米级
        （实测配错时 6678 ~ 48646 m）—— 两者差几个数量级，
        判定非常稳健。最小跨度仍超过 _MP_GF_SPAN_MAX 视为无双频。

        返回 (f1, f2, gf_span_med)；无法判定返回 None。
        """
        l1 = self.rnx.pick(sysc, pref['l1'])
        l2 = self.rnx.pick(sysc, pref['l2'])
        if not (l1 and l2):
            return None
        best = None
        for f1 in pref['f1']:
            for f2 in pref['f2']:
                if f1 == f2:
                    continue
                dual, _single = self._split_by_frequency(
                    epochs, None, l1, None, l2,
                    sysc=sysc, f1=f1, f2=f2)
                if not dual:
                    continue
                spans = []
                stride = max(1, len(epochs) // 3000)
                for k, (_, satdict) in enumerate(epochs):
                    if k % stride:
                        continue
                    for satid, v in satdict.items():
                        if satid[0] != sysc or satid not in dual:
                            continue
                        a, b = v.get(l1), v.get(l2)
                        if not a or not b:
                            continue
                        gf = (a * C_LIGHT / f1 - b * C_LIGHT / f2)
                        spans.append(gf)
                if len(spans) < 2:
                    continue
                span_med = _quantile(sorted(spans), 0.5)
                if best is None or span_med < best[2]:
                    best = (f1, f2, span_med)
        if best is None or best[2] > _MP_GF_SPAN_MAX:
            return None
        return best

    def _multipath_sys_stats(self, epochs):
        """多系统多路径 MP

        与 ``_multipath_stats``（GPS，结构保持不变供既有调用方使用）
        的区别：逐系统用**各自的频率对**计算同一形式的 MP 组合。
        MP 组合里 1/(1−γ) 这类系数取决于 f1/f2 —— 用 GPS 的系数去算
        Galileo（E1/E5a，γ 不同）会整体偏离，量级直接错。

        GLONASS 明确不支持（见 _MP_SYSTEM_PREFS 的注释）。
        """
        systems = {}
        rows = []
        for sysc, pref in _MP_SYSTEM_PREFS.items():
            name = sys_name(sysc)
            if not self.rnx.obs_codes(sysc):
                systems[name] = {'available': False,
                                 'reason': '无该系统观测值'}
                continue
            freq = self._resolve_system_freq(epochs, sysc, pref)
            if freq is None:
                systems[name] = {
                    'available': False, 'single_frequency': True,
                    'reason': ('未能从数据中确认可用的双频频率对：'
                               '或只有单频观测值，或第二频点列被'
                               '第一频点值填充'),
                }
                continue
            f1, f2, gf_span = freq
            c1 = self.rnx.pick(sysc, pref['c1'])
            l1 = self.rnx.pick(sysc, pref['l1'])
            c2 = self.rnx.pick(sysc, pref['c2'])
            l2 = self.rnx.pick(sysc, pref['l2'])
            if not all((c1, l1, c2, l2)):
                systems[name] = {'available': False,
                                 'reason': '缺少双频伪距/相位观测值'}
                continue
            dual, single = self._split_by_frequency(
                epochs, c1, l1, c2, l2,
                sysc=sysc, f1=f1, f2=f2)
            if not dual:
                systems[name] = {
                    'available': False, 'single_frequency': True,
                    'reason': ('第二频点列与第一频点重合'
                               '（接收机用第一频点值填充），无法计算'),
                    'excluded_sats': [sat_label(x) for x in sorted(single)],
                }
                continue
            mp_vals, _slip = self._mp_series(
                epochs, dual, c1, l1, c2, l2, f1=f1, f2=f2)
            srows = []
            for satid, arr in mp_vals.items():
                if len(arr) < MIN_SAMPLES_SAT:
                    continue
                rms = math.sqrt(sum(x * x for x in arr) / len(arr))
                srows.append({
                    'sat': satid, 'label': sat_label(satid),
                    'sys': sysc, 'sys_name': name,
                    'rms': rms, 'n': len(arr), 'series': arr,
                })
            srows.sort(key=lambda r: -r['rms'])
            mean_rms = (statistics.mean(r['rms'] for r in srows)
                        if srows else None)
            systems[name] = {
                'available': bool(srows),
                'rows': srows, 'mean_rms': mean_rms,
                'codes': {'c1': c1, 'l1': l1, 'c2': c2, 'l2': l2},
                # 判定出的频率对与 GF 跨度，报告里展示，
                # 便于核对频率自动识别是否合理
                'freq': {'f1': f1, 'f2': f2, 'gf_span_med': gf_span},
                'excluded_sats': [sat_label(x) for x in sorted(single)],
            }
            rows.extend(srows)
        return {
            'available': any(v.get('available')
                             for v in systems.values()),
            'systems': systems,
            'rows': rows,
        }

    @staticmethod
    def _pearson(xs, ys):
        """Pearson 相关系数；样本不足或零方差返回 None"""
        n = len(xs)
        if n < 3 or n != len(ys):
            return None
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx <= 0 or syy <= 0:
            return None
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        return sxy / math.sqrt(sxx * syy)

    @staticmethod
    def _mp_elev_verdict(r):
        """MP-仰角相关系数的解读

        低仰角 MP 大是正常物理（信号路径贴近地面，反射强）；
        关键是判断 MP 到底随仰角变化，还是与仰角无关地普遍偏大。
        """
        if r is None:
            return '样本不足，无法判断'
        if r <= -0.5:
            return ('MP 随仰角升高明显减小：以低仰角多径为主，'
                    '属正常物理，可抬高截止仰角观察整体改善')
        if r >= 0.5:
            return ('MP 随仰角升高反而增大：反常，'
                    '建议检查天线附近上方遮挡/反射或设备问题')
        return ('MP 与仰角无明显相关：多径来源更可能是固定方向的'
                '环境反射（天线附近的墙/金属面）或设备本身')

    def _mp_elev_corr(self, mp_sys, geometry, scatter_mp):
        """多路径与仰角的相关性（分系统）

        逐卫星：平均仰角（来自 geometry.per_sat，需星历）vs MP-RMS。
        GPS 另外给出**逐历元**相关（elev_scatter['mp'] 里已有
        (仰角, MP) 配对），粒度比逐卫星细得多。

        仰角必须靠星历算，无星历覆盖的卫星进不了统计 —— 这不是缺陷
        而是物理：没有仰角就没法谈"与仰角的相关性"。
        """
        per_sat = {r.get('sat'): r
                   for r in (geometry.get('per_sat') or [])}
        systems = {}
        for name, blk in (mp_sys.get('systems') or {}).items():
            if not blk.get('available'):
                continue
            pts = []
            for r in blk['rows']:
                g = per_sat.get(r['sat'])
                if not g or g.get('el_mean') is None:
                    continue
                pts.append((g['el_mean'], r['rms'], r['label']))
            if len(pts) < 3:
                systems[name] = {
                    'available': False,
                    'reason': (f'有仰角数据的卫星不足（{len(pts)} 颗，'
                               f'至少需要 3 颗；无星历的卫星算不出仰角）'),
                    'points': pts,
                }
                continue
            r_val = self._pearson([p[0] for p in pts],
                                  [p[1] for p in pts])
            systems[name] = {
                'available': True, 'n': len(pts), 'r': r_val,
                'points': pts,
                'verdict': self._mp_elev_verdict(r_val),
            }
        gps_epoch = None
        pts = [(el, mv)
               for seq in (scatter_mp or {}).values()
               for el, mv in seq]
        if len(pts) >= 30:
            r_val = self._pearson([p[0] for p in pts],
                                  [p[1] for p in pts])
            gps_epoch = {'n': len(pts), 'r': r_val,
                         'verdict': self._mp_elev_verdict(r_val)}
        return {
            'available': any(s.get('available')
                             for s in systems.values())
            or gps_epoch is not None,
            'systems': systems,
            'gps_epoch': gps_epoch,
        }

    def _iono_stats(self, epochs, geometry):
        """电离层变化率 ROT / IROT（几何无关组合的慢变部分）

        GF = λ1·L1 − λ2·L2 = −(1−γ)·I1 + 模糊度常数，
        其中 I1 是 L1 上的电离层延迟（米）。对时间求导，模糊度常数消失::

            ROT  = −f1² / (40.3·(1−γ)) · dGF/dt      [TECU/min]
            IROT = ROT / sin(仰角)

        经验量级（TECU/min）：安静 < 1，中扰 1~5，扰动 > 5。
        电离层扰动期会同时推高周跳率、拉长 RTK 固定时间 ——
        这是解释"固定慢/浮点久"的重要环境维度。

        两点取舍：
          * **周跳/丢锁的历元差不混进来**：单次 GF 跳变可达几十米
            （对应几十 TECU），一个就够把 ROT 统计带偏。
            判据：相邻历元 |ΔGF| > 5 m 视为不连续，丢弃该段。
          * **IROT 用该星的平均仰角换算**，不做逐历元 —— 逐历元仰角
            需要再跑一遍星历求解（本模块最耗时的一步），代价翻倍
            而收益只是把"每星一个数"细化成时序。要逐历元 IROT
            应在 _diagnostics_stats 的星历循环里加，那是正位置。

        GLONASS 不参与（FDMA，见 _MP_SYSTEM_PREFS 注释）。
        """
        GF_SLIP_M = 5.0
        TECU_MIN = 60.0          # TECU/s -> TECU/min
        # 电离层延迟常数：I = 40.3e16·TEC/f²（f 单位 Hz、TEC 单位 TECU、
        # I 单位米）。写成 40.3 会让 ROT 大 1e16 倍 —— 第一版就栽在这
        K = 40.3e16              # (m·Hz²/TECU)

        per_sat = {r.get('sat'): r
                   for r in (geometry.get('per_sat') or [])}
        systems = {}
        n_skip_slip = 0
        for sysc, pref in _MP_SYSTEM_PREFS.items():
            name = sys_name(sysc)
            if not self.rnx.obs_codes(sysc):
                continue
            freq = self._resolve_system_freq(epochs, sysc, pref)
            if freq is None:
                continue
            f1, f2, _gf_span = freq
            l1 = self.rnx.pick(sysc, pref['l1'])
            l2 = self.rnx.pick(sysc, pref['l2'])
            if not (l1 and l2):
                continue
            dual, _single = self._split_by_frequency(
                epochs, None, l1, None, l2,
                sysc=sysc, f1=f1, f2=f2)
            if not dual:
                systems[name] = {'available': False,
                                 'reason': '缺少可用的双频载波观测值'}
                continue
            gamma = (f1 / f2) ** 2
            coef = -f1 ** 2 / (K * (1.0 - gamma)) * TECU_MIN
            rot_acc = collections.defaultdict(list)
            prev = {}
            for secs, satdict in epochs:
                for satid, vals in satdict.items():
                    if satid[0] != sysc or satid not in dual:
                        continue
                    a, b = vals.get(l1), vals.get(l2)
                    if not a or not b:
                        prev.pop(satid, None)
                        continue
                    gf = a * C_LIGHT / f1 - b * C_LIGHT / f2
                    p = prev.get(satid)
                    prev[satid] = (secs, gf)
                    if p is None:
                        continue
                    dt = secs - p[0]
                    if dt <= 0:
                        continue
                    dgf = gf - p[1]
                    if abs(dgf) > GF_SLIP_M:
                        n_skip_slip += 1
                        continue
                    rot_acc[satid].append(abs(coef * dgf / dt))

            rows = []
            for satid, arr in rot_acc.items():
                if len(arr) < MIN_SAMPLES_SAT:
                    continue
                arr_sorted = sorted(arr)
                med = _quantile(arr_sorted, 0.5)
                mx = arr_sorted[-1]
                row = {
                    'sat': satid, 'label': sat_label(satid),
                    'n': len(arr),
                    'rot_med': med, 'rot_max': mx,
                }
                g = per_sat.get(satid)
                el = g.get('el_mean') if g else None
                if el is not None and el > 5.0:
                    row['irot_med'] = med / math.sin(math.radians(el))
                    row['el_mean'] = el
                else:
                    row['irot_med'] = None
                    row['el_mean'] = el
                rows.append(row)
            rows.sort(key=lambda r: -r['rot_med'])
            all_med = [r['rot_med'] for r in rows]
            all_max = [r['rot_max'] for r in rows]
            # 活动级别取"逐星中位数的中位数"（典型值）而不是最大值：
            # 个别低仰角卫星的 ROT 偏大会把级别抬高，误导整体判断
            rot_med_typ = (_quantile(sorted(all_med), 0.5)
                           if all_med else None)
            systems[name] = {
                'available': bool(rows),
                'rows': rows,
                'rot_med_max': max(all_med) if all_med else None,
                'rot_max_all': max(all_max) if all_max else None,
                'rot_med_typ': rot_med_typ,
                'level': _iono_level(rot_med_typ),
            }
        return {
            'available': any(v.get('available')
                             for v in systems.values()),
            'systems': systems,
            'n_skip_slip': n_skip_slip,
        }

    # ============ 观测诊断：DOP / 卫星数时序 + 仰角散点 ============

    def _diagnostics_stats(self, epochs):
        """DOP、可见卫星数时序，以及 SNR/MP–仰角散点（共用一趟星历计算）

        这几类指标都要逐历元的卫星仰角，放在同一趟遍历里算，
        避免重复调用星历 —— 求卫星位置是本模块最耗时的一步。

        DOP 用标准几何法（单位权、仅几何）::

            每颗卫星的单位视线向量 (e, n, u):
                e = cos(el)·sin(az)   n = cos(el)·cos(az)   u = sin(el)
            设计矩阵 G 的第 i 行为 [-e_i, -n_i, -u_i, 1]，末列 1 对应接收机钟差
            Q = (GᵀG)⁻¹
                GDOP = sqrt(tr Q)      PDOP = sqrt(q11+q22+q33)
                HDOP = sqrt(q11+q22)   VDOP = sqrt(q33)
                TDOP = sqrt(q44)

        与 RTKPLOT 的 DOP/NSat 图同口径（不做信噪比加权）。
        低于 ELEV_MASK_DEG 的卫星不参与解算，但仍计入「可见卫星数」。
        """
        if not self.eph or not self.eph.epochs:
            return {'dop': {'available': False, 'reason': '缺少广播星历'},
                    'scatter': {}}
        xyz = self._observer_xyz()
        if not xyz:
            return {'dop': {'available': False, 'reason': '头文件缺少近似坐标'},
                    'scatter': {}}
        t0, _ = self.rnx.time_span()
        if not t0:
            return {'dop': {'available': False, 'reason': '无法确定观测日期'},
                    'scatter': {}}

        from .rinex import LocalFrame
        frame = LocalFrame(*xyz)
        day0 = day0_timestamp(t0[0], doy_of(t0[0], t0[1], t0[2]))

        n_ep = len(epochs)
        dt = 1.0
        if n_ep >= 2:
            d = epochs[1][0] - epochs[0][0]
            if d > 0:
                dt = d
        step = max(1, n_ep // 1200)          # 最多约 1200 个历元

        # ---- MP 查表：历元序号 -> MP 值，供 MP–仰角散点使用 ----
        mp_at = {}
        gc1 = self.rnx.pick('G', ['C1', 'P1'])
        gl1 = self.rnx.pick('G', ['L1'])
        gc2 = self.rnx.pick('G', ['P2', 'C2'])
        gl2 = self.rnx.pick('G', ['L2'])
        if all((gc1, gl1, gc2, gl2)):
            dual, _ = self._split_by_frequency(epochs, gc1, gl1, gc2, gl2)
            ser, _s = self._mp_series(epochs, dual, gc1, gl1, gc2, gl2,
                                      with_time=True)
            for satid, arr in ser.items():
                mp_at[satid] = {int(round(t * 60.0 / dt)): v for t, v in arr}

        # ---- SNR 观测码表（系统 -> 频点）----
        sys_snr_codes = {}
        for sysc in set(list(self.rnx.obstypes.keys()) + ['']):
            codes = self.rnx.obs_codes(sysc)
            sys_snr_codes[sysc] = [(c, band_of(c)) for c in codes if is_snr(c)]

        times, gdops, pdops, hdops, vdops, tdops = [], [], [], [], [], []
        n_used, n_vis, n_eph = [], [], []
        used_sys, eph_sys = set(), set()
        sc_snr = collections.defaultdict(lambda: collections.defaultdict(list))
        sc_mp = collections.defaultdict(list)
        mask = ELEV_MASK_DEG

        for ei in range(0, n_ep, step):
            secs, satdict = epochs[ei]
            if not satdict:
                continue
            t = day0 + secs
            tmin = ei * dt / 60.0
            rows = []
            n_ok = 0
            for satid, vals in satdict.items():
                if satid[0] == 'R':                 # GLONASS 星历未解析
                    continue
                d = self.eph.best(satid, t)
                if d is None:
                    continue
                try:
                    sx, sy, sz = self.eph.sat_ecef(d, t)
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
                r = frame.az_el(sx, sy, sz)
                if r is None:
                    continue
                az, el, _ = r
                n_ok += 1
                eph_sys.add(satid[0])

                # ---- 仰角散点配对 ----
                sysc = satid[0]
                for code, band in (sys_snr_codes.get(sysc)
                                   or sys_snr_codes.get('', [])):
                    v = vals.get(code)
                    if v and v > 0:
                        sc_snr[(sysc, band)][satid].append((el, v))
                mv = mp_at.get(satid, {}).get(ei)
                if mv is not None:
                    sc_mp[satid].append((el, mv))

                # ---- 参与 DOP 解算的行 ----
                if el >= mask:
                    ce = math.cos(math.radians(el))
                    used_sys.add(satid[0])
                    rows.append((-ce * math.sin(math.radians(az)),
                                 -ce * math.cos(math.radians(az)),
                                 -math.sin(math.radians(el)), 1.0))
            if len(rows) < 4:
                continue

            g = [[0.0] * 4 for _ in range(4)]
            for rv in rows:
                for i in range(4):
                    for j in range(4):
                        g[i][j] += rv[i] * rv[j]
            q = _inv4(g)
            if q is None:
                continue
            tr = sum(q[i][i] for i in range(4))
            pd = q[0][0] + q[1][1] + q[2][2]
            hd = q[0][0] + q[1][1]
            vd = q[2][2]
            if min(tr, pd, hd, vd, q[3][3]) < 0:      # 数值病态，丢弃该历元
                continue
            try:
                times.append(tmin)
                gdops.append(math.sqrt(tr))
                pdops.append(math.sqrt(pd))
                hdops.append(math.sqrt(hd))
                vdops.append(math.sqrt(vd))
                tdops.append(math.sqrt(q[3][3]))
            except (ValueError, OverflowError):
                continue
            n_used.append(len(rows))
            n_vis.append(len(satdict))
            n_eph.append(n_ok)

        if not times:
            return {'dop': {'available': False,
                            'reason': '星历与观测时间不匹配，或可用卫星不足 4 颗'},
                    'scatter': {}}

        def _stat(arr):
            a = sorted(arr)
            return {'mean': statistics.mean(a), 'min': a[0], 'max': a[-1],
                    'p50': _quantile(a, 0.50), 'p95': _quantile(a, 0.95)}

        stats = {
            'gdop': _stat(gdops), 'pdop': _stat(pdops),
            'hdop': _stat(hdops), 'vdop': _stat(vdops),
            'nsat_used': _stat(n_used),
            'pdop_gt4': sum(1 for x in pdops if x > 4) / len(pdops) * 100,
            'pdop_gt6': sum(1 for x in pdops if x > 6) / len(pdops) * 100,
            'nsat_lt5': sum(1 for x in n_used if x < 5) / len(n_used) * 100,
            'min_sat': min(n_used),
        }
        return {
            'dop': {
                'available': True,
                'times': times, 'gdop': gdops, 'pdop': pdops,
                'hdop': hdops, 'vdop': vdops, 'tdop': tdops,
                'nsat_used': n_used, 'nsat_vis': n_vis, 'nsat_eph': n_eph,
                'stats': stats, 'mask': mask, 'step': step, 'n': len(times),
                # 星历覆盖哪些系统 —— 只有 GPS 星历时 DOP/仰角只能算 GPS，
                # 必须在界面上说清楚，否则用户会把它当成全星座结果
                'used_systems': [sys_name(s) for s in sorted(used_sys)],
                'eph_systems': [sys_name(s) for s in sorted(eph_sys)],
            },
            'scatter': {
                # 键统一成 (系统显示名, 频点)，与 obs_series 的键一致，
                # 便于界面用同一个频点下拉框同时驱动两张散点图
                'snr': {(sys_name(k[0]), k[1]): dict(v)
                        for k, v in sc_snr.items()},
                'mp': dict(sc_mp),
            },
        }

    # ============ 原始观测量时序 + 周跳 ============

    def _band_code_map(self):
        """逐系统挑出每个频点的 伪距/载波/多普勒 观测码

        返回 {系统字符: {频点: {'p':码, 'l':码, 'd':码}}}
        """
        out = {}
        for sysc in set(list(self.rnx.obstypes.keys()) + ['']):
            codes = self.rnx.obs_codes(sysc)
            if not codes:
                continue
            per = collections.defaultdict(dict)
            for c in codes:
                b = band_of(c)
                if not b:
                    continue
                slot = per[b]
                if is_pseudorange(c):
                    cur = slot.get('p')
                    # 优先 C 码（无码跟踪），其次 P 码
                    if cur is None or (cur[0] == 'P' and c[0] == 'C'):
                        slot['p'] = c
                elif is_phase(c):
                    slot.setdefault('l', c)
                elif is_doppler(c):
                    slot.setdefault('d', c)
            per = {b: v for b, v in per.items() if v}
            if per:
                out[sysc] = per
        return out

    def _obs_series_stats(self, epochs):
        """原始观测量时序（伪距/载波/多普勒）+ 周跳检测

        产出:
          * series : {(系统, 频点): {satid: (t, 伪距, 载波, 多普勒)}}
                     缺测处填 NaN，这样图上会自然断开
          * slips  : 逐卫星周跳事件与可用性判定

        周跳判据 —— 几何无关组合 (geometry-free)::

            GF = λ1·L1 - λ2·L2

        GF 中几何距离、接收机钟差、对流层均已抵消，只剩电离层延迟
        与模糊度常数。电离层变化平缓（实测历元间跳变中位 0.4 ~ 4 mm，
        P99 也只有 0.002 ~ 0.028 m），而任一颗星发生 1 周周跳都会让
        GF 突跳 λ ≈ 0.19 / 0.24 m，两者相差两个数量级，
        因此门限取 5 cm 时灵敏度极高而几乎没有虚警。

        自校验：若某颗卫星 GF 的中位跳变本身就很大（> SLIP_GF_CLEAN），
        说明该卫星两个频点的列彼此不自洽（接收机填充/异常），
        此时**不猜测**，直接标记为「无法检测」而不是给出错误的周跳数。

        为什么不用多普勒-相位一致性做判据（留档，勿重蹈）:
        实测该接收机的 |ΔL + D·dt| 残差在**所有卫星上分布几乎完全一致**
        （P50≈0.32 周、P90≈0.92、P99≈1.73，且每颗星都恰好有 1 个 >3 周、
        约 20 个 >2 周的事件），这是接收机多普勒估计的系统性共模效应，
        不是周跳；噪声底（P99 达 1.7 周）远高于 1 周周跳的幅度，
        该判据在本批数据上不具备可用性，故弃用。
        """
        n_ep = len(epochs)
        dt = 1.0
        if n_ep >= 2:
            d = epochs[1][0] - epochs[0][0]
            if d > 0:
                dt = d
        step = max(1, n_ep // 900)

        sys_band_codes = self._band_code_map()
        if not sys_band_codes:
            return {'series': {}, 'slips': {'available': False,
                                            'reason': '没有可识别的观测码'}}

        sys_present = set()
        for _, satdict in epochs:
            for satid in satdict:
                sys_present.add(satid[0])

        def codes_of(sysc):
            return sys_band_codes.get(sysc) or sys_band_codes.get('') or {}

        # ---------- 原始观测量序列（抽稀采样）----------
        series = collections.defaultdict(lambda: collections.defaultdict(
            lambda: ([], [], [], [])))
        nan = float('nan')
        for ei in range(0, n_ep, step):
            _, satdict = epochs[ei]
            tmin = ei * dt / 60.0
            for satid, vals in satdict.items():
                per = codes_of(satid[0])
                if not per:
                    continue
                for band, slot in per.items():
                    pc, lc, dc = slot.get('p'), slot.get('l'), slot.get('d')
                    p = vals.get(pc) if pc else None
                    l = vals.get(lc) if lc else None
                    d = vals.get(dc) if dc else None
                    if p is None and l is None and d is None:
                        continue
                    ts, ps, ls, ds = series[(satid[0], band)][satid]
                    ts.append(tmin)
                    ps.append(p if p is not None else nan)
                    ls.append(l if l is not None else nan)
                    ds.append(d if d is not None else nan)

        # 每个系统最多保留 3 个频点（按载波频率从高到低），并丢掉样本过少的
        out_series = {}
        by_sys = collections.defaultdict(list)
        for (sysc, band) in series:
            by_sys[sysc].append(band)
        for sysc, bands in by_sys.items():
            freq = _BAND_FREQ.get(sysc, {})
            bands.sort(key=lambda b: -(freq.get(b) or 0))
            for band in bands[:3]:
                sats = {s: tuple(v) for s, v in series[(sysc, band)].items()
                        if len(v[0]) >= 30}
                if sats:
                    out_series[(sys_name(sysc), band)] = sats

        # ---------- 周跳检测（全部历元，不抽稀）----------
        slip_rows = []
        total = 0
        total_loss = 0
        for sysc in sorted(sys_present):
            per = codes_of(sysc)
            if not per:
                continue
            pair = None
            for b1, b2 in _GF_BAND_PAIRS:
                if b1 in per and b2 in per:
                    pair = (b1, b2)
                    break
            if pair is None:                     # 退而求其次：任取两个频点
                cand = sorted(per)
                if len(cand) >= 2:
                    pair = (cand[0], cand[1])
            if pair is None:
                continue
            b1, b2 = pair
            freq = _BAND_FREQ.get(sysc, {})
            f1, f2 = freq.get(b1), freq.get(b2)
            code1, code2 = per[b1].get('l'), per[b2].get('l')
            if not f1 or not f2 or not code1 or not code2:
                continue
            lam1, lam2 = C_LIGHT / f1, C_LIGHT / f2

            prev = {}
            jumps = collections.defaultdict(list)
            for ei, (_, satdict) in enumerate(epochs):
                tmin = ei * dt / 60.0
                for satid, vals in satdict.items():
                    if satid[0] != sysc:
                        continue
                    a, b = vals.get(code1), vals.get(code2)
                    if a is None or b is None:
                        prev.pop(satid, None)
                        continue
                    gf = a * lam1 - b * lam2
                    if satid in prev:
                        pei, pgf = prev[satid]
                        if 0 < ei - pei <= 3:
                            jumps[satid].append((tmin, gf - pgf))
                    prev[satid] = (ei, gf)

            for satid, arr in jumps.items():
                if len(arr) < MIN_SAMPLES_SAT:
                    continue
                mags = sorted(abs(v) for _, v in arr)
                med = statistics.median(mags)
                row = {'sat': satid, 'label': sat_label(satid),
                       'sys': sys_name(sysc), 'n': len(arr),
                       'median_jump': med, 'band1': b1, 'band2': b2,
                       'n_slips': 0, 'n_loss': 0, 'n_raw': 0,
                       'int_ratio': None, 'chronic': False, 'events': [],
                       'usable': True, 'reason': ''}
                if med > SLIP_GF_CLEAN:
                    row['usable'] = False
                    row['reason'] = (f"{b1}/{b2} 两列不自洽"
                                     f"（中位跳变 {med:.3f} m），无法检测周跳")
                else:
                    ev = []
                    n_raw = n_int = 0
                    for t, v in arr:
                        if abs(v) <= SLIP_GF_THRESHOLD:
                            continue
                        n_raw += 1
                        if abs(v) > SLIP_LOSS_THRESHOLD:
                            # 幅度远超若干周，判为丢锁/大跳变，
                            # 此时去折算"第几周"没有意义
                            ev.append({'t': t, 'dgf': v, 'cycles': None,
                                       'kind': 'loss'})
                            continue
                        # 整周一致性检验：1 周周跳只会让 GF 变化 λ1 或 -λ2。
                        k1, k2 = v / lam1, -v / lam2
                        n1, n2 = int(round(k1)), int(round(k2))
                        r1 = abs(k1 - n1) if n1 else 9.9     # n=0 不算周跳
                        r2 = abs(k2 - n2) if n2 else 9.9
                        if min(r1, r2) <= SLIP_FIT_TOL:
                            n_int += 1
                            ev.append({'t': t, 'dgf': v,
                                       'cycles': n1 if r1 <= r2 else n2,
                                       'kind': 'slip'})
                        # 否则不计入周跳：不是整周跳变
                    row['n_raw'] = n_raw
                    # 「符合整周比例」是关键判据：真周跳必然 n·λ，
                    # 该比例应远高于随机水平（容差 0.10 周时随机约 36%）。
                    # 比例低而跳变密集 => 持续性跟踪异常，不是零散周跳。
                    row['int_ratio'] = (n_int / n_raw * 100) if n_raw else None
                    row['n_slips'] = sum(1 for e in ev if e['kind'] == 'slip')
                    row['n_loss'] = sum(1 for e in ev if e['kind'] == 'loss')
                    row['events'] = ev
                    row['chronic'] = bool(
                        n_raw >= SLIP_CHRONIC_MIN
                        and (row['int_ratio'] or 0) < SLIP_CHRONIC_RATIO)
                    # 持续性异常卫星的"周跳"只是噪声碰巧过筛，不计入总数
                    if not row['chronic']:
                        total += row['n_slips']
                    total_loss += row['n_loss']
                slip_rows.append(row)

        slip_rows.sort(key=lambda r: (-r['n_slips'], -r['n_loss']))
        return {
            'series': out_series,
            'slips': {
                'available': bool(slip_rows),
                'rows': slip_rows,
                'total': total,
                'total_loss': total_loss,
                'n_sats_checked': sum(1 for r in slip_rows if r['usable']),
                'n_sats_skipped': sum(1 for r in slip_rows if not r['usable']),
                'threshold': SLIP_GF_THRESHOLD,
                'fit_tol': SLIP_FIT_TOL,
                'loss_threshold': SLIP_LOSS_THRESHOLD,
                'method': '几何无关组合 GF = λ₁·L₁ − λ₂·L₂',
            },
        }

    @staticmethod
    def _split_by_frequency(epochs, c1, l1, c2, l2, stride=None,
                            sysc='G', f1=F_L1, f2=F_L2):
        """把卫星分成「真双频」和「第二频点列不可用」两组

        默认参数保持 GPS/L1-L2 行为不变；多系统 MP/电离层分析
        传入各自的系统字符与频率对。
        （原实现把系统字符硬编码成 'G'，频率硬编码成 F_L1/F_L2，
        只能算 GPS —— 多系统版本需要按系统传。）

        判据：**几何无关组合 GF = \u03bb\u2081\u00b7L1 \u2212 \u03bb\u2082\u00b7L2 的跨度**

        为什么改用载波、不再用伪距
        --------------------------
        原判据是 |C1\u2212P2| 的中位数 < 0.7 m，本意是识别"接收机没跟踪 L2、
        把 L1 的值填进 L2/P2 列"。但该判据**会误杀真卫星**：伪距差里混着
        接收机硬件偏差与观测噪声，而高仰角卫星的真实电离层延迟本来就小，
        两者叠加就可能落到亚米级。某些站点的 G11（中位 0.49 m）与
        G15（0.66 m）就是这样被剔除的 —— 可它们的 MP-RMS 只有
        0.845 / 1.048 m，都在站点均值（1.028 m）以下，是干净数据。

        载波 GF 是决定性的：若第二频点列真是第一频点的重复值，GF 会
        退化成 L1\u00b7(\u03bb\u2081\u2212\u03bb\u2082)，而 L1 \u2248 1e8 周，
        **跨度会是百万米级**；真双频时 GF 只剩电离层延迟差与模糊度常数，
        跨度是米级（实测四站 0.16 ~ 61.6 m）。两者差六个数量级，
        阈值取 GF_DUPLICATE_SPAN_M = 1000 m 既有巨大余量又不会误杀。

        为什么不用其他候选
        ------------------
        * 「|C1\u2212P2| 中位数」—— 见上，受噪声与硬件偏差影响，误杀真卫星。
        * 「GF 的**中位数**」—— GF 受周跳影响大，中位数不如跨度稳健；
          周跳只会把跨度推大，不会把重复值列的跨度压小。
        * 「P2 覆盖率」—— 部分卫星（GPS Block IIR/IIRM）只在过境的一段
          时间发射第二频点，覆盖率可能只有 52%~85%（实测 G17/G19/G22）。
          这类卫星在**有第二频点的历元上**数据完全真实，整颗剔除会白丢
          有效数据。正确做法是保留，仅在缺值的历元上跳过（由调用方处理）。

        返回 (dual_set, single_set)。
        """
        lam1 = C_LIGHT / f1
        lam2 = C_LIGHT / f2
        gf = collections.defaultdict(list)
        total = len(epochs)
        # 均匀抽样，覆盖整个观测时段
        if stride is None:
            stride = max(1, total // 3000)

        for k, (_, satdict) in enumerate(epochs):
            if k % stride:
                continue
            for satid, v in satdict.items():
                if not satid.startswith(sysc):
                    continue
                a, b = v.get(l1), v.get(l2)
                if a is not None and b is not None:
                    gf[satid].append(a * lam1 - b * lam2)

        dual, single = set(), set()

        # 样本量门槛：可见时间过短的卫星统计不可信（跨度算不准）
        usable = {s: v for s, v in gf.items() if len(v) >= 20}

        for satid, arr in usable.items():
            span = max(arr) - min(arr)
            if span > GF_DUPLICATE_SPAN_M:
                single.add(satid)
            else:
                dual.add(satid)

        # 抽样时样本不足的卫星（可见时间极短）一律排除，不作双频计算
        for satid in gf:
            if satid not in usable:
                single.add(satid)

        # 出现在历元里、但完全没有第二频点载波的 GPS 卫星也归入 single。
        # 必须遍历**全部**历元取并集：只看第一个历元会漏掉当时尚未升起
        # （或已落下）的卫星，把它们错判成可用。
        for _, satdict in epochs:
            for satid in satdict:
                if satid.startswith('G') and satid not in gf:
                    single.add(satid)
        return dual, single

    # ==================== 自动问题清单 ====================


def build_warnings(result):
    """对外暴露的问题清单构建（在 compute 之后调用，便于访问全量结果）"""
    warns = []
    info = result.get('info') or {}
    missing = result.get('missing') or {}
    snr = result.get('snr_band') or {}
    residual = result.get('residual') or {}
    continuity = result.get('continuity') or {}
    multipath = result.get('multipath') or {}

    overall = snr.get('overall')
    # ---- 严重: 频点整体缺失 ----
    for d in missing.get('dead_codes', [])[:6]:
        warns.append({
            'level': '严重',
            'title': f"{d['sys']} {d['code']} 观测值几乎全为空",
            'detail': (f"{d['sys']} 的 {d['code']} 观测值中 {d['rate']:.1f}% 为零或缺失"
                       f"（{d['zero']}/{d['total']}），"
                       f"说明该星座此频点未被接收或未记录。"),
        })

    # ---- 严重: 残差显著为负的卫星 ----
    bad_resid = [r for r in (residual.get('sats') or []) if r['residual'] < -2.5]
    for r in bad_resid[:8]:
        sev = '严重' if r['residual'] < -4 else '关注'
        warns.append({
            'level': sev,
            'title': f"{r['label']} 信号显著弱于同仰角基线",
            'detail': (f"实测 C/N0 {r['real']:.1f} dB-Hz，同仰角期望 {r['expect']:.1f} dB-Hz，"
                       f"残差 {r['residual']:+.1f} dB（样本 {r['n']}，"
                       f"平均仰角 {r['el_mean']:.1f}°）。"),
        })

    # ---- 关注: 真实失锁 ----
    for r in (continuity.get('loss_of_lock') or [])[:8]:
        warns.append({
            'level': '关注',
            'title': f"{r['label']} 存在真实失锁",
            'detail': (f"最长连续中断 {r['max_gap']} 历元"
                       f"（约 {r['max_gap'] / 60:.1f} 分钟），"
                       f"共 {r['n_gaps']} 次中断，完整率 {r['rate']:.1f}%。"),
        })

    # ---- 关注: 多路径偏大 ----
    mp_rows = multipath.get('rows') or []
    if mp_rows:
        worst = mp_rows[0]
        if worst['rms'] > 1.5:
            warns.append({
                'level': '关注',
                'title': f"{worst['label']} 多路径误差偏大",
                'detail': (f"MP-RMS = {worst['rms']:.3f} m"
                           f"（全站均值 {multipath.get('mean_rms', 0):.3f} m）。"),
            })

    # ---- 提示: 因 L2 非真双频而被剔除的卫星 ----
    # 接收机对未跟踪第二频点的卫星用第一频点值填充，这类"重复值列"会算出
    # 几十米的假多路径，必须剔除；但要告诉用户剔了哪些、为什么，否则会以为漏算。
    excl = multipath.get('excluded_sats') or []
    if excl:
        warns.append({
            'level': '提示',
            'title': f"{len(excl)} 颗 GPS 卫星未参与多路径计算",
            'detail': (f"涉及 {'、'.join(excl)}。"
                       f"判据是载波几何无关组合 "
                       f"GF = λ₁·L₁ − λ₂·L₂ 的跨度超过 "
                       f"{GF_DUPLICATE_SPAN_M:.0f} m —— 说明第二频点列是"
                       f"第一频点的重复值（此时 GF 会退化成 L₁·(λ₁−λ₂)，"
                       f"跨度达百万米级）。"
                       f"常见原因是接收机未跟踪第二频点、或该卫星可见时间"
                       f"过短导致样本不足。"),
        })
    if multipath.get('single_frequency'):
        warns.append({
            'level': '关注',
            'title': '该文件不具备 GPS L1/L2 真双频条件',
            'detail': ('所有 GPS 卫星的第二频点列都是第一频点的重复值，'
                       '多路径 MP 无法计算。'
                       '如需多路径分析，请确认接收机以双频模式记录。'),
        })

    # ---- 提示: 低信号占比高的卫星 ----
    if overall:
        for r in (snr.get('sats') or [])[:5]:
            if r['lt30'] > 30:
                warns.append({
                    'level': '提示',
                    'title': f"{r['label']} 低信号占比偏高",
                    'detail': (f"C/N0 < 30 dB-Hz 的比例为 {r['lt30']:.1f}%，"
                               f"均值 {r['mean']:.1f} dB-Hz。"),
                })

    # ---- 提示: 元数据不一致 ----
    if info.get('declared_interval') and info.get('modal_interval'):
        dec, modal = info['declared_interval'], info['modal_interval']
        if abs(dec - modal) > 0.01:
            warns.append({
                'level': '提示',
                'title': '采样间隔声明值与实际不符',
                'detail': (f"头文件声明 INTERVAL = {dec} s，"
                           f"实测众数间隔为 {modal} s。"),
            })
    if overall and overall.get('at_ceiling', 0) > 3:
        warns.append({
            'level': '提示',
            'title': f"C/N0 存在量化截断（上限 {overall['ceiling']:.0f} dB-Hz）",
            'detail': (f"{overall['at_ceiling']:.1f}% 的样本顶在最大值，"
                       f"说明接收机信噪比被硬件量化上限截断，"
                       f"真实信号可能还要更好。"),
        })

    # ---- 周跳 / 丢锁（几何无关组合判据）----
    slips = result.get('cycle_slips') or {}
    if slips.get('available'):
        n_loss = slips.get('total_loss', 0)
        n_slip = slips.get('total', 0)
        if n_loss:
            worst = max((r for r in slips.get('rows', []) if r.get('n_loss')),
                        key=lambda r: r['n_loss'], default=None)
            warns.append({
                'level': '严重',
                'title': f"检测到 {n_loss} 次载波丢锁 / 大跳变",
                'detail': (f"由几何无关组合 GF = λ₁L₁ − λ₂L₂ 判定，"
                           f"跳变幅度超过 {SLIP_LOSS_THRESHOLD} m（约 5 周以上）。"
                           + (f"最严重的是 {worst['label']}（{worst['n_loss']} 次）。"
                              if worst else '')
                           + "丢锁会迫使接收机重新初始化整周模糊度。"),
            })
        if n_slip:
            hot = [r for r in slips.get('rows', [])
                   if r.get('n_slips') and not r.get('chronic')][:5]
            desc = '、'.join(f"{r['label']} {r['n_slips']} 次" for r in hot)
            warns.append({
                'level': '关注',
                'title': f"检测到 {n_slip} 次整周周跳",
                'detail': (f"{desc}。均已通过整周一致性校验"
                           f"（与 n·λ 的偏差 ≤ {SLIP_FIT_TOL} 周），"
                           f"不符整周的跳变已剔除，未计入。"
                           f"存在持续性跟踪异常的卫星已单独统计，"
                           f"未重复计入此处。"),
            })
        skipped = slips.get('n_sats_skipped', 0)
        if skipped:
            names = [r['label'] for r in slips.get('rows', [])
                     if not r.get('usable')][:6]
            warns.append({
                'level': '提示',
                'title': f"{skipped} 颗卫星无法做周跳检测",
                'detail': (f"涉及 {'、'.join(names)}。这些卫星两个频点的载波列"
                           f"彼此不自洽（跳变幅度远超整周），"
                           f"为避免给出错误结论，本工具对其不做判定。"),
            })

        # ---- 持续性跟踪异常：跳变密集但大多不符整周 ----
        # 真周跳必然是 n·λ，单颗星若跳变很多而符合整周的比例却接近随机水平
        # （容差 0.10 周时随机约 36%），说明它在一段时间里反复失锁或输出
        # 非整周相位不连续 —— 这是比零散周跳更严重的问题，单独报出来。
        chronic = [r for r in slips.get('rows', []) if r.get('chronic')]
        if chronic:
            chronic.sort(key=lambda r: -r['n_raw'])
            names = '、'.join(f"{r['label']}（{r['n_raw']} 次，符合整周仅 "
                              f"{(r['int_ratio'] or 0):.0f}%）"
                             for r in chronic[:5])
            warns.append({
                'level': '严重',
                'title': f"{len(chronic)} 颗卫星出现持续性载波跟踪异常",
                'detail': (f"{names}。这些卫星的几何无关组合跳变非常密集，"
                           f"但绝大多数不符整周（真周跳必然是 n·λ，"
                           f"随机值约 36% 会碰巧通过校验）——"
                           f"说明不是零散的周跳，而是接收机在这段时间里"
                           f"反复失锁或输出非整周的相位不连续。"
                           f"请在「观测诊断」页的原始观测量上定位具体时段。"),
            })

    # ---- DOP / 几何强度 ----
    dop = result.get('dop_nsat') or {}
    if dop.get('available'):
        st = dop.get('stats') or {}
        eph_sys = list(dop.get('eph_systems') or [])
        all_sys = list((info.get('sys_sat_count') or {}).keys())
        miss = [s for s in all_sys if s not in eph_sys]
        if miss:
            warns.append({
                'level': '提示',
                'title': f"星历仅覆盖 {'、'.join(eph_sys)}，几何/仰角分析不完整",
                'detail': (f"观测文件里有 {'、'.join(all_sys)} 共 "
                           f"{len(all_sys)} 个系统，但星历文件只包含 "
                           f"{'、'.join(eph_sys)}。因此 DOP、卫星数、"
                           f"SNR/MP–仰角散点 **只覆盖 {'、'.join(eph_sys)}**，"
                           f"数值会明显偏大/偏少。"
                           f"点「🗺 加载星历」换用 IGS 的 BRDC 广播星历"
                           f"即可覆盖全部星座。"),
            })
        if st.get('pdop_gt6', 0) > 10:
            warns.append({
                'level': '关注',
                'title': f"PDOP 偏大：{st['pdop_gt6']:.1f}% 的历元超过 6",
                'detail': (f"PDOP 均值 {st['pdop']['mean']:.2f}、"
                           f"最大 {st['pdop']['max']:.2f}，"
                           f"参与解算卫星数最少 {st['min_sat']} 颗。"
                           f"几何强度不足会直接放大定位误差。"),
            })
    else:
        warns.append({
            'level': '提示',
            'title': 'DOP / 可见卫星数时序不可用',
            'detail': dop.get('reason') or '缺少星历或测站坐标。',
        })

    order = {'严重': 0, '关注': 1, '提示': 2}
    warns.sort(key=lambda w: order.get(w['level'], 9))
    return warns


# -------------------- 对外统一的入口函数 --------------------


def lite_summary(result):
    """把结果压缩成适合在状态栏/摘要面板显示的精简字典"""
    if not result or result.get('error'):
        return {'error': (result or {}).get('error', '无数据')}

    info = result.get('info', {})
    snr = (result.get('snr_band') or {}).get('overall') or {}
    cont = result.get('continuity') or {}
    mp = result.get('multipath') or {}

    return {
        'file': info.get('file'),
        'marker': info.get('marker'),
        'span': f"{info.get('t_first_str')} ~ {info.get('t_last_str')}",
        'n_epochs': info.get('n_epochs'),
        'n_sats': info.get('n_sats'),
        'avg_sats': info.get('avg_sats_per_epoch'),
        'snr_mean': snr.get('mean'),
        'snr_lt30': snr.get('lt30'),
        'rate_mean': cont.get('rate_mean'),
        'mp_rms': mp.get('mean_rms'),
    }
