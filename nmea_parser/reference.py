# -*- coding: utf-8 -*-
"""参考真值来源（ReferenceSource）

四种来源，接口统一
-----------------
RTK 测试的"真值"可以来自四个地方，它们的取数方式差别很大：

====================  ==================================================
类                     说明
====================  ==================================================
``KnownPointRef``     已知控制点，坐标恒定（最硬的真值）
``PosFileRef``        第三方后处理结果（RTKPOST 的 .pos），按时刻取
``PeerDeviceRef``     另一台设备的解算序列（基线互比，不上绝对真值）
``SelfMeanRef``       设备自身长时间观测的稳健均值（只能算重复性）
====================  ==================================================

把这四种**抽象成同一个接口**，是为了让精度评估引擎只认接口：
评估逻辑写一遍，四种真值都能用，场景层（静态/比对/动态/基线）
也就都变成薄封装，不用各自重写时间对齐与坐标转换。
"""

import math

from . import geodesy

# 时间对齐容差（秒）：参考点时刻与设备历元相差超过它就视为"无参考"
DEFAULT_MAX_DT = 1.5

# 解算质量（Q）**不是有序量**：4 是固定、5 是浮动，而 5 > 4。
# 所以"只要固定解"不能用 q >= 4 表达，必须用显式集合。
REQUIRES = {
    'any': None,                # 不限
    'fix': frozenset({4}),      # 仅 RTK 固定
    'fix_float': frozenset({4, 5}),
    'valid': frozenset({1, 2, 4, 5}),
}


def normalize_require(spec):
    """把质量要求归一成 frozenset 或 None

    接受 ``None`` / ``'any'`` / ``'fix'`` / ``'fix_float'`` / ``'valid'``
    或任意整数集合（如 ``(4,)``）。
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        key = spec.lower()
        if key not in REQUIRES:
            raise ValueError(f'未知的质量要求 {spec!r}，'
                             f'可选 {sorted(REQUIRES)}')
        return REQUIRES[key]
    return frozenset(int(x) for x in spec)


def require_ok(spec, q):
    """按质量要求判断某个历元是否可用"""
    if spec is None:
        return True
    return q is not None and q in spec


class RefPoint:
    """某一时刻的参考坐标

    ``t`` 是**请求时刻**（插值结果就对应该时刻）；
    ``src_t`` 是**取数的原始采样时刻**。两者不同：
    最近邻取数时参考值只在 src_t 上成立，时间容差必须按 src_t 判 ——
    若按 t 判，dt 恒为 0，容差形同虚设（本项目踩过这个坑）。
    """

    __slots__ = ('t', 'lat', 'lon', 'h', 'q', 'interpolated', 'note',
                 'src_t')

    def __init__(self, t, lat, lon, h, q=None, interpolated=False, note='',
                 src_t=None):
        self.t = t
        self.lat = lat
        self.lon = lon
        self.h = h
        self.q = q
        self.interpolated = interpolated
        self.note = note
        self.src_t = src_t if src_t is not None else t

    def __repr__(self):
        return (f'<RefPoint {self.t:%H:%M:%S} {self.lat:.9f},'
                f'{self.lon:.9f} h={self.h:.3f} q={self.q}>')


class ReferenceSource:
    """参考真值来源的抽象基类"""

    kind = 'base'
    name = '参考'

    def at(self, t):
        """给定时刻返回 RefPoint；无参考时返回 None"""
        raise NotImplementedError

    def info(self):
        """一行说明，写进报告"""
        return self.name

    def is_absolute(self):
        """是否是绝对真值（已知点/后处理）而非相对参考（自身均值/另一台设备）

        这个区分很重要：相对参考只能得出**重复性/一致性**，
        不能声称"精度"，报告里必须说清楚。
        """
        return True

    def summary(self):
        """结构化的说明，供界面与报告使用"""
        return {'kind': self.kind, 'name': self.name,
                'absolute': self.is_absolute(), 'detail': self.info()}


# ==================== 1. 已知控制点 ====================


class KnownPointRef(ReferenceSource):
    """已知控制点：坐标恒定

    支持三种输入方式（都归一成 BLH 存储）：
      * 经纬高 ``lat/lon/h``
      * ECEF ``xyz``
      * 高斯平面 ``gauss=(X, Y, zone, width)`` —— 会投影反算成经纬度

    天线高改正：若已知点是**标石面**而设备量的是**天线相位中心**，
    则应把天线高加进 h（``antenna_h`` 正值表示相位中心高于标石）。
    """

    kind = 'known_point'

    def __init__(self, lat=None, lon=None, h=0.0, xyz=None,
                 gauss=None, name='已知点', antenna_h=0.0,
                 ell=geodesy.WGS84, source_note=''):
        self.ell = ell
        if xyz is not None:
            lat, lon, h = geodesy.ecef_to_blh(*xyz, ell=ell)
            source_note = source_note or 'ECEF 输入'
        elif gauss is not None:
            X, Y, zone, width = gauss
            lat, lon = geodesy.gauss_inverse(X, Y, width=width, zone=zone,
                                             ell=geodesy.CGCS2000)
            h = h or 0.0
            source_note = source_note or f'高斯 {width}° 带 {zone} 带输入'
        if lat is None or lon is None:
            raise ValueError('已知点必须给出 lat/lon、xyz 或 gauss')
        self.lat = float(lat)
        self.lon = float(lon)
        self.h = float(h or 0.0) + float(antenna_h or 0.0)
        self.antenna_h = float(antenna_h or 0.0)
        self.name = name
        self.note = source_note

    def at(self, t):
        return RefPoint(t, self.lat, self.lon, self.h, note=self.note)

    @property
    def xyz(self):
        return geodesy.blh_to_ecef(self.lat, self.lon, self.h, self.ell)

    def info(self):
        s = (f'已知点 {self.name}: {self.lat:.9f}°, {self.lon:.9f}°, '
             f'{self.h:.4f} m')
        if self.antenna_h:
            s += f'（含天线高 {self.antenna_h:+.3f} m）'
        if self.note:
            s += f' [{self.note}]'
        return s


# ==================== 2/3. 序列型参考（.pos 或另一台设备） ====================


class SeriesRef(ReferenceSource):
    """以另一条解算序列为参考（.pos 后处理结果 / 另一台设备）

    取数策略：找时刻最近的一条记录；**时间容差按参考记录的原始采样时刻判**
    （见 RefPoint.src_t 的说明）。
    ``interpolate=True`` 时对前后两条做线性插值，适合参考序列采样率
    低于设备的场合。

    ``require_q`` 过滤参考记录的解算质量 —— 默认 ``None``（不筛），
    但 ``PosFileRef`` 默认只要固定解（``'fix'``）：拿浮动解当真值
    会把参考自身的误差算到设备头上。
    **注意 Q 不是有序量**，所以这里用集合而不是阈值（见 normalize_require）。
    """

    kind = 'series'

    def __init__(self, series, name='参考序列', max_dt=DEFAULT_MAX_DT,
                 interpolate=False, require_q=None, absolute=True,
                 note=''):
        self.series = series
        self.name = name
        self.max_dt = float(max_dt)
        self.interpolate = bool(interpolate)
        self.require_q = normalize_require(require_q)
        self._absolute = absolute
        self.note = note
        # 预建索引：按时间排序一次，之后二分查找
        self._ts = list(series.t)
        self._idx = sorted(range(len(self._ts)), key=lambda i: self._ts[i])
        self._ts_sorted = [self._ts[i] for i in self._idx]
        self.n_total = len(self._ts)
        self.n_usable = sum(1 for i in self._idx
                            if require_ok(self.require_q, series.q[i]))

    def _q_ok(self, q):
        return require_ok(self.require_q, q)

    def is_absolute(self):
        return self._absolute

    def _bisect(self, t):
        """返回 t 在排序后时间轴上的插入位置"""
        lo, hi = 0, len(self._ts_sorted)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ts_sorted[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def at(self, t):
        if not self._ts_sorted:
            return None
        pos = self._bisect(t)
        cands = [k for k in (pos - 1, pos) if 0 <= k < len(self._ts_sorted)]
        if not cands:
            return None
        if self.interpolate:
            # 需要前后两条都可用才插值，否则退回最近邻
            lo = pos - 1
            hi = pos
            if (0 <= lo and hi < len(self._ts_sorted)
                    and lo != hi):
                i0, i1 = self._idx[lo], self._idx[hi]
                t0, t1 = self._ts[i0], self._ts[i1]
                if (self._q_ok(self.series.q[i0])
                        and self._q_ok(self.series.q[i1])
                        and t0 <= t <= t1 and t1 > t0):
                    f = (t - t0).total_seconds() / (t1 - t0).total_seconds()
                    return RefPoint(
                        t,
                        self.series.lat[i0] + f * (self.series.lat[i1]
                                                   - self.series.lat[i0]),
                        self.series.lon[i0] + f * (self.series.lon[i1]
                                                   - self.series.lon[i0]),
                        self.series.h[i0] + f * (self.series.h[i1]
                                                 - self.series.h[i0]),
                        q=min(self.series.q[i0], self.series.q[i1]),
                        interpolated=True, src_t=t)
        # 最近邻
        best = None
        for k in cands:
            i = self._idx[k]
            if not self._q_ok(self.series.q[i]):
                continue
            dt = abs((self._ts[i] - t).total_seconds())
            if best is None or dt < best[0]:
                best = (dt, i)
        if best is None or best[0] > self.max_dt:
            return None
        dt, i = best
        # src_t 必须是参考记录的**真实采样时刻**，不能填请求时刻 t ——
        # 否则调用方算出的 dt 恒为 0，时间容差完全失效
        return RefPoint(t, self.series.lat[i], self.series.lon[i],
                        self.series.h[i], q=self.series.q[i],
                        src_t=self._ts[i])

    def info(self):
        s = (f'{self.name}: {self.n_total} 个历元'
             f'（质量达标 {self.n_usable}）')
        if self.series.t:
            s += (f'，{self.series.t[0]:%Y-%m-%d %H:%M:%S} ~ '
                  f'{self.series.t[-1]:%H:%M:%S}')
        s += f'，对齐容差 {self.max_dt:.1f} s'
        if self.interpolate:
            s += '，线性插值'
        if not self._absolute:
            s += '（相对参考，只能评估一致性）'
        if self.note:
            s += f' [{self.note}]'
        return s


class PosFileRef(SeriesRef):
    """第三方后处理结果（RTKPOST 的 .pos）作为参考

    默认只要**固定解**（require_q='fix'）—— 拿浮动解当真值会把参考
    自身的误差算到设备头上。若该 .pos 全程没有固定解，对齐结果会是 0，
    此时 ``align`` 的 ``n_no_ref`` 会把原因暴露出来。
    """

    kind = 'pos_file'

    def __init__(self, series, filename='', **kw):
        kw.setdefault('name', f'后处理参考 {filename}' if filename
                      else '后处理参考')
        kw.setdefault('absolute', True)
        kw.setdefault('require_q', 'fix')
        super().__init__(series, **kw)
        self.filename = filename


class PeerDeviceRef(SeriesRef):
    """另一台设备的解算序列作为参考（基线互比）

    **不是绝对真值** —— 两台设备可能同时有偏，只能评估一致性/相对精度。
    质量要求默认只要固定解（两台都在固定状态下比较才有意义）。
    """

    kind = 'peer_device'

    def __init__(self, series, label='另一台设备', **kw):
        kw.setdefault('name', f'对比设备 {label}')
        kw.setdefault('absolute', False)
        kw.setdefault('require_q', 'fix')
        super().__init__(series, **kw)
        self.label = label


# ==================== 4. 自建参考（设备自身稳健均值） ====================


class SelfMeanRef(ReferenceSource):
    """设备自身长时间观测的稳健均值

    **这是相对参考，不是真值** —— 只能评估重复性（重复观测的离散度），
    不能声称精度。报告与界面必须把这一点说清楚，否则用户会拿它当精度用。

    稳健做法：先取中位数作初值，按 ENU 径向误差做迭代 3σ 剔除
    （σ 用 MAD 估计，避免粗差把 σ 抬起来），再取剔除后的均值。
    """

    kind = 'self_mean'

    def __init__(self, series, require_q='fix', t_from=None, t_to=None,
                 sigma_k=3.0, max_iter=5, name='自建参考（自身均值）'):
        self.series = series
        self.name = name
        self.require_q = normalize_require(require_q)
        self.t_from = t_from
        self.t_to = t_to
        self.sigma_k = float(sigma_k)
        # max_iter 必须存到实例上 —— 迭代循环读的是 self.max_iter，
        # 少这一句会直接 AttributeError
        self.max_iter = int(max_iter)
        self.rejects = 0
        self.n_used = 0
        self.std_e = self.std_n = self.std_u = None
        self.horiz_rms = None
        self.converged = None
        self.n_iter = 0
        self.lat = self.lon = self.h = None
        self._estimate()

    def _mask(self):
        s = self.series
        out = []
        for i in range(len(s.t)):
            if not require_ok(self.require_q, s.q[i]):
                continue
            if self.t_from is not None and s.t[i] < self.t_from:
                continue
            if self.t_to is not None and s.t[i] > self.t_to:
                continue
            if s.lat[i] is None or s.lon[i] is None:
                continue
            out.append(i)
        return out

    def _estimate(self):
        idx = self._mask()
        if len(idx) < 2:
            raise ValueError(
                f'可用历元太少（{len(idx)}），无法建立自建参考。'
                f'请放宽质量要求或延长时间范围。')
        s = self.series
        # 初值用中位数，抗粗差
        lat = sorted(s.lat[i] for i in idx)[len(idx) // 2]
        lon = sorted(s.lon[i] for i in idx)[len(idx) // 2]
        hs = sorted(s.h[i] for i in idx if s.h[i] is not None)
        h = hs[len(hs) // 2] if hs else 0.0

        for it in range(self.max_iter):
            self.n_iter = it + 1
            to_enu = geodesy.make_local_frame(lat, lon, h)
            e, n, u = [], [], []
            for i in idx:
                a, b, c = to_enu(s.lat[i], s.lon[i],
                                 s.h[i] if s.h[i] is not None else h)
                e.append(a)
                n.append(b)
                u.append(c)
            rad = [math.hypot(e[k], n[k]) for k in range(len(idx))]
            med = sorted(rad)[len(rad) // 2]
            mad = sorted(abs(x - med) for x in rad)[len(rad) // 2]
            sigma = max(1.4826 * mad, 1e-6)
            keep = [k for k in range(len(idx))
                    if abs(rad[k] - med) <= self.sigma_k * sigma]
            if not keep:
                keep = list(range(len(idx)))
            if len(keep) == len(idx):
                # 已经收敛
                self.converged = True
                break
            self.rejects += len(idx) - len(keep)
            idx = [idx[k] for k in keep]
            lat = sum(s.lat[i] for i in idx) / len(idx)
            lon = sum(s.lon[i] for i in idx) / len(idx)
            hv = [s.h[i] for i in idx if s.h[i] is not None]
            h = sum(hv) / len(hv) if hv else h
        else:
            self.converged = False

        self.lat, self.lon, self.h = lat, lon, h
        self.n_used = len(idx)
        # 记录保留下来的**原始历元下标**：让调用方（如重复性统计）能用
        # 与稳健均值同一套子集，避免"均值剔了粗差、离散度没剔"的口径不一
        self.keep_epochs = list(idx)
        self.rejected_epochs = [i for i in self._mask() if i not in set(idx)]
        # 重复性指标
        to_enu = geodesy.make_local_frame(lat, lon, h)
        es, ns, us = [], [], []
        for i in idx:
            a, b, c = to_enu(s.lat[i], s.lon[i],
                             s.h[i] if s.h[i] is not None else h)
            es.append(a)
            ns.append(b)
            us.append(c)

        def rms(v):
            return math.sqrt(sum(x * x for x in v) / len(v)) if v else None
        self.std_e = rms(es)
        self.std_n = rms(ns)
        self.std_u = rms(us)
        self.horiz_rms = (math.sqrt(self.std_e ** 2 + self.std_n ** 2)
                          if self.std_e is not None else None)

    def at(self, t):
        return RefPoint(t, self.lat, self.lon, self.h,
                        note='自身稳健均值（相对参考）')

    def is_absolute(self):
        return False

    def info(self):
        return (f'{self.name}: {self.lat:.9f}°, {self.lon:.9f}°, {self.h:.4f} m'
                f'（用 {self.n_used} 个历元，剔除 {self.rejects} 个粗差）'
                f'  重复性 RMS: 水平 {self.horiz_rms:.4f} m / '
                f'高程 {self.std_u:.4f} m'
                f'  ⚠ 相对参考，只能评估重复性')


# ==================== 工厂 ====================


def build_reference(kind, **kw):
    """按类型名构建参考源（供界面下拉框使用）"""
    table = {
        'known_point': KnownPointRef,
        'pos_file': PosFileRef,
        'peer_device': PeerDeviceRef,
        'self_mean': SelfMeanRef,
    }
    cls = table.get(kind)
    if cls is None:
        raise ValueError(f'未知的参考类型: {kind}（可选 {list(table)}）')
    return cls(**kw)
