# -*- coding: utf-8 -*-
"""单点定位（SPP）—— 由广播星历逐历元解算接收机高程（纯逻辑，无 GUI 依赖）

用途
----
在静态观测质量分析里，用户希望看到「高程随时间的变化趋势」。不依赖外部
后处理 .pos 文件，直接用观测文件逐历元的伪距观测值 + 同批次广播星历，
做最小二乘单点定位，取解算位置的大地高作为高程。

原理
----
对每个历元，建立非线性伪距观测方程

    rho_i + c*dt_sat_i - c*dt_r = ||sat_i - rx|| + 误差项

对接收机位置 (x,y,z) 与接收机钟差 dt_r 四个未知数做牛顿拟合：
先由卫星钟差系数与相对论改正算出卫星钟误差，做地球自转（Sagnac）改正，
再迭代到位置收敛。

注意
----
* 精度为米级，只适合看**相对趋势**（高程漂移、跳变），不用于精密高程。
* GLONASS 星历未解析，解算自动只使用开普勒类卫星。
* 缺少广播星历或某一历元可用的伪距 - 卫星对不足 4 个时，该历元跳过。
"""

import math

from .rinex import C_LIGHT, GM, OMEGA_E, doy_of, day0_timestamp, is_pseudorange


def _pick_pseudorange(satdict):
    """从一颗卫星的观测值里挑一个伪距（C*/P*），返回 (code, value)。

    取该卫星列出的第一个有效伪距；跳过低/非正的不合理值。
    """
    pr = None
    pr_val = 0.0
    for code, val in satdict.items():
        if is_pseudorange(code):
            if pr is None or code[:2] < pr[:2]:
                pr, pr_val = code, val
    if pr is None or not pr_val or pr_val <= 1e4:
        return None
    return pr, pr_val


def _sat_clock(eph, tt):
    """由钟差多项式 + 相对论改正计算卫星钟误差 dt_sat（秒）"""
    dt = (eph.get('af0') or 0.0) \
        + (eph.get('af1') or 0.0) * (tt - eph['toc']) \
        + (eph.get('af2') or 0.0) * (tt - eph['toc']) ** 2
    # 相对论改正：-2 * sqrt(GM*A) / c^2 * e * sin(E)
    A = eph['sqrtA'] ** 2
    n0 = math.sqrt(GM / A ** 3)
    tk = tt - eph['toe_abs']
    if tk > 302400:
        tk -= 604800
    if tk < -302400:
        tk += 604800
    M = eph['M0'] + (n0 + (eph.get('deltaN') or 0.0)) * tk
    e = eph['e']
    E = M
    for _ in range(20):
        dE = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-13:
            break
    rel = -2.0 * math.sqrt(GM * A) / (C_LIGHT * C_LIGHT) * e * math.sin(E)
    return dt + rel


def _rotate_satellite(x, y, z, omega_dt):
    """地球自转改正：把卫星坐标绕 Z 轴转 omega_dt 弧度"""
    c, s = math.cos(omega_dt), math.sin(omega_dt)
    return c * x + s * y, -s * x + c * y, z


def compute_height_series(rinex_obs, eph, approx_xyz=None,
                          elev_mask=10.0, max_iter=12):
    """逐历元单点定位，返回高程序列（若数据不足返回 None）

    参数
    ----
    rinex_obs : RinexObs       已 parse 的观测文件
    eph       : BroadcastEphemeris 已 parse 的广播星历（含 af0/af1/af2）
    approx_xyz: 测站 ECEF 近似坐标（优先用 RINEX 头 APPROX POSITION XYZ）

    返回
    ----
    None 或 dict:
        {
          'available': bool,
          'reason':  str,                    # 不可用时给出原因
          'secs':      [浮点, ...],           # 当天秒数（观测历元时刻）
          't_abs':     [浮点, ...],           # 绝对 Unix 时间戳（UTC）
          'heights':   [浮点, ...],           # 大地高 H（米）
          'lats':[...], 'lons':[...],        # 解算的大地经纬度
          'n_sats':    [int, ...],           # 每历元使用的卫星数
          'rms':       [浮点, ...],           # 每历元伪距残差 RMS（米）
          'step':      float,                # 实际抽样的历元间隔秒
        }
    """
    from . import geodesy

    if eph is None or not eph.epochs:
        return {'available': False, 'reason': '未提供广播星历，无法进行单点定位。'}

    epochs = rinex_obs.epochs
    if not epochs:
        return {'available': False, 'reason': '观测文件中没有历元数据。'}

    t0, _ = rinex_obs.time_span()
    if not t0:
        return {'available': False, 'reason': '无法从观测文件确定日期。'}

    day0 = day0_timestamp(t0[0], doy_of(t0[0], t0[1], t0[2]))
    n_ep = len(epochs)
    dt = 1.0
    if n_ep >= 2:
        d = epochs[1][0] - epochs[0][0]
        if d > 0:
            dt = d
    step = max(1, n_ep // 2400)          # 最多约 2400 个历元，控住计算成本

    # 定位初值：优先 RINEX 头近似坐标，否则用首个可解历元的伪距加权均值
    x = list(approx_xyz) if approx_xyz else [0.0, 0.0, 0.0]

    secs_out, t_abs, hs, lats, lons, ns, rmss = [], [], [], [], [], [], []

    elev = math.radians(elev_mask)
    for d_, (secs, satdict) in enumerate(epochs):
        if d_ % step != 0:
            continue
        if not satdict:
            continue
        t_recv = day0 + secs

        # ---- 装配观测向量：伪距、卫星位置（含自转/钟差改正）----
        rows = []              # (x, y, z, rho_corrected, satid)
        for satid, vals in satdict.items():
            if satid[0] == 'R':
                continue       # GLONASS 星历未解析
            pr = _pick_pseudorange(vals)
            if pr is None:
                continue
            _code, rho = pr
            d_eph = eph.best(satid, t_recv)
            if d_eph is None:
                continue

            # 传输时间近似 + 地球自转改正
            tt = t_recv - rho / C_LIGHT
            try:
                sx, sy, sz = eph.sat_ecef(d_eph, tt)
            except (ValueError, TypeError, ZeroDivisionError):
                continue
            wdt = OMEGA_E * rho / C_LIGHT
            sx, sy, sz = _rotate_satellite(sx, sy, sz, wdt)

            # 卫星钟误差修正伪距
            rho_c = rho + C_LIGHT * _sat_clock(d_eph, tt)
            rows.append((sx, sy, sz, rho_c))

        if len(rows) < 4:
            continue

        # ---- 最小二乘迭代 ----
        ok = False
        clk = 0.0                      # 接收机钟差，秒（每次叠加增量）
        for _ in range(max_iter):
            A = []
            b = []
            for sx, sy, sz, rho_c in rows:
                r = math.sqrt((sx - x[0]) ** 2
                              + (sy - x[1]) ** 2
                              + (sz - x[2]) ** 2)
                if r < 1e-5:
                    continue
                A.append([(x[0] - sx) / r, (x[1] - sy) / r,
                          (x[2] - sz) / r, C_LIGHT])
                b.append(rho_c - r - C_LIGHT * clk)
            if len(A) < 4:
                break
            # 法方程 A^T A dx = A^T b
            AtA = [[sum(A[k][i] * A[k][j] for k in range(len(A)))
                    for j in range(4)] for i in range(4)]
            Atb = [sum(A[k][i] * b[k] for k in range(len(A)))
                   for i in range(4)]
            sol = _solve4(AtA, Atb)
            if sol is None:
                break
            dx, dy, dz, dtr = sol
            x = [x[0] + dx, x[1] + dy, x[2] + dz]
            clk += dtr
            # 位置与钟差的**增量**都收敛才算解出
            if (abs(dx) + abs(dy) + abs(dz) <= 1e-4
                    and abs(dtr) <= 1e-8):
                ok = True
                # 伪距后验残差 RMS：扣除接收机钟差 C*clk
                resid = [(rho_c - C_LIGHT * clk
                          - math.sqrt((sx - x[0]) ** 2
                                      + (sy - x[1]) ** 2
                                      + (sz - x[2]) ** 2))
                         for sx, sy, sz, rho_c in rows]
                rms = math.sqrt(sum(r * r for r in resid) / len(resid))
                break
        if not ok:
            continue
        if not (all(math.isfinite(v) for v in x)):
            continue

        try:
            lat, lon, h = geodesy.ecef_to_blh(*x, ell=geodesy.WGS84)
        except Exception:      # noqa: BLE001 - 极区数值奇异时跳过该历元
            continue
        if abs(lat) > 89.0 or abs(h) > 2e7:
            continue

        secs_out.append(secs)
        t_abs.append(t_recv)
        hs.append(h)
        lats.append(lat)
        lons.append(lon)
        ns.append(len(rows))
        rmss.append(rms)

    if not hs:
        return {'available': False,
                'reason': '可视卫星不足 4 颗，或星历与观测时间不匹配。'}

    return {
        'available': True,
        'secs': secs_out,
        't_abs': t_abs,
        'heights': hs,
        'lats': lats,
        'lons': lons,
        'n_sats': ns,
        'rms': rmss,
        'step': dt * step,
    }


def _solve4(a, b):
    """解 4x4 线性方程组 a x = b（高斯消元），奇异返回 None"""
    # 复制避免污染输入
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    n = 4
    for col in range(n):
        # 选主元
        piv = col
        best = abs(m[col][col])
        for r in range(col + 1, n):
            v = abs(m[r][col])
            if v > best:
                best, piv = v, r
        if best < 1e-18:
            return None
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        pivot = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pivot
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]