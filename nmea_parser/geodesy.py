# -*- coding: utf-8 -*-
"""大地测量坐标转换（纯逻辑，无 GUI 依赖）

RTK 测试为什么需要它
-------------------
设备输出的坐标、已知控制点的坐标、后处理结果的坐标，三者往往**不在同一
坐标系/同一表示形式**里：

  * 设备 NMEA 给的是 WGS84 的经纬度（度，十进制度）
  * 接收机原始日志（.XYZ）与 RINEX 头给的是 **ECEF 地心直角坐标**（米）
  * 测绘已知控制点通常是 **CGCS2000 高斯投影平面坐标**（3° 或 6° 带）
  * 工程上要的常常是**正高**，而设备给的是大地高，差一个高程异常 N

不把这些统一到同一个坐标系、同一种表示，就没法做"设备值 vs 真值"的比对。

本模块提供：
  * 椭球定义（WGS84 / CGCS2000）
  * ECEF ↔ 大地经纬高（BLH）
  * 局地 ENU / NED 增量换算
  * 横墨卡托投影（TM）与高斯-克吕格投影（CGCS2000 3°/6° 带）正反算
  * 四参数（平面相似）与七参数（Bursa-Wolf 空间相似）坐标转换
  * 大地水准面格网读取（GeographicLib PGM / NOAA GTX）与双线性插值

角度约定：对外一律用**十进制度**，内部转弧度。
"""

import math
import struct

# ==================== 椭球 ====================


class Ellipsoid:
    """参考椭球

    CGCS2000 与 WGS84 的 a 相同、扁率略异（1/f 差别在小数点后第 8 位），
    实际使用中差异远小于测量噪声，但保留区分以免概念混淆。
    """

    def __init__(self, a, inv_f, name):
        self.a = a
        self.inv_f = inv_f
        self.f = 1.0 / inv_f
        self.b = a * (1.0 - self.f)
        self.e2 = self.f * (2.0 - self.f)          # 第一偏心率平方
        self.ep2 = self.e2 / (1.0 - self.e2)       # 第二偏心率平方
        self.name = name

    def __repr__(self):
        return f'<Ellipsoid {self.name} a={self.a} 1/f={self.inv_f}>'


WGS84 = Ellipsoid(6378137.0, 298.257223563, 'WGS84')
CGCS2000 = Ellipsoid(6378137.0, 298.257222101, 'CGCS2000')

# 常用常量
DEG = math.pi / 180.0
RHO_SEC = 180.0 * 3600.0 / math.pi      # 1 弧度 = 多少秒


def _norm_lon(lon):
    """把经度归一到 (-180, 180]"""
    while lon > 180.0:
        lon -= 360.0
    while lon <= -180.0:
        lon += 360.0
    return lon


def _norm_lat(lat):
    """把纬度限制在 [-90, 90]"""
    return max(-90.0, min(90.0, lat))


# ==================== ECEF ↔ BLH ====================


def blh_to_ecef(lat, lon, h=0.0, ell=WGS84):
    """大地经纬高（十进制度, 米）-> ECEF 地心直角坐标（米）"""
    latr = _norm_lat(lat) * DEG
    lonr = _norm_lon(lon) * DEG
    sl, cl = math.sin(latr), math.cos(latr)
    so, co = math.sin(lonr), math.cos(lonr)
    n = ell.a / math.sqrt(1.0 - ell.e2 * sl * sl)
    x = (n + h) * cl * co
    y = (n + h) * cl * so
    z = (n * (1.0 - ell.e2) + h) * sl
    return x, y, z


def ecef_to_blh(x, y, z, ell=WGS84, eps=1e-12, max_iter=30):
    """ECEF -> 大地经纬高（十进制度, 米）

    用 Bowring 初值 + 牛顿迭代。纯 Bowring 闭式解在极区附近误差会到
    亚毫米以上，迭代到收敛更稳妥，迭代次数上界很小。
    """
    p = math.hypot(x, y)
    if p < 1e-9:
        # 极点
        lat = 90.0 if z >= 0 else -90.0
        return lat, 0.0, abs(z) - ell.b

    lon = math.atan2(y, x)
    # Bowring 初值
    theta = math.atan2(z * ell.a, p * ell.b)
    st, ct = math.sin(theta), math.cos(theta)
    lat = math.atan2(z + ell.ep2 * ell.b * st ** 3,
                     p - ell.e2 * ell.a * ct ** 3)

    for _ in range(max_iter):
        sl = math.sin(lat)
        n = ell.a / math.sqrt(1.0 - ell.e2 * sl * sl)
        h = p / math.cos(lat) - n
        new_lat = math.atan2(z, p * (1.0 - ell.e2 * n / (n + h)))
        if abs(new_lat - lat) < eps:
            lat = new_lat
            break
        lat = new_lat

    sl = math.sin(lat)
    n = ell.a / math.sqrt(1.0 - ell.e2 * sl * sl)
    # 用 cos(lat) 求 h 在极区不稳，改用 z 分量
    if abs(math.cos(lat)) > 1e-6:
        h = p / math.cos(lat) - n
    else:
        h = abs(z) - ell.b
    return lat / DEG, _norm_lon(lon / DEG), h


def ecef_delta_to_enu(dx, dy, dz, ref_lat, ref_lon):
    """ECEF 增量 (Δx,Δy,Δz) -> 局地 ENU 增量，参考点经纬度（度）"""
    latr = ref_lat * DEG
    lonr = ref_lon * DEG
    sl, cl = math.sin(latr), math.cos(latr)
    so, co = math.sin(lonr), math.cos(lonr)
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    return e, n, u


def enu_delta_to_ecef(e, n, u, ref_lat, ref_lon):
    """局地 ENU 增量 -> ECEF 增量（ecef_delta_to_enu 的逆）"""
    latr = ref_lat * DEG
    lonr = ref_lon * DEG
    sl, cl = math.sin(latr), math.cos(latr)
    so, co = math.sin(lonr), math.cos(lonr)
    dx = -so * e - sl * co * n + cl * co * u
    dy = co * e - sl * so * n + cl * so * u
    dz = cl * n + sl * u
    return dx, dy, dz


def enu_delta_to_ned(e, n, u):
    """ENU -> NED（仅换轴，不含位置项）"""
    return n, e, -u


def blh_diff_to_enu(lat, lon, h, ref_lat, ref_lon, ref_h, ell=WGS84):
    """两点 BLH 之差 -> 以参考点为原点的 ENU 增量（米）

    注意用的是**椭球高**。若要正高需先做高程异常改正。
    """
    x1, y1, z1 = blh_to_ecef(lat, lon, h, ell)
    x0, y0, z0 = blh_to_ecef(ref_lat, ref_lon, ref_h, ell)
    return ecef_delta_to_enu(x1 - x0, y1 - y0, z1 - z0, ref_lat, ref_lon)


def distance(a_lat, a_lon, b_lat, b_lon, ell=WGS84):
    """两点的大地线距离（米）—— 用 ECEF 弦长近似，短基线足够精确

    长基线（>100 km）用弦长会低估，此时应换 Vincenty。RTK 测试的基线
    一般在几十公里内，弦长与大地线之差可忽略。
    """
    x1, y1, z1 = blh_to_ecef(a_lat, a_lon, 0.0, ell)
    x2, y2, z2 = blh_to_ecef(b_lat, b_lon, 0.0, ell)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def azimuth(a_lat, a_lon, b_lat, b_lon):
    """a -> b 的方位角（度，自北顺时针）"""
    _, _, _ = 0, 0, 0
    e, n, _u = blh_diff_to_enu(b_lat, b_lon, 0.0, a_lat, a_lon, 0.0)
    return math.degrees(math.atan2(e, n)) % 360.0


def az_el(obs_xyz, sat_xyz, obs_lat, obs_lon,
          min_range=1.0e6, max_range=6.0e7):
    """测站看向卫星的 (方位角°, 仰角°, 斜距 m)；异常时返回 None

    斜距落在 [min_range, max_range] 之外一律判为不可用 —— 轨道算错、
    星历匹配到别的卫星这类问题都会表现成斜距离谱，这一步是极便宜的兜底
    校验，能挡住后面所有基于它的仰角/天顶图/RMS 指标。

    参数同时要测站 ECEF 与经纬度：ECEF 用来算增量，经纬度用来把增量转到
    ENU。不要由经纬度反复重算 ECEF —— 这个函数在逐历元循环里调用，
    重算是明显浪费。

    注意本函数是 ENU 变换的**唯一实现**：`rinex.LocalFrame.az_el`
    现在委托到这里，别再各写一份（本项目曾有三套并存的局面）。
    """
    if obs_xyz is None or sat_xyz is None:
        return None
    dx = sat_xyz[0] - obs_xyz[0]
    dy = sat_xyz[1] - obs_xyz[1]
    dz = sat_xyz[2] - obs_xyz[2]
    e, n, u = ecef_delta_to_enu(dx, dy, dz, obs_lat, obs_lon)
    rng = math.sqrt(e * e + n * n + u * u)
    if not (min_range < rng < max_range):
        return None
    el = math.degrees(math.asin(max(-1.0, min(1.0, u / rng))))
    az = math.degrees(math.atan2(e, n)) % 360.0
    return az, el, rng


# ==================== 横墨卡托 / 高斯-克吕格 ====================


def _meridian_arc(lat, ell):
    """子午线弧长（从赤道起算，米）"""
    e2 = ell.e2
    a = ell.a
    return a * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat)
        - (35 * e2 ** 3 / 3072) * math.sin(6 * lat)
    )


def _footpoint_lat(M, ell):
    """子午线弧长 -> 底点纬度（反算，级数展开）"""
    e2 = ell.e2
    a = ell.a
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    return (mu
            + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))


def tm_forward(lat, lon, lon0, k0=1.0, false_easting=500000.0,
               false_northing=0.0, ell=WGS84):
    """横墨卡托投影正算 -> (东坐标, 北坐标)，单位米

    高斯-克吕格就是 k0 = 1.0 的横墨卡托；UTM 是 k0 = 0.9996。
    级数在中央子午线 ±3.5° 内精度优于 1 mm。
    """
    latr = lat * DEG
    dlon = _norm_lon(lon - lon0) * DEG
    sl, cl = math.sin(latr), math.cos(latr)
    tl = math.tan(latr)
    n = ell.a / math.sqrt(1 - ell.e2 * sl * sl)
    t = tl * tl
    c = ell.ep2 * cl * cl
    a_ = dlon * cl
    m = _meridian_arc(latr, ell)

    x = m + n * tl * (a_ ** 2 / 2
                      + (5 - t + 9 * c + 4 * c * c) * a_ ** 4 / 24
                      + (61 - 58 * t + t * t + 600 * c - 330 * ell.ep2)
                      * a_ ** 6 / 720)
    y = n * (a_ + (1 - t + c) * a_ ** 3 / 6
             + (5 - 18 * t + t * t + 72 * c - 58 * ell.ep2) * a_ ** 5 / 120)

    easting = k0 * y + false_easting
    northing = k0 * x + false_northing
    return easting, northing


def tm_inverse(easting, northing, lon0, k0=1.0, false_easting=500000.0,
               false_northing=0.0, ell=WGS84):
    """横墨卡托投影反算 -> (纬度, 经度) 十进制度

    注意单位：内部级数全用**弧度**，最后统一转度。
    lon0 传入是度，必须先乘 DEG 再与弧度修正量相加 ——
    直接相加会得到荒谬的结果（度 + 弧度）。
    """
    x = (northing - false_northing) / k0
    y = (easting - false_easting) / k0

    lat_f = _footpoint_lat(x, ell)              # 弧度
    sl, cl = math.sin(lat_f), math.cos(lat_f)
    tl = math.tan(lat_f)

    n = ell.a / math.sqrt(1 - ell.e2 * sl * sl)
    r = ell.a * (1 - ell.e2) / (1 - ell.e2 * sl * sl) ** 1.5
    t = tl * tl
    c = ell.ep2 * cl * cl
    d = y / n

    lat = lat_f - (n * tl / r) * (
        d ** 2 / 2
        - (5 + 3 * t + 10 * c - 4 * c * c - 9 * ell.ep2) * d ** 4 / 24
        + (61 + 90 * t + 298 * c + 45 * t * t - 252 * ell.ep2
           - 3 * c * c) * d ** 6 / 720)
    lon = lon0 * DEG + (
        d
        - (1 + 2 * t + c) * d ** 3 / 6
        + (5 - 2 * c + 28 * t - 3 * c * c + 8 * ell.ep2 + 24 * t * t)
        * d ** 5 / 120) / cl
    return lat / DEG, _norm_lon(lon / DEG)


def gauss_zone(lon, width=3):
    """由经度求高斯投影带号与中央子午线经度

    **国内的带号编号不是 UTM 编号**，容易搞错：

    3° 带：带号 n = round(lon/3)，中央子午线 = 3n
        （如 121.3°E -> 40 带，中央子午线 120°）
    6° 带：带号 n = int(lon/6) + 1，中央子午线 = 6n − 3
        （如 121.3°E -> 21 带，中央子午线 123°）

    注意 6° 带不能用 UTM 的 ``int((lon+180)/6)+1``：那是以 180°W 起算的
    全球编号，对 121.3°E 会得出 51 带、中央子午线 303°，完全错误。
    国内 6° 带从 0° 起算，仅适用于 0~180°E 范围。
    """
    if width == 3:
        zone = int(round(lon / 3.0))
        if zone == 0:
            zone = 1
        return zone, zone * 3.0
    if width == 6:
        zone = int(lon / 6.0) + 1
        return zone, zone * 6.0 - 3.0
    raise ValueError('带宽只支持 3 或 6 度')


def gauss_forward(lat, lon, width=3, with_zone_prefix=True, ell=CGCS2000):
    """高斯-克吕格投影正算

    返回 (X 北坐标, Y 东坐标, 带号)。国内习惯 **X 为北、Y 为东**，
    且 Y 常加带号前缀（如 38 带写成 38500000 起）。
    """
    zone, lon0 = gauss_zone(lon, width)
    easting, northing = tm_forward(lat, lon, lon0, k0=1.0,
                                   false_easting=500000.0, ell=ell)
    if with_zone_prefix:
        easting += zone * 1000000.0
    return northing, easting, zone


def gauss_inverse(x_north, y_east, width=3, zone=None, ell=CGCS2000):
    """高斯-克吕格投影反算

    x_north 为北坐标；y_east 为东坐标（含带号前缀时 zone 可省）。
    3° 带中央子午线 = 3×带号；6° 带中央子午线 = 6×带号 − 3。
    """
    if zone is None:
        zone = int(y_east // 1000000.0)
        if zone <= 0:
            raise ValueError('东坐标没有带号前缀时必须显式给 zone')
    lon0 = zone * width if width == 3 else zone * width - 3.0
    easting = y_east - zone * 1000000.0
    return tm_inverse(easting, x_north, lon0, k0=1.0,
                      false_easting=500000.0, ell=ell)


def utm_forward(lat, lon, ell=WGS84):
    """UTM 投影（k0=0.9996，南半球加 10000 km 假北）"""
    zone = int((lon + 180.0) / 6.0) + 1
    lon0 = zone * 6.0 - 3.0
    fn = 0.0 if lat >= 0 else 10000000.0
    return tm_forward(lat, lon, lon0, k0=0.9996, false_easting=500000.0,
                      false_northing=fn, ell=ell)


def utm_inverse(easting, northing, zone, south=False, ell=WGS84):
    lon0 = zone * 6.0 - 3.0
    fn = 10000000.0 if south else 0.0
    return tm_inverse(easting, northing, lon0, k0=0.9996,
                      false_easting=500000.0, false_northing=fn, ell=ell)


# ==================== 四参数 / 七参数 ====================


def similarity4(x, y, dx, dy, rot_sec=0.0, scale_ppm=0.0):
    """平面四参数（相似变换）

    x,y 为源平面坐标；rot_sec 旋转（**角秒**）；scale_ppm 尺度差（**ppm**）。
    """
    th = rot_sec / RHO_SEC
    k = 1.0 + scale_ppm * 1e-6
    ct, st = math.cos(th), math.sin(th)
    return (dx + k * (ct * x - st * y),
            dy + k * (st * x + ct * y))


def similarity4_inverse(x, y, dx, dy, rot_sec=0.0, scale_ppm=0.0):
    """平面四参数逆变换"""
    th = rot_sec / RHO_SEC
    k = 1.0 + scale_ppm * 1e-6
    ct, st = math.cos(th), math.sin(th)
    xs = (x - dx) / k
    ys = (y - dy) / k
    return (ct * xs + st * ys,
            -st * xs + ct * ys)


def _mat3_inv(m):
    """3x3 矩阵求逆（伴随矩阵法）。奇异时抛 ValueError。"""
    (a, b, c), (d, e, f), (g, h, i) = m
    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    det = a * A + b * B + c * C
    if abs(det) < 1e-30:
        raise ValueError('矩阵奇异，无法求逆')
    return ((A / det, (c * h - b * i) / det, (b * f - c * e) / det),
            (B / det, (a * i - c * g) / det, (c * d - a * f) / det),
            (C / det, (b * g - a * h) / det, (a * e - b * d) / det))


def _mat3_mul(m, v):
    return tuple(sum(m[r][k] * v[k] for k in range(3)) for r in range(3))


def helmert7(x, y, z, dx=0.0, dy=0.0, dz=0.0,
             rx_sec=0.0, ry_sec=0.0, rz_sec=0.0, scale_ppm=0.0):
    """七参数空间相似变换（Bursa-Wolf，位置矢量约定）

    参数口径与国内常用说明一致：
      dx/dy/dz   平移（米）
      rx/ry/rz   旋转（**角秒**）
      scale_ppm  尺度差（**ppm**）

    约定::

        [X']   [dX]          [  1   -rz   ry ] [X]
        [Y'] = [dY] + (1+s)·[ rz     1  -rx ] [Y]
        [Z']   [dZ]          [-ry    rx    1 ] [Z]

    这是**线性化**形式（小角度近似），也是工程上通用的 Bursa-Wolf 模型；
    旋转角在几角秒量级时精度绰绰有余。注意另一套"坐标框架旋转"约定
    三个旋转符号相反，公式不变 —— 本项目统一用位置矢量约定。
    """
    s = 1.0 + scale_ppm * 1e-6
    rx = rx_sec / RHO_SEC
    ry = ry_sec / RHO_SEC
    rz = rz_sec / RHO_SEC
    x2 = dx + s * (x - rz * y + ry * z)
    y2 = dy + s * (rz * x + y - rx * z)
    z2 = dz + s * (-ry * x + rx * y + z)
    return x2, y2, z2


def helmert7_inverse(x, y, z, dx=0.0, dy=0.0, dz=0.0,
                     rx_sec=0.0, ry_sec=0.0, rz_sec=0.0, scale_ppm=0.0):
    """七参数逆变换：helmert7 的**精确代数逆**

    不能简单地"把参数取负再套一次正向公式" —— 那样残差量级是
    θ²·r（θ 为旋转角，r 为地心距），实测几个角秒就能带来 0.1 mm 级误差。
    这里对线性化的 3x3 旋转矩阵做真正的矩阵求逆，
    保证 helmert7 与 helmert7_inverse 互为逆变换（闭合到浮点精度）。
    """
    s = 1.0 + scale_ppm * 1e-6
    rx = rx_sec / RHO_SEC
    ry = ry_sec / RHO_SEC
    rz = rz_sec / RHO_SEC
    r = ((1.0, -rz, ry), (rz, 1.0, -rx), (-ry, rx, 1.0))
    ri = _mat3_inv(r)
    p = _mat3_mul(ri, (x - dx, y - dy, z - dz))
    return tuple(v / s for v in p)


# ==================== 大地水准面 ====================


class GeoidGrid:
    """高程异常格网：大地水准面高 N（米）

    支持两种常见格式：
      * GeographicLib ``.pgm``（P5 二进制，16 bit **大端**，像元 0 = 缺测）
        换算 ``N = Offset + Scale × pixel``，Offset/Scale 来自 ``#`` 注释
      * NOAA ``.gtx``（表头 4 个 little-endian 值 + float32 数据）
    """

    def __init__(self, lat_min, lat_max, lon_min, lon_max, nlat, nlon,
                 values, name=''):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.nlat = nlat
        self.nlon = nlon
        self.values = values          # 行自北向南、列自西向东；None = 缺测
        self.name = name

    def __repr__(self):
        return (f'<GeoidGrid {self.name} {self.nlat}x{self.nlon} '
                f'lat {self.lat_min}~{self.lat_max} '
                f'lon {self.lon_min}~{self.lon_max}>')

    # ---------- 读取 ----------

    @classmethod
    def load_pgm(cls, path):
        """读 GeographicLib PGM 大地水准面（load_pgm_raw 的别名）"""
        return cls.load_pgm_raw(path)

    @classmethod
    def load_pgm_at(cls, path, lat_max=None, lat_min=None, lon_min=None):
        """读 PGM 并显式覆盖纬度/经度范围

        GeographicLib 的 PGM 默认按全球等间距解读（行数覆盖 -90~+90），
        但区域模型（某省/某国）的行数与纬度跨度未必吻合，此时用本方法
        显式给范围。**范围给错会让高程异常整体平移**，
        换模型后应拿已知点复核一次。
        """
        g = cls.load_pgm_raw(path)
        if lat_max is not None:
            g.lat_max = lat_max
        if lat_min is not None:
            g.lat_min = lat_min
        if lon_min is not None:
            g.lon_min = lon_min
        return g

    @classmethod
    def load_pgm_raw(cls, path):
        with open(path, 'rb') as f:
            data = f.read()
        tokens, comments, i = [], [], 0
        while len(tokens) < 4:
            while i < len(data) and data[i:i + 1].isspace():
                i += 1
            if data[i:i + 1] == b'#':
                j = data.find(b'\n', i)
                comments.append(data[i + 1:j].decode('latin-1'))
                i = j + 1
                continue
            j = i
            while j < len(data) and not data[j:j + 1].isspace():
                j += 1
            tokens.append(data[i:j])
            i = j
        if tokens[0] != b'P5':
            raise ValueError(f'不是二进制 PGM: {tokens[0]!r}')
        ncol, nrow, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
        i += 1
        offset, scale = 0.0, 1.0
        origin_lat, origin_lon = None, None
        for c in comments:
            p = c.split()
            if len(p) >= 2 and p[0].lower() == 'offset':
                offset = float(p[1])
            elif len(p) >= 2 and p[0].lower() == 'scale':
                scale = float(p[1])
            elif len(p) >= 3 and p[0].lower() == 'origin':
                s = p[1].upper().replace('N', '').replace('S', '-')
                origin_lat = float(s) * (-1 if 'S' in p[1].upper() else 1)
                s = p[2].upper().replace('E', '').replace('W', '-')
                origin_lon = float(s)
        raw = data[i:i + nrow * ncol * 2]
        px = struct.unpack(f'>{nrow * ncol}H', raw[:nrow * ncol * 2])
        values = []
        for r in range(nrow):
            row = [None if px[r * ncol + c] == 0
                   else offset + scale * px[r * ncol + c]
                   for c in range(ncol)]
            values.append(row)
        lat0 = origin_lat if origin_lat is not None else 90.0
        lon0 = origin_lon if origin_lon is not None else 0.0
        # 等经纬间距：GeographicLib 用 dlat = dlon = 360/列数，
        # 行数覆盖 origin 纬度以南
        dlon = 360.0 / ncol
        dlat = 180.0 / (nrow - 1)
        return cls(lat_min=lat0 - dlat * (nrow - 1), lat_max=lat0,
                   lon_min=lon0, lon_max=lon0 + dlon * ncol,
                   nlat=nrow, nlon=ncol, values=values, name=path)

    @classmethod
    def load_gtx(cls, path):
        """读 NOAA .gtx 大地水准面"""
        with open(path, 'rb') as f:
            head = f.read(40)
            lat0, lon0, dlat, dlon = struct.unpack('<4d', head[:32])
            nrow, ncol = struct.unpack('<2i', head[32:40])
            data = f.read(nrow * ncol * 4)
        vals = struct.unpack(f'<{nrow * ncol}f', data[:nrow * ncol * 4])
        # gtx 行自南向北
        values = []
        for r in range(nrow - 1, -1, -1):
            values.append([None if vals[r * ncol + c] > 1e30
                           else float(vals[r * ncol + c])
                           for c in range(ncol)])
        return cls(lat_min=lat0, lat_max=lat0 + dlat * (nrow - 1),
                   lon_min=lon0, lon_max=lon0 + dlon * (ncol - 1),
                   nlat=nrow, nlon=ncol, values=values, name=path)

    @classmethod
    def load(cls, path):
        """按扩展名自动选择读取器"""
        low = path.lower()
        if low.endswith('.pgm'):
            return cls.load_pgm_raw(path)
        if low.endswith('.gtx'):
            return cls.load_gtx(path)
        raise ValueError(f'不认识的大地水准面格式: {path}')

    # ---------- 插值 ----------

    def height(self, lat, lon):
        """双线性插值求高程异常 N（米）；超出范围或缺测返回 None"""
        lat = min(max(lat, self.lat_min), self.lat_max)
        lon = _norm_lon(lon)
        if lon < self.lon_min:
            lon += 360.0
        if not (self.lon_min <= lon <= self.lon_max):
            return None
        dlat = (self.lat_max - self.lat_min) / max(self.nlat - 1, 1)
        dlon = (self.lon_max - self.lon_min) / max(self.nlon - 1, 1)
        fr = (self.lat_max - lat) / dlat            # 行自北向南
        fc = (lon - self.lon_min) / dlon
        r0 = int(math.floor(fr))
        c0 = int(math.floor(fc))
        r0 = max(0, min(self.nlat - 2, r0))
        c0 = max(0, min(self.nlon - 2, c0))
        tr = fr - r0
        tc = fc - c0
        v00 = self.values[r0][c0]
        v01 = self.values[r0][c0 + 1]
        v10 = self.values[r0 + 1][c0]
        v11 = self.values[r0 + 1][c0 + 1]
        if None in (v00, v01, v10, v11):
            # 有缺测就退回最近的有效像元，而不是返回 None 让调用方白跑
            for cand in (v00, v01, v10, v11):
                if cand is not None:
                    return cand
            return None
        return (v00 * (1 - tr) * (1 - tc) + v01 * (1 - tr) * tc
                + v10 * tr * (1 - tc) + v11 * tr * tc)

    def orthometric(self, lat, lon, h_ell):
        """椭球高 -> 正高：H = h - N"""
        n = self.height(lat, lon)
        if n is None or h_ell is None:
            return None
        return h_ell - n


# ==================== 便捷入口 ====================


def make_local_frame(ref_lat, ref_lon, ref_h=0.0, ell=WGS84):
    """返回一个把 BLH 转成以参考点为原点的 ENU 的可调用对象"""
    x0, y0, z0 = blh_to_ecef(ref_lat, ref_lon, ref_h, ell)

    def to_enu(lat, lon, h=0.0):
        x, y, z = blh_to_ecef(lat, lon, h, ell)
        return ecef_delta_to_enu(x - x0, y - y0, z - z0, ref_lat, ref_lon)

    return to_enu
