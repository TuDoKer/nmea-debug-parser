# -*- coding: utf-8 -*-
"""RINEX 观测文件 / 广播星历解析（纯逻辑，无 GUI 依赖）

支持:
  * RINEX 2.10/2.11 观测文件（历元行携带卫星列表，8 观测码分两行写入）
  * RINEX 3.x 观测文件（MIXED 多系统，SYS / # / OBS TYPES 逐系统观测码表）
  * RINEX 3.x 广播星历（BRDC）—— GPS / Galileo / 北斗 / QZSS 开普勒轨道

并提供:
  * ECEF <-> 大地坐标(BLH) 与 ENU 转换
  * 卫星位置(ECEF) 与 测站仰角/方位角计算

注: GLONASS 采用 PZ-90 数值积分星历，格式与开普勒星历不同，本模块不解析，
    相关卫星会在仰角/方位角计算中自动跳过（不影响其他系统）。
"""

import datetime
import math

from . import geodesy
from .geodesy import az_el as geodesy_az_el
from .geodesy import ecef_to_blh as geodesy_ecef_to_blh

# ==================== 物理 / 常量 ====================

C_LIGHT = 299792458.0                      # 光速 m/s
F_L1 = 1575.42e6                           # GPS L1 频率
F_L2 = 1227.60e6                           # GPS L2 频率
F_B1I = 1561.098e6                         # BDS B1I 频率
F_B2I = 1207.14e6                          # BDS B2I 频率（与 E5b 同频）
F_B3I = 1268.52e6                          # BDS B3I 频率
F_E1 = 1575.42e6                           # Galileo E1 频率
F_E5A = 1176.45e6                          # Galileo E5a 频率
F_E5B = 1207.14e6                          # Galileo E5b 频率
# 无电离层组合系数（用于多路径 MP 计算）
_AL = F_L1 ** 2 / (F_L1 ** 2 - F_L2 ** 2)
_BE = -F_L2 ** 2 / (F_L1 ** 2 - F_L2 ** 2)

GM = 3.986005e14                           # 地球引力常数 m^3/s^2
OMEGA_E = 7.2921151467e-5                  # 地球自转角速度 rad/s
GPS_EPOCH = 315964800.0                    # 1980-01-06 00:00:00 UTC 的 Unix 时间戳

# 椭球参数已统一到 geodesy（WGS84 / CGCS2000），本模块不再重复定义 ——
# 原先这里另有一份 WGS84_A/F/E2，与 geodesy 各占一套，改一处忘一处。
# 需要椭球请 `from . import geodesy` 后用 geodesy.WGS84。

# 系统标识符 -> 中文星座名
SYS_NAMES = {
    'G': 'GPS',
    'R': 'GLONASS',
    'E': 'Galileo',
    'C': '北斗',
    'J': 'QZSS',
    'I': 'IRNSS',
    'S': 'SBAS',
}


def _to_float(seg):
    """RINEX 数值转 float，兼容 Fortran 的 D 指数写法

    很多接收机导出的星历用 'D' 而非 'E' 表示指数
    （如 '1.90000000000D+01'），Python 的 float() 解析不了。
    """
    if seg is None:
        return None
    s = seg.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        try:
            return float(s.replace('D', 'E').replace('d', 'e'))
        except ValueError:
            return None


def sys_name(satid):
    """'G05' -> 'GPS'；未知标识原样返回"""
    if not satid:
        return '未知'
    return SYS_NAMES.get(satid[0], satid[0])


def sat_label(satid):
    """'G05' -> 'GPS-05'，便于在图表中区分星座"""
    if not satid or len(satid) < 2:
        return satid
    return f"{sys_name(satid)}-{satid[1:]}"


# ==================== RINEX 3 观测码 -> 频点归类 ====================


def band_of(obs_code):
    """把 RINEX 3 观测码映射到归纳频点，便于跨系统比较。

    返回 'L1' / 'L2' / 'L3' / 'L5' / 'L6' / 'L7' / 'L8' / None
    """
    if not obs_code or len(obs_code) < 2:
        return None
    band = obs_code[1]
    return {
        '1': 'L1', '2': 'L2', '3': 'L3', '5': 'L5',
        '6': 'L6', '7': 'L7', '8': 'L8', '9': 'L9',
    }.get(band)


def is_phase(obs_code):
    return obs_code[:1] == 'L'


def is_pseudorange(obs_code):
    return obs_code[:1] in ('C', 'P')


def is_snr(obs_code):
    return obs_code[:1] == 'S'


def is_doppler(obs_code):
    # RINEX 3 用 D，RINEX 2.10 老式写法也用 D
    return obs_code[:1] == 'D'


# ==================== 观测文件解析 ====================


class RinexObs:
    """RINEX 观测文件解析器（RINEX 2.10 与 3.x 自适应）

    用法::

        rnx = RinexObs(path)
        rnx.parse(progress_callback=cb)      # 可选回调 (current, total)
        rnx.epochs        # [(secs, {'G05': {...obs...}, ...}), ...]
        rnx.obstypes      # {'G': ['C1C','L1C',...], 'R': [...], ...}
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.version = 0.0
        self.rinex_type = ''
        self.marker = ''
        self.approx_xyz = None
        self.antenna_delta = (0.0, 0.0, 0.0)
        self.interval = None
        self.t_first = None            # (y, m, d, H, M, S)
        self.t_last = None
        self.obstypes = {}             # {系统标识: [观测码, ...]}；RINEX 2 用 ''
        self.epochs = []               # [(secs_of_day, {satid: {code: value}})]
        self.header_lines = 0
        self.total_lines = 0
        # WAVELENGTH FACT L1/2 声明的 L1/L2 波长因子，如 (1, 1) 表示
        # L2 列并非真正的第二频点（单频接收机常这样导出）。
        self.wavelength_fact = None

    # ---------- 头文件 ----------

    def _parse_header(self, f):
        """解析头文件，返回时文件指针停在头文件之后"""
        n = 0
        for line in f:
            n += 1
            label = line[60:].strip() if len(line) > 60 else ''

            if n == 1:
                # 版本号占前 9 列，右对齐（如 '     2.10'）；直接 float() 会因
                # 尾部空格报 ValueError，必须先剥掉空白。
                seg = line[:9].strip()
                if _is_float(seg):
                    self.version = float(seg)
                self.rinex_type = line[20:21].strip()
            if label == 'MARKER NAME':
                self.marker = line[:60].strip()
            elif label == 'APPROX POSITION XYZ':
                # 3 个 F14.4 定宽字段
                vals = self._fixed_floats(line, 3, 14, 0)
                if len(vals) == 3:
                    self.approx_xyz = tuple(vals)
            elif label == 'ANTENNA: DELTA H/E/N':
                vals = self._fixed_floats(line, 3, 14, 0)
                if len(vals) == 3:
                    self.antenna_delta = tuple(vals)
            elif label == 'INTERVAL':
                vals = self._fixed_floats(line, 1, 10, 0)
                if vals:
                    self.interval = vals[0]
            elif label == 'WAVELENGTH FACT L1/2':
                # 前两个字段是 L1、L2 的波长因子（整数）
                parts = line[:60].split()
                if len(parts) >= 2:
                    try:
                        self.wavelength_fact = (int(parts[0]), int(parts[1]))
                    except ValueError:
                        pass
            elif label in ('TIME OF FIRST OBS', 'TIME OF LAST OBS'):
                t = self._parse_time_line(line)
                if t is not None:
                    if label == 'TIME OF FIRST OBS':
                        self.t_first = t
                    else:
                        self.t_last = t

            # ---- RINEX 2: # / TYPES OF OBSERV ----
            elif label == '# / TYPES OF OBSERV':
                try:
                    cnt = int(line[:6])
                except ValueError:
                    cnt = 0
                codes = line[6:60].split()
                while len(codes) < cnt:
                    more = next(f)
                    n += 1
                    codes += more[6:60].split()
                self.obstypes[''] = self._normalize_codes(codes)

            # ---- RINEX 3: SYS / # / OBS TYPES ----
            elif label == 'SYS / # / OBS TYPES':
                sysc = line[0:1]
                try:
                    cnt = int(line[3:6])
                except ValueError:
                    cnt = 0
                codes = line[7:60].split()
                nblocks = (cnt + 12) // 13
                while len(codes) < cnt and nblocks > 1:
                    more = next(f)
                    n += 1
                    codes += more[7:60].split()
                    nblocks -= 1
                self.obstypes[sysc] = self._normalize_codes(codes)

            if label == 'END OF HEADER':
                break
        self.header_lines = n

    @staticmethod
    def _fixed_floats(line, count, width, offset=0):
        """按固定列宽解析浮点数（RINEX 头文件字段都是定宽格式）

        空字段会中断解析——这是有意的：RINEX 中数值字段是连续的，
        一旦出现空白说明后续字段不存在。
        """
        out = []
        for i in range(count):
            seg = line[offset + i * width:offset + (i + 1) * width].strip()
            if not _is_float(seg):
                break
            out.append(float(seg))
        return out

    @staticmethod
    def _normalize_codes(codes):
        """RINEX 2 观测码补全为 3 字符（C1 -> C1C 之类保持为 C1）"""
        out = []
        for c in codes:
            c = c.strip()
            if c:
                out.append(c)
        return out

    @staticmethod
    def _parse_time_line(line):
        parts = line[:43].split()
        if len(parts) < 6:
            return None
        try:
            return (int(parts[0]), int(parts[1]), int(parts[2]),
                    int(parts[3]), int(parts[4]), float(parts[5]))
        except ValueError:
            return None

    # ---------- 历元 ----------

    def _is_r3_epoch(self, line):
        """RINEX 3 历元行: > 2026 9 2 13 4 47.0000000  0 28"""
        if not line.startswith('>'):
            return False
        try:
            int(line[2:6])
            int(line[7:9])
            int(line[10:12])
            int(line[13:15])
            int(line[16:18])
        except (ValueError, IndexError):
            return False
        return True

    @staticmethod
    def _is_r2_epoch(line):
        """RINEX 2 历元行: ' 26  9  2 13  4 47.0000000  0 28G05G07...'"""
        if len(line) < 32:
            return False
        try:
            int(line[1:3])
            int(line[4:6])
            int(line[7:9])
            int(line[10:12])
            int(line[13:15])
            float(line[16:26])
            int(line[29:32])
        except (ValueError, IndexError):
            return False
        # 卫星列表段必须含有字母，否则可能是数据行被误判
        return any(ch.isalpha() for ch in line[32:68])

    # ---------- 主解析 ----------

    def parse(self, progress_callback=None):
        """解析整个观测文件

        progress_callback: 可选回调 (current_lines, total_lines)
        """
        with open(self.filepath, 'r', encoding='ascii', errors='replace') as f:
            self._parse_header(f)

            is_r3 = self.version >= 3.0
            ep_hdr = self._is_r3_epoch if is_r3 else self._is_r2_epoch

            if is_r3:
                self._parse_epochs_r3(f, ep_hdr, progress_callback)
            else:
                self._parse_epochs_r2(f, ep_hdr, progress_callback)

        return self

    # ---- RINEX 3 历元解析 ----

    def _parse_epochs_r3(self, f, ep_hdr, progress_callback):
        cur = None          # {satid: {code: value}}
        cur_secs = None
        version = self.version

        for line in f:
            self.total_lines += 1

            if line.startswith('>'):
                # 上一历元收尾
                if cur_secs is not None:
                    self.epochs.append((cur_secs, cur))

                try:
                    hh = int(line[13:15])
                    mi = int(line[16:18])
                    ss = float(line[19:30])
                except (ValueError, IndexError):
                    cur, cur_secs = {}, None
                    continue
                cur_secs = hh * 3600 + mi * 60 + ss
                cur = {}

                if progress_callback:
                    progress_callback(self.total_lines, 0)
                continue

            if cur is None or len(line) < 3:
                continue

            satid = line[0:3].strip()
            if len(satid) < 2 or not satid[0].isalpha():
                continue
            codes = self.obstypes.get(satid[0])
            if not codes:
                continue

            vals = {}
            for idx, code in enumerate(codes):
                col = 3 + idx * 16
                seg = line[col:col + 16].strip()
                if not seg:
                    continue
                try:
                    vals[code] = float(seg)
                except ValueError:
                    pass
            if vals:
                cur[satid] = vals

        if cur_secs is not None:
            self.epochs.append((cur_secs, cur))

    # ---- RINEX 2.10 历元解析 ----

    def _parse_epochs_r2(self, f, ep_hdr, progress_callback):
        codes = self.obstypes.get('', [])
        ncode = len(codes)
        # 每行 5 个 16 字符字段
        lines_per_sat = (ncode + 4) // 5

        cur = {}
        cur_secs = None
        pending_sats = []
        pend_idx = 0
        field_buf = []

        for line in f:
            self.total_lines += 1

            if ep_hdr(line):
                if cur_secs is not None:
                    self.epochs.append((cur_secs, cur))

                try:
                    hh = int(line[10:12])
                    mi = int(line[13:15])
                    ss = float(line[16:26])
                    nsat = int(line[29:32])
                except (ValueError, IndexError):
                    cur, cur_secs = {}, None
                    continue

                cur_secs = hh * 3600 + mi * 60 + ss
                cur = {}

                sats = [line[32 + j * 3:35 + j * 3].strip() for j in range(12)]
                sats = [s for s in sats if len(s) >= 2 and s[0].isalpha()]
                need = nsat - len(sats)
                # 卫星列表跨行（每行最多 12 颗）
                while need > 0:
                    cont = next(f, None)
                    if cont is None:
                        break
                    self.total_lines += 1
                    more = [cont[32 + j * 3:35 + j * 3].strip() for j in range(12)]
                    more = [s for s in more if len(s) >= 2 and s[0].isalpha()]
                    sats += more
                    need -= 12

                pending_sats = sats[:nsat]
                pend_idx = 0
                field_buf = []

                if progress_callback:
                    progress_callback(self.total_lines, 0)
                continue

            if cur_secs is None or pend_idx >= len(pending_sats):
                continue

            # 收集这一行的 5 个字段
            row = [line[j * 16:(j + 1) * 16].strip() for j in range(5)]
            if not field_buf:
                field_buf = row
            else:
                field_buf += row

            if len(field_buf) < lines_per_sat * 5:
                # 该卫星仍未攒满
                continue

            satid = pending_sats[pend_idx]
            vals = {}
            for ci, code in enumerate(codes):
                if ci >= len(field_buf):
                    break
                seg = field_buf[ci]
                if not seg:
                    continue
                try:
                    vals[code] = float(seg)
                except ValueError:
                    pass
            if vals:
                cur[satid] = vals

            pend_idx += 1
            field_buf = []

        if cur_secs is not None:
            self.epochs.append((cur_secs, cur))

    # ---------- 便捷访问 ----------

    def obs_codes(self, sysc):
        """取某系统可用观测码列表"""
        return self.obstypes.get(sysc, []) or self.obstypes.get('', [])

    def pick(self, sysc, prefixes):
        """按优先顺序挑观测码：prefixes 例如 ('S1',) -> 'S1C' / 'S1'"""
        codes = self.obs_codes(sysc)
        for pref in prefixes:
            for code in codes:
                if code.startswith(pref):
                    return code
        return None

    def time_span(self):
        """返回 (t_first, t_last) —— 若头文件没写则用历元推算"""
        t0, t1 = self.t_first, self.t_last
        if t0 is None and self.epochs:
            t0 = self._secs_to_hms(self.epochs[0][0])
        if t1 is None and self.epochs:
            t1 = self._secs_to_hms(self.epochs[-1][0])
        return t0, t1

    @staticmethod
    def _secs_to_hms(secs):
        secs = int(round(secs))
        return (0, 0, 0, secs // 3600, (secs // 60) % 60, float(secs % 60))


def _is_float(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


# ==================== 广播星历解析 ====================


class BroadcastEphemeris:
    """RINEX 3 广播星历（BRDC）解析器

    只解析开普勒类星历（GPS / Galileo / 北斗 / QZSS）。
    GLONASS 使用完全不同的 PZ-90 状态向量格式，本类自动跳过。
    """

    # 允许解析的系统（跳过 R = GLONASS）
    KEPLERIAN = ('G', 'E', 'C', 'J')
    KEPLERIAN2 = ('G', 'E', 'C', 'J', 'S')   # RINEX 2：首字符是 PRN

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.satellite_count = 0
        self.epochs = {}       # {satid: [eph_dict, ...]}
        self.version = 3.0
        self.format = 'rinex3'

    @staticmethod
    def _fields(line, base=4, count=4, width=19):
        """从定宽星历行中取 count 个浮点数，缺失置 None"""
        return [_to_float(line[base + j * width:base + (j + 1) * width])
                for j in range(count)]

    @staticmethod
    def _fields2(line, count=4, width=19, base=3):
        """RINEX 2 NAV 定宽取值：列 3 起，每 19 列一个字段"""
        return [_to_float(line[base + j * width:base + (j + 1) * width])
                for j in range(count)]

    def parse(self, progress_callback=None):
        if not self.filepath:
            return self

        with open(self.filepath, 'r', encoding='ascii', errors='replace') as f:
            lines = f.readlines()

        i = 0
        # 跳过头文件，顺带读版本号（前 9 列右对齐，如 '     2.10'）
        while i < len(lines):
            if lines[i][60:].strip() == 'END OF HEADER':
                break
            if i == 0:
                seg = lines[i][:9].strip()
                try:
                    self.version = float(seg)
                except ValueError:
                    self.version = 3.0
            i += 1
        i += 1

        # RINEX 2 星历：历元行首字符是 PRN 数字（如 '06 26  9  8 12  0  0.0'），
        # 与 RINEX 3 的 'G06 2026 09 08 ...' 完全不同，必须分别处理。
        self.format = 'rinex2' if self.version < 3.0 else 'rinex3'
        if self.format == 'rinex2':
            return self._parse_nav2(lines, i, progress_callback)

        while i < len(lines):
            line = lines[i]
            if len(line) < 3 or not line.strip():
                i += 1
                continue

            sysc = line[0]
            if sysc not in self.KEPLERIAN:
                # GLONASS(R)/其他：跳过（R 为 4 行一组）
                i += 4 if sysc == 'R' else 1
                continue

            parts = line.split()
            try:
                prn = int(parts[0][1:])
                yr = int(parts[1])
                mo = int(parts[2])
                dy = int(parts[3])
                hh = int(parts[4])
                mi = int(parts[5])
                ss = float(parts[6])
                # RINEX 3 历元行第 8~10 字段是卫星钟差系数 af0/af1/af2，SPP 需要
                af0 = _to_float(parts[7]) if len(parts) > 7 else None
                af1 = _to_float(parts[8]) if len(parts) > 8 else None
                af2 = _to_float(parts[9]) if len(parts) > 9 else None
            except (ValueError, IndexError):
                i += 1
                continue

            yr += 2000 if yr < 100 else 0
            try:
                toc = datetime.datetime(
                    yr, mo, dy, hh, mi, int(ss),
                    tzinfo=datetime.timezone.utc).timestamp()
            except ValueError:
                i += 1
                continue

            # 需要后续 7 行
            if i + 7 >= len(lines):
                break

            # 第 0 行第 2~4 字段（Crs 等）：注意非标准 BRDC 首行含 4 个字段
            l1 = self._fields(lines[i + 1])
            l2 = self._fields(lines[i + 2])
            l3 = self._fields(lines[i + 3])
            l4 = self._fields(lines[i + 4])
            l5 = self._fields(lines[i + 5])

            satid = f"{sysc}{prn:02d}"
            eph = {
                'sys': sysc, 'prn': prn, 'toc': toc,
                'af0': af0, 'af1': af1, 'af2': af2,
                'Crs': l1[0], 'deltaN': l1[2], 'M0': l1[3],
                'Cuc': l2[0], 'e': l2[1], 'Cus': l2[2], 'sqrtA': l2[3],
                'Toe': l3[0], 'Cic': l3[1], 'OMEGA0': l3[2], 'Cis': l3[3],
                'i0': l4[0], 'Crc': l4[1], 'omega': l4[2], 'OMEGAdot': l4[3],
                'IDOT': l5[0], 'GPSWeek': l5[2],
            }
            # 必需字段齐全才纳入
            if all(eph.get(k) is not None for k in
                   ('sqrtA', 'e', 'M0', 'Toe', 'i0', 'OMEGA0', 'omega')):
                self.epochs.setdefault(satid, []).append(eph)
                self.satellite_count += 1

            i += 8

        self._resolve_toe_absolute()

        if progress_callback:
            progress_callback(len(lines), len(lines))
        return self

    # RINEX 2 各系统的默认星座前缀（首字符是 PRN 时无从判断系统）
    _R2_SYS_BY_PRN = ((1, 32, 'G'), (33, 64, 'S'), (65, 96, 'J'),
                      (97, 120, 'C'), (121, 140, 'E'))

    @classmethod
    def _r2_satid(cls, prn):
        """RINEX 2 星历只有 PRN，按 PRN 号段反推星座"""
        for lo, hi, s in cls._R2_SYS_BY_PRN:
            if lo <= prn <= hi:
                return f"{s}{prn:02d}"
        return f"G{prn:02d}"

    def _parse_nav2(self, lines, i, progress_callback=None):
        """解析 RINEX 2 广播星历（GPS/QZSS/Galileo/BDS 开普勒型，跳过 GLONASS）

        典型来源：接收机随观测文件一起导出的 <站号>.<yy>N（如 STATION.26N）。
        """
        n = len(lines)
        while i < n:
            line = lines[i]
            if len(line) < 22 or not line[:2].strip():
                i += 1
                continue

            parts = line.split()
            if len(parts) < 8:
                i += 1
                continue
            try:
                prn = int(parts[0])
                yr = int(parts[1])
                mo = int(parts[2])
                dy = int(parts[3])
                hh = int(parts[4])
                mi = int(parts[5])
                ss = float(parts[6])
                # RINEX 2 历元行第 8~10 字段同样是卫星钟差 af0/af1/af2
                af0 = _to_float(parts[7]) if len(parts) > 7 else None
                af1 = _to_float(parts[8]) if len(parts) > 8 else None
                af2 = _to_float(parts[9]) if len(parts) > 9 else None
            except ValueError:
                i += 1
                continue

            # RINEX 2 里 GPS 用的是「周数」而在混合文件里是年；两值都不在
            # 合理年段时按 2 位年份处理。
            if not (1990 <= (yr + 2000 if yr < 100 else yr) <= 2100):
                i += 1
                continue
            yr = yr + 2000 if yr < 100 else yr

            try:
                toc = datetime.datetime(
                    yr, mo, dy, hh, mi, int(ss),
                    tzinfo=datetime.timezone.utc).timestamp()
            except ValueError:
                i += 1
                continue

            if i + 7 >= n:
                break

            l1 = self._fields2(lines[i + 1])
            l2 = self._fields2(lines[i + 2])
            l3 = self._fields2(lines[i + 3])
            l4 = self._fields2(lines[i + 4])
            l5 = self._fields2(lines[i + 5])

            satid = self._r2_satid(prn)
            eph = {
                'sys': satid[0], 'prn': prn, 'toc': toc,
                'af0': af0, 'af1': af1, 'af2': af2,
                'Crs': l1[0], 'deltaN': l1[2], 'M0': l1[3],
                'Cuc': l2[0], 'e': l2[1], 'Cus': l2[2], 'sqrtA': l2[3],
                'Toe': l3[0], 'Cic': l3[1], 'OMEGA0': l3[2], 'Cis': l3[3],
                'i0': l4[0], 'Crc': l4[1], 'omega': l4[2], 'OMEGAdot': l4[3],
                'IDOT': l5[0], 'GPSWeek': l5[2],
            }
            if all(eph.get(k) is not None for k in
                   ('sqrtA', 'e', 'M0', 'Toe', 'i0', 'OMEGA0', 'omega')):
                self.epochs.setdefault(satid, []).append(eph)
                self.satellite_count += 1

            i += 8

        self._resolve_toe_absolute()

        if progress_callback:
            progress_callback(n, n)
        return self

    def _resolve_toe_absolute(self):
        """把 Toe(周内秒) 折算成绝对 Unix 时间戳，便于按时间挑最新星历"""
        for lst in self.epochs.values():
            for d in lst:
                wk = d.get('GPSWeek') or 0
                toe = d.get('Toe') or 0
                if wk and wk > 100:
                    t = GPS_EPOCH + wk * 604800.0 + toe
                    # 与 toc 对齐到同一周
                    while t - d['toc'] > 302400:
                        t -= 604800
                    while d['toc'] - t > 302400:
                        t += 604800
                    d['toe_abs'] = t
                else:
                    d['toe_abs'] = d['toc']
                d['toe_abs'] = d['toe_abs'] or d['toc']

    def best(self, satid, t, max_dt=7200.0):
        """取距时刻 t 最近的星历（超过 max_dt 秒则视为不可用）"""
        lst = self.epochs.get(satid)
        if not lst:
            return None
        d = min(lst, key=lambda e: abs(e['toc'] - t))
        return d if abs(d['toc'] - t) <= max_dt else None

    def sat_ecef(self, eph, t):
        """由开普勒星历计算 t 时刻卫星 ECEF 坐标（m）"""
        A = eph['sqrtA'] ** 2
        n0 = math.sqrt(GM / A ** 3)
        tk = t - eph['toe_abs']
        if tk > 302400:
            tk -= 604800
        if tk < -302400:
            tk += 604800

        M = eph['M0'] + (n0 + eph['deltaN']) * tk
        e = eph['e']
        E = M
        for _ in range(40):
            dE = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
            E -= dE
            if abs(dE) < 1e-13:
                break

        v = math.atan2(math.sqrt(1 - e * e) * math.sin(E), math.cos(E) - e)
        phi = v + eph['omega']
        du = eph['Cus'] * math.sin(2 * phi) + eph['Cuc'] * math.cos(2 * phi)
        dr = eph['Crs'] * math.sin(2 * phi) + eph['Crc'] * math.cos(2 * phi)
        di = eph['Cis'] * math.sin(2 * phi) + eph['Cic'] * math.cos(2 * phi)

        u = phi + du
        r = A * (1 - e * math.cos(E)) + dr
        inc = eph['i0'] + di + eph['IDOT'] * tk
        xp = r * math.cos(u)
        yp = r * math.sin(u)
        Om = eph['OMEGA0'] + (eph['OMEGAdot'] - OMEGA_E) * tk - OMEGA_E * eph['Toe']

        return (xp * math.cos(Om) - yp * math.cos(inc) * math.sin(Om),
                xp * math.sin(Om) + yp * math.cos(inc) * math.cos(Om),
                yp * math.sin(inc))


# ==================== 坐标转换 ====================


def ecef_to_blh(x, y, z):
    """ECEF -> (纬度°, 经度°, 大地高 m)，WGS84

    实现委托给 `geodesy.ecef_to_blh`（Bowring 初值 + 牛顿迭代），
    本函数保留只为兼容既有调用。**不要再在这里另写一份** ——
    项目里曾经同时存在三套 ECEF/ENU 实现，改一处忘一处是必然结果。
    """
    return geodesy_ecef_to_blh(x, y, z, ell=geodesy.WGS84)


class LocalFrame:
    """以测站为原点的 ENU 局部坐标系，用于算仰角/方位角

    保留本类只是为了不改动既有调用点（rinex_statistics 里有多处使用），
    内部计算全部委托给 ``geodesy``。
    """

    def __init__(self, x, y, z):
        self.xyz = (x, y, z)
        lat, lon, _ = ecef_to_blh(x, y, z)
        self.lat, self.lon = lat, lon

    def az_el(self, sx, sy, sz):
        """返回 (方位角° 自北顺时针, 仰角°, 斜距 m)；不可用返回 None"""
        return geodesy_az_el(self.xyz, (sx, sy, sz), self.lat, self.lon)


def doy_of(year, month, day):
    """公历日期 -> 年积日 DOY"""
    return datetime.date(year, month, day).timetuple().tm_yday


def day0_timestamp(year, doy):
    """某年 DOY 当天 00:00:00 UTC 的 Unix 时间戳"""
    base = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    return base + (doy - 1) * 86400.0
