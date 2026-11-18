# -*- coding: utf-8 -*-
"""GGA 报文统计分析（纯逻辑，无GUI依赖）"""

import math

import numpy as np

from . import timebase
from .solution import Q_NAMES, SolutionSeries, analyse_rtk_state


class GGAStatistics:
    """GGA报文统计分析"""

    # 异常值判定阈值（米）—— 单一来源，改这里即可全局生效。
    # 不同检验标准要求不同，界面上可按标准覆盖这两个值。
    HORIZ_OUTLIER_M = 0.02    # 水平距离误差阈值 2cm
    VERT_OUTLIER_M = 0.03     # 高程误差阈值 3cm

    @staticmethod
    def compute_sample_errors(gga_messages):
        """为每个GGA样本计算ENU误差并标记异常（numpy向量化）。

        各返回数组与 gga_messages 等长、按索引对齐；无效样本对应位置为 NaN，
        异常标记为 False。

        返回 dict:
            'ref':          {'lat','lon','alt'} 参考点(有效样本中位数)，alt 可能为 None
            'lats':         np.ndarray[float] 纬度(°)
            'lons':         np.ndarray[float] 经度(°)
            'alts':         np.ndarray[float] 高程(m)
            'eastings':     np.ndarray[float] 东向误差(m)
            'northings':    np.ndarray[float] 北向误差(m)
            'horiz_errors': np.ndarray[float] 水平距离误差(m)
            'vert_errors':  np.ndarray[float] 高程误差(m)
            'valid':        np.ndarray[bool]  经纬度是否有效
            'outliers':     np.ndarray[bool]  异常(水平或|高程|超阈值，见 HORIZ/VERT_OUTLIER_M)
        """
        n = len(gga_messages)
        parsed_list = [m.get('parsed', {}) for m in gga_messages]
        lats = np.array([p.get('latitude') for p in parsed_list], dtype=np.float64)
        lons = np.array([p.get('longitude') for p in parsed_list], dtype=np.float64)
        alts = np.array([p.get('altitude') for p in parsed_list], dtype=np.float64)

        valid = ~np.isnan(lats) & ~np.isnan(lons)
        n_valid = int(valid.sum())

        if n_valid > 0:
            ref_lat = float(np.median(lats[valid]))
            ref_lon = float(np.median(lons[valid]))
            alt_avail = alts[valid & ~np.isnan(alts)]
            ref_alt = float(np.median(alt_avail)) if len(alt_avail) > 0 else None
        else:
            ref_lat = ref_lon = 0.0
            ref_alt = None

        # ENU：这里是**为了性能保留的向量化近似**（等距圆柱投影）。
        # 与 geodesy 里的真 ENU 数值等价 —— 实测 959 个历元最大差
        # 水平 0.00003 mm / 高程 0.000002 mm（见 tests/verify_coord_unify.py），
        # 所以在毫米级的定位质量统计里没有影响。
        # 之所以不换成 geodesy：这里是整批 GGA 的 numpy 向量化计算，
        # 换成逐点调用会把 O(1) 的数组运算变成 Python 循环。
        # **注意别在外面再写第三份实现** —— 需要单点换算请用 geodesy。
        lat_rad = np.radians(ref_lat)
        mpl = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
        mplon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)
        northings = (lats - ref_lat) * mpl
        eastings = (lons - ref_lon) * mplon
        horiz_errors = np.sqrt(eastings ** 2 + northings ** 2)
        vert_errors = (alts - ref_alt) if ref_alt is not None else np.full(n, np.nan)

        outliers = np.zeros(n, dtype=bool)
        if n_valid > 0:
            outliers[valid] = (
                (horiz_errors[valid] > GGAStatistics.HORIZ_OUTLIER_M)
                | (np.abs(vert_errors[valid]) > GGAStatistics.VERT_OUTLIER_M)
            )

        return {
            'ref': {'lat': ref_lat, 'lon': ref_lon, 'alt': ref_alt},
            'lats': lats, 'lons': lons, 'alts': alts,
            'eastings': eastings, 'northings': northings,
            'horiz_errors': horiz_errors, 'vert_errors': vert_errors,
            'valid': valid, 'outliers': outliers,
        }

    @staticmethod
    def compute_stats(gga_messages):
        """计算GGA报文的统计量"""
        se = GGAStatistics.compute_sample_errors(gga_messages)
        valid = se['valid']
        n_valid = int(valid.sum())
        if n_valid < 2:
            return {'error': '有效GGA数据点不足（至少需要2个）'}

        lats = se['lats'][valid]
        lons = se['lons'][valid]
        alts = se['alts'][valid]
        eastings = se['eastings'][valid]
        northings = se['northings'][valid]
        horiz_errors = se['horiz_errors'][valid]
        vert_errors = se['vert_errors'][valid]
        outliers_valid = se['outliers'][valid]

        ref = se['ref']

        # 高程/垂直误差：仅统计有高程的样本
        alt_valid = ~np.isnan(alts)
        alts_ok = alts[alt_valid]
        vert_ok = vert_errors[alt_valid]

        # HDOP / 卫星数：仅统计有效定位样本（与原逻辑一致）
        hdops, num_satss = [], []
        for i, msg in enumerate(gga_messages):
            if not valid[i]:
                continue
            p = msg.get('parsed', {})
            if p.get('hdop') is not None:
                hdops.append(p['hdop'])
            if p.get('num_sats') is not None:
                num_satss.append(p['num_sats'])

        horiz_stats = {
            'count': n_valid,
            'lat_mean': float(np.mean(lats)), 'lat_median': float(np.median(lats)),
            'lat_std': float(np.std(lats, ddof=1)),
            'lat_min': float(np.min(lats)), 'lat_max': float(np.max(lats)),
            'lat_range': float(np.max(lats) - np.min(lats)),
            'lon_mean': float(np.mean(lons)), 'lon_median': float(np.median(lons)),
            'lon_std': float(np.std(lons, ddof=1)),
            'lon_min': float(np.min(lons)), 'lon_max': float(np.max(lons)),
            'lon_range': float(np.max(lons) - np.min(lons)),
            'east_mean': float(np.mean(eastings)), 'east_median': float(np.median(eastings)),
            'east_std': float(np.std(eastings, ddof=1)),
            'east_rms': float(math.sqrt(np.mean(eastings**2))),
            'north_mean': float(np.mean(northings)), 'north_median': float(np.median(northings)),
            'north_std': float(np.std(northings, ddof=1)),
            'north_rms': float(math.sqrt(np.mean(northings**2))),
            'horiz_error_mean': float(np.mean(horiz_errors)),
            'horiz_error_median': float(np.median(horiz_errors)),
            'horiz_error_std': float(np.std(horiz_errors, ddof=1)),
            'horiz_error_rms': float(math.sqrt(np.mean(horiz_errors**2))),
            'horiz_error_max': float(np.max(horiz_errors)),
            'horiz_error_min': float(np.min(horiz_errors)),
            'cep50': float(np.median(horiz_errors)),
            'cep68': float(np.percentile(horiz_errors, 68)),
            'cep95': float(np.percentile(horiz_errors, 95)),
            'cep99': float(np.percentile(horiz_errors, 99)),
            'drms': float(math.sqrt(np.mean(eastings**2) + np.mean(northings**2))),
            '2drms': float(2 * math.sqrt(np.mean(eastings**2) + np.mean(northings**2))),
        }

        vert_stats = {}
        if len(vert_ok) > 0:
            vert_stats = {
                'alt_mean': float(np.mean(alts_ok)), 'alt_median': float(np.median(alts_ok)),
                'alt_std': float(np.std(alts_ok, ddof=1)),
                'alt_min': float(np.min(alts_ok)), 'alt_max': float(np.max(alts_ok)),
                'alt_range': float(np.max(alts_ok) - np.min(alts_ok)),
                'vert_error_mean': float(np.mean(np.abs(vert_ok))),
                'vert_error_median': float(np.median(np.abs(vert_ok))),
                'vert_error_std': float(np.std(vert_ok, ddof=1)),
                'vert_error_rms': float(math.sqrt(np.mean(vert_ok**2))),
                'vert_error_max': float(np.max(np.abs(vert_ok))),
                'vert_cep68': float(np.percentile(np.abs(vert_ok), 68)),
                'vert_cep95': float(np.percentile(np.abs(vert_ok), 95)),
                'vert_cep99': float(np.percentile(np.abs(vert_ok), 99)),
            }
        else:
            vert_stats = {'error': '无有效高度数据'}

        other_stats = {}
        if hdops:
            other_stats.update({
                'hdop_mean': float(np.mean(hdops)), 'hdop_median': float(np.median(hdops)),
                'hdop_min': float(np.min(hdops)), 'hdop_max': float(np.max(hdops)),
            })
        if num_satss:
            other_stats.update({
                'sats_mean': float(np.mean(num_satss)), 'sats_median': float(np.median(num_satss)),
                'sats_min': float(np.min(num_satss)), 'sats_max': float(np.max(num_satss)),
            })

        # 异常值统计
        n_outliers = int(outliers_valid.sum())
        outlier_stats = {
            'threshold_h': GGAStatistics.HORIZ_OUTLIER_M,
            'threshold_v': GGAStatistics.VERT_OUTLIER_M,
            'n_valid': n_valid,
            'n_outliers': n_outliers,
            'rate': float(outliers_valid.mean()) if n_valid else 0.0,
        }

        return {
            'reference': ref,
            'horizontal': horiz_stats,
            'vertical': vert_stats,
            'other': other_stats,
            'outliers': outlier_stats,
            'raw': {
                'lats': lats, 'lons': lons, 'alts': alts_ok,
                'eastings': eastings, 'northings': northings,
                'horiz_errors': horiz_errors, 'vert_errors': vert_ok,
                'outliers': outliers_valid,
                'times': list(range(n_valid))
            }
        }

class SATSINFOStatistics:
    """SATSINFO 报文统计分析（多星座、多频点信噪比、天顶图）

    输出接口与 GSVStatistics.compute_stats 对齐，便于"卫星分析"对话框复用
    天空图(天顶图)/SNR直方图/星座统计/逐星SNR时序等面板。
    同时在 per_satellite 中附带逐频点 SNR(freqs)，用于展示多频点信息。
    """

    # 系统标识 → 星座显示名（与 parser.SYSTEM_NAMES 一致）
    SYSTEM_NAMES = {
        0: 'GPS', 1: 'GLONASS', 2: 'SBAS',
        3: 'Galileo', 4: '北斗', 5: 'QZSS',
    }

    @staticmethod
    def collect_satellites(satsinfo_messages):
        """收集所有卫星观测样本（每卫星/每报文一条，SNR 取该星所有频点中的最大值）
        """
        samples = []
        for msg in satsinfo_messages:
            p = msg.get('parsed', {})
            line_no = msg.get('line_no', 0)
            for sat in p.get('satellites', []):
                az = sat.get('azimuth')
                el = sat.get('elevation')
                freqs = sat.get('freqs') or []
                if az is None or el is None or not freqs:
                    continue
                # 该星系统取其首个频点的系统标识
                sys_id = freqs[0].get('sys')
                const = SATSINFOStatistics.SYSTEM_NAMES.get(sys_id, '未知')
                snr_max = max((f.get('snr') for f in freqs
                               if f.get('snr') is not None), default=None)
                samples.append({
                    'line_no': line_no,
                    'prn': str(sat.get('prn')),
                    'constellation': const,
                    'sys': sys_id,
                    'elevation': el,
                    'azimuth': az,
                    'snr': snr_max,
                    'freqs': [{'sys': f.get('sys'), 'snr': f.get('snr'),
                               'freq_status': f.get('freq_status')} for f in freqs],
                })
        return samples

    @staticmethod
    def compute_stats(satsinfo_messages):
        """计算 SATSINFO 统计量，返回与 GSV 兼容的聚合数据"""
        from .parser import NMEAParser
        samples = SATSINFOStatistics.collect_satellites(satsinfo_messages)

        snrs = [s['snr'] for s in samples if s['snr'] is not None]
        n_blank = len(samples) - len(snrs)

        # 逐卫星数据（星座-PRN 为键），附带逐频点 SNR 聚合
        per_sat = {}
        for s in samples:
            key = f"{s['constellation']}-{s['prn']}"
            v = per_sat.setdefault(key, {
                'prn': s['prn'], 'constellation': s['constellation'],
                'sys': s['sys'], 'lines': [], 'snrs': [], 'n_obs': 0,
                '_freq_snrs': {},
            })
            v['lines'].append(s['line_no'])
            v['n_obs'] += 1
            if s['snr'] is not None:
                v['snrs'].append(s['snr'])
            # 该星各频点统计：freq_status -> [snr,...]，按系统转换频点名
            const = v['constellation']
            for f in s['freqs']:
                fs = f.get('freq_status')
                name = NMEAParser.freq_name(f.get('sys'), fs)
                v['_freq_snrs'].setdefault(name, []).append(f['snr'])

        for v in per_sat.values():
            v['lines'] = np.array(v['lines'], dtype=float)
            v['snrs'] = np.array(v['snrs'], dtype=float)
            if len(v['snrs']):
                v['snr_mean'] = float(np.mean(v['snrs']))
                v['snr_max'] = float(np.max(v['snrs']))
                v['snr_min'] = float(np.min(v['snrs']))
            else:
                v['snr_mean'] = v['snr_max'] = v['snr_min'] = None
            # 频点详情：按频点名聚合 min/mean/max/best
            v['freqs'] = []
            for name, fs_snrs in sorted(v['_freq_snrs'].items()):
                fs_snrs = [x for x in fs_snrs if x is not None]
                if not fs_snrs:
                    continue
                v['freqs'].append({
                    'name': name,
                    'snr_mean': float(np.mean(fs_snrs)),
                    'snr_max': float(np.max(fs_snrs)),
                    'n_obs': len(fs_snrs),
                })
            del v['_freq_snrs']

        # 分星座（按观测次数降序）
        per_const = {}
        for v in per_sat.values():
            c = per_const.setdefault(v['constellation'], {
                'obs': 0, 'sats': set(), 'snrs': [],
            })
            c['obs'] += v['n_obs']
            c['sats'].add(v['prn'])
            c['snrs'].extend(v['snrs'])
        per_const = {
            name: {
                'obs': c['obs'], 'sats': len(c['sats']),
                'snr_mean': float(np.mean(c['snrs'])) if c['snrs'] else None,
                'snr_max': float(np.max(c['snrs'])) if c['snrs'] else None,
            }
            for name, c in sorted(per_const.items(), key=lambda kv: -kv[1]['obs'])
        }

        # 每条报文跟踪卫星数时序（作为"搜星数"）
        siv_lines, siv_vals = [], []
        for msg in satsinfo_messages:
            n_sat = len(msg.get('parsed', {}).get('satellites') or [])
            siv_lines.append(msg.get('line_no', 0))
            siv_vals.append(n_sat)

        def quality_count(lo, hi):
            return sum(1 for x in snrs if lo <= x < hi)

        quality = {
            'good': quality_count(GSVStatistics.SNR_GOOD, 1e9),
            'mid': quality_count(GSVStatistics.SNR_POOR, GSVStatistics.SNR_GOOD),
            'poor': quality_count(0, GSVStatistics.SNR_POOR),
        }

        return {
            'n_gsv': len(satsinfo_messages),
            'n_samples': len(samples),
            'n_blank_snr': n_blank,
            'n_unique_sats': len(per_sat),
            'snr': {
                'n': len(snrs),
                'min': float(np.min(snrs)) if snrs else None,
                'mean': float(np.mean(snrs)) if snrs else None,
                'median': float(np.median(snrs)) if snrs else None,
                'max': float(np.max(snrs)) if snrs else None,
                'std': float(np.std(snrs, ddof=1)) if len(snrs) > 1 else 0.0,
            },
            'quality': quality,
            'per_constellation': per_const,
            'per_satellite': per_sat,
            'sats_in_view': {'lines': np.array(siv_lines), 'values': np.array(siv_vals)},
            'samples': samples,
        }


class GSVStatistics:
    """GSV报文统计分析（卫星在视、信噪比、星座）"""

    # SNR 质量分档阈值（dBHz）
    SNR_GOOD = 40   # ≥40 优
    SNR_POOR = 30   # <30 差（30~40 良）

    @staticmethod
    def collect_satellites(gsv_messages):
        """收集所有卫星观测样本（只保留有仰角/方位角的卫星）"""
        samples = []
        for msg in gsv_messages:
            p = msg.get('parsed', {})
            line_no = msg.get('line_no', 0)
            for sat in p.get('satellites', []):
                if sat.get('elevation') is None and sat.get('azimuth') is None:
                    continue
                samples.append({
                    'line_no': line_no,
                    'prn': sat.get('prn', ''),
                    'constellation': sat.get('constellation', '未知'),
                    'elevation': sat.get('elevation'),
                    'azimuth': sat.get('azimuth'),
                    'snr': sat.get('snr'),
                })
        return samples

    @staticmethod
    def compute_stats(gsv_messages):
        """计算GSV统计量，返回聚合后的分析数据"""
        samples = GSVStatistics.collect_satellites(gsv_messages)

        # 有效 SNR 与未锁定(SNR空)
        snrs = [s['snr'] for s in samples if s['snr'] is not None]
        n_blank = len(samples) - len(snrs)

        # 逐卫星数据（星座-PRN 为键）
        per_sat = {}
        for s in samples:
            key = f"{s['constellation']}-{s['prn']}"
            v = per_sat.setdefault(key, {
                'prn': s['prn'], 'constellation': s['constellation'],
                'lines': [], 'snrs': [], 'n_obs': 0,
            })
            v['lines'].append(s['line_no'])
            v['n_obs'] += 1
            if s['snr'] is not None:
                v['snrs'].append(s['snr'])

        for v in per_sat.values():
            v['lines'] = np.array(v['lines'])
            v['snrs'] = np.array(v['snrs'])
            if len(v['snrs']):
                v['snr_mean'] = float(np.mean(v['snrs']))
                v['snr_max'] = float(np.max(v['snrs']))
                v['snr_min'] = float(np.min(v['snrs']))
            else:
                v['snr_mean'] = v['snr_max'] = v['snr_min'] = None

        # 分星座（按观测次数降序）
        per_const = {}
        for v in per_sat.values():
            c = per_const.setdefault(v['constellation'], {
                'obs': 0, 'sats': set(), 'snrs': [],
            })
            c['obs'] += v['n_obs']
            c['sats'].add(v['prn'])
            c['snrs'].extend(v['snrs'])
        per_const = {
            name: {
                'obs': c['obs'], 'sats': len(c['sats']),
                'snr_mean': float(np.mean(c['snrs'])) if c['snrs'] else None,
                'snr_max': float(np.max(c['snrs'])) if c['snrs'] else None,
            }
            for name, c in sorted(per_const.items(), key=lambda kv: -kv[1]['obs'])
        }

        # 搜星数(在视卫星数)时序
        siv_lines, siv_vals = [], []
        for msg in gsv_messages:
            p = msg.get('parsed', {})
            if p.get('sats_in_view') is not None:
                siv_lines.append(msg.get('line_no', 0))
                siv_vals.append(p['sats_in_view'])

        # SNR 质量分档
        def quality_count(lo, hi):
            return sum(1 for x in snrs if lo <= x < hi)

        quality = {
            'good': quality_count(GSVStatistics.SNR_GOOD, 1e9),   # ≥40
            'mid': quality_count(GSVStatistics.SNR_POOR, GSVStatistics.SNR_GOOD),  # 30~40
            'poor': quality_count(0, GSVStatistics.SNR_POOR),       # <30
        }

        return {
            'n_gsv': len(gsv_messages),
            'n_samples': len(samples),
            'n_blank_snr': n_blank,
            'n_unique_sats': len(per_sat),
            'snr': {
                'n': len(snrs),
                'min': float(np.min(snrs)) if snrs else None,
                'mean': float(np.mean(snrs)) if snrs else None,
                'median': float(np.median(snrs)) if snrs else None,
                'max': float(np.max(snrs)) if snrs else None,
                'std': float(np.std(snrs, ddof=1)) if len(snrs) > 1 else 0.0,
            },
            'quality': quality,
            'per_constellation': per_const,
            'per_satellite': per_sat,
            'sats_in_view': {'lines': np.array(siv_lines), 'values': np.array(siv_vals)},
            'samples': samples,
        }


class QualityStatistics:
    """测量环境质量统计（综合 GGA/GSA/GST/GSV/RMC/SATSINFO）

    集中计算定位质量、几何质量、信号质量、误差椭圆四大板块指标，
    返回结构化 dict 供 QualityReportDialog 绘图与摘要展示。
    """

    # Fix quality 显示名 —— 引用 solution.Q_NAMES 作为**唯一来源**。
    # 这里曾经另有一份表，对 5/6/8 的命名与 solution 不一致
    # （"RTK浮点"/"DR"/"模拟" vs "RTK浮动"/"估算"/"仿真"），
    # 同一个 Q 值在两个对话框里显示不同标签。别再另起一份。
    FIX_NAMES = Q_NAMES

    # 差分龄期告警阈值（秒）
    AGE_WARN_S = 60.0

    # RMC 判"运动"的速度阈值（节）。低于此视为静止/未运动。
    # 0.2 节 ≈ 0.1 m/s：正常架站/静态测试速度应远低于此。
    MOVING_KNOTS = 0.2

    @staticmethod
    def compute_quality(all_nmea_messages, options=None):
        """计算测量环境质量指标

        参数:
            all_nmea_messages — NMEAParser.parse_file 返回的 nmea_messages 列表
            options — 可选配置，目前支持:
                'recovery_min_s'  失锁最短时长（秒），短于此的丢固定视为抖动
                'age_warn_s'      差分龄期告警阈值（秒）
                'rejected'        parse_file 的拒收计数（按原因分类），
                                  并入完整性板块，避免"丢了多少"留白
                'rejected_samples' parse_file 的拒收样本（按原因留少量原文），
                                  供界面列出具体是哪几行被丢
        返回: 结构化 dict，各板块独立，缺失数据对应字段为 None

        关于时间轴：原先所有时序的横轴都是**文件行号**，既不能定位时刻也
        不能换算秒。这里先给报文补上绝对 UTC 时刻（timebase），各分项再
        额外给出 `times_utc`；`times`（行号）保留不删，避免破坏既有调用。
        """
        opts = dict(options or {})
        # 给每条报文补 utc_dt（原地），失败时 axis['available'] 为 False，
        # 各分项的 times_utc 里会是 None，界面据此退回行号并给出提示
        time_axis = timebase.build_time_axis(all_nmea_messages)

        # 按类型分拣
        gga = [m for m in all_nmea_messages if m.get('type') == 'GGA']
        gsa = [m for m in all_nmea_messages if m.get('type') == 'GSA']
        gst = [m for m in all_nmea_messages if m.get('type') == 'GST']
        gsv = [m for m in all_nmea_messages if m.get('type') == 'GSV']
        rmc = [m for m in all_nmea_messages if m.get('type') == 'RMC']
        satsinfo = [m for m in all_nmea_messages
                    if m.get('type') in ('SATSINFOA', 'SATSINFOB')]
        # REFSTATIONA 等基准站/板卡健康报文
        refstation = [m for m in all_nmea_messages
                      if (m.get('type') or '').startswith('REFSTATION')]

        # RTK 时效状态机（固定率/TTFF/失锁恢复/固定段）
        try:
            series = SolutionSeries.from_messages(
                all_nmea_messages, time_axis=time_axis)
            rtk = (analyse_rtk_state(
                series, recovery_min_s=opts.get('recovery_min_s', 1.0))
                if len(series) else {'available': False,
                                     'reason': '没有可用的解算历元'})
            rtk['series_note'] = {
                'n_gga_raw': series.n_gga_raw,
                'n_dup_gga': series.n_dup_gga,
                'n_unrecoverable': series.n_unrecoverable,
            }
        except Exception as e:                                # noqa: BLE001
            rtk = {'available': False,
                   'reason': f'RTK 时效分析失败: {type(e).__name__}: {e}'}

        return {
            'integrity': QualityStatistics._compute_integrity(
                all_nmea_messages, opts.get('rejected'),
                opts.get('rejected_samples')),
            'fix_quality': QualityStatistics._compute_fix(gga, rmc),
            'dop': QualityStatistics._compute_dop(gsa),
            'age': QualityStatistics._compute_age(
                gga, opts.get('age_warn_s')),
            'gst': QualityStatistics._compute_gst(gst),
            'signal': QualityStatistics._compute_signal(gsv, satsinfo),
            'lock_rate': QualityStatistics._compute_lock_rate(gsa, gsv),
            'cpu': QualityStatistics._compute_cpu(refstation, satsinfo),
            'rmc_motion': QualityStatistics._compute_rmc_motion(rmc),
            'rtk': rtk,
            'time_axis': time_axis,
            'msg_counts': {
                'GGA': len(gga), 'GSA': len(gsa), 'GST': len(gst),
                'GSV': len(gsv), 'RMC': len(rmc),
                'SATSINFO': len(satsinfo), 'REFSTATION': len(refstation),
            },
        }

    # -------------------- 报文完整性 --------------------

    @staticmethod
    def _compute_integrity(all_nmea_messages, rejected=None,
                           rejected_samples=None):
        """校验和完整性：通过 / 失败 / 无法判定，以及二进制帧修复与拒收条数

        为什么单列一块
        --------------
        解析器原先只检查校验和的**格式**，从不比对数值，被干扰的句子
        会和正常句子一样进入统计，且事后无法区分。对 RTK 调试来说，
        "这条链路上有多少报文本身是坏的"是结论，不是噪声。

        三态口径
        --------
        checksum_ok 为 None 表示**无法判定**（缺校验和、或私有报文用
        32 位 CRC），必须与 False 分开统计 —— 混成"失败"会冤枉正常报文，
        报出来的失败率就没人敢用了。

        为什么要带 rejected
        -------------------
        下面统计的只是"已经解析成功的报文"。在解析入口被拒掉的句子
        （超长、无逗号、不可修复、类型不支持）不在这个列表里，若不
        单独带进来，"丢了多少"永远是空白 —— 失败率再低也证明不了链路干净。
        rejected_samples 进一步给出"丢的是哪几行"，只留样本不留全量。
        """
        rejected = dict(rejected or {})
        if not all_nmea_messages and not rejected:
            return None

        n_ok = n_bad = n_unknown = n_repaired = 0
        by_type = {}
        bad_samples = []

        for m in all_nmea_messages:
            t = m.get('type') or m.get('full_type') or '?'
            rec = by_type.setdefault(
                t, {'total': 0, 'ok': 0, 'bad': 0, 'unknown': 0, 'repaired': 0})
            rec['total'] += 1

            if m.get('repaired'):
                n_repaired += 1
                rec['repaired'] += 1

            ck = m.get('checksum_ok')
            if ck is True:
                n_ok += 1
                rec['ok'] += 1
            elif ck is False:
                n_bad += 1
                rec['bad'] += 1
                if len(bad_samples) < 10:
                    bad_samples.append({
                        'line_no': m.get('line_no'),
                        'type': t,
                        'raw': (m.get('raw') or '')[:120],
                    })
            else:
                n_unknown += 1
                rec['unknown'] += 1

        total = len(all_nmea_messages)
        return {
            'total': total,
            'n_ok': n_ok,
            'n_bad': n_bad,
            'n_unknown': n_unknown,
            'n_repaired': n_repaired,
            # 全部被拒时 total 为 0，占比按 0 处理而不是除零
            'ok_rate': n_ok / total if total else 0.0,
            'bad_rate': n_bad / total if total else 0.0,
            'unknown_rate': n_unknown / total if total else 0.0,
            'by_type': by_type,
            'bad_samples': bad_samples,
            'rejected': rejected,
            'rejected_total': sum(rejected.values()),
            'rejected_samples': dict(rejected_samples or {}),
        }

    # -------------------- 定位质量 --------------------

    @staticmethod
    def _compute_fix(gga_msgs, rmc_msgs):
        """Fix 质量统计：解算类型分布、固定率、切换次数、时序

        「成功率」与「固定率」是两件事，必须分开：
          * success_rate = fix_quality > 0  —— 只是"有解率"，
            把单点/差分/浮点/固定混在一起，反映不出 RTK 能力
          * fix_rate     = fix_quality == 4 —— RTK 固定率，RTK 测试的核心指标
        """
        if not gga_msgs and not rmc_msgs:
            return None

        times, times_utc, values = [], [], []
        for m in gga_msgs:
            p = m.get('parsed', {})
            fq = p.get('fix_quality')
            if fq is not None:
                times.append(m.get('line_no', 0))
                times_utc.append(m.get('utc_dt'))
                values.append(fq)

        # RMC status A/V 补充（无 GGA 时用 RMC）
        if not values and rmc_msgs:
            for m in rmc_msgs:
                p = m.get('parsed', {})
                st = p.get('status', '')
                if st:
                    times.append(m.get('line_no', 0))
                    times_utc.append(m.get('utc_dt'))
                    values.append(1 if st == 'A' else 0)

        if not values:
            return None

        arr = np.array(values, dtype=float)
        times_arr = np.array(times, dtype=float)
        valid = arr[arr > 0]
        total = len(arr)
        valid_count = len(valid)

        # 质量分布
        dist = {}
        for v in values:
            dist[int(v)] = dist.get(int(v), 0) + 1

        # 切换次数
        switch_count = 0
        for i in range(1, len(values)):
            if values[i] != values[i - 1]:
                switch_count += 1

        n_fix = dist.get(4, 0)
        n_float = dist.get(5, 0)

        def _pct(n):
            return n / total * 100.0 if total else 0.0

        return {
            'total': total,
            'valid_count': valid_count,
            'success_rate': valid_count / total if total else 0.0,
            # ---- RTK 口径（各解算类型占比）----
            'n_fix': n_fix,
            'n_float': n_float,
            'fix_rate': _pct(n_fix),
            'fix_rate_valid': (n_fix / valid_count * 100.0) if valid_count
            else 0.0,
            'float_rate': _pct(n_float),
            'dgps_rate': _pct(dist.get(2, 0)),
            'single_rate': _pct(dist.get(1, 0)),
            'invalid_rate': _pct(dist.get(0, 0)),
            'quality_distribution': dist,
            'switch_count': switch_count,
            'times': times_arr,
            'times_utc': times_utc,
            'values': arr,
        }

    # -------------------- 几何质量 --------------------

    @staticmethod
    def _compute_dop(gsa_msgs):
        """DOP 时序统计：PDOP/HDOP/VDOP"""
        if not gsa_msgs:
            return None

        times, times_utc, pdops, hdops, vdops = [], [], [], [], []
        for m in gsa_msgs:
            p = m.get('parsed', {})
            ln = m.get('line_no', 0)
            if p.get('pdop') is not None:
                times.append(ln)
                times_utc.append(m.get('utc_dt'))
                pdops.append(p['pdop'])
                hdops.append(p.get('hdop') or np.nan)
                vdops.append(p.get('vdop') or np.nan)

        if not times:
            return None

        times_arr = np.array(times, dtype=float)
        times_utc_list = list(times_utc)

        def dop_stat(vals):
            arr = np.array(vals, dtype=float)
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                return None
            return {
                'mean': float(np.mean(valid)),
                'min': float(np.min(valid)),
                'max': float(np.max(valid)),
                'times': times_arr,
                'times_utc': times_utc_list,
                'values': arr,
            }

        pdop_s = dop_stat(pdops)
        hdop_s = dop_stat(hdops)
        vdop_s = dop_stat(vdops)

        # HDOP/VDOP 比值均值
        ratio = None
        if hdop_s and vdop_s:
            h_v = hdop_s['values'] / np.where(vdop_s['values'] > 0,
                                             vdop_s['values'], np.nan)
            valid_r = h_v[~np.isnan(h_v)]
            if len(valid_r):
                ratio = float(np.mean(valid_r))

        return {'pdop': pdop_s, 'hdop': hdop_s, 'vdop': vdop_s,
                'hdop_vdop_ratio': ratio}

    @staticmethod
    def _compute_lock_rate(gsa_msgs, gsv_msgs):
        """锁定率：参与解算卫星数 / 在视卫星数"""
        sats_used_vals = []
        for m in gsa_msgs:
            p = m.get('parsed', {})
            sats = p.get('sats_used', [])
            sats_used_vals.append(len(sats))

        sats_view_vals = []
        for m in gsv_msgs:
            p = m.get('parsed', {})
            siv = p.get('sats_in_view')
            if siv is not None:
                sats_view_vals.append(siv)

        used_mean = float(np.mean(sats_used_vals)) if sats_used_vals else None
        view_mean = float(np.mean(sats_view_vals)) if sats_view_vals else None
        lock_rate = None
        if used_mean is not None and view_mean and view_mean > 0:
            lock_rate = used_mean / view_mean

        return {
            'sats_used_mean': used_mean,
            'sats_in_view_mean': view_mean,
            'lock_rate': lock_rate,
        }

    # -------------------- 差分龄期 --------------------

    @staticmethod
    def _compute_age(gga_msgs, age_warn_s=None):
        """差分龄期统计；age_warn_s 可覆盖默认告警阈值"""
        warn_s = (float(age_warn_s) if age_warn_s is not None
                  else QualityStatistics.AGE_WARN_S)
        times, times_utc, ages = [], [], []
        for m in gga_msgs:
            p = m.get('parsed', {})
            age = p.get('age')
            if age is not None:
                times.append(m.get('line_no', 0))
                times_utc.append(m.get('utc_dt'))
                ages.append(age)

        if not ages:
            return None

        arr = np.array(ages, dtype=float)
        return {
            'mean': float(np.mean(arr)),
            'max': float(np.max(arr)),
            'age_warn_s': warn_s,
            'over_warn_count': int(np.sum(arr > warn_s)),
            'over_60s_count': int(np.sum(arr > QualityStatistics.AGE_WARN_S)),
            'times': np.array(times, dtype=float),
            'times_utc': times_utc,
            'values': arr,
        }

    # -------------------- 误差椭圆 (GST) --------------------

    @staticmethod
    def _compute_gst(gst_msgs):
        """GST 统计：RMS 时序、误差椭圆面积/长短轴比"""
        if not gst_msgs:
            return None

        times, times_utc, rms_vals = [], [], []
        smajs, smis, orient = [], [], []
        lat_errs, lon_errs, alt_errs = [], [], []

        for m in gst_msgs:
            p = m.get('parsed', {})
            ln = m.get('line_no', 0)
            rms = p.get('rms')
            if rms is not None:
                times.append(ln)
                times_utc.append(m.get('utc_dt'))
                rms_vals.append(rms)
            smaj = p.get('semi_major')
            smi = p.get('semi_minor')
            if smaj is not None:
                smajs.append(smaj)
            if smi is not None:
                smis.append(smi)
            if p.get('orientation') is not None:
                orient.append(p['orientation'])
            if p.get('lat_error') is not None:
                lat_errs.append(p['lat_error'])
            if p.get('lon_error') is not None:
                lon_errs.append(p['lon_error'])
            if p.get('alt_error') is not None:
                alt_errs.append(p['alt_error'])

        if not rms_vals:
            return None

        rms_arr = np.array(rms_vals, dtype=float)
        times_arr = np.array(times, dtype=float)

        # 误差椭圆面积 = π × semi_major × semi_minor
        areas = []
        ratios = []
        for ma, mi in zip(smajs, smis):
            if ma is not None and mi is not None and mi > 0:
                areas.append(math.pi * ma * mi)
                ratios.append(ma / mi)

        def safe_mean(lst):
            return float(np.mean(lst)) if lst else None

        return {
            'rms': {
                'mean': float(np.mean(rms_arr)),
                'min': float(np.min(rms_arr)),
                'max': float(np.max(rms_arr)),
                'times': times_arr,
                'times_utc': times_utc,
                'values': rms_arr,
            },
            'ellipse_area_mean': safe_mean(areas),
            'axis_ratio_mean': safe_mean(ratios),
            'lat_error_mean': safe_mean(lat_errs),
            'lon_error_mean': safe_mean(lon_errs),
            'alt_error_mean': safe_mean(alt_errs),
        }

    # -------------------- 信号质量 --------------------

    @staticmethod
    def _compute_signal(gsv_msgs, satsinfo_msgs):
        """信号质量：复用 GSVStatistics 或 SATSINFOStatistics"""
        if satsinfo_msgs:
            stats = SATSINFOStatistics.compute_stats(satsinfo_msgs)
            source = 'SATSINFO'
        elif gsv_msgs:
            stats = GSVStatistics.compute_stats(gsv_msgs)
            source = 'GSV'
        else:
            return None

        return {
            'source': source,
            'snr': stats['snr'],
            'quality': stats['quality'],
            'per_constellation': stats['per_constellation'],
            'n_samples': stats['n_samples'],
            'n_unique_sats': stats['n_unique_sats'],
            'samples': stats['samples'],
        }

    # -------------------- 接收机 CPU 健康 --------------------

    @staticmethod
    def _compute_cpu(ref_msgs, satsinfo_msgs):
        """接收机板卡 CPU 负载统计

        来源两种，量纲相反：
          * REFSTATION.cpu_load —— CPU 利用率(%)，值越高越忙
          * SATSINFO.cpuidle    —— CPU 空闲率(%)，值越低越忙，换算成利用率 100−cpuidle
        两者合并为统一"CPU 利用率"时序，便于判断板卡是否过载/异常。
        """
        times, times_utc, values = [], [], []
        for m in ref_msgs:
            p = m.get('parsed', {})
            v = p.get('cpu_load')
            if v is not None:
                times.append(m.get('line_no', 0))
                times_utc.append(m.get('utc_dt'))
                values.append(float(v))
        for m in satsinfo_msgs:
            p = m.get('parsed', {})
            idle = p.get('cpuidle')
            if idle is not None:
                times.append(m.get('line_no', 0))
                times_utc.append(m.get('utc_dt'))
                values.append(100.0 - float(idle))

        if not values:
            return None

        arr = np.array(values, dtype=float)
        return {
            'times': np.array(times, dtype=float),
            'times_utc': times_utc,
            'values': arr,
            'mean': float(np.mean(arr)),
            'max': float(np.max(arr)),
            'n_ref': len(ref_msgs),
            'n_satsinfo': len(satsinfo_msgs),
        }

    # -------------------- RMC 运动动态 --------------------

    @staticmethod
    def _compute_rmc_motion(rmc_msgs):
        """RMC 运动动态：速度/航向时序 + 动静态判断

        依据 MOVING_KNOTS 阈值统计运动/静止比例，用于确认
        接收机是否满足静态测试前提（架站不动）。
        """
        if not rmc_msgs:
            return None

        times, times_utc, speeds, courses = [], [], [], []
        for m in rmc_msgs:
            p = m.get('parsed', {})
            sp = p.get('speed_knots')
            if sp is not None:
                times.append(m.get('line_no', 0))
                times_utc.append(m.get('utc_dt'))
                speeds.append(float(sp))
                courses.append(p.get('course'))

        if not speeds:
            return None

        arr = np.array(speeds, dtype=float)
        total = len(arr)
        moving = int(np.sum(arr > QualityStatistics.MOVING_KNOTS))
        frac = moving / total if total else 0.0

        if moving == 0:
            state = '静止'
        elif frac >= 0.8:
            state = '运动'
        elif frac <= 0.2:
            state = '基本静止'
        else:
            state = '间歇运动'

        return {
            'times': np.array(times, dtype=float),
            'times_utc': times_utc,
            'speeds': arr,
            'courses': courses,
            'speed_mean': float(np.mean(arr)),
            'speed_max': float(np.max(arr)),
            'moving_count': moving,
            'still_count': total - moving,
            'moving_fraction': frac,
            'motion_state': state,
            'n_samples': total,
        }
