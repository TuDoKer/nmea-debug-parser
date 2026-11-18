# -*- coding: utf-8 -*-
"""临时验证脚本：用合成几何验证 SPP 解算不崩溃且结果合理"""
import math
from nmea_parser import geodesy
from nmea_parser.spp import compute_height_series, _solve4

# 测站真实位置
rx_lat, rx_lon, rx_h = 30.0, 114.0, 50.0
rx_xyz = geodesy.blh_to_ecef(rx_lat, rx_lon, rx_h, ell=geodesy.WGS84)

# 5 颗静止卫星（固定 ECEF，每颗 ~2.6e7 m，含一点点高程散布）
sats = {}
for i in range(5):
    slat = 10.0 + 5.0 * i
    slon = 100.0 + 8.0 * i
    sats[f'G{1+i:02d}'] = geodesy.blh_to_ecef(slat, slon, 20200000.0,
                                              ell=geodesy.WGS84)

class FakeEph:
    epochs = {'G01': [1], 'G02': [1], 'G03': [1], 'G04': [1], 'G05': [1]}
    def best(self, satid, t):
        return {'toc': t, 'toe_abs': t, 'af0': 0.0, 'af1': 0.0, 'af2': 0.0}
    def sat_ecef(self, d, t):
        return sats.get(d if False else list(sats)[0][:0] and [0])

# 覆盖 sat_ecef：直接返回每颗卫星固定坐标
class FakeEph2(FakeEph):
    def best(self, satid, t):
        return {'satid': satid, 'toc': t, 'toe_abs': t,
                'af0': 0.0, 'af1': 0.0, 'af2': 0.0,
                'sqrtA': math.sqrt(2.656e7), 'e': 0.0, 'M0': 0.0,
                'deltaN': 0.0}
    def sat_ecef(self, eph, t):
        # 返回「发射时刻 ECEF」坐标，使 SPP 内部 +wdt 旋转后落到真值
        from nmea_parser.rinex import C_LIGHT, OMEGA_E
        sx, sy, sz = sats[eph['satid']]
        d = math.sqrt((sx - rx_xyz[0])**2 + (sy - rx_xyz[1])**2 + (sz - rx_xyz[2])**2)
        wdt = OMEGA_E * d / C_LIGHT
        c, s = math.cos(-wdt), math.sin(-wdt)
        return c * sx + s * sy, -s * sx + c * sy, sz

# 构造观测文件：3 个历元，伪距 = 真距 + 10m 接收机钟漂 + 噪声
class FakeRnx:
    def __init__(self):
        self.t_first = (2026, 9, 2, 13, 0, 0.0)
        self.t_last = (2026, 9, 2, 13, 0, 10.0)
        self.approx_xyz = rx_xyz
        self.epochs = []
        for k, secs in enumerate([0.0, 30.0, 60.0]):
            b = 0.01 * k          # 接收机钟漂
            satdict = {}
            for i in range(5):
                sid = f'G{1+i:02d}'
                sx, sy, sz = sats[sid]
                d = math.sqrt((sx - rx_xyz[0])**2 + (sy - rx_xyz[1])**2 + (sz - rx_xyz[2])**2)
                rho = d + 299792458.0 * b + 0.05 * (k + i)   # 钟偏 + 小噪声
                satdict[sid] = {'C1C': rho}
            self.epochs.append((secs, satdict))
    def time_span(self):
        return self.t_first, self.t_last

eph = FakeEph2()
rnx = FakeRnx()

for approx in (rx_xyz, [0.0, 0.0, 0.0]):
    res = compute_height_series(rnx, eph, approx_xyz=list(approx))
    print('--- approx start', approx)
    print('available =', res.get('available'))
    if res.get('available'):
        print('n_epochs  =', len(res['heights']))
        print('heights   =', [round(h, 2) for h in res['heights']])
        print('expected  ~', rx_h)
        print('lats      =', [round(l, 5) for l in res['lats']])
        print('rms       =', [round(r, 2) for r in res['rms']])
    else:
        print('reason    =', res.get('reason'))

# 简单测试 4x4 求解
a = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
b = [1, 2, 3, 4]
print('solve4    =', _solve4(a, b))