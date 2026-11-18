# -*- coding: utf-8 -*-
"""静态观测数据（RINEX）分析工作线程

信号契约与 worker.ParseWorker 完全一致，便于主窗口复用同一套
QThread + moveToThread 生命周期管理:
    progress(int, str) / finished(dict) / error(str)
"""

import os

from PyQt5.QtCore import QObject, pyqtSignal


class RinexWorker(QObject):
    """RINEX 观测文件解析 + 静态质量分析工作线程对象"""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, obs_path, nav_path=None):
        super().__init__()
        self.obs_path = obs_path
        self.nav_path = nav_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """在子线程中执行解析与分析"""
        try:
            # 延迟导入：避免主窗口启动时就引入星历/数值计算依赖
            from .rinex import RinexObs, BroadcastEphemeris
            from .rinex_statistics import StaticQuality

            base = os.path.basename(self.obs_path)
            self.progress.emit(2, f"正在读取观测文件: {base} ...")

            rnx = RinexObs(self.obs_path)
            rnx.parse()
            if self._cancelled:
                return

            n_ep = len(rnx.epochs)
            if n_ep == 0:
                self.error.emit(
                    "未能从文件中解析出任何观测历元。\n"
                    "请确认这是完整的 RINEX 观测文件（.26O / .rnx / .obs 等）。")
                return
            self.progress.emit(25, f"观测历元 {n_ep} 个，正在加载广播星历...")

            eph = None
            if self.nav_path:
                eph = BroadcastEphemeris(self.nav_path).parse()
                self.progress.emit(
                    40, f"广播星历已加载: {eph.satellite_count} 条记录")
            if self._cancelled:
                return

            # 分析阶段把内部百分比映射到 40~100
            def sub_progress(pct, msg):
                if self._cancelled:
                    raise InterruptedError("用户取消了分析")
                mapped = 40 + int(pct * 0.6)
                self.progress.emit(mapped, msg)

            analyzer = StaticQuality(rnx, eph)
            result = analyzer.compute(progress_callback=sub_progress)

            if self._cancelled:
                return

            if result.get('error'):
                self.error.emit(result['error'])
                return

            result['nav_file'] = self.nav_path
            self.finished.emit(result)

        except InterruptedError:
            self.error.emit("分析已取消")
        except MemoryError:
            self.error.emit("内存不足：文件过大，建议先按时间段裁切后再分析。")
        except Exception as e:      # noqa: BLE001 - 需把任意异常回传给 GUI
            self.error.emit(f"分析失败: {type(e).__name__}: {e}")


class EphemerisDownloadWorker(QObject):
    """广播星历下载工作线程

    与 RinexWorker 同样的信号契约，复用主窗口的进度条与线程管理。
    finished 传出下载（必要时合并）好的星历文件路径列表。
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, dates, dest_dir=None):
        """dates: [(year, doy), ...] 需要下载的日期"""
        super().__init__()
        self.dates = list(dates or [])
        self.dest_dir = dest_dir or os.path.join(
            os.path.expanduser('~'), '.gnss_qc', 'brdc')
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from . import rinex_ephemeris as RXE

            if not self.dates:
                self.error.emit("没有可下载的日期信息。")
                return

            def cb(msg):
                if self._cancelled:
                    raise InterruptedError("用户取消了下载")
                self.progress.emit(50, msg)

            self.progress.emit(5, "正在连接 IGS 数据服务器...")
            paths = RXE.download_many(self.dates, self.dest_dir,
                                      progress_callback=cb)

            if self._cancelled:
                return

            paths = [p for p in paths if p and os.path.exists(p)]
            if not paths:
                self.error.emit("下载完成但未得到可用的星历文件。")
                return

            self.progress.emit(100, f"星历已就绪: {os.path.basename(paths[0])}")
            self.finished.emit(paths)

        except InterruptedError:
            self.error.emit("下载已取消")
        except (ConnectionError, FileNotFoundError) as e:
            self.error.emit(str(e))
        except Exception as e:      # noqa: BLE001 - 需把异常回传给 GUI
            self.error.emit(f"星历下载失败: {type(e).__name__}: {e}")
