# -*- coding: utf-8 -*-
"""解析工作线程 (ParseWorker)

使用 QObject.moveToThread 模式，避免 GUI 卡死。
线程生命周期由主窗口统一管理（见 ui/main_window.py）。
"""

import os

from PyQt5.QtCore import QObject, pyqtSignal

from .parser import NMEAParser


class ParseWorker(QObject):
    """文件解析工作线程对象"""
    progress = pyqtSignal(int, str)       # (percent, message)
    finished = pyqtSignal(dict)           # 解析结果
    error = pyqtSignal(str)               # 错误信息

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        """在子线程中执行解析"""
        try:
            self.progress.emit(0, f"正在读取文件: {os.path.basename(self.filepath)}...")

            def progress_cb(current, total):
                if self._cancelled:
                    raise InterruptedError("用户取消了解析")
                percent = int(current / total * 100) if total > 0 else 0
                self.progress.emit(percent, f"解析中... {current}/{total} 行 ({percent}%)")

            result = NMEAParser.parse_file(self.filepath, progress_callback=progress_cb)

            if self._cancelled:
                return

            n_nmea = len(result['nmea_messages'])
            n_bin = len(result['binary_packets'])
            n_unk = len(result['unknown_lines'])
            self.progress.emit(100, f"解析完成: NMEA {n_nmea}条, 二进制 {n_bin}条, 未知 {n_unk}条")
            self.finished.emit(result)

        except InterruptedError:
            self.error.emit("解析已取消")
        except Exception as e:
            self.error.emit(f"解析失败: {str(e)}")
