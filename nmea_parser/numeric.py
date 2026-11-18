# -*- coding: utf-8 -*-
"""共用的数值统计 helper

为什么单独一个模块
------------------
分位数/均方根这类函数原先在四个地方各写了一份：

  * ``accuracy._pct``
  * ``track._pct``（同一文件里还有两处嵌套定义）
  * ``rinex_statistics._quantile``

各写一份的直接后果是**bug 只修在一处**。本项目就踩过：
``rinex_statistics._quantile`` 曾经既取向上取整的"次序统计量"、
又被传入**未排序**的列表，导致真双频判定整体错乱 ——
这个 bug 在另外三份里根本不存在，因为它们恰好都在调用前排了序。

所以这里统一实现，并把「容易踩的两个坑」在接口上堵死：

  1. ``quantile`` 明确要求**已排序**输入（名字里就写着），性能好；
  2. ``quantile_of`` 自动排序，是"安全但略慢"的入口。

调用方按需要挑：已经在循环里维护了有序列表就用前者，
手上是随手收集的原始值就用后者 —— 不要自己再写第三份。
"""

import math


def quantile(sorted_vals, q):
    """线性插值分位数（与 numpy.percentile 默认算法一致）

    **要求 sorted_vals 已升序排序**。传入未排序的列表不会报错，
    但取出的是"某个时间点的值"而不是分位数 —— 这种错误极其隐蔽，
    本项目踩过一次（见模块 docstring）。拿不准就用 quantile_of。
    """
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def quantile_of(vals, q):
    """自动排序的安全版本；输入可以是任意可迭代对象"""
    seq = sorted(vals)
    return quantile(seq, q)


def mean(vals):
    if not vals:
        return None
    return sum(vals) / len(vals)


def median(vals):
    return quantile_of(vals, 0.5)


def rms(vals):
    """均方根。注意这是**未去均值**的 RMS（含系统性偏移）"""
    if not vals:
        return None
    return math.sqrt(sum(x * x for x in vals) / len(vals))


def pstdev(vals):
    """总体标准差（除以 n，与 statistics.pstdev 一致）"""
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def percentile_range(vals, qs=(0.5, 0.95)):
    """一次排序取多个分位，避免重复排序"""
    seq = sorted(vals)
    return {q: quantile(seq, q) for q in qs}
