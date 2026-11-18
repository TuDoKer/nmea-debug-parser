# -*- coding: utf-8 -*-
"""广播星历自动获取（纯逻辑，无 GUI 依赖）

提供两条路径：
  1. auto_find()   —— 在观测文件附近/常见目录下搜索匹配日期的星历文件
  2. download()    —— 按观测文件日期从 IGS(BKG) 镜像下载 BRDC 星历

同时提供 candidate_dates() 处理跨日观测（一个观测文件可能跨 UTC 午夜，
需要同时拿到前后两天的星历才能覆盖整个时段）。
"""

import datetime
import gzip
import os
import re
import shutil
import urllib.error
import urllib.request

# IGS BKG 镜像的 BRDC 目录（无需认证，实测可用）
# 长文件名 8.3 格式：BRDC00IGS_R_YYYYDOY0000_01D_MN.rnx.gz
BRDC_URL = ("https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/"
            "BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz")

# 星历文件扩展名（RINEX NAV），用于目录扫描
NAV_EXTS = ('.rnx', '.nav', '.26p', '.25p', '.24p', '.26n', '.25n',
            '.26b', '.25b', '.26g', '.25g', '.26q', '.25q')

# 扫描星历时优先搜索的目录名
COMMON_DIRS = ('brdc', 'BRDC', 'BRDC_2026', 'nav', 'NAV', 'ephemeris', '星历')

# 星历文件优先级（数值越小越优先）
#   GPS/Galileo/混合 > GPS > QZSS/BDS > GLONASS
# GLONASS 用的是地心笛卡尔坐标数值积分，本工具的解析器不处理，故最低。
NAV_EXT_RANK = {
    '.rnx': 0, '.nav': 0,                    # RINEX 3 多系统
    '.26n': 1, '.25n': 1, '.24n': 1,         # RINEX 2 GPS
    '.26p': 2, '.25p': 2, '.24p': 2,         # RINEX 2 混合(GPS+GLO)
    '.26g': 3, '.25g': 3, '.26q': 3, '.25q': 3,   # Galileo / QZSS
    '.26b': 4, '.25b': 4,                    # BeiDou
    '.26c': 4, '.25c': 4,                    # BeiDou（接收机用 C 表示）
    '.26l': 5, '.25l': 5,                    # Galileo（接收机用 L 表示）
    '.26o': 9, '.25o': 9,                    # 观测文件，绝不应作为星历
}


def nav_rank(path):
    """星历文件的可用性优先级（越小越好），供调用方排序选择"""
    name = os.path.basename(path).lower()
    if name.endswith('.gz'):
        name = name[:-3]
    ext = os.path.splitext(name)[1]
    return NAV_EXT_RANK.get(ext, 6)

_USER_AGENT = 'Mozilla/5.0 (compatible; GNSS-QC-Tool/1.1)'


# ==================== 日期推算 ====================


def candidate_dates(year, month, day, hours_span=0.0):
    """返回覆盖观测时段所需的 DOY 列表

    跨 UTC 午夜的观测需要前后两天的星历。hours_span 为观测时长（小时）。
    返回 [(year, doy), ...]
    """
    dates = []
    try:
        d0 = datetime.date(year, month, day)
    except ValueError:
        return dates
    dates.append((d0.year, d0.timetuple().tm_yday))

    # 起点 + 时长跨过午夜 -> 需要次日星历
    end = (datetime.datetime(year, month, day)
           + datetime.timedelta(hours=hours_span))
    if end.date() != d0:
        d1 = end.date()
        dates.append((d1.year, d1.timetuple().tm_yday))
    return dates


def dates_from_result(result):
    """从 StaticQuality 分析结果中推出候选 DOY 列表"""
    info = (result or {}).get('info') or {}
    t0 = info.get('t_first')
    t1 = info.get('t_last')
    if not t0:
        return []

    span_h = 0.0
    if t1:
        try:
            a = datetime.datetime(*[int(x) for x in t0[:5]],
                                  tzinfo=datetime.timezone.utc)
            b = datetime.datetime(*[int(x) for x in t1[:5]],
                                  tzinfo=datetime.timezone.utc)
            span_h = max(0.0, (b - a).total_seconds() / 3600.0)
        except (TypeError, ValueError):
            span_h = 0.0
    return candidate_dates(t0[0], t0[1], t0[2], span_h)


def dates_from_obs_header(obs_path, max_header_lines=200):
    """只读 RINEX 观测文件头，推出候选 DOY 列表

    这是轻量版：不解析整个观测文件，只扫头文件里的
    TIME OF FIRST OBS / TIME OF LAST OBS，用于在正式分析前
    决定该去找/下载哪一天的星历。

    返回 [(year, doy), ...]；解析失败返回 []。
    """
    first = last = None
    try:
        with open(obs_path, 'r', encoding='ascii', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= max_header_lines:
                    break
                label = line[60:].strip() if len(line) > 60 else ''
                if label in ('TIME OF FIRST OBS', 'TIME OF LAST OBS'):
                    parts = line[:43].split()
                    if len(parts) >= 6:
                        try:
                            t = (int(parts[0]), int(parts[1]), int(parts[2]),
                                 int(parts[3]), int(parts[4]), float(parts[5]))
                        except ValueError:
                            continue
                        if label == 'TIME OF FIRST OBS':
                            first = t
                        else:
                            last = t
                if label == 'END OF HEADER':
                    break
    except OSError:
        return []

    if not first:
        return []

    span_h = 0.0
    if last:
        try:
            a = datetime.datetime(*first[:5], tzinfo=datetime.timezone.utc)
            b = datetime.datetime(*last[:5], tzinfo=datetime.timezone.utc)
            span_h = max(0.0, (b - a).total_seconds() / 3600.0)
        except (TypeError, ValueError):
            span_h = 0.0
    return candidate_dates(first[0], first[1], first[2], span_h)


# ==================== 本地搜索 ====================

_DOY_PATTERNS = (
    # RINEX 3 长名: BRDC00IGS_R_20262510000_01D_MN.rnx
    #                ^^^^yyy^^^^^doy^  （必须放在最前，避免被短名规则误匹配）
    re.compile(r'_R_(\d{4})(\d{3})\d{4}_', re.IGNORECASE),
    # 短名 RINEX NAV: brdc2510.26n（4 位 DOY，末位取整） / brdc251.26n
    re.compile(r'^brdc(\d{3})\d?\.\d{2}[npqbg]', re.IGNORECASE),
    # 短名 RINEX OBS 形式: brdc251.rnx（不带 0.26n 后缀）
    re.compile(r'^brdc(\d{3})(\d)?\.(rnx|nav)', re.IGNORECASE),
)


def _doy_of_filename(name):
    """从星历文件名中解析出 (year, doy)，识别不了返回 None

    注意: 短名（如 brdc251.rnx）本身不含年份，需要调用方按上下文补年份，
    因此这里对短名返回 year=None，由上层用观测年份填充。
    """
    for idx, pat in enumerate(_DOY_PATTERNS):
        m = pat.search(name)
        if not m:
            continue

        # 长名规则（第 0 条）：group(1)=年, group(2)=DOY
        if idx == 0:
            try:
                y, d = int(m.group(1)), int(m.group(2))
            except (IndexError, ValueError, TypeError):
                continue
            if 1 <= d <= 366 and 1990 <= y <= 2100:
                return y, d
            continue

        # 短名规则：group(1) 实际是 DOY（无年份），返回 year=None 由上层补
        try:
            d = int(m.group(1))
        except (IndexError, ValueError, TypeError):
            continue
        if 1 <= d <= 366:
            return None, d
    return None


# 接收机自带 NAV 文件（<站号>.<yy>N 等）文件名里没有年积日，
# 只能靠读文件内容里的历元时间来判断。只读开头若干行即可。
_YEAR_IN_NAME = re.compile(r'\.(\d{2})[npqbgcl]$', re.IGNORECASE)
_SYS_IN_NAME = re.compile(r'\.(\d{2})([npqbgcl])$', re.IGNORECASE)


def _nav_dates_from_content(path, max_lines=400):
    """读取 RINEX NAV 文件内容，返回其中出现的 (year, doy) 集合

    用于文件名不含年积日的情况：站文件如 STATION.26N。
    只需读文件开头一小段即可拿到历元日期（NAV 数据按时间顺序排列）。

    历元行格式（RINEX 2 NAV，2 位年份）:
        '06 26  9  8 12  0  0.0 ...'
         PRN yy mo dd hh mm ss —— 空格分隔，前 5 个字段足够定位日期
    """
    out = set()
    end_of_header = False
    try:
        with open(path, 'r', encoding='ascii', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                label = line[60:].strip() if len(line) > 60 else ''
                if label == 'END OF HEADER':
                    end_of_header = True
                    continue
                if not end_of_header:
                    continue

                parts = line[:22].split()
                # 只认「PRN + 2位年 + 月 + 日 ...」形式的历元行
                if len(parts) < 5:
                    continue
                if not (parts[0].isdigit() and len(parts[1]) == 2):
                    continue
                try:
                    yy = int(parts[1])
                    mo = int(parts[2])
                    dd = int(parts[3])
                except ValueError:
                    continue
                yy = yy + 2000 if yy < 80 else yy + 1900
                if not (1 <= mo <= 12 and 1 <= dd <= 31):
                    continue
                try:
                    out.add((yy, datetime.date(yy, mo, dd)
                             .timetuple().tm_yday))
                except ValueError:
                    pass
    except OSError:
        return set()
    return out


def _iter_scan_dirs(obs_path):
    """产出搜索星历的候选目录（去重）"""
    obs_dir = os.path.dirname(os.path.abspath(obs_path))
    seen = set()
    cands = [obs_dir]

    # 逐级向上两层
    parent = obs_dir
    for _ in range(2):
        parent = os.path.dirname(parent)
        if parent and parent != obs_dir:
            cands.append(parent)

    # 常见子目录名
    for base in (obs_dir, os.path.dirname(obs_dir)):
        for sub in COMMON_DIRS:
            cands.append(os.path.join(base, sub))

    for d in cands:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            yield d


def auto_find(obs_path, wanted_dates, fallback_year=None, max_depth=3):
    """在观测文件附近搜索匹配日期的星历文件

    搜索策略（按优先级）:
      1. 观测文件所在目录及其上两层
      2. 上述目录下常见星历子目录（brdc / nav / 星历 等）
      3. 以观测文件所在目录及上一层为根，做受限深度的递归遍历
         （覆盖 brdc/2026/251/ 这类按年/年积日分层的结构）

    参数:
        obs_path: 观测文件路径
        wanted_dates: [(year, doy), ...] —— 需要覆盖的日期
        fallback_year: 短名星历（brdc251.rnx 不含年份）使用的年份
    返回:
        {'found': [(path, (year,doy), size), ...], 'missing': [(year,doy), ...]}
    """
    wanted = {(y, d) for y, d in wanted_dates}
    found = []                    # 每个日期可能命中多个候选，最后再挑
    seen_files = set()

    # 按扩展名判断是否是星历文件（.rnx / .nav / .26N ...）
    def _looks_like_nav(name):
        low = name.lower()
        if low.endswith('.gz'):
            low = low[:-3]
        ext = os.path.splitext(low)[1]
        return ext in NAV_EXTS or bool(_SYS_IN_NAME.search(low))

    def _consider(full):
        if full in seen_files:
            return
        seen_files.add(full)
        name = os.path.basename(full)
        if not os.path.isfile(full):
            return
        if not _looks_like_nav(name):
            return
        info = _doy_of_filename(name)
        if info is not None:
            y, d = info
            if y is None:                   # 短名无年份，用观测年份兜底
                y = fallback_year
            if y is not None and (y, d) in wanted:
                found.append((full, (y, d), os.path.getsize(full)))
            return

        # 文件名没有年积日（接收机自带 NAV，如 STATION.26N）
        # -> 读文件内容里的历元日期来判断。压缩文件跳过。
        if name.lower().endswith('.gz'):
            return
        if os.path.getsize(full) > 80 * 1024 * 1024:
            return                          # 过大跳过，避免拖慢扫描
        for key in _nav_dates_from_content(full):
            if key in wanted:
                found.append((full, key, os.path.getsize(full)))

    def _scan_dir(directory):
        try:
            for name in os.listdir(directory):
                _consider(os.path.join(directory, name))
        except OSError:
            pass

    # --- 第 1、2 步：直接目录 + 常见子目录 ---
    roots = list(_iter_scan_dirs(obs_path))
    for d in roots:
        _scan_dir(d)

    # --- 第 3 步：受限深度递归遍历 ---
    if {k for _, k, _ in found} < wanted:
        has_abs = ':' in obs_path or obs_path.startswith(('/', '\\'))
        obs_dir = os.path.dirname(os.path.abspath(obs_path)) if has_abs \
            else os.path.dirname(obs_path)

        # 关键: 星历常与观测数据是「兄弟目录」（如 SITE_A/Base 与 brdc/），
        # 因此必须从两者的共同祖先开始遍历，而不是从观测目录本身。
        # 统一再上跳两层作为遍历根，保证能覆盖兄弟目录。
        walk_roots = []
        node = os.path.dirname(obs_dir) or obs_dir
        for _ in range(2):
            walk_roots.append(node)
            parent = os.path.dirname(node)
            if not parent or parent == node:
                break
            node = parent

        for root in walk_roots:
            if not os.path.isdir(root):
                continue
            base_depth = root.rstrip('/\\').count(os.sep)
            for cur, dirs, files in os.walk(root):
                if cur.count(os.sep) - base_depth >= max_depth:
                    dirs[:] = []
                # 跳过明显无关的目录，避免拖慢扫描
                dirs[:] = [x for x in dirs
                           if x.lower() not in ('windows', '$recycle.bin',
                                                'system volume information',
                                                'node_modules', '.git',
                                                '__pycache__', 'appdata')]
                for name in files:
                    _consider(os.path.join(cur, name))
            if {k for _, k, _ in found} >= wanted:
                break

    # 同一日期可能命中多个文件（如 .26N / .26C / .26G / .26P），
    # 按星历类型优先级去重；同名多份时保留较大的（通常覆盖时段更全）。
    best = {}
    for path, key, size in found:
        cand = (nav_rank(path), -size)
        if key not in best or cand < best[key][0]:
            best[key] = (cand, path, size)

    picked = sorted(((v[1], k, v[2]) for k, v in best.items()),
                    key=lambda it: it[1])
    return {'found': picked, 'missing': sorted(wanted - set(best))}


# ==================== 在线下载 ====================


def download(year, doy, dest_dir, timeout=90, progress_callback=None):
    """从 IGS(BKG) 下载指定日期的 BRDC 星历并解压

    返回解压后的本地路径；失败抛异常（由调用方转成用户可读信息）。
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = BRDC_URL.format(year=year, doy=doy)
    gz_name = os.path.basename(url)
    gz_path = os.path.join(dest_dir, gz_name)
    rnx_path = os.path.join(dest_dir, gz_name[:-3])       # 去掉 .gz

    if os.path.exists(rnx_path) and os.path.getsize(rnx_path) > 1024:
        return rnx_path                                   # 已有缓存

    if progress_callback:
        progress_callback(f"正在下载 {gz_name} ...")

    req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get('Content-Length') or 0)
            done = 0
            tmp = gz_path + '.part'
            with open(tmp, 'wb') as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress_callback and total:
                        pct = int(done / total * 100)
                        progress_callback(
                            f"正在下载星历 {gz_name}  {pct}% "
                            f"({done / 1048576:.1f}/{total / 1048576:.1f} MB)")
            os.replace(tmp, gz_path)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(
                f"该日期（{year} 年第 {doy} 天）的星历在服务器上不存在，"
                f"可能尚未发布。") from e
        raise
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"网络连接失败：{e.reason}。请检查网络后重试，"
            f"或改为手动指定本地星历文件。") from e
    except TimeoutError as e:
        raise ConnectionError("下载超时，请检查网络后重试。") from e

    if progress_callback:
        progress_callback("正在解压星历...")

    # 解压（用 shutil.copyfileobj 处理 .gz 中可能的内嵌文件名）
    with gzip.open(gz_path, 'rb') as fin, open(rnx_path, 'wb') as fout:
        shutil.copyfileobj(fin, fout, 1024 * 1024)

    return rnx_path


def download_many(date_list, dest_dir, progress_callback=None):
    """批量下载并合并候选日期的星历

    多个日期的星历需要合并成一个文件，因为解析器按单文件读取。
    返回合并后的路径列表（通常只需第一个，因为已合并）。
    """
    paths = []
    for y, d in date_list:
        paths.append(download(y, d, dest_dir,
                              progress_callback=progress_callback))
    if len(paths) <= 1:
        return paths

    merged = os.path.join(dest_dir, 'brdc_merged.rnx')
    _merge_nav(paths, merged)
    return [merged]


def _merge_nav(sources, dest):
    """把多个 RINEX NAV 文件合并成一个（保留第一个的头，拼接数据段）"""
    with open(dest, 'wb') as out:
        first = True
        for src in sources:
            with open(src, 'rb') as f:
                lines = f.readlines()
            i = 0
            while i < len(lines) and lines[i][60:].strip() != b'END OF HEADER':
                if first:
                    out.write(lines[i])
                i += 1
            if first and i < len(lines):
                out.write(lines[i])          # 写入 END OF HEADER
                first = False
            i += 1
            for line in lines[i:]:
                out.write(line)
    return dest
