# -*- coding: utf-8 -*-
"""
baseline.py — 两台接收机 RINEX 3.x 静态数据 → 相对坐标(ENU基线)与差值报告

原理：调用 RTKLIB 的 rnx2rtkp.exe 做静态相对定位（双差 + LAMBDA 整周模糊度固定），
      本模块只负责：找工具、配参数、收集导航文件、解析输出、做统计。
      第一个文件 = rover（待定站），第二个文件 = base（基准站）。
      RTKLIB 的 rnx2rtkp.exe 需自行准备（2.4.3 b34 或 demo5 均可），
      通过 --rtklib 指定其路径；本模块只负责驱动与结果解析。

用法（命令行）：
    python -m nmea_parser.baseline <rover观测> <base观测> [选项]
    观测参数可以直接给 .26O 文件，也可以直接给目录（自动在目录里找唯一的观测文件）

选项：
    --rtklib  路径    指定 rnx2rtkp.exe 或其所在目录（默认自动查找）
    --outdir  目录    结果 .pos 保存位置（默认当前目录）
    --sys     字符串   参与解算的卫星系统，默认 G,E,C（GLONASS 见下方说明）
    --elev    度数    截止高度角，默认 15
    --include-glo     尝试纳入 GLONASS（仅当你的接收机 .26G 是标准"千米"单位时可用）
    --debug           出错时生成 RTKLIB trace 文件便于排查

作为库使用（GUI 走这条入口，保证与命令行结果一致）：
    import rinex_baseline as rb
    res = rb.solve(rover, base, outdir="结果目录", log=print)
    res["report"]     # 报告文本（含固定率警告/精度声明）
    res["rows"]       # 逐历元 E/N/U 明细
    res["out_pos"]    # .pos 结果文件路径
    业务错误一律抛 BaselineError（CLI 打印友好提示，GUI 弹窗显示）

关于 GLONASS 的坑（已内置自动修复）：部分接收机导出的 .26G 把轨道
三要素单位写成"米/米每秒/米每二次方秒"，而 RINEX 标准是"千米/千米每秒/千米每二次方秒"，
RTKLIB 直接用会把 GLO 卫星位置放大 1000 倍导致解算输出假坐标。
当参与系统包含 R 时，本脚本自动检测该 bug 并生成 ÷1000 的修复副本（写入临时目录，
绝不改动原始数据）。默认参与系统仍为 G,E,C；要纳入 GLO 用 --sys G,R,C。
"""

import argparse
import contextlib
import hashlib
import io
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class BaselineError(Exception):
    """业务错误（文件缺失/解算失败/结果异常）：CLI 打印友好提示，GUI 捕获弹窗展示"""

# ----------------------------------------------------------------------------
# 第 1 步：找到能跑的 rnx2rtkp.exe（demo5 版依赖同目录 librtklib.dll，要先处理）
# ----------------------------------------------------------------------------
def find_exe(user_path=None):
    cands = []
    if user_path:
        p = Path(user_path)
        cands.append(p / "rnx2rtkp.exe" if p.is_dir() else p)
    if getattr(sys, "frozen", False):       # 打包成 exe 后：优先用随包附带的 RTKLIB
        cands.append(Path(sys._MEIPASS) / "rtklib" / "rnx2rtkp.exe")
    # 随包附带的解算器（源码运行时的首选，与 frozen 分支同一目录约定）
    cands.append(Path(__file__).resolve().parent / "rtklib" / "rnx2rtkp.exe")
    cands += [
        Path(tempfile.gettempdir()) / "rtkdl" / "rnx2rtkp.exe",  # 已修复 DLL 依赖的临时副本
    ]
    which = shutil.which("rnx2rtkp")
    if which:
        cands.append(Path(which))
    for c in cands:
        if c and Path(c).exists():
            return ensure_dll(Path(c))
    raise BaselineError("未找到 rnx2rtkp.exe，请用 --rtklib 指定 RTKLIB 目录")


def ensure_dll(exe):
    """demo5 版如果同目录没有 librtklib.dll（它放在 ../lib 下），就拷到临时目录组好"""
    if (exe.parent / "librtklib.dll").exists() or exe.parent.name == "rtkdl":
        return exe
    lib = exe.parent.parent / "lib" / "librtklib.dll"
    if lib.exists():
        stage = Path(tempfile.mkdtemp(prefix="rtkstage_"))
        shutil.copy2(exe, stage / exe.name)
        shutil.copy2(lib, stage / lib.name)
        return stage / exe.name
    return exe


# ----------------------------------------------------------------------------
# 第 2 步：观测文件参数 —— 传目录也行，自动识别 .26O/.rnx/.obs
# ----------------------------------------------------------------------------
def resolve_obs(arg, log=print):
    p = Path(arg)
    if p.is_file():
        return p
    if p.is_dir():
        hits = sorted(f for f in p.iterdir() if f.is_file() and (
            re.fullmatch(r"\.\d\d[Oo]", f.suffix) or f.suffix.lower() in (".rnx", ".obs")))
        if len(hits) == 1:
            return hits[0]
        if hits:
            log(f"提示: 目录 {p} 有 {len(hits)} 个观测文件，取第一个: {hits[0].name}")
            return hits[0]
    raise BaselineError(f"观测文件不存在或目录里找不到 RINEX 观测: {arg}")


# ----------------------------------------------------------------------------
# 第 3 步：收集导航文件 —— 读文件头第一行判断卫星系统，过滤坏的 GLONASS
# ----------------------------------------------------------------------------
def nav_system(f):
    """返回导航文件的系统字母 (G/R/E/C/...)，不是导航文件或读不出则返回 None"""
    try:
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            line = fh.readline()
    except OSError:
        return None
    if "NAV DATA" not in line.upper():
        return None
    m = re.search(r"NAV DATA\s+([A-Z])\s*:", line)      # 例: "3.02  N: GNSS NAV DATA    G: GPS"
    if m:
        return m.group(1)
    # 兜底：按常见扩展名猜 (.26N=GPS, .26G=GLO, .26L=GAL, .26C=BDS)
    return {"N": "G", "G": "R", "L": "E", "C": "C"}.get(f.suffix[-1:].upper())


_GLO_NUM = re.compile(r"[-+]?\d+\.\d+[DE][-+]\d+")   # RINEX 数值（D/E 指数两种写法）


def fix_glo_units(path, log=print):
    """GLONASS 单位 bug 自动修复。

    检测 .26G 的轨道三要素（位置/速度/加速度）单位：RINEX 标准为 千米 体系，
    部分接收机写成 米 体系。违规时生成 ÷1000 的修复副本（写入系统临时目录，
    用文件路径+修改时间做指纹，绝不改动你的原始数据），返回可用路径；
    合规则直接返回原路径。时钟参数与健康/频率/龄期等无量纲字段保持不变。
    """
    src = Path(path)
    try:
        text = src.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise BaselineError(f"GLONASS 导航文件读取失败: {src} ({e})")
    lines = text.splitlines()
    hdr = next((i for i, l in enumerate(lines) if "END OF HEADER" in l), None)
    if hdr is None:
        raise BaselineError(f"GLONASS 导航文件没有文件头（疑似损坏）: {src}")

    body = [l for l in lines[hdr + 1:] if l.strip()]
    if len(body) < 4 or not re.match(r"R\d\d\s", body[0]):
        raise BaselineError(f"GLONASS 导航记录结构异常（应为 历元行+3行轨道）: {src}")
    m = _GLO_NUM.search(body[1])           # 第一条记录的 X 位置
    if not m:
        raise BaselineError(f"GLONASS 导航轨道行解析失败: {src}")
    if abs(float(m.group(0).replace("D", "E"))) <= 1e6:
        return src                         # 位置量级 ≤1e6 km → 单位本来就是千米

    def conv(line):
        """一行 4 个字段：前 3 个(位置/速度/加速度)÷1000，第 4 个原样"""
        nums = _GLO_NUM.findall(line)
        if len(nums) < 4:
            return line                    # 结构不符宁可原样交给 RTKLIB 报错，也不乱改
        vals = [float(n.replace("D", "E")) for n in nums]
        vals[0] /= 1000.0                  # 位置 m   → km
        vals[1] /= 1000.0                  # 速度 m/s → km/s
        vals[2] /= 1000.0                  # 加速度    → km/s²
        # 必须保留 4 字符缩进：RTKLIB 读 RINEX3 正文行从 buff+4 起按 19 列取字段
        # （rnx2rtkp 源码 readrnxnavb: p=buff+sp, sp=4），丢了缩进会整行错位
        return "    " + "".join(f"{v:19.12E}".replace("E", "D") for v in vals)

    out = list(lines[:hdr + 1])
    rec = lines[hdr + 1:]
    j = 0
    while j < len(rec):
        line = rec[j]
        if re.match(r"R\d\d\s", line):     # 历元行：时钟参数单位正确，不动
            out.append(line)
            for k in range(1, 4):          # 紧随的 3 行轨道要素：÷1000
                if j + k < len(rec):
                    nxt = rec[j + k]
                    out.append(conv(nxt) if nxt.strip() else nxt)
            j += 4
        else:
            out.append(line)
            j += 1

    cache = Path(tempfile.gettempdir()) / "rinex_glo_fixed"
    cache.mkdir(parents=True, exist_ok=True)
    fp = f"{src.resolve()}|{src.stat().st_mtime_ns}"
    dst = cache / f"{hashlib.md5(fp.encode('utf-8')).hexdigest()[:10]}_{src.name}"
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    log(f"[修复] GLONASS {src.name}: 轨道单位'米'→'千米'，修复副本 {dst.name}")
    return dst


def collect_navs(obs_list, sys_set, include_glo, log=print):
    """收集两个目录的导航文件取并集（不同接收机导出的星历记录数可能差很多，
    只用一边可能导致某些时段缺星历、解算质量下降）。返回 [路径]"""
    by_sys = {}
    printed = set()                                   # 防止同一文件名在两个目录里重复打印提示
    for obs in obs_list:
        for f in sorted(obs.parent.iterdir()):
            if not f.is_file() or f == obs:
                continue
            if f.stat().st_size > 10_000_000:         # 观测文件很大，跳过
                continue
            s = nav_system(f)
            if s is None:
                continue
            if s == "R":
                if "R" not in sys_set:
                    if f.name not in printed:
                        printed.add(f.name)
                        log(f"[跳过] GLONASS 导航 {f.name}（GLONASS 未参与；"
                            "参与系统加 R 即自动修复单位并纳入解算）")
                    continue
                # R 已参与：自动检测单位 bug，违规则改用 ÷1000 的修复副本
                by_sys.setdefault("R", []).append(fix_glo_units(f, log=log))
                continue
            if s in sys_set:
                by_sys.setdefault(s, []).append(f)
    # 展平并按路径去重（用户两次传同一目录时避免重复）
    seen, flat = set(), []
    for s in sorted(by_sys):
        for f in by_sys[s]:
            key = str(f.resolve())
            if key not in seen:
                seen.add(key)
                flat.append(f)
    if not flat:
        raise BaselineError("两个观测文件目录里都找不到可用的导航文件（.26N/.26L/.26C 等）")
    log(f"导航文件 ({len(flat)}): {', '.join(f.name for f in flat)}")
    return flat


# ----------------------------------------------------------------------------
# 第 4 步：调用 RTKLIB 做静态相对定位
# ----------------------------------------------------------------------------
def _kill_process_tree(proc):
    """终止子进程及其全部后代，再回收管道 —— 必须整树杀

    只 kill 直接子进程（rnx2rtkp / cmd 包装层）时，孙进程仍握着
    stdout/stderr 管道句柄，随后的 communicate() 会一直等到管道 EOF，
    表现为「点了终止却要等子任务跑完才返回」。taskkill /T 先清掉
    整棵树，管道随之关闭，communicate 秒回。
    """
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        proc.kill()                        # 兜底：直接子进程（taskkill 失败时）
    except OSError:
        pass
    try:
        proc.communicate(timeout=5)        # 回收句柄；句柄被弃管则超时放行
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_solve(exe, rover, base, navs, out_pos, sys_str, elev, debug,
              log=print, cancelled=None):
    cmd = [str(exe), "-p", "3", "-sys", sys_str, "-f", "2",
           "-m", str(elev), "-v", "3", "-h", "-c", "-a", "-o", str(out_pos)]
    if debug:
        cmd += ["-x", "2"]
    cmd += [str(rover), str(base)] + [str(n) for n in navs]
    disp = " ".join(f'"{c}"' if (" " in c or any(ord(ch) > 127 for ch in c)) else c for c in cmd)
    log(f"执行: {disp}")
    # CREATE_NO_WINDOW：GUI 程序（窗口子系统）拉起控制台程序 rnx2rtkp 时，
    # Windows 会给子进程新开一个黑色控制台窗口，点「开始解算」会闪出来。
    # 用该标志让子进程附着到隐藏控制台；stdout/stderr 走管道不受影响。
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, creationflags=flags)
    except OSError as e:
        raise BaselineError(f"无法启动解算器 {exe}: {e}")

    # 不用 subprocess.run(timeout=)：它阻塞等待，中途无法响应取消。
    # 这里按 0.3 s 轮询 communicate —— 既保超时看门狗，也保用户取消
    # （GUI 的「⏹ 终止」/关窗走 cancelled 回调，置真即杀进程返回）。
    deadline = time.monotonic() + 1800
    while True:
        try:
            out, err = proc.communicate(timeout=0.3)
            break
        except subprocess.TimeoutExpired:
            if cancelled is not None and cancelled():
                _kill_process_tree(proc)
                raise BaselineError("解算已取消（用户终止）")
            if time.monotonic() > deadline:
                _kill_process_tree(proc)
                raise BaselineError("解算超时（超过 30 分钟）——数据量异常或 RTKLIB 卡住，可加 --debug 排查")
    if (proc.returncode & 0xFFFFFFFF) == 0xC0000135:
        raise BaselineError("RTKLIB 启动失败：缺少 DLL（0xC0000135），请用 --rtklib 指向一个完整的 RTKLIB 目录")
    if proc.returncode != 0:
        log((err or b"").decode("mbcs", errors="replace")[-800:])
        raise BaselineError(f"rnx2rtkp 退出码 {proc.returncode}，可加 --debug 生成 trace 排查")


# ----------------------------------------------------------------------------
# 第 5 步：解析 .pos（-a 模式列: 周 tow E N U Q ns sde sdn sdu ... ratio）
# ----------------------------------------------------------------------------
def parse_pos(path, log=print):
    rows = []
    header = []
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        raise BaselineError(f"没有生成结果文件: {path}")
    with fh:
        for line in fh:
            if line.startswith("%"):
                header.append(line.rstrip())
                continue
            if not line.strip():
                continue
            p = line.split()
            if len(p) < 15:
                continue
            try:
                rows.append(dict(t=float(p[1]), e=float(p[2]), n=float(p[3]), u=float(p[4]),
                                 q=int(p[5]), ns=int(p[6]), ratio=float(p[14]),
                                 week=int(float(p[0]))))
            except ValueError:
                continue
    if not rows:
        raise BaselineError("结果文件里没有任何解算历元（检查观测/导航是否匹配、时段是否重叠）")
    # 关键校验：-a 基线格式、内核版本、基准站坐标是否落在合理范围
    if not any("baseline" in h.lower() and "e/n/u" in h.lower() for h in header):
        raise BaselineError("输出不是 ENU 基线格式：解算器可能不支持 -a，请改用 demo5 版 rnx2rtkp")
    if not any("demo5" in h.lower() for h in header):
        log("[警告] 解算器不是 demo5，RINEX 3.02 多系统/基线输出支持可能不完整")
    for h in header:
        if "ref pos" in h.lower():
            try:
                v = h.split(":", 1)[1].split()
                lat, lon, hgt = float(v[0]), float(v[1]), float(v[2])
            except (ValueError, IndexError):
                break
            if abs(lat) > 89.9 or abs(hgt) > 10000:
                raise BaselineError(f"基准站坐标异常: {h.strip()} —— 请显式指定基准站 XYZ 后重跑")
            log(f"[校验] 基准站位置 {lat:.6f}° {lon:.6f}° {hgt:.2f} m")
            break
    return rows


def mean_std(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
    return m, s


def median(v):
    s = sorted(v)
    return s[len(s) // 2]


# ----------------------------------------------------------------------------
# 第 6 步：打印报告 —— 相对坐标(逐段统计) + 两站差值 + 质量指标
# ----------------------------------------------------------------------------
def report(rows, out_pos, rover, base, exe):
    fix = [r for r in rows if r["q"] == 1]
    n, nf = len(rows), len(fix)
    if nf < 0.9 * n:
        print(f"[警告] 固定率仅 {100.0*nf/n:.1f}%（{nf}/{n}），低于 90% 时结果不可用于毫米级判定；"
              "请先排查基准站坐标、星历、周跳与截止高度角")
    if fix:
        use = fix                                  # 只用固定解，绝不把 SINGLE/FLOAT 混进毫米级统计
        kind = f"固定解(Q=1, n={nf})"
    else:
        use = rows
        kind = "全部解(无固定解，仅参考，不可用于精度判定)"

    t_mid = median([r["t"] for r in use])
    A = [r for r in use if r["t"] < t_mid]
    B = [r for r in use if r["t"] >= t_mid]

    print()
    print("=" * 62)
    print("  相对定位结果   base(基准) = " + base.parent.name +
          "   rover(待定) = " + rover.parent.name)
    print("=" * 62)
    print(f"  解算器      : {exe.name}")
    print(f"  历元 / 固定 : {n} / {nf}  ({nf * 100.0 / n:.2f}%)   统计样本: {kind}")
    if A and B:
        print(f"  分段        : 前半 {len(A)} 历元 | 后半 {len(B)} 历元")
    print("-" * 62)
    print(f"  基线分量 ENU ({rover.parent.name} − {base.parent.name})，单位 mm，mean ± 样本标准差")
    print("-" * 62)
    halves = {}
    for key, label in (("e", "ΔE"), ("n", "ΔN"), ("u", "ΔU")):
        m, s = mean_std([r[key] for r in use])
        line = f"  {label} = {m * 1000:+9.2f} ± {s * 1000:6.2f} mm"
        if A and B:
            ma, _ = mean_std([r[key] for r in A])
            mb, _ = mean_std([r[key] for r in B])
            halves[key] = (mb - ma) * 1000
            line += f"   前半 {ma * 1000:+8.2f} | 后半 {mb * 1000:+8.2f} | 互差 {halves[key]:+6.2f}"
        print(line)
    print("-" * 62)
    e = mean_std([r["e"] for r in use])[0]
    n_ = mean_std([r["n"] for r in use])[0]
    u = mean_std([r["u"] for r in use])[0]
    horiz = math.hypot(e, n_) * 1000
    slope = math.sqrt(e * e + n_ * n_ + u * u) * 1000
    az = (math.degrees(math.atan2(e, n_)) + 360.0) % 360.0
    print(f"  基线长度    : 水平 {horiz:.1f} mm | 斜距 {slope:.1f} mm | 方位角 {az:.1f}°")
    ns_med = median([r["ns"] for r in use])
    ratios = [r["ratio"] for r in use if r["ratio"] > 0]
    rat = f"{median(ratios):.1f}" if ratios else "n/a"
    print(f"  卫星数 / ratio 中位: {ns_med} / {rat}")
    u_range = (max(r["u"] for r in use) - min(r["u"] for r in use)) * 1000
    print(f"  ΔU 极差     : {u_range:.1f} mm（时段性波动，多路径/天线相位中心引起）")
    if halves:
        print("-" * 62)
        print(f"  内符合重复性: 水平 {max(abs(halves.get('e', 0)), abs(halves.get('n', 0))):.1f} mm"
              f" | 高程 {abs(halves.get('u', 0)):.1f} mm  （前后半段互差；不含系统偏差）")
    print("=" * 62)
    print(f"  结果文件: {out_pos}")
    print(f"  反向 ({base.parent.name} − {rover.parent.name}) 取上面 ENU 三个分量的相反数。")
    print("  注意: 上面的 mm 级是内符合精度, 不是绝对精度;")
    print("        两站天线型号不同时, ΔU 含未知的差分相位中心系统差(mm~cm 级);")
    print("        要把高差核到亚毫米, 请原地交换两台设备再测一段做交换校准。")


# ----------------------------------------------------------------------------
# solve()：一次完整解算 —— CLI 与 GUI 的唯一编排入口（保证两条路径结果一致）
# ----------------------------------------------------------------------------
def solve(rover_arg, base_arg, outdir=".", sys_str="G,E,C", elev=15,
          include_glo=False, rtklib=None, debug=False, out_name=None,
          log=print, cancelled=None):
    """执行完整解算流程，返回:
        {"rows": [...], "report": "报告文本", "out_pos": Path,
         "rover": Path, "base": Path, "exe": Path}
    业务错误抛 BaselineError；cancelled() 返回 True 时杀掉子进程抛
    BaselineError("解算已取消（用户终止）")，供 GUI 终止/关窗使用。"""
    rover = resolve_obs(rover_arg, log=log)
    base = resolve_obs(base_arg, log=log)
    if rover.resolve() == base.resolve():
        raise BaselineError("rover 和 base 不能是同一个观测文件")

    # 系统列表：保持用户给定顺序，去重；--include-glo 时自动补 R
    sys_list = []
    for s in str(sys_str).split(","):
        s = s.strip().upper()
        if s and s not in sys_list:
            sys_list.append(s)
    sys_set = set(sys_list)
    if include_glo and "R" not in sys_set:
        sys_list.append("R")
        sys_set.add("R")
        log("[提示] include_glo 生效：已把 R 加入参与系统（请确认 .26G 轨道单位为千米）")

    exe = find_exe(rtklib)
    log(f"解算器: {exe}")

    navs = collect_navs([rover, base], sys_set, include_glo, log=log)

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    if not out_name:
        out_name = f"{rover.parent.name}_rel_{base.parent.name}.pos"
    out_pos = outdir_p / out_name

    run_solve(exe, rover, base, navs, out_pos, ",".join(sys_list),
              elev, debug, log=log, cancelled=cancelled)
    rows = parse_pos(out_pos, log=log)

    # 报告由 report() 打印 —— 这里捕获成文本返回（GUI 直接显示；CLI 再打一遍）
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(rows, out_pos, rover, base, exe)
    return {"rows": rows, "report": buf.getvalue(), "out_pos": out_pos,
            "rover": rover, "base": base, "exe": exe}


# ----------------------------------------------------------------------------
# CLI main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="两台接收机 RINEX 3.x 静态数据 → 相对坐标(ENU)与差值报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python rinex_baseline.py 2026314\\03140614.26O X1\\03140614.26O\n"
               "  python rinex_baseline.py C:\\data\\站A C:\\data\\站B --outdir D:\\结果")
    ap.add_argument("rover", help="待定站观测文件（或其所在目录）")
    ap.add_argument("base", help="基准站观测文件（或其所在目录）")
    ap.add_argument("--rtklib", help="rnx2rtkp.exe 或 RTKLIB 目录")
    ap.add_argument("--outdir", default=".", help="结果保存目录（默认当前目录）")
    ap.add_argument("--sys", default="G,E,C", help="参与解算的系统，默认 G,E,C")
    ap.add_argument("--elev", type=int, default=15, help="截止高度角(度)，默认 15")
    ap.add_argument("--include-glo", action="store_true", help="尝试纳入 GLONASS")
    ap.add_argument("--debug", action="store_true", help="生成 RTKLIB trace")
    args = ap.parse_args()

    try:
        res = solve(args.rover, args.base, outdir=args.outdir, sys_str=args.sys,
                    elev=args.elev, include_glo=args.include_glo,
                    rtklib=args.rtklib, debug=args.debug)
    except BaselineError as e:
        sys.exit(str(e))
    print(res["report"])


if __name__ == "__main__":
    main()
