# -*- coding: utf-8 -*-
"""NMEA0183 报文解析（纯逻辑，无GUI依赖）"""

import re


class NMEAParser:
    """NMEA0183 报文解析器"""

    # 已知的NMEA报文类型白名单
    KNOWN_NMEA_TYPES = {
        'GGA', 'GSA', 'GST', 'RMC', 'GLL', 'VTG', 'ZDA', 'GSV',
        'GNS', 'DTM', 'GBS', 'GRS',
        'REFSTATIONA', 'REFSTATIONB',
        'SATSINFOA', 'SATSINFOB',
    }

    # GSV 报文 talker → 星座
    TALKER_CONSTELLATION = {
        'GP': 'GPS',
        'GL': 'GLONASS',
        'GA': 'Galileo',
        'BD': '北斗',
        'GB': '北斗',
        'GQ': 'QZSS',
        'GI': 'IRNSS',
    }

    # 扩展报文系统标识 → 星座显示名
    SYSTEM_NAMES = {
        0: 'GPS',
        1: 'GLONASS',
        2: 'SBAS',
        3: 'Galileo',
        4: '北斗',
        5: 'QZSS',
    }

    # 扩展报文频点标识 → 频点显示名（按系统）
    FREQ_NAMES = {
        'GPS': {0: 'L1C/A', 9: 'L2P(Y)', 3: 'L1C(P)', 11: 'L1C(D)',
                6: 'L5(D)', 14: 'L5(P)', 17: 'L2C(L)'},
        'GLONASS': {0: 'L1C/A', 5: 'L2C/A', 6: 'G3I', 7: 'G3Q'},
        'QZSS': {0: 'L1C/A', 6: 'L5(D)', 14: 'L5(P)', 17: 'L2C(L)',
                 18: 'L61(D)', 22: 'L61(P)'},
        '北斗': {0: 'B1I', 4: 'B1Q', 8: 'B1C(P)', 23: 'B1C(D)', 5: 'B2Q',
                 17: 'B2I', 12: 'B2a(P)', 28: 'B2a(D)', 6: 'B3Q',
                 21: 'B3I', 13: 'B2b(I)'},
        'Galileo': {1: 'E1B', 2: 'E1C', 12: 'E5A(P)', 17: 'E5B(P)',
                    18: 'E6B', 22: 'E6C'},
        'SBAS': {0: 'L1C/A', 6: 'L5(I)', 24: 'L62(D1)', 25: 'L62(D2)'},
    }

    @staticmethod
    def freq_name(sys_id, freq_status):
        """根据系统标识与频点标识解析频点名称"""
        const = NMEAParser.SYSTEM_NAMES.get(sys_id)
        if const is None:
            return f'SYS{sys_id}-F{freq_status}'
        m = NMEAParser.FREQ_NAMES.get(const, {})
        return m.get(freq_status, f'F{freq_status}')

    @staticmethod
    def parse_file(filepath, progress_callback=None):
        """
        解析.dbg文件，返回结构化数据
        progress_callback: 可选回调函数(current, total) 用于进度报告
        返回: {
            'nmea_messages': [{'type': 'GGA', 'raw': '...', 'parsed': {...}}, ...],
            'binary_packets': [{'raw_bytes': b'...', 'hex_preview': '...', 'line_no': 1}, ...],
            'unknown_lines': [{'content': '...', 'line_no': 1}, ...],
            'rejected': {'too_long': 3, 'unrepairable': 12, ...},
            'rejected_samples': {'too_long': [{'line_no': 8, 'content': '...'}], ...}
        }
        """
        result = {
            'nmea_messages': [],
            'binary_packets': [],
            'unknown_lines': [],
            'rejected': {},
            'rejected_samples': {},
        }

        with open(filepath, 'rb') as f:
            raw_data = f.read()

        # 按原始字节分割行（支持\n和\r\n）
        raw_lines = []
        start = 0
        for i, b in enumerate(raw_data):
            if b == 0x0A:  # \n
                end = i
                if i > 0 and raw_data[i - 1] == 0x0D:  # \r\n
                    end = i - 1
                raw_lines.append(raw_data[start:end])
                start = i + 1
        if start < len(raw_data):
            raw_lines.append(raw_data[start:])

        total_lines = len(raw_lines)
        rejected = result['rejected']
        rejected_samples = result['rejected_samples']
        samples_per_reason = NMEAParser.REJECT_SAMPLES_PER_REASON

        def _reject(reason, text=''):
            """记录被拒收的报文：按原因计数，并各留少量原文

            原先这些分支都是静默 continue，被丢掉的句子不留任何痕迹，
            事后既算不出"丢了多少"，也说不清"为什么丢"。
            计数之外还留样，是因为排查干扰段时需要看到被打断的位置 ——
            只报"不可修复 403 条"仍然不知道该改帧长还是查串口。
            留样按原因封顶，坏行可以上万条，全留会把内存吃光。
            """
            rejected[reason] = rejected.get(reason, 0) + 1
            bucket = rejected_samples.setdefault(reason, [])
            if len(bucket) < samples_per_reason:
                shown = ''.join(c if 32 <= ord(c) <= 126 else '·'
                                for c in text[:160])
                bucket.append({'line_no': line_no, 'content': shown})

        skip_until = 0
        for line_no, raw_line in enumerate(raw_lines, 1):
            if not raw_line:
                continue

            # 报告进度（每100行或最后一行）
            if progress_callback and (line_no % 100 == 0 or line_no == total_lines):
                progress_callback(line_no, total_lines)

            # 已被上一行拼接成完整句子的续行，不再单独分类
            if line_no <= skip_until:
                continue

            # 解码为文本（用于NMEA提取），保留原始字节
            line = NMEAParser._decode_line(raw_line)

            stripped = line.strip()

            # 尝试提取NMEA报文（可能在二进制数据中混杂）
            nmea_matches = list(NMEAParser._NMEA_RE.finditer(line))

            has_nmea = False
            for match in nmea_matches:
                msg = match.group(0)
                # 严格验证：
                # 1. 长度合理
                # 2. 包含逗号分隔字段
                # 3. 可打印字符比例高（>95%）
                if len(msg) < 10:
                    _reject('too_short', msg)
                    continue
                # SATSINFO 报文很长（数十颗卫星×多频点），不受 300 上限限制
                if len(msg) > 300 and not msg.startswith('#SATSINFO'):
                    _reject('too_long', msg)
                    continue
                if ',' not in msg:
                    _reject('no_comma', msg)
                    continue
                printable = sum(1 for c in msg if 32 <= ord(c) <= 126)
                was_repaired = False
                if printable / len(msg) < 0.95:
                    # 句子可能被二进制帧从中间打断，先尝试修复再决定丢弃
                    repaired = NMEAParser.repair_interleaved(msg)
                    if repaired is None:
                        _reject('unrepairable', msg)
                        continue
                    msg = repaired
                    was_repaired = True

                # 半句：'$' 报文以 *XX 收尾，没有终止符说明这一行被截断了
                # （典型是二进制帧把句子尾巴连同校验和一起吃掉）。
                # 这种片段原先会被当成"没有校验和"的正常报文收下，
                # 统计里表现为一条无法判定校验和的 RMC —— 其实是坏数据。
                joined = False
                if (msg.startswith('$')
                        and not NMEAParser._TERMINATOR_RE.search(msg)):
                    tail = NMEAParser._join_continuation(
                        raw_lines, line_no - 1, msg)
                    if tail is None:
                        _reject('truncated', msg)
                        continue
                    msg, last_idx = tail
                    skip_until = last_idx + 1      # 续行不再单独处理
                    joined = True

                parsed = NMEAParser.parse_nmea_message(msg, line_no)
                # 拼接出来的句子必须靠校验和自证：拼错了就会校验失败，
                # 不能因为"拼上了"就当成有效数据
                if parsed and joined and parsed.get('checksum_ok') is not True:
                    parsed = None
                if parsed:
                    parsed['repaired'] = was_repaired
                    if joined:
                        parsed['joined'] = True
                    result['nmea_messages'].append(parsed)
                    has_nmea = True
                else:
                    _reject('truncated' if joined else
                            NMEAParser._reject_reason(msg), msg)

            # 判断整行是否为二进制数据（基于原始字节）
            printable_count = sum(1 for b in raw_line if 32 <= b <= 126 or b in (9, 10, 13))
            if len(raw_line) > 0 and (printable_count / len(raw_line)) < 0.3 and not has_nmea:
                # 二进制数据行
                hex_preview = raw_line[:64].hex(' ').upper()
                result['binary_packets'].append({
                    'raw_bytes': bytes(raw_line),
                    'hex_preview': hex_preview,
                    'line_no': line_no,
                    'length': len(raw_line)
                })
            elif not has_nmea and stripped:
                # 未知行
                result['unknown_lines'].append({
                    'content': stripped[:200],
                    'line_no': line_no
                })

        return result

    # 报文提取正则。
    # '$' 报文按 *XX 收尾定界，而不是贪婪吃到行尾 —— 一行里可能有多条
    # 报文（或报文后面紧跟垃圾），贪婪匹配会把后一条一起吞进来，再按
    # 最后一个 '*' 切字段就串了，finditer 也就形同虚设。
    # '#' 私有报文用 8 位 CRC，不能按两位截断，仍整行取。
    _NMEA_RE = re.compile(r'\$[A-Z]{2,6}(?:[^*\r\n]*\*[0-9A-Fa-f]{2}|[^\r\n]*)'
                          r'|#[A-Z]{2,6}[^\r\n]*')

    # 每种拒收原因最多留多少条原文样本（够定位即可，不随文件大小增长）
    REJECT_SAMPLES_PER_REASON = 10

    # '$' 报文的终止符：'*' 加两位十六进制，且落在句尾
    _TERMINATOR_RE = re.compile(r'\*[0-9A-Fa-f]{2}$')

    # 跨行拼接最多往下看几行（真被换行劈开的句子只差一两行）
    JOIN_MAX_LINES = 3

    @staticmethod
    def _decode_line(raw_line):
        """按 utf-8 → gbk → latin-1 依次尝试解码，保证不抛异常"""
        try:
            return raw_line.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return raw_line.decode('gbk')
            except UnicodeDecodeError:
                return raw_line.decode('latin-1', errors='replace')

    @staticmethod
    def _join_continuation(raw_lines, idx, seg):
        """把被换行劈开的半句与后续行接回去

        返回 (拼接后的句子, 末行索引) 或 None。

        为什么要设边界：拼接的前提是"下一行是这条句子的续尾"。
        一旦下一行出现新的报文起始符（$ / #），那就是另一条报文，
        再往下拼只会把两条句子粘成一条 —— 拼出来的东西校验和必然
        不过，还会白白吃掉后面那条好报文。
        拼接结果还必须全为可打印字符：续行可能是二进制帧。
        """
        parts = [seg]
        for k in range(idx + 1,
                       min(idx + 1 + NMEAParser.JOIN_MAX_LINES,
                           len(raw_lines))):
            nxt = NMEAParser._decode_line(raw_lines[k])
            if '$' in nxt or '#' in nxt:
                return None
            parts.append(nxt)
            joined = ''.join(parts)
            if len(joined) > 300:
                return None
            if not NMEAParser._TERMINATOR_RE.search(joined):
                continue
            if all(32 <= ord(c) <= 126 for c in joined):
                return joined, k
            return None
        return None

    # 各 NMEA 报文的合法字段数（按逗号分割后的个数，含开头那个 $TYPE 字段）。
    # 用来给「修复被打断句子」时定位正确切点，允许几个变体以防版本差异。
    _FIELD_COUNT = {
        'GGA': (15,), 'RMC': (12, 13, 14), 'GST': (9,), 'GSA': (18, 19),
    }

    @staticmethod
    def nmea_checksum(body):
        """计算 NMEA 校验和（$ 与 * 之间所有字符的异或），返回两位大写十六进制"""
        c = 0
        for ch in body[1:]:              # 跳过开头的 '$'
            c ^= ord(ch)
        return f'{c:02X}'

    @staticmethod
    def _verify_checksum(msg, body, checksum):
        """校验和判定：True 通过 / False 不符 / None 无法判定

        三态而不是两态，是因为"判不了"和"判错"是两件事：
          * 只有 ``$`` 开头且校验和恰为两位十六进制的句子，才符合
            NMEA 标准的异或校验，能判定真伪；
          * ``#`` 开头的私有报文用的是 32 位 CRC，
            8 位十六进制无法用异或复现 —— 返回 None，不能算失败；
          * 根本没带校验和的句子同样返回 None（未提供，不是错）。

        判定结果只做标记，不丢弃句子 —— 坏句子也要留下痕迹，
        否则质量报告里"少了多少条"永远查不出来。
        """
        if not checksum or len(checksum) != 2:
            return None
        if not msg.startswith('$'):
            return None
        return NMEAParser.nmea_checksum(body) == checksum.upper()

    @staticmethod
    def repair_interleaved(seg):
        """修复被二进制调试帧从中间打断的 NMEA 句子

        问题现象
        --------
        实测这批 .dbg 里，接收机的二进制调试帧会**插在 NMEA 句子中间**::

            $GPRMC,093259.00,A,3109.9646839,N,12117>>>\\x00"\\x00\\xEF\\xBF\\xBD\\x00\\x03\\x00.4009231,E,000.002,320.6,310726,0.0,W,D*2F
                                                     ^^^ 二进制帧插在这里，把句子劈成两半

        这样的片段可打印率只有 0.91，低于 0.95 的过滤阈值，会被当成噪声
        整条丢掉。实测影响很大：I5 丢了 403/907 条 RMC、120/907 条 GST，
        而 GGA 只丢 1/1088（GGA 恰好在帧边界上，很少被插）。

        修复思路
        --------
        垃圾是**一段连续字节**：起点已知（第一个 ``>>>``），终点未知。
        用 NMEA 校验和反推终点 —— 从头往后枚举每个候选切点，重建句子，
        谁的校验和与字段数同时对上，谁就是正确切点。

        三道关卡缺一不可：
          1. **必须全部是可打印 ASCII**。这条不能省 —— NUL(0x00) 异或
             0x00 等于没异或，所以只靠校验和会把残留的 \\x00 放过去，
             恢复出 ``...12117\\x00.4009231...`` 这种字段被污染的结果。
          2. 字段数必须精确匹配（GGA 15 / RMC 13 / GST 9 / GSA 18~19）
          3. 校验和必须一致

        校验和只有 8 位，单靠它随机命中率 1/256；叠加前两条后误判概率
        可以忽略。修复失败一律返回 None，绝不返回可疑结果。
        修复后仍会走正常的校验和验证流程，不绕过任何既有校验。
        """
        if '>>>' not in seg or '*' not in seg:
            return None
        # 句子必须正好以 *XX 收尾。若校验和后面还跟着二进制，说明这条
        # 修复不了，直接放弃 —— 宁可不修，也不要猜。
        m = re.fullmatch(r'(.*\*)([0-9A-Fa-f]{2})', seg, re.S)
        if not m:
            return None
        head, ck = m.group(1), m.group(2)
        cut = head.find('>>>')
        if cut < 0:
            return None
        dollar = head.rfind('$', 0, cut)
        if dollar < 0:
            return None
        mtype = head[dollar + 3:dollar + 6]
        want = NMEAParser._FIELD_COUNT.get(mtype)
        if want is None:
            return None
        prefix = head[:cut]
        for r in range(cut + 3, len(head) + 1):
            cand = prefix + head[r:] + ck
            if any(not (32 <= ord(c) <= 126) for c in cand):
                continue
            if cand.count(',') + 1 not in want:
                continue
            # cand 形如 '$GPRMC,...,D*2F'，取 '-3' 去掉 '*' 与两位校验和，
            # checksum 只对 '$' 之后到 '*' 之前的字符做异或
            if NMEAParser.nmea_checksum(cand[:-3]) == ck.upper():
                return cand
        return None

    @staticmethod
    def _reject_reason(msg):
        """parse_nmea_message 返回 None 时归类原因，供完整性统计

        只做归类，不做二次判定 —— 真正的解析决策仍然只有
        parse_nmea_message 一处，避免两边口径漂移。
        """
        body = msg.rsplit('*', 1)[0] if '*' in msg else msg
        if '*' in msg:
            ck = msg.rsplit('*', 1)[1].strip()
            if ck and (len(ck) > 8
                       or not all(c in '0123456789ABCDEFabcdef' for c in ck)):
                return 'bad_checksum_format'
        if msg.startswith('$'):
            if len(body) >= 6 and body[3:6] not in NMEAParser.KNOWN_NMEA_TYPES:
                return 'unsupported_type'
        elif msg.startswith('#') and ',' in body:
            full_type = body[1:body.find(',')]
            if full_type not in NMEAParser.KNOWN_NMEA_TYPES:
                return 'unsupported_type'
        return 'other'

    @staticmethod
    def parse_nmea_message(raw_msg, line_no=0):
        """解析单条NMEA报文"""
        msg = raw_msg.strip()
        if not msg:
            return None

        # 分离校验和
        if '*' in msg:
            body, checksum_str = msg.rsplit('*', 1)
            checksum = checksum_str.strip()
            # 验证校验和格式（NMEA标准2位，NovAtel等可能有8位CRC）
            if not all(c in '0123456789ABCDEFabcdef' for c in checksum):
                if len(checksum) > 0:
                    return None
            elif len(checksum) > 8:
                return None
        else:
            body = msg
            checksum = ''

        checksum_ok = NMEAParser._verify_checksum(msg, body, checksum)

        # 提取报文类型
        if msg.startswith('$'):
            talker = body[1:3] if len(body) >= 3 else ''
            msg_type = body[3:6] if len(body) >= 6 else ''
            full_type = body[1:6] if len(body) >= 6 else ''
        elif msg.startswith('#'):
            end = body.find(',')
            full_type = body[1:end] if end > 0 else body[1:]
            talker = ''
            msg_type = full_type
        else:
            return None

        # 报文类型白名单检查
        if msg_type not in NMEAParser.KNOWN_NMEA_TYPES:
            return None

        fields = body.split(',')
        parsed_data = {'talker': talker, 'type': msg_type, 'full_type': full_type,
                       'field_count': len(fields)}

        # 门槛按"标准允许的最短形式"给，而不是标准条数：
        # GGA 的 station ID 可省略（14 字段）、老版 RMC 没有 mode 指示符
        # （12 字段）。卡太死会让合法短报文静默降级成"只有原始字段"，
        # 在统计里表现为"这条报文没有坐标"，而且查不出原因。
        # 各 _parse_* 内部另有 len(fields) > n 的取值保护，不会越界。
        if msg_type == 'GGA' and len(fields) >= 14:
            parsed_data.update(NMEAParser._parse_gga(fields))
        elif msg_type == 'GSA' and len(fields) >= 18:
            parsed_data.update(NMEAParser._parse_gsa(fields))
        elif msg_type == 'GST' and len(fields) >= 9:
            parsed_data.update(NMEAParser._parse_gst(fields))
        elif msg_type == 'RMC' and len(fields) >= 12:
            parsed_data.update(NMEAParser._parse_rmc(fields))
        elif msg_type == 'GSV' and len(fields) >= 4:
            parsed_data.update(NMEAParser._parse_gsv(fields, talker))
        elif msg_type == 'REFSTATIONA':
            parsed_data.update(NMEAParser._parse_refstation(fields))
        elif full_type in ('SATSINFOA', 'SATSINFOB'):
            parsed_data.update(NMEAParser._parse_satsinfo(body))
        else:
            parsed_data['fields'] = fields

        return {
            'type': msg_type if msg_type else full_type,
            'full_type': full_type,
            'raw': msg,
            'parsed': parsed_data,
            'line_no': line_no,
            'checksum': checksum,
            'checksum_ok': checksum_ok,
            'repaired': False,
        }

    @staticmethod
    def _parse_gga(fields):
        """解析GGA报文"""
        try:
            lat = NMEAParser._parse_latlon(fields[2], fields[3]) if len(fields) > 3 else None
            lon = NMEAParser._parse_latlon(fields[4], fields[5], is_lon=True) if len(fields) > 5 else None
            alt = float(fields[9]) if len(fields) > 9 and fields[9] else None
            fix_quality = int(fields[6]) if len(fields) > 6 and fields[6] else None
            num_sats = int(fields[7]) if len(fields) > 7 and fields[7] else None
            hdop = float(fields[8]) if len(fields) > 8 and fields[8] else None
            geoid_sep = float(fields[11]) if len(fields) > 11 and fields[11] else None
            age = float(fields[13]) if len(fields) > 13 and fields[13] else None
            station_id = fields[14] if len(fields) > 14 else ''

            utc_time = fields[1] if len(fields) > 1 else ''
            time_str = ''
            if utc_time and len(utc_time) >= 6:
                time_str = f"{utc_time[:2]}:{utc_time[2:4]}:{utc_time[4:6]}"

            return {
                'utc_time': utc_time, 'time_str': time_str,
                'latitude': lat, 'longitude': lon, 'altitude': alt,
                'fix_quality': fix_quality, 'num_sats': num_sats,
                'hdop': hdop, 'geoid_sep': geoid_sep, 'age': age,
                'station_id': station_id,
                'lat_dir': fields[3] if len(fields) > 3 else '',
                'lon_dir': fields[5] if len(fields) > 5 else ''
            }
        except (ValueError, IndexError) as e:
            return {'parse_error': str(e), 'fields': fields}

    @staticmethod
    def _parse_gsa(fields):
        """解析GSA报文"""
        try:
            mode = fields[1] if len(fields) > 1 else ''
            fix_type = int(fields[2]) if len(fields) > 2 and fields[2] else None
            sats = [f for f in fields[3:15] if f]
            pdop = float(fields[15]) if len(fields) > 15 and fields[15] else None
            hdop = float(fields[16]) if len(fields) > 16 and fields[16] else None
            vdop = float(fields[17]) if len(fields) > 17 and fields[17] else None
            return {'mode': mode, 'fix_type': fix_type, 'sats_used': sats,
                    'pdop': pdop, 'hdop': hdop, 'vdop': vdop}
        except (ValueError, IndexError) as e:
            return {'parse_error': str(e)}

    @staticmethod
    def _parse_gst(fields):
        """解析GST报文"""
        try:
            return {
                'utc_time': fields[1] if len(fields) > 1 else '',
                'rms': float(fields[2]) if len(fields) > 2 and fields[2] else None,
                'semi_major': float(fields[3]) if len(fields) > 3 and fields[3] else None,
                'semi_minor': float(fields[4]) if len(fields) > 4 and fields[4] else None,
                'orientation': float(fields[5]) if len(fields) > 5 and fields[5] else None,
                'lat_error': float(fields[6]) if len(fields) > 6 and fields[6] else None,
                'lon_error': float(fields[7]) if len(fields) > 7 and fields[7] else None,
                'alt_error': float(fields[8]) if len(fields) > 8 and fields[8] else None,
            }
        except (ValueError, IndexError) as e:
            return {'parse_error': str(e)}

    @staticmethod
    def _parse_gsv(fields, talker=''):
        """解析GSV报文（在视卫星数 + 每颗卫星的PRN/仰角/方位角/SNR）

        GSV 每帧最多携带4颗卫星，字段偏移为 4+4i：卫星号/仰角/方位角/SNR。
        SNR 为空表示该星未锁定（不参与解算）。
        """
        try:
            num_msgs = int(fields[1]) if len(fields) > 1 and fields[1] else None
            msg_idx = int(fields[2]) if len(fields) > 2 and fields[2] else None
            sats_in_view = int(fields[3]) if len(fields) > 3 and fields[3] else None
            constellation = NMEAParser.TALKER_CONSTELLATION.get(talker, talker or '未知')

            satellites = []
            for g in range(4):
                base = 4 + g * 4
                if len(fields) >= base + 4 and fields[base]:
                    satellites.append({
                        'prn': fields[base],
                        'elevation': float(fields[base + 1]) if fields[base + 1] else None,
                        'azimuth': float(fields[base + 2]) if fields[base + 2] else None,
                        'snr': float(fields[base + 3]) if fields[base + 3] else None,
                        'constellation': constellation,
                    })

            return {
                'num_msgs': num_msgs,
                'msg_idx': msg_idx,
                'sats_in_view': sats_in_view,
                'satellites': satellites,
            }
        except (ValueError, IndexError) as e:
            return {'parse_error': str(e), 'fields': fields}

    @staticmethod
    def _parse_rmc(fields):
        """解析RMC报文"""
        try:
            lat = NMEAParser._parse_latlon(fields[3], fields[4]) if len(fields) > 4 else None
            lon = NMEAParser._parse_latlon(fields[5], fields[6], is_lon=True) if len(fields) > 6 else None
            return {
                'utc_time': fields[1] if len(fields) > 1 else '',
                'status': fields[2] if len(fields) > 2 else '',
                'latitude': lat, 'longitude': lon,
                'speed_knots': float(fields[7]) if len(fields) > 7 and fields[7] else None,
                'course': float(fields[8]) if len(fields) > 8 and fields[8] else None,
                'date': fields[9] if len(fields) > 9 else '',
                'mag_var': float(fields[10]) if len(fields) > 10 and fields[10] else None,
            }
        except (ValueError, IndexError) as e:
            return {'parse_error': str(e)}

    @staticmethod
    def _parse_refstation(fields):
        """解析REFSTATIONA报文"""
        try:
            return {
                'station_name': fields[1] if len(fields) > 1 else '',
                'port': fields[2] if len(fields) > 2 else '',
                'flags': fields[3] if len(fields) > 3 else '',
                'cpu_load': float(fields[4]) if len(fields) > 4 and fields[4] else None,
                'mode': fields[5] if len(fields) > 5 else '',
                'gps_week': int(fields[6]) if len(fields) > 6 and fields[6] else None,
                'gps_ms': float(fields[7]) if len(fields) > 7 and fields[7] else None,
                'status': fields[8] if len(fields) > 8 else '',
            }
        except (ValueError, IndexError) as e:
            return {'parse_error': str(e)}

    @staticmethod
    def _parse_satsinfo(body):
        """解析扩展 SATSINFO 报文

        SATSINFO 包含板卡跟踪到的全部卫星信息：卫星PRN、方位角、高度角，
        以及不同频点的信噪比。支持 ASCII(#SATSINFOA) 与二进制(#SATSINFOB) 两种形式。

        报文结构：
          载荷头: Sat number, Version, reserve×3, Frq flag
          逐卫星: PRN, Azimuth(°), Elevation(°), 然后 F×(Sys, SNR, FreqStatus, FreqNo)
        F 由每颗卫星首个频点的 FreqNo 字段给出（各频点组内 FreqNo 相同）。
        """
        body = body.lstrip('#').strip()
        comma = body.find(',')
        if comma <= 0:
            return {'parse_error': '报文头格式错误'}
        name = body[:comma]
        rest = body[comma + 1:]
        if name.endswith('B'):
            return NMEAParser._parse_satsinfo_binary(rest)
        return NMEAParser._parse_satsinfo_ascii(rest)

    @staticmethod
    def _parse_satsinfo_ascii(rest):
        """解析 ASCII 形式 SATSINFO：#SATSINFOA,<通用ASCII头9字段>;<卫星数据...>*CRC"""
        try:
            header, payload = rest.split(';', 1)
        except ValueError:
            return {'parse_error': '缺失分号分隔符，无法拆分载荷'}
        hf = header.split(',') if header else []

        def try_int(v, default=None):
            try:
                return int(v)
            except (TypeError, ValueError):
                return default

        # ASCII 消息头: CPUIdle,TimeRef,TimeStatus,Wn,Ms,reserved,version,Leapsec,OutputDelay
        info = {
            'cpuidle': try_int(hf[0]) if len(hf) > 0 else None,
            'time_ref': hf[1] if len(hf) > 1 else '',
            'time_status': hf[2] if len(hf) > 2 else '',
            'week': try_int(hf[3]) if len(hf) > 3 else None,
            'ms': try_int(hf[4]) if len(hf) > 4 else None,
        }

        vals = payload.split(',')
        i = 0
        n = len(vals)

        def next_int():
            nonlocal i
            if i >= n:
                raise IndexError
            v = int(vals[i])
            i += 1
            return v

        try:
            sat_number = next_int()          # 载荷头 Sat number
            version = next_int()             # Version
            for _ in range(3):               # reserve×3
                next_int()
            frq_flag = next_int()            # Frq flag
        except (IndexError, ValueError):
            return {'parse_error': 'SATSINFO 载荷头解析失败', **info}

        satellites = []
        for _ in range(sat_number):
            try:
                prn = next_int()
                azimuth = float(next_int())
                elevation = float(next_int())
            except (IndexError, ValueError):
                break

            freqs = []
            expected = None
            while True:
                if i + 3 >= n:
                    break
                try:
                    sys_id = next_int()
                    snr = next_int()
                    freq_status = next_int()
                    freq_no = next_int()
                except (IndexError, ValueError):
                    break
                freqs.append({'sys': sys_id, 'snr': snr, 'freq_status': freq_status})
                if expected is None and freq_no > 0:
                    expected = freq_no
                if expected is not None and len(freqs) >= expected:
                    break

            satellites.append({
                'prn': prn,
                'azimuth': azimuth,
                'elevation': elevation,
                'freqs': freqs,
            })

        info.update({
            'sat_number': sat_number,
            'version': version,
            'frq_flag': frq_flag,
            'satellites': satellites,
        })
        return info

    @staticmethod
    def _parse_satsinfo_binary(rest):
        """解析二进制形式 SATSINFO：#SATSINFOB,<十六进制载荷>*CRC

        载荷可能带 24 字节厂商消息头，
        也可能为纯载荷，自动识别。
        """
        hex_str = rest.strip()
        try:
            data = bytes.fromhex(hex_str)
        except ValueError:
            return {'parse_error': '十六进制数据非法'}

        # 识别 24 字节消息头
        if len(data) >= 3 and data[0:3] == b'\xaa\x44\xb5':
            off = 24
        else:
            off = 0

        if off + 6 > len(data):
            return {'parse_error': '载荷过短'}

        def u8(idx):
            return data[idx] if idx < len(data) else None

        sat_number = u8(off)
        version = u8(off + 1)
        frq_flag = u8(off + 5)

        satellites = []
        p = off + 6
        for _ in range(sat_number if sat_number else 0):
            if p + 4 > len(data):
                break
            prn = data[p]
            azimuth = int.from_bytes(data[p + 1:p + 3], 'little', signed=True)
            elevation = data[p + 3]
            p += 4
            if p + 4 > len(data):
                break
            # 首个频点组的 FreqNo 给出本卫星频点数量
            sys_id = u8(p)
            snr = u8(p + 1)
            freq_status = u8(p + 2)
            freq_no = u8(p + 3)
            p += 4
            freqs = [{'sys': sys_id, 'snr': snr, 'freq_status': freq_status}]
            if freq_no:
                for _ in range(freq_no - 1):
                    if p + 4 > len(data):
                        break
                    freqs.append({'sys': u8(p), 'snr': u8(p + 1),
                                  'freq_status': u8(p + 2)})
                    p += 4
            satellites.append({
                'prn': prn,
                'azimuth': azimuth,
                'elevation': elevation,
                'freqs': freqs,
            })

        return {
            'sat_number': sat_number,
            'version': version,
            'frq_flag': frq_flag,
            'satellites': satellites,
        }

    @staticmethod
    def _parse_latlon(value, direction, is_lon=False):
        """将NMEA度分格式转换为度"""
        if not value:
            return None
        try:
            if is_lon:
                degrees = int(value[:3])
                minutes = float(value[3:])
            else:
                degrees = int(value[:2])
                minutes = float(value[2:])
            result = degrees + minutes / 60.0
            if direction in ['S', 'W']:
                result = -result
            return result
        except (ValueError, IndexError):
            return None
