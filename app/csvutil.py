# -*- coding: utf-8 -*-
"""方阵 CSV 解析。

规则（对应验收要求）：
- 第一行可为表头：若首单元格（忽略大小写）为 row/行 等关键字，则整行跳过。
- 每行字段数必须完全一致，且行数 == 列数 == 2 的幂。
- 字段为空（含纯空白）表示该格未测；否则必须是整数摄氏度。
- 温度范围做合理性限制（可按工艺调整），防止误录。
"""

from __future__ import annotations

import csv
import io
from typing import List, Optional, Tuple

from .quadtree import is_power_of_two

TEMP_MIN = -100          # 摄氏度，工艺合理下限
TEMP_MAX = 3000          # 摄氏度，工艺合理上限
MAX_SIDE = 512           # 单表最大边长，控制节点表规模
HEADER_TOKENS = {"row", "r", "行", "行号", "rows"}


class CSVParseError(ValueError):
    pass


def parse_square_csv(raw: bytes) -> Tuple[int, List[List[Optional[int]]]]:
    """
    解析上传的方阵 CSV，返回 (边长, 二维数据)，元素为 int 或 None（未测）。
    """
    if not raw or not raw.strip():
        raise CSVParseError("文件为空")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gbk")
        except UnicodeDecodeError:
            raise CSVParseError("文件编码无法识别，请使用 UTF-8 或 GBK 编码的 CSV")

    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader]
    # 去掉文件末尾完全为空的行（常见于导出工具）
    while rows and all((cell is None or cell.strip() == "") for cell in rows[-1]):
        rows.pop()
    if not rows:
        raise CSVParseError("CSV 中没有数据行")

    # 可选表头：仅当首格为行号类关键字时整行跳过；不做隐式列数猜测，避免误伤数据行
    first = rows[0]
    if first and first[0].strip().lower() in HEADER_TOKENS:
        rows = rows[1:]

    if not rows:
        raise CSVParseError("去掉表头后没有数据")

    n = len(rows)
    if not is_power_of_two(n):
        raise CSVParseError(f"行数 {n} 不是 2 的幂（要求 1、2、4、8、…、{MAX_SIDE}）")
    if n > MAX_SIDE:
        raise CSVParseError(f"边长 {n} 超过上限 {MAX_SIDE}")

    width = len(rows[0])
    data: List[List[Optional[int]]] = []
    for i, row in enumerate(rows):
        if len(row) != width:
            raise CSVParseError(f"第 {i + 1} 行有 {len(row)} 列，与首行 {width} 列不一致")
        parsed_row: List[Optional[int]] = []
        for j, cell in enumerate(row):
            token = cell.strip()
            if token == "":
                parsed_row.append(None)
                continue
            try:
                value = int(token)
            except ValueError:
                raise CSVParseError(
                    f"第 {i + 1} 行第 {j + 1} 列 “{token}” 不是整数，"
                    "格内只能填整数摄氏度，留空表示未测"
                )
            if not (TEMP_MIN <= value <= TEMP_MAX):
                raise CSVParseError(
                    f"第 {i + 1} 行第 {j + 1} 列温度 {value} 超出合理范围 "
                    f"[{TEMP_MIN}, {TEMP_MAX}] ℃"
                )
            parsed_row.append(value)
        data.append(parsed_row)

    if width != n:
        raise CSVParseError(f"必须是方阵：当前 {n} 行 × {width} 列")
    if n == 0:
        raise CSVParseError("没有有效数据")
    return n, data


def blocks_to_csv(blocks, n: int) -> str:
    """
    把现行供数（块列表）导出为 CSV（一次性返回，适合中小网格）。
    行、列号采用界面一致的 1 基编号；字段：
    block_id, row_start, col_start, side, temp_max_c, temp_min_c, cell_count
    """
    return "".join(iter_blocks_csv(blocks, n))


def iter_blocks_csv(blocks, n: int):
    """流式生成供数 CSV，逐行 yield，避免大网格一次性占用内存。"""
    import io as _io
    header = ["block_id", "row_start", "col_start", "side",
              "temp_max_c", "temp_min_c", "cell_count", "grid_side"]
    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    yield buf.getvalue()
    for idx, b in enumerate(blocks, start=1):
        buf = _io.StringIO()
        csv.writer(buf).writerow([
            idx, b.row + 1, b.col + 1, b.size,
            b.tmax, b.tmin, b.count, n,
        ])
        yield buf.getvalue()
