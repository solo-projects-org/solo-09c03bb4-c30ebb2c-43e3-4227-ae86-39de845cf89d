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
from .alignment import (FILL_TOKENS, MAX_SHEET_POINTS, REF_TOKENS,
                        SheetError, SheetPoint)

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


# ----------------------- 热像仪补测表 CSV -----------------------
#
# 固定四列：local_row, local_col, temp, purpose
# （局部行、局部列均为 1 基正整数；温度为整数摄氏度；
#   purpose 为“参照/补测”类标记，中英文均可，见 alignment.REF_TOKENS）
ALIGN_HEADER_TOKENS = {"local_row", "row", "r", "行", "行号", "局部行"}


def parse_alignment_csv(raw: bytes) -> List[SheetPoint]:
    """
    解析热像仪补测表，返回 SheetPoint 列表。

    规则：
    - 恰好 4 列：局部行、局部列、整数温度、用途；
    - 局部行列必须是 >=1 的整数（补测表自身的 1 基编号）；
    - 温度为 [TEMP_MIN, TEMP_MAX] 内整数；
    - 用途只接受参照/补测两类标记；
    - 同一 (局部行, 局部列) 不得重复；至少要有一个补测点。
    """
    if not raw or not raw.strip():
        raise SheetError("补测表文件为空")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gbk")
        except UnicodeDecodeError:
            raise SheetError("文件编码无法识别，请使用 UTF-8 或 GBK 编码的 CSV")

    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader]
    while rows and all((cell is None or cell.strip() == "") for cell in rows[-1]):
        rows.pop()
    if not rows:
        raise SheetError("补测表中没有数据行")

    first = rows[0]
    if first and first[0].strip().lower() in ALIGN_HEADER_TOKENS:
        rows = rows[1:]
    if not rows:
        raise SheetError("去掉表头后补测表没有数据")

    points: List[SheetPoint] = []
    seen = set()
    for i, row in enumerate(rows):
        if len(row) != 4:
            raise SheetError(
                f"第 {i + 1} 行有 {len(row)} 列，补测表必须为 4 列："
                "局部行, 局部列, 温度, 用途（参照/补测）")
        r_token, c_token, t_token, p_token = (cell.strip() for cell in row)
        try:
            lr = int(r_token)
            lc = int(c_token)
        except ValueError:
            raise SheetError(
                f"第 {i + 1} 行的局部行列号 “{r_token},{c_token}” 不是整数")
        if lr < 1 or lc < 1:
            raise SheetError(
                f"第 {i + 1} 行的局部行列号必须 >=1（补测表为 1 基编号）")
        try:
            temp = int(t_token)
        except ValueError:
            raise SheetError(
                f"第 {i + 1} 行温度 “{t_token}” 不是整数，补测表只接受整数摄氏度")
        if not (TEMP_MIN <= temp <= TEMP_MAX):
            raise SheetError(
                f"第 {i + 1} 行温度 {temp} 超出合理范围 "
                f"[{TEMP_MIN}, {TEMP_MAX}] ℃")
        purpose = p_token.lower()
        if purpose in REF_TOKENS:
            is_ref = True
        elif purpose in FILL_TOKENS:
            is_ref = False
        else:
            raise SheetError(
                f"第 {i + 1} 行用途 “{p_token}” 无法识别，"
                "只能填“参照”或“补测”（也接受 ref/fill）")
        key = (lr, lc)
        if key in seen:
            raise SheetError(
                f"局部坐标 ({lr},{lc}) 在补测表中重复出现，每个点只能出现一次")
        seen.add(key)
        points.append(SheetPoint(lr, lc, temp, is_ref))

    if len(points) > MAX_SHEET_POINTS:
        raise SheetError(f"补测表点数 {len(points)} 超过上限 {MAX_SHEET_POINTS}")
    if not any(not p.is_ref for p in points):
        raise SheetError("补测表至少要包含一个“补测”用途的点，全是参照点无法补测")
    return points
