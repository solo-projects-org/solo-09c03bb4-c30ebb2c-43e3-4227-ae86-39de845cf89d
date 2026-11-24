# -*- coding: utf-8 -*-
"""
四叉树分区核心算法。

约定：
- 网格边长 N 为 2 的幂，行、列号均为 0 基（界面对 1 基做转换）。
- 一个方块（节点）合法，当且仅当：
    1) 方块内不含未测格（空白格）；
    2) 方块内最高温与任一读数之差 <= 允许温差 tol
       （整数温度下等价于 max-min <= tol）。
- 合法性是“遗传”的：合法方块的任意子方块也合法（子区域不增未测格，温差不扩大）。
  因此最优（块数最少、块间不重叠）覆盖就是：取所有“最大合法节点”——
  自身合法、但父节点不合法的四叉树节点，外加根节点自身合法的退化情形。
- 任何不重叠方块覆盖都不可能跨越四叉树边界（归纳可证），
  所以每个最大合法节点至少需要一个块来覆盖，本方案每节点恰用一块，块数最少。

本模块只做纯计算，不接触数据库，便于单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Cell = Optional[int]          # None 表示未测；否则为整数摄氏度
Grid = Sequence[Sequence[Cell]]
Coord = Tuple[int, int, int]   # (row, col, size)，行、列均为 0 基
NodeStat = Tuple[int, int, int, int]
# (measured_count, unmeasured_count, temp_min, temp_max)
# 无实测格时 temp_min/temp_max 用 None，存储层以 NULL 表示。


def is_power_of_two(n: int) -> bool:
    return isinstance(n, int) and n >= 1 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class Block:
    """一块最终覆盖方块。row/col 为 0 基左上角，size 为边长。"""
    row: int
    col: int
    size: int
    tmax: int
    tmin: int
    count: int  # 覆盖的实测格数（合法块内全部为实测格，等于 size*size）


def _merge_stats(a: NodeStat, b: NodeStat) -> NodeStat:
    am, au, amin, amax = a
    bm, bu, bmin, bmax = b
    measured = am + bm
    unmeasured = au + bu
    if amin is None:
        tmin, tmax = bmin, bmax
    elif bmin is None:
        tmin, tmax = amin, amax
    else:
        tmin = min(amin, bmin)
        tmax = max(amax, bmax)
    return measured, unmeasured, tmin, tmax


def stat_is_valid(stat: NodeStat, tol: int) -> bool:
    """节点是否可作为覆盖方块：无未测格且温差不超限。"""
    measured, unmeasured, tmin, tmax = stat
    if measured == 0 or unmeasured != 0:
        return False
    return tmax - tmin <= tol


def compute_all_node_stats(cells: Dict[Tuple[int, int], Cell], n: int) -> Dict[Coord, NodeStat]:
    """
    自底向上计算整张四叉树所有内部节点（size >= 2）的统计值。

    cells: {(row, col): 温度整数或 None}，未出现的键按未测处理。
    返回 {(row, col, size): (measured, unmeasured, tmin, tmax)}。
    """
    if not is_power_of_two(n):
        raise ValueError("网格边长必须是 2 的幂")
    nodes: Dict[Coord, NodeStat] = {}
    if n == 1:
        return nodes

    # size=1 层不入库，直接由格子构成；逐层向上合并。
    size = 2
    while size <= n:
        half = size // 2
        for r in range(0, n, size):
            for c in range(0, n, size):
                if half == 1:
                    children = []
                    for dr in (0, 1):
                        for dc in (0, 1):
                            v = cells.get((r + dr, c + dc))
                            if v is None:
                                children.append((0, 1, None, None))
                            else:
                                children.append((1, 0, v, v))
                else:
                    children = [
                        nodes[(r, c, half)],
                        nodes[(r, c + half, half)],
                        nodes[(r + half, c, half)],
                        nodes[(r + half, c + half, half)],
                    ]
                stat = children[0]
                for child in children[1:]:
                    stat = _merge_stats(stat, child)
                nodes[(r, c, size)] = stat
        size *= 2
    return nodes


def collect_blocks(nodes: Dict[Coord, NodeStat], cells: Dict[Tuple[int, int], Cell],
                   n: int, tol: int) -> List[Block]:
    """
    自上而下取“最大合法节点”，得到最少块数的不重叠覆盖。

    叶子（size=1）为实测格时本身视为合法（单格温差为 0）。
    只覆盖实测格，未测格不产出任何块。
    """
    blocks: List[Block] = []

    def visit(r: int, c: int, size: int) -> None:
        if size == 1:
            v = cells.get((r, c))
            if v is not None:
                blocks.append(Block(r, c, 1, v, v, 1))
            return
        stat = nodes[(r, c, size)]
        if stat_is_valid(stat, tol):
            _, _, tmin, tmax = stat
            blocks.append(Block(r, c, size, tmax, tmin, stat[0]))
            return
        half = size // 2
        visit(r, c, half)
        visit(r, c + half, half)
        visit(r + half, c, half)
        visit(r + half, c + half, half)

    visit(0, 0, n)
    return blocks


def ancestor_nodes(r: int, c: int, n: int, min_size: int = 2) -> List[Coord]:
    """
    返回叶子格 (r,c) 的全部四叉树祖先节点，按 size 升序排列。
    每个层级上的节点左上角坐标由对齐得到。
    """
    out: List[Coord] = []
    size = 2
    while size <= n:
        out.append(((r // size) * size, (c // size) * size, size))
        size *= 2
    return out


def children_of(node: Coord) -> List[Coord]:
    r, c, size = node
    half = size // 2
    if half == 1:
        # size=1 叶子不是节点表成员，用特殊约定返回四个叶子坐标（size=1）
        return [(r + dr, c + dc, 1) for dr in (0, 1) for dc in (0, 1)]
    return [
        (r, c, half),
        (r, c + half, half),
        (r + half, c, half),
        (r + half, c + half, half),
    ]
