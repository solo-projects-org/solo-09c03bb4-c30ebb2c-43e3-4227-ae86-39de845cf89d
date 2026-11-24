# -*- coding: utf-8 -*-
"""
热像仪补测表对位（纯算法，不接触数据库）。

背景
----
补测表只保留了**局部行列号**（热像仪自身坐标系，1 基整数），丢失了
它在炉壁格网上的位置与朝向。需要在用户给定的搜索区矩形内，枚举
- 平移：整组点的局部坐标加同一个 (dr, dc) 平移量；
- 朝向：0° / 90° / 180° / 270°（顺时针），**不做镜像**。

摆法（placement）可行的充要条件
------------------------------
1. 整组点（参照点 + 补测点）旋转、平移后全部落在搜索区矩形内；
2. 每个参照点落点处的**已有实测温度**与表中整数温度完全相等。

注意：补测点的目标格是否已测**不参与**摆法筛选——已测目标格只在
确认整批填入时拦截，不能因此排除摆法（验收硬性要求）。

预览唯一性
----------
不同可行摆法可能产生同一组补测落点（同一坐标 → 同一值）。只有
**所有**可行摆法给出的补测坐标+值集合完全相同时，才预览这组落点；
否则报“多组歧义”。无解时报具体原因（越界 / 参照不吻合）。

坐标约定：外部（炉壁格网）一律 0 基 (row, col)；补测表局部坐标 1 基。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

# 补测表最大点数，防止异常大文件触发海量枚举
MAX_SHEET_POINTS = 20000

# 用途列的可接受取值（小写比较）
REF_TOKENS = {"ref", "reference", "参照", "参照点", "参考", "参考点"}
FILL_TOKENS = {"fill", "remeasure", "re-measure", "measure",
               "补测", "补测点", "复测", "复测点"}

Point = Tuple[int, int]                  # (row, col)，局部表内 1 基
Landing = Tuple[int, int, int]           # (row, col, temp)，格网 0 基
FillKey = Tuple[int, int, int]           # (row, col, temp)：补测落点 + 值


class SheetError(ValueError):
    """补测表 CSV 结构/内容错误。"""


@dataclass(frozen=True)
class SheetPoint:
    """补测表中的一个点。"""
    local_row: int
    local_col: int
    temp: int
    is_ref: bool          # True=参照点，False=补测点


@dataclass(frozen=True)
class Placement:
    """一种可行摆法。"""
    orientation: int                       # 0 / 90 / 180 / 270
    translate: Tuple[int, int]             # (dr, dc)，加到归一化后坐标上
    ref_landings: Tuple[Landing, ...]      # 参照落点 (row,col,temp)
    fill_landings: Tuple[Landing, ...]     # 补测落点 (row,col,temp)

    def fill_set(self) -> FrozenSet[FillKey]:
        return frozenset(self.fill_landings)


@dataclass
class SolveResult:
    """对位求解结果。"""
    status: str                            # "unique" / "ambiguous" / "none"
    placements: List[Placement] = field(default_factory=list)
    # unique 时的唯一补测落点（含值），其余状态为空
    fill_landings: Tuple[Landing, ...] = ()
    ref_landings: Tuple[Landing, ...] = ()
    distinct_fill_count: int = 0
    no_solution_reason: Optional[str] = None


# --------------------------- 旋转（无镜像） ---------------------------

def rotate_points(points: Sequence[Tuple[int, int]],
                  orientation: int) -> List[Point]:
    """
    把局部 1 基坐标点集按 orientation（0/90/180/270，顺时针）旋转。

    旋转后重新归一化，使最小行、最小列都回到 1：等价于以点集包围盒
    左上角为锚点旋转，平移量在求解阶段统一枚举。返回顺序与输入逐点对应。
    """
    if orientation == 0:
        rotated = [(r, c) for (r, c) in points]
    elif orientation == 90:
        # 顺时针 90°：(r, c) -> (c, -r)
        rotated = [(c, -r) for (r, c) in points]
    elif orientation == 180:
        rotated = [(-r, -c) for (r, c) in points]
    elif orientation == 270:
        # 顺时针 270°：(r, c) -> (-c, r)
        rotated = [(-c, r) for (r, c) in points]
    else:
        raise ValueError(f"不支持的朝向 {orientation}")
    min_r = min(p[0] for p in rotated)
    min_c = min(p[1] for p in rotated)
    # 归一化到 1 基，保持输入顺序
    return [(r - min_r + 1, c - min_c + 1) for (r, c) in rotated]


ORIENTATIONS = (0, 90, 180, 270)


# ----------------------------- 求解 -----------------------------

def solve(points: Sequence[SheetPoint],
          region: Tuple[int, int, int, int],
          cells: Dict[Point, Optional[int]]) -> SolveResult:
    """
    在搜索区内枚举全部可行摆法并判定补测落点是否唯一。

    points  : 补测表点集（至少含 1 个补测点）
    region  : (r0, r1, c0, c1)，格网 0 基闭区间搜索区
    cells   : 整张格网 {(row,col): 温度整数或 None(未测)}
    """
    r0, r1, c0, c1 = region
    width_r = r1 - r0 + 1
    width_c = c1 - c0 + 1

    # 分类记录越界与参照不吻合，用于无解时给出原因
    any_fits_region = False
    any_ref_mismatch = False
    placements: List[Placement] = []
    seen: Set[Tuple[int, int, int]] = set()

    local = [(p.local_row, p.local_col) for p in points]

    for orientation in ORIENTATIONS:
        rotated_all = rotate_points(local, orientation)
        # rotate_points 按输入顺序返回；字典兜底，避免重复局部坐标在
        # 归一化后点序重排，导致参照/补测属性错位到别的落点。
        rot_by_local: Dict[Point, Point] = dict(zip(local, rotated_all))
        ext_r = max(p[0] for p in rotated_all)       # 归一化后最大 1 基行
        ext_c = max(p[1] for p in rotated_all)
        # 平移量（0 基格网）：归一化点 (lr,lc) -> (r0+dr+lr-1, c0+dc+lc-1)
        max_dr = width_r - ext_r
        max_dc = width_c - ext_c
        if max_dr < 0 or max_dc < 0:
            continue                                 # 该朝向下点集比搜索区还大
        for dr in range(max_dr + 1):
            for dc in range(max_dc + 1):
                region_ok = True
                landed: List[Tuple[SheetPoint, int, int]] = []
                for sp in points:
                    lr, lc = rot_by_local[(sp.local_row, sp.local_col)]
                    gr = r0 + dr + lr - 1
                    gc = c0 + dc + lc - 1
                    if not (r0 <= gr <= r1 and c0 <= gc <= c1):
                        region_ok = False
                        break
                    landed.append((sp, gr, gc))
                if not region_ok:
                    continue
                any_fits_region = True
                # 参照点必须与已有实测吻合；补测目标格是否已测不参与筛选
                ref_landings: List[Landing] = []
                fill_landings: List[Landing] = []
                ref_ok = True
                for sp, gr, gc in landed:
                    if sp.is_ref:
                        actual = cells.get((gr, gc))
                        if actual is None or actual != sp.temp:
                            ref_ok = False
                            break
                        ref_landings.append((gr, gc, sp.temp))
                    else:
                        fill_landings.append((gr, gc, sp.temp))
                if not ref_ok:
                    any_ref_mismatch = True
                    continue
                key = (orientation, dr, dc)
                if key in seen:
                    continue
                seen.add(key)
                placements.append(Placement(
                    orientation=orientation, translate=(dr, dc),
                    ref_landings=tuple(sorted(ref_landings)),
                    fill_landings=tuple(sorted(fill_landings)),
                ))

    if not placements:
        if not any_fits_region:
            reason = ("没有可行摆法：补测表点集在任何允许的朝向下都无法"
                      "整体落入搜索区（点集包围盒大于搜索区或搜索区为空）")
        elif any_ref_mismatch:
            reason = ("没有可行摆法：所有能落入搜索区的摆法中，参照点温度"
                      "都与已有实测不吻合（或参照落点当前未测）")
        else:
            reason = "没有可行摆法"
        return SolveResult(status="none", no_solution_reason=reason)

    distinct: Dict[FrozenSet[FillKey], Tuple[Landing, ...]] = {}
    for pl in placements:
        distinct.setdefault(pl.fill_set(), pl.fill_landings)

    if len(distinct) > 1:
        return SolveResult(status="ambiguous", placements=placements,
                           distinct_fill_count=len(distinct))

    fills = next(iter(distinct.values()))
    # 预览展示首个摆法的参照落点；确认重核以重新求解的结果为准
    return SolveResult(
        status="unique", placements=placements,
        fill_landings=fills,
        ref_landings=placements[0].ref_landings,
        distinct_fill_count=1,
    )


def validate_region(region: Tuple[int, int, int, int],
                    n: int) -> Tuple[int, int, int, int]:
    """校验搜索区：0 基闭区间，落在 n×n 格网内且起止有序。"""
    r0, r1, c0, c1 = region
    for v, name in ((r0, "起始行"), (r1, "结束行"),
                    (c0, "起始列"), (c1, "结束列")):
        if not (0 <= v < n):
            raise SheetError(f"{name} {v + 1} 超出格网范围 1..{n}")
    if r1 < r0 or c1 < c0:
        raise SheetError("搜索区起止顺序有误：结束行/列不能小于起始行/列")
    return region
