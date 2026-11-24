# -*- coding: utf-8 -*-
"""
SQLite 持久层与核算服务。

数据模型
--------
grid      : 一张方阵（一次上传一行），记录边长与当前允许温差
cells     : 每个格子一行，温度 NULL 表示未测
nodes     : 四叉树内部节点（size>=2）的统计与合法标志，供增量核算复用
batch     : 批次。上传/改温差/补测清空成功各生成一个批次，
            is_current=1 的批次对应“现行供数”；批次一经生效不可变
block_op  : 块变更日志（delta 模型）
            * 上传、改温差写一整套 op='base' 的现行块；
            * 补测/清空只在受影响子树写 op='add'（新块）与
              op='tombstone'（旧块作废），写入量与改动量成正比；
            * 现行块 = 每个块坐标 (row,col,side) 上批次号最大的一条操作，
              add 生效、tombstone 丢弃。下载与页面同源读取，保证
              位置、边长、温度来自同一批已生效数据。
event_log : 操作审计

增量核算（补测/清空）
--------------------
1. 取改动格的全部祖先节点；
2. 自底向上（size 升序）只重算这些节点（及其直接孩子）的统计与合法标志；
3. 自根向下收集块增量：未受影响子树整棵跳过（旧块继续生效），
   受影响路径上按新合法性写 add / tombstone；
4. 全流程在一个 BEGIN IMMEDIATE 事务内，异常回滚，旧批次与旧供数不变。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from . import alignment, csvutil, quadtree
from .alignment import SheetError
from .quadtree import Block

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS grid (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    side        INTEGER NOT NULL,
    tol         INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cells (
    grid_id  INTEGER NOT NULL REFERENCES grid(id),
    row_idx  INTEGER NOT NULL,
    col_idx  INTEGER NOT NULL,
    temp     INTEGER,            -- NULL = 未测
    PRIMARY KEY (grid_id, row_idx, col_idx)
);

CREATE TABLE IF NOT EXISTS nodes (
    grid_id        INTEGER NOT NULL REFERENCES grid(id),
    row_idx        INTEGER NOT NULL,
    col_idx        INTEGER NOT NULL,
    side           INTEGER NOT NULL,
    measured_cnt   INTEGER NOT NULL,
    unmeasured_cnt INTEGER NOT NULL,
    temp_min       INTEGER,
    temp_max       INTEGER,
    is_valid       INTEGER NOT NULL,   -- 1=可作为覆盖方块
    PRIMARY KEY (grid_id, row_idx, col_idx, side)
);

CREATE TABLE IF NOT EXISTS batch (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    grid_id     INTEGER NOT NULL REFERENCES grid(id),
    reason      TEXT NOT NULL,        -- upload / tolerance / edit
    is_current  INTEGER NOT NULL DEFAULT 0,
    tol         INTEGER NOT NULL,
    block_count INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS block_op (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id   INTEGER NOT NULL REFERENCES batch(id),
    row_idx    INTEGER NOT NULL,
    col_idx    INTEGER NOT NULL,
    side       INTEGER NOT NULL,
    op         TEXT NOT NULL CHECK (op IN ('base','add','tombstone')),
    temp_max   INTEGER,
    temp_min   INTEGER,
    cell_count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_block_op_batch ON block_op(batch_id);
CREATE INDEX IF NOT EXISTS idx_block_op_pos ON block_op(row_idx, col_idx, side);

-- 现行块物化表：与 block_op 在同一事务内维护，下载/页面直接顺序读取，
-- 无需跨批次还原；表内只保留 is_current 批次对应的块，天然同源。
CREATE TABLE IF NOT EXISTS blocks_current (
    row_idx    INTEGER NOT NULL,
    col_idx    INTEGER NOT NULL,
    side       INTEGER NOT NULL,
    temp_max   INTEGER NOT NULL,
    temp_min   INTEGER NOT NULL,
    cell_count INTEGER NOT NULL,
    PRIMARY KEY (row_idx, col_idx, side)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS event_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    kind       TEXT NOT NULL,
    detail     TEXT NOT NULL
);

-- 热像仪补测表对位：同一时刻只保留一份预览（重传底图/新预览会整表替换）。
-- 只有所有可行摆法给出同一组补测落点时才会写入；确认时用最新实测重核。
CREATE TABLE IF NOT EXISTS alignment_preview (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    grid_id             INTEGER NOT NULL REFERENCES grid(id),
    region_r0           INTEGER NOT NULL,   -- 搜索区，0 基闭区间
    region_r1           INTEGER NOT NULL,
    region_c0           INTEGER NOT NULL,
    region_c1           INTEGER NOT NULL,
    placement_count     INTEGER NOT NULL,   -- 可行摆法数（落点唯一时可能 >1）
    point_count         INTEGER NOT NULL,
    raw_csv             BLOB NOT NULL,      -- 原始补测表，确认时重新解析重核
    filename            TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alignment_preview_point (
    preview_id  INTEGER NOT NULL REFERENCES alignment_preview(id) ON DELETE CASCADE,
    row_idx     INTEGER NOT NULL,           -- 格网 0 基落点
    col_idx     INTEGER NOT NULL,
    temp        INTEGER NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('ref', 'fill')),
    PRIMARY KEY (preview_id, row_idx, col_idx)
) WITHOUT ROWID;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db_path() -> str:
    return os.environ.get("FURNACE_DB",
                          os.path.join(os.getcwd(), "instance", "furnace.db"))


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")   # 64 MiB 页缓存
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection):
    """显式事务：异常回滚，正常提交。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ----------------------------- 读取 --------------------------------

def get_grid(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM grid WHERE id=1").fetchone()


def load_cells(conn: sqlite3.Connection) -> Dict[Tuple[int, int], Optional[int]]:
    rows = conn.execute(
        "SELECT row_idx, col_idx, temp FROM cells WHERE grid_id=1").fetchall()
    return {(r["row_idx"], r["col_idx"]): r["temp"] for r in rows}


def load_node_map(conn: sqlite3.Connection
                  ) -> Dict[Tuple[int, int, int], sqlite3.Row]:
    rows = conn.execute(
        "SELECT row_idx, col_idx, side, measured_cnt, unmeasured_cnt, "
        "temp_min, temp_max, is_valid FROM nodes WHERE grid_id=1"
    ).fetchall()
    return {(r["row_idx"], r["col_idx"], r["side"]): r for r in rows}


def current_batch_id(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute(
        "SELECT id FROM batch WHERE grid_id=1 AND is_current=1").fetchone()
    return row["id"] if row else None


# 现行块直接读物化表；block_op 仅作为批次历史与回滚依据保留。
_CURRENT_BLOCK_SQL = """
SELECT row_idx, col_idx, side, temp_max, temp_min, cell_count
FROM blocks_current
ORDER BY row_idx, col_idx, side DESC
"""


def load_current_blocks(conn: sqlite3.Connection) -> List[Block]:
    if current_batch_id(conn) is None:
        return []
    rows = conn.execute(_CURRENT_BLOCK_SQL).fetchall()
    return [Block(r["row_idx"], r["col_idx"], r["side"], r["temp_max"],
                  r["temp_min"], r["cell_count"]) for r in rows]


def iter_current_blocks(conn: sqlite3.Connection):
    """流式迭代现行块，供下载大网格时避免一次性构建 20 万个对象。"""
    if current_batch_id(conn) is None:
        return
    cur = conn.execute(_CURRENT_BLOCK_SQL)
    for r in cur:
        yield Block(r["row_idx"], r["col_idx"], r["side"], r["temp_max"],
                    r["temp_min"], r["cell_count"])


def current_block_count(conn: sqlite3.Connection) -> int:
    if current_batch_id(conn) is None:
        return 0
    return conn.execute("SELECT COUNT(*) FROM blocks_current").fetchone()[0]


# --------------------------- 内部工具 ------------------------------

class _MemRow:
    """与 sqlite3.Row 同接口（按列名取值）的轻量行对象。"""

    def __init__(self, coord: quadtree.Coord, stat: quadtree.NodeStat, tol: int):
        r, c, size = coord
        measured, unmeasured, tmin, tmax = stat
        self._d = {
            "row_idx": r, "col_idx": c, "side": size,
            "measured_cnt": measured, "unmeasured_cnt": unmeasured,
            "temp_min": tmin, "temp_max": tmax,
            "is_valid": 1 if quadtree.stat_is_valid(stat, tol) else 0,
        }

    def __getitem__(self, key):
        return self._d[key]


def _new_batch(conn: sqlite3.Connection, reason: str,
               tol: int, block_count: int) -> int:
    conn.execute("UPDATE batch SET is_current=0 WHERE grid_id=1")
    cur = conn.execute(
        "INSERT INTO batch(grid_id,reason,is_current,tol,block_count,created_at) "
        "VALUES(1,?,1,?,?,?)", (reason, tol, block_count, _now()))
    return cur.lastrowid


def _log(conn: sqlite3.Connection, kind: str, detail: str) -> None:
    conn.execute("INSERT INTO event_log(ts,kind,detail) VALUES(?,?,?)",
                 (_now(), kind, detail))


def _materialize_subtree(node_rows, coord, adds, leaf_need) -> None:
    """
    祖先合法块被拆除后，为未受影响子树按现存结构产出块意图：
    size>=2 的合法节点直接补发；非法节点下钻；到 size=2 仍非法时，
    把四个叶子坐标记入 leaf_need，由调用方统一批量读温度后补发。
    """
    r, c, size = coord
    row = node_rows[coord]
    if row["is_valid"] == 1:
        adds.append(Block(r, c, size, row["temp_max"], row["temp_min"],
                          row["measured_cnt"]))
        return
    if size == 2:
        for dr in (0, 1):
            for dc in (0, 1):
                leaf_need.add((r + dr, c + dc))
        return
    for child in quadtree.children_of(coord):
        _materialize_subtree(node_rows, child, adds, leaf_need)


def _collect_delta(node_rows, coord, affected, changed_leaves,
                   old_valid: Dict[Tuple[int, int, int], bool],
                   adds: List[Block], tombstones: List,
                   leaf_need: set, split: bool = False) -> None:
    """
    自受影响根（恒为整表根）向下收集本次编辑的块增量。

    split=True 表示该节点上方存在“旧合法、新非法”的祖先：
    旧整块已被区域墓碑拆除，本区域保留下来的块都必须显式 add。

    叶子不直接读温度：需要作为块输出的叶子坐标统一记入 leaf_need，
    由调用方在遍历结束后批量加载，保证叶子读取量与实际块输出成正比。

    - 新合法：区域墓碑 + 一个 add（吸收内部全部旧块），结束下钻；
    - 新非法、旧合法（或 split）：本块被拆除，置 split=True 继续下钻；
    - 新非法且一直非法：未受影响孩子旧块仍有效、跳过，仅受影响路径递归；
    - size=2 非法：四个叶子按“改动/拆分补发”记入 leaf_need，改动叶子另记墓碑。
    """
    r, c, size = coord

    # size=2 是最小内部节点，其孩子为叶子，在此终止递归
    if size == 2:
        row = node_rows[coord]
        if row["is_valid"] == 1:
            tombstones.append(("region", r, c, size))
            adds.append(Block(r, c, size, row["temp_max"], row["temp_min"],
                              row["measured_cnt"]))
            return
        now_split = split or bool(old_valid.get(coord, False))
        if now_split and not split:
            tombstones.append((r, c, size))
        for dr in (0, 1):
            for dc in (0, 1):
                pos = (r + dr, c + dc)
                if pos in changed_leaves:
                    tombstones.append((pos[0], pos[1], 1))
                    leaf_need.add(pos)
                elif now_split:
                    leaf_need.add(pos)
        return

    # n==1 退化情形
    if size == 1:
        if (r, c) in changed_leaves:
            tombstones.append((r, c, 1))
            leaf_need.add((r, c))
        elif split:
            leaf_need.add((r, c))
        return

    row = node_rows[coord]
    if row["is_valid"] == 1:
        tombstones.append(("region", r, c, size))
        adds.append(Block(r, c, size, row["temp_max"], row["temp_min"],
                          row["measured_cnt"]))
        return

    now_split = split or bool(old_valid.get(coord, False))
    if now_split and not split:
        tombstones.append((r, c, size))

    for child in quadtree.children_of(coord):
        cr, cc, cs = child
        if cs == 1:
            # 父节点必为 size=2，已在上面的基例处理，理论上不可达
            pos = (cr, cc)
            if pos in changed_leaves:
                tombstones.append((cr, cc, 1))
                leaf_need.add(pos)
            elif now_split:
                leaf_need.add(pos)
            continue
        if child not in affected:
            if now_split:
                _materialize_subtree(node_rows, child, adds, leaf_need)
            continue
        _collect_delta(node_rows, child, affected, changed_leaves,
                       old_valid, adds, tombstones, leaf_need, now_split)


def _write_base_blocks(conn: sqlite3.Connection, bid: int,
                       blocks: List[Block]) -> None:
    conn.executemany(
        "INSERT INTO block_op(batch_id,row_idx,col_idx,side,op,"
        "temp_max,temp_min,cell_count) VALUES(?,?,?,?,'base',?,?,?)",
        [(bid, b.row, b.col, b.size, b.tmax, b.tmin, b.count) for b in blocks],
    )


def _rebuild_materialized(conn: sqlite3.Connection,
                          blocks: List[Block]) -> None:
    """整体重建现行块物化表（用于上传、改温差这类全局重建）。"""
    conn.execute("DELETE FROM blocks_current")
    conn.executemany(
        "INSERT INTO blocks_current(row_idx,col_idx,side,temp_max,temp_min,cell_count) "
        "VALUES(?,?,?,?,?,?)",
        [(b.row, b.col, b.size, b.tmax, b.tmin, b.count) for b in blocks],
    )


def _apply_delta_materialized(conn: sqlite3.Connection,
                              tombstones: List[Tuple[int, int, int]],
                              adds: List[Block]) -> None:
    """把局部块增量应用到物化表：同址 UPSERT，删除坐标先删后插。"""
    conn.executemany(
        "DELETE FROM blocks_current WHERE row_idx=? AND col_idx=? AND side=?",
        tombstones)
    conn.executemany(
        "INSERT INTO blocks_current(row_idx,col_idx,side,temp_max,temp_min,cell_count) "
        "VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(row_idx,col_idx,side) DO UPDATE SET "
        "temp_max=excluded.temp_max, temp_min=excluded.temp_min, "
        "cell_count=excluded.cell_count",
        [(b.row, b.col, b.size, b.tmax, b.tmin, b.count) for b in adds],
    )


# ----------------------------- 服务 --------------------------------

def create_grid(conn: sqlite3.Connection, n: int,
                data: Sequence[Sequence[Optional[int]]],
                tol: int) -> Dict[str, int]:
    """全新上传：清空旧数据，整体建表、建四叉树、建首批次与整套 base 块。"""
    metrics = {"nodes_recomputed": 0, "blocks_added": 0, "blocks_removed": 0}
    with transaction(conn):
        # 对位预览外键引用 grid，必须先作废预览再删格网
        clear_alignment_preview(conn)
        conn.execute("DELETE FROM block_op")
        conn.execute("DELETE FROM batch")
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM cells")
        conn.execute("DELETE FROM grid")
        # 重传底图：格网整体替换，旧对位预览已作废，
        # 用户须基于新底图重新预览（改温差不经过这里，预览保留）。

        now = _now()
        conn.execute(
            "INSERT INTO grid(id,side,tol,created_at,updated_at) "
            "VALUES(1,?,?,?,?)", (n, tol, now, now))
        conn.executemany(
            "INSERT INTO cells(grid_id,row_idx,col_idx,temp) VALUES(1,?,?,?)",
            [(r, c, data[r][c]) for r in range(n) for c in range(n)])
        cells = {(r, c): data[r][c] for r in range(n) for c in range(n)}

        stats = quadtree.compute_all_node_stats(cells, n)
        conn.executemany(
            "INSERT INTO nodes(grid_id,row_idx,col_idx,side,measured_cnt,"
            "unmeasured_cnt,temp_min,temp_max,is_valid) "
            "VALUES(1,?,?,?,?,?,?,?,?)",
            [(*coord, *stat, 1 if quadtree.stat_is_valid(stat, tol) else 0)
             for coord, stat in stats.items()])
        metrics["nodes_recomputed"] = len(stats)

        blocks = quadtree.collect_blocks(stats, cells, n, tol)
        bid = _new_batch(conn, "upload", tol, len(blocks))
        _write_base_blocks(conn, bid, blocks)
        _rebuild_materialized(conn, blocks)
        metrics["blocks_added"] = len(blocks)
        _log(conn, "upload",
             f"新建方阵 {n}x{n}，tol={tol}，块数={len(blocks)}")
    return metrics


def update_tolerance(conn: sqlite3.Connection, tol: int) -> Dict[str, int]:
    """改允许温差：节点统计不变，只重算合法标志，并以整套 base 块重建新批次。"""
    grid = get_grid(conn)
    if grid is None:
        raise RuntimeError("尚未上传方阵")
    n = grid["side"]
    metrics = {"nodes_recomputed": 0, "blocks_added": 0, "blocks_removed": 0}
    with transaction(conn):
        conn.execute("UPDATE grid SET tol=?, updated_at=? WHERE id=1",
                     (tol, _now()))
        conn.execute(
            "UPDATE nodes SET is_valid = CASE WHEN unmeasured_cnt=0 "
            "AND measured_cnt>0 AND temp_max-temp_min<=? THEN 1 ELSE 0 END",
            (tol,))
        metrics["nodes_recomputed"] = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE grid_id=1").fetchone()[0]

        cells = load_cells(conn)
        node_rows = load_node_map(conn)
        stats = {k: (v["measured_cnt"], v["unmeasured_cnt"],
                     v["temp_min"], v["temp_max"])
                 for k, v in node_rows.items()}
        blocks = quadtree.collect_blocks(stats, cells, n, tol)
        # 温差是全局变化，旧 delta 链不再有参考价值，整体以 base 重建
        conn.execute("DELETE FROM block_op")
        bid = _new_batch(conn, "tolerance", tol, len(blocks))
        _write_base_blocks(conn, bid, blocks)
        _rebuild_materialized(conn, blocks)
        metrics["blocks_added"] = len(blocks)
        _log(conn, "tolerance", f"允许温差改为 {tol}，块数={len(blocks)}")
    return metrics


def apply_edits(conn: sqlite3.Connection,
                edits: Dict[Tuple[int, int], Optional[int]]) -> Dict[str, int]:
    """
    批量补测/清空。edits 值为 int（补测）或 None（清空）。

    只重算包含改动格的节点，块变更以 add/tombstone 增量写入新批次，
    写入量与改动子树规模成正比，不再复制全量块。
    """
    grid = get_grid(conn)
    if grid is None:
        raise RuntimeError("尚未上传方阵")
    with transaction(conn):
        return _apply_edits_locked(conn, edits)


def _apply_edits_locked(conn: sqlite3.Connection,
                        edits: Dict[Tuple[int, int], Optional[int]]) -> Dict[str, int]:
    """apply_edits 的事务内实现；调用方必须已持有写事务（BEGIN IMMEDIATE）。

    对位确认整批填入复用本函数，保证“重核 + 填入 + 供数块更新”原子完成。
    """
    grid = get_grid(conn)
    if grid is None:
        raise RuntimeError("尚未上传方阵")
    n, tol = grid["side"], grid["tol"]

    # 只取改动格的现值做校验与空改剔除，避免全表读取 cells。
    # 用临时表按 cells 主键点查，避免 OR 条件在大表上退化为扫描。
    edit_keys = list(edits.keys())
    for (r, c) in edit_keys:
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError(f"坐标 ({r + 1},{c + 1}) 超出 {n}x{n} 网格")
    conn.execute("CREATE TEMP TABLE t_edit(r INT, c INT)")
    try:
        conn.executemany("INSERT INTO t_edit VALUES(?,?)", edit_keys)
        old_rows = conn.execute(
            "SELECT ce.row_idx, ce.col_idx, ce.temp FROM cells ce "
            "JOIN t_edit e ON ce.row_idx=e.r AND ce.col_idx=e.c "
            "WHERE ce.grid_id=1").fetchall()
    finally:
        conn.execute("DROP TABLE t_edit")
    old_vals = {(r["row_idx"], r["col_idx"]): r["temp"] for r in old_rows}

    real: Dict[Tuple[int, int], Optional[int]] = {}
    for (r, c), val in edits.items():
        if val is not None and not isinstance(val, int):
            raise ValueError(f"坐标 ({r + 1},{c + 1}) 的温度必须是整数")
        if old_vals.get((r, c)) != val:
            real[(r, c)] = val
    if not real:
        _log(conn, "edit", "提交的改动与现值一致，未产生新批次")
        return {"nodes_recomputed": 0, "blocks_added": 0,
                "blocks_removed": 0, "changed_cells": 0}

    # 1) 批量写格子
    conn.executemany(
        "UPDATE cells SET temp=? WHERE grid_id=1 AND row_idx=? AND col_idx=?",
        [(val, r, c) for (r, c), val in real.items()])
    # 后续叶子统计只需改动格的值；未改动叶子的统计经由 nodes 表获得
    cells = dict(old_vals)
    cells.update(real)

    # 2) 受影响祖先并集，自底向上重算（n==1 无内部节点，单独处理）
    affected: set = set()
    for (r, c) in real:
        affected.update(quadtree.ancestor_nodes(r, c, n))

    if n == 1:
        v = cells.get((0, 0))
        bid = _new_batch(conn, "edit", tol, 1 if v is not None else 0)
        conn.execute(
            "INSERT INTO block_op(batch_id,row_idx,col_idx,side,op,"
            "temp_max,temp_min,cell_count) "
            "VALUES(?,0,0,1,'tombstone',NULL,NULL,NULL)", (bid,))
        added = 0
        if v is not None:
            conn.execute(
                "INSERT INTO block_op(batch_id,row_idx,col_idx,side,op,"
                "temp_max,temp_min,cell_count) "
                "VALUES(?,0,0,1,'add',?,?,1)", (bid, v, v))
            added = 1
        conn.execute("DELETE FROM blocks_current WHERE row_idx=0 AND col_idx=0 AND side=1")
        if v is not None:
            conn.execute(
                "INSERT INTO blocks_current(row_idx,col_idx,side,temp_max,temp_min,cell_count) "
                "VALUES(0,0,1,?,?,1)", (v, v))
        conn.execute("UPDATE grid SET updated_at=? WHERE id=1", (_now(),))
        _log(conn, "edit", f"n=1 单格改动，现行块数 {1 if v is not None else 0}")
        return {"nodes_recomputed": 0, "blocks_added": added,
                "blocks_removed": 1, "changed_cells": 1}

    ordered = sorted(affected, key=lambda x: x[2])
    # 重算节点还要读其孩子，加载集合 = 受影响节点 ∪ 其 size>=2 子节点
    load_set = set(affected)
    for coord in ordered:
        for kid in quadtree.children_of(coord):
            if kid[2] >= 2:
                load_set.add(kid)
    # 用临时表 JOIN 按主键索引取节点，避免大串 OR 触发全表扫描
    conn.execute("CREATE TEMP TABLE t_want_nodes(r INT, c INT, s INT)")
    try:
        conn.executemany("INSERT INTO t_want_nodes VALUES(?,?,?)",
                         sorted(load_set))
        node_rows = {
            (r["row_idx"], r["col_idx"], r["side"]): r
            for r in conn.execute(
                "SELECT n.row_idx, n.col_idx, n.side, n.measured_cnt, "
                "n.unmeasured_cnt, n.temp_min, n.temp_max, n.is_valid "
                "FROM nodes n JOIN t_want_nodes w "
                "ON n.row_idx=w.r AND n.col_idx=w.c AND n.side=w.s "
                "WHERE n.grid_id=1")
        }
    finally:
        conn.execute("DROP TABLE t_want_nodes")

    # 重算 size=2 节点要读其叶子真实温度；只加载受影响的 size=2
    # 节点覆盖的叶子，规模 ≤4×改动数。按 cells 主键点查，走索引。
    leaf_quads = sorted(coord for coord in affected if coord[2] == 2)
    if leaf_quads:
        leaf_positions = sorted({
            (rr + dr, cc + dc)
            for (rr, cc, _ss) in leaf_quads
            for dr in (0, 1) for dc in (0, 1)
        })
        conn.execute("CREATE TEMP TABLE t_want_leaf2(r INT, c INT)")
        try:
            conn.executemany("INSERT INTO t_want_leaf2 VALUES(?,?)",
                             leaf_positions)
            for row in conn.execute(
                    "SELECT ce.row_idx, ce.col_idx, ce.temp FROM cells ce "
                    "JOIN t_want_leaf2 w ON ce.row_idx=w.r AND ce.col_idx=w.c "
                    "WHERE ce.grid_id=1"):
                cells[(row["row_idx"], row["col_idx"])] = row["temp"]
        finally:
            conn.execute("DROP TABLE t_want_leaf2")

    def stat_of(coord: quadtree.Coord) -> quadtree.NodeStat:
        r2, c2, s2 = coord
        if s2 == 1:
            v = cells.get((r2, c2))
            return (1, 0, v, v) if v is not None else (0, 1, None, None)
        row = node_rows[coord]
        return (row["measured_cnt"], row["unmeasured_cnt"],
                row["temp_min"], row["temp_max"])

    # 重算前的合法标志，供块增量判断“旧合法块是否被拆除”。
    # 覆盖整个加载集合：未受影响兄弟在补发子树时也要读旧合法性。
    old_valid = {coord: (row["is_valid"] == 1)
                 for coord, row in node_rows.items()}

    node_upserts = []
    for coord in ordered:
        kids = quadtree.children_of(coord)
        stat = stat_of(kids[0])
        for kid in kids[1:]:
            stat = quadtree._merge_stats(stat, stat_of(kid))
        r0, c0, s0 = coord
        measured, unmeasured, tmin, tmax = stat
        valid = 1 if quadtree.stat_is_valid(stat, tol) else 0
        node_upserts.append(
            (r0, c0, s0, measured, unmeasured, tmin, tmax, valid))
        node_rows[coord] = _MemRow(coord, stat, tol)
    conn.executemany(
        "INSERT INTO nodes(grid_id,row_idx,col_idx,side,measured_cnt,"
        "unmeasured_cnt,temp_min,temp_max,is_valid) "
        "VALUES(1,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(grid_id,row_idx,col_idx,side) DO UPDATE SET "
        "measured_cnt=excluded.measured_cnt, "
        "unmeasured_cnt=excluded.unmeasured_cnt, "
        "temp_min=excluded.temp_min, temp_max=excluded.temp_max, "
        "is_valid=excluded.is_valid", node_upserts)

    # 3) 内存收集块增量（tombstones 含精确 3 元组与 ('region',r,c,s) 区域元组）
    adds: List[Block] = []
    tombstones: List = []
    leaf_need: set = set()
    _collect_delta(node_rows, (0, 0, n), affected,
                   set(real.keys()), old_valid, adds, tombstones, leaf_need)

    # 收集器只记录叶子意图；这里按实际需要批量读温度，
    # 实测格产出 size=1 块，未测格不产出。读取量与实际块输出成正比。
    leaf_list = sorted(leaf_need)
    leaf_temp: Dict[Tuple[int, int], Optional[int]] = {}
    if leaf_list:
        conn.execute("CREATE TEMP TABLE t_want_leaf(r INT, c INT)")
        try:
            conn.executemany("INSERT INTO t_want_leaf VALUES(?,?)",
                             leaf_list)
            for row in conn.execute(
                    "SELECT ce.row_idx, ce.col_idx, ce.temp FROM cells ce "
                    "JOIN t_want_leaf w ON ce.row_idx=w.r AND ce.col_idx=w.c "
                    "WHERE ce.grid_id=1"):
                leaf_temp[(row["row_idx"], row["col_idx"])] = row["temp"]
        finally:
            conn.execute("DROP TABLE t_want_leaf")
    for pos in leaf_list:
        v = leaf_temp.get(pos)
        if v is not None:
            adds.append(Block(pos[0], pos[1], 1, v, v, 1))

    exact_tomb = [t for t in tombstones if t[0] != "region"]
    region_tomb = [t for t in tombstones if t[0] == "region"]

    # 同批次同坐标既有精确墓碑又有 add 时，add 本身即覆盖旧状态，取消该墓碑
    add_positions = {(b.row, b.col, b.size) for b in adds}
    exact_tomb = [t for t in exact_tomb if t not in add_positions]

    # 精确墓碑：只有 delta 链上该坐标最新操作仍是有效块时才写。
    # 不能只查物化表：被更大祖先块覆盖的坐标没有物化行，
    # 但 delta 里可能残留更早的 base/add，祖先拆分后必须显式作废。
    if exact_tomb:
        active_pos = set()
        for i in range(0, len(exact_tomb), 300):
            chunk = exact_tomb[i:i + 300]
            t_where = " OR ".join(
                "(o.row_idx=? AND o.col_idx=? AND o.side=?)" for _ in chunk)
            t_keys = [v for tup in chunk for v in tup]
            for r in conn.execute(
                    "WITH latest AS ("
                    "SELECT o.row_idx, o.col_idx, o.side, MAX(o.id) AS mid "
                    "FROM block_op o JOIN batch b ON b.id=o.batch_id "
                    f"WHERE b.grid_id=1 AND ({t_where}) "
                    "GROUP BY o.row_idx, o.col_idx, o.side) "
                    "SELECT l.row_idx, l.col_idx, l.side FROM latest l "
                    "JOIN block_op x ON x.id=l.mid WHERE x.op<>'tombstone'",
                    t_keys):
                active_pos.add((r["row_idx"], r["col_idx"], r["side"]))
        exact_tomb = [t for t in exact_tomb if t in active_pos]

    # 区域墓碑：展开为该区域内 delta 链上仍有效的全部块坐标。
    # 块合并时这些后代块都被新的大块吸收，必须逐条作废，
    # 否则 delta 还原与物化表都会残留被吸收的旧块。
    region_positions: List[Tuple[int, int, int]] = []
    for (_, rr, cc, ss) in region_tomb:
        rows = conn.execute(
            "WITH latest AS ("
            "SELECT o.row_idx, o.col_idx, o.side, MAX(o.id) AS mid "
            "FROM block_op o JOIN batch b ON b.id=o.batch_id "
            "WHERE b.grid_id=1 AND o.row_idx>=? AND o.col_idx>=? "
            "AND o.row_idx+o.side<=? AND o.col_idx+o.side<=? "
            "GROUP BY o.row_idx, o.col_idx, o.side) "
            "SELECT l.row_idx, l.col_idx, l.side FROM latest l "
            "JOIN block_op x ON x.id=l.mid WHERE x.op<>'tombstone'",
            (rr, cc, rr + ss, cc + ss)).fetchall()
        for row in rows:
            pos = (row["row_idx"], row["col_idx"], row["side"])
            if pos not in add_positions:
                region_positions.append(pos)
    # 去重（多个区域/精确墓碑可能指向同一坐标）
    all_tomb = list(dict.fromkeys(exact_tomb + region_positions))

    # 4) 建新批次，只写局部增量，并同步物化现行块表
    bid = _new_batch(conn, "edit", tol, 0)
    conn.executemany(
        "INSERT INTO block_op(batch_id,row_idx,col_idx,side,op,"
        "temp_max,temp_min,cell_count) "
        "VALUES(?,?,?,?,'tombstone',NULL,NULL,NULL)",
        [(bid, r, c, s) for (r, c, s) in all_tomb])
    conn.executemany(
        "INSERT INTO block_op(batch_id,row_idx,col_idx,side,op,"
        "temp_max,temp_min,cell_count) "
        "VALUES(?,?,?,?,'add',?,?,?)",
        [(bid, b.row, b.col, b.size, b.tmax, b.tmin, b.count)
         for b in adds])
    _apply_delta_materialized(conn, all_tomb, adds)

    count = current_block_count(conn)
    conn.execute("UPDATE batch SET block_count=? WHERE id=?", (count, bid))
    conn.execute("UPDATE grid SET updated_at=? WHERE id=1", (_now(),))
    _log(conn, "edit",
         f"改动 {len(real)} 格，重算节点 {len(ordered)} 个，"
         f"add {len(adds)} / tombstone {len(all_tomb)}，现行块数 {count}")
    return {"nodes_recomputed": len(ordered),
            "blocks_added": len(adds),
            "blocks_removed": len(all_tomb),
            "changed_cells": len(real)}


# ======================= 热像仪补测表对位预览 =======================

class AlignmentConfirmError(RuntimeError):
    """确认重核失败：读数变化、目标格已被占用或预览失效。"""


def clear_alignment_preview(conn: sqlite3.Connection) -> None:
    """作废当前对位预览（重传底图、新一轮预览替换时调用，事务内执行）。"""
    conn.execute("DELETE FROM alignment_preview_point")
    conn.execute("DELETE FROM alignment_preview")


def cancel_alignment_preview(conn: sqlite3.Connection) -> bool:
    """用户主动放弃预览：独立短事务作废；无预览返回 False。"""
    with transaction(conn):
        head = conn.execute(
            "SELECT 1 FROM alignment_preview WHERE id=1").fetchone()
        if head is None:
            return False
        clear_alignment_preview(conn)
        _log(conn, "alignment_cancel", "用户放弃对位预览，未写入任何读数")
        return True


def save_alignment_preview(conn: sqlite3.Connection,
                           region: Tuple[int, int, int, int],
                           result, raw_csv: bytes, filename: str) -> None:
    """把唯一落点的对位结果落库为当前预览（同一时刻只保留一份）。"""
    r0, r1, c0, c1 = region
    with transaction(conn):
        clear_alignment_preview(conn)
        conn.execute(
            "INSERT INTO alignment_preview(id,grid_id,region_r0,region_r1,"
            "region_c0,region_c1,placement_count,point_count,raw_csv,"
            "filename,created_at) VALUES(1,1,?,?,?,?,?,?,?,?,?)",
            (r0, r1, c0, c1, len(result.placements),
             len(result.fill_landings) + len(result.ref_landings),
             sqlite3.Binary(raw_csv), filename, _now()))
        conn.executemany(
            "INSERT INTO alignment_preview_point(preview_id,row_idx,col_idx,"
            "temp,kind) VALUES(1,?,?,?,'fill')",
            [(r, c, t) for (r, c, t) in result.fill_landings])
        conn.executemany(
            "INSERT INTO alignment_preview_point(preview_id,row_idx,col_idx,"
            "temp,kind) VALUES(1,?,?,?,'ref')",
            [(r, c, t) for (r, c, t) in result.ref_landings])
        _log(conn, "alignment_preview",
             f"对位预览：搜索区 行{r0 + 1}..{r1 + 1}/列{c0 + 1}..{c1 + 1}，"
             f"可行摆法 {len(result.placements)} 种且补测落点唯一，"
             f"补测 {len(result.fill_landings)} 点、参照 "
             f"{len(result.ref_landings)} 点（{filename}）")


def load_alignment_preview(conn: sqlite3.Connection) -> Optional[dict]:
    """读取当前预览（含参照/补测落点）；无预览返回 None。"""
    head = conn.execute("SELECT * FROM alignment_preview WHERE id=1").fetchone()
    if head is None:
        return None
    pts = conn.execute(
        "SELECT row_idx, col_idx, temp, kind FROM alignment_preview_point "
        "WHERE preview_id=1 ORDER BY kind, row_idx, col_idx").fetchall()
    return {
        "head": head,
        "fills": [(r["row_idx"], r["col_idx"], r["temp"])
                  for r in pts if r["kind"] == "fill"],
        "refs": [(r["row_idx"], r["col_idx"], r["temp"])
                 for r in pts if r["kind"] == "ref"],
    }


def confirm_alignment_preview(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    用最新实测重核当前预览并整批填入补测点，全程在一个写事务内：

    1. 重新解析原始补测表，按原搜索区用**最新 cells** 重新对位；
    2. 仍须落点唯一，且唯一补测坐标+值与预览完全一致；
    3. 参照点仍与最新实测吻合（重算已隐含校验，再断言一次）；
    4. 全部补测目标格当前必须未测（不能覆盖任何已测格）。
    任一不满足则回滚、读数与供数块不变，并抛出 AlignmentConfirmError；
    成功则删除预览、复用增量核算写格并更新现行供数块。
    """
    grid = get_grid(conn)
    if grid is None:
        raise AlignmentConfirmError("请先上传方阵 CSV")
    n = grid["side"]

    with transaction(conn):
        preview = load_alignment_preview(conn)
        if preview is None:
            raise AlignmentConfirmError("当前没有可确认的对位预览，请先上传补测表做对位预览")
        head = preview["head"]
        region = (head["region_r0"], head["region_r1"],
                  head["region_c0"], head["region_c1"])
        raw = bytes(head["raw_csv"])

        try:
            points = csvutil.parse_alignment_csv(raw)
        except SheetError as exc:
            raise AlignmentConfirmError(f"预览中的补测表无法重新解析：{exc}")

        cells = load_cells(conn)
        result = alignment.solve(points, region, cells)

        if result.status != "unique":
            raise AlignmentConfirmError(
                "重核失败：最新实测下对位结果已不再是唯一落点"
                "（参照读数可能已变化），请重新上传补测表对位后再试")
        new_fills = sorted(result.fill_landings)
        old_fills = sorted(preview["fills"])
        if new_fills != old_fills:
            raise AlignmentConfirmError(
                "重核失败：最新实测算出的补测坐标/值与预览不一致，"
                "请重新上传补测表对位后再试")
        # 参照点必须仍与最新实测吻合（solve 已过滤，显式断言防回归）
        for (r, c, t) in result.ref_landings:
            if cells.get((r, c)) != t:
                raise AlignmentConfirmError(
                    f"重核失败：参照点 ({r + 1},{c + 1}) 的最新实测不再是 {t} ℃，"
                    "请重新对位后再试")
        # 目标格必须全部未测——已测格绝不覆盖
        occupied = [(r, c) for (r, c, _t) in new_fills
                    if cells.get((r, c)) is not None]
        if occupied:
            desc = "、".join(f"({r + 1},{c + 1})" for r, c in occupied[:8])
            more = " 等" if len(occupied) > 8 else ""
            raise AlignmentConfirmError(
                f"重核失败：{len(occupied)} 个补测目标格已被其他补测占用"
                f"（{desc}{more}），本次整批未写入；可调整搜索区/补测表后重新对位")

        edits = {(r, c): t for (r, c, t) in new_fills}
        # 防御：occupied 检查已保证目标格全未测，edits 必然全部是真实改动；
        # 若断言失败说明上面的拦截被绕过，宁可回滚也不能留下失效预览。
        if any(cells.get(pos) is not None for pos in edits):
            raise AlignmentConfirmError("重核失败：存在已测的补测目标格，整批未写入")
        metrics = _apply_edits_locked(conn, edits)
        if metrics.get("changed_cells") != len(new_fills):
            raise AlignmentConfirmError(
                "重核失败：实际改动格数与预览补测点数不一致，整批回滚")
        clear_alignment_preview(conn)
        _log(conn, "alignment_confirm",
             f"对位确认：整批填入补测 {len(new_fills)} 格，"
             f"重算节点 {metrics['nodes_recomputed']} 个，"
             f"块增 {metrics['blocks_added']} / 删 {metrics['blocks_removed']}")
        metrics["filled_cells"] = len(new_fills)
        return metrics
