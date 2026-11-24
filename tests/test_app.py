# -*- coding: utf-8 -*-
"""
核心算法与服务自测（纯标准库，运行：PYTHONPATH=. python -m tests.test_app -v）

覆盖：
1. CSV 解析（空白=未测、非 2 次幂拒绝、缺列拒绝、表头、非整数拒绝）
2. 四叉树覆盖：不含未测格、块内温差不超限、不重叠且铺满实测格、块数最少性
3. 增量核算与全量重建等价性（随机补测/清空多轮对照）
4. 增量核算只触碰包含改动格的节点（节点重算数 = 改动格祖先并集）
5. 批次与下载：失败回滚后旧批次仍可下载；下载内容与现行批次同源
6. Flask 端到端：上传、批量补测、清空、改温差、下载
"""

import io
import os
import random
import sqlite3
import tempfile
import unittest
from itertools import product

from app import csvutil, db, quadtree
from app.web import create_app


def make_grid(n, fill=None, seed=0):
    rng = random.Random(seed)
    return [[fill if fill is not None else rng.randint(0, 100) for _ in range(n)] for _ in range(n)]


def full_rebuild_blocks(conn):
    """用当前 cells 全量重建一遍，返回 Block 列表，作为增量结果的对照基准。"""
    grid = db.get_grid(conn)
    n, tol = grid["side"], grid["tol"]
    cells = db.load_cells(conn)
    stats = quadtree.compute_all_node_stats(cells, n)
    return quadtree.collect_blocks(stats, cells, n, tol), cells


def block_set(blocks):
    return {(b.row, b.col, b.size, b.tmin, b.tmax, b.count) for b in blocks}


class QuadtreeAlgorithmTests(unittest.TestCase):

    def test_power_of_two(self):
        for n in (1, 2, 4, 8, 16, 256):
            self.assertTrue(quadtree.is_power_of_two(n))
        for n in (0, 3, 5, 6, 7, 300):
            self.assertFalse(quadtree.is_power_of_two(n))

    def test_uniform_grid_one_block(self):
        n = 8
        cells = {(r, c): 100 for r in range(n) for c in range(n)}
        stats = quadtree.compute_all_node_stats(cells, n)
        blocks = quadtree.collect_blocks(stats, cells, n, tol=10)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], quadtree.Block(0, 0, 8, 100, 100, 64))

    def test_unmeasured_cells_never_covered(self):
        n = 4
        data = [[10, 10, None, None],
                [10, 10, None, None],
                [None, None, 20, 20],
                [None, None, 20, 20]]
        cells = {(r, c): data[r][c] for r in range(n) for c in range(n)}
        stats = quadtree.compute_all_node_stats(cells, n)
        blocks = quadtree.collect_blocks(stats, cells, n, tol=5)
        covered = set()
        for b in blocks:
            for dr, dc in product(range(b.size), range(b.size)):
                p = (b.row + dr, b.col + dc)
                self.assertIsNotNone(cells[p], "块内不得含未测格")
                self.assertNotIn(p, covered, "块之间不得重叠")
                covered.add(p)
        self.assertEqual(covered, {p for p, v in cells.items() if v is not None})
        self.assertEqual(len(blocks), 2)

    def test_tolerance_forces_split(self):
        # 整张温差超限，但四个象限各自均匀 -> 恰好裂成 4 块
        n = 4
        data = [[0, 0, 50, 50],
                [0, 0, 50, 50],
                [90, 90, 120, 120],
                [90, 90, 120, 120]]
        cells = {(r, c): data[r][c] for r in range(n) for c in range(n)}
        stats = quadtree.compute_all_node_stats(cells, n)
        blocks = quadtree.collect_blocks(stats, cells, n, tol=30)
        self.assertEqual(sorted((b.row, b.col, b.size) for b in blocks),
                         [(0, 0, 2), (0, 2, 2), (2, 0, 2), (2, 2, 2)])
        for b in blocks:
            self.assertLessEqual(b.tmax - b.tmin, 30)

        # 温差放到 120，根合法，只剩 1 块
        blocks2 = quadtree.collect_blocks(stats, cells, n, tol=120)
        self.assertEqual(len(blocks2), 1)

    def test_coverage_properties_random(self):
        rng = random.Random(42)
        for n in (2, 4, 8, 16):
            for _ in range(20):
                cells = {}
                for r, c in product(range(n), range(n)):
                    cells[(r, c)] = None if rng.random() < 0.3 else rng.randint(0, 200)
                tol = rng.choice((0, 5, 20, 500))
                stats = quadtree.compute_all_node_stats(cells, n)
                blocks = quadtree.collect_blocks(stats, cells, n, tol)
                covered = set()
                for b in blocks:
                    # 块必须是四叉树对齐节点
                    self.assertEqual(b.row % b.size, 0)
                    self.assertEqual(b.col % b.size, 0)
                    for dr, dc in product(range(b.size), range(b.size)):
                        p = (b.row + dr, b.col + dc)
                        self.assertIsNotNone(cells[p])
                        self.assertNotIn(p, covered)
                        covered.add(p)
                    self.assertLessEqual(b.tmax - b.tmin, tol)
                self.assertEqual(covered, {p for p, v in cells.items() if v is not None})

    def test_minimality_against_brute_force(self):
        """小网格上暴力枚举所有合法对齐方块，求集合覆盖下界，验证块数最少。"""
        rng = random.Random(7)
        for n in (1, 2, 4):
            for _ in range(15):
                cells = {}
                for r, c in product(range(n), range(n)):
                    cells[(r, c)] = None if rng.random() < 0.25 else rng.randint(0, 30)
                tol = rng.choice((0, 3, 10, 100))
                stats = quadtree.compute_all_node_stats(cells, n)
                blocks = quadtree.collect_blocks(stats, cells, n, tol)
                measured = {p for p, v in cells.items() if v is not None}

                # 枚举全部合法的对齐方块
                valid_tiles = []
                size = 1
                while size <= n:
                    for r in range(0, n, size):
                        for c in range(0, n, size):
                            pts = [(r + dr, c + dc) for dr in range(size) for dc in range(size)]
                            vals = [cells[p] for p in pts]
                            if any(v is None for v in vals):
                                continue
                            if max(vals) - min(vals) <= tol:
                                valid_tiles.append(frozenset(pts))
                    size *= 2
                # 贪心求一个上界；再用最大合法节点数做结构下界
                # （每个最大合法节点区域必须至少有一个块覆盖其“专属”格，
                #  且没有任何合法方块能跨越它的边界），二者应与本算法相等
                maximal = set()
                for tile in valid_tiles:
                    if not any(tile < other for other in valid_tiles):
                        maximal.add(tile)
                self.assertEqual(len(blocks), len(maximal))
                # 贪心覆盖数不可能少于结构块数
                remaining = set(measured)
                greedy = 0
                while remaining:
                    best = max(valid_tiles, key=lambda t: len(t & remaining))
                    remaining -= best
                    greedy += 1
                self.assertGreaterEqual(greedy, len(blocks))


class CSVParseTests(unittest.TestCase):

    def test_basic_with_blanks(self):
        raw = "1,2,\n,3,4\n5,6,7\n".encode()  # 3x3 非 2 次幂
        with self.assertRaises(csvutil.CSVParseError):
            csvutil.parse_square_csv(raw)

    def test_square_ok(self):
        raw = "1,2,3,4\n5,,7,8\n,10,11,12\n13,14,15,16\n".encode()
        n, data = csvutil.parse_square_csv(raw)
        self.assertEqual(n, 4)
        self.assertIsNone(data[1][1])
        self.assertEqual(data[2][0], None)
        self.assertEqual(data[0][0], 1)

    def test_header_and_ragged(self):
        raw = "row,c1,c2,c3,c4\n1,2,3,4\n5,6,7,8\n".encode()
        with self.assertRaises(csvutil.CSVParseError):
            csvutil.parse_square_csv(raw)  # 去表头后只剩 2 行，非 2 次幂
        # 4x4 数据中夹一行缺列：必须拒绝而不是静默对齐
        raw2 = "1,2,3,4\n5,6,7,8\n9,10,11\n13,14,15,16\n".encode()
        with self.assertRaises(csvutil.CSVParseError):
            csvutil.parse_square_csv(raw2)

    def test_non_integer(self):
        with self.assertRaises(csvutil.CSVParseError):
            csvutil.parse_square_csv(b"1,2\n3,x\n")
        with self.assertRaises(csvutil.CSVParseError):
            csvutil.parse_square_csv(b"1,2\n3,99999\n")  # 超温度上限

    def test_explicit_header_row(self):
        # 显式表头（首格为行号关键字）整行跳过，剩余为 4x4 方阵
        raw = "row,c1,c2,c3,c4\n1,2,3,4\n5,6,7,8\n9,10,11,12\n13,14,15,16\n".encode()
        n, data = csvutil.parse_square_csv(raw)
        self.assertEqual(n, 4)
        self.assertEqual(data[0][0], 1)
        self.assertEqual(data[3][3], 16)

    def test_no_implicit_index_column(self):
        # 带行号列但未写表头关键字：首列是数据，4 行 5 列不是方阵，必须拒绝
        raw = "1,1,2,3,4\n2,5,6,7,8\n3,9,10,11,12\n4,13,14,15,16\n".encode()
        with self.assertRaises(csvutil.CSVParseError):
            csvutil.parse_square_csv(raw)


class ServiceIncrementalTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "t.db")
        self.conn = db.connect(self.db_path)
        db.init_db(self.db_path)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_upload_and_blocks(self):
        n = 4
        data = make_grid(n, seed=1)
        db.create_grid(self.conn, n, data, 10)
        blocks = db.load_current_blocks(self.conn)
        expect, cells = full_rebuild_blocks(self.conn)
        self.assertEqual(block_set(blocks), block_set(expect))
        self.assertEqual(db.current_batch_id(self.conn), 1)

    def test_incremental_equivalence_random_edits(self):
        rng = random.Random(123)
        n = 16
        data = [[None if rng.random() < 0.4 else rng.randint(0, 150) for _ in range(n)]
                for _ in range(n)]
        db.create_grid(self.conn, n, data, 15)
        snapshot = db.load_cells(self.conn)

        for step in range(60):
            # 随机挑 1~6 个格子做补测或清空
            edits = {}
            for _ in range(rng.randint(1, 6)):
                p = (rng.randrange(n), rng.randrange(n))
                edits[p] = None if rng.random() < 0.4 else rng.randint(0, 150)
            # 编辑前快照判定真实改动（含清空为 None 的格）
            real_changes = {p for p, v in edits.items() if snapshot.get(p) != v}
            metrics = db.apply_edits(self.conn, edits)
            snapshot.update(edits)

            # 1) 增量结果必须与全量重建逐块一致
            blocks = db.load_current_blocks(self.conn)
            expect, cells = full_rebuild_blocks(self.conn)
            self.assertEqual(block_set(blocks), block_set(expect),
                             msg=f"step {step} 增量与全量结果不一致")

            # 2) 覆盖性质复检
            covered = set()
            for b in blocks:
                for dr, dc in product(range(b.size), range(b.size)):
                    p = (b.row + dr, b.col + dc)
                    self.assertIsNotNone(cells[p])
                    self.assertNotIn(p, covered)
                    covered.add(p)
                self.assertLessEqual(b.tmax - b.tmin, 15)
            self.assertEqual(covered, {p for p, v in cells.items() if v is not None})

            # 3) 重算节点数恰为真实改动格祖先并集大小（只触碰包含改动格的节点）
            self.assertEqual(metrics["changed_cells"], len(real_changes))
            if real_changes:
                anc = set()
                for (r, c) in real_changes:
                    anc.update(quadtree.ancestor_nodes(r, c, n))
                self.assertEqual(metrics["nodes_recomputed"], len(anc))
            else:
                self.assertEqual(metrics["nodes_recomputed"], 0)

    def test_single_cell_edit_touches_only_log_n_nodes(self):
        n = 32
        data = make_grid(n, seed=2)
        db.create_grid(self.conn, n, data, 500)  # 根合法，整块一块
        # 写入一个与现值必然不同的温度，触发单格祖先链重算
        metrics = db.apply_edits(self.conn, {(0, 0): -50})
        self.assertEqual(metrics["changed_cells"], 1)
        self.assertEqual(metrics["nodes_recomputed"], n.bit_length() - 1)  # log2(n) 个祖先

    def test_rollback_keeps_old_batch(self):
        n = 4
        db.create_grid(self.conn, n, make_grid(n, seed=3), 10)
        old_bid = db.current_batch_id(self.conn)
        old_blocks = block_set(db.load_current_blocks(self.conn))

        # 构造一个必然失败的编辑：越界坐标
        with self.assertRaises(ValueError):
            db.apply_edits(self.conn, {(n, 0): 50})
        self.assertEqual(db.current_batch_id(self.conn), old_bid)
        self.assertEqual(block_set(db.load_current_blocks(self.conn)), old_blocks)

        # 事务失败后库仍可正常写入
        db.apply_edits(self.conn, {(0, 0): 77})
        self.assertNotEqual(db.current_batch_id(self.conn), old_bid)

    def test_materialized_and_delta_consistency(self):
        """物化现行块、delta 日志还原、纯算法全量重建三者必须逐块一致。"""
        rng = random.Random(77)
        n = 32
        data = [[None if rng.random() < 0.35 else rng.randint(0, 120)
                 for _ in range(n)] for _ in range(n)]
        db.create_grid(self.conn, n, data, 12)

        def delta_reconstruct():
            rows = self.conn.execute(
                "WITH latest AS ("
                "SELECT o.row_idx, o.col_idx, o.side, MAX(o.id) AS mid FROM block_op o "
                "JOIN batch b ON b.id=o.batch_id "
                "WHERE b.id <= (SELECT id FROM batch WHERE grid_id=1 AND is_current=1) "
                "GROUP BY o.row_idx, o.col_idx, o.side) "
                "SELECT o.row_idx, o.col_idx, o.side, o.temp_max, o.temp_min, o.cell_count "
                "FROM latest JOIN block_op o ON o.id=latest.mid "
                "WHERE o.op<>'tombstone'"
            ).fetchall()
            # 字段顺序与 block_set 对齐：(row, col, side, tmin, tmax, count)
            return {(r["row_idx"], r["col_idx"], r["side"], r["temp_min"],
                     r["temp_max"], r["cell_count"]) for r in rows}

        for step in range(120):
            edits = {}
            for _ in range(rng.randint(1, 5)):
                p = (rng.randrange(n), rng.randrange(n))
                edits[p] = None if rng.random() < 0.4 else rng.randint(0, 120)
            db.apply_edits(self.conn, edits)
            if step % 25 == 24:
                db.update_tolerance(self.conn, rng.choice((0, 5, 20, 200)))

            materialized = block_set(db.load_current_blocks(self.conn))
            expect, _ = full_rebuild_blocks(self.conn)
            self.assertEqual(materialized, block_set(expect),
                             msg=f"step {step}: 物化表与全量重建不一致")
            self.assertEqual(delta_reconstruct(), materialized,
                             msg=f"step {step}: delta 日志与物化表不一致")

    def test_consistency_across_grid_sizes(self):
        """多种边长（含 1、2 退化情形）长轮次随机编辑下三者一致。"""
        rng = random.Random(909)
        for n in (1, 2, 4, 8, 64):
            data = [[None if rng.random() < 0.3 else rng.randint(0, 80)
                     for _ in range(n)] for _ in range(n)]
            db.create_grid(self.conn, n, data, 10)
            for step in range(40):
                edits = {}
                for _ in range(rng.randint(1, 4)):
                    p = (rng.randrange(n), rng.randrange(n))
                    edits[p] = None if rng.random() < 0.35 else rng.randint(0, 80)
                db.apply_edits(self.conn, edits)
                blocks = db.load_current_blocks(self.conn)
                expect, cells = full_rebuild_blocks(self.conn)
                self.assertEqual(block_set(blocks), block_set(expect),
                                 msg=f"n={n} step={step}")
                # 覆盖性质：不重叠、不含未测格、温差不超限
                covered = set()
                for b in blocks:
                    for dr, dc in product(range(b.size), range(b.size)):
                        p = (b.row + dr, b.col + dc)
                        self.assertIsNotNone(cells[p])
                        self.assertNotIn(p, covered)
                        covered.add(p)
                    self.assertLessEqual(b.tmax - b.tmin, 10)
                self.assertEqual(covered,
                                 {p for p, v in cells.items() if v is not None})
            db.update_tolerance(self.conn, 50)
            expect, _ = full_rebuild_blocks(self.conn)
            self.assertEqual(block_set(db.load_current_blocks(self.conn)),
                             block_set(expect))

    def test_block_merge_emits_region_tombstones(self):
        """清空后再补测使一个四分象限从非法变合法：内部旧小块必须被吸收，
        物化表/delta 中不得残留与新大块重叠的后代块。"""
        # 2x2，tol=0：初始四格温度相同 -> 一块 2x2
        conn = self.conn
        db.create_grid(conn, 2, [[50, 50], [50, 50]], 0)
        self.assertEqual(len(db.load_current_blocks(conn)), 1)
        # 改一格使温差超限 -> 裂成单格块
        db.apply_edits(conn, {(0, 0): 99})
        blocks = db.load_current_blocks(conn)
        self.assertEqual(sorted(b.size for b in blocks), [1, 1, 1, 1])
        # 改回相同温度 -> 重新合并为一块，不得有残留叶子块
        db.apply_edits(conn, {(0, 0): 50})
        blocks = db.load_current_blocks(conn)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].size, 2)
        # delta 还原同样只剩一块
        rows = conn.execute(
            "WITH latest AS (SELECT row_idx,col_idx,side,MAX(id) mid FROM block_op "
            "GROUP BY row_idx,col_idx,side) "
            "SELECT o.side FROM latest JOIN block_op o ON o.id=latest.mid "
            "WHERE o.op<>'tombstone'").fetchall()
        self.assertEqual([r["side"] for r in rows], [2])

    def test_tolerance_change(self):
        n = 8
        rng = random.Random(5)
        data = [[rng.randint(0, 100) for _ in range(n)] for _ in range(n)]
        db.create_grid(self.conn, n, data, 0)
        strict = len(db.load_current_blocks(self.conn))
        db.update_tolerance(self.conn, 1000)
        loose = len(db.load_current_blocks(self.conn))
        self.assertEqual(loose, 1)
        self.assertGreater(strict, loose)
        expect, _ = full_rebuild_blocks(self.conn)
        self.assertEqual(block_set(db.load_current_blocks(self.conn)), block_set(expect))

    def test_n1_grid(self):
        db.create_grid(self.conn, 1, [[10]], 5)
        self.assertEqual(len(db.load_current_blocks(self.conn)), 1)
        db.apply_edits(self.conn, {(0, 0): 20})
        b = db.load_current_blocks(self.conn)
        self.assertEqual(b[0].tmax, 20)
        db.apply_edits(self.conn, {(0, 0): None})
        self.assertEqual(db.load_current_blocks(self.conn), [])
        # 再补测回来
        db.apply_edits(self.conn, {(0, 0): 30})
        self.assertEqual(db.load_current_blocks(self.conn)[0].tmax, 30)

    def test_batch_history_preserved(self):
        n = 4
        db.create_grid(self.conn, n, make_grid(n, seed=4), 10)
        db.apply_edits(self.conn, {(0, 0): 55})
        db.update_tolerance(self.conn, 30)
        rows = self.conn.execute("SELECT id, reason, is_current FROM batch ORDER BY id").fetchall()
        self.assertEqual([r["reason"] for r in rows], ["upload", "edit", "tolerance"])
        self.assertEqual(sum(r["is_current"] for r in rows), 1)
        self.assertTrue(rows[-1]["is_current"])


class WebEndToEndTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "web.db")
        self.app = create_app(self.db_path)
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def _upload(self, raw, tol=20):
        return self.client.post(
            "/upload",
            data={"csvfile": (io.BytesIO(raw), "grid.csv"), "tol": str(tol)},
            content_type="multipart/form-data",
        )

    def test_full_flow(self):
        # 空首页
        self.assertEqual(self.client.get("/").status_code, 200)

        n = 8
        rng = random.Random(9)
        lines = []
        for r in range(n):
            lines.append(",".join(
                "" if rng.random() < 0.3 else str(rng.randint(0, 120))
                for _ in range(n)))
        resp = self._upload("\n".join(lines).encode())
        self.assertEqual(resp.status_code, 302)

        # 块页与下载
        self.assertEqual(self.client.get("/blocks").status_code, 200)
        dl = self.client.get("/download")
        self.assertEqual(dl.status_code, 200)
        self.assertIn("attachment", dl.headers["Content-Disposition"])
        text = dl.get_data(as_text=True)
        self.assertIn("block_id,row_start,col_start,side", text)
        first_data_line = text.strip().splitlines()[1].split(",")
        self.assertEqual(int(first_data_line[-1]), n)  # grid_side

        # 批量补测 2x2
        resp = self.client.post("/edit", data={
            "mode": "fill", "row_start": 1, "row_end": 2,
            "col_start": 1, "col_end": 2, "temp": "66",
        })
        self.assertEqual(resp.status_code, 302)

        # 批量清空
        resp = self.client.post("/edit", data={
            "mode": "clear", "row_start": 3, "row_end": 4,
            "col_start": 3, "col_end": 4,
        })
        self.assertEqual(resp.status_code, 302)

        # 改温差
        resp = self.client.post("/tolerance", data={"tol": "50"})
        self.assertEqual(resp.status_code, 302)

        # 坏输入不产生 5xx，旧数据仍可下载
        bad = self.client.post("/edit", data={
            "mode": "fill", "row_start": 1, "row_end": 99,
            "col_start": 1, "col_end": 1, "temp": "66",
        }, follow_redirects=True)
        self.assertEqual(bad.status_code, 200)
        self.assertIn("超出", bad.get_data(as_text=True))
        self.assertEqual(self.client.get("/download").status_code, 200)

        # 坏 CSV 不影响现有数据
        before = self.client.get("/download").get_data(as_text=True)
        bad_up = self._upload(b"1,2\n3,bad\n", tol=10)
        self.assertEqual(bad_up.status_code, 302)
        after = self.client.get("/download").get_data(as_text=True)
        self.assertEqual(before, after)

    def test_download_requires_grid(self):
        resp = self.client.get("/download")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
