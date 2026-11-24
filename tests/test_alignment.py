# -*- coding: utf-8 -*-
"""
热像仪补测表对位自测（纯标准库）。

覆盖验收要求：
1. 对位求解：平移 + 0/90/180/270 四朝向、无镜像、整组落区、参照吻合；
2. 唯一性：多摆法同落点可预览；落点多组 -> ambiguous；越界/参照失配 -> none；
3. 目标格已测不参与摆法筛选，只在确认时拦截；
4. 解析：4 列、用途标记、重复坐标、全参照、温度/行列校验；
5. 服务层：预览落库、重传底图作废预览、改温差保留预览；
6. 确认重核：交错补测占用目标格 / 参照读数变化 -> 整批回滚；
7. 成功确认后供数块与全量重建一致；失败不改读数与供数；
8. Web 端到端：预览、确认、放弃、坏输入不污染。

运行：PYTHONPATH=. python -m tests.test_alignment -v
"""

import io
import os
import random
import tempfile
import unittest
from itertools import product

from app import alignment, csvutil, db
from app.alignment import SheetPoint
from app.web import create_app


def sp(lr, lc, temp, is_ref):
    return SheetPoint(lr, lc, temp, is_ref)


def cell_map(grid):
    n = len(grid)
    return {(r, c): grid[r][c] for r in range(n) for c in range(n)}


def fill_dict(result):
    return {(r, c): t for (r, c, t) in result.fill_landings}


class RotationTests(unittest.TestCase):

    def test_rotations_are_pure_turns_no_mirror(self):
        # L 形三点，四种朝向后相对手性保持（只旋转不镜像）
        pts = [(1, 1), (1, 2), (2, 1)]
        r0 = alignment.rotate_points(pts, 0)
        r90 = alignment.rotate_points(pts, 90)
        r180 = alignment.rotate_points(pts, 180)
        r270 = alignment.rotate_points(pts, 270)
        # 0° 归一化后等于原点
        self.assertEqual(r0, pts)
        # 90° 与 270° 包围盒宽高互换
        self.assertEqual(max(p[0] for p in r90), 2)
        self.assertEqual(max(p[1] for p in r90), 2)
        # 手性：原三点构成 (1,1)->(1,2) 右、(1,1)->(2,1) 下；
        # 旋转后有向面积符号不变（镜像会变号）
        def signed_area(ps):
            (a, b, c) = ps
            return ((b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0]))
        signs = {signed_area(p) for p in (r0, r90, r180, r270)}
        self.assertEqual(len(signs), 1)
        # 转四次回到原位
        self.assertEqual(
            alignment.rotate_points(
                alignment.rotate_points(
                    alignment.rotate_points(
                        alignment.rotate_points(pts, 90), 90), 90), 90),
            pts)

    def test_rotation_preserves_point_order(self):
        pts = [(3, 5), (1, 2), (2, 8)]
        for ori in alignment.ORIENTATIONS:
            out = alignment.rotate_points(pts, ori)
            self.assertEqual(len(out), len(pts))


class SolveUniqueTests(unittest.TestCase):

    def setUp(self):
        # 8x8 全实测底图，温度按位置唯一可区分
        self.n = 8
        self.grid = [[100 + r * 10 + c for c in range(self.n)]
                     for r in range(self.n)]
        self.cells = cell_map(self.grid)
        # 表锚点 A=局部(1,1)、B=局部(1,2)，真实落点 (2,3)/(2,4)（0 基）
        self.a = (2, 3)
        self.b = (2, 4)

    def _sheet(self, fill_temp=500):
        ta = self.grid[self.a[0]][self.a[1]]
        tb = self.grid[self.b[0]][self.b[1]]
        return [
            sp(1, 1, ta, True),    # A 参照
            sp(1, 2, tb, True),    # B 参照
            sp(2, 1, fill_temp, False),   # 补测 P，0° 时落在 A 正下方 (3,3)
        ]

    def test_unique_placement_zero_degree(self):
        region = (0, 7, 0, 7)
        result = alignment.solve(self._sheet(), region, self.cells)
        self.assertEqual(result.status, "unique")
        self.assertEqual(len(result.placements), 1)
        self.assertEqual(result.placements[0].orientation, 0)
        self.assertEqual(fill_dict(result), {(3, 3): 500})

    def test_target_already_measured_does_not_filter_placement(self):
        # 补测落点 (3,3) 当前已测且温度不同：摆法仍可行（筛选不看目标格）
        self.assertIsNotNone(self.cells[(3, 3)])
        result = alignment.solve(self._sheet(fill_temp=999), (0, 7, 0, 7),
                                 self.cells)
        self.assertEqual(result.status, "unique")
        self.assertEqual(fill_dict(result), {(3, 3): 999})

    def test_region_constraint_excludes_placements(self):
        # 搜索区不含真实锚点：能落区的摆法参照全部失配 -> 无解
        result = alignment.solve(self._sheet(), (4, 7, 4, 7), self.cells)
        self.assertEqual(result.status, "none")
        self.assertIn("没有可行摆法", result.no_solution_reason)

    def test_reference_mismatch_no_solution(self):
        sheet = self._sheet()
        sheet[0] = sp(1, 1, sheet[0].temp + 1, True)   # 参照温度改错
        result = alignment.solve(sheet, (0, 7, 0, 7), self.cells)
        self.assertEqual(result.status, "none")
        self.assertIn("参照", result.no_solution_reason)

    def test_reference_landing_unmeasured_no_solution(self):
        cells = dict(self.cells)
        cells[self.a] = None                            # 参照落点未测
        result = alignment.solve(self._sheet(), (0, 7, 0, 7), cells)
        self.assertEqual(result.status, "none")

    def test_ninety_degree_orientation(self):
        # 真实锚点竖向排列 A=(2,3)、B=(3,3)，表中横向 (1,1)/(1,2)
        # -> 只有 90° 朝向能对上
        a, b = (2, 3), (3, 3)
        sheet = [
            sp(1, 1, self.grid[a[0]][a[1]], True),
            sp(1, 2, self.grid[b[0]][b[1]], True),
            sp(2, 1, 700, False),
        ]
        result = alignment.solve(sheet, (0, 7, 0, 7), self.cells)
        self.assertEqual(result.status, "unique")
        pl = result.placements[0]
        self.assertEqual(pl.orientation, 90)
        # 90° 顺时针：局部 (2,1) 旋转归一化后落在锚点 A 的左侧 (2,2)
        self.assertEqual(fill_dict(result), {(2, 2): 700})

    def test_multiple_placements_same_fill_is_unique(self):
        # 单点补测、无参照、1x1 搜索区：四种朝向都只能落在同一格，
        # 4 种摆法但补测落点完全相同 -> 允许预览
        sheet = [sp(1, 1, 424, False)]
        result = alignment.solve(sheet, (3, 3, 3, 3), self.cells)
        self.assertEqual(result.status, "unique")
        self.assertEqual(len(result.placements), 4)
        self.assertEqual(result.fill_landings, ((3, 3, 424),))

    def test_single_point_multiple_translations_is_ambiguous(self):
        # 单点在 3x3 搜索区有 9 个不同落点（每朝向相同），落点多组 -> 歧义
        sheet = [sp(1, 1, 424, False)]
        result = alignment.solve(sheet, (1, 3, 1, 3), self.cells)
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.distinct_fill_count, 9)

    def test_ambiguous_when_distinct_fills(self):
        # 无参照两点：不同平移给出不同补测落点组
        sheet = [sp(1, 1, 424, False), sp(1, 2, 425, False)]
        result = alignment.solve(sheet, (1, 4, 1, 4), self.cells)
        self.assertEqual(result.status, "ambiguous")
        self.assertGreater(result.distinct_fill_count, 1)
        self.assertGreater(len(result.placements), 1)

    def test_sheet_larger_than_region_no_solution(self):
        sheet = [sp(1, 1, 100, True), sp(3, 3, 100, False)]
        # 包围盒 3x3，搜索区 2x2，任何朝向都放不下
        result = alignment.solve(sheet, (0, 1, 0, 1), self.cells)
        self.assertEqual(result.status, "none")


class SheetParseTests(unittest.TestCase):

    def test_parse_ok_with_header_and_purposes(self):
        raw = (
            "local_row,local_col,temp,purpose\n"
            "1,1,421,参照\n"
            "1,2,422,ref\n"
            "2,1,500,补测\n"
            "2,2,501,fill\n"
        ).encode()
        pts = csvutil.parse_alignment_csv(raw)
        self.assertEqual(len(pts), 4)
        self.assertTrue(pts[0].is_ref and pts[1].is_ref)
        self.assertFalse(pts[2].is_ref and pts[3].is_ref)
        self.assertEqual(pts[2].temp, 500)

    def test_parse_gbk_encoding(self):
        raw = ("局部行,局部列,温度,用途\n1,1,421,参照\n1,2,500,补测\n"
               ).encode("gbk")
        pts = csvutil.parse_alignment_csv(raw)
        self.assertEqual(len(pts), 2)
        self.assertTrue(pts[0].is_ref)

    def test_bad_column_count(self):
        with self.assertRaises(alignment.SheetError):
            csvutil.parse_alignment_csv(b"1,1,421\n")

    def test_bad_purpose(self):
        with self.assertRaises(alignment.SheetError):
            csvutil.parse_alignment_csv("1,1,421,未知用途\n".encode())

    def test_duplicate_local_coord(self):
        raw = "1,1,421,参照\n1,1,422,补测\n".encode()
        with self.assertRaises(alignment.SheetError):
            csvutil.parse_alignment_csv(raw)

    def test_all_refs_rejected(self):
        raw = "1,1,421,参照\n1,2,422,参照\n".encode()
        with self.assertRaises(alignment.SheetError):
            csvutil.parse_alignment_csv(raw)

    def test_non_positive_local_and_bad_temp(self):
        with self.assertRaises(alignment.SheetError):
            csvutil.parse_alignment_csv("0,1,421,补测\n".encode())
        with self.assertRaises(alignment.SheetError):
            csvutil.parse_alignment_csv("1,1,99999,补测\n".encode())

    def test_empty_file(self):
        with self.assertRaises(alignment.SheetError):
            csvutil.parse_alignment_csv(b"  \n")


class AlignmentServiceTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "t.db")
        self.conn = db.connect(self.db_path)
        db.init_db(self.db_path)
        self.n = 8
        # 留出 (3,3) 未测，作为预期补测落点
        self.grid = [[100 + r * 10 + c for c in range(self.n)]
                     for r in range(self.n)]
        self.grid[3][3] = None
        db.create_grid(self.conn, self.n, self.grid, 20)
        self.a, self.b = (2, 3), (2, 4)
        self.sheet_bytes = self._sheet_bytes()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _sheet_bytes(self, fill_temp=500):
        ta = self.grid[self.a[0]][self.a[1]]
        tb = self.grid[self.b[0]][self.b[1]]
        return (
            "local_row,local_col,temp,purpose\n"
            f"1,1,{ta},参照\n"
            f"1,2,{tb},参照\n"
            f"2,1,{fill_temp},补测\n"
        ).encode()

    def _preview(self):
        pts = csvutil.parse_alignment_csv(self.sheet_bytes)
        cells = db.load_cells(self.conn)
        result = alignment.solve(pts, (0, 7, 0, 7), cells)
        self.assertEqual(result.status, "unique")
        db.save_alignment_preview(self.conn, (0, 7, 0, 7), result,
                                  self.sheet_bytes, "sheet.csv")
        return result

    def test_preview_persisted_and_loaded(self):
        result = self._preview()
        preview = db.load_alignment_preview(self.conn)
        self.assertIsNotNone(preview)
        self.assertEqual(preview["fills"], [(3, 3, 500)])
        self.assertEqual(preview["head"]["placement_count"], 1)
        # 预览不改任何读数
        self.assertIsNone(db.load_cells(self.conn)[(3, 3)])

    def test_repreview_replaces_old(self):
        self._preview()
        # 再做一份不同落点的预览：1x1 搜索区保证单点四朝向同落点、唯一
        pts = [sp(1, 1, 600, False)]
        cells = db.load_cells(self.conn)
        result = alignment.solve(pts, (5, 5, 5, 5), cells)
        self.assertEqual(result.status, "unique")
        db.save_alignment_preview(self.conn, (5, 5, 5, 5), result,
                                  b"1,1,600,fill\n", "s2.csv")
        preview = db.load_alignment_preview(self.conn)
        self.assertEqual(len(preview["fills"]), 1)
        self.assertEqual(preview["fills"][0][:2], (5, 5))
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM alignment_preview").fetchone()[0], 1)

    def test_confirm_success_fills_and_updates_blocks(self):
        self._preview()
        before_blocks = {
            (b.row, b.col, b.size) for b in db.load_current_blocks(self.conn)}
        metrics = db.confirm_alignment_preview(self.conn)
        self.assertEqual(metrics["filled_cells"], 1)
        cells = db.load_cells(self.conn)
        self.assertEqual(cells[(3, 3)], 500)
        # 预览已清除
        self.assertIsNone(db.load_alignment_preview(self.conn))
        # 现行供数块与全量重建一致
        from app import quadtree
        n = self.n
        stats = quadtree.compute_all_node_stats(cells, n)
        expect = quadtree.collect_blocks(stats, cells, n, 20)
        got = db.load_current_blocks(self.conn)
        self.assertEqual({(b.row, b.col, b.size, b.tmin, b.tmax, b.count)
                          for b in got},
                         {(b.row, b.col, b.size, b.tmin, b.tmax, b.count)
                          for b in expect})
        self.assertNotEqual(
            {(b.row, b.col, b.size) for b in got}, before_blocks)

    def test_confirm_without_preview_raises(self):
        with self.assertRaises(db.AlignmentConfirmError):
            db.confirm_alignment_preview(self.conn)

    def test_interleaved_edit_occupies_target_blocks_confirm(self):
        """交错补测：对位预览期间另一笔补测占用目标格，确认必须整批拒绝。"""
        self._preview()
        # 其他补测正常入库（要求：对位计算期间其他补测仍可入库）
        db.apply_edits(self.conn, {(3, 3): 424})
        snapshot_cells = dict(db.load_cells(self.conn))
        snapshot_blocks = [
            (b.row, b.col, b.size, b.tmin, b.tmax, b.count)
            for b in db.load_current_blocks(self.conn)]
        with self.assertRaises(db.AlignmentConfirmError) as ctx:
            db.confirm_alignment_preview(self.conn)
        self.assertIn("占用", str(ctx.exception))
        # 读数与供数块均未变，预览保留以便重试
        self.assertEqual(db.load_cells(self.conn), snapshot_cells)
        self.assertEqual(
            [(b.row, b.col, b.size, b.tmin, b.tmax, b.count)
             for b in db.load_current_blocks(self.conn)], snapshot_blocks)
        self.assertIsNotNone(db.load_alignment_preview(self.conn))

    def test_interleaved_reference_change_blocks_confirm(self):
        """重核后参照读数变化：重新对位无解，整批拒绝并提示重试。"""
        self._preview()
        db.apply_edits(self.conn, {self.a: 999})
        with self.assertRaises(db.AlignmentConfirmError) as ctx:
            db.confirm_alignment_preview(self.conn)
        self.assertIn("重核", str(ctx.exception))
        # 目标格仍未测、读数未被本次确认改动
        self.assertIsNone(db.load_cells(self.conn)[(3, 3)])

    def test_reupload_grid_clears_preview(self):
        self._preview()
        new_grid = [[r * 10 + c for c in range(self.n)] for r in range(self.n)]
        db.create_grid(self.conn, self.n, new_grid, 10)
        self.assertIsNone(db.load_alignment_preview(self.conn))
        # 旧预览的点表也被清掉
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM alignment_preview_point").fetchone()[0], 0)

    def test_tolerance_change_keeps_preview(self):
        self._preview()
        db.update_tolerance(self.conn, 50)
        preview = db.load_alignment_preview(self.conn)
        self.assertIsNotNone(preview)
        self.assertEqual(preview["fills"], [(3, 3, 500)])

    def test_cancel_preview(self):
        self._preview()
        self.assertTrue(db.cancel_alignment_preview(self.conn))
        self.assertIsNone(db.load_alignment_preview(self.conn))
        self.assertFalse(db.cancel_alignment_preview(self.conn))

    def test_confirm_rejects_even_when_temp_matches_existing(self):
        """目标格已测时即使温度恰好相等也必须拦截：规则要求目标格全未测。"""
        self._preview()
        db.apply_edits(self.conn, {(3, 3): 500})   # 与补测值相同
        before_bid = db.current_batch_id(self.conn)
        with self.assertRaises(db.AlignmentConfirmError):
            db.confirm_alignment_preview(self.conn)
        # 没有产生新批次，读数不变，预览保留
        self.assertEqual(db.current_batch_id(self.conn), before_bid)
        self.assertEqual(db.load_cells(self.conn)[(3, 3)], 500)
        self.assertIsNotNone(db.load_alignment_preview(self.conn))

    def test_failed_confirm_rolls_back_when_multiple_targets(self):
        # 两个补测点，其中一个被占用 -> 整批回滚，另一个也不得写入
        ta = self.grid[self.a[0]][self.a[1]]
        tb = self.grid[self.b[0]][self.b[1]]
        raw = (
            "local_row,local_col,temp,purpose\n"
            f"1,1,{ta},参照\n1,2,{tb},参照\n"
            "2,1,501,补测\n2,2,502,补测\n"
        ).encode()
        pts = csvutil.parse_alignment_csv(raw)
        cells = db.load_cells(self.conn)
        result = alignment.solve(pts, (0, 7, 0, 7), cells)
        self.assertEqual(result.status, "unique")
        db.save_alignment_preview(self.conn, (0, 7, 0, 7), result,
                                  raw, "multi.csv")
        # (3,4) 被其他补测占用；(3,3) 仍未测
        db.apply_edits(self.conn, {(3, 4): 424})
        with self.assertRaises(db.AlignmentConfirmError):
            db.confirm_alignment_preview(self.conn)
        cells = db.load_cells(self.conn)
        self.assertIsNone(cells[(3, 3)])       # 未被连带写入
        self.assertEqual(cells[(3, 4)], 424)   # 交错补测保留


class AlignmentWebTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "web.db")
        self.app = create_app(self.db_path)
        self.client = self.app.test_client()
        self.n = 8
        self.grid = [[100 + r * 10 + c for c in range(self.n)]
                     for r in range(self.n)]
        self.grid[3][3] = None
        self._upload_grid()
        self.a, self.b = (2, 3), (2, 4)

    def tearDown(self):
        self.tmp.cleanup()

    def _upload_grid(self):
        lines = [",".join("" if v is None else str(v) for v in row)
                 for row in self.grid]
        return self.client.post(
            "/upload",
            data={"csvfile": (io.BytesIO("\n".join(lines).encode()), "g.csv"),
                  "tol": "20"},
            content_type="multipart/form-data")

    def _sheet(self, fill_temp=500):
        ta = self.grid[self.a[0]][self.a[1]]
        tb = self.grid[self.b[0]][self.b[1]]
        return (
            "local_row,local_col,temp,purpose\n"
            f"1,1,{ta},参照\n1,2,{tb},参照\n"
            f"2,1,{fill_temp},补测\n"
        ).encode()

    def test_preview_confirm_cancel_flow(self):
        resp = self.client.post("/alignment/preview", data={
            "row_start": 1, "row_end": 8, "col_start": 1, "col_end": 8,
            "sheet": (io.BytesIO(self._sheet()), "sheet.csv"),
        }, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        text = resp.get_data(as_text=True)
        self.assertIn("对位成功", text)
        self.assertIn("当前对位预览", text)
        # 0 基 (3,3) 的补测点以 1 基行列 4 / 4 分两列展示
        self.assertIn("<td>4</td><td>4</td>", text)
        self.assertIn("500", text)

        # 确认写入
        resp = self.client.post("/alignment/confirm", follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("重核通过", resp.get_data(as_text=True))
        # 下载供数仍正常
        self.assertEqual(self.client.get("/download").status_code, 200)

    def test_ambiguous_flash_and_no_preview(self):
        raw = b"1,1,424,fill\n1,2,425,fill\n"
        resp = self.client.post("/alignment/preview", data={
            "row_start": 1, "row_end": 8, "col_start": 1, "col_end": 8,
            "sheet": (io.BytesIO(raw), "a.csv"),
        }, content_type="multipart/form-data", follow_redirects=True)
        text = resp.get_data(as_text=True)
        self.assertIn("多组", text)
        self.assertNotIn("当前对位预览", text)

    def test_bad_sheet_does_not_pollute(self):
        before = self.client.get("/download").get_data(as_text=True)
        resp = self.client.post("/alignment/preview", data={
            "row_start": 1, "row_end": 8, "col_start": 1, "col_end": 8,
            "sheet": (io.BytesIO("1,1,421,乱填\n".encode()), "bad.csv"),
        }, content_type="multipart/form-data", follow_redirects=True)
        self.assertIn("用途", resp.get_data(as_text=True))
        after = self.client.get("/download").get_data(as_text=True)
        self.assertEqual(before, after)

    def test_confirm_after_interleaved_fill_is_rejected(self):
        self.client.post("/alignment/preview", data={
            "row_start": 1, "row_end": 8, "col_start": 1, "col_end": 8,
            "sheet": (io.BytesIO(self._sheet()), "sheet.csv"),
        }, content_type="multipart/form-data")
        # 交错补测占用 (3,3)（1 基 4,4）
        self.client.post("/edit", data={
            "mode": "fill", "row_start": 4, "row_end": 4,
            "col_start": 4, "col_end": 4, "temp": "424",
        })
        resp = self.client.post("/alignment/confirm",
                                follow_redirects=True)
        self.assertIn("占用", resp.get_data(as_text=True))

    def test_cancel_then_confirm_is_empty(self):
        self.client.post("/alignment/preview", data={
            "row_start": 1, "row_end": 8, "col_start": 1, "col_end": 8,
            "sheet": (io.BytesIO(self._sheet()), "s.csv"),
        }, content_type="multipart/form-data")
        self.client.post("/alignment/cancel")
        resp = self.client.post("/alignment/confirm", follow_redirects=True)
        self.assertIn("没有可确认", resp.get_data(as_text=True))

    def test_preview_requires_grid(self):
        # 新应用、未上传底图
        app2 = create_app(os.path.join(self.tmp.name, "empty.db"))
        client2 = app2.test_client()
        resp = client2.post("/alignment/preview", data={
            "row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2,
            "sheet": (io.BytesIO(b"1,1,10,fill\n"), "s.csv"),
        }, content_type="multipart/form-data", follow_redirects=True)
        self.assertIn("先上传", resp.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
