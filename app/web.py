# -*- coding: utf-8 -*-
"""Flask 路由：上传、补测/清空、温差维护、块查询、下载。"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

from flask import (Flask, Response, abort, flash, redirect, render_template,
                   request, url_for)

from . import alignment, csvutil, db
from .alignment import SheetError
from .csvutil import CSVParseError, TEMP_MAX, TEMP_MIN
from .db import AlignmentConfirmError
from .quadtree import is_power_of_two


def create_app(db_path: Optional[str] = None) -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "furnace-acceptance-dev-key"
    app.config["DB_PATH"] = db_path
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    db.init_db(db_path)

    def conn():
        return db.connect(app.config["DB_PATH"])

    # --------------------------- 主页 ---------------------------
    @app.route("/")
    def index():
        c = conn()
        try:
            grid = db.get_grid(c)
            if grid is None:
                return render_template("index.html", grid=None)
            n = grid["side"]
            cells = db.load_cells(c)
            blocks = db.load_current_blocks(c)
            bid = db.current_batch_id(c)
            batch = c.execute("SELECT * FROM batch WHERE id=?", (bid,)).fetchone() if bid else None
            measured = sum(1 for v in cells.values() if v is not None)
            t_values = [v for v in cells.values() if v is not None]
            stats = {
                "measured": measured,
                "unmeasured": n * n - measured,
                "tmax": max(t_values) if t_values else None,
                "tmin": min(t_values) if t_values else None,
                "block_count": len(blocks),
            }
            # 小网格整表渲染；大网格只渲染前 32 行/列，界面提示用批量操作
            preview_limit = 32
            show_full = n <= preview_limit
            view_n = n if show_full else preview_limit
            grid_rows = []
            for r in range(view_n):
                row = []
                for cc in range(view_n):
                    row.append(cells.get((r, cc)))
                grid_rows.append(row)
            # 预览视图内可见的块（左上角落在预览区）
            view_blocks = [b for b in blocks if b.row < view_n and b.col < view_n]
            align_preview = db.load_alignment_preview(c)
            return render_template(
                "index.html", grid=grid, stats=stats, batch=batch,
                grid_rows=grid_rows, view_n=view_n, show_full=show_full,
                blocks=view_blocks, total_blocks=len(blocks), n=n,
                align_preview=align_preview, cells_map=cells,
            )
        finally:
            c.close()

    # --------------------------- 上传 ---------------------------
    @app.route("/upload", methods=["POST"])
    def upload():
        f = request.files.get("csvfile")
        if f is None or f.filename == "":
            flash("请选择要上传的 CSV 文件", "error")
            return redirect(url_for("index"))
        tol_raw = request.form.get("tol", "").strip()
        try:
            tol = int(tol_raw)
            if tol < 0:
                raise ValueError
        except ValueError:
            flash("允许温差必须是不小于 0 的整数（℃）", "error")
            return redirect(url_for("index"))
        try:
            raw = f.read()
            n, data = csvutil.parse_square_csv(raw)
            c = conn()
            try:
                metrics = db.create_grid(c, n, data, tol)
            finally:
                c.close()
        except CSVParseError as exc:
            flash(f"CSV 解析失败，未写入任何数据：{exc}", "error")
            return redirect(url_for("index"))
        except Exception as exc:  # 入库失败同样不能污染旧数据
            app.logger.exception("upload failed")
            flash(f"写入失败，已回滚，旧表仍可下载：{exc}", "error")
            return redirect(url_for("index"))
        flash(f"上传成功：{n}×{n} 方阵，重算节点 {metrics['nodes_recomputed']} 个，"
              f"生成覆盖块 {metrics['blocks_added']} 个", "ok")
        return redirect(url_for("index"))

    # --------------------- 补测 / 清空（批量） ---------------------
    @app.route("/edit", methods=["POST"])
    def edit():
        mode = request.form.get("mode", "fill")
        try:
            r1, r2, c1, c2 = _parse_range(request.form)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

        c = conn()
        try:
            grid = db.get_grid(c)
            if grid is None:
                flash("请先上传方阵 CSV", "error")
                return redirect(url_for("index"))
            n = grid["side"]
            for v, name in ((r1, "起始行"), (r2, "结束行"), (c1, "起始列"), (c2, "结束列")):
                if not (1 <= v <= n):
                    raise ValueError(f"{name} {v} 超出 1..{n} 范围")
            if r2 < r1 or c2 < c1:
                raise ValueError("起止顺序有误：结束行/列不能小于起始行/列")

            edits: Dict[Tuple[int, int], Optional[int]] = {}
            if mode == "clear":
                for r in range(r1 - 1, r2):
                    for cc in range(c1 - 1, c2):
                        edits[(r, cc)] = None
            else:
                temp_raw = request.form.get("temp", "").strip()
                try:
                    temp = int(temp_raw)
                except ValueError:
                    raise ValueError("补测温度必须是整数摄氏度")
                if not (TEMP_MIN <= temp <= TEMP_MAX):
                    raise ValueError(f"补测温度 {temp} 超出合理范围 [{TEMP_MIN}, {TEMP_MAX}] ℃")
                for r in range(r1 - 1, r2):
                    for cc in range(c1 - 1, c2):
                        edits[(r, cc)] = temp

            try:
                metrics = db.apply_edits(c, edits)
            except Exception as exc:
                app.logger.exception("edit failed")
                flash(f"补测/清空失败，已回滚，旧表仍可下载：{exc}", "error")
                return redirect(url_for("index"))
        except ValueError as exc:
            flash(f"输入有误，未写入：{exc}", "error")
            return redirect(url_for("index"))
        finally:
            c.close()

        if metrics.get("changed_cells", 0) == 0:
            flash("所选格子的现值与提交值一致，未生成新批次", "ok")
        else:
            flash(
                f"改动 {metrics['changed_cells']} 格；仅重算包含改动格的节点 "
                f"{metrics['nodes_recomputed']} 个，块增 {metrics['blocks_added']} / "
                f"删 {metrics['blocks_removed']}", "ok")
        return redirect(url_for("index"))

    # ------------------- 热像仪补测表对位 -------------------
    @app.route("/alignment/preview", methods=["POST"])
    def alignment_preview():
        """上传补测表并在搜索区内对位；唯一落点才生成预览，否则只说明原因。"""
        f = request.files.get("sheet")
        if f is None or f.filename == "":
            flash("请选择热像仪补测表 CSV（局部行, 局部列, 温度, 用途）", "error")
            return redirect(url_for("index"))
        try:
            r1, r2, c1, c2 = _parse_range(request.form)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

        c = conn()
        try:
            grid = db.get_grid(c)
            if grid is None:
                flash("请先上传方阵 CSV 底图", "error")
                return redirect(url_for("index"))
            n = grid["side"]
            try:
                region = alignment.validate_region((r1 - 1, r2 - 1, c1 - 1, c2 - 1), n)
            except SheetError as exc:
                flash(f"搜索区有误，未做对位：{exc}", "error")
                return redirect(url_for("index"))
            raw = f.read()
            try:
                points = csvutil.parse_alignment_csv(raw)
            except SheetError as exc:
                flash(f"补测表解析失败，未生成预览：{exc}", "error")
                return redirect(url_for("index"))

            cells = db.load_cells(c)
            result = alignment.solve(points, region, cells)
            if result.status == "none":
                flash(result.no_solution_reason, "error")
                return redirect(url_for("index"))
            if result.status == "ambiguous":
                flash(
                    f"存在多组可行摆法（{len(result.placements)} 种摆法、"
                    f"{result.distinct_fill_count} 组不同的补测落点），"
                    "按规则不得预览；请缩小搜索区、增加参照点或核对补测表后重试",
                    "error")
                return redirect(url_for("index"))

            db.save_alignment_preview(c, region, result, raw, f.filename)
        except Exception:
            app.logger.exception("alignment preview failed")
            flash("对位预览失败，已回滚，既有数据与预览不变", "error")
            return redirect(url_for("index"))
        finally:
            c.close()
        flash(
            f"对位成功：{len(result.placements)} 种可行摆法均给出同一组补测落点，"
            f"共 {len(result.fill_landings)} 个补测点（参照点 "
            f"{len(result.ref_landings)} 个已与实测吻合）；落点见下方预览，"
            "确认前不会写入任何读数", "ok")
        return redirect(url_for("index"))

    @app.route("/alignment/confirm", methods=["POST"])
    def alignment_confirm():
        """用最新实测重核预览；通过则整批填入并更新供数块，否则回滚并提示重试。"""
        c = conn()
        try:
            try:
                metrics = db.confirm_alignment_preview(c)
            except AlignmentConfirmError as exc:
                flash(f"{exc}；读数与供数块均未改动", "error")
                return redirect(url_for("index"))
            except Exception:
                app.logger.exception("alignment confirm failed")
                flash("对位确认失败，已回滚，读数与供数块均未改动", "error")
                return redirect(url_for("index"))
        finally:
            c.close()
        flash(
            f"重核通过，整批填入补测 {metrics['filled_cells']} 格；"
            f"重算节点 {metrics['nodes_recomputed']} 个，现行供数块增 "
            f"{metrics['blocks_added']} / 删 {metrics['blocks_removed']}", "ok")
        return redirect(url_for("index"))

    @app.route("/alignment/cancel", methods=["POST"])
    def alignment_cancel():
        c = conn()
        try:
            existed = db.cancel_alignment_preview(c)
        finally:
            c.close()
        flash("已放弃当前对位预览，未写入任何读数" if existed
              else "当前没有可放弃的对位预览", "ok" if existed else "error")
        return redirect(url_for("index"))

    # --------------------------- 温差 ---------------------------
    @app.route("/tolerance", methods=["POST"])
    def tolerance():
        try:
            tol = int(request.form.get("tol", "").strip())
            if tol < 0:
                raise ValueError
        except ValueError:
            flash("允许温差必须是不小于 0 的整数（℃）", "error")
            return redirect(url_for("index"))
        c = conn()
        try:
            if db.get_grid(c) is None:
                flash("请先上传方阵 CSV", "error")
                return redirect(url_for("index"))
            try:
                metrics = db.update_tolerance(c, tol)
            except Exception as exc:
                app.logger.exception("tolerance update failed")
                flash(f"温差更新失败，已回滚，旧表仍可下载：{exc}", "error")
                return redirect(url_for("index"))
        finally:
            c.close()
        flash(f"允许温差已更新为 {tol} ℃，重算节点 {metrics['nodes_recomputed']} 个，"
              f"现行块数 {metrics['blocks_added']}", "ok")
        return redirect(url_for("index"))

    # --------------------------- 块列表 ---------------------------
    @app.route("/blocks")
    def blocks_view():
        c = conn()
        try:
            grid = db.get_grid(c)
            if grid is None:
                flash("请先上传方阵 CSV", "error")
                return redirect(url_for("index"))
            blocks = db.load_current_blocks(c)
            bid = db.current_batch_id(c)
            batch = c.execute("SELECT * FROM batch WHERE id=?", (bid,)).fetchone()
            return render_template("blocks.html", blocks=blocks, grid=grid, batch=batch)
        finally:
            c.close()

    # --------------------------- 下载 ---------------------------
    @app.route("/download")
    def download():
        c = conn()
        grid = db.get_grid(c)
        if grid is None:
            c.close()
            abort(404, description="尚无已生效数据，无法下载")
        bid = db.current_batch_id(c)
        if bid is None:
            c.close()
            abort(404, description="尚无已生效批次")

        n = grid["side"]

        def generate():
            # 块与批次在同一连接的同一读取流中产出，
            # 保证位置/边长/温度来自同一批已生效数据
            try:
                yield from csvutil.iter_blocks_csv(
                    db.iter_current_blocks(c), n)
            finally:
                c.close()

        fname = f"furnace_blocks_{n}x{n}_batch{bid}.csv"
        return Response(
            generate(),
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={fname}"},
        )

    # --------------------------- 事件 ---------------------------
    @app.route("/events")
    def events():
        c = conn()
        try:
            rows = c.execute(
                "SELECT * FROM event_log ORDER BY id DESC LIMIT 100").fetchall()
            grid = db.get_grid(c)
            return render_template("events.html", rows=rows, grid=grid)
        finally:
            c.close()

    @app.errorhandler(404)
    def not_found(err):
        return render_template("error.html", message=err.description), 404

    return app


def _parse_range(form) -> Tuple[int, int, int, int]:
    try:
        r1 = int(form.get("row_start", "").strip())
        r2 = int(form.get("row_end", "").strip())
        c1 = int(form.get("col_start", "").strip())
        c2 = int(form.get("col_end", "").strip())
    except (AttributeError, ValueError):
        raise ValueError("行、列号必须是 1 基整数")
    return r1, r2, c1, c2
