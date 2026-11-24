# -*- coding: utf-8 -*-
"""Flask 路由：上传、补测/清空、温差维护、块查询、下载。"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

from flask import (Flask, Response, abort, flash, redirect, render_template,
                   request, url_for)

from . import csvutil, db
from .csvutil import CSVParseError, TEMP_MAX, TEMP_MIN
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
            return render_template(
                "index.html", grid=grid, stats=stats, batch=batch,
                grid_rows=grid_rows, view_n=view_n, show_full=show_full,
                blocks=view_blocks, total_blocks=len(blocks), n=n,
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
