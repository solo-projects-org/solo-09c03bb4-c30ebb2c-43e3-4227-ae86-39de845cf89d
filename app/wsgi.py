# -*- coding: utf-8 -*-
"""gunicorn / flask 运行入口：python -m app.wsgi 可本地起开发服务器。"""

from .web import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
