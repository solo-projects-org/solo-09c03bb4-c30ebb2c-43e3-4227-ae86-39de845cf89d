FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FURNACE_DB=/data/furnace.db

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 单 worker + 多线程：SQLite 单库写事务串行，单 worker 可避免库级写锁竞争；
# WAL 模式下读写互不阻塞，线程内短事务足以支撑验收现场使用。
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", \
     "--timeout", "120", "app.wsgi:app"]
