FROM python:3.10-slim

WORKDIR /app

# 安装 PyTorch 等原生扩展可能依赖的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制整个项目到容器（包含代码、模型、配置以及预构建的 .venv）
COPY . /app/

# 修复 .venv 中的 Python 解释器软链接。
# 宿主机上 uv 创建的 .venv/bin/python 可能是指向宿主机绝对路径的软链接，
# 在容器内会失效，因此重新链接到容器自带的 Python 3.10。
RUN rm -f /app/.venv/bin/python /app/.venv/bin/python3 /app/.venv/bin/python3.10 && \
    ln -s $(which python3.10) /app/.venv/bin/python && \
    ln -s $(which python3.10) /app/.venv/bin/python3 && \
    ln -s $(which python3.10) /app/.venv/bin/python3.10

EXPOSE 5000

# 使用容器内 .venv 的 Python 直接启动服务
CMD ["/app/.venv/bin/python", "main.py"]
