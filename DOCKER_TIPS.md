# Docker 部署踩坑总结（通用 checklist）

> 基于 VisualHunt 项目部署经验整理，适用于后续其他项目的 Docker 化部署。

---

## 一、构建阶段

### 1.1 Python 版本必须一致

如果采用**宿主机预构建 .venv + COPY 进容器**的方案：

- 宿主机的 Python 版本必须和 Dockerfile 基础镜像一致
- 例：宿主机用 Python 3.12 构建 .venv，Dockerfile 必须是 `FROM python:3.12-slim`
- **不能**宿主机 3.12 + 容器 3.10，否则 `.venv/lib/python3.12/site-packages` 容器找不到

```dockerfile
# ✅ 正确
FROM python:3.12-slim

# ❌ 错误（和宿主机 .venv 版本不一致）
FROM python:3.10-slim
```

### 1.2 .venv 软链接修复

`uv sync` 创建的 `.venv/bin/python` 通常是指向宿主机绝对路径的软链接：

```
.venv/bin/python -> /home/ubuntu/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
```

COPY 到容器后这个路径不存在，需要在 Dockerfile 里重新链接：

```dockerfile
RUN rm -f /app/.venv/bin/python /app/.venv/bin/python3 /app/.venv/bin/python3.12 && \
    ln -s $(which python3.12) /app/.venv/bin/python && \
    ln -s $(which python3.12) /app/.venv/bin/python3 && \
    ln -s $(which python3.12) /app/.venv/bin/python3.12
```

> 注意：`rm -f` 要把目标文件名都删掉，否则 `ln -s` 会报 "File exists"。

### 1.3 apt 源换国内镜像

Debian/Ubuntu 默认源在国外，构建时 `apt-get update` 可能卡几分钟。Dockerfile 里先换源：

```dockerfile
RUN sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' \
    /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' \
    /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*
```

### 1.4 .venv 必须在 Linux 下构建

Windows 的 `.venv` 不能 COPY 到 Linux 容器运行（PyTorch 等包含平台特定二进制 `.dll` / `.so`）。

**必须在 Linux 服务器（或 Linux 构建机）上执行 `uv sync`**。

---

## 二、端口配置

### 2.1 三处端口必须统一

| 位置 | 说明 |
|------|------|
| `runtime_config.json` / 代码 | 服务实际监听的端口 |
| `Dockerfile` | `EXPOSE xxx` |
| `docker-compose.yml` | `ports: - "xxx:xxx"` |

### 2.2 容器内必须监听 0.0.0.0

Flask/Django/FastAPI 默认监听 `127.0.0.1`，容器外无法访问：

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 5176
  }
}
```

> 生产环境 `debug` 设为 `false`。

---

## 三、Nginx 反向代理（多项目共存）

### 3.1 前端请求必须用相对路径

前端 `fetch` / `axios` 不能用绝对路径：

```javascript
// ❌ 错误：浏览器会请求 http://IP/api/xxx（丢失 /projectA 前缀）
fetch('/api/check_models')

// ✅ 正确：浏览器根据当前页面路径自动拼接前缀
fetch('./api/check_models')
```

### 3.2 Nginx 强制末尾斜杠跳转

访问 `/projectA`（无斜杠）时，浏览器解析 `./api/xxx` 会变成 `/api/xxx`，导致 404。

必须加跳转规则：

```nginx
location = /projectA {
    return 301 /projectA/;
}

location /projectA/ {
    proxy_pass http://container:port/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

### 3.3 proxy_pass 末尾斜杠的含义

```nginx
# ✅ 正确：会把 /projectA/xxx 转发为 /xxx
proxy_pass http://container:port/;

# ❌ 错误（无斜杠）：会把 /projectA/xxx 转发为 /projectA/xxx
proxy_pass http://container:port;
```

---

## 四、代码热更新（不用反复 build 镜像）

### 4.1 挂载源代码目录

默认 `COPY . /app/` 进镜像后，代码修改必须重新 `docker build`。

在 `docker-compose.yml` 里把源码目录挂载为 volume：

```yaml
services:
  app:
    volumes:
      - ./src:/app/src
      - ./main.py:/app/main.py
      - ./config.json:/app/config.json
```

> 只挂载代码，不挂载 `.venv`（容器内用镜像里的 .venv）。

### 4.2 修改 docker-compose.yml 后必须重新创建容器

**`docker compose restart` 不会重新读取 docker-compose.yml 的变更！**

修改 volume 映射、端口、环境变量等配置后，必须：

```bash
# ✅ 正确：重新创建容器
sudo docker compose down
sudo docker compose up -d

# ❌ 错误：restart 只重启现有容器，不会应用新配置
sudo docker compose restart
```

### 4.3 更新流程总结

| 变更内容 | 操作 |
|---------|------|
| 前端/后端代码 | `git pull` + `docker compose restart` |
| docker-compose.yml / Dockerfile | `docker compose down && docker compose up -d` |
| 依赖（pyproject.toml）| `uv sync` + `docker compose up -d --build` |
| 配置（runtime_config.json）| 直接改文件，`docker compose restart` |

---

## 五、网络配置

### 5.1 Nginx 和项目容器要在同一个网络

多项目共存时，Nginx 容器需要能解析项目容器的主机名：

```yaml
# docker-compose.yml
services:
  app:
    networks:
      - projects-net

networks:
  projects-net:
    external: true
```

创建公共网络：

```bash
docker network create projects-net
```

### 5.2 Nginx 里用容器名访问

```nginx
proxy_pass http://visualhunt:5176/;
```

> 不要写 `127.0.0.1:5176`，那是宿主机端口；容器间通信用容器名 + 容器内部端口。

---

## 六、浏览器缓存

前端修改后，浏览器可能缓存旧的 `index.html`。用户需要强制刷新：

- **Windows**: `Ctrl + F5`
- **Mac**: `Cmd + Shift + R`

或者 Nginx 里加禁用缓存头（开发环境）：

```nginx
location / {
    add_header Cache-Control "no-cache, no-store, must-revalidate";
}
```

---

## 七、快速 checklist

配置新项目时，逐条检查：

- [ ] 宿主机 Python 版本和 Dockerfile 基础镜像版本一致
- [ ] Dockerfile 里修复 `.venv/bin/python` 软链接
- [ ] Dockerfile apt 源换成国内镜像
- [ ] 端口三处统一（代码 / Dockerfile / docker-compose）
- [ ] 容器内服务监听 `0.0.0.0`
- [ ] 前端 `fetch` 使用相对路径（`./api/xxx`）
- [ ] Nginx 配置 `/project` → `/project/` 跳转
- [ ] `proxy_pass` 末尾带 `/`
- [ ] docker-compose.yml 挂载 `src/` 和入口文件
- [ ] Nginx 容器和项目容器在同一 Docker network
- [ ] 修改 compose 配置后执行 `down && up -d`，不是 `restart`
