# VisualHunt Docker 部署文档（Ubuntu）

> 目标服务器：Ubuntu 20.04/22.04/24.04  
> 部署路径：`/opt/visualHunt`  
> 代码仓库：https://gitee.com/myparadises/visual-hunt.git  
> 服务端口：5000（容器内部端口，对外通过 Nginx 反向代理暴露）

---

## 前置要求

- 服务器已安装 **Docker** 和 **Docker Compose**
- 服务器已有 **Python 3.10+** 和 **uv**（用于宿主机预构建 `.venv`）
- **`.venv` 必须在 Linux x86_64 环境下构建**，Windows/macOS 构建的 `.venv` 无法直接复制到 Linux 容器运行

---

## 一、服务器环境准备

### 1.1 安装 Docker & Docker Compose

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Docker
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 验证
sudo docker --version
sudo docker compose version
```

### 1.2 安装 uv（宿主机包管理器）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

---

## 二、拉取项目代码

```bash
sudo mkdir -p /opt
sudo git clone -b dev https://gitee.com/myparadises/visual-hunt.git /opt/visualHunt
sudo chown -R $(whoami):$(whoami) /opt/visualHunt
```

> **注意**：`dev` 分支已配置为 CPU 版 PyTorch，且仓库中已包含 `finetuned/` 目录下的预训练模型。

---

## 三、宿主机预构建 .venv

**核心原则**：在宿主机上先用 `uv sync` 装好所有依赖，构建 Docker 镜像时直接把 `.venv` 复制进去，避免容器内重复下载。

```bash
cd /opt/visualHunt

# 创建虚拟环境并安装依赖（CPU 版 PyTorch）
uv sync

# 验证 CLI
uv run vh --help
```

> 如果执行报错，确认 `pyproject.toml` 中 PyTorch 源为 `https://download.pytorch.org/whl/cpu`。

---

## 四、修改运行配置

容器内 Flask 必须监听 `0.0.0.0`，否则宿主机和 Nginx 无法访问到容器服务。

```bash
cd /opt/visualHunt
vim runtime_config.json
```

改为：

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false
  }
}
```

> - `host` 必须是 `0.0.0.0`，容器内只监听 `127.0.0.1` 会导致外部无法访问。
> - `debug` 生产环境设为 `false`。

---

## 五、构建并启动容器

### 5.1 构建镜像

```bash
cd /opt/visualHunt
sudo docker build -t visualhunt:dev .
```

构建过程说明：
- Dockerfile 会复制整个项目（含 `.venv`、代码、模型）到 `/app`
- 修复 `.venv/bin/python*` 的软链接，使其指向容器内的 Python 3.10
- 暴露 5000 端口

### 5.2 使用 Docker Compose 启动（推荐）

```bash
sudo docker compose up -d
```

查看日志：

```bash
sudo docker logs -f visualhunt
```

常用命令：

```bash
sudo docker compose up -d        # 后台启动
sudo docker compose down         # 停止并移除容器
sudo docker compose restart      # 重启
sudo docker compose pull && sudo docker compose up -d --build   # 更新后重新构建
```

> `docker-compose.yml` 中已将 `./data`、`./finetuned` 和 `runtime_config.json` 挂载为 volume，后续修改数据和配置无需重新构建镜像。

### 5.3 不使用 Compose 直接运行

```bash
sudo docker run -d \
  --name visualhunt \
  -p 5000:5000 \
  -v /opt/visualHunt/data:/app/data \
  -v /opt/visualHunt/finetuned:/app/finetuned \
  -v /opt/visualHunt/runtime_config.json:/app/runtime_config.json \
  --restart unless-stopped \
  visualhunt:dev
```

---

## 六、Nginx 反向代理配置（多项目共存）

服务器上还有其他项目，使用 Nginx 做反向代理，通过**路径区分**或**子域名**暴露 VisualHunt。

### 方式 A：路径区分（推荐）

假设服务器 IP 为 `1.2.3.4`，将 VisualHunt 挂在 `/visualhunt/` 路径下：

```bash
sudo vim /etc/nginx/sites-available/visualhunt
```

```nginx
server {
    listen 80;
    server_name 1.2.3.4;  # 或你的域名

    location /visualhunt/ {
        proxy_pass http://127.0.0.1:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Prefix /visualhunt;
        proxy_redirect off;
    }

    location /visualhunt/static/ {
        alias /opt/visualHunt/src/api/static/;
    }
}
```

启用：

```bash
sudo ln -sf /etc/nginx/sites-available/visualhunt /etc/nginx/sites-enabled/visualhunt
sudo nginx -t
sudo systemctl reload nginx
```

> 如果前端链接异常，可在 `src/api/app.py` 中增加 `ProxyFix`：
> ```python
> from werkzeug.middleware.proxy_fix import ProxyFix
> app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)
> ```

### 方式 B：独立子域名

```nginx
server {
    listen 80;
    server_name visualhunt.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static/ {
        alias /opt/visualHunt/src/api/static/;
    }
}
```

---

## 七、防火墙配置

开放 80 端口（Nginx），容器映射的 5000 端口**不需要**对外暴露：

```bash
sudo ufw allow 80/tcp
sudo ufw reload
```

> 云服务器还需在厂商控制台的安全组中放行 80 端口。

---

## 八、验证部署

1. **容器状态检查**：
   ```bash
   sudo docker ps | grep visualhunt
   sudo docker logs visualhunt
   ```

2. **本机访问测试**：
   ```bash
   curl http://127.0.0.1:5000
   ```

3. **Nginx 代理测试**：
   ```bash
   curl http://1.2.3.4/visualhunt/        # 方式 A
   curl http://visualhunt.yourdomain.com   # 方式 B
   ```

---

## 九、后续维护

### 9.1 更新代码

```bash
cd /opt/visualHunt
git pull origin dev

# 如果依赖有变化，需要重新构建 .venv 和镜像
uv sync
sudo docker compose down
sudo docker compose up -d --build
```

### 9.2 重新训练模型（可选）

```bash
cd /opt/visualHunt

# 在宿主机上训练（直接使用 .venv）
uv run vh train --task classification
uv run vh train --task denoising
uv run vh train --task similarity

# 重启容器加载新模型
sudo docker compose restart
```

### 9.3 查看日志与排查

```bash
# 容器实时日志
sudo docker logs -f visualhunt

# Nginx 日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## 十、目录结构（部署后）

```text
/opt/visualHunt/
├── .venv/                  # 宿主机预构建的虚拟环境（会被复制到镜像）
├── data/
│   ├── dataset/            # 样本图片
│   └── fashion-labels.csv
├── finetuned/              # 预训练模型
│   ├── classifier.pt
│   ├── decoder.pt
│   ├── denoiser.pt
│   ├── encoder.pt
│   └── embeddings.npy
├── src/
│   ├── api/                # Flask 服务
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── models.py
│   └── utils.py
├── Dockerfile              # Docker 构建文件
├── docker-compose.yml      # Docker Compose 配置
├── .dockerignore           # Docker 构建上下文排除规则
├── main.py                 # 启动入口
├── runtime_config.json     # 运行配置（host 需为 0.0.0.0）
├── pyproject.toml
└── README.md
```

---

## 附录：一键部署脚本

在服务器上保存为 `deploy.sh` 并执行：

```bash
#!/bin/bash
set -e

PROJECT_DIR="/opt/visualHunt"
REPO_URL="https://gitee.com/myparadises/visual-hunt.git"

echo "=== 1. 克隆项目 ==="
sudo mkdir -p /opt
if [ ! -d "$PROJECT_DIR/.git" ]; then
    sudo git clone -b dev "$REPO_URL" "$PROJECT_DIR"
fi
sudo chown -R "$(whoami):$(whoami)" "$PROJECT_DIR"

echo "=== 2. 安装 uv ==="
if [ ! -f "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

echo "=== 3. 宿主机构建 .venv ==="
cd "$PROJECT_DIR"
uv sync

echo "=== 4. 修改 runtime_config.json ==="
# 确保 host 为 0.0.0.0
python3 -c "
import json
with open('runtime_config.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)
cfg['api']['host'] = '0.0.0.0'
cfg['api']['debug'] = False
with open('runtime_config.json', 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
"

echo "=== 5. 构建并启动 Docker 容器 ==="
sudo docker compose -f "$PROJECT_DIR/docker-compose.yml" down || true
sudo docker compose -f "$PROJECT_DIR/docker-compose.yml" up -d --build

echo "=== 部署完成 ==="
echo "查看容器状态: sudo docker ps | grep visualhunt"
echo "查看实时日志: sudo docker logs -f visualhunt"
echo "请手动配置 Nginx 反向代理"
```

执行：

```bash
chmod +x deploy.sh
./deploy.sh
```

---

> 如有问题，请检查 `sudo docker logs visualhunt` 和 Nginx 错误日志进行排查。
