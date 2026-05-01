# VisualHunt Ubuntu 服务器部署文档

> 目标服务器：Ubuntu 20.04/22.04/24.04  
> 部署路径：`/opt/visualHunt`  
> 代码仓库：https://gitee.com/myparadises/visual-hunt.git  
> 服务端口：5176（Flask 内部端口，对外通过 Nginx 反向代理暴露）

---

## 一、服务器环境准备

### 1.1 更新系统并安装基础依赖

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git wget curl vim nginx
```

### 1.2 确认 Python 版本

本项目需要 Python >= 3.10，你的服务器已有 Python 3.10.12，无需额外安装：

```bash
python3 --version   # Python 3.10.12
```

### 1.3 安装 uv（包管理器）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

> uv 默认安装在 `$HOME/.local/bin`，安装后执行 `source $HOME/.local/bin/env` 加载 PATH。

如果镜像较慢，可配置国内 pip 源：

```bash
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF
```

---

## 二、拉取项目代码

```bash
sudo mkdir -p /opt
sudo git clone https://gitee.com/myparadises/visual-hunt.git /opt/visualHunt
sudo chown -R $(whoami):$(whoami) /opt/visualHunt
```

> **注意**：仓库中已包含 `finetuned/` 目录下的预训练模型（`classifier.pt`、`decoder.pt`、`denoiser.pt`、`encoder.pt`、`embeddings.npy`），无需单独下载。

---

## 三、安装项目依赖（CPU 版本）

`dev` 分支已配置为 CPU 版 PyTorch：

```bash
cd /opt/visualHunt

# 使用 uv 创建虚拟环境并安装依赖
uv sync

# 验证 CLI
uv run vh --help
```

> 当前 `pyproject.toml` 中 PyTorch 源已指向 `https://download.pytorch.org/whl/cpu`，适用于无 GPU 的服务器。

---

## 四、配置运行参数

编辑 `runtime_config.json`，根据生产环境调整 API 参数：

```bash
cd /opt/visualHunt
vim runtime_config.json
```

建议生产环境配置：

```json
{
  "api": {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": false
  }
}
```

> - `host` 设为 `127.0.0.1` 而非 `0.0.0.0`，让 Flask 只监听本机，由 Nginx 反向代理对外提供服务，更安全。
> - `debug` 设为 `false`，避免暴露调试信息。

---

## 五、使用 Systemd 管理服务

创建 systemd 服务文件，让 VisualHunt 随系统启动自动运行。

```bash
sudo vim /etc/systemd/system/visualhunt.service
```

写入以下内容（**注意替换 `User=ubuntu` 为你实际运行的用户**）：

```ini
[Unit]
Description=VisualHunt Flask Web Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/visualHunt
ExecStart=/home/ubuntu/.local/bin/uv run python main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONIOENCODING=utf-8

[Install]
WantedBy=multi-user.target
```

> **注意**：`ExecStart` 中的 `/home/ubuntu/.local/bin/uv` 是 uv 的安装路径。如果是其他用户，请替换为 `$(whoami)` 对应的路径，或用 `which uv` 查询。

启动并启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable visualhunt
sudo systemctl start visualhunt
sudo systemctl status visualhunt
```

常用命令：

```bash
sudo systemctl start visualhunt    # 启动
sudo systemctl stop visualhunt     # 停止
sudo systemctl restart visualhunt  # 重启
sudo systemctl status visualhunt   # 查看状态
sudo journalctl -u visualhunt -f   # 实时查看日志
```

---

## 六、Nginx 反向代理配置（多项目共存）

服务器上已有其他项目，建议使用 **Nginx 反向代理 + 路径区分** 或 **子域名** 方式部署。

### 方式 A：路径区分（推荐，无需额外域名）

假设服务器 IP 为 `1.2.3.4`，将 VisualHunt 挂在 `/visualhunt/` 路径下：

```bash
sudo vim /etc/nginx/sites-available/visualhunt
```

写入：

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

    # 静态资源（如有需要）
    location /visualhunt/static/ {
        alias /opt/visualHunt/src/api/static/;
    }
}
```

然后创建软链接启用：

```bash
sudo ln -sf /etc/nginx/sites-available/visualhunt /etc/nginx/sites-enabled/visualhunt
sudo nginx -t
sudo systemctl reload nginx
```

> ⚠️ Flask 的 `url_for` 可能需要适配 `APPLICATION_ROOT` 或反向代理头。如果前端链接异常，可在 `src/api/app.py` 中增加 `ProxyFix`：
> ```python
> from werkzeug.middleware.proxy_fix import ProxyFix
> app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)
> ```

### 方式 B：独立子域名

如果有域名，可为 VisualHunt 配置独立子域名（如 `visualhunt.yourdomain.com`）：

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

如果服务器启用了 UFW，开放 80 端口即可，Flask 的 5000 端口**不需要**对外暴露：

```bash
sudo ufw allow 80/tcp
sudo ufw reload
```

> 如果没有启用 UFW，通常是云厂商的安全组控制，请在控制台放行 80 端口。

---

## 八、验证部署

1. **服务状态检查**：
   ```bash
   sudo systemctl status visualhunt
   curl http://127.0.0.1:5000
   ```

2. **Nginx 代理检查**：
   ```bash
   # 方式 A（路径区分）
   curl http://1.2.3.4/visualhunt/

   # 方式 B（子域名）
   curl http://visualhunt.yourdomain.com
   ```

3. **浏览器访问**：
   - 路径区分：`http://<服务器IP>/visualhunt/`
   - 子域名：`http://visualhunt.yourdomain.com`

---

## 九、后续维护

### 9.1 更新代码

```bash
cd /opt/visualHunt
git pull origin dev
sudo systemctl restart visualhunt
```

### 9.2 重新训练模型（可选）

```bash
cd /opt/visualHunt

# 图像分类
uv run vh train --task classification

# 图像去噪
uv run vh train --task denoising

# 图像相似度
uv run vh train --task similarity

# 重启服务生效
sudo systemctl restart visualhunt
```

### 9.3 日志排查

```bash
# 查看服务日志
sudo journalctl -u visualhunt -n 100 --no-pager

# 查看 Nginx 访问日志
sudo tail -f /var/log/nginx/access.log

# 查看 Nginx 错误日志
sudo tail -f /var/log/nginx/error.log
```

---

## 十、目录结构（部署后）

```text
/opt/visualHunt/
├── .venv/                  # uv 虚拟环境
├── data/
│   ├── dataset/            # 样本图片
│   └── fashion-labels.csv
├── finetuned/              # 预训练模型（已包含）
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
├── main.py                 # 启动入口
├── runtime_config.json     # 运行配置
├── pyproject.toml
└── README.md
```

---

## 附录：一键部署脚本（可选）

将以下内容保存为 `deploy.sh`，在服务器上执行：

```bash
#!/bin/bash
set -e

PROJECT_DIR="/opt/visualHunt"
REPO_URL="https://gitee.com/myparadises/visual-hunt.git"
UV_BIN="$HOME/.local/bin/uv"

echo "=== 1. 安装基础依赖 ==="
sudo apt update -y
sudo apt install -y git wget curl vim nginx

echo "=== 2. 安装 uv ==="
if [ ! -f "$UV_BIN" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

echo "=== 3. 克隆项目 ==="
sudo mkdir -p /opt
if [ ! -d "$PROJECT_DIR/.git" ]; then
    sudo git clone -b dev "$REPO_URL" "$PROJECT_DIR"
else
    echo "项目已存在，跳过克隆"
fi
sudo chown -R "$(whoami):$(whoami)" "$PROJECT_DIR"

echo "=== 4. 安装依赖 ==="
cd "$PROJECT_DIR"
$UV_BIN sync

echo "=== 5. 配置 systemd ==="
sudo tee /etc/systemd/system/visualhunt.service > /dev/null <<EOF
[Unit]
Description=VisualHunt Flask Web Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_DIR
ExecStart=$UV_BIN run python main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONIOENCODING=utf-8

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable visualhunt
sudo systemctl start visualhunt

echo "=== 6. 启动 Nginx ==="
sudo systemctl enable nginx
sudo systemctl start nginx

echo "=== 部署完成 ==="
echo "请手动配置 Nginx 反向代理，并修改 runtime_config.json 中的 api.debug 为 false"
echo "查看服务状态: sudo systemctl status visualhunt"
```

运行：

```bash
chmod +x deploy.sh
./deploy.sh
```

---

> 如有问题，请检查 `journalctl -u visualhunt` 和 Nginx 错误日志进行排查。
