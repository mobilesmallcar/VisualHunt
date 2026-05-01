# VisualHunt 服务器部署流程

> 部署目标：Ubuntu 服务器 + Docker  
> 代码仓库：https://gitee.com/nlp-learning/visual-hunt.git  
> 分支：`cpu-部署`  
> 部署路径：`/opt/projects/visual-hunt`

---

## 一、环境准备（服务器上执行）

```bash
# 1. 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 2. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

> 装完后退出 SSH 重新登录，让 docker 用户组生效。

---

## 二、拉取代码

```bash
cd /opt
sudo git clone -b cpu-部署 https://gitee.com/nlp-learning/visual-hunt.git projects/visual-hunt
sudo chown -R $(whoami):$(whoami) projects/visual-hunt
cd projects/visual-hunt
```

---

## 三、宿主机构建 .venv

```bash
uv sync
```

> 这一步在**宿主机**执行，装的是 Linux 版 PyTorch，构建完成后 `.venv`（约 1.1G）会被 Docker 镜像复制进去，避免容器内重复下载。

---

## 四、启动容器

```bash
sudo docker compose up -d
```

服务默认监听容器内 `5176` 端口，通过 Nginx 反向代理对外暴露。

---

## 五、后续更新代码

### 5.1 只改前端/后端代码（不涉及依赖）

```bash
cd /opt/projects/visual-hunt
git pull origin cpu-部署
sudo docker compose restart
```

> `docker-compose.yml` 已挂载 `./src` 和 `./main.py`，重启容器即可加载最新代码，**不需要重新 build 镜像**。

### 5.2 改了依赖（pyproject.toml / uv.lock）

```bash
cd /opt/projects/visual-hunt
git pull origin cpu-部署
uv sync                    # 重新安装依赖
sudo docker compose down
sudo docker compose up -d   # 重新创建容器，复制新的 .venv
```

> 依赖变更后必须重新 build 镜像，因为 `.venv` 是构建时 COPY 进镜像的，不是挂载的。

### 5.3 改了配置（runtime_config.json）

```bash
# 直接修改文件即可，已挂载为 volume
vim runtime_config.json
sudo docker compose restart
```

---

## 六、分支说明

| 分支 | 用途 |
|------|------|
| `master` | 原始代码（CUDA 版 PyTorch） |
| `cpu-部署` | **当前部署分支**（CPU 版 PyTorch + Docker 配置 + 前端相对路径） |

所有部署相关的修改都在 `cpu-部署` 分支，后续更新代码时切记拉取这个分支。

---

## 七、常用命令速查

```bash
# 查看容器状态
sudo docker ps | grep visualhunt

# 查看实时日志
sudo docker logs -f visualhunt

# 停止服务
sudo docker compose down

# 启动/重启
sudo docker compose up -d
sudo docker compose restart

# 进入容器排查
sudo docker exec -it visualhunt bash
```

---

## 八、已知注意事项（随口提）

- 容器内 Flask 监听的是 `0.0.0.0:5176`，不是 `127.0.0.1`
- 前端 `fetch` 已改为相对路径 `./api/xxx`，适配 Nginx `/visualhunt` 代理
- `.venv` 是宿主机预构建后 COPY 进镜像的，所以**必须在 Linux 服务器上**执行 `uv sync`
- 修改 `docker-compose.yml` 后要用 `down && up -d`，`restart` 不会重新读取配置
- 前端更新后浏览器可能需要 `Ctrl + F5` 强制刷新
