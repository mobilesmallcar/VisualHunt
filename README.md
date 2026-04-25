# 🌟 VisualHunt — 视觉猎人

> 基于深度学习的智能图像检索系统
> 集 **图像相似度检索**、**去噪修复**、**商品分类** 于一体的综合性计算机视觉项目。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ 功能特性

- 🔍 **以图搜图**：基于自编码器提取图像嵌入，结合 KNN 实现毫秒级相似图像检索。
- 🎨 **图像去噪**：卷积自编码器学习从噪声图像恢复清晰图像。
- 🏷️ **商品分类**：轻量 CNN 对时尚商品进行 5 类分类（上衣 / 鞋 / 包 / 下身衣服 / 手表）。
- 🌐 **Web 可视化平台**：基于 Flask 提供配置管理、训练监控、模型评估、图片上传预测等全流程 Web 界面。
- ⚡ **纯 PyTorch 实现**：代码简洁，模块化设计，易于二次开发与学习。

---

## 🚀 快速开始

### 1. 安装依赖

本项目使用 [**uv**](https://github.com/astral-sh/uv) 进行环境与包管理。

```bash
# 克隆仓库
git clone https://github.com/A-project-nlp/VisualHunt.git
cd VisualHunt

# 创建虚拟环境并安装依赖
uv sync

# 验证 CLI 可用
uv run vh --help
```

> 若需使用 Jupyter Notebook，可安装可选依赖：
> ```bash
> uv sync --extra notebook
> ```

### 2. 准备数据

本仓库**携带了 50 张样本图片**（`data/dataset/0.jpg` ~ `49.jpg`）以及完整的 `fashion-labels.csv` 标签映射表，可直接运行测试与体验。若需完整训练，请自行准备更大规模数据集：

```text
data/
├── dataset/               # 原始图片（默认含 50 张样本）
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── fashion-labels.csv     # 图片 id 与类别标签映射
```

详细数据说明与推荐来源请参考 [**DATA.md**](DATA.md)。

### 3. CLI 命令行使用

#### 训练

```bash
# 图像分类
uv run vh train --task classification

# 图像去噪
uv run vh train --task denoising

# 图像相似度（同时生成嵌入矩阵）
uv run vh train --task similarity
```

#### 测试（带可视化窗口）

```bash
# 图像分类测试（弹出 Matplotlib 窗口展示预测结果）
uv run vh test --task classification

# 图像去噪测试（展示去噪前后对比）
uv run vh test --task denoising

# 图像相似度测试（展示检索结果）
uv run vh test --task similarity
```

> ⚠️ `test` 命令会调用 `plt.show()` 弹出图形窗口，在无 GUI 环境（如纯 SSH）中可能无法正常显示。

#### 命令行参数覆盖

```bash
# 自定义训练轮次、学习率、批大小与模型保存目录
uv run vh train --task classification --epochs 30 --lr 5e-4 --batch-size 64 --model-dir ./checkpoints
```

也可以直接修改 `runtime_config.json` 中的默认值，无需每次传参。

### 4. 启动 Web 服务

项目内置了 Flask Web 平台，支持可视化训练、评估与预测：

```bash
uv run python main.py
```

然后浏览器访问 http://localhost:5000

**Web 端功能一览：**

| 功能 | 说明 |
|------|------|
| 配置管理 | 查看与修改 `runtime_config.json` 中的训练参数 |
| 数据预览 | 查看原生样本、预处理效果、DataLoader 批次格式 |
| 训练监控 | 启动后台训练，实时查看 loss / accuracy 曲线 |
| 模型评估 | 在测试集上评估已训练模型 |
| 单任务预测 | 上传图片，执行分类 / 去噪 / 相似度检索 |
| 综合分析 | 一次上传，同时执行三种任务并对比结果 |
| 模型检查 | 查看各任务模型文件是否存在 |

服务启动参数（host / port / debug）可在 `runtime_config.json` 的 `api` 段中修改。

---

## ⚙️ 配置说明

所有运行时参数集中在项目根目录的 `runtime_config.json` 中：

```json
{
  "classification": { "epochs": 20, "lr": 0.001, "batch_size": 32, ... },
  "denoising": { "epochs": 30, "lr": 0.001, "noise_ratio": 0.5, ... },
  "similarity": { "epochs": 20, "lr": 0.001, "num_similar": 5, ... },
  "global": { "train_ratio": 0.75, "seed": 42, "model_dir": "finetuned" },
  "api": { "host": "0.0.0.0", "port": 5000, "debug": true }
}
```

CLI 与 Web 服务均会读取该文件。命令行传参的优先级高于 JSON 配置。

---

## 📂 项目结构

```text
VisualHunt/
├── src/                       # 核心包（已整合三大模块）
│   ├── __init__.py
│   ├── cli.py                 # 统一命令行入口
│   ├── config.py              # 任务配置与预设（读取 runtime_config.json）
│   ├── data.py                # 统一数据集构建
│   ├── engine.py              # 训练/测试/检索引擎
│   ├── models.py              # 分类器、去噪器、编码器/解码器
│   ├── utils.py               # 工具函数
│   └── api/                   # Flask Web 服务
│       ├── app.py             # API 路由与业务逻辑
│       └── templates/         # Web 前端模板
├── test/                      # Notebook 实验与测试
├── data/                      # 数据目录（含 50 张样本）
├── pyproject.toml             # UV 项目配置
├── runtime_config.json        # 运行时参数配置
├── main.py                    # Web 服务启动入口
├── README.md
└── DATA.md                    # 数据说明与来源
```

---

## 🛠️ 开发规范

本项目使用 [Ruff](https://docs.astral.sh/ruff/) 进行代码格式与质量检查（需先安装开发依赖）：

```bash
# 安装开发依赖
uv sync --group dev

# 格式化代码
uv run ruff format .

# 静态检查
uv run ruff check . --fix
```

---

## 📄 许可证

[MIT License](LICENSE)
