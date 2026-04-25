# 📁 数据说明（Data Guide）

> ⚠️ 本仓库**携带了 50 张样本图片**（`data/dataset/0.jpg` ~ `49.jpg`）以及完整的 `fashion-labels.csv` 标签映射表，方便你直接体验项目功能。
>
> 若要进行完整训练，建议按下方说明准备约 25,000 张图片规模的完整数据集。其余数据文件仍已加入 `.gitignore`，不会被追踪。

---

## 目录结构要求

```text
VisualHunt/
├── data/
│   ├── dataset/               # 存放原始图片（默认含 50 张样本，完整训练建议约 25,000 张）
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── fashion-labels.csv     # 图片 id 与类别标签的映射表
```

### `fashion-labels.csv` 格式

| 列名   | 类型 | 说明                               |
|--------|------|------------------------------------|
| `id`   | int  | 图片文件名对应的数字（如 `0` 对应 `0.jpg`） |
| `target` | int  | 类别标签，共 5 类：`0=上衣, 1=鞋, 2=包, 3=下身衣服, 4=手表` |

> 如果你使用其他数据集，请自行修改 `img_classification/classification_config.py` 中的 `classification_names` 映射。

---

## 推荐数据来源

本项目使用的时尚商品图片可来自以下公开数据集（**任选其一或混合使用均可**）：

| 数据集 | 来源 | 说明 |
|--------|------|------|
| **Fashion Product Images** | [Kaggle](https://www.kaggle.com/datasets/utkarshsaxenadn/fashion-product-images-small) | 约 44k 张时尚商品图，含多类别标签，可筛选出上衣、鞋、包、下身衣服、手表等类别 |
| **DeepFashion (Category and Attribute Prediction)** | [GitHub / 官网](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) | 大规模时尚数据集，类别丰富，需按类别筛选并整理为所需 5 类 |
| **自定义爬取** | — | 可使用爬虫从电商平台采集同类商品图，并自行标注类别 |

### 数据预处理建议

1. **统一命名**：将所有图片按顺序重命名为 `0.jpg`, `1.jpg`, `2.jpg` ...，以便与 `fashion-labels.csv` 中的 `id` 一一对应。
2. **格式统一**：建议统一转换为 `.jpg` 格式，单通道图请转换为 RGB（代码中会执行 `.convert('RGB')`）。
3. **数量参考**：当前代码在约 25,000 张图片规模下验证通过。若数据量差异较大，可能需要调整 `config` 中的批量大小（`BATCH_SIZE`）与训练轮次（`EPOCHS`）。

---

## 各模块数据依赖

| 模块 | 依赖路径 | 说明 |
|------|----------|------|
| `img_classification` | `../data/dataset/` + `../data/fashion-labels.csv` | 需要图片与 CSV 标签文件 |
| `img_denoising` | `../data/dataset/` | 仅需图片，无需标签 |
| `img_similarity` | `../data/dataset/` | 仅需图片，无需标签 |

> 所有模块默认以脚本所在目录为基准，通过 `../data/` 相对路径访问数据。若你调整了目录结构，请同步修改各 `*_config.py` 中的 `IMG_PATH` 与 `FASHION_LABELS_PATH`。

---

## 版权与许可

请确保你使用的数据符合其原始许可协议。若使用 Kaggle 数据集，请遵守相应的 Dataset License；若使用 DeepFashion，请引用其官方论文。
