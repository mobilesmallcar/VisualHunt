"""VisualHunt Flask Web API — 完整的深度学习训练与推理平台

提供配置管理、数据预处理检查、训练监控、评估、测试、预测等全流程 Web 接口。

启动方式（在项目根目录执行）：
    uv run python src/api/app.py

然后浏览器访问 http://localhost:5000
"""

from __future__ import annotations

import base64
import io
import json
import random
import sys
import threading
import time
from pathlib import Path

# 将项目根目录加入 Python 路径，确保能导入 src 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from torch.utils.data import DataLoader

from src.config import Config
from src.data import ImageDataset, create_datasets, _get_transform
from src.engine import (
    compute_similarity,
    create_embeddings,
    test_epoch,
    test_epoch_with_acc,
    train_epoch,
)
from src.models import ClassifierModel, ConvDecoder, ConvDenoiser, ConvEncoder
from src.utils import seed_everything

app = Flask(__name__, template_folder="templates", static_folder="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 全局运行时配置（可被前端修改，持久化到 JSON）
# ---------------------------------------------------------------------------
CFG_FILE = PROJECT_ROOT / "runtime_config.json"

DEFAULT_RUNTIME_CFG = {
    "classification": {
        "epochs": 20,
        "lr": 1e-3,
        "batch_size": 32,
        "img_h": 64,
        "img_w": 64,
        "max_samples": None,
    },
    "denoising": {
        "epochs": 30,
        "lr": 1e-3,
        "batch_size": 32,
        "img_h": 68,
        "img_w": 68,
        "max_samples": None,
        "noise_ratio": 0.5,
    },
    "similarity": {
        "epochs": 30,
        "lr": 1e-3,
        "batch_size": 32,
        "img_h": 64,
        "img_w": 64,
        "max_samples": None,
        "full_batch_size": 32,
        "num_similar": 5,
    },
    "global": {
        "train_ratio": 0.75,
        "seed": 42,
        "model_dir": "finetuned",
        "img_path": "data/dataset",
        "labels_path": "data/fashion-labels.csv",
    },
    "huggingface": {
        "model_repo": "",
        "dataset_repo": "",
    },
}


def _load_runtime_cfg() -> dict:
    if CFG_FILE.exists():
        try:
            with open(CFG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # 合并默认值（防止新增字段缺失）
                merged = DEFAULT_RUNTIME_CFG.copy()
                for k, v in merged.items():
                    if isinstance(v, dict) and k in loaded:
                        merged[k] = {**v, **loaded[k]}
                    elif k in loaded:
                        merged[k] = loaded[k]
                return merged
        except Exception:
            pass
    return DEFAULT_RUNTIME_CFG.copy()


def _save_runtime_cfg(cfg: dict) -> None:
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


RUNTIME_CFG = _load_runtime_cfg()


def _build_cfg(task: str) -> Config:
    """根据运行时配置构建 Config 对象。"""
    task_cfg = RUNTIME_CFG.get(task, {})
    glob = RUNTIME_CFG.get("global", {})

    base = Config(
        task=task,
        img_path=PROJECT_ROOT / glob.get("img_path", "data/dataset"),
        labels_path=PROJECT_ROOT / glob.get("labels_path", "data/fashion-labels.csv") if task == "classification" else None,
        img_h=task_cfg.get("img_h", 64),
        img_w=task_cfg.get("img_w", 64),
        seed=glob.get("seed", 42),
        train_ratio=glob.get("train_ratio", 0.75),
        batch_size=task_cfg.get("batch_size", 32),
        epochs=task_cfg.get("epochs", 20),
        lr=task_cfg.get("lr", 1e-3),
        model_dir=PROJECT_ROOT / glob.get("model_dir", "finetuned"),
        device="auto",
        max_samples=task_cfg.get("max_samples", None),
        noise_ratio=task_cfg.get("noise_ratio", 0.5) if task == "denoising" else 0.5,
        full_batch_size=task_cfg.get("full_batch_size", 32) if task == "similarity" else 32,
        num_similar=task_cfg.get("num_similar", 5) if task == "similarity" else 5,
    )
    return base


# ---------------------------------------------------------------------------
# 训练状态（内存中，非持久化）
# ---------------------------------------------------------------------------
TRAIN_STATE = {
    "running": False,
    "task": None,
    "epoch": 0,
    "total_epochs": 0,
    "train_losses": [],
    "val_losses": [],
    "val_accs": [],
    "message": "",
    "error": None,
    "start_time": None,
}
TRAIN_LOCK = threading.Lock()
STOP_FLAG = threading.Event()


def _tensor_to_base64(tensor: torch.Tensor) -> str:
    """将 (C,H,W) 张量转为 base64 PNG 字符串。"""
    img = tensor.detach().cpu().clamp(0, 1)
    pil = transforms.ToPILImage()(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_to_tensor(pil_img: Image.Image, cfg: Config) -> torch.Tensor:
    transform = _get_transform(cfg)
    return transform(pil_img)


def _validate_model(path: str, model_class, *args, **kwargs):
    """校验并加载指定路径的模型。"""
    try:
        p = Path(path)
        if not p.exists():
            return None, f"文件不存在: {path}"
        model = model_class(*args, **kwargs).to(device)
        state = torch.load(p, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, None
    except RuntimeError as e:
        return None, f"模型结构不匹配: {e}"
    except Exception as e:
        return None, f"加载失败: {e}"


# ---------------------------------------------------------------------------
# 训练后台线程
# ---------------------------------------------------------------------------

def _train_thread(task: str) -> None:
    """在后台线程中执行训练。"""
    global TRAIN_STATE
    cfg = _build_cfg(task)
    STOP_FLAG.clear()

    with TRAIN_LOCK:
        TRAIN_STATE.update({
            "running": True,
            "task": task,
            "epoch": 0,
            "total_epochs": cfg.epochs,
            "train_losses": [],
            "val_losses": [],
            "val_accs": [],
            "message": "初始化中...",
            "error": None,
            "start_time": time.time(),
        })

    try:
        seed_everything(cfg.seed)
        train_ds, test_ds, full_ds = create_datasets(cfg)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

        with TRAIN_LOCK:
            TRAIN_STATE["message"] = f"训练集: {len(train_ds)} 张 | 测试集: {len(test_ds)} 张"

        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        min_val_loss = float("inf")

        if task == "classification":
            model = ClassifierModel(cfg.num_classes).to(device)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
            for epoch in range(cfg.epochs):
                if STOP_FLAG.is_set():
                    break
                train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
                val_loss, val_acc = test_epoch_with_acc(model, test_loader, loss_fn, device)
                if val_loss < min_val_loss:
                    torch.save(model.state_dict(), cfg.model_path)
                    min_val_loss = val_loss
                with TRAIN_LOCK:
                    TRAIN_STATE["epoch"] = epoch + 1
                    TRAIN_STATE["train_losses"].append(round(train_loss, 6))
                    TRAIN_STATE["val_losses"].append(round(val_loss, 6))
                    TRAIN_STATE["val_accs"].append(round(val_acc, 6))
                    TRAIN_STATE["message"] = f"Epoch {epoch+1}/{cfg.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}"

        elif task == "denoising":
            model = ConvDenoiser().to(device)
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            for epoch in range(cfg.epochs):
                if STOP_FLAG.is_set():
                    break
                train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
                val_loss = test_epoch(model, test_loader, loss_fn, device)
                if val_loss < min_val_loss:
                    torch.save(model.state_dict(), cfg.model_path)
                    min_val_loss = val_loss
                with TRAIN_LOCK:
                    TRAIN_STATE["epoch"] = epoch + 1
                    TRAIN_STATE["train_losses"].append(round(train_loss, 6))
                    TRAIN_STATE["val_losses"].append(round(val_loss, 6))
                    TRAIN_STATE["message"] = f"Epoch {epoch+1}/{cfg.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}"

        elif task == "similarity":
            encoder = ConvEncoder().to(device)
            decoder = ConvDecoder().to(device)
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr)
            for epoch in range(cfg.epochs):
                if STOP_FLAG.is_set():
                    break
                train_loss = train_epoch([encoder, decoder], train_loader, loss_fn, optimizer, device)
                val_loss = test_epoch([encoder, decoder], test_loader, loss_fn, device)
                if val_loss < min_val_loss:
                    torch.save(encoder.state_dict(), cfg.model_path)
                    torch.save(decoder.state_dict(), cfg.decoder_path)
                    min_val_loss = val_loss
                with TRAIN_LOCK:
                    TRAIN_STATE["epoch"] = epoch + 1
                    TRAIN_STATE["train_losses"].append(round(train_loss, 6))
                    TRAIN_STATE["val_losses"].append(round(val_loss, 6))
                    TRAIN_STATE["message"] = f"Epoch {epoch+1}/{cfg.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}"

            # 生成 embeddings
            if not STOP_FLAG.is_set():
                full_loader = DataLoader(full_ds, batch_size=cfg.full_batch_size, shuffle=False)
                encoder.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
                embeddings = create_embeddings(encoder, full_loader, device)
                np.save(cfg.embedding_path, embeddings)
                with TRAIN_LOCK:
                    TRAIN_STATE["message"] += f" | Embeddings: {embeddings.shape}"

        with TRAIN_LOCK:
            TRAIN_STATE["message"] += " | 训练完成！"

    except Exception as e:
        with TRAIN_LOCK:
            TRAIN_STATE["error"] = str(e)
            TRAIN_STATE["message"] = f"训练出错: {e}"
    finally:
        with TRAIN_LOCK:
            TRAIN_STATE["running"] = False


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    return render_template("index.html")


# ---------------------------------------------------------------------------
# 配置接口
# ---------------------------------------------------------------------------

@app.route("/api/config", methods=["GET", "POST"])
def config_api():
    global RUNTIME_CFG
    if request.method == "GET":
        return jsonify(RUNTIME_CFG)
    data = request.get_json(silent=True) or {}
    for key, val in data.items():
        if key in RUNTIME_CFG and isinstance(RUNTIME_CFG[key], dict):
            RUNTIME_CFG[key].update(val)
        else:
            RUNTIME_CFG[key] = val
    _save_runtime_cfg(RUNTIME_CFG)
    return jsonify({"success": True, "config": RUNTIME_CFG})


# ---------------------------------------------------------------------------
# 数据预处理检查接口
# ---------------------------------------------------------------------------

@app.route("/api/raw_samples")
def raw_samples():
    """返回几张原生数据样本（未预处理）。"""
    img_dir = PROJECT_ROOT / RUNTIME_CFG["global"].get("img_path", "data/dataset")
    files = sorted([p for p in img_dir.iterdir() if p.is_file()])[:20]
    if not files:
        return jsonify({"error": "数据目录为空"}), 400
    samples = random.sample(files, min(4, len(files)))
    images = []
    for p in samples:
        with Image.open(p) as img:
            images.append({
                "name": p.name,
                "size": f"{img.width} x {img.height}",
                "base64": f"data:image/png;base64,{_pil_to_base64(img)}",
            })
    return jsonify({"images": images})


@app.route("/api/preprocess_sample")
def preprocess_sample():
    """随机返回一张图片的原生图和预处理后图。"""
    task = request.args.get("task", "classification")
    cfg = _build_cfg(task)
    files = sorted([p for p in cfg.img_path.iterdir() if p.is_file()])[:20]
    if not files:
        return jsonify({"error": "数据目录为空"}), 400

    sample_path = random.choice(files)
    raw_img = Image.open(sample_path).convert("RGB")
    tensor = _pil_to_tensor(raw_img, cfg)

    result = {
        "name": sample_path.name,
        "raw_size": f"{raw_img.width} x {raw_img.height}",
        "raw": f"data:image/png;base64,{_pil_to_base64(raw_img)}",
        "preprocessed": f"data:image/png;base64,{_tensor_to_base64(tensor)}",
        "preprocessed_shape": list(tensor.shape),
        "task": task,
    }

    if task == "denoising":
        noise = cfg.noise_ratio * torch.randn_like(tensor)
        noisy = torch.clamp(tensor + noise, 0.0, 1.0)
        result["noisy"] = f"data:image/png;base64,{_tensor_to_base64(noisy)}"

    if task == "classification":
        labels = {}
        if cfg.labels_path and cfg.labels_path.exists():
            import pandas as pd
            df = pd.read_csv(cfg.labels_path)
            labels = dict(zip(df["id"].astype(str), df["target"].astype(str)))
        idx = int(Path(sample_path).stem)
        label = labels.get(str(idx), "未知")
        names = cfg.classification_names
        result["label"] = names.get(int(label), "未知")
        result["label_id"] = int(label)

    return jsonify(result)


@app.route("/api/dataloader_batch")
def dataloader_batch():
    """返回一个 DataLoader 批次的数据，用于检查格式。"""
    task = request.args.get("task", "classification")
    cfg = _build_cfg(task)

    try:
        train_ds, _, _ = create_datasets(cfg)
    except Exception as e:
        return jsonify({"error": f"创建数据集失败: {e}"}), 400

    loader = DataLoader(train_ds, batch_size=min(cfg.batch_size, 8), shuffle=True)
    batch = next(iter(loader))

    data, target = batch
    images = []
    for i in range(min(len(data), 8)):
        images.append(_tensor_to_base64(data[i]))

    target_info = []
    if task == "classification":
        for i in range(min(len(target), 8)):
            target_info.append(cfg.classification_names.get(int(target[i]), "未知"))
    else:
        target_info = ["self" if task == "similarity" else "original"] * min(len(target), 8)

    has_padding = False  # 当前实现无 padding，直接 resize

    return jsonify({
        "batch_size": len(data),
        "tensor_shape": list(data.shape),
        "dtype": str(data.dtype),
        "value_range": f"[{data.min():.3f}, {data.max():.3f}]",
        "has_padding": has_padding,
        "images": [f"data:image/png;base64,{b}" for b in images],
        "targets": target_info,
        "task": task,
    })


# ---------------------------------------------------------------------------
# 训练接口
# ---------------------------------------------------------------------------

@app.route("/api/train", methods=["POST"])
def train_api():
    global TRAIN_STATE
    data = request.get_json(silent=True) or {}
    task = data.get("task", "classification")
    if task not in ("classification", "denoising", "similarity"):
        return jsonify({"error": "未知任务类型"}), 400

    with TRAIN_LOCK:
        if TRAIN_STATE["running"]:
            return jsonify({"error": "已有训练任务在运行中"}), 400

    thread = threading.Thread(target=_train_thread, args=(task,), daemon=True)
    thread.start()
    return jsonify({"success": True, "message": f"已启动 {task} 训练"})


@app.route("/api/train_status")
def train_status():
    with TRAIN_LOCK:
        return jsonify(TRAIN_STATE.copy())


@app.route("/api/stop_train", methods=["POST"])
def stop_train():
    STOP_FLAG.set()
    with TRAIN_LOCK:
        TRAIN_STATE["message"] = "正在停止..."
    return jsonify({"success": True, "message": "已发送停止信号"})


# ---------------------------------------------------------------------------
# 评估接口
# ---------------------------------------------------------------------------

@app.route("/api/evaluate", methods=["POST"])
def evaluate_api():
    data = request.get_json(silent=True) or {}
    task = data.get("task", "classification")
    cfg = _build_cfg(task)

    try:
        _, test_ds, _ = create_datasets(cfg)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)
    except Exception as e:
        return jsonify({"error": f"加载数据失败: {e}"}), 400

    if task == "classification":
        model = ClassifierModel(cfg.num_classes).to(device)
        if not cfg.model_path.exists():
            return jsonify({"error": "模型文件不存在，请先训练"}), 400
        model.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
        loss_fn = torch.nn.CrossEntropyLoss()
        test_loss, test_acc = test_epoch_with_acc(model, test_loader, loss_fn, device)
        return jsonify({"loss": round(test_loss, 6), "accuracy": round(test_acc, 6), "task": task})

    elif task == "denoising":
        model = ConvDenoiser().to(device)
        if not cfg.model_path.exists():
            return jsonify({"error": "模型文件不存在，请先训练"}), 400
        model.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
        loss_fn = torch.nn.MSELoss()
        test_loss = test_epoch(model, test_loader, loss_fn, device)
        return jsonify({"loss": round(test_loss, 6), "task": task})

    elif task == "similarity":
        encoder = ConvEncoder().to(device)
        decoder = ConvDecoder().to(device)
        if not cfg.model_path.exists() or not cfg.decoder_path.exists():
            return jsonify({"error": "模型文件不存在，请先训练"}), 400
        encoder.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
        decoder.load_state_dict(torch.load(cfg.decoder_path, map_location=device, weights_only=True))
        loss_fn = torch.nn.MSELoss()
        test_loss = test_epoch([encoder, decoder], test_loader, loss_fn, device)
        return jsonify({"loss": round(test_loss, 6), "task": task})

    return jsonify({"error": "未知任务"}), 400


# ---------------------------------------------------------------------------
# 预测接口（含原生效果对比）
# ---------------------------------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """统一分析接口：一次上传图片，同时执行分类、去噪、相似度检索。"""
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "请上传图片"}), 400

    pil_img = Image.open(file.stream).convert("RGB")

    # 可选自定义模型路径
    cls_path = request.form.get("classifier_path", str(_build_cfg("classification").model_path))
    den_path = request.form.get("denoiser_path", str(_build_cfg("denoising").model_path))
    enc_path = request.form.get("encoder_path", str(_build_cfg("similarity").model_path))

    # 分类
    cfg_cls = _build_cfg("classification")
    tensor_cls = _pil_to_tensor(pil_img, cfg_cls).unsqueeze(0).to(device)
    model_cls, err_cls = _validate_model(cls_path, ClassifierModel, cfg_cls.num_classes)
    if model_cls is None:
        return jsonify({"error": f"分类模型校验失败: {err_cls}"}), 400
    with torch.no_grad():
        output = model_cls(tensor_cls)
        prob = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = prob[0][pred].item()
    label = cfg_cls.classification_names[pred]

    # 去噪
    cfg_den = _build_cfg("denoising")
    tensor_den = _pil_to_tensor(pil_img, cfg_den).unsqueeze(0).to(device)
    noise = cfg_den.noise_ratio * torch.randn_like(tensor_den)
    noisy = torch.clamp(tensor_den + noise, 0.0, 1.0)
    model_den, err_den = _validate_model(den_path, ConvDenoiser)
    if model_den is None:
        return jsonify({"error": f"去噪模型校验失败: {err_den}"}), 400
    with torch.no_grad():
        denoised = model_den(noisy)

    # 相似度
    cfg_sim = _build_cfg("similarity")
    tensor_sim = _pil_to_tensor(pil_img, cfg_sim).unsqueeze(0).to(device)
    encoder, err_enc = _validate_model(enc_path, ConvEncoder)
    if encoder is None:
        return jsonify({"error": f"编码器校验失败: {err_enc}"}), 400

    if enc_path != str(cfg_sim.model_path):
        full_ds = ImageDataset(cfg_sim.img_path, _get_transform(cfg_sim))
        full_loader = DataLoader(full_ds, batch_size=cfg_sim.full_batch_size, shuffle=False)
        embeddings = create_embeddings(encoder, full_loader, device)
    else:
        embeddings = np.load(cfg_sim.embedding_path)

    indices = compute_similarity(encoder, tensor_sim, cfg_sim.num_similar, embeddings, device)
    full_ds = ImageDataset(cfg_sim.img_path, _get_transform(cfg_sim))
    similar_images = []
    for idx in indices[0]:
        sim_img, _ = full_ds[idx]
        similar_images.append(_tensor_to_base64(sim_img))

    # 原生效果（仅预处理后的图，无模型处理）
    raw_preprocessed = _tensor_to_base64(_pil_to_tensor(pil_img, cfg_cls))

    return jsonify({
        "raw": f"data:image/png;base64,{raw_preprocessed}",
        "classification": {"label": label, "confidence": round(confidence, 4), "model": cls_path},
        "denoising": {
            "noisy": f"data:image/png;base64,{_tensor_to_base64(noisy.squeeze(0))}",
            "denoised": f"data:image/png;base64,{_tensor_to_base64(denoised.squeeze(0))}",
            "model": den_path,
        },
        "similarity": {
            "images": [f"data:image/png;base64,{img}" for img in similar_images],
            "model": enc_path,
        },
    })


@app.route("/api/predict", methods=["POST"])
def predict_api():
    """单任务预测，用于测试 Tab。"""
    file = request.files.get("image")
    task = request.form.get("task", "classification")
    if not file:
        return jsonify({"error": "请上传图片"}), 400

    pil_img = Image.open(file.stream).convert("RGB")
    cfg = _build_cfg(task)
    tensor = _pil_to_tensor(pil_img, cfg).unsqueeze(0).to(device)
    raw_b64 = _tensor_to_base64(_pil_to_tensor(pil_img, cfg))

    if task == "classification":
        model = ClassifierModel(cfg.num_classes).to(device)
        if cfg.model_path.exists():
            model.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                confidence = prob[0][pred].item()
            return jsonify({
                "raw": f"data:image/png;base64,{raw_b64}",
                "result": f"预测类别: {cfg.classification_names[pred]} (置信度: {(confidence*100):.2f}%)",
                "model_used": str(cfg.model_path),
            })
        return jsonify({"raw": f"data:image/png;base64,{raw_b64}", "result": "模型不存在，请先训练"})

    elif task == "denoising":
        noise = cfg.noise_ratio * torch.randn_like(tensor)
        noisy = torch.clamp(tensor + noise, 0.0, 1.0)
        if cfg.model_path.exists():
            model = ConvDenoiser().to(device)
            model.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                denoised = model(noisy)
            return jsonify({
                "raw": f"data:image/png;base64,{raw_b64}",
                "noisy": f"data:image/png;base64,{_tensor_to_base64(noisy.squeeze(0))}",
                "result": f"data:image/png;base64,{_tensor_to_base64(denoised.squeeze(0))}",
                "model_used": str(cfg.model_path),
            })
        return jsonify({
            "raw": f"data:image/png;base64,{raw_b64}",
            "noisy": f"data:image/png;base64,{_tensor_to_base64(noisy.squeeze(0))}",
            "result": "模型不存在，请先训练",
        })

    elif task == "similarity":
        if cfg.model_path.exists():
            encoder = ConvEncoder().to(device)
            encoder.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
            encoder.eval()
            embeddings = np.load(cfg.embedding_path)
            indices = compute_similarity(encoder, tensor, cfg.num_similar, embeddings, device)
            full_ds = ImageDataset(cfg.img_path, _get_transform(cfg))
            similar_images = []
            for idx in indices[0]:
                sim_img, _ = full_ds[idx]
                similar_images.append(_tensor_to_base64(sim_img))
            return jsonify({
                "raw": f"data:image/png;base64,{raw_b64}",
                "result": [f"data:image/png;base64,{img}" for img in similar_images],
                "model_used": str(cfg.model_path),
            })
        return jsonify({"raw": f"data:image/png;base64,{raw_b64}", "result": "模型不存在，请先训练"})

    return jsonify({"error": "未知任务"}), 400


@app.route("/api/check_models")
def check_models():
    """检查三个任务的模型文件是否都已存在。"""
    result = {}
    for task in ("classification", "denoising", "similarity"):
        cfg = _build_cfg(task)
        exists = cfg.model_path.exists()
        result[task] = {
            "exists": exists,
            "path": str(cfg.model_path),
        }
    result["all_ready"] = all(v["exists"] for v in result.values())
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
