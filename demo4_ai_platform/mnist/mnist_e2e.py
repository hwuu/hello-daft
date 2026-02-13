"""
MNIST 端到端流式处理脚本（Level 2）。

在同一个 Daft on Ray 执行图里完成清洗和训练，中间数据在内存中流转不落盘。
与分开两个 Task（mnist_clean.py + mnist_cnn.py）相比，省去中间 Lance 写入。

资源分步：清洗阶段用 2 CPU，训练阶段用更多 CPU（可选 GPU）。
通过嵌套 Ray Task 实现——外层 run() 是轻量协调者，内层 step 各自声明资源。

数据流: Lance -> Daft lazy 清洗 (2 CPU) -> 训练 (N CPU / GPU) -> 模型写入 Lance
"""

import io
import json
import logging
from datetime import datetime, timezone

import daft
import numpy as np
import pyarrow as pa
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class MnistCNN(nn.Module):
    """与 mnist_cnn.py 中完全一致的 CNN 结构。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# --- 嵌套 Ray Task：每个 step 独立声明资源 ---


@ray.remote(num_cpus=2)
def clean_step(input_path: str) -> dict:
    """阶段 1: 数据清洗（轻量 CPU）。

    Daft lazy evaluation 构建计算图，.to_pandas() 触发执行。
    数据在 Ray 集群内存中流转，不写中间 Lance 文件。
    返回 dict 而非 DataFrame，因为 Ray 需要序列化跨 Task 传递。
    """
    logger.info(f"[clean_step] 读取数据: {input_path}")
    df = daft.read_lance(input_path)

    # 可以在这里加更多清洗/特征工程步骤，都是 lazy 的
    train_pdf = df.where(daft.col("split") == "train").to_pandas()
    test_pdf = df.where(daft.col("split") == "test").to_pandas()

    logger.info(f"[clean_step] 训练集: {len(train_pdf)} 条, 测试集: {len(test_pdf)} 条")

    # 序列化为 dict，跨 Ray Task 传递
    return {
        "train": train_pdf.to_dict("list"),
        "test": test_pdf.to_dict("list"),
    }


@ray.remote(num_cpus=4)
def train_step(clean_data: dict, output_path: str, params: dict) -> dict:
    """阶段 2: 模型训练（重量 CPU，可选 GPU）。

    接收清洗后的数据，训练 CNN，保存模型到 Lance。
    资源声明比 clean_step 更多。如果有 GPU，可以用
    train_step.options(num_gpus=1) 动态覆盖。
    """
    import pandas as pd

    epochs = params.get("epochs", 10)
    lr = params.get("learning_rate", 0.001)
    batch_size = params.get("batch_size", 64)
    device = torch.device(params.get("device", "cpu"))

    train_pdf = pd.DataFrame(clean_data["train"])
    test_pdf = pd.DataFrame(clean_data["test"])
    logger.info(f"[train_step] 训练集: {len(train_pdf)}, 测试集: {len(test_pdf)}, device: {device}")

    train_loader = _make_loader(train_pdf, batch_size, shuffle=True)
    test_loader = _make_loader(test_pdf, batch_size, shuffle=False)

    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"[train_step] Epoch {epoch + 1}/{epochs}, loss: {total_loss / len(train_loader):.4f}")

    # 评估
    model.eval()
    correct = total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"[train_step] accuracy={accuracy:.4f}, loss={avg_test_loss:.4f}")

    # 保存模型到 Lance
    _save_model(model, output_path, params, {"accuracy": accuracy, "test_loss": avg_test_loss})

    return {"accuracy": accuracy, "test_loss": avg_test_loss, "mode": "e2e_streaming"}


def run(input_path: str, output_path: str, params: dict) -> dict:
    """端到端流式处理：清洗 + 训练，通过嵌套 Ray Task 分步调度。

    run() 本身是轻量协调者，不做计算。实际工作由 clean_step 和
    train_step 两个 Ray Task 完成，各自声明不同的资源。

    资源分配:
        clean_step: 2 CPU（数据读取和清洗）
        train_step: 4 CPU（模型训练，可通过 .options(num_gpus=1) 加 GPU）

    Args:
        input_path: Lance 数据集路径
        output_path: 模型输出 Lance 路径
        params: 超参数（同 mnist_cnn.py）
    """
    logger.info(f"提交 e2e workflow: {input_path} -> {output_path}")

    # 阶段 1: 清洗（2 CPU）
    clean_ref = clean_step.remote(input_path)

    # 阶段 2: 训练（4 CPU）
    # clean_ref 作为依赖传入，Ray 自动等 clean_step 完成后再调度 train_step
    # 如果有 GPU: train_step.options(num_gpus=1).remote(clean_ref, ...)
    train_ref = train_step.remote(clean_ref, output_path, params)

    # 等待最终结果
    result = ray.get(train_ref)
    logger.info(f"e2e 完成: {result}")
    return result


def _make_loader(pdf, batch_size: int, shuffle: bool) -> DataLoader:
    images = np.array(pdf["image"].tolist(), dtype=np.float32).reshape(-1, 1, 28, 28)
    labels = np.array(pdf["label"].tolist(), dtype=np.int64)
    return DataLoader(TensorDataset(torch.from_numpy(images), torch.from_numpy(labels)),
                      batch_size=batch_size, shuffle=shuffle)


def _save_model(model: nn.Module, output_path: str, params: dict, metrics: dict):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    table = pa.table({
        "weights": [buf.getvalue()],
        "params": [json.dumps(params)],
        "metrics": [json.dumps(metrics)],
        "created_at": [datetime.now(timezone.utc).isoformat()],
    })
    daft.from_arrow(table).write_lance(output_path, mode="overwrite")
