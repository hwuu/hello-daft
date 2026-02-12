"""
MNIST 端到端流式处理脚本（Level 2）。

在同一个 Daft on Ray 执行图里完成清洗和训练，中间数据在内存中流转不落盘。
与分开两个 Task（mnist_clean.py + mnist_cnn.py）相比，省去中间 Lance 写入。

数据流: 原始 IDX 文件 -> Daft lazy 清洗 -> .to_pandas() 触发执行 -> PyTorch 训练 -> 模型写入 Lance
"""

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import daft
import numpy as np
import pyarrow as pa
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


def run(input_path: str, output_path: str, params: dict) -> dict:
    """端到端流式处理：清洗 + 训练在同一个执行图中完成。

    Args:
        input_path: Lance 数据集路径（已有的 mnist_clean.lance，或原始数据路径）
        output_path: 模型输出 Lance 路径
        params: 超参数（同 mnist_cnn.py）
    Returns:
        训练指标: {"accuracy": float, "test_loss": float, "mode": "e2e_streaming"}
    """
    epochs = params.get("epochs", 10)
    lr = params.get("learning_rate", 0.001)
    batch_size = params.get("batch_size", 64)
    device = torch.device(params.get("device", "cpu"))

    # --- 阶段 1: Daft lazy 数据处理 ---
    # 构建计算图，此时不执行。在 Level 2 下 Daft 使用 Ray 后端，
    # .to_pandas() 触发时数据在 Ray 集群内存中流转。
    logger.info(f"构建 Daft 计算图: {input_path}")
    df = daft.read_lance(input_path)

    # 可以在这里加更多清洗/特征工程步骤，都是 lazy 的
    # 例如: df = df.where(col("label").between(0, 9))

    # --- 阶段 2: 触发执行 + 训练 ---
    # .to_pandas() 触发 Daft 执行图，数据从 Lance 读取后在内存中流转
    # 不写中间 Lance 文件
    logger.info("触发 Daft 执行，数据在内存中流转...")
    train_pdf = df.where(daft.col("split") == "train").to_pandas()
    test_pdf = df.where(daft.col("split") == "test").to_pandas()
    logger.info(f"训练集: {len(train_pdf)} 条, 测试集: {len(test_pdf)} 条")

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
        logger.info(f"Epoch {epoch + 1}/{epochs}, loss: {total_loss / len(train_loader):.4f}")

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
    logger.info(f"e2e 完成: accuracy={accuracy:.4f}, loss={avg_test_loss:.4f}")

    # --- 阶段 3: 只有最终结果写 Lance ---
    _save_model(model, output_path, params, {"accuracy": accuracy, "test_loss": avg_test_loss})

    return {"accuracy": accuracy, "test_loss": avg_test_loss, "mode": "e2e_streaming"}


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
