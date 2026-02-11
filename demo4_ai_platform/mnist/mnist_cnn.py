"""
MNIST CNN 训练脚本。

从 Lance 数据湖读取 MNIST 数据，训练简单的 CNN 模型，将权重保存回 Lance。
实现 run(input_path, output_path, params) 接口，供 Server 调用。

数据流: Lance 数据集 -> PyTorch DataLoader -> 训练 -> 模型权重写入 Lance
模型结构: Conv(1→16) -> Pool -> Conv(16→32) -> Pool -> FC(1568→128) -> FC(128→10)
"""

import io
import json
import logging
from datetime import datetime, timezone

import daft
import numpy as np
import pyarrow as pa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class MnistCNN(nn.Module):
    """简单的 MNIST CNN 分类器。

    两层卷积 + 两层全连接，输入 28x28 灰度图，输出 10 类概率。
    注意: mnist/mnist_serve.py 中的 MnistCNN 必须与此结构完全一致。
    """

    def __init__(self):
        super().__init__()
        # 卷积层: 提取图像特征
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 1 通道 -> 16 通道, 28x28 不变
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 28x28 -> 14x14
            nn.Conv2d(16, 32, 3, padding=1),   # 16 通道 -> 32 通道, 14x14 不变
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 14x14 -> 7x7
        )
        # 全连接层: 分类
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),        # 展平后 1568 维 -> 128 维
            nn.ReLU(),
            nn.Linear(128, 10),                # 128 维 -> 10 类（数字 0-9）
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平: (batch, 32, 7, 7) -> (batch, 1568)
        return self.fc(x)


def run(input_path: str, output_path: str, params: dict) -> dict:
    """训练 MNIST CNN 模型。

    Args:
        input_path: Lance 数据集路径（需包含 image, label, split 列）
        output_path: 模型输出 Lance 路径
        params: 超参数
            - epochs: 训练轮数（默认 10）
            - learning_rate: 学习率（默认 0.001）
            - batch_size: 批大小（默认 64）
            - device: 计算设备（默认 "cpu"）
    Returns:
        训练指标: {"accuracy": float, "test_loss": float}
    """
    # 解析超参数
    epochs = params.get("epochs", 10)
    lr = params.get("learning_rate", 0.001)
    batch_size = params.get("batch_size", 64)
    device = torch.device(params.get("device", "cpu"))

    logger.info(f"训练参数: epochs={epochs}, lr={lr}, batch_size={batch_size}, device={device}")

    # 从 Lance 数据湖读取数据
    logger.info(f"读取训练数据: {input_path}")
    df = daft.read_lance(input_path)
    pdf = df.to_pandas()

    # 按 split 列拆分训练集和测试集
    train_pdf = pdf[pdf["split"] == "train"]
    test_pdf = pdf[pdf["split"] == "test"]
    logger.info(f"训练集: {len(train_pdf)} 条, 测试集: {len(test_pdf)} 条")

    # 构建 PyTorch DataLoader
    train_loader = _make_loader(train_pdf, batch_size, shuffle=True)
    test_loader = _make_loader(test_pdf, batch_size, shuffle=False)

    # 初始化模型、优化器、损失函数
    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(epochs):
        model.train()  # 切换到训练模式
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()           # 清零梯度
            output = criterion(model(images), labels)  # 前向传播 + 计算损失
            output.backward()               # 反向传播
            optimizer.step()                 # 更新参数
            total_loss += output.item()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}")

    # 评估模型
    logger.info("开始评估模型...")
    model.eval()  # 切换到评估模式（关闭 dropout 等）
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():  # 评估时不需要计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"评估完成: 准确率={accuracy:.4f}, 测试损失={avg_test_loss:.4f}")

    # 将模型权重 + 元数据保存到 Lance 数据湖
    _save_model(model, output_path, params, {"accuracy": accuracy, "test_loss": avg_test_loss})
    logger.info(f"模型已保存: {output_path}")

    return {"accuracy": accuracy, "test_loss": avg_test_loss}


def _make_loader(pdf, batch_size: int, shuffle: bool) -> DataLoader:
    """将 Pandas DataFrame 转为 PyTorch DataLoader。

    Args:
        pdf: 包含 image 和 label 列的 DataFrame
        batch_size: 批大小
        shuffle: 是否打乱顺序（训练集 True，测试集 False）
    """
    # image 列是 784 维浮点数组，reshape 为 (N, 1, 28, 28)
    images = np.array(pdf["image"].tolist(), dtype=np.float32)
    images = images.reshape(-1, 1, 28, 28)  # (N, channel, height, width)
    labels = np.array(pdf["label"].tolist(), dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _save_model(model: nn.Module, output_path: str, params: dict, metrics: dict):
    """将模型权重和元数据保存到 Lance。

    Lance 表 schema:
    - weights: Binary — PyTorch state_dict 的序列化字节
    - params: String — 超参数 JSON
    - metrics: String — 训练指标 JSON
    - created_at: String — 创建时间 ISO 格式
    """
    # 序列化模型权重为字节
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    weights_bytes = buf.getvalue()
    logger.info(f"模型权重大小: {len(weights_bytes)} 字节")

    # 构建 Arrow Table 并写入 Lance
    table = pa.table({
        "weights": [weights_bytes],
        "params": [json.dumps(params)],
        "metrics": [json.dumps(metrics)],
        "created_at": [datetime.now(timezone.utc).isoformat()],
    })

    df = daft.from_arrow(table)
    df.write_lance(output_path, mode="overwrite")
