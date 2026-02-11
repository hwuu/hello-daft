"""
MNIST 推理服务脚本。

从 Lance 数据湖加载模型权重，启动 FastAPI 子服务提供推理 API。
实现 run(input_path, output_path, params) 接口，供 Executor 调用。

数据流: Lance 模型文件 -> PyTorch 加载权重 -> FastAPI 提供 /predict 端点
"""

import io
import logging

import daft
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MnistCNN(nn.Module):
    """MNIST CNN 模型结构。

    必须与 scripts/training/mnist_cnn.py 中的 MnistCNN 保持一致。
    结构: Conv(1->16) -> Pool -> Conv(16->32) -> Pool -> FC(1568->128) -> FC(128->10)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def run(input_path: str, output_path: str, params: dict) -> dict:
    """启动推理服务。

    从 Lance 文件加载模型权重，启动 FastAPI 子服务。
    此函数会阻塞（uvicorn.run），直到进程被 kill。

    Args:
        input_path: Lance 模型文件路径（包含 weights 列）
        output_path: 未使用（推理服务不产生输出文件）
        params: 服务参数
            - device: 计算设备（默认 "cpu"）
            - port: 服务端口（默认 8080）
    Returns:
        不会正常返回（阻塞直到进程终止）
    """
    device = params.get("device", "cpu")
    port = params.get("port", 8080)

    # 从 Lance 数据湖读取模型权重
    logger.info(f"加载模型: {input_path}")
    df = daft.read_lance(input_path)
    pdf = df.to_pandas()
    weights_bytes = pdf["weights"].iloc[0]
    logger.info(f"已读取模型权重: {len(weights_bytes)} 字节")

    # 反序列化权重并加载到 PyTorch 模型
    model = MnistCNN()
    buf = io.BytesIO(weights_bytes)
    model.load_state_dict(torch.load(buf, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"模型已加载，设备: {device}")

    # 构建 FastAPI 子服务
    app = FastAPI(title="MNIST Inference")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class PredictRequest(BaseModel):
        """推理请求体。"""
        image: list[float]  # 784 维浮点数组（28x28 归一化像素值）

    @app.post("/predict")
    def predict(body: PredictRequest) -> dict:
        """执行推理，返回预测结果。"""
        tensor = torch.tensor(body.image, dtype=torch.float32).reshape(1, 1, 28, 28)
        with torch.no_grad():
            output = model(tensor)
        probs = torch.softmax(output, dim=1)
        prediction = probs.argmax().item()
        confidence = probs.max().item()
        logger.info(f"推理完成: 预测={prediction}, 置信度={confidence:.4f}")
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs[0].tolist(),
        }

    @app.get("/health")
    def health() -> dict:
        """健康检查端点。"""
        return {"status": "ok"}

    # 阻塞运行，直到进程被 kill
    logger.info(f"推理服务启动: http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

    return {"status": "stopped"}
