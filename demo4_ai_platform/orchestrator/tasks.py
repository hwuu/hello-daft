"""
统一任务管理模块。

管理三种任务类型：
- ingestion: 数据入库（批处理，运行完自动 completed）
- training:  模型训练（批处理，运行完自动 completed）
- inference: 推理服务（常驻，一直 running 直到 cancel）

ingestion 和 training 通过 Executor API 提交脚本任务；
inference 在 Orchestrator 进程内加载模型并提供 predict 方法。
"""

import io
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from threading import Lock

import httpx
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """任务类型枚举。"""
    INGESTION = "ingestion"   # 数据入库
    TRAINING = "training"     # 模型训练
    INFERENCE = "inference"   # 推理服务


class TaskStatus(str, Enum):
    """任务状态枚举。状态流转见 design.md 4.3 节。"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class _MnistCNN(nn.Module):
    """MNIST CNN 模型结构。

    必须与 scripts/training/mnist_cnn.py 中的 MnistCNN 保持一致，
    否则无法正确加载权重。

    结构: Conv(1→16) -> Pool -> Conv(16→32) -> Pool -> FC(1568→128) -> FC(128→10)
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 28x28 -> 14x14
            nn.Conv2d(16, 32, 3, padding=1),   # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 14x14 -> 7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),        # 1568 -> 128
            nn.ReLU(),
            nn.Linear(128, 10),                # 128 -> 10（0-9 十个数字）
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平: (batch, 32, 7, 7) -> (batch, 1568)
        return self.fc(x)


class TaskManager:
    """统一任务管理器。

    对外提供 create / get / list / cancel / predict 方法。
    内部通过 Executor HTTP API 提交批处理任务，
    通过 PyTorch 加载模型处理推理任务。
    """

    def __init__(self, executor_url: str):
        self.executor_url = executor_url.rstrip("/")
        self._tasks: dict[str, dict] = {}       # task_id -> 任务状态字典
        self._models: dict[str, nn.Module] = {}  # task_id -> 已加载的 PyTorch 模型
        self._lock = Lock()
        logger.info(f"任务管理器初始化完成，Executor 地址: {self.executor_url}")

    def create(self, task_type: TaskType, payload: dict) -> dict:
        """创建并启动任务。

        Args:
            task_type: 任务类型（ingestion/training/inference）
            payload: 请求体（包含 name, input, script 等字段）
        Returns:
            任务状态字典（过滤掉内部字段和 None 值）
        """
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        task = {
            "id": task_id,
            "type": task_type,
            "name": payload.get("name", ""),
            "status": TaskStatus.RUNNING,
            "created_at": now,
            "completed_at": None,
            "result": None,
            "error": None,
            # 保留请求体中的其他字段（如 input, script, params, model 等）
            **{k: v for k, v in payload.items() if k not in ("type", "name")},
        }

        with self._lock:
            self._tasks[task_id] = task

        logger.info(f"创建任务: {task_id}, 类型: {task_type}, 名称: {payload.get('name', '')}")

        # 根据任务类型分发处理
        if task_type in (TaskType.INGESTION, TaskType.TRAINING):
            self._submit_batch(task_id, payload)
        elif task_type == TaskType.INFERENCE:
            self._start_inference(task_id, payload)

        with self._lock:
            return self._public_view(self._tasks[task_id])

    def get(self, task_id: str) -> dict | None:
        """查询任务状态。

        对于运行中的批处理任务，会主动轮询 Executor 获取最新状态。
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return None

        # 批处理任务运行中时，主动同步 Executor 的状态
        if task["type"] in (TaskType.INGESTION, TaskType.TRAINING) and task["status"] == TaskStatus.RUNNING:
            executor_task_id = task.get("_executor_task_id")
            if executor_task_id:
                self._sync_batch_status(task_id, executor_task_id)

        with self._lock:
            return self._public_view(self._tasks[task_id])

    def list_all(self, task_type: TaskType | None = None) -> list[dict]:
        """列出所有任务，可按类型过滤。"""
        with self._lock:
            tasks = list(self._tasks.values())
        if task_type:
            tasks = [t for t in tasks if t["type"] == task_type]
        return [self._public_view(t) for t in tasks]

    def cancel(self, task_id: str) -> bool:
        """取消/停止任务。

        批处理任务: 向 Executor 发送取消请求
        推理任务: 卸载模型，释放内存
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task["status"] != TaskStatus.RUNNING:
                return False

        logger.info(f"取消任务: {task_id}")

        # 如果是批处理任务，通知 Executor 取消
        executor_task_id = task.get("_executor_task_id")
        if executor_task_id:
            try:
                httpx.post(f"{self.executor_url}/api/v1/tasks/{executor_task_id}/cancel")
                logger.info(f"已通知 Executor 取消任务: {executor_task_id}")
            except httpx.HTTPError as e:
                logger.warning(f"通知 Executor 取消失败: {e}")

        # 如果是推理任务，卸载模型
        if task_id in self._models:
            del self._models[task_id]
            logger.info(f"已卸载模型: {task_id}")

        with self._lock:
            task["status"] = TaskStatus.FAILED
            task["error"] = "cancelled"
            task["completed_at"] = datetime.now(timezone.utc).isoformat()
        return True

    def _submit_batch(self, task_id: str, payload: dict):
        """向 Executor 提交批处理任务（ingestion/training）。

        Orchestrator 不执行脚本，只是将任务转发给 Executor。
        Executor 负责加载脚本、执行 run()、写入数据湖。
        """
        try:
            logger.info(f"向 Executor 提交任务: {task_id}, 脚本: {payload['script']}")
            resp = httpx.post(
                f"{self.executor_url}/api/v1/tasks",
                json={
                    "script": payload["script"],
                    "input": payload["input"],
                    "output": payload["output"],
                    "params": payload.get("params", {}),
                },
            )
            resp.raise_for_status()
            executor_task_id = resp.json()["id"]
            with self._lock:
                # 保存 Executor 侧的任务 ID，用于后续状态同步
                self._tasks[task_id]["_executor_task_id"] = executor_task_id
            logger.info(f"Executor 已接受任务: {executor_task_id}")
        except Exception as e:
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.FAILED
                self._tasks[task_id]["error"] = str(e)
                self._tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"提交任务到 Executor 失败: {task_id}, 错误: {e}")

    def _sync_batch_status(self, task_id: str, executor_task_id: str):
        """从 Executor 同步批处理任务的最新状态。

        Orchestrator 通过轮询 Executor 的 GET /tasks/{id} 接口，
        将 Executor 侧的状态同步到本地。
        """
        try:
            resp = httpx.get(f"{self.executor_url}/api/v1/tasks/{executor_task_id}")
            resp.raise_for_status()
            data = resp.json()
            with self._lock:
                task = self._tasks[task_id]
                if data["status"] == "completed":
                    task["status"] = TaskStatus.COMPLETED
                    task["result"] = data.get("result")
                    task["completed_at"] = data.get("completed_at")
                    logger.info(f"任务已完成: {task_id}, 结果: {data.get('result')}")
                elif data["status"] == "failed":
                    task["status"] = TaskStatus.FAILED
                    task["error"] = data.get("error")
                    task["completed_at"] = data.get("completed_at")
                    logger.error(f"任务已失败: {task_id}, 错误: {data.get('error')}")
        except httpx.HTTPError as e:
            logger.warning(f"同步任务状态失败: {task_id}, 错误: {e}")

    def _start_inference(self, task_id: str, payload: dict):
        """启动推理服务：从数据湖加载模型权重到内存。

        流程:
        1. 通过 Executor API 获取模型的 Lance 文件路径
        2. 用 Daft 读取 Lance 文件中的权重字节
        3. 用 PyTorch 反序列化权重并加载到模型
        4. 模型进入 eval 模式，准备接收 predict 请求
        """
        model_name = payload.get("model", "")
        device = payload.get("device", "cpu")
        port = payload.get("port", 8080)

        try:
            # 第 1 步: 从 Executor 获取模型元信息（主要是 Lance 文件路径）
            logger.info(f"加载模型: {model_name}")
            resp = httpx.get(f"{self.executor_url}/api/v1/models/{model_name}")
            resp.raise_for_status()
            model_info = resp.json()
            model_path = model_info["path"]

            # 第 2 步: 从 Lance 文件读取模型权重（二进制）
            import daft
            df = daft.read_lance(model_path)
            pdf = df.to_pandas()
            weights_bytes = pdf["weights"].iloc[0]
            logger.info(f"已读取模型权重: {len(weights_bytes)} 字节")

            # 第 3 步: 反序列化权重并加载到 PyTorch 模型
            model = _MnistCNN()
            buf = io.BytesIO(weights_bytes)
            model.load_state_dict(torch.load(buf, map_location=device, weights_only=True))
            model.eval()  # 切换到推理模式（关闭 dropout、batchnorm 等）

            with self._lock:
                self._models[task_id] = model
                self._tasks[task_id]["endpoint"] = f"http://localhost:{port}"
            logger.info(f"推理服务就绪: {task_id}, 模型: {model_name}, 设备: {device}")
        except Exception as e:
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.FAILED
                self._tasks[task_id]["error"] = str(e)
                self._tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"启动推理服务失败: {task_id}, 错误: {e}")

    def predict(self, task_id: str, image_data: list[float]) -> dict:
        """执行推理。

        Args:
            task_id: 推理任务 ID
            image_data: 784 维浮点数组（28x28 归一化像素值，范围 [0, 1]）
        Returns:
            {"prediction": 预测数字, "confidence": 置信度, "probabilities": 10 类概率}
        Raises:
            ValueError: 模型未加载
        """
        model = self._models.get(task_id)
        if model is None:
            raise ValueError("模型未加载")

        # 将一维数组转为 (1, 1, 28, 28) 的四维张量: (batch, channel, height, width)
        tensor = torch.tensor(image_data, dtype=torch.float32).reshape(1, 1, 28, 28)
        with torch.no_grad():  # 推理时不需要计算梯度
            output = model(tensor)
        # softmax 将 logits 转为概率分布
        probs = torch.softmax(output, dim=1)

        prediction = probs.argmax().item()
        confidence = probs.max().item()
        logger.info(f"推理完成: 预测={prediction}, 置信度={confidence:.4f}")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs[0].tolist(),
        }

    def _public_view(self, task: dict) -> dict:
        """返回任务的公开视图，过滤掉内部字段（_开头）和 None 值。"""
        return {k: v for k, v in task.items() if not k.startswith("_") and v is not None}
