"""
Orchestrator HTTP API 模块。

用户唯一入口。提供统一的任务管理 API 和数据湖代理查询。
Orchestrator 不直接操作数据，而是通过 Executor API 代理。

启动方式:
    uvicorn orchestrator.app:app --port 8000
"""

import logging
from contextlib import asynccontextmanager

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .tasks import TaskManager, TaskType

logger = logging.getLogger(__name__)

# 模块级变量，在 lifespan 中初始化
_manager: TaskManager | None = None
_executor_url: str = ""


def _load_config() -> dict:
    """加载配置文件。找不到时使用默认值。"""
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("未找到 config.yaml，使用默认配置")
        return {"executor": {"host": "http://localhost:8001"}, "orchestrator": {"host": "http://localhost:8000"}}


def create_app() -> FastAPI:
    """创建 Orchestrator FastAPI 应用。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理：启动时加载配置、初始化任务管理器。"""
        global _manager, _executor_url
        config = _load_config()
        _executor_url = config["executor"]["host"].rstrip("/")
        _manager = TaskManager(_executor_url)
        logger.info(f"Orchestrator 启动完成，Executor 地址: {_executor_url}")
        yield
        logger.info("Orchestrator 关闭")

    app = FastAPI(title="Orchestrator", lifespan=lifespan)

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # --- 数据集代理路由（转发到 Executor） ---

    @app.get("/api/v1/datasets")
    def list_datasets():
        """列出所有数据集（代理 Executor）。"""
        logger.info("收到请求: 列出数据集")
        return _proxy_get("/api/v1/datasets")

    @app.get("/api/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        """获取数据集详情（代理 Executor）。"""
        logger.info(f"收到请求: 查看数据集 {dataset_id}")
        return _proxy_get(f"/api/v1/datasets/{dataset_id}")

    @app.delete("/api/v1/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        """删除数据集（代理 Executor）。"""
        logger.info(f"收到请求: 删除数据集 {dataset_id}")
        return _proxy_delete(f"/api/v1/datasets/{dataset_id}")

    # --- 模型代理路由（转发到 Executor） ---

    @app.get("/api/v1/models")
    def list_models():
        """列出所有模型（代理 Executor）。"""
        logger.info("收到请求: 列出模型")
        return _proxy_get("/api/v1/models")

    @app.get("/api/v1/models/{model_id}")
    def get_model(model_id: str):
        """获取模型详情（代理 Executor）。"""
        logger.info(f"收到请求: 查看模型 {model_id}")
        return _proxy_get(f"/api/v1/models/{model_id}")

    @app.delete("/api/v1/models/{model_id}")
    def delete_model(model_id: str):
        """删除模型（代理 Executor）。"""
        logger.info(f"收到请求: 删除模型 {model_id}")
        return _proxy_delete(f"/api/v1/models/{model_id}")

    # --- 统一任务路由 ---

    class TaskRequest(BaseModel):
        """创建任务的请求体。

        三种任务类型共用同一请求体，通过 type 区分：
        - ingestion/training: 需要 input, script, output
        - inference: 需要 model
        """
        type: TaskType
        name: str
        # 批处理字段（ingestion/training 使用）
        input: str | None = None
        script: str | None = None
        params: dict = {}
        output: str | None = None
        # 推理字段（inference 使用）
        model: str | None = None
        device: str | None = None
        port: int | None = None

    @app.post("/api/v1/tasks", status_code=201)
    def create_task(req: TaskRequest):
        """创建任务。

        根据 type 字段校验必填参数，然后交给 TaskManager 处理。
        """
        logger.info(f"收到请求: 创建任务, 类型: {req.type}, 名称: {req.name}")
        # 参数校验：批处理任务需要 input/script/output
        if req.type in (TaskType.INGESTION, TaskType.TRAINING):
            if not all([req.input, req.script, req.output]):
                raise HTTPException(400, {"code": "INVALID_REQUEST", "message": "批处理任务需要 input, script, output 字段"})
        # 参数校验：推理任务需要 model
        elif req.type == TaskType.INFERENCE:
            if not req.model:
                raise HTTPException(400, {"code": "INVALID_REQUEST", "message": "推理任务需要 model 字段"})
        return _manager.create(req.type, req.model_dump(exclude_none=True))

    @app.get("/api/v1/tasks")
    def list_tasks(type: TaskType | None = Query(None)):
        """列出所有任务，可通过 ?type=ingestion 按类型过滤。"""
        logger.info(f"收到请求: 列出任务, 过滤类型: {type}")
        return _manager.list_all(type)

    @app.get("/api/v1/tasks/{task_id}")
    def get_task(task_id: str):
        """查询任务状态和详情。"""
        result = _manager.get(task_id)
        if result is None:
            raise HTTPException(404, {"code": "TASK_NOT_FOUND", "message": f"任务 '{task_id}' 不存在"})
        return result

    @app.post("/api/v1/tasks/{task_id}/cancel")
    def cancel_task(task_id: str):
        """取消/停止任务。"""
        logger.info(f"收到请求: 取消任务 {task_id}")
        if not _manager.cancel(task_id):
            raise HTTPException(404, {"code": "TASK_NOT_FOUND", "message": f"任务 '{task_id}' 不存在或未在运行"})
        return {"status": "cancelled"}

    class PredictRequest(BaseModel):
        """推理请求体。"""
        image: list[float]  # 784 维浮点数组（28x28 归一化像素值）

    @app.post("/api/v1/tasks/{task_id}/predict")
    def predict(task_id: str, body: PredictRequest):
        """调用推理（仅 type=inference 的任务支持）。"""
        task = _manager.get(task_id)
        if task is None:
            raise HTTPException(404, {"code": "TASK_NOT_FOUND", "message": f"任务 '{task_id}' 不存在"})
        if task.get("type") != TaskType.INFERENCE:
            raise HTTPException(400, {"code": "INVALID_REQUEST", "message": "只有推理任务支持 predict"})
        if task.get("status") != "running":
            raise HTTPException(400, {"code": "INVALID_REQUEST", "message": "推理服务未在运行"})
        logger.info(f"收到推理请求: {task_id}")
        try:
            return _manager.predict(task_id, body.image)
        except ValueError as e:
            raise HTTPException(500, {"code": "INFERENCE_ERROR", "message": str(e)})

    return app


def _proxy_get(path: str):
    """代理 GET 请求到 Executor。"""
    try:
        resp = httpx.get(f"{_executor_url}{path}")
        if resp.status_code == 404:
            raise HTTPException(404, resp.json())
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        logger.error(f"无法连接 Executor: {_executor_url}")
        raise HTTPException(502, {"code": "EXECUTOR_UNAVAILABLE", "message": "无法连接 Executor"})


def _proxy_delete(path: str):
    """代理 DELETE 请求到 Executor。"""
    try:
        resp = httpx.delete(f"{_executor_url}{path}")
        if resp.status_code == 404:
            raise HTTPException(404, resp.json())
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        logger.error(f"无法连接 Executor: {_executor_url}")
        raise HTTPException(502, {"code": "EXECUTOR_UNAVAILABLE", "message": "无法连接 Executor"})


# 默认应用实例，供 uvicorn 直接启动
app = create_app()
