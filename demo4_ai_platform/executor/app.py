"""
Executor HTTP API 模块。

提供数据湖存储和计算任务的 RESTful API。
Executor 是内部服务，由 Orchestrator 调用，不直接面向用户。

启动方式:
    uvicorn executor.app:app --port 8001
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .runner import TaskRunner
from .storage import Storage

logger = logging.getLogger(__name__)

# 模块级变量，在 lifespan 中初始化
_storage: Storage | None = None
_runner: TaskRunner | None = None


def create_app(storage_path: str = "./lance_storage") -> FastAPI:
    """创建 Executor FastAPI 应用。

    Args:
        storage_path: 数据湖根目录路径
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理：启动时初始化存储和执行器。"""
        global _storage, _runner
        _storage = Storage(storage_path)
        _runner = TaskRunner()
        logger.info(f"Executor 启动完成，存储路径: {storage_path}")
        yield
        logger.info("Executor 关闭")

    app = FastAPI(title="Executor", lifespan=lifespan)

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # --- 数据集路由（供 Orchestrator 代理查询） ---

    @app.get("/api/v1/datasets")
    def list_datasets():
        """列出数据湖中所有数据集。"""
        logger.info("收到请求: 列出数据集")
        return _storage.list_datasets()

    @app.get("/api/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        """获取数据集详情（schema、行数等）。"""
        logger.info(f"收到请求: 查看数据集 {dataset_id}")
        result = _storage.get_dataset(dataset_id)
        if result is None:
            raise HTTPException(404, {"code": "DATASET_NOT_FOUND", "message": f"数据集 '{dataset_id}' 不存在"})
        return result

    @app.delete("/api/v1/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        """删除数据集。"""
        logger.info(f"收到请求: 删除数据集 {dataset_id}")
        if not _storage.delete_dataset(dataset_id):
            raise HTTPException(404, {"code": "DATASET_NOT_FOUND", "message": f"数据集 '{dataset_id}' 不存在"})
        return {"status": "deleted"}

    # --- 模型路由（供 Orchestrator 代理查询） ---

    @app.get("/api/v1/models")
    def list_models():
        """列出数据湖中所有模型。"""
        logger.info("收到请求: 列出模型")
        return _storage.list_models()

    @app.get("/api/v1/models/{model_id}")
    def get_model(model_id: str):
        """获取模型详情（权重 schema、指标等）。"""
        logger.info(f"收到请求: 查看模型 {model_id}")
        result = _storage.get_model(model_id)
        if result is None:
            raise HTTPException(404, {"code": "MODEL_NOT_FOUND", "message": f"模型 '{model_id}' 不存在"})
        return result

    @app.delete("/api/v1/models/{model_id}")
    def delete_model(model_id: str):
        """删除模型。"""
        logger.info(f"收到请求: 删除模型 {model_id}")
        if not _storage.delete_model(model_id):
            raise HTTPException(404, {"code": "MODEL_NOT_FOUND", "message": f"模型 '{model_id}' 不存在"})
        return {"status": "deleted"}

    # --- 计算任务路由 ---

    class TaskRequest(BaseModel):
        """计算任务请求体。

        Executor 的任务不区分 type（ingestion/training），
        统一执行用户脚本的 run(input, output, params) 函数。
        type 的区分由 Orchestrator 负责。
        """
        script: str   # 用户脚本路径
        input: str    # 输入数据路径
        output: str   # 输出数据路径
        params: dict = {}  # 用户自定义参数

    @app.post("/api/v1/tasks", status_code=201)
    def submit_task(req: TaskRequest):
        """提交计算任务（执行用户脚本）。"""
        logger.info(f"收到请求: 提交任务, 脚本: {req.script}")
        return _runner.submit(req.script, req.input, req.output, req.params)

    @app.get("/api/v1/tasks")
    def list_tasks():
        """列出所有计算任务。"""
        return _runner.list_all()

    @app.get("/api/v1/tasks/{task_id}")
    def get_task(task_id: str):
        """查询任务状态和详情。"""
        result = _runner.get(task_id)
        if result is None:
            raise HTTPException(404, {"code": "TASK_NOT_FOUND", "message": f"任务 '{task_id}' 不存在"})
        return result

    @app.post("/api/v1/tasks/{task_id}/cancel")
    def cancel_task(task_id: str):
        """取消运行中的任务。"""
        logger.info(f"收到请求: 取消任务 {task_id}")
        if not _runner.cancel(task_id):
            raise HTTPException(404, {"code": "TASK_NOT_FOUND", "message": f"任务 '{task_id}' 不存在或未在运行"})
        return {"status": "cancelled"}

    return app


# 默认应用实例，供 uvicorn 直接启动
app = create_app()
