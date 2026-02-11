"""
AI Platform HTTP API 模块。

提供数据湖存储、计算任务、数据集/模型查询的 RESTful API。
用户唯一入口。

启动方式:
    uvicorn server.app:app --port 8000
"""

import logging
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .runner import TaskRunner
from .storage import Storage

logger = logging.getLogger(__name__)

# 模块级变量，在 lifespan 中初始化
_storage: Storage | None = None
_runner: TaskRunner | None = None


def _load_config() -> dict:
    """加载配置文件。找不到时使用默认值。"""
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("未找到 config.yaml，使用默认配置")
        return {"server": {"host": "http://localhost:8000", "storage_path": "./lance_storage"}}


def create_app(storage_path: str | None = None) -> FastAPI:
    """创建 AI Platform FastAPI 应用。

    Args:
        storage_path: 数据湖根目录路径（为 None 时从 config.yaml 读取）
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """应用生命周期管理：启动时初始化存储和执行器。"""
        global _storage, _runner
        path = storage_path
        if path is None:
            config = _load_config()
            path = config.get("server", {}).get("storage_path", "./lance_storage")
        _storage = Storage(path)
        _runner = TaskRunner()
        logger.info(f"AI Platform 启动完成，存储路径: {path}")
        yield
        logger.info("AI Platform 关闭")

    app = FastAPI(title="AI Platform", lifespan=lifespan)

    # 允许 Web 页面跨域访问 API（本地开发用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # --- 数据集路由 ---

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

    # --- 模型路由 ---

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
        """计算任务请求体。统一执行用户脚本的 run(input, output, params) 函数。"""
        name: str         # 任务名称
        script: str       # 用户脚本路径
        input: str = ""   # 输入数据路径
        output: str = ""  # 输出数据路径
        params: dict = {} # 用户自定义参数

    @app.post("/api/v1/tasks", status_code=201)
    def create_task(req: TaskRequest):
        """创建计算任务（执行用户脚本）。"""
        logger.info(f"收到请求: 创建任务, 名称: {req.name}, 脚本: {req.script}")
        return _runner.submit(req.name, req.script, req.input, req.output, req.params)

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
