"""Orchestrator HTTP API: unified task management + proxy to Executor."""

from contextlib import asynccontextmanager

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .tasks import TaskManager, TaskType

_manager: TaskManager | None = None
_executor_url: str = ""


def _load_config() -> dict:
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"executor": {"host": "http://localhost:8001"}, "orchestrator": {"host": "http://localhost:8000"}}


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _manager, _executor_url
        config = _load_config()
        _executor_url = config["executor"]["host"].rstrip("/")
        _manager = TaskManager(_executor_url)
        yield

    app = FastAPI(title="Orchestrator", lifespan=lifespan)

    # --- Proxy: Datasets (forward to Executor) ---

    @app.get("/api/v1/datasets")
    def list_datasets():
        return _proxy_get("/api/v1/datasets")

    @app.get("/api/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        return _proxy_get(f"/api/v1/datasets/{dataset_id}")

    @app.delete("/api/v1/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        return _proxy_delete(f"/api/v1/datasets/{dataset_id}")

    # --- Proxy: Models (forward to Executor) ---

    @app.get("/api/v1/models")
    def list_models():
        return _proxy_get("/api/v1/models")

    @app.get("/api/v1/models/{model_id}")
    def get_model(model_id: str):
        return _proxy_get(f"/api/v1/models/{model_id}")

    @app.delete("/api/v1/models/{model_id}")
    def delete_model(model_id: str):
        return _proxy_delete(f"/api/v1/models/{model_id}")

    # --- Tasks ---

    class TaskRequest(BaseModel):
        type: TaskType
        name: str
        # Batch fields (ingestion/training)
        input: str | None = None
        script: str | None = None
        params: dict = {}
        output: str | None = None
        # Inference fields
        model: str | None = None
        device: str | None = None
        port: int | None = None

    @app.post("/api/v1/tasks", status_code=201)
    def create_task(req: TaskRequest):
        if req.type in (TaskType.INGESTION, TaskType.TRAINING):
            if not all([req.input, req.script, req.output]):
                raise HTTPException(400, {"code": "INVALID_REQUEST", "message": "Batch tasks require input, script, and output"})
        elif req.type == TaskType.INFERENCE:
            if not req.model:
                raise HTTPException(400, {"code": "INVALID_REQUEST", "message": "Inference tasks require model"})
        return _manager.create(req.type, req.model_dump(exclude_none=True))

    @app.get("/api/v1/tasks")
    def list_tasks(type: TaskType | None = Query(None)):
        return _manager.list_all(type)

    @app.get("/api/v1/tasks/{task_id}")
    def get_task(task_id: str):
        result = _manager.get(task_id)
        if result is None:
            raise HTTPException(404, {"code": "TASK_NOT_FOUND", "message": f"Task '{task_id}' not found"})
        return result

    @app.post("/api/v1/tasks/{task_id}/cancel")
    def cancel_task(task_id: str):
        if not _manager.cancel(task_id):
            raise HTTPException(404, {"code": "TASK_NOT_FOUND", "message": f"Task '{task_id}' not found or not running"})
        return {"status": "cancelled"}

    class PredictRequest(BaseModel):
        image: list[float]  # 784 floats (28x28 normalized pixel values)

    @app.post("/api/v1/tasks/{task_id}/predict")
    def predict(task_id: str, body: PredictRequest):
        task = _manager.get(task_id)
        if task is None:
            raise HTTPException(404, {"code": "TASK_NOT_FOUND", "message": f"Task '{task_id}' not found"})
        if task.get("type") != TaskType.INFERENCE:
            raise HTTPException(400, {"code": "INVALID_REQUEST", "message": "Only inference tasks support predict"})
        if task.get("status") != "running":
            raise HTTPException(400, {"code": "INVALID_REQUEST", "message": "Inference service is not running"})
        try:
            return _manager.predict(task_id, body.image)
        except ValueError as e:
            raise HTTPException(500, {"code": "INFERENCE_ERROR", "message": str(e)})

    return app


def _proxy_get(path: str):
    try:
        resp = httpx.get(f"{_executor_url}{path}")
        if resp.status_code == 404:
            raise HTTPException(404, resp.json())
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        raise HTTPException(502, {"code": "EXECUTOR_UNAVAILABLE", "message": "Cannot connect to Executor"})


def _proxy_delete(path: str):
    try:
        resp = httpx.delete(f"{_executor_url}{path}")
        if resp.status_code == 404:
            raise HTTPException(404, resp.json())
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        raise HTTPException(502, {"code": "EXECUTOR_UNAVAILABLE", "message": "Cannot connect to Executor"})


app = create_app()
