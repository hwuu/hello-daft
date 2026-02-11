"""
统一任务管理模块。

平台只做一件事——跑用户脚本。
脚本跑完就 completed，报错就 failed，cancel 就 cancelled。
所有任务统一通过 Executor API 提交脚本执行。
"""

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from threading import Lock

import httpx

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态枚举。状态流转见 design.md 4.3 节。"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskManager:
    """统一任务管理器。

    对外提供 create / get / list / cancel 方法。
    所有任务统一通过 Executor HTTP API 提交脚本执行。
    """

    def __init__(self, executor_url: str):
        self.executor_url = executor_url.rstrip("/")
        self._tasks: dict[str, dict] = {}
        self._lock = Lock()
        logger.info(f"任务管理器初始化完成，Executor 地址: {self.executor_url}")

    def create(self, payload: dict) -> dict:
        """创建并启动任务。

        Args:
            payload: 请求体（包含 name, input, script, output, params 字段）
        Returns:
            任务状态字典（过滤掉内部字段和 None 值）
        """
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        task = {
            "id": task_id,
            "name": payload.get("name", ""),
            "input": payload.get("input", ""),
            "script": payload.get("script", ""),
            "output": payload.get("output", ""),
            "params": payload.get("params", {}),
            "status": TaskStatus.RUNNING,
            "created_at": now,
            "completed_at": None,
            "result": None,
            "error": None,
        }

        with self._lock:
            self._tasks[task_id] = task

        logger.info(f"创建任务: {task_id}, 名称: {payload.get('name', '')}")

        self._submit_to_executor(task_id, payload)

        with self._lock:
            return self._public_view(self._tasks[task_id])

    def get(self, task_id: str) -> dict | None:
        """查询任务状态。

        对于运行中的任务，会主动轮询 Executor 获取最新状态。
        """
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return None

        # 运行中的任务主动同步 Executor 的状态
        if task["status"] == TaskStatus.RUNNING:
            executor_task_id = task.get("_executor_task_id")
            if executor_task_id:
                self._sync_executor_status(task_id, executor_task_id)

        with self._lock:
            return self._public_view(self._tasks[task_id])

    def list_all(self) -> list[dict]:
        """列出所有任务。"""
        with self._lock:
            tasks = list(self._tasks.values())
        return [self._public_view(t) for t in tasks]

    def cancel(self, task_id: str) -> bool:
        """取消/停止任务。向 Executor 发送取消请求。"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task["status"] != TaskStatus.RUNNING:
                return False

        logger.info(f"取消任务: {task_id}")

        executor_task_id = task.get("_executor_task_id")
        if executor_task_id:
            try:
                httpx.post(f"{self.executor_url}/api/v1/tasks/{executor_task_id}/cancel")
                logger.info(f"已通知 Executor 取消任务: {executor_task_id}")
            except httpx.HTTPError as e:
                logger.warning(f"通知 Executor 取消失败: {e}")

        with self._lock:
            task["status"] = TaskStatus.FAILED
            task["error"] = "cancelled"
            task["completed_at"] = datetime.now(timezone.utc).isoformat()
        return True

    def _submit_to_executor(self, task_id: str, payload: dict):
        """向 Executor 提交任务。

        Orchestrator 不执行脚本，只是将任务转发给 Executor。
        Executor 负责加载脚本、执行 run()、写入数据湖。
        """
        try:
            logger.info(f"向 Executor 提交任务: {task_id}, 脚本: {payload['script']}")
            resp = httpx.post(
                f"{self.executor_url}/api/v1/tasks",
                json={
                    "script": payload["script"],
                    "input": payload.get("input", ""),
                    "output": payload.get("output", ""),
                    "params": payload.get("params", {}),
                },
            )
            resp.raise_for_status()
            executor_task_id = resp.json()["id"]
            with self._lock:
                self._tasks[task_id]["_executor_task_id"] = executor_task_id
            logger.info(f"Executor 已接受任务: {executor_task_id}")
        except Exception as e:
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.FAILED
                self._tasks[task_id]["error"] = str(e)
                self._tasks[task_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"提交任务到 Executor 失败: {task_id}, 错误: {e}")

    def _sync_executor_status(self, task_id: str, executor_task_id: str):
        """从 Executor 同步任务的最新状态。"""
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

    def _public_view(self, task: dict) -> dict:
        """返回任务的公开视图，过滤掉内部字段（_开头）和 None 值。"""
        return {k: v for k, v in task.items() if not k.startswith("_") and v is not None}
