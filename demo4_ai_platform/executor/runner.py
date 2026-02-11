"""
脚本执行器模块。

负责加载用户脚本、调用 run() 函数、管理任务生命周期。
每个任务在独立线程中执行，通过线程锁保证状态一致性。

用户脚本接口约定：
    def run(input_path: str, output_path: str, params: dict) -> dict
"""

import importlib.util
import logging
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """任务状态枚举。

    状态流转: RUNNING -> COMPLETED / FAILED
    注意: Level 1 中任务直接进入 RUNNING（无 PENDING），因为是单机串行。
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRunner:
    """脚本执行器。

    管理任务的提交、执行、查询和取消。
    任务在后台线程中执行用户脚本的 run() 函数。
    """

    def __init__(self):
        self._tasks: dict[str, dict] = {}  # task_id -> 任务状态字典
        self._lock = threading.Lock()       # 保护 _tasks 的并发访问

    def submit(self, script: str, input_path: str, output_path: str, params: dict) -> dict:
        """提交一个新任务。

        加载用户脚本并在后台线程中执行 run() 函数。

        Args:
            script: 用户脚本路径（需包含 run() 函数）
            input_path: 输入数据路径
            output_path: 输出数据路径
            params: 用户自定义参数
        Returns:
            任务摘要（id, status, created_at）
        """
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        task = {
            "id": task_id,
            "script": script,
            "input": input_path,
            "output": output_path,
            "params": params,
            "status": TaskStatus.RUNNING,
            "created_at": now,
            "completed_at": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._tasks[task_id] = task

        logger.info(f"任务已提交: {task_id}, 脚本: {script}")

        # 在后台线程中执行，避免阻塞 API 请求
        thread = threading.Thread(target=self._execute, args=(task_id,), daemon=True)
        thread.start()
        return {"id": task_id, "status": task["status"], "created_at": now}

    def get(self, task_id: str) -> dict | None:
        """查询任务状态。返回 None 表示任务不存在。"""
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return None
        # 不暴露脚本路径（内部信息）
        return {k: v for k, v in task.items() if k != "script"}

    def list_all(self) -> list[dict]:
        """列出所有任务的摘要信息。"""
        with self._lock:
            tasks = list(self._tasks.values())
        return [
            {"id": t["id"], "status": t["status"], "created_at": t["created_at"]}
            for t in tasks
        ]

    def cancel(self, task_id: str) -> bool:
        """取消运行中的任务。

        注意: Level 1 中无法真正中断线程，只是标记状态为 FAILED。
        脚本可能仍在后台运行直到自然结束。
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task["status"] == TaskStatus.RUNNING:
                task["status"] = TaskStatus.FAILED
                task["error"] = "cancelled"
                task["completed_at"] = datetime.now(timezone.utc).isoformat()
                logger.info(f"任务已取消: {task_id}")
                return True
        return False

    def _execute(self, task_id: str):
        """在后台线程中执行用户脚本。

        流程: 加载脚本 -> 调用 run() -> 更新任务状态
        """
        with self._lock:
            task = self._tasks[task_id]
            script = task["script"]
            input_path = task["input"]
            output_path = task["output"]
            params = task["params"]

        try:
            logger.info(f"开始执行任务 {task_id}: {script}")
            # 动态加载用户脚本并调用 run()
            run_fn = _load_run_function(script)
            result = run_fn(input_path, output_path, params)
            with self._lock:
                # 检查是否已被取消（取消会将状态改为 FAILED）
                if task["status"] == TaskStatus.RUNNING:
                    task["status"] = TaskStatus.COMPLETED
                    task["result"] = result
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"任务完成: {task_id}, 结果: {result}")
        except Exception as e:
            with self._lock:
                if task["status"] == TaskStatus.RUNNING:
                    task["status"] = TaskStatus.FAILED
                    task["error"] = str(e)
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"任务失败: {task_id}, 错误: {e}")


def _load_run_function(script_path: str):
    """动态加载用户脚本，返回其中的 run() 函数。

    用户脚本必须定义 run(input_path, output_path, params) -> dict 函数。
    使用 importlib 动态加载，避免硬编码依赖。

    Args:
        script_path: 脚本文件路径
    Returns:
        脚本中的 run 函数引用
    Raises:
        FileNotFoundError: 脚本文件不存在
        AttributeError: 脚本中没有定义 run() 函数
    """
    spec = importlib.util.spec_from_file_location("user_script", script_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"脚本未找到: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"脚本 {script_path} 必须定义 run() 函数")
    logger.info(f"已加载用户脚本: {script_path}")
    return module.run
