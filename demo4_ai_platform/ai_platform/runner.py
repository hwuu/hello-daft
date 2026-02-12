"""
脚本执行器模块。

负责加载用户脚本、调用 run() 函数、管理任务生命周期。
提供 BaseRunner 基类和 create_runner() 工厂函数，通过 PLATFORM_LEVEL 环境变量切换后端。

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
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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


class BaseRunner:
    """Runner 基类，管理任务状态。

    子类只需实现 _execute(task_id) 方法，定义具体的执行策略。
    """

    def __init__(self):
        self._tasks: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, script: str, input_path: str, output_path: str, params: dict) -> dict:
        """提交一个新任务。

        Args:
            name: 任务名称
            script: 用户脚本路径（需包含 run() 函数）
            input_path: 输入数据路径
            output_path: 输出数据路径
            params: 用户自定义参数
        Returns:
            任务公开视图
        """
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        task = {
            "id": task_id,
            "name": name,
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

        logger.info(f"任务已提交: {task_id}, 名称: {name}, 脚本: {script}")
        self._start(task_id)
        return self._public_view(task)

    def get(self, task_id: str) -> dict | None:
        """查询任务状态。返回 None 表示任务不存在。"""
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            return None
        return self._public_view(task)

    def list_all(self) -> list[dict]:
        """列出所有任务的摘要信息。"""
        with self._lock:
            tasks = list(self._tasks.values())
        return [
            {"id": t["id"], "name": t["name"], "status": t["status"], "created_at": t["created_at"]}
            for t in tasks
        ]

    def cancel(self, task_id: str) -> bool:
        """取消运行中的任务。"""
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

    def _public_view(self, task: dict) -> dict:
        """返回任务的公开视图，过滤掉内部字段（script）和 None 值。"""
        return {k: v for k, v in task.items() if k != "script" and v is not None}

    def _start(self, task_id: str):
        """启动任务执行。子类实现具体的调度策略。"""
        raise NotImplementedError

    def _execute(self, task_id: str):
        """执行用户脚本并更新任务状态。"""
        with self._lock:
            task = self._tasks[task_id]
            script = task["script"]
            input_path = task["input"]
            output_path = task["output"]
            params = task["params"]

        try:
            logger.info(f"开始执行任务 {task_id}: {script}")
            run_fn = _load_run_function(script)
            result = run_fn(input_path, output_path, params)
            with self._lock:
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


def create_runner(backend: str = "local") -> BaseRunner:
    """创建 Runner 实例。

    Args:
        backend: "local"（Level 1，线程执行）或 "ray"（Level 2/3，Ray Task 执行）
    """
    if backend == "ray":
        from .runners.ray import RayRunner
        return RayRunner()
    from .runners.local import LocalRunner
    return LocalRunner()
