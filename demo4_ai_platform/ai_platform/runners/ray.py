"""Level 2/3 Ray 执行器。任务提交为 Ray Task。"""

import logging
import threading
from datetime import datetime, timezone

from ..runner import BaseRunner, TaskStatus, _load_run_function

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RayRunner(BaseRunner):
    """Ray Task 执行器（Level 2/3）。"""

    def __init__(self):
        if not _RAY_AVAILABLE:
            raise RuntimeError("Ray 未安装。Level 2/3 需要: pip install ray")
        super().__init__()
        if not ray.is_initialized():
            ray.init()

    def _start(self, task_id: str):
        thread = threading.Thread(target=self._execute, args=(task_id,), daemon=True)
        thread.start()

    def _execute(self, task_id: str):
        """重写基类 _execute，将脚本执行提交为 Ray Task。"""
        with self._lock:
            task = self._tasks[task_id]
            script = task["script"]
            input_path = task["input"]
            output_path = task["output"]
            params = task["params"]

        try:
            logger.info(f"提交 Ray Task: {task_id}, 脚本: {script}")
            ref = _ray_execute.remote(script, input_path, output_path, params)
            result = ray.get(ref)
            with self._lock:
                if task["status"] == TaskStatus.RUNNING:
                    task["status"] = TaskStatus.COMPLETED
                    task["result"] = result
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"Ray Task 完成: {task_id}")
        except Exception as e:
            with self._lock:
                if task["status"] == TaskStatus.RUNNING:
                    task["status"] = TaskStatus.FAILED
                    task["error"] = str(e)
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"Ray Task 失败: {task_id}, 错误: {e}")


if _RAY_AVAILABLE:
    @ray.remote
    def _ray_execute(script: str, input_path: str, output_path: str, params: dict) -> dict:
        """在 Ray Worker 上执行用户脚本。"""
        run_fn = _load_run_function(script)
        return run_fn(input_path, output_path, params)
