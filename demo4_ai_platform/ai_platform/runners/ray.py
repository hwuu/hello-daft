"""Level 2/3 Ray 执行器。任务提交为 Ray Task，Daft 使用 Ray 后端。"""

import threading

from ..runner import BaseRunner, _load_run_function

try:
    import daft
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False


class RayRunner(BaseRunner):
    """Ray Task 执行器（Level 2/3）。"""

    def __init__(self):
        if not _RAY_AVAILABLE:
            raise RuntimeError("Ray 未安装。Level 2/3 需要: pip install ray")
        super().__init__()
        if not ray.is_initialized():
            ray.init()

    def _start(self, task_id: str):
        # 在后台线程中提交 Ray Task 并等待结果
        # 这样不阻塞 API，同时复用 BaseRunner 的状态管理
        thread = threading.Thread(target=self._run_on_ray, args=(task_id,), daemon=True)
        thread.start()

    def _run_on_ray(self, task_id: str):
        """提交 Ray Task 并等待结果。"""
        with self._lock:
            task = self._tasks[task_id]
            script = task["script"]
            input_path = task["input"]
            output_path = task["output"]
            params = task["params"]

        try:
            ref = _ray_execute.remote(script, input_path, output_path, params)
            result = ray.get(ref)
            with self._lock:
                from ..runner import TaskStatus
                from datetime import datetime, timezone
                if task["status"] == TaskStatus.RUNNING:
                    task["status"] = TaskStatus.COMPLETED
                    task["result"] = result
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            with self._lock:
                from ..runner import TaskStatus
                from datetime import datetime, timezone
                if task["status"] == TaskStatus.RUNNING:
                    task["status"] = TaskStatus.FAILED
                    task["error"] = str(e)
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()


if _RAY_AVAILABLE:
    @ray.remote
    def _ray_execute(script: str, input_path: str, output_path: str, params: dict) -> dict:
        """在 Ray Worker 上执行用户脚本。"""
        daft.context.set_runner_ray()
        run_fn = _load_run_function(script)
        return run_fn(input_path, output_path, params)
