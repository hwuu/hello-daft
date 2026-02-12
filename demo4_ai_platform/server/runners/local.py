"""Level 1 本地执行器。每个任务在独立线程中运行用户脚本。"""

import threading

from ..runner import BaseRunner


class LocalRunner(BaseRunner):
    """本地线程执行器（Level 1）。"""

    def _start(self, task_id: str):
        thread = threading.Thread(target=self._execute, args=(task_id,), daemon=True)
        thread.start()
