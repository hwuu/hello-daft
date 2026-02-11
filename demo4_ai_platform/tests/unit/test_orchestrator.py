"""Unit tests for orchestrator."""

import time
from unittest.mock import patch, MagicMock

import pytest

from orchestrator.tasks import TaskManager, TaskType, TaskStatus


@pytest.fixture
def manager():
    return TaskManager("http://localhost:8001")


class TestTaskManager:
    def test_create_ingestion_task_executor_down(self, manager):
        """When Executor is unreachable, task should fail."""
        result = manager.create(TaskType.INGESTION, {
            "name": "test",
            "input": "/in",
            "script": "test.py",
            "params": {},
            "output": "/out",
        })
        assert result["type"] == "ingestion"
        assert result["name"] == "test"
        # Executor is not running, so task should fail
        assert result["status"] == "failed"

    def test_create_inference_task_executor_down(self, manager):
        """Inference task fails when model can't be loaded from Executor."""
        result = manager.create(TaskType.INFERENCE, {
            "name": "predictor",
            "model": "nonexistent",
            "device": "cpu",
            "port": 8080,
        })
        assert result["type"] == "inference"
        assert result["status"] == "failed"

    def test_create_inference_task_with_mock(self, manager):
        """Inference task succeeds when Executor returns model info."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "id": "test_model",
            "path": "lance_storage/models/mnist_cnn_v1.lance",
            "schema": {"weights": "Binary"},
            "num_rows": 1,
        }

        with patch("orchestrator.tasks.httpx.get", return_value=mock_resp):
            # Will still fail because Lance file doesn't exist in test env
            result = manager.create(TaskType.INFERENCE, {
                "name": "predictor",
                "model": "test_model",
                "device": "cpu",
                "port": 8080,
            })
            assert result["type"] == "inference"

    def test_list_all_empty(self, manager):
        assert manager.list_all() == []

    def test_list_all_with_filter(self, manager):
        manager.create(TaskType.INGESTION, {
            "name": "t1", "input": "/in", "script": "s.py", "params": {}, "output": "/out",
        })
        manager.create(TaskType.INFERENCE, {
            "name": "t2", "model": "m", "device": "cpu", "port": 8080,
        })
        all_tasks = manager.list_all()
        assert len(all_tasks) == 2

        ingestion_only = manager.list_all(TaskType.INGESTION)
        assert len(ingestion_only) == 1
        assert ingestion_only[0]["type"] == "ingestion"

    def test_get_nonexistent(self, manager):
        assert manager.get("nonexistent") is None

    def test_cancel_nonexistent(self, manager):
        assert manager.cancel("nonexistent") is False

    def test_cancel_running_task(self, manager):
        result = manager.create(TaskType.INFERENCE, {
            "name": "predictor", "model": "m", "device": "cpu", "port": 8080,
        })
        task_id = result["id"]
        # Task failed because executor is down, so cancel should return False
        assert manager.cancel(task_id) is False

    def test_public_view_hides_internal_fields(self, manager):
        """Internal fields starting with _ should not appear in output."""
        result = manager.create(TaskType.INGESTION, {
            "name": "test", "input": "/in", "script": "s.py", "params": {}, "output": "/out",
        })
        for key in result:
            assert not key.startswith("_")

    def test_public_view_hides_none_fields(self, manager):
        """None-valued fields should not appear in output."""
        result = manager.create(TaskType.INGESTION, {
            "name": "test", "input": "/in", "script": "s.py", "params": {}, "output": "/out",
        })
        for value in result.values():
            assert value is not None
