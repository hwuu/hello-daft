"""Unit tests for orchestrator."""

from unittest.mock import patch, MagicMock

import pytest

from orchestrator.tasks import TaskManager, TaskStatus


@pytest.fixture
def manager():
    return TaskManager("http://localhost:8001")


class TestTaskManager:
    def test_create_task_executor_down(self, manager):
        """When Executor is unreachable, task should fail."""
        result = manager.create({
            "name": "test",
            "input": "/in",
            "script": "test.py",
            "params": {},
            "output": "/out",
        })
        assert result["name"] == "test"
        assert result["script"] == "test.py"
        # Executor is not running, so task should fail
        assert result["status"] == "failed"

    def test_create_task_with_mock(self, manager):
        """Task succeeds when Executor accepts it."""
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "id": "task-abc123",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
        }

        with patch("orchestrator.tasks.httpx.post", return_value=mock_resp):
            result = manager.create({
                "name": "mnist_ingestion",
                "input": "download",
                "script": "mnist/mnist_clean.py",
                "output": "lance_storage/datasets/mnist_clean.lance",
                "params": {"normalize": True},
            })
            assert result["status"] == "running"
            assert result["name"] == "mnist_ingestion"

    def test_create_serving_task_with_mock(self, manager):
        """Serving task (port in params) succeeds when Executor accepts it."""
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "id": "task-serve1",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
        }

        with patch("orchestrator.tasks.httpx.post", return_value=mock_resp):
            result = manager.create({
                "name": "mnist_serve",
                "input": "lance_storage/models/mnist_cnn_v1.lance",
                "script": "mnist/mnist_serve.py",
                "output": "",
                "params": {"device": "cpu", "port": 8080},
            })
            assert result["status"] == "running"
            assert result["params"] == {"device": "cpu", "port": 8080}

    def test_list_all_empty(self, manager):
        assert manager.list_all() == []

    def test_list_all_returns_all_tasks(self, manager):
        manager.create({
            "name": "t1", "input": "/in", "script": "s.py", "params": {}, "output": "/out",
        })
        manager.create({
            "name": "t2", "input": "/in2", "script": "s2.py", "params": {}, "output": "/out2",
        })
        all_tasks = manager.list_all()
        assert len(all_tasks) == 2

    def test_get_nonexistent(self, manager):
        assert manager.get("nonexistent") is None

    def test_cancel_nonexistent(self, manager):
        assert manager.cancel("nonexistent") is False

    def test_cancel_failed_task_returns_false(self, manager):
        result = manager.create({
            "name": "test", "input": "/in", "script": "s.py", "params": {}, "output": "/out",
        })
        task_id = result["id"]
        # Task failed because executor is down, so cancel should return False
        assert manager.cancel(task_id) is False

    def test_cancel_running_task_with_mock(self, manager):
        """Cancel a running task."""
        mock_post = MagicMock()
        mock_post.status_code = 201
        mock_post.raise_for_status = MagicMock()
        mock_post.json.return_value = {
            "id": "task-exec1",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
        }

        with patch("orchestrator.tasks.httpx.post", return_value=mock_post):
            result = manager.create({
                "name": "test", "input": "/in", "script": "s.py", "params": {}, "output": "/out",
            })
            task_id = result["id"]
            assert result["status"] == "running"

        # Cancel the running task
        with patch("orchestrator.tasks.httpx.post") as mock_cancel:
            assert manager.cancel(task_id) is True

        # Verify task is now failed with cancelled error
        task = manager.get(task_id)
        assert task["status"] == "failed"
        assert task["error"] == "cancelled"

    def test_public_view_hides_internal_fields(self, manager):
        """Internal fields starting with _ should not appear in output."""
        result = manager.create({
            "name": "test", "input": "/in", "script": "s.py", "params": {}, "output": "/out",
        })
        for key in result:
            assert not key.startswith("_")

    def test_public_view_hides_none_fields(self, manager):
        """None-valued fields should not appear in output."""
        result = manager.create({
            "name": "test", "input": "/in", "script": "s.py", "params": {}, "output": "/out",
        })
        for value in result.values():
            assert value is not None

    def test_no_type_field_in_response(self, manager):
        """Response should not contain a 'type' field."""
        result = manager.create({
            "name": "test", "input": "/in", "script": "s.py", "params": {}, "output": "/out",
        })
        assert "type" not in result
