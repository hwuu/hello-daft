"""Unit tests for AI Platform server."""

import time

import daft
import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from ai_platform.app import create_app
from ai_platform.runner import _load_run_function
from ai_platform.runners.local import LocalRunner
from ai_platform.storage import Storage


# --- Fixtures ---


@pytest.fixture
def tmp_storage(tmp_path):
    storage = Storage(str(tmp_path))
    return storage


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a small Lance dataset for testing."""
    storage_path = tmp_path / ".ai_platform"
    datasets_path = storage_path / "datasets"
    datasets_path.mkdir(parents=True)

    table = pa.table({
        "image": [[0.1] * 784, [0.2] * 784, [0.3] * 784],
        "label": [0, 1, 2],
        "split": ["train", "train", "test"],
    })
    df = daft.from_arrow(table)
    output_path = str(datasets_path / "test_data.lance")
    df.write_lance(output_path, mode="overwrite")
    return storage_path


@pytest.fixture
def sample_script(tmp_path):
    """Create a simple user script for testing."""
    script = tmp_path / "test_script.py"
    script.write_text(
        "def run(input_path, output_path, params):\n"
        "    return {'status': 'ok', 'input': input_path, 'params': params}\n"
    )
    return str(script)


@pytest.fixture
def failing_script(tmp_path):
    script = tmp_path / "fail_script.py"
    script.write_text(
        "def run(input_path, output_path, params):\n"
        "    raise ValueError('intentional error')\n"
    )
    return str(script)


@pytest.fixture
def slow_script(tmp_path):
    """Create a script that sleeps, for testing cancel."""
    script = tmp_path / "slow_script.py"
    script.write_text(
        "import time\n"
        "def run(input_path, output_path, params):\n"
        "    time.sleep(10)\n"
        "    return {'status': 'ok'}\n"
    )
    return str(script)


@pytest.fixture
def client(sample_dataset):
    app = create_app(str(sample_dataset))
    with TestClient(app) as client:
        yield client


# --- Storage Tests ---


class TestStorage:
    def test_list_datasets_empty(self, tmp_storage):
        assert tmp_storage.list_datasets() == []

    def test_list_models_empty(self, tmp_storage):
        assert tmp_storage.list_models() == []

    def test_list_datasets(self, sample_dataset):
        storage = Storage(str(sample_dataset))
        datasets = storage.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["id"] == "test_data"
        assert datasets[0]["num_rows"] == 3

    def test_get_dataset(self, sample_dataset):
        storage = Storage(str(sample_dataset))
        ds = storage.get_dataset("test_data")
        assert ds is not None
        assert ds["id"] == "test_data"
        assert "image" in ds["schema"]
        assert "label" in ds["schema"]

    def test_get_dataset_not_found(self, tmp_storage):
        assert tmp_storage.get_dataset("nonexistent") is None

    def test_delete_dataset(self, sample_dataset):
        storage = Storage(str(sample_dataset))
        assert storage.delete_dataset("test_data") is True
        assert storage.list_datasets() == []

    def test_delete_dataset_not_found(self, tmp_storage):
        assert tmp_storage.delete_dataset("nonexistent") is False


# --- Runner Tests ---


class TestRunner:
    def test_load_run_function(self, sample_script):
        fn = _load_run_function(sample_script)
        result = fn("/input", "/output", {"key": "val"})
        assert result == {"status": "ok", "input": "/input", "params": {"key": "val"}}

    def test_load_missing_script(self):
        with pytest.raises(FileNotFoundError):
            _load_run_function("/nonexistent/script.py")

    def test_load_script_without_run(self, tmp_path):
        script = tmp_path / "no_run.py"
        script.write_text("x = 1\n")
        with pytest.raises(AttributeError):
            _load_run_function(str(script))

    def test_submit_and_complete(self, sample_script):
        runner = LocalRunner()
        result = runner.submit("test", sample_script, "/in", "/out", {"a": 1})
        assert "id" in result
        assert result["name"] == "test"
        assert result["status"] == "running"

        # Wait for completion
        time.sleep(0.5)
        task = runner.get(result["id"])
        assert task["status"] == "completed"
        assert task["result"]["status"] == "ok"

    def test_submit_failing_script(self, failing_script):
        runner = LocalRunner()
        result = runner.submit("fail_test", failing_script, "/in", "/out", {})
        time.sleep(0.5)
        task = runner.get(result["id"])
        assert task["status"] == "failed"
        assert "intentional error" in task["error"]

    def test_list_tasks(self, sample_script):
        runner = LocalRunner()
        runner.submit("test", sample_script, "/in", "/out", {})
        tasks = runner.list_all()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "test"

    def test_get_nonexistent(self):
        runner = LocalRunner()
        assert runner.get("nonexistent") is None

    def test_cancel_nonexistent(self):
        runner = LocalRunner()
        assert runner.cancel("nonexistent") is False

    def test_cancel_running_task(self, slow_script):
        runner = LocalRunner()
        result = runner.submit("slow", slow_script, "/in", "/out", {})
        task_id = result["id"]
        assert result["status"] == "running"

        assert runner.cancel(task_id) is True
        task = runner.get(task_id)
        assert task["status"] == "failed"
        assert task["error"] == "cancelled"

    def test_cancel_completed_task_returns_false(self, sample_script):
        runner = LocalRunner()
        result = runner.submit("test", sample_script, "/in", "/out", {})
        time.sleep(0.5)
        assert runner.cancel(result["id"]) is False

    def test_public_view_hides_none_fields(self, sample_script):
        runner = LocalRunner()
        result = runner.submit("test", sample_script, "/in", "/out", {})
        for value in result.values():
            assert value is not None

    def test_public_view_hides_script_field(self, sample_script):
        runner = LocalRunner()
        result = runner.submit("test", sample_script, "/in", "/out", {})
        assert "script" not in result

    def test_no_type_field_in_response(self, sample_script):
        runner = LocalRunner()
        result = runner.submit("test", sample_script, "/in", "/out", {})
        assert "type" not in result


# --- API Tests ---


class TestAPI:
    def test_list_datasets(self, client):
        resp = client.get("/api/v1/datasets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "test_data"

    def test_get_dataset(self, client):
        resp = client.get("/api/v1/datasets/test_data")
        assert resp.status_code == 200
        assert resp.json()["num_rows"] == 3

    def test_get_dataset_not_found(self, client):
        resp = client.get("/api/v1/datasets/nonexistent")
        assert resp.status_code == 404

    def test_list_models_empty(self, client):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_task(self, client, sample_script):
        resp = client.post("/api/v1/tasks", json={
            "name": "test_task",
            "script": sample_script,
            "input": "/in",
            "output": "/out",
            "params": {},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "running"
        assert data["name"] == "test_task"

        # Wait and check
        time.sleep(0.5)
        resp = client.get(f"/api/v1/tasks/{data['id']}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_create_task_with_params(self, client, sample_script):
        resp = client.post("/api/v1/tasks", json={
            "name": "mnist_serve",
            "script": sample_script,
            "input": ".ai_platform/models/mnist_cnn_v1.lance",
            "output": "",
            "params": {"device": "cpu", "port": 8080},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["params"] == {"device": "cpu", "port": 8080}

    def test_get_task_not_found(self, client):
        resp = client.get("/api/v1/tasks/nonexistent")
        assert resp.status_code == 404

    def test_list_tasks(self, client, sample_script):
        client.post("/api/v1/tasks", json={
            "name": "test", "script": sample_script, "input": "/in", "output": "/out", "params": {},
        })
        resp = client.get("/api/v1/tasks")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_cancel_task(self, client, slow_script):
        resp = client.post("/api/v1/tasks", json={
            "name": "slow", "script": slow_script, "input": "/in", "output": "/out", "params": {},
        })
        task_id = resp.json()["id"]

        resp = client.post(f"/api/v1/tasks/{task_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_nonexistent_task(self, client):
        resp = client.post("/api/v1/tasks/nonexistent/cancel")
        assert resp.status_code == 404
