# Demo 4: AI Platform 设计文档

## 目录

- [1. 背景与目标](#1-背景与目标)
  - [1.1 问题](#11-问题)
  - [1.2 火山引擎的启发](#12-火山引擎的启发)
  - [1.3 设计目标](#13-设计目标)
- [2. 架构设计](#2-架构设计)
  - [2.1 核心分层](#21-核心分层)
  - [2.2 解耦设计](#22-解耦设计)
- [3. 任务模型](#3-任务模型)
  - [3.1 统一脚本接口](#31-统一脚本接口)
  - [3.2 任务状态机](#32-任务状态机)
  - [3.3 数据入库](#33-数据入库)
  - [3.4 模型训练](#34-模型训练)
  - [3.5 推理服务](#35-推理服务)
- [4. Server](#4-server)
  - [4.1 组件职责](#41-组件职责)
  - [4.2 RESTful API](#42-restful-api)
  - [4.3 数据湖目录结构](#43-数据湖目录结构)
  - [4.4 错误处理](#44-错误处理)
- [5. MNIST 实现规划](#5-mnist-实现规划)
  - [5.1 目录结构](#51-目录结构)
  - [5.2 实现步骤](#52-实现步骤)
  - [5.3 技术选型](#53-技术选型)
  - [5.4 配置](#54-配置)
- [6. 部署级别](#6-部署级别)
  - [6.1 Level 1: 单机单任务](#61-level-1-单机单任务)
  - [6.2 Level 2: 单机多任务](#62-level-2-单机多任务)
  - [6.3 Level 3: 多机多任务](#63-level-3-多机多任务)
  - [6.4 Ray on K8s 的能力边界](#64-ray-on-k8s-的能力边界)
  - [6.5 多租户与资源分配](#65-多租户与资源分配)
- [参考文献](#参考文献)

## 1. 背景与目标

### 1.1 问题

在实际的 AI/ML 工作流中，数据处理和模型训练/推理是两个不同的关注点：

- **数据工程师**关心：数据如何入库、清洗、存储、查询
- **算法工程师**关心：如何获取训练数据、训练模型、部署推理服务

这两类工作通常耦合在一起，导致数据处理逻辑和训练逻辑混杂，难以独立演进。

### 1.2 火山引擎的启发

参考火山引擎 LAS（LakeHouse AI Service）的架构，Daft + Ray + Lance 组合提供了一套完整的数据平台能力：

- **Lance 列式存储的高压缩比**：100GB Tensor 数据可压缩到 2GB，适合存储模型权重和图像数据
- **零拷贝 Schema 演进**：数据湖中的数据集可以灵活添加新列（如特征列），不需要重写整个数据集
- **Row ID 替代内容 Join**：在多模态场景下，用 Row ID 关联图像和标签，避免大数据量 Join
- **懒加载**：Daft 的 URL 列 + Lance 的按需读取，只在真正需要时才加载数据

### 1.3 设计目标

构建一个单服务的 **AI Platform**，统一提供数据湖存储、脚本执行和 API 查询：

- **存储层**：基于 Lance + LanceDB，统一存储数据集和模型
- **计算层**：基于 Daft（可选 Ray），执行用户脚本
- **API 层**：基于 FastAPI，提供 RESTful API

平台不绑定特定数据集或模型，所有业务逻辑由用户脚本实现。

## 2. 架构设计

### 2.1 核心分层

```
+----------------------------------------------------------------------+
|                           AI Platform                                |
|                                                                      |
|  +----------------------------------------------------------------+  |
|  |  Server (RESTful API: FastAPI)                                 |  |
|  |  +----------------------------------------------------------+  |  |
|  |  | Task Runner          | Storage          | API Routes     |  |  |
|  |  | (Script Executor)    | (Lance + LanceDB)| (REST)         |  |  |
|  |  +----------------------------------------------------------+  |  |
|  |  +-----------------+  +--------------------+  +--------------+ |  |
|  |  | Daft            |  | Ray                |  | Lance +      | |  |
|  |  | (Compute)       |  | (Optional Runtime) |  | LanceDB      | |  |
|  |  |                 |  |                    |  | (Storage)    | |  |
|  |  +-----------------+  +--------------------+  +--------------+ |  |
|  +----------------------------------------------------------------+  |
+----------------------------------------------------------------------+
```

用户通过 Server 的 RESTful API 操作全部功能。任务是高度可定制的——用户提供自己的脚本，平台只负责调度和存储（详见 [2.2 解耦设计](#22-解耦设计)）。

### 2.2 解耦设计

平台与用户脚本通过 **Lance 格式 + run() 接口**解耦。

#### 脚本与平台的解耦

平台不关心业务逻辑，只提供基础设施：

| 平台提供的 | 用户脚本负责的 |
|---|---|
| 数据读写（Lance） | 清洗逻辑、特征工程 |
| 计算资源（Daft/Ray） | 训练代码和超参数 |
| 存储管理 | 模型版本管理策略 |
| 任务调度和生命周期 | 推理服务和 API 设计 |

用户脚本只需实现 `run(input_path, output_path, params) → dict`，平台负责调用和管理。

#### 可替换性

- **替换存储**（如 Lance → Parquet + Milvus）：改 Storage 模块即可，用户脚本不受影响
- **替换计算**（如 Daft → Spark）：改 Runner 后端即可，只要脚本接口不变
- **替换训练框架**（如 PyTorch → TensorFlow）：平台不受影响，用户脚本自行选择

## 3. 任务模型

平台不绑定特定数据集或模型。所有任务采用统一的**脚本模式**：平台负责调度和存储，用户脚本负责业务逻辑。

### 3.1 统一脚本接口

所有任务共用同一个接口约定：

```python
def run(input_path: str, output_path: str, params: dict) -> dict
```

平台不区分批处理和服务，不关心脚本内部做什么。三种典型任务的 API 调用：

**数据入库：**

```
POST /api/v1/tasks
{
    "name": "mnist_ingestion",
    "input": "/data/raw/mnist/",
    "script": "mnist/mnist_clean.py",
    "params": {"normalize": true, "flatten": true},
    "output": "mnist_clean"
}
```

**模型训练：**

```
POST /api/v1/tasks
{
    "name": "mnist_cnn_v1",
    "input": "mnist_clean",
    "script": "mnist/mnist_cnn.py",
    "params": {"epochs": 10, "learning_rate": 0.001, "batch_size": 64, "device": "cpu"},
    "output": "mnist_cnn_v1"
}
```

**推理服务：**

```
POST /api/v1/tasks
{
    "name": "mnist_serve",
    "input": "lance_storage/models/mnist_cnn_v1.lance",
    "script": "mnist/mnist_serve.py",
    "output": "",
    "params": {"device": "cpu", "port": 8080}
}
```

### 3.2 任务状态机

所有任务共享同一状态机。脚本跑完就 completed，报错就 failed，cancel 就 cancelled：

```
+----------+     submit    +-----------+     done      +-----------+
| pending  | ------------> | running   | ------------> | completed |
+----------+               +-----------+               +-----------+
                                |
                                | error / cancel
                                v
                           +-----------+
                           | failed    |
                           +-----------+
```

客户端通过轮询 `GET /tasks/{id}` 获取状态。

### 3.3 数据入库

```
+-------------+     +------------------+     +------------------+
| Raw Data    |     | User Script      |     | Data Lake        |
| (MNIST zip) | --> | (on Daft/Ray)    | --> | (lance_storage/  |
|             |     |                  |     |  datasets/)      |
+-------------+     +------------------+     +------------------+
```

Server 接收任务请求，加载用户脚本并执行：

```python
# mnist/mnist_clean.py — 用户编写
def run(input_path: str, output_path: str, params: dict) -> dict:
    import daft
    from daft import col

    # 读取 IDX 格式的 MNIST 数据
    df = load_mnist_idx(input_path)

    # 归一化像素值到 [0, 1]
    if params.get("normalize"):
        df = df.with_column("image", col("image") / 255.0)

    # 验证标签范围
    df = df.where(col("label").between(0, 9))

    # 添加 train/test 拆分
    df = df.with_column("split", assign_split(col("index"), test_ratio=0.14))

    # 写入数据湖
    df.write_lance(output_path, mode="overwrite")

    return {"total_records": df.count_rows()}
```

MNIST 的具体清洗步骤：
1. 读取 IDX 格式的图像和标签文件
2. 将 28x28 图像展平为 784 维向量，归一化到 [0, 1]
3. 验证标签范围 [0, 9]
4. 添加 `split` 列（train/test）
5. 写入 Lance

### 3.4 模型训练

```
+------------------+     +------------------+     +------------------+
| Data Lake        |     | User Script      |     | Data Lake        |
| (datasets/)      | --> | (PyTorch on CPU) | --> | (models/)        |
+------------------+     +------------------+     +------------------+
```

```python
# mnist/mnist_cnn.py — 用户编写
def run(input_path: str, output_path: str, params: dict) -> dict:
    import daft
    from daft import col

    # 从数据湖读取训练数据
    df = daft.read_lance(input_path)
    train_data = df.where(col("split") == "train").to_pandas()
    test_data = df.where(col("split") == "test").to_pandas()

    # 训练模型（PyTorch）
    model = MnistCNN()
    train_model(model, train_data, params)
    metrics = evaluate_model(model, test_data)

    # 将模型权重 + 元数据写入数据湖
    save_model(model, metrics, params, output_path)

    return {"accuracy": metrics["accuracy"], "loss": metrics["loss"]}
```

### 3.5 推理服务

```
+------------------+     +------------------+     +------------------+
| Data Lake        |     | User Script      |     | API / Web        |
| (models/)        | --> | (FastAPI subprocess) | --> | (localhost:port) |
+------------------+     +------------------+     +------------------+
```

推理服务由用户脚本实现，平台只负责启动脚本。用户脚本从数据湖加载模型，启动 FastAPI 子服务：

```python
# mnist/mnist_serve.py — 用户编写
def run(input_path: str, output_path: str, params: dict) -> dict:
    import daft, torch, uvicorn
    from fastapi import FastAPI

    # 从数据湖读取模型权重
    df = daft.read_lance(input_path)
    pdf = df.to_pandas()
    weights_bytes = pdf["weights"].iloc[0]

    # 加载 PyTorch 模型
    model = MnistCNN()
    model.load_state_dict(torch.load(io.BytesIO(weights_bytes), weights_only=True))
    model.eval()

    # 启动 FastAPI 子服务
    app = FastAPI()

    @app.post("/predict")
    def predict(body: PredictRequest):
        tensor = torch.tensor(body.image).reshape(1, 1, 28, 28)
        with torch.no_grad():
            output = model(tensor)
        probs = torch.softmax(output, dim=1)
        return {
            "prediction": probs.argmax().item(),
            "confidence": probs.max().item(),
            "probabilities": probs[0].tolist(),
        }

    # 阻塞运行，直到进程被 kill
    uvicorn.run(app, host="0.0.0.0", port=params.get("port", 8080))
```

## 4. Server

数据湖存储 + 脚本执行器 + API 层，统一为一个服务。

### 4.1 组件职责

| 组件 | 职责 |
|------|------|
| **Lance** | 列式存储**格式**（类似 Parquet），针对 ML 优化：高压缩、支持大 BLOB、零拷贝 schema 演进 |
| **LanceDB** | 基于 Lance 格式的**数据库引擎**，提供建表、查询、向量搜索等管理 API |
| **Daft** | 计算引擎：可通过 `read_lance()` 直接读写 Lance 文件，也可通过 LanceDB API 操作 |
| **Ray** | 分布式运行时：Daft 的可选执行后端，也可直接用于训练任务 |

Lance 和 LanceDB 的关系类似 Parquet 和 DuckDB：

```python
# LanceDB — 数据库 API（建表、搜索）
db = lancedb.connect("./lance_storage")
db.create_table("products", data)
db.open_table("products").search(query_vec)

# Lance — Daft 直接读底层文件（绕过 LanceDB，用于大规模计算）
daft.read_lance("./lance_storage/products.lance")
```

Server 内部分三个模块：

| 模块 | 职责 |
|------|------|
| **Storage** (`storage.py`) | Lance 读写封装（列出、查看 schema、删除） |
| **Runner** (`runner.py` + `runners/`) | 脚本执行器基类 + 后端实现（`local.py` 线程执行，`ray.py` Ray Task 执行） |
| **App** (`app.py`) | FastAPI 路由（数据集、模型、任务 API），通过 `PLATFORM_LEVEL` 环境变量选择后端 |

### 4.2 RESTful API

```
/api/v1/
├── datasets/                    # 数据集管理
│   ├── GET    /                 # 列出数据湖中的数据集
│   ├── GET    /{id}             # 查看数据集详情和 schema
│   └── DELETE /{id}             # 删除数据集
│
├── models/                      # 模型管理
│   ├── GET    /                 # 列出数据湖中的模型
│   ├── GET    /{id}             # 查看模型详情（权重、指标、超参数）
│   └── DELETE /{id}             # 删除模型
│
└── tasks/                       # 计算任务
    ├── POST   /                 # 创建任务（执行用户脚本）
    ├── GET    /                 # 列出所有任务
    ├── GET    /{id}             # 查询任务状态
    └── POST   /{id}/cancel      # 取消任务
```

#### 任务字段

| 字段 | 说明 |
|------|------|
| `name` | 任务名称 |
| `input` | 输入路径 |
| `script` | 用户脚本路径 |
| `output` | 输出路径 |
| `params` | 用户自定义参数（如 port, device 等） |

#### 示例

**创建任务：**

```
POST /api/v1/tasks
{
    "name": "mnist_ingestion",
    "input": "/data/raw/mnist/",
    "script": "mnist/mnist_clean.py",
    "output": "lance_storage/datasets/mnist_clean.lance",
    "params": {"normalize": true, "flatten": true}
}
```

```
201 Created
{
    "id": "task-001",
    "name": "mnist_ingestion",
    "status": "running",
    "created_at": "2026-02-10T10:00:00Z"
}
```

**查询任务状态：**

```
GET /api/v1/tasks/task-001
```

```
200 OK
{
    "id": "task-001",
    "name": "mnist_ingestion",
    "status": "completed",
    "created_at": "2026-02-10T10:00:00Z",
    "completed_at": "2026-02-10T10:01:30Z",
    "result": {"total_records": 70000}
}
```

**列出数据集：**

```
GET /api/v1/datasets
```

```
200 OK
[
    {
        "id": "mnist_clean",
        "path": "lance_storage/datasets/mnist_clean.lance",
        "schema": {"image": "List[Float64]", "label": "Int64", "split": "String"},
        "num_rows": 70000
    }
]
```

### 4.3 数据湖目录结构

Lance 作为统一存储格式，数据集和模型都存在数据湖中：

```
lance_storage/
├── datasets/
│   ├── mnist_raw.lance          # Raw data (images + labels)
│   ├── mnist_clean.lance        # Cleaned data
│   └── mnist_features.lance     # Feature data (optional)
└── models/
    ├── mnist_cnn_v1.lance       # Model weights + metadata
    └── mnist_cnn_v2.lance       # Model versioning
```

### 4.4 错误处理

统一错误响应格式：

```json
{
    "error": {
        "code": "DATASET_NOT_FOUND",
        "message": "Dataset 'mnist_clean' does not exist"
    }
}
```

| HTTP Status | Code | 说明 |
|-------------|------|------|
| 400 | INVALID_REQUEST | 请求参数校验失败 |
| 404 | DATASET_NOT_FOUND | 数据集不存在 |
| 404 | MODEL_NOT_FOUND | 模型不存在 |
| 404 | TASK_NOT_FOUND | 任务不存在 |
| 500 | INTERNAL_ERROR | 内部错误 |

## 5. MNIST 实现规划

### 5.1 目录结构

```
demo4_ai_platform/
├── design.md                    # 本文档
├── README.md
├── requirements.txt
├── lance_storage/               # 数据湖根目录（共享存储，gitignored）
│   ├── datasets/
│   └── models/
├── server/                      # AI Platform 服务
│   ├── __init__.py
│   ├── app.py                   # FastAPI 主应用（API 路由）
│   ├── storage.py               # Lance 读写封装
│   ├── runner.py                # Runner 基类 + 工厂函数
│   └── runners/
│       ├── local.py             # Level 1: 线程执行
│       └── ray.py               # Level 2/3: Ray Task 执行
├── mnist/                       # 用户脚本 + Web Demo
│   ├── mnist_clean.py           # MNIST 清洗脚本
│   ├── mnist_cnn.py             # MNIST CNN 训练脚本
│   ├── mnist_serve.py           # MNIST 推理服务脚本
│   └── index.html               # 手写数字识别 Web Demo
└── tests/
    └── unit/
```

> `lance_storage/` 是运行时数据，应加入 `.gitignore`。

### 5.2 实现步骤

1. `server/storage.py` — Lance 读写封装（列出、查看 schema、删除）
2. `server/runner.py` — 脚本执行器（加载用户脚本、调用 `run()`、管理任务生命周期）
3. `server/app.py` — HTTP API（数据集、模型、任务路由）
4. `mnist/mnist_clean.py` — MNIST 清洗脚本（实现 `run()` 接口）
5. `mnist/mnist_cnn.py` — PyTorch CNN 训练脚本（实现 `run()` 接口）
6. `mnist/mnist_serve.py` — 推理服务脚本（实现 `run()` 接口，启动 FastAPI 子服务）
7. 单元测试

### 5.3 技术选型

| 需求 | 选型 | 理由 |
|------|------|------|
| 数据存储 | LanceDB + Lance | 统一存储结构化数据和二进制数据（模型权重） |
| 数据处理 | Daft | 支持 Lance 原生读写，lazy evaluation |
| 分布式运行时 | Ray | Daft 后端，也可用于分布式训练 |
| 模型训练 | PyTorch (CPU) | 轻量，MNIST 不需要 GPU |
| API 框架 | FastAPI | 异步支持好，自动生成 OpenAPI 文档 |

### 5.4 配置

默认值内置在 `server/app.py` 中：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `storage_path` | `./lance_storage` | 数据湖根目录 |

`create_app(storage_path=...)` 可在代码中覆盖。

Level 1 下任务状态存储在内存中。Level 2+ 切换到 Ray 后，任务状态由 Ray Tasks/Workflows 管理。

## 6. 部署级别

根据负载规模，系统有三个部署级别。Server API 层不变，底层三个维度逐步升级：

| 维度 | Level 1 | Level 2 | Level 3 |
|------|---------|---------|---------|
| **Storage** | Lance local files | Lance local files | Lance on S3/MinIO |
| **Compute** | User Script (Daft local) | Ray Task (Daft on Ray) | Ray Task on K8s |
| **Serving** | User Script (uvicorn) | Ray Serve | Ray Serve on K8s |

### 6.1 Level 1: 单机单任务

最简部署，适合开发调试和小规模数据（如 MNIST）。

```
+--------------------------------------------------------------------+
|  Single Machine                                                    |
|                                                                    |
|  +--------------------------------------------------------------+  |
|  |  Server (FastAPI, :8000)                                     |  |
|  |  +------------+  +-------------+  +-----------------------+  |  |
|  |  | GET /data  |  | GET /models |  | POST/GET /tasks       |  |  |
|  |  +------+-----+  +------+------+  +-----------+-----------+  |  |
|  |         |               |                     |              |  |
|  |         v               v                     v              |  |
|  |  +------------+  +------------+  +------------------------+  |  |
|  |  | Storage    |  | Storage    |  | TaskRunner             |  |  |
|  |  | (Lance R/W)|  | (Lance R/W)|  | (thread per task)      |  |  |
|  |  +------+-----+  +------+-----+  +-----+----------+-------+  |  |
|  |         |               |              |          |          |  |
|  +--------------------------------------------------------------+  |
|            |               |              |          |             |
|            v               v              v          v             |
|  +-----------------------------+  +------------+ +------------+    |
|  |  lance_storage/             |  | User       | | User       |    |
|  |  +----------+ +---------+   |  | Script     | | Script     |    |
|  |  | datasets/| | models/ |   |  | (Daft      | | (uvicorn   |    |
|  |  | *.lance  | | *.lance |   |  |  local)    | |  :8080)    |    |
|  |  +----------+ +---------+   |  |            | |            |    |
|  +-----------------------------+  +------------+ +------------+    |
|   Storage                          Compute        Serving          |
+--------------------------------------------------------------------+
```

- 任务串行执行，每个任务在独立线程中运行用户脚本
- 推理服务由用户脚本自行启动 FastAPI 子进程（如 :8080）
- 依赖：不需要 Ray，不需要 K8s

### 6.2 Level 2: 单机多任务

引入 Ray 作为本地运行时，支持并发任务和资源隔离。

```
+--------------------------------------------------------------------+
|  Single Machine                                                    |
|                                                                    |
|  +--------------------------------------------------------------+  |
|  |  Server (FastAPI, :8000)                                     |  |
|  |  +------------+  +-------------+  +-----------------------+  |  |
|  |  | GET /data  |  | GET /models |  | POST/GET /tasks       |  |  |
|  |  +------+-----+  +------+------+  +-----------+-----------+  |  |
|  |         |               |                     |              |  |
|  |         v               v                     v              |  |
|  |  +------------+  +------------+  +------------------------+  |  |
|  |  | Storage    |  | Storage    |  | TaskRunner             |  |  |
|  |  | (Lance R/W)|  | (Lance R/W)|  | (submit to Ray)        |  |  |
|  |  +------+-----+  +------+-----+  +-----+----------+-------+  |  |
|  |         |               |              |          |          |  |
|  +--------------------------------------------------------------+  |
|            |               |              |          |             |
|            v               v              v          v             |
|  +-----------------------------+  +------------------------+       |
|  |  lance_storage/             |  | Ray (local cluster)    |       |
|  |  +----------+ +---------+   |  | +----------+ +-------+ |       |
|  |  | datasets/| | models/ |   |  | | Ray Task | | Ray   | |       |
|  |  | *.lance  | | *.lance |   |  | | (Daft    | | Serve | |       |
|  |  +----------+ +---------+   |  | |  on Ray) | |       | |       |
|  +-----------------------------+  | +----------+ +-------+ |       |
|                                   +------------------------+       |
|   Storage                            Compute      Serving          |
+--------------------------------------------------------------------+
```

- Daft 切换 Ray runner（`set_runner_ray()`），多任务并发
- Ray Serve 管理模型副本，支持多模型同时服务
- 依赖：需要 Ray，不需要 K8s

**Level 1 → 2 的改动点：**

| 维度 | 改动 |
|------|------|
| Compute | User Script → Ray Task |
| Serving | User Script → Ray Serve |
| Storage | 不变 |

**TaskRunner 改动示例：**

Level 1 在线程中直接调用用户脚本：

```python
# Level 1: runner.py — 线程执行
def _execute(self, task_id, script, input_path, output_path, params):
    fn = _load_run_function(script)
    result = fn(input_path, output_path, params)
    self._tasks[task_id]["status"] = "completed"
    self._tasks[task_id]["result"] = result
```

Level 2 改为提交 Ray Task，Daft 计算也自动分布到 Ray 集群：

```python
# Level 2: runner.py — Ray Task 执行
@ray.remote
def _run_script(script: str, input_path: str, output_path: str, params: dict) -> dict:
    daft.context.set_runner_ray()
    fn = _load_run_function(script)
    return fn(input_path, output_path, params)

def _execute(self, task_id, script, input_path, output_path, params):
    ref = _run_script.remote(script, input_path, output_path, params)
    result = ray.get(ref)
    self._tasks[task_id]["status"] = "completed"
    self._tasks[task_id]["result"] = result
```

**流式处理示例：**

Level 1 下，数据入库和训练是两个独立任务，中间结果写 Lance 文件：

```
POST /tasks  {script: "mnist_clean.py", output: "mnist_clean.lance"}
# 等完成...
POST /tasks  {script: "mnist_cnn.py", input: "mnist_clean.lance", ...}
```

Level 2 下，用户可以编写端到端脚本，在同一个 Daft on Ray 执行图里完成清洗和训练，中间数据在内存中流转不落盘：

```python
# mnist/mnist_e2e.py — 端到端流式处理（Level 2）
def run(input_path: str, output_path: str, params: dict) -> dict:
    import daft
    from daft import col

    # 阶段 1: 数据清洗（CPU）
    # Daft lazy evaluation，此时只构建计算图，不执行
    df = load_mnist_idx(input_path)
    if params.get("normalize"):
        df = df.with_column("image", col("image") / 255.0)
    df = df.where(col("label").between(0, 9))

    # 阶段 2: 训练（GPU）
    # .to_pandas() 触发执行，数据在 Ray 集群内存中流转
    # 清洗结果不写 Lance，直接进训练
    train_data = df.where(col("split") == "train").to_pandas()
    test_data = df.where(col("split") == "test").to_pandas()

    model = MnistCNN()
    train_model(model, train_data, params)
    metrics = evaluate_model(model, test_data)

    # 只有最终结果写 Lance
    save_model(model, metrics, params, output_path)

    return {"accuracy": metrics["accuracy"], "loss": metrics["loss"]}
```

两种方式对比：

| | 分开两个 Task | 单个 e2e 脚本 |
|---|---|---|
| 中间数据 | 写 Lance 文件 | 内存中流转 |
| 调度 | 两次 HTTP 调用 | 一次 |
| 灵活性 | 可单独重跑清洗或训练 | 必须一起跑 |
| 适用场景 | 开发调试、数据复用 | 生产跑批、追求性能 |

平台 API 不变，流式编排由用户脚本自己实现。Daft on Ray 的优势（内存流式、异构调度）在用户脚本内部自然生效。

### 6.3 Level 3: 多机多任务

Ray 集群部署在 K8s 上，存储切换到共享对象存储。

```
+--------------------------------------------------------------------+
|  K8s Cluster                                                       |
|                                                                    |
|  +--------------------------------------------------------------+  |
|  |  Server Pod (FastAPI, :8000)                                 |  |
|  |  +------------+  +-------------+  +-----------------------+  |  |
|  |  | GET /data  |  | GET /models |  | POST/GET /tasks       |  |  |
|  |  +------+-----+  +------+------+  +-----------+-----------+  |  |
|  |         |               |                     |              |  |
|  |         v               v                     v              |  |
|  |  +------------+  +------------+  +------------------------+  |  |
|  |  | Storage    |  | Storage    |  | TaskRunner             |  |  |
|  |  | (Lance R/W)|  | (Lance R/W)|  | (Ray Workflows)        |  |  |
|  |  +------+-----+  +------+-----+  +-----+----------+-------+  |  |
|  |         |               |              |          |          |  |
|  +--------------------------------------------------------------+  |
|            |               |              |          |             |
|            v               v              v          v             |
|  +-----------------------------+  +------------------------+       |
|  |  S3 / MinIO (shared)        |  | Ray on K8s (KubeRay)   |       |
|  |  s3://bucket/lance_storage/ |  | +----------+ +-------+ |       |
|  |  +----------+ +---------+   |  | | Ray Task | | Ray   | |       |
|  |  | datasets/| | models/ |   |  | | (Daft    | | Serve | |       |
|  |  | *.lance  | | *.lance |   |  | |  on Ray) | | on K8s| |       |
|  |  +----------+ +---------+   |  | +----------+ +-------+ |       |
|  +-----------------------------+  +------------------------+       |
|   Storage                            Compute      Serving          |
+--------------------------------------------------------------------+
```

- Ray on K8s 自动扩缩容，多节点并行
- Ray Workflows 持久化任务状态，支持故障恢复
- Lance 文件在 S3/MinIO 上，多节点共享
- 依赖：需要 Ray、K8s、MinIO/S3

**Level 2 → 3 的改动点：**

| 维度 | 改动 |
|------|------|
| Compute | Ray Task → Ray Task on K8s |
| Serving | Ray Serve → Ray Serve on K8s |
| Storage | `./lance_storage/` → `s3://bucket/lance_storage/` |

### 6.4 Ray on K8s 的能力边界

Ray 的定位是**分布式计算运行时**，不是完整的任务编排平台。理解它做什么、不做什么，有助于在 Level 3 架构中合理分工。

#### 隔离性

Ray 的隔离是**进程级**的，不是容器级的：

- 每个 Worker 是独立进程，内存隔离
- 在 K8s 上，KubeRay 把每个 Worker 放在独立 Pod 里，跨节点有 Pod 级隔离
- 但同一个 Worker 进程可以跑多个 Task，这些 Task 之间没有隔离

Ray 不做安全沙箱级的任务隔离。设计假设是**单信任域**——同一个团队的协作任务，不是多租户平台。如果需要多租户隔离，做法是每个租户跑独立的 Ray 集群。

#### 资源声明

CPU/GPU/内存是一等资源，可以在 `@ray.remote` 上声明：

```python
@ray.remote(num_cpus=4, num_gpus=1, memory=8 * 1024**3)
def train_model(input_path, output_path, params):
    ...
```

- 还支持自定义资源（如 `resources={"TPU": 1}`）
- KubeRay 会把 Ray 的资源请求映射到 K8s Pod 的 resource requests
- Ray Autoscaler 在资源不足时自动向 K8s 申请新 Pod

**网络带宽不是一等资源。** 网络带宽难以精确度量和独占分配，硬隔离需要依赖底层（K8s NetworkPolicy、CNI 插件），Ray 选择不管这层。

#### 生命周期管理

| 能力 | Ray Tasks | Ray Workflows |
|------|-----------|---------------|
| 提交 | `task.remote()` | `workflow.run()` |
| 状态查询 | `ray.get()` 阻塞等待 | `workflow.get_status()` |
| 重试 | `@ray.remote(max_retries=3)` | step 级别重试 |
| 取消 | `ray.cancel(ref)` 协作式 | `workflow.cancel()` |
| 故障恢复 | 无（内存丢了就丢了） | 从 checkpoint 恢复 |
| 挂起/继续 | 不支持 | 不支持 |

**挂起/继续不做。** 暂停一个分布式计算需要序列化任意 Python 运行时状态（调用栈、锁、网络连接、GPU 上下文），这是应用特定的，通用方案做不了。如果需要"暂停"，只能靠应用自己 checkpoint 然后重新提交。

**取消是协作式的。** `ray.cancel()` 发 SIGTERM，Task 需要自己处理信号退出。不能强杀，因为强杀可能导致 GPU 显存泄漏、文件写一半等问题。

#### 职责分工

| 关注点 | 谁负责 |
|--------|--------|
| 容器隔离、网络策略 | K8s |
| 计算调度、资源分配 | Ray |
| 复杂 DAG 编排、定时调度 | Airflow / Prefect |
| 持久化任务队列 | 外部消息队列 |

Ray 不是要替代 K8s，而是在 K8s 之上提供一层对 ML 友好的计算抽象——用户不用关心 Pod 怎么调度，只需要声明"我要 4 CPU + 1 GPU"，Ray + KubeRay 搞定剩下的。

### 6.5 多租户与资源分配

Level 3 部署在 K8s 上后，自然会面临多租户问题。Ray 本身不做多租户，隔离和资源管理交给 K8s。

#### 隔离方案

| 方案 | 隔离强度 | 资源利用率 | 适用场景 |
|------|----------|-----------|----------|
| 共享 Ray 集群 + 逻辑隔离 | 弱（进程级） | 高 | 内部团队 |
| 每租户独立 Ray 集群 | 中（Pod + Namespace） | 中 | 企业多部门 / SaaS |
| 每租户独立 K8s 集群 | 强（集群级） | 低 | 强合规（金融、医疗） |

推荐方案是**每租户独立 Ray 集群**：

```
K8s Cluster
├── Namespace: tenant-a
│   ├── RayCluster A (KubeRay CR)
│   ├── Server Pod A
│   └── NetworkPolicy (deny cross-namespace)
├── Namespace: tenant-b
│   ├── RayCluster B (KubeRay CR)
│   ├── Server Pod B
│   └── NetworkPolicy (deny cross-namespace)
└── S3 (IAM per tenant)
```

隔离靠 K8s 原生能力（Namespace、NetworkPolicy、IAM），Ray 不需要感知多租户。KubeRay 天然支持一个 K8s 集群里跑多个 RayCluster CR，运维上就是多一份 YAML。空闲时 Autoscaler 可以缩容到 0，节省资源。

#### 资源分配策略

| 策略 | 做法 | 适用场景 |
|------|------|----------|
| 静态配额 | K8s ResourceQuota 写死上限 | 负载稳定 |
| Guaranteed + Burstable | requests 保底 + limits 上限 | 负载波动大（推荐） |
| Preemption | K8s PriorityClass 高优先级驱逐低优先级 | 资源紧张且有明确优先级 |

推荐 **Guaranteed + Burstable**——每个租户有保底资源（requests），超出部分按集群余量弹性分配：

| 租户 | 保底 (requests) | 上限 (limits) |
|------|----------------|---------------|
| A | 8 CPU, 1 GPU | 32 CPU, 4 GPU |
| B | 4 CPU, 0 GPU | 16 CPU, 2 GPU |

#### 抢占

K8s PriorityClass 支持跨租户抢占（高优先级 Pod 驱逐低优先级 Pod），但 Ray 自身不感知抢占——K8s 杀 Pod 后 Ray 只看到 Worker 丢了，触发 fault tolerance。这意味着被抢占的训练任务如果没做 checkpoint，进度全丢。所以抢占场景下，训练脚本必须自己做 epoch 级 checkpoint。

> 多租户是 Level 3 之上的进阶话题，本项目不涉及实现。

## 参考文献

### 核心组件

- [Daft 官方文档](https://www.getdaft.io/projects/docs/en/stable/) — 分布式 DataFrame 框架
- [Ray 官方文档](https://docs.ray.io/en/latest/) — 分布式计算运行时
- [LanceDB 官方文档](https://lancedb.github.io/lancedb/) — 嵌入式向量数据库
- [Lance 格式规范](https://lancedb.github.io/lance/) — 列式存储格式

### 架构参考

- [火山引擎：多模态数据湖 Daft + Lance 架构](https://developer.volcengine.com/articles/7551415232963608602) — LAS（LakeHouse AI Service）设计理念
- [LanceDB: AI Data Infrastructure](https://blog.lancedb.com/) — LanceDB 团队博客，涵盖向量搜索、多模态存储等主题

### AI Platform 设计

- [FastAPI 官方文档](https://fastapi.tiangolo.com/) — 异步 Web 框架
- [PyTorch 官方教程](https://pytorch.org/tutorials/) — 深度学习框架
- [Ray Serve 文档](https://docs.ray.io/en/latest/serve/) — 可扩展的模型推理服务
- [Ray Train 文档](https://docs.ray.io/en/latest/train/) — 分布式训练

### 进阶阅读

- [Designing Data-Intensive Applications](https://dataintensive.net/) — Martin Kleppmann，数据系统设计经典
- [MLOps: Machine Learning Operations](https://ml-ops.org/) — MLOps 实践和模式
- [MinIO 文档](https://min.io/docs/minio/linux/index.html) — 本地部署的 S3 兼容对象存储
