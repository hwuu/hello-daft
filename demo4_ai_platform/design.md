# Demo 4: AI Platform 设计文档

## 目录

- [1. 背景与目标](#1-背景与目标)
  - [1.1 问题](#11-问题)
  - [1.2 火山引擎的启发](#12-火山引擎的启发)
  - [1.3 设计目标](#13-设计目标)
- [2. 架构总览](#2-架构总览)
  - [2.1 核心分层](#21-核心分层)
  - [2.2 部署级别](#22-部署级别)
  - [2.3 可定制性设计](#23-可定制性设计)
- [3. Server](#3-server)
  - [3.1 组件职责](#31-组件职责)
  - [3.2 RESTful API](#32-restful-api)
  - [3.3 数据湖目录结构](#33-数据湖目录结构)
  - [3.4 任务状态机](#34-任务状态机)
  - [3.5 错误处理](#35-错误处理)
- [4. 解耦设计](#4-解耦设计)
  - [4.1 脚本与平台的解耦](#41-脚本与平台的解耦)
  - [4.2 可替换性](#42-可替换性)
- [5. 核心任务流](#5-核心任务流)
  - [5.1 数据入库](#51-数据入库)
  - [5.2 模型训练](#52-模型训练)
  - [5.3 推理服务](#53-推理服务)
- [6. MNIST 实现规划](#6-mnist-实现规划)
  - [6.1 目录结构](#61-目录结构)
  - [6.2 实现步骤](#62-实现步骤)
  - [6.3 技术选型](#63-技术选型)
  - [6.4 配置](#64-配置)
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

## 2. 架构总览

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

用户通过 Server 的 RESTful API 操作全部功能。任务是高度可定制的——用户提供自己的脚本，平台只负责调度和存储（详见 [2.3 可定制性设计](#23-可定制性设计)）。

### 2.2 部署级别

根据负载规模，系统有三个部署级别。架构不变，只是底层运行时逐步升级：

#### Level 1: 单机单任务

最简部署，适合开发调试和小规模数据（如 MNIST）。

```
+----------------------------+
|  Single Machine            |
|                            |
|  Server (FastAPI, :8000)   |
|  +----------------------+  |
|  | Task: sequential     |  |
|  | Daft: local runner   |  |
|  | Lance: local files   |  |
|  | Serving: user script |  |
|  +----------------------+  |
+----------------------------+
```

- **计算**：Daft 本地执行（默认 runner），任务串行
- **存储**：Lance 文件在本地磁盘
- **推理**：用户脚本启动 FastAPI 子服务
- **任务调度**：内存中的状态机，同一时间只运行一个任务
- **依赖**：不需要 Ray，不需要 K8s

#### Level 2: 单机多任务

引入 Ray 作为本地运行时，支持并发任务和资源隔离。

```
+-----------------------------------------+
|  Single Machine                         |
|                                         |
|  Server (FastAPI, :8000)                |
|  +-----------------------------------+  |
|  |  Ray (local cluster)              |  |
|  |  +--------+ +--------+ +-------+  |  |
|  |  | Task 1 | | Task 2 | | Serve |  |  |
|  |  | (Daft) | | (Train)| | (Ray  |  |  |
|  |  |        | |        | | Serve)|  |  |
|  |  +--------+ +--------+ +-------+  |  |
|  +-----------------------------------+  |
|  Lance: local files                     |
+-----------------------------------------+
```

- **计算**：Daft 切换 Ray runner（`set_runner_ray()`），多任务并发
- **存储**：Lance 文件仍在本地磁盘
- **推理**：Ray Serve 管理模型副本，支持多模型同时服务
- **任务调度**：Ray Tasks/Actors 提供并发和资源隔离
- **依赖**：需要 Ray，不需要 K8s

**Level 1 → 2 的改动点：**

| 组件 | 改动 |
|------|------|
| Daft | `daft.context.set_runner_ray()` |
| 任务调度 | 内存状态机 → Ray Tasks |
| 推理 | FastAPI 单进程 → Ray Serve |
| 代码改动量 | 小：切换 runner + 任务提交方式 |

#### Level 3: 多机多任务

Ray 集群部署在 K8s 上，存储切换到共享对象存储。

```
+------------------+     +------------------+
|  K8s Node 1      |     |  K8s Node 2      |
|  +-------------+ |     |  +-------------+ |
|  | Ray Worker  | |     |  | Ray Worker  | |
|  | (Daft tasks)| |     |  | (Training)  | |
|  +-------------+ |     |  +-------------+ |
|  +-------------+ |     |  +-------------+ |
|  | Ray Serve   | |     |  | Ray Serve   | |
|  | (Inference) | |     |  | (Inference) | |
|  +-------------+ |     |  +-------------+ |
+------------------+     +------------------+
         |                        |
         v                        v
+-------------------------------------------+
|  Shared Storage (MinIO / S3)              |
|  Lance files: s3://bucket/lance_storage/  |
+-------------------------------------------+
```

- **计算**：Ray on K8s，自动扩缩容
- **存储**：Lance 文件在 S3/MinIO，多节点共享
- **推理**：Ray Serve 多副本，K8s 负载均衡
- **任务调度**：Ray Workflows 持久化任务状态
- **依赖**：需要 Ray、K8s、MinIO/S3

**Level 2 → 3 的改动点：**

| 组件 | 改动 |
|------|------|
| Ray | 本地集群 → K8s KubeRay Operator |
| 存储路径 | `./lance_storage/` → `s3://bucket/lance_storage/` |
| 任务调度 | Ray Tasks → Ray Workflows（持久化） |
| 推理 | Ray Serve 本地 → Ray Serve on K8s |
| 代码改动量 | 小：改存储路径 + 部署配置 |

### 2.3 可定制性设计

平台不绑定特定数据集或模型。所有任务采用统一的**脚本模式**：平台负责调度和存储，用户脚本负责业务逻辑。

**数据入库 — 用户提供清洗脚本：**

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

清洗脚本需要遵循接口约定：

```python
# mnist/mnist_clean.py
def run(input_path: str, output_path: str, params: dict) -> dict:
    """
    Args:
        input_path: 原始数据路径
        output_path: 输出 Lance 路径
        params: 用户自定义参数
    Returns:
        stats: {"total_records": 70000, ...}
    """
```

**模型训练 — 用户提供训练脚本：**

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

训练脚本需要遵循接口约定：

```python
# mnist/mnist_cnn.py
def run(input_path: str, output_path: str, params: dict) -> dict:
    """
    Args:
        input_path: Lance 数据集路径
        output_path: 模型输出路径
        params: 用户定义的超参数
    Returns:
        metrics: {"accuracy": 0.98, "loss": 0.03, ...}
    """
```

**推理服务 — 用户提供推理脚本：**

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

推理脚本需要遵循接口约定：

```python
# mnist/mnist_serve.py
def run(input_path: str, output_path: str, params: dict) -> dict:
    """
    Args:
        input_path: Lance 模型文件路径
        output_path: 未使用
        params: 服务参数（port, device 等）
    Returns:
        不会正常返回（阻塞直到进程终止）
    """
```

所有任务的接口完全一致：`run(input_path, output_path, params) → dict`。平台不区分批处理和服务，不关心脚本内部做什么。

## 3. Server

数据湖存储 + 脚本执行器 + API 层，统一为一个服务。

### 3.1 组件职责

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
| **TaskRunner** (`runner.py`) | 脚本执行器（加载用户脚本、调用 `run()`、管理任务生命周期） |
| **App** (`app.py`) | FastAPI 路由（数据集、模型、任务 API） |

### 3.2 RESTful API

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

### 3.3 数据湖目录结构

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

### 3.4 任务状态机

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

### 3.5 错误处理

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

## 4. 解耦设计

平台与用户脚本通过 **Lance 格式 + run() 接口**解耦。

### 4.1 脚本与平台的解耦

平台不关心业务逻辑，只提供基础设施：

| 平台提供的 | 用户脚本负责的 |
|---|---|
| 数据读写（Lance） | 清洗逻辑、特征工程 |
| 计算资源（Daft/Ray） | 训练代码和超参数 |
| 存储管理 | 模型版本管理策略 |
| 任务调度和生命周期 | 推理服务和 API 设计 |

用户脚本只需实现 `run(input_path, output_path, params) → dict`，平台负责调用和管理。

### 4.2 可替换性

- **替换存储**（如 Lance → Parquet + Milvus）：改 Storage 模块即可，用户脚本不受影响
- **替换计算**（如 Daft → Spark）：改 TaskRunner 即可，只要脚本接口不变
- **替换训练框架**（如 PyTorch → TensorFlow）：平台不受影响，用户脚本自行选择

## 5. 核心任务流

### 5.1 数据入库

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

### 5.2 模型训练

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

### 5.3 推理服务

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

## 6. MNIST 实现规划

### 6.1 目录结构

```
demo4_ai_platform/
├── design.md                    # 本文档
├── README.md
├── requirements.txt
├── config.yaml                  # 配置文件（存储路径等）
├── lance_storage/               # 数据湖根目录（共享存储，gitignored）
│   ├── datasets/
│   └── models/
├── server/                      # AI Platform 服务
│   ├── __init__.py
│   ├── app.py                   # FastAPI 主应用（API 路由）
│   ├── storage.py               # Lance 读写封装
│   └── runner.py                # 脚本执行器（加载脚本、调用 run()）
├── mnist/                       # 用户脚本 + Web Demo
│   ├── mnist_clean.py           # MNIST 清洗脚本
│   ├── mnist_cnn.py             # MNIST CNN 训练脚本
│   ├── mnist_serve.py           # MNIST 推理服务脚本
│   └── index.html               # 手写数字识别 Web Demo
└── tests/
    └── unit/
```

> `lance_storage/` 是运行时数据，应加入 `.gitignore`。

### 6.2 实现步骤

1. `server/storage.py` — Lance 读写封装（列出、查看 schema、删除）
2. `server/runner.py` — 脚本执行器（加载用户脚本、调用 `run()`、管理任务生命周期）
3. `server/app.py` — HTTP API（数据集、模型、任务路由）
4. `mnist/mnist_clean.py` — MNIST 清洗脚本（实现 `run()` 接口）
5. `mnist/mnist_cnn.py` — PyTorch CNN 训练脚本（实现 `run()` 接口）
6. `mnist/mnist_serve.py` — 推理服务脚本（实现 `run()` 接口，启动 FastAPI 子服务）
7. `config.yaml` — 配置文件
8. 单元测试

### 6.3 技术选型

| 需求 | 选型 | 理由 |
|------|------|------|
| 数据存储 | LanceDB + Lance | 统一存储结构化数据和二进制数据（模型权重） |
| 数据处理 | Daft | 支持 Lance 原生读写，lazy evaluation |
| 分布式运行时 | Ray | Daft 后端，也可用于分布式训练 |
| 模型训练 | PyTorch (CPU) | 轻量，MNIST 不需要 GPU |
| API 框架 | FastAPI | 异步支持好，自动生成 OpenAPI 文档 |

### 6.4 配置

```yaml
# config.yaml
server:
  host: "http://localhost:8000"    # 服务地址
  storage_path: "./lance_storage"  # 数据湖根目录
```

Level 1 下任务状态存储在内存中。Level 2+ 切换到 Ray 后，任务状态由 Ray Tasks/Workflows 管理。

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
