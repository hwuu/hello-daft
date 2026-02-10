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
- [3. Data Platform](#3-data-platform)
  - [3.1 组件职责](#31-组件职责)
  - [3.2 数据湖目录结构](#32-数据湖目录结构)
  - [3.3 对外接口](#33-对外接口)
- [4. AI Platform](#4-ai-platform)
  - [4.1 模块职责](#41-模块职责)
  - [4.2 RESTful API](#42-restful-api)
  - [4.3 任务状态机](#43-任务状态机)
  - [4.4 错误处理](#44-错误处理)
- [5. 解耦设计](#5-解耦设计)
  - [5.1 交互方式](#51-交互方式)
  - [5.2 各平台的边界](#52-各平台的边界)
  - [5.3 可替换性](#53-可替换性)
- [6. 核心任务流](#6-核心任务流)
  - [6.1 数据入库](#61-数据入库)
  - [6.2 模型训练](#62-模型训练)
  - [6.3 推理服务](#63-推理服务)
- [7. MNIST 实现规划](#7-mnist-实现规划)
  - [7.1 目录结构](#71-目录结构)
  - [7.2 实现步骤](#72-实现步骤)
  - [7.3 技术选型](#73-技术选型)
  - [7.4 火山引擎理念在 MNIST 中的体现](#74-火山引擎理念在-mnist-中的体现)
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

将系统拆分为两个解耦的平台：

- **Data Platform**（数据平台）：基于 Daft + Ray + LanceDB，提供存储和计算原语
- **AI Platform**（AI 平台）：基于 Data Platform，提供 RESTful API 管理数据入库、模型训练、推理服务

两者通过 **Lance 格式**作为契约层连接，互不依赖对方的实现细节。

## 2. 架构总览

### 2.1 核心分层

```
+-------------------------------------------------------------------+
|                        AI Platform                                |
|  (RESTful API: FastAPI)                                           |
|  +------------------+  +------------------+  +-----------------+  |
|  | Ingestion Task   |  | Training Task    |  | Inference       |  |
|  | Manager          |  | Manager          |  | Service         |  |
|  +--------+---------+  +--------+---------+  +--------+--------+  |
|           |                      |                     |          |
+-----------+----------------------+---------------------+----------+
            |                      |                     |
            v                      v                     v
+-------------------------------------------------------------------+
|                       Data Platform                               |
|  +------------------+  +---------------------+  +---------------+ |
|  | Daft             |  | Ray                 |  | LanceDB/Lance | |
|  | (Compute Engine) |  | (Optional Runtime)  |  | (Storage)     | |
|  +------------------+  +---------------------+  +---------------+ |
+-------------------------------------------------------------------+
```

用户通过 AI Platform 的 RESTful API 操作全部功能，不直接接触 Data Platform。

任务（Ingestion Task、Training Task）是高度可定制的：
- **Ingestion Task** 通过可插拔的 pipeline 定义清洗逻辑，不同数据集使用不同的 pipeline
- **Training Task** 通过指定训练脚本路径来定制训练逻辑，平台不限定模型架构和训练框架

### 2.2 部署级别

根据负载规模，系统有三个部署级别。架构分层不变，只是底层实现逐步升级：

#### Level 1: 单机单任务

最简部署，适合开发调试和小规模数据（如 MNIST）。

```
+----------------------------+
|  Single Machine            |
|                            |
|  FastAPI (1 process)       |
|  +----------------------+  |
|  | Task: sequential     |  |
|  | Daft: local runner   |  |
|  | Lance: local files   |  |
|  | Serving: in-process  |  |
|  +----------------------+  |
+----------------------------+
```

- **计算**：Daft 本地执行（默认 runner），任务串行
- **存储**：Lance 文件在本地磁盘
- **推理**：FastAPI 单进程，模型加载在同一进程内
- **任务调度**：内存中的状态机，同一时间只运行一个任务
- **依赖**：不需要 Ray，不需要 K8s

#### Level 2: 单机多任务

引入 Ray 作为本地运行时，支持并发任务和资源隔离。

```
+-----------------------------------------+
|  Single Machine                         |
|                                         |
|  FastAPI (1 process)                    |
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

平台不绑定特定数据集或模型。Ingestion 和 Training 采用统一的**脚本模式**：平台负责调度和存储，用户脚本负责业务逻辑。

**Ingestion Task — 用户提供清洗脚本：**

```
POST /api/v1/ingestion-tasks
{
    "name": "mnist_ingestion",
    "source_path": "/data/raw/mnist/",
    "script": "pipelines/mnist_clean.py",
    "params": {"normalize": true, "flatten": true},
    "output_dataset": "mnist_clean"
}
```

清洗脚本需要遵循接口约定：

```python
# pipelines/mnist_clean.py
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

**Training Task — 用户提供训练脚本：**

```
POST /api/v1/training-tasks
{
    "name": "mnist_cnn_v1",
    "dataset": "mnist_clean",
    "script": "training/mnist_cnn.py",
    "params": {"epochs": 10, "learning_rate": 0.001, "batch_size": 64},
    "output_model": "mnist_cnn_v1"
}
```

训练脚本需要遵循接口约定：

```python
# training/mnist_cnn.py
def run(data_path: str, output_path: str, params: dict) -> dict:
    """
    Args:
        data_path: Lance 数据集路径
        output_path: 模型输出路径
        params: 用户定义的超参数
    Returns:
        metrics: {"accuracy": 0.98, "loss": 0.03, ...}
    """
```

两种任务的接口完全一致：`run(input, output, params) → stats/metrics`。

## 3. Data Platform

底层基础设施，提供存储和计算原语。

### 3.1 组件职责

| 组件 | 职责 |
|------|------|
| **LanceDB / Lance** | 数据湖存储：原始数据、清洗数据、特征、模型文件、向量索引 |
| **Daft** | 计算引擎：数据读取、清洗、转换、嵌入生成 |
| **Ray** | 分布式运行时：Daft 的执行后端，也可直接用于训练任务 |

Data Platform 不关心业务逻辑，只提供：
- 读写 Lance 格式数据
- 执行 Daft DataFrame 计算
- 分配 Ray 计算资源

### 3.2 数据湖目录结构

LanceDB 作为统一存储层，所有数据（原始数据、清洗数据、模型权重、任务记录）都存在数据湖中：

```
lance_storage/
├── datasets/
│   ├── mnist_raw.lance          # Raw data (images + labels)
│   ├── mnist_clean.lance        # Cleaned data
│   └── mnist_features.lance     # Feature data (optional)
├── models/
│   ├── mnist_cnn_v1.lance       # Model weights + metadata
│   └── mnist_cnn_v2.lance       # Model versioning
└── metadata/
    ├── tasks.lance              # Task execution records
    └── metrics.lance            # Training metrics
```

### 3.3 对外接口

Data Platform 作为独立微服务，通过 HTTP API 对外提供能力。数据本身不走 HTTP，而是通过共享存储（Lance 文件）交换：

```
AI Platform                     Data Platform
     |                               |
     |  POST /compute/submit         |
     |  {"script": "clean.py",       |
     |   "input": "s3://.../raw",    |
     |   "output": "s3://.../clean"} |
     +------------------------------>|
     |                               |  Daft on Ray 执行
     |  200 {"task_id": "..."}       |  读 Lance → 处理 → 写 Lance
     |<------------------------------+
     |                               |
     |  两边都能直接读 Lance 文件     |
     |  (共享存储，不经过 HTTP)       |
```

#### 存储 API

```
GET    /api/v1/storage/datasets              # 列出数据湖中的数据集
GET    /api/v1/storage/datasets/{id}         # 查看数据集详情和 schema
DELETE /api/v1/storage/datasets/{id}         # 删除数据集
```

#### 计算 API

```
POST   /api/v1/compute/submit               # 提交计算任务（执行用户脚本）
GET    /api/v1/compute/{id}                  # 查询任务状态
POST   /api/v1/compute/{id}/cancel           # 取消任务
```

## 4. AI Platform

上层业务编排，通过 RESTful API 管理 AI 工作流。

### 4.1 模块职责

| 模块 | 职责 |
|------|------|
| **Ingestion Task Manager** | 定义和执行数据入库任务（数据源 → 用户脚本 → 数据湖） |
| **Training Task Manager** | 定义和执行训练任务（数据湖 → 用户脚本 → 模型 → 数据湖） |
| **Inference Service** | 加载模型，提供推理 API |

### 4.2 RESTful API

#### 资源模型

```
/api/v1/
├── datasets/                    # 数据集管理
│   ├── GET    /                 # 列出所有数据集
│   ├── POST   /                 # 创建数据集（元数据）
│   ├── GET    /{id}             # 获取数据集详情
│   └── DELETE /{id}             # 删除数据集
│
├── ingestion-tasks/             # 数据入库任务
│   ├── GET    /                 # 列出所有任务
│   ├── POST   /                 # 创建并提交任务
│   ├── GET    /{id}             # 获取任务状态和详情
│   └── POST   /{id}/cancel      # 取消运行中的任务
│
├── models/                      # 模型管理
│   ├── GET    /                 # 列出所有模型
│   ├── GET    /{id}             # 获取模型详情（含指标）
│   └── DELETE /{id}             # 删除模型
│
├── training-tasks/              # 训练任务
│   ├── GET    /                 # 列出所有任务
│   ├── POST   /                 # 创建并提交任务
│   ├── GET    /{id}             # 获取任务状态和详情
│   └── POST   /{id}/cancel      # 取消运行中的任务
│
└── inference-services/          # 推理服务
    ├── GET    /                 # 列出所有服务
    ├── POST   /                 # 创建并启动服务
    ├── GET    /{id}             # 获取服务状态
    ├── DELETE /{id}             # 停止并删除服务
    └── POST   /{id}/predict     # 调用推理
```

#### 用户使用流程（MNIST 示例）

**Step 1: 创建数据入库任务**

```
POST /api/v1/ingestion-tasks
{
    "name": "mnist_ingestion",
    "source_path": "/data/raw/mnist/",
    "script": "pipelines/mnist_clean.py",
    "params": {"normalize": true, "flatten": true},
    "output_dataset": "mnist_clean"
}
```

```
201 Created
{
    "id": "ingest-001",
    "name": "mnist_ingestion",
    "status": "running",
    "created_at": "2026-02-10T10:00:00Z",
    "output_dataset": "mnist_clean"
}
```

**Step 2: 查看任务状态**

```
GET /api/v1/ingestion-tasks/ingest-001
```

```
200 OK
{
    "id": "ingest-001",
    "name": "mnist_ingestion",
    "status": "completed",
    "created_at": "2026-02-10T10:00:00Z",
    "completed_at": "2026-02-10T10:01:30Z",
    "output_dataset": "mnist_clean",
    "stats": {
        "total_records": 70000,
        "train_records": 60000,
        "test_records": 10000
    }
}
```

**Step 3: 查看数据集**

```
GET /api/v1/datasets/mnist_clean
```

```
200 OK
{
    "id": "mnist_clean",
    "created_at": "2026-02-10T10:01:30Z",
    "source_task": "ingest-001",
    "storage_path": "lance_storage/datasets/mnist_clean.lance",
    "schema": {
        "image": "binary",
        "label": "int64",
        "split": "string"
    },
    "stats": {
        "total_records": 70000,
        "size_bytes": 52428800
    }
}
```

**Step 4: 创建训练任务**

```
POST /api/v1/training-tasks
{
    "name": "mnist_cnn_v1",
    "dataset": "mnist_clean",
    "script": "training/mnist_cnn.py",
    "params": {
        "epochs": 10,
        "learning_rate": 0.001,
        "batch_size": 64
    },
    "device": "cpu",
    "output_model": "mnist_cnn_v1"
}
```

```
201 Created
{
    "id": "train-001",
    "name": "mnist_cnn_v1",
    "status": "running",
    "created_at": "2026-02-10T10:05:00Z"
}
```

**Step 5: 查看训练进度**

```
GET /api/v1/training-tasks/train-001
```

```
200 OK
{
    "id": "train-001",
    "name": "mnist_cnn_v1",
    "status": "completed",
    "created_at": "2026-02-10T10:05:00Z",
    "completed_at": "2026-02-10T10:12:00Z",
    "output_model": "mnist_cnn_v1",
    "metrics": {
        "train_loss": 0.032,
        "test_accuracy": 0.987
    }
}
```

**Step 6: 启动推理服务**

```
POST /api/v1/inference-services
{
    "name": "mnist_predictor",
    "model": "mnist_cnn_v1",
    "device": "cpu",
    "port": 8080
}
```

```
201 Created
{
    "id": "serve-001",
    "name": "mnist_predictor",
    "status": "running",
    "endpoint": "http://localhost:8080",
    "created_at": "2026-02-10T10:15:00Z"
}
```

**Step 7: 调用推理**

```
POST /api/v1/inference-services/serve-001/predict
Content-Type: application/json
{
    "image": "<base64 encoded image>"
}
```

```
200 OK
{
    "prediction": 7,
    "confidence": 0.983,
    "probabilities": [0.001, 0.002, 0.003, 0.001, 0.002, 0.001, 0.003, 0.983, 0.002, 0.002]
}
```

### 4.3 任务状态机

所有异步任务（ingestion-tasks、training-tasks）共享同一状态机：

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

客户端通过轮询 `GET /{id}` 获取状态。

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
| 409 | TASK_ALREADY_RUNNING | 同名任务正在运行 |
| 409 | SERVICE_ALREADY_EXISTS | 同名服务已存在 |
| 500 | INTERNAL_ERROR | 内部错误 |

## 5. 解耦设计

两个平台通过 **Lance 格式**解耦，Lance 是唯一的数据交换格式。

### 5.1 交互方式

```
AI Platform                     Data Platform
     |                               |
     |  POST /compute/submit         |
     |  {"script": "clean.py",       |
     |   "input": "s3://.../raw",    |
     |   "output": "s3://.../clean"} |
     +------------------------------>|  执行用户脚本（Daft on Ray）
     |                               |
     |  GET /compute/{id}            |
     |<------------------------------+  {"status": "completed"}
     |                               |
     |  两边都能直接读 Lance 文件     |
     |  (共享存储，不经过 HTTP)       |
```

AI Platform 只看到 Lance 文件路径，不关心底层是本地磁盘还是 S3。

### 5.2 各平台的边界

**AI Platform 不依赖 Data Platform 的实现细节：**

| AI Platform 看到的 | Data Platform 内部实现 |
|---|---|
| `POST /compute/submit` | Daft on Ray 执行用户脚本 |
| `GET /storage/datasets` | 扫描 Lance 文件目录 |
| Lance 文件路径（共享存储） | 本地 / S3 / MinIO |

**Data Platform 不关心 AI 业务逻辑：**

| Data Platform 提供的 | AI Platform 自行处理的 |
|---|---|
| 数据读写 | 清洗 pipeline 定义 |
| 分布式计算资源 | 训练代码和超参数 |
| 存储管理 | 模型版本管理策略 |
| 向量索引和搜索 | 推理服务和 API 设计 |

### 5.3 可替换性

- **替换 Data Platform 存储**（如 Lance → Parquet + Milvus）：AI Platform 不受影响，Data Platform 内部改实现即可
- **替换 Data Platform 计算**（如 Daft → Spark）：AI Platform 代码不变，只要 Data Platform API 不变
- **替换 AI Platform 训练框架**（如 PyTorch → TensorFlow）：Data Platform 不受影响

## 6. 核心任务流

### 6.1 数据入库

```
+-------------+     +------------------+     +------------------+
| Raw Data    |     | User Script      |     | LanceDB          |
| (MNIST zip) | --> | (on Daft/Ray)    | --> | (lance_storage/  |
|             |     |                  |     |  datasets/)      |
+-------------+     +------------------+     +------------------+
```

AI Platform 提交任务，Data Platform 执行用户脚本：

```python
# pipelines/mnist_clean.py — 用户编写
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

### 6.2 模型训练

```
+------------------+     +------------------+     +------------------+
| LanceDB          |     | User Script      |     | LanceDB          |
| (datasets/)      | --> | (PyTorch on CPU) | --> | (models/)        |
+------------------+     +------------------+     +------------------+
```

AI Platform 提交任务，Data Platform 执行用户脚本：

```python
# training/mnist_cnn.py — 用户编写
def run(data_path: str, output_path: str, params: dict) -> dict:
    import daft
    from daft import col

    # 从数据湖读取训练数据
    df = daft.read_lance(data_path)
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

### 6.3 推理服务

```
+------------------+     +------------------+     +------------------+
| LanceDB          |     | Model Loading    |     | API / Web        |
| (models/)        | --> | (PyTorch)        | --> | (FastAPI)        |
+------------------+     +------------------+     +------------------+
```

从数据湖加载模型，启动 FastAPI 服务：

```python
# 从数据湖加载模型
table = db.open_table(model_name)
record = table.to_pandas().iloc[0]
model = MnistCNN()
model.load_state_dict(deserialize(record["weights"]))
model.eval()

# 启动 FastAPI
@app.post("/predict")
async def predict(request: PredictRequest):
    image = decode_base64_image(request.image)
    tensor = preprocess(image)
    with torch.no_grad():
        output = model(tensor)
    probs = torch.softmax(output, dim=1)
    return {
        "prediction": probs.argmax().item(),
        "confidence": probs.max().item(),
        "probabilities": probs[0].tolist(),
    }
```

## 7. MNIST 实现规划

### 7.1 目录结构

```
demo4_ai_platform/
├── design.md                    # 本文档
├── README.md
├── requirements.txt
├── lance_storage/               # 数据湖根目录（共享存储）
│   ├── datasets/
│   └── models/
├── data_platform/               # 数据平台（独立微服务）
│   ├── __init__.py
│   ├── app.py                   # FastAPI 主应用（存储 + 计算 API）
│   ├── storage.py               # Lance 读写封装
│   └── compute.py               # Daft + Ray 计算封装
├── ai_platform/                 # AI 平台（独立微服务）
│   ├── __init__.py
│   ├── app.py                   # FastAPI 主应用（任务 + 推理 API）
│   ├── ingestion.py             # 数据入库任务管理
│   ├── training.py              # 模型训练任务管理
│   └── serving.py               # 推理服务管理
├── scripts/                     # 用户脚本（清洗 + 训练）
│   ├── pipelines/
│   │   └── mnist_clean.py       # MNIST 清洗脚本
│   └── training/
│       └── mnist_cnn.py         # MNIST CNN 训练脚本
└── tests/
    └── unit/
```

### 7.2 实现步骤

1. `data_platform/storage.py` — Lance 读写封装（列出、查看 schema、删除）
2. `data_platform/compute.py` — Daft 计算封装（执行用户脚本、管理 Ray runner）
3. `data_platform/app.py` — Data Platform HTTP API（存储 + 计算路由）
4. `scripts/pipelines/mnist_clean.py` — MNIST 清洗脚本（实现 `run()` 接口）
5. `scripts/training/mnist_cnn.py` — PyTorch CNN 训练脚本（实现 `run()` 接口）
6. `ai_platform/ingestion.py` — 数据入库任务管理（调用 Data Platform API）
7. `ai_platform/training.py` — 训练任务管理（调用 Data Platform API）
8. `ai_platform/serving.py` — 推理服务管理
9. `ai_platform/app.py` — AI Platform HTTP API（任务 + 推理路由）
10. 单元测试

### 7.3 技术选型

| 需求 | 选型 | 理由 |
|------|------|------|
| 数据存储 | LanceDB + Lance | 统一存储结构化数据和二进制数据（模型权重） |
| 数据处理 | Daft | 支持 Lance 原生读写，lazy evaluation |
| 分布式运行时 | Ray | Daft 后端，也可用于分布式训练 |
| 模型训练 | PyTorch (CPU) | 轻量，MNIST 不需要 GPU |
| API 框架 | FastAPI | 异步支持好，自动生成 OpenAPI 文档 |
| Web 界面 | Gradio（可选） | 快速搭建手写体识别 demo 页面 |

### 7.4 火山引擎理念在 MNIST 中的体现

- **Lance 列式存储**：图像数据以 Lance 格式存储，按需加载
- **懒加载**：训练时只读取需要的列（image + label），不加载元数据
- **零拷贝 Schema 演进**：模型表可以灵活添加 metrics 列，不需要重写
- **统一存储**：数据集和模型都在同一个数据湖中，通过路径区分

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
