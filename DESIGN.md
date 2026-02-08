# Hello Daft 系列教程 - 设计文档

## 1. 项目概述

这是一个面向初学者的系列教程项目，通过循序渐进的方式学习现代数据处理技术栈。

### 1.1 目标受众
- Python 开发者（有基本的 Python 和 Pandas 经验）
- 对分布式数据处理感兴趣的学习者
- 希望了解 Daft、Ray、LanceDB 的初学者
- 需要构建数据清洗管道的工程师

### 1.2 学习目标
通过本系列教程，学习者将掌握：
1. Daft 分布式数据框架的基本使用
2. Ray 分布式计算框架及其在 Kubernetes 上的部署
3. LanceDB 向量数据库的使用方法
4. 如何组合这些技术构建完整的数据清洗管道

## 2. 教程架构

本项目采用**渐进式学习路径**，分为 4 个独立的 Demo：

```
Demo 1: Daft 基础
    ↓
Demo 2: Ray on Kubernetes
    ↓
Demo 3: LanceDB 基础
    ↓
Demo 4: 综合应用 - 数据清洗管道
```

每个 Demo 都是独立可运行的，但建议按顺序学习。

## 3. Demo 详细设计

### Demo 1: Daft 基础使用

**学习目标**：
- 理解 Daft 的核心概念和优势
- 掌握 Daft DataFrame 的基本操作
- 了解 Daft 与 Pandas 的区别
- 学习 Daft 的延迟执行机制

**内容大纲**：
1. **安装和环境设置**
   - 安装 Daft
   - 验证安装

2. **基础操作**
   - 创建 DataFrame
   - 读取数据（CSV、Parquet、JSON）
   - 基本查询和过滤
   - 列操作和转换

3. **数据处理**
   - 聚合和分组
   - 排序和去重
   - 连接操作（Join）
   - 处理缺失值

4. **高级特性**
   - 用户自定义函数（UDF）
   - 复杂数据类型（嵌套结构）
   - 延迟执行和查询优化
   - 性能对比（Daft vs Pandas）

5. **AI Functions 与多模态**
   - 文本分类（classify_text）
   - 文本嵌入（embed_text）
   - 语义搜索（cosine_distance）
   - LLM 结构化提取（prompt）

**示例数据集**：
- 电商产品数据（100K 条记录）
- 包含：产品ID、名称、类别、价格、评分、描述等

**目录结构**：
```
demo1_daft/
├── README.md                    # Demo 说明文档
├── requirements.txt             # 依赖包
├── notebooks/                   # Notebook 自包含，代码全部内联
│   ├── 01_introduction.ipynb    # Daft 介绍
│   ├── 02_basic_operations.ipynb
│   ├── 03_data_processing.ipynb
│   ├── 04_advanced_features.ipynb
│   └── 05_ai_multimodal.ipynb   # AI Functions（需要 OpenAI API Key）
└── data/
    ├── products.csv
    └── generate_data.py         # 生成示例数据
```

**关键代码示例**：
```python
import daft

# 读取数据
df = daft.read_csv("data/products.csv")

# 基本操作
df_filtered = df.where(df["price"] > 100)
df_sorted = df_filtered.sort("rating", desc=True)

# 聚合
df_agg = df.groupby("category").agg(
    daft.col("price").mean().alias("avg_price"),
    daft.col("product_id").count().alias("count")
)

# 执行
result = df_agg.collect()
```

---

### Demo 2: Ray on Kubernetes

**学习目标**：
- 理解 Ray 的分布式计算模型
- 掌握 Ray 的基本 API（Tasks、Actors、Object Store）
- 学习 Daft + Ray 分布式数据处理
- 了解在 Kubernetes 上部署 Ray 集群（可选/进阶）

**内容大纲**：
1. **Ray 基础**（01_ray_basics.ipynb）
   - Ray 核心概念（Tasks、Actors、Objects）
   - 本地 Ray 集群初始化
   - 串行 vs 并行对比、Task 依赖链
   - Actor 有状态服务、Object Store 共享大对象
   - 资源管理和错误处理

2. **Daft + Ray 分布式数据处理**（02_daft_on_ray.ipynb）
   - Native Runner vs Ray Runner 架构
   - 配置 Ray Runner、复用 Demo 1 产品数据
   - 过滤/聚合/UDF 在 Ray 上执行
   - Native vs Ray 性能对比（100 万条数据）
   - 分区并行处理、Ray Dashboard 监控

3. **Kubernetes 部署**（03_kubernetes_deployment.ipynb，可选/进阶）
   - KubeRay Operator 安装
   - RayCluster CRD 配置详解
   - 连接远程集群、Dashboard 监控
   - 自动扩缩容、故障排除

**数据**：
- 复用 Demo 1 产品数据集（`demo1_daft/data/products.parquet`）
- 性能测试时通过 `generate_data.py` 动态生成 100 万条数据

**目录结构**：
```
demo2_ray/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_ray_basics.ipynb
│   ├── 02_daft_on_ray.ipynb
│   └── 03_kubernetes_deployment.ipynb
├── k8s/
│   ├── namespace.yaml
│   └── ray-cluster.yaml
└── scripts/
    ├── setup_ray_k8s.sh         # 安装 KubeRay + 部署集群
    └── cleanup.sh               # 清理资源
```

**关键代码示例**：
```python
import ray
import daft
from daft import col

# 初始化 Ray 并配置 Daft Ray Runner
ray.init()
daft.set_runner_ray()

# 读取 Demo 1 产品数据（与 Native Runner 代码完全相同）
df = daft.read_parquet("../../demo1_daft/data/products.parquet")

# 分布式数据处理
result = (
    df
    .where(col("price") > 100)
    .groupby("category")
    .agg(
        col("price").mean().alias("avg_price"),
        col("product_id").count().alias("count"),
    )
    .sort("count", desc=True)
    .collect()
)
```

**Kubernetes 配置示例**：
```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ray-demo-cluster
  namespace: ray-demo
spec:
  rayVersion: '2.53.0'
  headGroupSpec:
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.53.0
          resources:
            limits:
              cpu: "2"
              memory: "4Gi"
  workerGroupSpecs:
  - replicas: 2
    minReplicas: 1
    maxReplicas: 4
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.53.0
          resources:
            limits:
              cpu: "2"
              memory: "4Gi"
```

---

### Demo 3: LanceDB 基础使用

**学习目标**：
- 理解向量数据库的概念和应用场景
- 掌握 LanceDB 的基本操作
- 学习向量嵌入（Embeddings）的生成
- 实现语义搜索功能

**内容大纲**：
1. **LanceDB 介绍**
   - 向量数据库概念
   - LanceDB 特性和优势
   - 安装和配置

2. **基础操作**
   - 创建表和插入数据
   - 向量搜索
   - 过滤和查询
   - 更新和删除

3. **嵌入生成**
   - 使用 sentence-transformers
   - 文本嵌入
   - 图像嵌入（可选）

4. **实战应用**
   - 构建语义搜索引擎
   - 相似商品推荐
   - 问答系统

**示例数据集**：
- 商品评论数据（50K 条）
- 包含：评论ID、商品ID、用户ID、评论文本、评分等

**目录结构**：
```
demo3_lancedb/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_lancedb_introduction.ipynb
│   ├── 02_basic_operations.ipynb
│   ├── 03_embeddings.ipynb
│   └── 04_semantic_search.ipynb
├── data/
│   ├── reviews.csv
│   └── generate_reviews.py
└── lancedb_data/                # LanceDB 数据目录
```

**关键代码示例**：
```python
import lancedb
from sentence_transformers import SentenceTransformer

# 连接数据库
db = lancedb.connect("lancedb_data")

# 创建嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 准备数据
data = [
    {
        "id": 1,
        "text": "这个产品质量很好",
        "vector": model.encode("这个产品质量很好").tolist()
    }
]

# 创建表
table = db.create_table("reviews", data)

# 向量搜索
query = "质量不错的商品"
query_vector = model.encode(query)
results = table.search(query_vector).limit(5).to_list()
```

---

### Demo 4: 综合应用 - 数据清洗管道

**学习目标**：
- 整合 Daft、Ray、LanceDB 构建完整管道
- 实现端到端的数据清洗流程
- 在 Kubernetes 上部署完整系统
- 理解分布式数据处理的最佳实践

**内容大纲**：
1. **系统架构**
   - 整体架构设计
   - 组件交互流程
   - 数据流设计

2. **数据清洗管道**
   - 数据摄取（Daft）
   - 分布式清洗（Daft + Ray）
   - 嵌入生成（Ray）
   - 存储和索引（LanceDB）

3. **Kubernetes 部署**
   - 完整系统部署
   - 服务编排
   - 资源配置
   - 监控和日志

4. **端到端演示**
   - 处理真实数据集
   - 性能测试
   - 结果验证

**应用场景**：
处理电商评论数据，实现以下功能：
1. 清洗和标准化评论文本
2. 去重和质量过滤
3. 生成语义嵌入
4. 存储到 LanceDB
5. 提供语义搜索 API

**目录结构**：
```
demo4_integrated/
├── README.md
├── requirements.txt
├── ARCHITECTURE.md              # 架构文档
├── notebooks/
│   ├── 01_pipeline_overview.ipynb
│   ├── 02_data_ingestion.ipynb
│   ├── 03_cleaning_process.ipynb
│   ├── 04_embedding_generation.ipynb
│   └── 05_query_and_validation.ipynb
├── k8s/
│   ├── namespace.yaml
│   ├── ray/
│   │   └── ray-cluster.yaml
│   ├── lancedb/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── pvc.yaml
│   ├── minio/                   # 对象存储（可选）
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── pvc.yaml
│   └── kustomization.yaml
├── data/
│   ├── raw/
│   │   └── reviews_raw.csv      # 原始数据
│   └── processed/               # 处理后数据
├── scripts/
│   ├── generate_sample_data.py  # 生成测试数据
│   ├── deploy_all.sh            # 一键部署
│   ├── run_pipeline.py          # 运行管道
│   └── cleanup.sh
└── docker/
    ├── Dockerfile.pipeline
    └── Dockerfile.jupyter
```

**数据流程图**：
```
┌─────────────────┐
│  原始数据源      │
│  (CSV/JSON)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  阶段 1: 数据摄取 (Daft)     │
│  - 读取多种格式              │
│  - 模式推断                  │
│  - 初步验证                  │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  阶段 2: 数据清洗            │
│  (Daft + Ray 分布式执行)    │
│  - 去重                      │
│  - 缺失值处理                │
│  - 文本标准化                │
│  - 数据验证                  │
│  - 异常值检测                │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  阶段 3: 特征工程            │
│  (Ray 并行处理)              │
│  - 文本预处理                │
│  - 生成嵌入向量              │
│  - 提取元数据                │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  阶段 4: 存储 (LanceDB)      │
│  - 写入清洗后数据            │
│  - 创建向量索引              │
│  - 元数据索引                │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  阶段 5: 查询和验证          │
│  - 数据质量检查              │
│  - 语义搜索测试              │
│  - 性能基准测试              │
└─────────────────────────────┘
```

**核心代码示例**：
```python
import daft
import ray
import lancedb
from sentence_transformers import SentenceTransformer

# 初始化 Ray
ray.init(address="ray://ray-head:10001")

# 阶段 1: 数据摄取
df = daft.read_csv("s3://bucket/reviews/*.csv")

# 阶段 2: 数据清洗
df_clean = (
    df
    .drop_duplicates()
    .where(df["text"].is_not_null())
    .where(df["text"].str.length() > 10)
    .with_column("text_clean",
                 df["text"].str.lower().str.strip())
)

# 阶段 3: 生成嵌入
@ray.remote
def generate_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text).tolist()

# 使用 Daft UDF 集成 Ray
def embed_udf(text_series):
    futures = [generate_embedding.remote(t) for t in text_series]
    return ray.get(futures)

df_embedded = df_clean.with_column(
    "embedding",
    daft.udf(embed_udf, return_dtype=daft.DataType.list(daft.DataType.float32()))(df_clean["text_clean"])
)

# 阶段 4: 存储到 LanceDB
data = df_embedded.to_pandas()
db = lancedb.connect("lancedb_data")
table = db.create_table("reviews_clean", data)

# 阶段 5: 查询验证
query = "产品质量很好"
query_vector = generate_embedding.remote(query)
results = table.search(ray.get(query_vector)).limit(10).to_list()
```

---

## 4. 技术栈

### 4.1 核心依赖
```
Python >= 3.10
daft >= 0.2.0
ray >= 2.9.0
lancedb >= 0.4.0
sentence-transformers >= 2.2.0
pandas >= 2.0.0
pyarrow >= 14.0.0
```

### 4.2 Kubernetes 环境
```
Kubernetes >= 1.27
KubeRay Operator >= 1.0
kubectl >= 1.27
```

### 4.3 可选组件
```
MinIO (对象存储)
Jupyter Notebook
Prometheus + Grafana (监控)
```

## 5. 项目根目录结构

```
hello_daft/
├── README.md                    # 项目总览
├── DESIGN.md                    # 本设计文档
├── CONTRIBUTING.md              # 贡献指南
├── LICENSE
│
├── demo1_daft/                  # Demo 1
├── demo2_ray/                   # Demo 2
├── demo3_lancedb/               # Demo 3
├── demo4_integrated/            # Demo 4
│
└── docs/                        # 共享文档
    ├── setup_guide.md           # 环境设置指南
    ├── troubleshooting.md       # 常见问题
    └── best_practices.md        # 最佳实践
```

## 6. 学习路径建议

### 6.1 初学者路径（推荐）
1. **第 1 周**：Demo 1 - Daft 基础
   - 完成所有 notebook
   - 理解 Daft 的核心概念
   - 完成练习题

2. **第 2 周**：Demo 2 - Ray on Kubernetes
   - 本地 Ray 实验
   - 搭建 K8s 环境
   - 部署 Ray 集群

3. **第 3 周**：Demo 3 - LanceDB 基础
   - 学习向量数据库概念
   - 实现语义搜索
   - 完成小项目

4. **第 4 周**：Demo 4 - 综合应用
   - 理解整体架构
   - 部署完整系统
   - 运行端到端管道

### 6.2 快速上手路径
- 有经验的开发者可以直接从 Demo 4 开始
- 遇到不熟悉的技术再回看对应的基础 Demo

### 6.3 深入学习路径
- 完成所有 Demo 后，尝试以下进阶任务：
  1. 优化管道性能
  2. 添加实时流处理
  3. 实现数据版本控制
  4. 集成 MLOps 工具链

## 7. 示例数据集

### 7.1 数据集规模
- **Demo 1**: 100K 条产品记录（~50MB）
- **Demo 2**: 复用 Demo 1 数据 + 动态生成 100 万条大数据集用于性能测试
- **Demo 3**: 50K 条评论（~100MB）
- **Demo 4**: 1M 条评论（~2GB）

### 7.2 数据生成
每个 Demo 都包含数据生成脚本，可以自定义数据规模：
```bash
python data/generate_data.py --size 100000 --output data/products.csv
```

### 7.3 真实数据集（可选）
- Amazon Reviews Dataset
- Yelp Reviews Dataset
- 可以替换示例数据进行实验

## 8. 部署环境要求

### 8.1 本地开发环境
- **CPU**: 4 核以上
- **内存**: 16GB 以上
- **磁盘**: 50GB 可用空间
- **操作系统**: Linux / macOS / Windows (WSL2)

### 8.2 Kubernetes 集群
- **节点数**: 最少 3 个节点
- **每节点配置**:
  - CPU: 4 核
  - 内存: 16GB
  - 磁盘: 100GB
- **推荐环境**:
  - 本地: minikube / kind / k3s
  - 云端: GKE / EKS / AKS

### 8.3 网络要求
- 能够访问 Docker Hub 或配置镜像加速
- 能够访问 PyPI 或配置国内镜像
- K8s 集群内部网络互通

## 9. 评估标准

### 9.1 Demo 完成标准
每个 Demo 完成后，学习者应该能够：

**Demo 1**:
- [ ] 独立编写 Daft 数据处理脚本
- [ ] 理解延迟执行的优势
- [ ] 能够对比 Daft 和 Pandas 的性能差异

**Demo 2**:
- [ ] 编写 Ray 分布式任务
- [ ] 在 K8s 上部署 Ray 集群
- [ ] 使用 Ray Dashboard 监控任务

**Demo 3**:
- [ ] 创建和查询 LanceDB 表
- [ ] 生成文本嵌入
- [ ] 实现基本的语义搜索功能

**Demo 4**:
- [ ] 理解完整的数据管道架构
- [ ] 部署所有组件到 K8s
- [ ] 运行端到端的数据清洗流程
- [ ] 验证数据质量和性能指标

### 9.2 性能基准
**Demo 4 性能目标**:
- 数据摄取: > 50K 条/秒
- 数据清洗: > 20K 条/秒
- 嵌入生成: > 1K 条/秒
- 向量搜索: < 100ms 延迟
- 端到端处理 1M 条记录: < 15 分钟

## 10. 常见问题和故障排除

### 10.1 环境问题
- Python 版本不兼容
- 依赖包冲突
- K8s 集群资源不足

### 10.2 性能问题
- Ray 集群配置不当
- 内存溢出
- 网络瓶颈

### 10.3 数据问题
- 数据格式错误
- 编码问题
- 缺失值处理

详细的故障排除指南见 `docs/troubleshooting.md`

## 11. 扩展和进阶

### 11.1 可选扩展
1. **实时流处理**
   - 集成 Kafka
   - 实现增量更新

2. **高级监控**
   - Prometheus + Grafana
   - 自定义指标

3. **CI/CD**
   - GitHub Actions
   - 自动化测试和部署

4. **安全加固**
   - RBAC 配置
   - 密钥管理
   - 网络策略

### 11.2 进阶项目建议
- 构建推荐系统
- 实现实时异常检测
- 开发数据质量监控平台
- 集成 MLflow 进行模型管理

## 12. 贡献指南

欢迎贡献！可以通过以下方式参与：
1. 报告 Bug 和问题
2. 提交新的示例代码
3. 改进文档
4. 添加新的 Demo

详见 `CONTRIBUTING.md`

## 13. 参考资源

### 13.1 官方文档
- [Daft 文档](https://www.getdaft.io/projects/docs/)
- [Ray 文档](https://docs.ray.io/)
- [LanceDB 文档](https://lancedb.github.io/lancedb/)
- [KubeRay 文档](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)

### 13.2 推荐阅读
- 《Designing Data-Intensive Applications》
- 《Kubernetes in Action》
- Ray 官方教程
- Daft 博客文章

### 13.3 社区资源
- Daft Discord 社区
- Ray Slack 频道
- LanceDB GitHub Discussions

## 14. 版本规划

### v1.0 (当前版本)
- [x] 设计文档
- [x] Demo 1: Daft 基础
- [x] Demo 2: Ray on Kubernetes
- [ ] Demo 3: LanceDB 基础
- [ ] Demo 4: 综合应用

### v1.1 (计划中)
- [ ] 添加视频教程
- [ ] 增加更多练习题
- [ ] 性能优化指南
- [ ] 多语言支持（英文版）

### v2.0 (未来)
- [ ] 实时流处理 Demo
- [ ] MLOps 集成
- [ ] 生产环境最佳实践
- [ ] 高级调优指南

---

**文档版本**: 1.0
**最后更新**: 2026-02-08
**作者**: 系统设计
**状态**: 待审核
