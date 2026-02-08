# Demo 4: 综合应用 - 数据清洗管道

欢迎来到综合应用教程！这是 Hello Daft 系列的第四个（也是最后一个）Demo。

## 📖 学习目标

通过本 Demo，你将学习：
- ✅ 整合 Daft、Ray、LanceDB 构建完整数据管道
- ✅ 实现端到端的数据清洗流程
- ✅ 分布式嵌入生成（Daft UDF + Ray）
- ✅ 向量存储与语义搜索（LanceDB）
- ✅ 在 Kubernetes 上部署完整系统
- ✅ 数据质量验证和性能基准测试

## 🎯 适合人群

- 已完成 Demo 1-3 的学习者
- 需要构建数据清洗管道的工程师
- 希望了解分布式数据处理最佳实践的开发者
- 对端到端数据系统感兴趣的学习者

## ⏱️ 预计学习时间

- **快速浏览**: 3-4 小时
- **深入学习**: 2-3 天
- **完成 K8s 部署**: 额外 1-2 天

## 📚 内容结构

### Notebook 教程

1. **01_pipeline_overview.ipynb** - 管道概览
   - 系统架构介绍
   - 组件交互流程
   - 数据流设计
   - 运行环境准备

2. **02_data_ingestion.ipynb** - 数据摄取
   - 使用 Daft 读取多种格式
   - 模式推断和初步验证
   - 数据质量预检

3. **03_cleaning_process.ipynb** - 数据清洗
   - Daft + Ray 分布式清洗
   - 去重和缺失值处理
   - 文本标准化
   - 异常值检测

4. **04_embedding_generation.ipynb** - 嵌入生成
   - Ray 并行嵌入计算
   - Daft UDF 集成 Ray
   - 批量处理优化

5. **05_query_and_validation.ipynb** - 查询与验证
   - 数据质量检查
   - 语义搜索测试
   - 性能基准测试

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                   Kubernetes Cluster                     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              Ray Cluster (KubeRay)                 │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐        │  │
│  │  │ Head Node│  │ Worker 1 │  │ Worker 2 │  ...   │  │
│  │  └──────────┘  └──────────┘  └──────────┘        │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                │
│            ┌────────────┼────────────┐                   │
│            ▼            ▼            ▼                   │
│  ┌──────────────┐ ┌──────────┐ ┌──────────────┐        │
│  │ Daft Engine  │ │ Embedding│ │   LanceDB    │        │
│  │ (Ingestion + │ │ Service  │ │  (Storage +  │        │
│  │  Cleaning)   │ │ (Ray)    │ │   Search)    │        │
│  └──────────────┘ └──────────┘ └──────────────┘        │
│                                                          │
│  ┌──────────────┐                                       │
│  │ MinIO (S3)   │  ← Optional object storage            │
│  └──────────────┘                                       │
└──────────────────────────────────────────────────────────┘
```

## 🔄 数据清洗管道流程

```
┌─────────────────┐
│  Raw Data       │
│  (CSV/JSON)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Stage 1: Ingestion (Daft)  │
│  - Read multiple formats    │
│  - Schema inference         │
│  - Initial validation       │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Stage 2: Cleaning          │
│  (Daft + Ray distributed)   │
│  - Deduplication            │
│  - Null handling            │
│  - Text normalization       │
│  - Data validation          │
│  - Outlier detection        │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Stage 3: Feature Eng.      │
│  (Ray parallel)             │
│  - Text preprocessing       │
│  - Embedding generation     │
│  - Metadata extraction      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Stage 4: Storage (LanceDB) │
│  - Write cleaned data       │
│  - Create vector index      │
│  - Metadata index           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Stage 5: Query & Validate  │
│  - Data quality check       │
│  - Semantic search test     │
│  - Performance benchmark    │
└─────────────────────────────┘
```

## 🚀 快速开始

### 前置要求

- 已完成 Demo 1-3（或具备 Daft、Ray、LanceDB 基础知识）
- Python 3.10+
- 至少 16GB 内存

**Kubernetes 部署（可选）**：
- Kubernetes 集群（minikube/kind/k3s 或云端）
- kubectl 已安装并配置
- KubeRay Operator 已部署（参考 Demo 2）

### 1. 安装依赖

```bash
cd demo4_integrated

# 安装依赖
pip install -r ../requirements.txt

# 验证安装
python -c "import daft, ray, lancedb; print('All dependencies OK')"
```

### 2. 生成示例数据

```bash
python scripts/generate_sample_data.py --size 1000000
```

这将生成：
- `data/raw/reviews_raw.csv` - 1M 条电商评论数据
- 包含评论文本、评分、商品信息等

### 3. 本地运行管道

```bash
# 运行完整管道
python scripts/run_pipeline.py

# 或按阶段运行
python scripts/run_pipeline.py --stage ingestion
python scripts/run_pipeline.py --stage cleaning
python scripts/run_pipeline.py --stage embedding
python scripts/run_pipeline.py --stage storage
python scripts/run_pipeline.py --stage validation
```

### 4. 启动 Jupyter Notebook

```bash
jupyter notebook notebooks/01_pipeline_overview.ipynb
```

### 5. Kubernetes 部署（可选）

```bash
# 一键部署所有组件
./scripts/deploy_all.sh

# 或手动部署
kubectl apply -k k8s/

# 检查部署状态
kubectl get pods -n hello-daft
```

## 💡 核心概念

### 1. 管道编排

管道由 `src/main.py` 编排，按阶段顺序执行：

```python
import daft
import ray
import lancedb

class DataPipeline:
    def __init__(self, config):
        self.config = config
        ray.init(address=config.ray_address)

    def run(self):
        # 阶段 1: 数据摄取
        df = self.ingest()
        # 阶段 2: 数据清洗
        df_clean = self.clean(df)
        # 阶段 3: 嵌入生成
        df_embedded = self.embed(df_clean)
        # 阶段 4: 存储
        self.store(df_embedded)
        # 阶段 5: 验证
        self.validate()
```

### 2. Daft + Ray 分布式清洗

利用 Daft 的延迟执行和 Ray 的分布式计算能力：

```python
import daft

# 使用 Ray 后端执行 Daft 操作
daft.context.set_runner_ray()

df = daft.read_csv("data/raw/reviews_raw.csv")

df_clean = (
    df
    .drop_duplicates()
    .where(df["text"].is_not_null())
    .where(df["text"].str.length() > 10)
    .with_column("text_clean",
                 df["text"].str.lower().str.strip())
)
```

### 3. Ray 并行嵌入生成

通过 Daft UDF 集成 Ray 实现分布式嵌入计算：

```python
import ray
from sentence_transformers import SentenceTransformer

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
    daft.udf(
        embed_udf,
        return_dtype=daft.DataType.list(daft.DataType.float32())
    )(df_clean["text_clean"])
)
```

### 4. LanceDB 存储与搜索

将清洗后的数据和嵌入向量存储到 LanceDB：

```python
import lancedb

# 存储
data = df_embedded.to_pandas()
db = lancedb.connect("lancedb_data")
table = db.create_table("reviews_clean", data)

# 语义搜索验证
query = "产品质量很好"
query_vector = ray.get(generate_embedding.remote(query))
results = table.search(query_vector).limit(10).to_list()
```

## 📊 示例数据集

### 电商评论数据集（reviews_raw.csv）

**字段说明**：
- `review_id`: 评论唯一标识符
- `product_id`: 商品ID
- `user_id`: 用户ID
- `rating`: 评分 1-5
- `title`: 评论标题
- `text`: 评论内容
- `timestamp`: 评论时间
- `verified_purchase`: 是否验证购买
- `helpful_votes`: 有用投票数

**数据规模**：
- 默认：1,000,000 条评论
- 大小：约 2GB
- 可自定义大小

**数据特点**：
- 包含重复记录（约 3%）
- 包含缺失值（约 5%）
- 包含异常值（评分超范围等）
- 包含脏数据（特殊字符、HTML 标签等）
- 适合演示完整的数据清洗流程

## 🏗️ Kubernetes 部署

### 部署架构

```
┌─────────────────────────────────────────┐
│         Namespace: hello-daft           │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Ray Cluster (KubeRay)          │   │
│  │  - Head: 1 pod (2C/4Gi)        │   │
│  │  - Workers: 3-5 pods (4C/8Gi)  │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  LanceDB                        │   │
│  │  - Deployment + PVC             │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  MinIO (Optional)               │   │
│  │  - Object storage for raw data  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 部署步骤

```bash
# 1. 创建命名空间
kubectl apply -f k8s/namespace.yaml

# 2. 部署 Ray 集群
kubectl apply -f k8s/ray/ray-cluster.yaml

# 3. 部署 LanceDB
kubectl apply -f k8s/lancedb/deployment.yaml
kubectl apply -f k8s/lancedb/service.yaml
kubectl apply -f k8s/lancedb/pvc.yaml

# 4. (可选) 部署 MinIO
kubectl apply -f k8s/minio/

# 5. 验证部署
kubectl get pods -n hello-daft
```

### 连接集群运行管道

```python
import ray

# 连接 K8s 上的 Ray 集群
ray.init(address="ray://ray-cluster-head-svc:10001")

# 运行管道
from src.main import DataPipeline
pipeline = DataPipeline(config)
pipeline.run()
```

### 监控和日志

```bash
# Ray Dashboard
kubectl port-forward -n hello-daft service/ray-cluster-head-svc 8265:8265
# 访问 http://localhost:8265

# 查看管道日志
kubectl logs -n hello-daft -l app=pipeline -f

# 查看资源使用
kubectl top pods -n hello-daft
```

## 📈 性能基准

**Demo 4 性能目标**（1M 条记录）：

| 阶段 | 目标吞吐量 | 说明 |
|------|-----------|------|
| 数据摄取 | > 50K 条/秒 | Daft 读取 CSV/Parquet |
| 数据清洗 | > 20K 条/秒 | Daft + Ray 分布式清洗 |
| 嵌入生成 | > 1K 条/秒 | Ray 并行嵌入计算 |
| 向量搜索 | < 100ms 延迟 | LanceDB 向量检索 |

## 🎓 练习题

### 初级练习

1. **运行管道**
   - 使用默认配置运行完整管道
   - 观察每个阶段的输出
   - 检查最终数据质量

2. **修改清洗规则**
   - 添加新的文本清洗规则（如去除 HTML 标签）
   - 调整去重策略
   - 修改缺失值处理方式

3. **查询验证**
   - 使用不同的查询测试语义搜索
   - 对比清洗前后的搜索质量
   - 测试混合搜索（向量 + 过滤）

### 中级练习

4. **性能优化**
   - 调整 Ray Worker 数量观察性能变化
   - 优化嵌入生成的批量大小
   - 对比不同数据格式（CSV vs Parquet）的摄取速度

5. **管道扩展**
   - 添加新的清洗阶段
   - 实现增量处理（只处理新数据）
   - 添加数据质量报告生成

6. **监控集成**
   - 为管道添加日志记录
   - 实现进度追踪
   - 添加错误告警

### 高级练习

7. **Kubernetes 部署**
   - 部署完整系统到 K8s
   - 配置 Ray 集群自动扩缩容
   - 实现资源限制和配额

8. **生产化改造**
   - 添加配置管理
   - 实现优雅关闭和断点续传
   - 添加端到端测试

9. **架构优化**
   - 实现流式处理模式
   - 添加缓存层减少重复计算
   - 集成 Prometheus + Grafana 监控

## 🐛 常见问题

### Q1: Ray 集群连接失败

```python
# 检查 Ray 集群状态
ray.is_initialized()

# 本地模式运行（不需要集群）
ray.init()  # 不指定 address，使用本地模式

# K8s 环境下检查服务
# kubectl get svc -n hello-daft
```

### Q2: 嵌入生成速度慢

```python
# 1. 增加 Ray Worker 数量
# 编辑 k8s/ray/ray-cluster.yaml，增加 replicas

# 2. 使用批量编码
@ray.remote
def generate_embeddings_batch(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts).tolist()

# 3. 使用更轻量的模型
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 维，速度快
# 而不是
# model = SentenceTransformer('all-mpnet-base-v2')  # 768 维，较慢
```

### Q3: LanceDB 写入失败

```python
# 检查磁盘空间
import shutil
usage = shutil.disk_usage("lancedb_data")
print(f"可用空间: {usage.free / 1e9:.1f} GB")

# 分批写入
batch_size = 10000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    if i == 0:
        table = db.create_table("reviews_clean", batch)
    else:
        table.add(batch)
```

### Q4: 内存溢出（OOM）

```python
# 1. 减少数据量进行测试
python scripts/generate_sample_data.py --size 10000

# 2. 调整 Daft 分区数
df = daft.read_csv("data/raw/reviews_raw.csv").repartition(100)

# 3. K8s 环境下增加 Worker 内存
# 编辑 k8s/ray/ray-cluster.yaml
# resources.requests.memory: "16Gi"
```

## 📚 参考资源

### 官方文档
- [Daft 文档](https://www.getdaft.io/projects/docs/)
- [Ray 文档](https://docs.ray.io/)
- [LanceDB 文档](https://lancedb.github.io/lancedb/)
- [KubeRay 文档](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)

### 相关教程
- [Demo 1: Daft 基础](../demo1_daft/) - Daft DataFrame 操作
- [Demo 2: Ray on Kubernetes](../demo2_ray/) - Ray 分布式计算
- [Demo 3: LanceDB 基础](../demo3_lancedb/) - 向量数据库和语义搜索

### 推荐阅读
- 《Designing Data-Intensive Applications》
- 《Kubernetes in Action》
- [Daft + Ray 集成指南](https://www.getdaft.io/projects/docs/en/latest/user_guide/integrations/ray.html)

## ✅ 完成检查清单

完成本 Demo 后，你应该能够：

- [ ] 理解完整的数据管道架构
- [ ] 使用 Daft 进行数据摄取和清洗
- [ ] 使用 Ray 进行分布式嵌入生成
- [ ] 使用 LanceDB 存储和查询向量数据
- [ ] 通过 Daft UDF 集成 Ray 实现分布式处理
- [ ] 部署所有组件到 Kubernetes
- [ ] 运行端到端的数据清洗流程
- [ ] 验证数据质量和性能指标
- [ ] 处理常见的部署和运行问题
- [ ] 完成至少 3 个练习题

## 🎯 下一步

恭喜你完成了 Hello Daft 系列的全部教程！接下来你可以：

- 🔧 **优化管道性能** - 调优各阶段参数，提升吞吐量
- 📡 **添加实时流处理** - 集成 Kafka 实现增量更新
- 📊 **集成监控** - 使用 Prometheus + Grafana 监控系统
- 🤖 **集成 MLOps** - 使用 MLflow 管理模型和实验

更多进阶内容请参考 [设计文档](../DESIGN.md) 中的扩展和进阶章节。

---

**祝学习愉快！** 🚀

如有问题，请查看 [故障排除指南](../docs/troubleshooting.md) 或提交 [Issue](https://github.com/your-username/hello_daft/issues)。
