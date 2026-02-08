# Demo 2: Ray on Kubernetes

欢迎来到 Ray on Kubernetes 教程！这是 Hello Daft 系列的第二个 Demo。

## 学习目标

通过本 Demo，你将学习：
- Ray 分布式计算框架的核心概念（Tasks、Actors、Object Store）
- Daft + Ray 分布式数据处理
- Native Runner 与 Ray Runner 的性能对比
- 在 Kubernetes 上部署 Ray 集群（可选/进阶）

## 前置要求

- 完成 [Demo 1: Daft 基础](../demo1_daft/)
- Python 3.10+
- 至少 8GB 内存

## 内容结构

### Notebook 教程

| 编号 | 文件 | 内容 | 级别 |
|------|------|------|------|
| 01 | `01_ray_basics.ipynb` | Ray 核心概念：Tasks、Actors、Object Store、资源管理、错误处理 | 基础 |
| 02 | `02_daft_on_ray.ipynb` | Daft + Ray 集成：Ray Runner 配置、分布式数据处理、性能对比 | 基础 |
| 03 | `03_kubernetes_deployment.ipynb` | K8s 部署 Ray 集群：KubeRay Operator、自动扩缩容、故障排除 | **进阶（可选）** |

### 数据

本 Demo 复用 Demo 1 的产品数据集（`demo1_daft/data/products.parquet`），性能测试时通过 `generate_data.py` 动态生成大数据集。

## 快速开始

### 1. 安装依赖

```bash
cd demo2_ray
pip install -r requirements.txt
```

### 2. 运行 Notebook

```bash
# 从 Ray 基础开始
jupyter notebook notebooks/01_ray_basics.ipynb
```

### 3. K8s 部署（可选）

需要额外安装 `kubectl`、`helm`、`minikube`/`kind`。

```bash
# 一键部署
bash scripts/setup_ray_k8s.sh

# 一键清理
bash scripts/cleanup.sh
```

## 目录结构

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
    ├── setup_ray_k8s.sh
    └── cleanup.sh
```

## 核心概念

### Ray 三大抽象

| 抽象 | 说明 | 适用场景 |
|------|------|----------|
| **Tasks** | 无状态远程函数 | 并行计算、批处理 |
| **Actors** | 有状态远程类 | 有状态服务、计数器 |
| **Objects** | 分布式共享对象 | 大数据传递、中间结果共享 |

### Daft + Ray

```python
import daft
from daft import col

# 配置 Ray Runner（一行代码切换）
daft.set_runner_ray()

# 之后的操作与 Native Runner 完全相同
df = daft.read_parquet("products.parquet")
result = df.where(col("price") > 100).groupby("category").agg(
    col("price").mean().alias("avg_price"),
).collect()
```

## 常见问题

### Ray 初始化失败

```python
# 检查 Ray 是否已经运行
ray.is_initialized()

# 如果已运行，先关闭
ray.shutdown()

# 重新初始化
ray.init()
```

### Daft Runner 不能切换

Daft 的 Runner 在一个 Python 进程中只能设置一次。如需切换，请重启 Python 进程/Kernel。

## 参考资源

- [Ray 官方文档](https://docs.ray.io/)
- [Daft Ray Runner 文档](https://www.getdaft.io/projects/docs/en/latest/user_guide/integrations/ray.html)
- [KubeRay 文档](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)

## 下一步

完成本 Demo 后，继续学习：

[Demo 3: LanceDB 基础](../demo3_lancedb/) — 学习向量数据库和语义搜索
