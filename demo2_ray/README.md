# Demo 2: Ray on Kubernetes

学习 Ray 分布式计算框架，并将 Daft 运行在 Ray 集群上实现分布式数据处理。

## 前置要求

- 完成 [Demo 1: Daft 基础](../demo1_daft/)
- Python 3.10+
- 至少 8GB 内存

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Notebook
jupyter notebook notebooks/01_ray_basics.ipynb

# 3. K8s 部署（可选，需要 kubectl/helm/minikube）
bash scripts/setup_ray_k8s.sh
```

## Notebook 列表

| 编号 | 文件 | 内容 | 级别 |
|------|------|------|------|
| 01 | `01_ray_basics.ipynb` | Ray 核心概念：Tasks、Actors、Object Store、资源管理、错误处理 | 基础 |
| 02 | `02_daft_on_ray.ipynb` | Daft + Ray 集成：Ray Runner 配置、分布式数据处理、性能对比 | 基础 |
| 03 | `03_kubernetes_deployment.ipynb` | K8s 部署 Ray 集群：KubeRay Operator、自动扩缩容、故障排除 | 进阶（可选） |

按编号顺序依次学习，Notebook 03 为可选的进阶内容。

## 下一步

完成本 Demo 后，继续学习 [Demo 3: LanceDB 基础](../demo3_lancedb/) — 向量数据库和语义搜索。
