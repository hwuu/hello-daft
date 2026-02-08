# Hello Daft - 分布式数据处理学习教程

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Daft](https://img.shields.io/badge/Daft-0.2+-green.svg)](https://www.getdaft.io/)
[![Ray](https://img.shields.io/badge/Ray-2.9+-orange.svg)](https://docs.ray.io/)
[![LanceDB](https://img.shields.io/badge/LanceDB-0.4+-purple.svg)](https://lancedb.github.io/lancedb/)

一个面向初学者的系列教程，通过循序渐进的方式学习现代分布式数据处理技术栈。

## 📚 教程概览

本项目包含 4 个独立的 Demo，每个 Demo 专注于一个核心技术：

| Demo | 主题 | 学习内容 | 难度 |
|------|------|----------|------|
| [Demo 1](./demo1_daft/) | Daft 基础 | 分布式数据框架的使用 | ⭐ |
| [Demo 2](./demo2_ray/) | Ray on K8s | 分布式计算和 K8s 部署 | ⭐⭐ |
| [Demo 3](./demo3_lancedb/) | LanceDB 基础 | 向量数据库和语义搜索 | ⭐⭐ |
| [Demo 4](./demo4_integrated/) | 综合应用 | 完整的数据清洗管道 | ⭐⭐⭐ |

## 🎯 适合人群

- 有 Python 和 Pandas 基础的开发者
- 对分布式数据处理感兴趣的学习者
- 希望了解现代数据工程技术栈的工程师
- 需要构建数据清洗管道的实践者

## 🚀 快速开始

### 前置要求

- Python 3.10 或更高版本
- Docker（用于 Kubernetes 部署）
- kubectl（用于 K8s 管理）
- 至少 16GB 内存和 50GB 磁盘空间

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-username/hello_daft.git
cd hello_daft
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装基础依赖**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **验证安装**
```bash
python -c "import daft; print(f'Daft version: {daft.__version__}')"
python -c "import ray; print(f'Ray version: {ray.__version__}')"
python -c "import lancedb; print(f'LanceDB version: {lancedb.__version__}')"
```

## 📖 学习路径

### 推荐学习顺序

```
Demo 1: Daft 基础
    ↓ (1-2 天)
Demo 2: Ray on Kubernetes
    ↓ (2-3 天)
Demo 3: LanceDB 基础
    ↓ (1-2 天)
Demo 4: 综合应用
    ↓ (3-4 天)
完成！🎉
```

### Demo 1: Daft 基础使用

学习 Daft 分布式数据框架的核心概念和基本操作。

**内容**：
- Daft DataFrame 基础操作
- 数据读取和写入
- 聚合、过滤、连接
- 用户自定义函数（UDF）
- 性能对比（Daft vs Pandas）

**开始学习**：
```bash
cd demo1_daft
jupyter notebook notebooks/01_introduction.ipynb
```

### Demo 2: Ray on Kubernetes

学习 Ray 分布式计算框架及其在 Kubernetes 上的部署。

**内容**：
- Ray Tasks 和 Actors
- 分布式并行计算
- KubeRay Operator 部署
- Ray Dashboard 监控
- 实战：分布式图像处理

**开始学习**：
```bash
cd demo2_ray
jupyter notebook notebooks/01_ray_basics.ipynb
```

### Demo 3: LanceDB 基础使用

学习向量数据库的概念和 LanceDB 的使用方法。

**内容**：
- 向量数据库基础
- LanceDB CRUD 操作
- 文本嵌入生成
- 向量相似度搜索
- 实战：语义搜索引擎

**开始学习**：
```bash
cd demo3_lancedb
jupyter notebook notebooks/01_lancedb_introduction.ipynb
```

### Demo 4: 综合应用 - 数据清洗管道

整合所有技术，构建完整的分布式数据清洗管道。

**内容**：
- 端到端数据管道设计
- Daft + Ray 分布式清洗
- 嵌入生成和存储
- Kubernetes 完整部署
- 性能测试和优化

**开始学习**：
```bash
cd demo4_integrated
jupyter notebook notebooks/01_pipeline_overview.ipynb
```

## 🏗️ 项目结构

```
hello_daft/
├── README.md                    # 本文件
├── DESIGN.md                    # 详细设计文档
├── requirements.txt             # Python 依赖
│
├── demo1_daft/                  # Demo 1: Daft 基础
│   ├── README.md
│   ├── notebooks/
│   ├── src/
│   └── data/
│
├── demo2_ray/                   # Demo 2: Ray on K8s
│   ├── README.md
│   ├── notebooks/
│   ├── k8s/
│   ├── src/
│   └── scripts/
│
├── demo3_lancedb/               # Demo 3: LanceDB
│   ├── README.md
│   ├── notebooks/
│   ├── src/
│   └── data/
│
├── demo4_integrated/            # Demo 4: 综合应用
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── notebooks/
│   ├── k8s/
│   ├── src/
│   └── docker/
│
└── docs/                        # 共享文档
    ├── setup_guide.md
    ├── troubleshooting.md
    └── best_practices.md
```

## 💡 核心技术

### Daft
分布式数据框架，提供类似 Pandas 的 API，但能够处理大规模数据集。

**特点**：
- 延迟执行和查询优化
- 原生支持复杂数据类型
- 与 Ray 深度集成
- 高性能的分布式计算

### Ray
通用的分布式计算框架，简化并行和分布式应用的开发。

**特点**：
- 简单的 API（@ray.remote）
- 自动资源管理
- 容错和重试机制
- 丰富的生态系统

### LanceDB
现代化的向量数据库，专为 AI 应用设计。

**特点**：
- 快速的向量相似度搜索
- 支持结构化和非结构化数据
- ACID 事务保证
- 嵌入式和服务器模式

### Kubernetes
容器编排平台，用于自动化部署、扩展和管理容器化应用。

**特点**：
- 自动扩缩容
- 服务发现和负载均衡
- 自我修复
- 声明式配置

## 📊 示例数据集

每个 Demo 都包含示例数据集和生成脚本：

- **Demo 1**: 100K 条电商产品数据（~50MB）
- **Demo 2**: 10K 张示例图片（~500MB）
- **Demo 3**: 50K 条商品评论（~100MB）
- **Demo 4**: 1M 条评论数据（~2GB）

生成自定义数据：
```bash
cd demo1_daft/data
python generate_data.py --size 100000 --output products.csv
```

## 🔧 环境要求

### 本地开发
- **CPU**: 4 核以上
- **内存**: 16GB 以上
- **磁盘**: 50GB 可用空间
- **操作系统**: Linux / macOS / Windows (WSL2)

### Kubernetes 集群（Demo 2 和 Demo 4）
- **节点数**: 最少 3 个节点
- **每节点**: 4 CPU, 16GB 内存, 100GB 磁盘
- **推荐**: minikube / kind / k3s（本地）或 GKE / EKS / AKS（云端）

## 🐛 故障排除

### 常见问题

**1. 依赖安装失败**
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**2. Jupyter Notebook 无法启动**
```bash
# 重新安装 Jupyter
pip install --upgrade jupyter notebook
jupyter notebook --generate-config
```

**3. Ray 集群连接失败**
```bash
# 检查 Ray 状态
ray status
# 重启 Ray
ray stop
ray start --head
```

**4. Kubernetes 资源不足**
```bash
# 检查节点资源
kubectl top nodes
kubectl describe nodes
```

更多问题请查看 [故障排除指南](./docs/troubleshooting.md)

## 📈 性能基准

**Demo 4 性能目标**（1M 条记录）：

| 阶段 | 目标性能 |
|------|----------|
| 数据摄取 | > 50K 条/秒 |
| 数据清洗 | > 20K 条/秒 |
| 嵌入生成 | > 1K 条/秒 |
| 向量搜索 | < 100ms |
| 端到端 | < 15 分钟 |

## 🤝 贡献

欢迎贡献！请查看 [贡献指南](./CONTRIBUTING.md)

贡献方式：
- 🐛 报告 Bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- ⭐ 添加新的示例

## 📚 参考资源

### 官方文档
- [Daft 文档](https://www.getdaft.io/projects/docs/)
- [Ray 文档](https://docs.ray.io/)
- [LanceDB 文档](https://lancedb.github.io/lancedb/)
- [Kubernetes 文档](https://kubernetes.io/docs/)

### 推荐阅读
- [Daft 博客](https://blog.getdaft.io/)
- [Ray 教程](https://docs.ray.io/en/latest/ray-overview/index.html)
- [向量数据库介绍](https://www.pinecone.io/learn/vector-database/)

### 社区
- [Daft Discord](https://discord.gg/daft)
- [Ray Slack](https://forms.gle/9TSdDYUgxYs8SA9e8)
- [LanceDB GitHub](https://github.com/lancedb/lancedb)

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](./LICENSE) 文件

## 🙏 致谢

感谢以下开源项目：
- [Daft](https://github.com/Eventual-Inc/Daft) - 分布式数据框架
- [Ray](https://github.com/ray-project/ray) - 分布式计算框架
- [LanceDB](https://github.com/lancedb/lancedb) - 向量数据库
- [Kubernetes](https://github.com/kubernetes/kubernetes) - 容器编排

## 📧 联系方式

- 问题反馈：[GitHub Issues](https://github.com/your-username/hello_daft/issues)
- 讨论交流：[GitHub Discussions](https://github.com/your-username/hello_daft/discussions)

---

**开始你的学习之旅吧！** 🚀

从 [Demo 1: Daft 基础](./demo1_daft/) 开始，或查看 [设计文档](./DESIGN.md) 了解更多细节。
