# Demo 1: Daft 基础使用

通过 5 个 Notebook 从零掌握 Daft DataFrame 的核心操作与 AI 能力。

## 前置要求

- Python 3.10+
- 已安装项目依赖（`pip install -r requirements.txt`）

## 快速开始

```bash
# 1. 生成示例数据（默认 100K 条产品记录）
python data/generate_data.py

# 2. 启动 Notebook
jupyter notebook notebooks/01_introduction.ipynb
```

## Notebook 列表

| 编号 | 文件 | 内容 |
|------|------|------|
| 01 | `01_introduction.ipynb` | Daft 介绍：安装、核心概念、第一个程序 |
| 02 | `02_basic_operations.ipynb` | 基础操作：读取数据、选择过滤、排序去重 |
| 03 | `03_data_processing.ipynb` | 数据处理：聚合分组、Join、缺失值、窗口函数 |
| 04 | `04_advanced_features.ipynb` | 高级特性：UDF、复杂类型、查询计划、性能对比 |
| 05 | `05_ai_multimodal.ipynb` | AI Functions：文本分类、嵌入、语义搜索、LLM 提取（需要 OpenAI API Key） |

按编号顺序依次学习，每个 Notebook 包含概念讲解、代码示例和练习题。

## 下一步

完成本 Demo 后，继续学习 [Demo 2: Ray on Kubernetes](../demo2_ray/) — 分布式计算与 K8s 部署。
