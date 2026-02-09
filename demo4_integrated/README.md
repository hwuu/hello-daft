# Demo 4: 智能商品搜索系统

整合 Daft、LanceDB 构建端到端的智能搜索系统：从原始数据清洗到语义搜索与商品推荐。复用 Demo 1 的产品数据和 Demo 3 的评论数据，将前三个 Demo 的技术有机串联。

## 前置要求

- 已完成 Demo 1 和 Demo 3（需要其生成的数据文件）
- SiliconFlow API Key（用于嵌入生成，Notebook 02/03 需要）
  ```bash
  export OPENAI_API_KEY='your-siliconflow-key'
  export OPENAI_BASE_URL='https://api.siliconflow.cn/v1'
  ```

## 快速开始

```bash
# 1. 安装依赖
pip install -r demo4_integrated/requirements.txt

# 2. 确保 demo1 和 demo3 数据已就绪
python demo1_daft/data/generate_data.py --size 1000 --output demo1_daft/data
python demo3_lancedb/data/prepare_data.py --output demo3_lancedb/data

# 3. 启动 Notebook
jupyter notebook demo4_integrated/notebooks/
```

## Notebook 列表

| 序号 | Notebook | 简介 |
|------|----------|------|
| 01 | [数据整合与清洗](notebooks/01_data_integration.ipynb) | Daft 读取产品和评论数据，清洗去重，构建搜索文本（无需 API Key） |
| 02 | [嵌入生成与向量存储](notebooks/02_embedding_storage.ipynb) | Daft embed_text 批量生成嵌入，写入 LanceDB，创建索引 |
| 03 | [智能搜索与推荐](notebooks/03_search_recommend.ipynb) | 语义搜索、混合过滤、相似商品推荐、评论洞察 |
