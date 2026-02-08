# Demo 3: LanceDB 基础使用

欢迎来到 LanceDB 基础教程！这是 Hello Daft 系列的第三个 Demo。

## 📖 学习目标

通过本 Demo，你将学习：
- ✅ 向量数据库的概念和应用场景
- ✅ LanceDB 的核心特性和优势
- ✅ LanceDB 的 CRUD 操作
- ✅ 文本嵌入（Embeddings）的生成
- ✅ 向量相似度搜索
- ✅ 实战：构建语义搜索引擎

## 🎯 适合人群

- 对 AI 应用开发感兴趣的开发者
- 需要实现语义搜索的工程师
- 希望了解向量数据库的学习者
- 构建推荐系统的开发者

## ⏱️ 预计学习时间

- **快速浏览**: 2-3 小时
- **深入学习**: 1-2 天
- **完成项目**: 额外 1-2 天

## 📚 内容结构

### Notebook 教程

1. **01_lancedb_introduction.ipynb** - LanceDB 介绍
   - 什么是向量数据库？
   - LanceDB 特性和优势
   - 安装和基本使用
   - 第一个 LanceDB 程序

2. **02_basic_operations.ipynb** - 基础操作
   - 创建数据库和表
   - 插入数据
   - 查询和过滤
   - 更新和删除
   - 索引管理

3. **03_embeddings.ipynb** - 嵌入生成
   - 什么是嵌入（Embeddings）？
   - 使用 sentence-transformers
   - 文本嵌入
   - 多语言支持
   - 嵌入模型选择

4. **04_semantic_search.ipynb** - 语义搜索
   - 向量相似度搜索
   - 混合搜索（向量 + 过滤）
   - 相似商品推荐
   - 问答系统
   - 性能优化

## 🚀 快速开始

### 1. 安装依赖

```bash
cd demo3_lancedb

# 确保已安装基础依赖
pip install -r ../requirements.txt

# 验证安装
python -c "import lancedb; print(f'LanceDB version: {lancedb.__version__}')"
```

### 2. 生成示例数据

```bash
python data/generate_reviews.py --size 50000
```

这将生成：
- `data/reviews.csv` - 50K 条商品评论
- 包含评论文本、评分、商品信息等

### 3. 启动 Jupyter Notebook

```bash
jupyter notebook notebooks/01_lancedb_introduction.ipynb
```

### 4. 按顺序学习

依次打开并运行每个 notebook：
1. 01_lancedb_introduction.ipynb
2. 02_basic_operations.ipynb
3. 03_embeddings.ipynb
4. 04_semantic_search.ipynb

## 💡 核心概念

### 1. 什么是向量数据库？

向量数据库是专门用于存储和查询高维向量的数据库，主要用于：
- **语义搜索**：根据含义而非关键词搜索
- **推荐系统**：找到相似的商品、内容
- **图像搜索**：以图搜图
- **问答系统**：找到最相关的答案

### 2. 什么是嵌入（Embeddings）？

嵌入是将文本、图像等数据转换为高维向量的过程：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# 文本转向量
text = "这个产品质量很好"
embedding = model.encode(text)  # 返回 384 维向量

print(embedding.shape)  # (384,)
```

**相似的文本会有相似的向量**：
- "质量很好" 和 "品质不错" → 向量相似
- "质量很好" 和 "价格便宜" → 向量不太相似

### 3. LanceDB 核心特性

**优势**：
- 🚀 **快速**：高性能的向量搜索
- 💾 **嵌入式**：无需单独服务器
- 🔄 **ACID**：事务保证
- 📊 **多模态**：支持结构化和向量数据
- 🔍 **混合搜索**：向量搜索 + SQL 过滤
- 🐍 **Python 友好**：简单的 API

**架构**：
```
┌─────────────────────────────────┐
│      Python Application         │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│         LanceDB API             │
│  - create_table()               │
│  - search()                     │
│  - add()                        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│      Lance Format Storage       │
│  - Columnar format              │
│  - Vector indexes               │
│  - Metadata                     │
└─────────────────────────────────┘
```

## 📊 示例数据集

### 商品评论数据集（reviews.csv）

**字段说明**：
- `review_id`: 评论唯一标识符
- `product_id`: 商品ID
- `user_id`: 用户ID
- `rating`: 评分 1-5
- `title`: 评论标题
- `text`: 评论内容（主要用于生成嵌入）
- `timestamp`: 评论时间
- `verified_purchase`: 是否验证购买
- `helpful_votes`: 有用投票数

**数据规模**：
- 默认：50,000 条评论
- 大小：约 100MB
- 可自定义大小

## 🔍 关键操作示例

### 1. 创建数据库和表

```python
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

# 连接数据库（如果不存在会自动创建）
db = lancedb.connect("lancedb_data")

# 准备数据
model = SentenceTransformer('all-MiniLM-L6-v2')

data = [
    {
        "id": 1,
        "text": "这个产品质量很好，非常满意",
        "rating": 5,
        "vector": model.encode("这个产品质量很好，非常满意").tolist()
    },
    {
        "id": 2,
        "text": "价格有点贵，但是物有所值",
        "rating": 4,
        "vector": model.encode("价格有点贵，但是物有所值").tolist()
    }
]

# 创建表
table = db.create_table("reviews", data)
```

### 2. 插入数据

```python
# 插入单条
new_review = {
    "id": 3,
    "text": "发货速度很快",
    "rating": 5,
    "vector": model.encode("发货速度很快").tolist()
}
table.add([new_review])

# 批量插入
df = pd.read_csv("data/reviews.csv")
df["vector"] = df["text"].apply(lambda x: model.encode(x).tolist())
table.add(df.to_dict('records'))
```

### 3. 向量搜索

```python
# 语义搜索
query = "质量不错的商品"
query_vector = model.encode(query)

results = (
    table
    .search(query_vector)
    .limit(10)
    .to_list()
)

for result in results:
    print(f"评分: {result['rating']}, 文本: {result['text']}")
```

### 4. 混合搜索（向量 + 过滤）

```python
# 搜索高评分的相似评论
results = (
    table
    .search(query_vector)
    .where("rating >= 4")
    .limit(10)
    .to_list()
)
```

### 5. 更新和删除

```python
# 更新（通过删除再插入）
table.delete("id = 1")
table.add([updated_record])

# 批量删除
table.delete("rating < 3")
```

## 🎯 实战项目：语义搜索引擎

### 项目目标

构建一个商品评论语义搜索系统：
1. 导入 50K 条评论数据
2. 生成文本嵌入
3. 存储到 LanceDB
4. 实现语义搜索 API
5. 支持过滤和排序

### 实现步骤

```python
import lancedb
from sentence_transformers import SentenceTransformer
import pandas as pd

# 1. 初始化
db = lancedb.connect("lancedb_data")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. 加载数据
df = pd.read_csv("data/reviews.csv")

# 3. 生成嵌入
print("生成嵌入...")
df["vector"] = df["text"].apply(lambda x: model.encode(x).tolist())

# 4. 创建表
table = db.create_table("reviews", df.to_dict('records'))

# 5. 搜索函数
def semantic_search(query, min_rating=None, limit=10):
    query_vector = model.encode(query)

    search = table.search(query_vector).limit(limit)

    if min_rating:
        search = search.where(f"rating >= {min_rating}")

    return search.to_pandas()

# 6. 使用
results = semantic_search("质量很好的产品", min_rating=4, limit=5)
print(results[["text", "rating"]])
```

### 性能优化

```python
# 创建 IVF 索引加速搜索
table.create_index(
    metric="cosine",
    num_partitions=256,
    num_sub_vectors=96
)

# 搜索时指定 nprobes
results = (
    table
    .search(query_vector)
    .nprobes(20)
    .limit(10)
    .to_list()
)
```

## 🎓 练习题

### 初级练习

1. **基础操作**
   - 创建 LanceDB 数据库
   - 插入 100 条测试数据
   - 执行简单的向量搜索

2. **嵌入生成**
   - 使用不同的嵌入模型
   - 对比不同模型的效果
   - 测试多语言支持

3. **数据查询**
   - 实现基本的语义搜索
   - 添加评分过滤
   - 按时间排序结果

### 中级练习

4. **混合搜索**
   - 结合向量搜索和属性过滤
   - 实现多条件查询
   - 优化搜索性能

5. **推荐系统**
   - 根据商品找相似商品
   - 根据用户历史推荐
   - 实现协同过滤

6. **数据管理**
   - 实现增量更新
   - 处理重复数据
   - 数据备份和恢复

### 高级练习

7. **性能优化**
   - 创建和调优索引
   - 批量处理优化
   - 内存使用优化

8. **高级应用**
   - 构建问答系统
   - 实现多模态搜索
   - 集成到 Web 应用

9. **生产化**
   - 添加缓存层
   - 实现 API 服务
   - 监控和日志

## 🐛 常见问题

### Q1: 嵌入模型选择

```python
# 中文推荐
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 英文推荐
model = SentenceTransformer('all-MiniLM-L6-v2')

# 高质量（较慢）
model = SentenceTransformer('all-mpnet-base-v2')
```

### Q2: 搜索结果不准确

```python
# 1. 尝试不同的相似度度量
table.search(query_vector, metric="cosine")  # 或 "l2", "dot"

# 2. 调整搜索参数
results = table.search(query_vector).nprobes(50).limit(20)

# 3. 使用更好的嵌入模型
```

### Q3: 内存占用过大

```python
# 批量处理
batch_size = 1000
for i in range(0, len(df), batch_size):
    batch = df[i:i+batch_size]
    batch["vector"] = batch["text"].apply(lambda x: model.encode(x).tolist())
    table.add(batch.to_dict('records'))
```

### Q4: 如何更新已有记录？

```python
# LanceDB 目前不支持直接更新，需要删除后重新插入
table.delete(f"id = {record_id}")
table.add([updated_record])
```

## 📚 参考资源

### 官方文档
- [LanceDB 文档](https://lancedb.github.io/lancedb/)
- [LanceDB Python API](https://lancedb.github.io/lancedb/python/)
- [Lance Format](https://github.com/lancedb/lance)

### 嵌入模型
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Models](https://huggingface.co/models?pipeline_tag=sentence-similarity)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

### 应用案例
- [语义搜索教程](https://lancedb.github.io/lancedb/notebooks/semantic_search/)
- [推荐系统](https://lancedb.github.io/lancedb/notebooks/recommender_system/)
- [问答系统](https://lancedb.github.io/lancedb/notebooks/youtube_transcript_search/)

## ✅ 完成检查清单

完成本 Demo 后，你应该能够：

- [ ] 理解向量数据库的概念和应用
- [ ] 创建和管理 LanceDB 数据库
- [ ] 生成文本嵌入
- [ ] 执行向量相似度搜索
- [ ] 实现混合搜索（向量 + 过滤）
- [ ] 选择合适的嵌入模型
- [ ] 优化搜索性能
- [ ] 构建简单的语义搜索应用
- [ ] 处理常见问题
- [ ] 完成至少 3 个练习题

## 🎯 下一步

完成本 Demo 后，继续学习：

👉 [Demo 4: 综合应用 - 数据清洗管道](../demo4_integrated/) - 整合所有技术构建完整系统

---

**祝学习愉快！** 🚀

如有问题，请查看 [故障排除指南](../docs/troubleshooting.md) 或提交 [Issue](https://github.com/your-username/hello_daft/issues)。
