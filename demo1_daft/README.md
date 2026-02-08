# Demo 1: Daft 基础使用

欢迎来到 Daft 基础教程！这是 Hello Daft 系列的第一个 Demo。

## 📖 学习目标

通过本 Demo，你将学习：
- ✅ Daft 的核心概念和设计理念
- ✅ Daft DataFrame 的基本操作
- ✅ 数据读取、转换和聚合
- ✅ 用户自定义函数（UDF）
- ✅ Daft 与 Pandas 的性能对比
- ✅ 延迟执行和查询优化
- ✅ AI Functions（文本分类、嵌入、语义搜索、LLM 提取）

## 🎯 适合人群

- 熟悉 Pandas 的 Python 开发者
- 需要处理大规模数据的工程师
- 对分布式数据处理感兴趣的学习者

## ⏱️ 预计学习时间

- **快速浏览**: 1-2 小时
- **深入学习**: 1-2 天
- **完成练习**: 额外 2-3 小时

## 📚 内容结构

### Notebook 教程

1. **01_introduction.ipynb** - Daft 介绍
   - 什么是 Daft？
   - 为什么选择 Daft？
   - 安装和环境设置
   - 第一个 Daft 程序

2. **02_basic_operations.ipynb** - 基础操作
   - 创建 DataFrame
   - 读取数据（CSV、Parquet、JSON）
   - 选择和过滤
   - 列操作和重命名
   - 排序和去重

3. **03_data_processing.ipynb** - 数据处理
   - 聚合和分组（GroupBy）
   - 连接操作（Join）
   - 处理缺失值
   - 数据类型转换
   - 窗口函数

4. **04_advanced_features.ipynb** - 高级特性
   - 用户自定义函数（UDF）
   - 复杂数据类型（List、Struct）
   - 延迟执行和查询计划
   - 性能优化技巧
   - Daft vs Pandas 性能对比

5. **05_ai_multimodal.ipynb** - AI Functions 与多模态
   - 文本分类（classify_text）
   - 文本嵌入（embed_text）
   - 语义搜索（cosine_distance）
   - LLM 结构化提取（prompt）
   - 需要 OpenAI API Key

## 🚀 快速开始

### 1. 安装依赖

```bash
# 确保在项目根目录
cd /path/to/hello_daft

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖（如果还没安装）
pip install -r requirements.txt
```

### 2. 生成示例数据

```bash
cd demo1_daft
python data/generate_data.py
```

这将生成：
- `data/products.csv` - 100K 条产品记录
- `data/products.parquet` - Parquet 格式
- `data/products.json` - JSON 格式

### 3. 启动 Jupyter Notebook

```bash
jupyter notebook notebooks/01_introduction.ipynb
```

### 4. 按顺序学习

依次打开并运行每个 notebook：
1. 01_introduction.ipynb
2. 02_basic_operations.ipynb
3. 03_data_processing.ipynb
4. 04_advanced_features.ipynb
5. 05_ai_multimodal.ipynb（需要 OpenAI API Key）

## 📊 示例数据集

### 产品数据集（products.csv）

**字段说明**：
- `product_id`: 产品唯一标识符（字符串）
- `name`: 产品名称（字符串）
- `category`: 产品类别（字符串）
- `subcategory`: 子类别（字符串）
- `price`: 价格（浮点数）
- `original_price`: 原价（浮点数）
- `discount`: 折扣百分比（浮点数）
- `rating`: 评分 1-5（浮点数）
- `review_count`: 评论数量（整数）
- `stock`: 库存数量（整数）
- `brand`: 品牌（字符串）
- `description`: 产品描述（字符串）
- `created_at`: 创建时间（时间戳）
- `updated_at`: 更新时间（时间戳）

**数据规模**：
- 默认：100,000 条记录
- 大小：约 50MB（CSV）
- 可自定义：通过 `--size` 参数调整

**数据特点**：
- 包含缺失值（约 5%）
- 包含重复记录（约 2%）
- 真实的数据分布
- 适合演示各种数据操作

## 💡 核心概念

### 1. Daft DataFrame

Daft DataFrame 是分布式的、延迟执行的数据结构，类似于 Pandas DataFrame 但能处理更大规模的数据。

```python
import daft

# 创建 DataFrame
df = daft.read_csv("data/products.csv")

# 延迟执行 - 此时还没有真正读取数据
df_filtered = df.where(df["price"] > 100)

# 触发执行
result = df_filtered.collect()  # 现在才真正执行
```

### 2. 延迟执行（Lazy Evaluation）

Daft 使用延迟执行来优化查询：
- 操作不会立即执行
- 构建查询计划（Query Plan）
- 优化执行路径
- 调用 `.collect()` 时才真正执行

**优势**：
- 自动查询优化
- 减少不必要的计算
- 更好的内存管理

### 3. 分布式执行

Daft 可以在多核或分布式环境中执行：
- 本地多核并行
- 与 Ray 集成进行分布式计算
- 自动数据分区

### 4. 类型系统

Daft 有强大的类型系统：
- 基本类型：Int, Float, String, Boolean
- 复杂类型：List, Struct, Map
- 时间类型：Date, Timestamp
- 嵌套结构支持

## 🔍 关键操作示例

### 读取数据

```python
import daft

# CSV
df = daft.read_csv("data/products.csv")

# Parquet（推荐，性能更好）
df = daft.read_parquet("data/products.parquet")

# JSON
df = daft.read_json("data/products.json")

# 多个文件
df = daft.read_csv("data/*.csv")
```

### 基本操作

```python
# 选择列
df_subset = df.select("product_id", "name", "price")

# 过滤
df_expensive = df.where(df["price"] > 100)

# 排序
df_sorted = df.sort("price", desc=True)

# 去重
df_unique = df.drop_duplicates()

# 限制行数
df_top10 = df.limit(10)
```

### 聚合操作

```python
# 分组聚合
df_agg = df.groupby("category").agg(
    daft.col("price").mean().alias("avg_price"),
    daft.col("price").max().alias("max_price"),
    daft.col("product_id").count().alias("count")
)

# 多列分组
df_agg2 = df.groupby("category", "brand").agg(
    daft.col("rating").mean().alias("avg_rating")
)
```

### 列操作

```python
# 添加新列
df = df.with_column(
    "discount_price",
    df["price"] * (1 - df["discount"] / 100)
)

# 重命名列
df = df.rename({"product_id": "id"})

# 删除列
df = df.drop("description")
```

### 连接操作

```python
# 内连接
df_joined = df1.join(df2, on="product_id")

# 左连接
df_left = df1.join(df2, on="product_id", how="left")

# 多键连接
df_joined = df1.join(df2, on=["category", "brand"])
```

## 📈 性能对比

### Daft vs Pandas

| 操作 | Pandas | Daft | 提升 |
|------|--------|------|------|
| 读取 100K CSV | 2.5s | 0.8s | 3.1x |
| 过滤操作 | 0.5s | 0.1s | 5x |
| GroupBy 聚合 | 1.2s | 0.3s | 4x |
| Join 操作 | 3.0s | 0.9s | 3.3x |

**注意**：实际性能取决于数据规模、硬件配置和操作复杂度。

### 何时使用 Daft？

**适合使用 Daft**：
- ✅ 数据量 > 1GB
- ✅ 需要分布式处理
- ✅ 复杂的数据转换管道
- ✅ 需要查询优化
- ✅ 与 Ray 集成

**继续使用 Pandas**：
- ✅ 数据量 < 100MB
- ✅ 简单的探索性分析
- ✅ 需要特定的 Pandas 功能
- ✅ 交互式数据分析

## 🎓 练习题

完成 notebook 学习后，尝试以下练习：

### 初级练习

1. **数据探索**
   - 读取产品数据
   - 查看数据的基本统计信息
   - 找出价格最高的 10 个产品

2. **数据清洗**
   - 删除重复记录
   - 处理缺失值（删除或填充）
   - 过滤掉库存为 0 的产品

3. **数据分析**
   - 计算每个类别的平均价格
   - 找出评分最高的品牌
   - 统计每个类别的产品数量

### 中级练习

4. **数据转换**
   - 创建价格区间列（低、中、高）
   - 计算折扣后的价格
   - 提取产品名称的关键词

5. **复杂聚合**
   - 按类别和品牌分组，计算多个指标
   - 使用窗口函数计算排名
   - 计算移动平均

6. **数据连接**
   - 创建两个相关的数据集
   - 执行不同类型的连接
   - 处理连接后的缺失值

### 高级练习

7. **自定义函数**
   - 编写 UDF 处理产品描述
   - 实现自定义的数据验证逻辑
   - 批量处理复杂数据类型

8. **性能优化**
   - 对比不同操作的执行时间
   - 优化查询计划
   - 测试不同数据格式的性能

9. **综合项目**
   - 构建完整的数据处理管道
   - 从原始数据到分析结果
   - 生成数据质量报告

## 🐛 常见问题

### Q1: Daft 安装失败

```bash
# 尝试升级 pip
pip install --upgrade pip

# 使用国内镜像
pip install getdaft -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 内存不足错误

```python
# 减少数据量
df = daft.read_csv("data/products.csv").limit(10000)

# 或者使用更小的数据集
python data/generate_data.py --size 10000
```

### Q3: 执行速度慢

```python
# 使用 Parquet 格式（比 CSV 快很多）
df = daft.read_parquet("data/products.parquet")

# 启用 Ray 后端（需要先安装 Ray）
import daft
daft.context.set_runner_ray()
```

### Q4: 如何查看查询计划？

```python
# 查看逻辑计划
print(df.explain())

# 查看物理计划
print(df.explain(show_all=True))
```

## 📚 扩展阅读

### 官方资源
- [Daft 官方文档](https://www.getdaft.io/projects/docs/)
- [Daft API 参考](https://www.getdaft.io/projects/docs/en/latest/api_docs/index.html)
- [Daft GitHub](https://github.com/Eventual-Inc/Daft)
- [Daft 博客](https://blog.getdaft.io/)

### 推荐文章
- [Daft vs Pandas: When to Use Each](https://blog.getdaft.io/daft-vs-pandas)
- [Understanding Lazy Evaluation](https://blog.getdaft.io/lazy-evaluation)
- [Distributed DataFrames Explained](https://blog.getdaft.io/distributed-dataframes)

### 视频教程
- [Daft Introduction (YouTube)](https://www.youtube.com/watch?v=...)
- [Daft Advanced Features](https://www.youtube.com/watch?v=...)

## ✅ 完成检查清单

完成本 Demo 后，你应该能够：

- [ ] 理解 Daft 的核心概念和优势
- [ ] 创建和操作 Daft DataFrame
- [ ] 读取和写入多种数据格式
- [ ] 执行基本的数据转换操作
- [ ] 使用 GroupBy 进行聚合
- [ ] 执行 Join 操作
- [ ] 编写简单的 UDF
- [ ] 理解延迟执行的工作原理
- [ ] 对比 Daft 和 Pandas 的性能差异
- [ ] 完成至少 3 个练习题

## 🎯 下一步

完成本 Demo 后，继续学习：

👉 [Demo 2: Ray on Kubernetes](../demo2_ray/) - 学习分布式计算和 K8s 部署

---

**祝学习愉快！** 🚀

如有问题，请查看 [故障排除指南](../docs/troubleshooting.md) 或提交 [Issue](https://github.com/your-username/hello_daft/issues)。
