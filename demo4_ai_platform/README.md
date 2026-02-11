# Demo 4: AI Platform — 微服务架构的 ML 平台

基于 Daft + Lance + PyTorch 构建的教学级 AI 平台，演示数据入库、模型训练、推理服务的完整流程。

## 架构

```
用户 --> Orchestrator (8000) --> Executor (8001) --> Lance 数据湖
```

| 服务 | 职责 |
|------|------|
| **Orchestrator** | 用户唯一入口，任务编排、代理查询 |
| **Executor** | 数据湖存储（Lance）+ 脚本执行器（Daft） |

## 快速开始

```bash
# 1. 安装依赖
pip install -r demo4_ai_platform/requirements.txt

# 2. 启动服务
cd demo4_ai_platform
uvicorn executor.app:app --port 8001 &
uvicorn orchestrator.app:app --port 8000 &

# 3. 打开 Notebook 教程
jupyter notebook 01_ai_platform_tutorial.ipynb
```

## 内容

| 内容 | 说明 |
|------|------|
| [Notebook 教程](01_ai_platform_tutorial.ipynb) | 端到端演示：数据入库 → 训练 → 推理 → 可视化 |
| [Web Demo](web/index.html) | 手写数字识别页面（Canvas 手写 + API 调用） |
| [设计文档](design.md) | 架构设计、API 定义、部署级别、任务状态机 |

## Web Demo 使用

```bash
# 1. 确保服务已启动（见上方）
# 2. 创建推理任务
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"name":"mnist_serve","input":"lance_storage/models/mnist_cnn_v1.lance","script":"scripts/serving/mnist_serve.py","output":"","params":{"device":"cpu","port":8080}}'

# 3. 用浏览器打开 web/index.html
# 4. 配置区填入推理服务地址（默认 http://localhost:8080）
```

## 目录结构

```
demo4_ai_platform/
├── design.md                    # 设计文档
├── config.yaml                  # 配置文件
├── requirements.txt
├── 01_ai_platform_tutorial.ipynb
├── web/
│   └── index.html               # 手写数字识别 Web Demo
├── executor/                    # Executor 微服务
│   ├── app.py                   # HTTP API
│   ├── storage.py               # Lance 存储封装
│   └── runner.py                # 脚本执行器
├── orchestrator/                # Orchestrator 微服务
│   ├── app.py                   # HTTP API
│   └── tasks.py                 # 统一任务管理
├── scripts/                     # 用户脚本
│   ├── pipelines/mnist_clean.py # MNIST 数据清洗
│   ├── training/mnist_cnn.py    # CNN 训练
│   └── serving/mnist_serve.py   # 推理服务
└── tests/unit/                  # 单元测试
```
