# Demo 4: AI Platform — 单服务 ML 平台

基于 Daft + Lance + PyTorch 构建的教学级 AI 平台，演示数据入库、模型训练、推理服务的完整流程。

## 架构

```
用户 --> Server (8000) --> Lance 数据湖
```

Server 统一提供数据湖存储（Lance）、脚本执行（Daft）和 RESTful API。

## 快速开始

```bash
# 1. 安装依赖
pip install -r demo4_ai_platform/requirements.txt

# 2. 启动服务
cd demo4_ai_platform
uvicorn ai_platform.server.app:app --port 8000

# 3. 打开 Notebook 教程
jupyter notebook notebooks/01_level1_tutorial.ipynb
```

## 内容

| 内容 | 说明 |
|------|------|
| [Level 1 教程](notebooks/01_level1_tutorial.ipynb) | 端到端演示：数据入库 → 训练 → 推理 → 可视化 |
| [Web Demo](mnist/index.html) | 手写数字识别页面（Canvas 手写 + API 调用） |
| [设计文档](ai_platform/design.md) | 架构设计、设计决策、API 定义、部署级别 |

## Web Demo 使用

```bash
# 1. 确保服务已启动（见上方）
# 2. 创建推理任务
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"name":"mnist_serve","input":"lance_storage/models/mnist_cnn_v1.lance","script":"mnist/mnist_serve.py","output":"","params":{"device":"cpu","port":8080}}'

# 3. 用浏览器打开 mnist/index.html
# 4. 配置区填入推理服务地址（默认 http://localhost:8080）
```

## 目录结构

```
demo4_ai_platform/
├── README.md                    # 教学入口
├── requirements.txt
├── notebooks/                   # 教学 notebook
│   └── 01_level1_tutorial.ipynb
├── ai_platform/                 # 平台源码
│   ├── design.md               # 设计文档
│   ├── server/                 # Server 服务
│   │   ├── app.py              # HTTP API
│   │   ├── storage.py          # Lance 存储封装
│   │   ├── runner.py           # Runner 基类 + 工厂函数
│   │   └── runners/
│   │       ├── local.py        # Level 1: 线程执行
│   │       └── ray.py          # Level 2/3: Ray Task 执行
│   └── tests/unit/             # 单元测试
├── mnist/                       # 用户脚本 + Web Demo（平台用户编写）
│   ├── mnist_clean.py          # MNIST 数据清洗
│   ├── mnist_cnn.py            # CNN 训练
│   ├── mnist_serve.py          # 推理服务
│   └── index.html              # 手写数字识别 Web Demo
└── lance_storage/               # 运行时数据（gitignored）
```
