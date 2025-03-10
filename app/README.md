# 微小语言模型部署指南

这个项目展示了如何在本地部署一个极微小的语言模型，并通过API接口提供对话功能。

## 文件结构

- `model.py`: 微小模型的核心实现
- `train.py`: 用于训练模型的脚本
- `api.py`: 提供REST API接口的服务
- `training_data1.txt`和`training_data2.txt`: 用于训练模型的示例数据
- `micro_model.joblib`: 训练好的模型文件（运行train.py后生成）

## 使用步骤

### 1. 安装依赖

首先，安装所需的依赖库：

```bash
pip install -r requirements.txt
```

### 2. 训练模型

在使用API之前，需要先训练模型：

```bash
cd model
python train.py
```

这将从训练数据文件加载问答对，训练模型并保存为`micro_model.joblib`文件。

### 3. 启动API服务

训练完成后，启动API服务：

```bash
cd model
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 4. 使用API

启动服务后，可以通过以下方式使用API：

- 浏览器访问：[http://localhost:8000/docs](http://localhost:8000/docs) 查看API文档
- 使用curl或其他工具发送POST请求：

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "今天天气怎么样"}'
```

## 自定义训练数据

你可以通过编辑现有的训练数据文件或创建新的训练数据文件来扩展模型的能力。每个训练数据文件应遵循以下格式：

```text
问题: 你的问题
回答: 你的回答

问题: 另一个问题
回答: 另一个回答
```

添加完新的训练数据后，重新运行`train.py`即可更新模型。

## 注意事项

- 这是一个极简的模型，适合学习和理解基本流程，不适合生产环境
- 模型基于TF-IDF和余弦相似度，只能找到与训练数据中最相似的问题并返回对应答案
- 可以通过增加训练数据和改进算法来提升模型性能
