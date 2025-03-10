import os
from model import MicroModel

def train_model():
    # 创建模型实例
    model = MicroModel()
    
    # 获取当前目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载训练数据
    num_samples = model.load_training_data(current_dir)
    print(f"已加载 {num_samples} 个问答对")
    
    # 训练模型
    model.train()
    print("模型训练完成")
    
    # 保存模型
    model_path = os.path.join(current_dir, "micro_model.joblib")
    model.save(model_path)
    print(f"模型已保存到 {model_path}")
    
    # 简单测试
    test_queries = [
        "你好啊",
        "今天的天气",
        "推荐一本好书",
        "如何学习编程"
    ]
    
    print("\n模型测试:")
    for query in test_queries:
        results = model.predict(query)
        if results:
            print(f"查询: {query}")
            print(f"最佳回答: {results[0]['answer']}")
            print(f"相似度分数: {results[0]['score']:.4f}")
            print("-" * 50)
        else:
            print(f"查询: {query} - 找不到匹配的回答")
            print("-" * 50)

if __name__ == "__main__":
    train_model()
