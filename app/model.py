import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class MicroModel:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.vectorizer = None
        self.question_vectors = None
        
    def load_training_data(self, data_dir):
        """从指定目录加载训练数据文件"""
        training_files = [f for f in os.listdir(data_dir) if f.startswith('training_data') and f.endswith('.txt')]
        
        for file_name in training_files:
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                qa_pairs = re.split(r'\n\n', content.strip())
                
                for pair in qa_pairs:
                    lines = pair.strip().split('\n')
                    if len(lines) >= 2:
                        question = lines[0].replace('问题: ', '', 1).strip()
                        answer = lines[1].replace('回答: ', '', 1).strip()
                        self.questions.append(question)
                        self.answers.append(answer)
                        
        return len(self.questions)
    
    def train(self):
        """训练模型"""
        if not self.questions:
            raise ValueError("没有训练数据，请先加载训练数据")
            
        # 使用TF-IDF向量化问题
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        
        return True
    
    def predict(self, query, top_n=1):
        """预测用户查询的回答"""
        if not self.vectorizer or self.question_vectors is None:
            raise ValueError("模型未经训练，请先训练模型")
            
        # 向量化查询
        query_vector = self.vectorizer.transform([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        
        # 获取相似度最高的top_n个问题的索引
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0:  # 只返回有相似度的结果
                results.append({
                    'question': self.questions[idx],
                    'answer': self.answers[idx],
                    'score': float(score)
                })
        
        return results
    
    def save(self, model_path):
        """保存模型到指定路径"""
        model_data = {
            'questions': self.questions,
            'answers': self.answers,
            'vectorizer': self.vectorizer,
            'question_vectors': self.question_vectors
        }
        joblib.dump(model_data, model_path)
        return True
    
    def load(self, model_path):
        """从指定路径加载模型"""
        model_data = joblib.load(model_path)
        self.questions = model_data['questions']
        self.answers = model_data['answers']
        self.vectorizer = model_data['vectorizer']
        self.question_vectors = model_data['question_vectors']
        return True
