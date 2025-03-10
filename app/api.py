from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
from model import MicroModel
from app.logger import logger

# 初始化FastAPI应用
app = FastAPI(title="微小语言模型API", description="一个简单的本地部署微小模型的API接口")

# 请求模型
class QueryRequest(BaseModel):
    query: str

# 响应模型
class QueryResponse(BaseModel):
    answer: str
    confidence: float
    original_question: str

# 全局变量存储模型实例
model = None

# 启动事件
@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = MicroModel()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "micro_model.joblib")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}，请先运行train.py训练模型")
            
        # 加载模型
        model.load(model_path)
        print(f"模型已加载，包含 {len(model.questions)} 个问答对")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        model = None

# 根路径
@app.get("/")
async def root():
    return {"message": "欢迎使用微小语言模型API！请使用 /chat 端点进行对话。"}

# 对话端点
@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    global model
    
    # 检查模型是否已加载
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请检查服务器日志")
    
    # 获取查询
    query = request.query
    
    # 预测回答
    try:
        results = model.predict(query)
        
        if not results:
            return QueryResponse(
                answer="抱歉，我不理解您的问题，能否换个方式提问？",
                confidence=0.0,
                original_question=""
            )
        
        # 获取最佳匹配
        best_match = results[0]
        
        return QueryResponse(
            answer=best_match["answer"],
            confidence=best_match["score"],
            original_question=best_match["question"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测过程出错: {str(e)}")

# 健康检查端点
@app.get("/health")
async def health():
    global model
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "questions_count": len(model.questions) if model is not None else 0
    }
