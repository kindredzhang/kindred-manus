import asyncio
import os
import subprocess
import sys
from app.logger import logger

async def train_model():
    """训练微小语言模型"""
    logger.info("开始训练微小语言模型")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
    train_script = os.path.join(model_dir, "train.py")
    
    try:
        result = subprocess.run([sys.executable, train_script], 
                              cwd=model_dir,
                              capture_output=True,
                              text=True,
                              check=True)
        logger.info(f"模型训练成功:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"模型训练失败: {str(e)}\n{e.stderr}")
        return False


async def start_api_server():
    """启动API服务器"""
    logger.info("启动API服务器")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    api_script = os.path.join(model_dir, "api.py")
    
    # 使用uvicorn启动API服务器
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "model.api:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    try:
        # 使用非阻塞方式启动服务器
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 读取一些初始输出确认服务器启动
        for _ in range(10):  # 读取前10行或直到没有更多输出
            line = process.stdout.readline()
            if not line:
                break
            logger.info(f"API服务器: {line.strip()}")
            if "Application startup complete" in line:
                logger.info("API服务器已成功启动")
                break
                
        return process
    except Exception as e:
        logger.error(f"启动API服务器失败: {str(e)}")
        return None


async def main():
    logger.info("启动应用程序")
    
    # 训练模型
    train_success = await train_model()
    if not train_success:
        logger.error("模型训练失败，无法继续")
        return
    
    # 启动API服务器
    server_process = await start_api_server()
    if server_process is None:
        logger.error("API服务器启动失败，无法继续")
        return
    
    logger.info("应用程序已完全启动")
    logger.info("可以访问 http://localhost:8000/docs 查看API文档并测试对话功能")
    
    # 保持主程序运行
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("接收到终止信号，正在关闭API服务器...")
        server_process.terminate()
        server_process.wait()
        logger.info("API服务器已关闭，程序结束")


if __name__ == "__main__":
    asyncio.run(main())
    