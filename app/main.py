from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.rag import answer

class ChatRequest(BaseModel):
    question: str

app = FastAPI()

# 👇 获取当前 main.py 所在的目录（即 app 目录）
CURRENT_DIR = Path(__file__).parent

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(CURRENT_DIR)), name="static")

@app.get("/")
async def root():
    # 👇 使用绝对路径返回 index.html
    html_path = CURRENT_DIR / "index.html"
    return FileResponse(html_path)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: ChatRequest):
    result = answer(request.question)
    return {"reply": result["answer"]}