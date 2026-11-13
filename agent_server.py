# -*- coding: utf-8 -*-
"""
agent_server.py —— Agent 独立服务（W8D6）

设计原则：
    ⚠️ 完全不动 api_service.py（线上已跑稳定）
    Agent 用自己的端口（8001），通过 HTTP 调用 8000 的 RAG

架构：
    浏览器 ──→ :8001 (agent_server) ──→ :8000 (api_service / RAG)

端点：
    GET  /health           → Agent 自身健康
    POST /agent            → 跑 Agent（返回完整结果）
    POST /agent/stream     → 跑 Agent（SSE 流式，可选）
    GET  /                 → Agent 前端页面

启动：
    uvicorn agent_server:app --host 0.0.0.0 --port 8001

环境变量：
    RAG_BASE     RAG 服务地址（默认 http://127.0.0.1:8000）
    AGENT_PORT   端口（默认 8001）
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
AGENT_DIR = BASE / "agent"
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(BASE))

# 默认让 Agent 调本地 RAG；线上部署时改环境变量
os.environ.setdefault("RAG_BASE", "http://127.0.0.1:8000")

from main_agent import TOOLS_MODE, run_agent  # noqa: E402

START_TIME = time.time()

app = FastAPI(
    title="河南专升本 Agent API",
    version="1.0.0",
    description="LangGraph Agent：路由决策 + RAG 工具调用 + 降级容错",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class AgentReq(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = None


@app.get("/health")
def health():
    """Agent 自身健康 + RAG 依赖状态"""
    from tools_robust import health_check
    rag = health_check()
    return {
        "status": "healthy" if rag["ok"] else "degraded",
        "agent": "up",
        "tools_mode": TOOLS_MODE,
        "rag_base": os.environ.get("RAG_BASE"),
        "rag_status": rag.get("status"),
        "rag_chunks": rag.get("chunks_count", 0),
        "uptime_seconds": int(time.time() - START_TIME),
    }


@app.post("/agent")
def agent_endpoint(req: AgentReq):
    """
    跑一次 Agent。

    返回：
        {
            answer, sources, route, route_reason,
            trace, elapsed_ms, degraded, degrade_reason
        }
    """
    try:
        return run_agent(req.question)
    except Exception as e:
        print(f"[Agent异常] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 执行失败：{e}")


@app.get("/", include_in_schema=False)
def index():
    """Agent 前端页面"""
    for name in ("index_agent.html", "docs/index_agent.html"):
        p = BASE / name
        if p.exists():
            return FileResponse(str(p), media_type="text/html; charset=utf-8")
    return HTMLResponse(
        "<h1>Agent 前端缺失</h1><p>请把 index_agent.html 放到项目根目录</p>"
        "<p>API 可直用：<code>POST /agent</code>，文档在 <a href='/docs'>/docs</a></p>",
        status_code=404,
    )


@app.on_event("startup")
def _startup():
    print("=" * 62)
    print(f"[Agent 启动] 工具模式={TOOLS_MODE}")
    print(f"[Agent 依赖] RAG_BASE={os.environ.get('RAG_BASE')}")
    try:
        from tools_robust import health_check
        h = health_check()
        print(f"[Agent 依赖] RAG 状态={h.get('status')} | {h.get('chunks_count', 0)} 块")
    except Exception as e:
        print(f"[Agent 依赖] 检查失败：{e}")
    print("=" * 62)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("AGENT_PORT", "8001")))
