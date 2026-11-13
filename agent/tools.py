# -*- coding: utf-8 -*-
"""
tools.py —— 把 RAG 包成 Agent 可调用的工具（W8D2）

设计原则：
    1. 用 HTTP 调用，不直接 import —— 见文件末尾"自检题答案"
    2. 每个工具只做一件事，返回结构化 dict
    3. 统一异常处理：失败返回 {"ok": False, "error": ...}，不抛异常
    4. 支持本地和远程两种后端

环境变量：
    RAG_BASE   RAG 服务地址（默认本地 8000，可改为公网）

用法：
    python agent/tools.py         # 自测 3 个工具
"""
import os
import time
from typing import Optional

import requests

# 后端地址：优先环境变量，默认本地
BASE = os.environ.get("RAG_BASE", "http://127.0.0.1:8000").rstrip("/")

# 超时配置（秒）：(连接超时, 读取超时)
# ⚠️ 跨境到香港服务器，TCP 连接要 2-5 秒，3 秒会误判"不可达"
TIMEOUT_SEARCH = (10, 25)
TIMEOUT_CHAT = (10, 100)
TIMEOUT_HEALTH = (10, 15)


# ============================================================
# 工具 1：search_docs —— 纯检索，不生成
# ============================================================
def search_docs(query: str, top_k: int = 5, filters: Optional[dict] = None) -> dict:
    """
    检索知识库，返回相关文档片段（不调用 LLM 生成）。

    适用场景：
        - 需要给用户看原文列表
        - 想让 Agent 自己判断哪些片段有用
        - 省成本（不调 LLM）

    返回：
        {"ok": True, "hits": [...], "total": N, "latency_ms": X}
        {"ok": False, "error": "..."}
    """
    payload = {"query": query, "top_k": top_k}
    if filters:
        payload["filters"] = filters

    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/search", json=payload, timeout=TIMEOUT_SEARCH)
        r.raise_for_status()
        d = r.json()
        return {
            "ok": True,
            "hits": d.get("hits", []),
            "total": d.get("total", 0),
            "filters_applied": d.get("filters_applied", []),
            "fallback_used": d.get("fallback_used", False),
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except requests.Timeout:
        return {"ok": False, "error": "检索超时"}
    except requests.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.response.status_code if e.response else '?'}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ============================================================
# 工具 2：ask_rag —— 检索 + 生成
# ============================================================
def ask_rag(question: str, mode: str = "fast") -> dict:
    """
    问答（检索 + LLM 生成），返回答案与来源。

    适用场景：
        - 需要一个完整的自然语言答案
        - 直接给用户看

    注意：
        - 这是最慢的工具（LLM 生成占 90%）
        - 一次 ~5-7 秒

    返回：
        {"ok": True, "answer": "...", "sources": [...], "refused": bool, "latency_ms": X}
        {"ok": False, "error": "..."}
    """
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/chat",
            json={"question": question, "mode": mode},
            timeout=TIMEOUT_CHAT,
        )
        r.raise_for_status()
        d = r.json()
        return {
            "ok": True,
            "answer": d.get("answer", ""),
            "sources": d.get("sources", []),
            "refused": d.get("refused", False),
            "mode_used": d.get("mode_used", mode),
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except requests.Timeout:
        return {"ok": False, "error": "问答超时（>90s）"}
    except requests.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.response.status_code if e.response else '?'}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ============================================================
# 工具 3：health_check —— 服务可用性
# ============================================================
def health_check() -> dict:
    """
    检查 RAG 服务是否可用。

    适用场景：
        - Agent 执行前先确认依赖可用
        - 失败时降级（不调用其他工具，直接提示）

    返回：
        {"ok": True, "status": "healthy", "chunks_count": 142, ...}
        {"ok": False, "error": "..."}
    """
    try:
        r = requests.get(f"{BASE}/health", timeout=TIMEOUT_HEALTH)
        r.raise_for_status()
        d = r.json()
        return {
            "ok": d.get("status") == "healthy",
            "status": d.get("status"),
            "chunks_count": d.get("chunks_count", 0),
            "same_source": d.get("same_source"),
            "uptime_seconds": d.get("uptime_seconds", 0),
        }
    except Exception as e:
        return {"ok": False, "status": "unreachable", "error": f"{type(e).__name__}: {e}"}


# ============================================================
# LangChain Tool 封装（可选，Agent 自动决策时用）
# ============================================================
def get_langchain_tools():
    """
    把 3 个函数包成 LangChain Tool。
    Agent 会读取 description 来判断何时调用。

    注意：只有 W9-W11 做"LLM 自主决策"时才需要。
    W8 的规则式路由可以直接调上面的普通函数。
    """
    try:
        from langchain_core.tools import tool
    except ImportError:
        return None

    @tool
    def search_policy_docs(query: str) -> str:
        """检索河南专升本政策知识库。当你需要查看政策原文、需要列出多个院校/专业/分数线时使用。输入查询语句，返回相关文档片段。"""
        r = search_docs(query, top_k=5)
        if not r["ok"]:
            return f"检索失败：{r['error']}"
        if not r["hits"]:
            return "未检索到相关文档"
        lines = []
        for i, h in enumerate(r["hits"][:5], 1):
            lines.append(f"[{i}] {h.get('source', '-')} | {h.get('chunk_id', '-')}\n{h.get('snippet', '')[:200]}")
        return "\n\n".join(lines)

    @tool
    def ask_policy_question(question: str) -> str:
        """回答河南专升本政策问题（检索+生成）。当你需要一个直接可用的完整答案时使用。输入完整问句。"""
        r = ask_rag(question, mode="fast")
        if not r["ok"]:
            return f"问答失败：{r['error']}"
        if r["refused"]:
            return f"未找到相关信息。{r['answer']}"
        src = ", ".join(s.get("chunk_id", "") for s in r["sources"][:3])
        return f"{r['answer']}\n\n[来源: {src}]"

    @tool
    def check_rag_status() -> str:
        """检查政策问答服务是否正常。在开始查询前或怀疑服务异常时使用。"""
        r = health_check()
        if r["ok"]:
            return f"服务正常，知识库 {r['chunks_count']} 块"
        return f"服务异常：{r.get('error') or r.get('status')}"

    return [search_policy_docs, ask_policy_question, check_rag_status]


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    print(f"[后端] {BASE}")
    print("=" * 70)

    print("\n【工具 1】health_check()")
    h = health_check()
    print(f"  ok={h['ok']} status={h.get('status')} chunks={h.get('chunks_count')} same_source={h.get('same_source')}")
    if not h["ok"]:
        print(f"  error: {h.get('error')}")
        print("\n⚠️ 服务不可达，请先启动：")
        print("    本地：cd D:\\vscode\\ls-rag && venv\\Scripts\\python.exe -m uvicorn api_service:app --port 8000")
        print("    或改环境变量指向公网：$env:RAG_BASE='http://43.128.31.138'")
        raise SystemExit(1)

    print("\n【工具 2】search_docs('2026年录取控制分数线', top_k=3)")
    s = search_docs("2026年录取控制分数线", top_k=3)
    print(f"  ok={s['ok']} total={s.get('total')} latency={s.get('latency_ms')}ms")
    print(f"  过滤条件: {s.get('filters_applied')}")
    for i, hit in enumerate(s.get("hits", [])[:3], 1):
        print(f"  [{i}] {hit['chunk_id']} [{hit['doc_type']}] score={hit['score']}")
        print(f"      {hit['text'][:60]}...")

    print("\n【工具 3】ask_rag('河南专升本考试考哪几科？')")
    a = ask_rag("河南专升本考试考哪几科？")
    print(f"  ok={a['ok']} refused={a.get('refused')} latency={a.get('latency_ms')}ms")
    print(f"  answer: {a.get('answer', '')[:100]}...")
    print(f"  sources: {len(a.get('sources', []))} 条")

    print("\n【测试拒答】ask_rag('英语怎么从60提到100？')")
    a2 = ask_rag("英语怎么从60提到100？")
    print(f"  ok={a2['ok']} refused={a2.get('refused')}")
    print(f"  answer: {a2.get('answer', '')[:80]}")

    print("\n【LangChain Tool 封装】")
    tools = get_langchain_tools()
    if tools:
        print(f"  ✅ 已封装 {len(tools)} 个 tool：")
        for t in tools:
            print(f"     - {t.name}: {t.description[:50]}...")
    else:
        print("  ⚠️ langchain_core 未安装（W9 需要时再装）")

    print("\n" + "=" * 70)
    print("自检：为什么用 HTTP 调用而不是直接 import 函数？")
    print("=" * 70)
    print("""
1. 可独立部署：RAG 在服务器上，Agent 可以在本地/另一台机器
2. 可复用：其他服务（前端、其他 Agent）也能调同一套接口
3. 不改冻结代码：RAG 已冻结，通过 HTTP 调用不会误改它
4. 可观测：能看到完整的请求/响应，便于调试和监控
5. 可替换：换实现（比如换向量库）不影响 Agent

代价：
- 网络往返开销（本地 ~1ms，公网 ~50-100ms）
- 需要处理网络异常（超时、断连）
- 比函数调用多一层序列化/反序列化
""")
