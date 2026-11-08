#!/usr/bin/env python
# -*- coding: utf-8 -*-
# api_service.py —— RAG API 服务（W6D5 + W6D6 最终版）
"""
用法：
    uvicorn api_service:app --port 8000

端点：
    GET  /health   健康检查 + 同源校验
    POST /search   纯检索（不生成）
    POST /chat     对话问答（fast / high 两种模式）

设计要点：
    1. 复用 docs/eval/eval_metadata_filter.py 的 infer_filters / qdrant_hits
    2. 启动时强制同源校验：Qdrant 点数 == 文件块数，不一致则降级
    3. chunk_id 全程透传，每条结果带 in_corpus 标记
    4. 拒答检测只看开头（避免末尾"建议咨询"误判）
    5. LLM 强制 max_tokens，控制延迟
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

load_dotenv()

# ---------------- 复用 eval 的过滤逻辑 ----------------
BASE = Path(__file__).resolve().parent
EVAL_DIR = BASE / "docs" / "eval"
sys.path.insert(0, str(EVAL_DIR))
try:
    from eval_metadata_filter import EMBED_MODEL, EMBED_URL, infer_filters, qdrant_hits
except ImportError as e:
    sys.exit(f"[错误] 无法导入 eval_metadata_filter：{e}")

EMBED_DIM = 1024   # BGE-M3 维度（eval 里未导出）

# ---------------- 配置 ----------------
CHUNKS_PATH = BASE / "docs" / "chunks" / "chunks_fixed_v2.jsonl"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION", "chunks")
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

LLM_URL = "https://api.siliconflow.cn/v1/chat/completions"
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek-ai/DeepSeek-V3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

CAND_K = 20              # high 模式多路召回数
MAX_TOKENS = 600         # 限制生成长度（控制延迟与成本）
CTX_BLOCKS = 5           # 喂给 LLM 的块数（必须 5，减到 3 会丢答案）
RERANK_DOC_CHARS = 800   # rerank 每个文档截断长度

# 向量相似度阈值（fast 模式，低于此认为不相关）
VECTOR_MIN_SCORE = 0.25
# LLM 拒答但 top1 分数高于此值时强制重试（提高阈值，避免延迟翻倍）
RETRY_MIN_SCORE = 0.60
FORCE_RETRY = os.environ.get("FORCE_RETRY", "1") == "1"

START_TIME = time.time()

# ---------------- 拒答检测 ----------------
# 三级词表
HARD_REFUSE = ["未找到相关信息", "没有相关信息", "无法回答", "暂未找到",
               "无法确定", "无法提供", "不清楚", "无法判断", "超出范围",
               "材料中没有", "材料未提供", "材料里没有", "无法获取"]

SOFT_MARKERS = ["未明确", "并未明确", "没有明确", "未具体说明", "未列出",
                "未提及", "材料未涉及", "未涵盖", "未说明", "不包含", "没有提供"]

# 软标记正则（允许中间插入"直接/具体/详细"等修饰词）
SOFT_PATTERNS = [
    r"(并未|没有|未能|未能直接|未)[^，。；]{0,6}(明确|列出|提及|说明|涉及|提供|涵盖)",
    r"(材料|资料|文中)[^，。；]{0,8}(没有|未|无)[^，。；]{0,4}(相关|具体|明确|相应)",
    r"并未[^，。；]{0,4}(列出|明确|给出|提供)",
]

GUIDE_WORDS = ["建议咨询", "请咨询", "需咨询", "以官方为准", "通过其他渠道",
               "咨询报考院校", "咨询当地"]

REFUSE_WORDS = HARD_REFUSE + SOFT_MARKERS + GUIDE_WORDS + [
    "资料中未找到", "资料中暂未有", "抱歉", "参考资料中未",
    "提供的参考资料中未", "没有找到", "暂无相关",
]


def is_refuse_text(answer):
    """
    判断回答是否表达了"不知道"。

    判定顺序（严格按此优先级）：
      A. 空或过短(<4字)                      → 拒答
      B. 开头 20 字含硬拒答词                 → 拒答
      C. 含软标记(未明确/未列出…) 且 <300字   → 拒答
      D. 含条款引用(第X条)                    → 不拒答
      E. 含分数型数据(\\d{2,}分)              → 不拒答（有实质内容）
      F. 含引导词(建议咨询…) 且 <60字          → 拒答
      G. <20字且含任意拒答词                  → 拒答
      H. 其他                                → 不拒答
    """
    import re
    a = "".join(str(answer or "").split())
    if len(a) < 4:                                        # A
        return True

    head20 = a[:20]
    for w in HARD_REFUSE:                                 # B
        if "".join(w.split()) in head20:
            return True

    # C. 软标记：字符串精确匹配 或 正则模糊匹配（允许修饰词）+ 长度 <300
    has_soft = any("".join(w.split()) in a for w in SOFT_MARKERS) or \
               any(re.search(p, a) for p in SOFT_PATTERNS)
    if has_soft and len(a) < 300:
        return True

    if re.search(r"第[一二三四五六七八九十百]+条", a):       # D
        return False

    if re.search(r"\d{2,}\s*分", a):                       # E
        return False

    if len(a) < 60 and any("".join(w.split()) in a for w in GUIDE_WORDS):     # F
        return True

    if len(a) < 20:                                        # G
        for w in REFUSE_WORDS:
            if "".join(w.split()) in a:
                return True

    return False                                           # H


REFUSE_ANSWER = "未找到相关信息。本系统仅覆盖河南专升本政策类问题，不提供备考建议、生活信息或个人决策建议。"

ANSWER_PROMPT = """你是河南专升本政策问答助手。根据材料回答问题。

规则：
1. 材料有相关内容，必须基于它作答，不要轻易说找不到
2. 只有材料完全无关（备考方法、生活建议、院校体验、个人决策）才回答「未找到相关信息」
3. 不要编造材料中没有的具体数字、日期
4. 回答简洁，控制在 300 字以内

材料：
{context}

问题：{question}

回答："""

FORCE_PROMPT = """以下材料与用户问题相关，请基于材料给出具体回答。
禁止回答"未找到相关信息"——如果材料只覆盖部分内容，就回答能确定的部分。

材料：
{context}

问题：{question}

回答："""


# ---------------- 启动：加载语料 + 同源校验 ----------------
def load_chunks():
    if not CHUNKS_PATH.exists():
        sys.exit(f"[错误] 语料文件不存在：{CHUNKS_PATH}")
    raw = CHUNKS_PATH.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    return [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]


CHUNKS = load_chunks()
BY_ID = {str(c["id"]): c for c in CHUNKS}
CHUNK_IDS = set(BY_ID)
CORPUS_YEARS = {str(c.get("year", "")) for c in CHUNKS}
CLIENT = QdrantClient(url=QDRANT_URL)


def check_same_source():
    try:
        info = CLIENT.get_collection(COLLECTION)
    except Exception as e:
        return False, f"Qdrant 不可连：{e}"
    pts = info.points_count
    dim = info.config.params.vectors.size
    if pts != len(CHUNKS):
        return False, f"点数不一致：Qdrant {pts} vs 文件 {len(CHUNKS)}"
    if dim != EMBED_DIM:
        return False, f"维度不一致：Qdrant {dim} vs 期望 {EMBED_DIM}"
    return True, f"{pts} 点 / {dim} 维 / 文件 {len(CHUNKS)} 块"


SAME_SOURCE_OK, SAME_SOURCE_MSG = check_same_source()


# ---------------- 检索 ----------------
def embed_one(text: str, retries: int = 3) -> list:
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                EMBED_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": EMBED_MODEL, "input": [text]},
                timeout=(10, 60),
            )
            r.raise_for_status()
            v = r.json()["data"][0]["embedding"]
            if len(v) != EMBED_DIM:
                raise ValueError(f"维度异常 {len(v)}")
            return v
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"embedding 失败：{last}")


def _to_hit(h):
    p = getattr(h, "payload", None) or {}
    cid = str(p.get("chunk_id") or getattr(h, "id", ""))
    chunk = BY_ID.get(cid, {})
    return {
        "chunk_id": cid,
        "source": p.get("source") or chunk.get("source", ""),
        "doc_type": p.get("doc_type") or chunk.get("doc_type", ""),
        "year": p.get("year") or chunk.get("year", ""),
        "score": round(float(getattr(h, "score", 0.0)), 4),
        "text": p.get("text") or chunk.get("text", ""),
        "in_corpus": cid in CHUNK_IDS,
    }


def search_filtered(query: str, top_k: int, filters: Optional[dict] = None):
    """检索（带元数据过滤）。返回 (hits, filters_applied, fallback_used)"""
    if filters:
        must = [{"key": k, "value": str(v)} for k, v in filters.items() if k in ("doc_type", "year")]
    else:
        must = infer_filters(query, CORPUS_YEARS)

    vec = embed_one(query)
    fallback = False
    try:
        if must:
            raw = qdrant_hits(CLIENT, vec, top_k, must)
            if not raw:
                raw = qdrant_hits(CLIENT, vec, top_k)
                fallback = True
        else:
            raw = qdrant_hits(CLIENT, vec, top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{e}")

    return [_to_hit(h) for h in raw], must, fallback


def rerank_hits(query, hits, top_n=5, timeout=25):
    """high 模式重排。失败时降级返回原顺序"""
    if not hits:
        return hits, 0.0
    docs = [h["text"][:RERANK_DOC_CHARS] for h in hits]
    t0 = time.time()
    try:
        r = requests.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": RERANK_MODEL, "query": query, "documents": docs,
                  "top_n": top_n, "return_documents": False},
            timeout=timeout,
        )
        if r.status_code >= 400:
            print(f"[rerank降级] HTTP {r.status_code}")
            return hits[:top_n], (time.time() - t0) * 1000
        out = []
        for it in r.json().get("results", []):
            idx = it.get("index", -1)
            if 0 <= idx < len(hits):
                h = dict(hits[idx])
                h["score"] = round(float(it.get("relevance_score", 0.0)), 4)
                out.append(h)
        return (out or hits[:top_n]), (time.time() - t0) * 1000
    except Exception as e:
        print(f"[rerank降级] {type(e).__name__}: {e}")
        return hits[:top_n], (time.time() - t0) * 1000


def llm_call(prompt: str, timeout=60):
    try:
        r = requests.post(
            LLM_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": GEN_MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        if r.status_code >= 400:
            print(f"[LLM错误] {r.status_code} {r.text[:150]}")
            return None
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM失败] {type(e).__name__}: {e}")
        return None


def build_context(hits, n=CTX_BLOCKS):
    return "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits[:n]))


def hybrid_candidates(question):
    """high 模式：BM25 + 向量 → RRF 合并 top CAND_K"""
    import jieba
    from rank_bm25 import BM25Okapi

    must = infer_filters(question, CORPUS_YEARS)
    sub = [c for c in CHUNKS if all(str(c.get(k["key"], "")) == k["value"] for k in must)] if must else CHUNKS
    if must and not sub:
        sub = CHUNKS

    vec = embed_one(question)
    raw = qdrant_hits(CLIENT, vec, CAND_K, must) if must else qdrant_hits(CLIENT, vec, CAND_K)
    if must and not raw:
        raw = qdrant_hits(CLIENT, vec, CAND_K)
    vec_c = [h["chunk_id"] for h in (_to_hit(x) for x in raw)]

    bm = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in sub])
    scores = bm.get_scores(list(jieba.cut_for_search(question)))
    idx = sorted(range(len(scores)), key=lambda j: -scores[j])[:CAND_K]
    bm_c = [str(sub[j]["id"]) for j in idx]

    rrf = {}
    for lst in (bm_c, vec_c):
        for rank, cid in enumerate(lst, 1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank)
    cand_ids = [c for c, _ in sorted(rrf.items(), key=lambda kv: -kv[1])][:CAND_K]

    return [{
        "chunk_id": cid,
        "source": BY_ID[cid].get("source", ""),
        "doc_type": BY_ID[cid].get("doc_type", ""),
        "year": BY_ID[cid].get("year", ""),
        "score": 0.0,
        "text": BY_ID[cid].get("text", ""),
        "in_corpus": True,
    } for cid in cand_ids if cid in BY_ID]


# ---------------- FastAPI ----------------
app = FastAPI(title="河南专升本政策问答 RAG API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SearchReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[dict] = None


class ChatReq(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    mode: str = Field("fast", pattern="^(fast|high)$")
    session_id: Optional[str] = None


@app.get("/health")
def health():
    ok, msg = check_same_source()
    if "不可连" in msg:
        return {"status": "unhealthy", "detail": msg, "chunks_count": 0,
                "collection": COLLECTION, "rerank_enabled": bool(API_KEY),
                "uptime_seconds": int(time.time() - START_TIME)}
    return {
        "status": "healthy" if ok else "degraded",
        "chunks_count": len(CHUNKS),
        "collection": COLLECTION,
        "vector_dim": EMBED_DIM,
        "embedding_model": EMBED_MODEL,
        "rerank_enabled": bool(API_KEY),
        "uptime_seconds": int(time.time() - START_TIME),
        "same_source": ok,
        "same_source_detail": msg,
    }


@app.post("/search")
def search(req: SearchReq):
    if not SAME_SOURCE_OK:
        raise HTTPException(status_code=503, detail=f"知识库未就绪：{SAME_SOURCE_MSG}")
    t0 = time.time()
    hits, must, fallback = search_filtered(req.query, req.top_k, req.filters)
    return {"hits": hits, "total": len(hits), "latency_ms": int((time.time() - t0) * 1000),
            "filters_applied": must, "fallback_used": fallback}


@app.post("/chat")
def chat(req: ChatReq):
    if not SAME_SOURCE_OK:
        raise HTTPException(status_code=503, detail=f"知识库未就绪：{SAME_SOURCE_MSG}")

    t0 = time.time()
    t_rr = 0.0

    if req.mode == "high":
        cands = hybrid_candidates(req.question)
        t_ret = (time.time() - t0) * 1000
        hits5, t_rr = rerank_hits(req.question, cands, top_n=5)
    else:
        hits5, _, _ = search_filtered(req.question, 5)
        t_ret = (time.time() - t0) * 1000

    # 空结果 / 低分 → 直接拒答（不调 LLM，省时省钱）
    empty = not hits5
    if req.mode == "fast" and hits5:
        empty = all(h.get("score", 1.0) < VECTOR_MIN_SCORE for h in hits5)
    if empty:
        return {"answer": REFUSE_ANSWER, "sources": [], "latency_ms": int((time.time() - t0) * 1000),
                "mode_used": req.mode, "refused": True,
                "timings": {"retrieve_ms": int(t_ret), "rerank_ms": int(t_rr), "llm_ms": 0}}

    # LLM 生成
    t_l0 = time.time()
    answer = llm_call(ANSWER_PROMPT.format(context=build_context(hits5), question=req.question))
    t_llm = (time.time() - t_l0) * 1000

    if answer is None:
        answer = "（答案生成暂不可用，以下为检索到的原文）\n\n" + build_context(hits5, 2)
        refused = False
    else:
        refused = is_refuse_text(answer)

    # 保护重试：LLM 拒答但 top1 向量分数偏高 → 强制基于材料作答
    if refused and req.mode == "fast" and FORCE_RETRY and hits5[0].get("score", 0) >= RETRY_MIN_SCORE:
        print(f"[重试] 拒答但 top1={hits5[0]['score']:.2f}，强制作答")
        t_r0 = time.time()
        a2 = llm_call(FORCE_PROMPT.format(context=build_context(hits5), question=req.question))
        t_llm += (time.time() - t_r0) * 1000
        if a2 and not is_refuse_text(a2):
            answer, refused = a2, False
            print("[重试成功]")

    sources = [] if refused else [{
        "chunk_id": h["chunk_id"], "source": h["source"], "doc_type": h["doc_type"],
        "year": h["year"], "score": h["score"], "snippet": h["text"][:200],
    } for h in hits5[:5]]

    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": int((time.time() - t0) * 1000),
        "mode_used": req.mode,
        "refused": refused,
        "timings": {"retrieve_ms": int(t_ret), "rerank_ms": int(t_rr), "llm_ms": int(t_llm)},
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatReq):
    """
    流式问答（SSE）。
    事件格式：
        event: meta    data: {"sources":[...], "retrieve_ms":..., "mode":"fast"}
        event: token   data: {"text":"..."}
        event: done    data: {"llm_ms":..., "refused":false, "first_token_ms":...}
        event: error   data: {"message":"..."}
    """
    import asyncio
    from fastapi.responses import StreamingResponse

    if not SAME_SOURCE_OK:
        raise HTTPException(status_code=503, detail=f"知识库未就绪：{SAME_SOURCE_MSG}")

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def gen():
        loop = asyncio.get_event_loop()
        t0 = time.time()

        # ---------- 1. 检索（在线程池里跑，不阻塞事件循环）----------
        try:
            if req.mode == "high":
                cands = await loop.run_in_executor(None, hybrid_candidates, req.question)
                t_ret = (time.time() - t0) * 1000
                hits5, t_rr = await loop.run_in_executor(
                    None, lambda: rerank_hits(req.question, cands, top_n=5))
            else:
                hits5, _, _ = await loop.run_in_executor(
                    None, lambda: search_filtered(req.question, 5))
                t_ret, t_rr = (time.time() - t0) * 1000, 0.0
        except Exception as e:
            yield sse("error", {"message": f"检索失败：{e}"})
            return

        # ---------- 2. 空结果 → 直接拒答 ----------
        empty = not hits5
        if req.mode == "fast" and hits5:
            empty = all(h.get("score", 1.0) < VECTOR_MIN_SCORE for h in hits5)
        if empty:
            yield sse("meta", {"sources": [], "retrieve_ms": int(t_ret),
                               "rerank_ms": int(t_rr), "mode": req.mode})
            yield sse("token", {"text": REFUSE_ANSWER})
            yield sse("done", {"llm_ms": 0, "refused": True,
                               "first_token_ms": int((time.time() - t0) * 1000)})
            return

        # ---------- 3. 先发 meta（含来源），用户马上能看到出处 ----------
        sources = [{
            "chunk_id": h["chunk_id"], "source": h["source"], "doc_type": h["doc_type"],
            "year": h["year"], "score": h["score"],
            "snippet": h["text"][:400],      # W7D6 修复：片段 200 → 400 字
            "text": h["text"],               # 全文，供前端"展开全文"
        } for h in hits5[:5]]
        yield sse("meta", {"sources": sources, "retrieve_ms": int(t_ret),
                           "rerank_ms": int(t_rr), "mode": req.mode,
                           "meta_ms": int((time.time() - t0) * 1000)})

        # ---------- 4. LLM 流式生成 ----------
        prompt = ANSWER_PROMPT.format(context=build_context(hits5), question=req.question)
        payload = {
            "model": GEN_MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }

        def _post_stream():
            return requests.post(
                LLM_URL,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json",
                         "Accept": "text/event-stream"},
                json=payload, stream=True, timeout=(10, 120),
            )

        try:
            resp = await loop.run_in_executor(None, _post_stream)
        except Exception as e:
            yield sse("error", {"message": f"LLM 连接失败：{e}"})
            return

        if resp.status_code >= 400:
            yield sse("error", {"message": f"LLM HTTP {resp.status_code}: {resp.text[:200]}"})
            return

        buf, full, first_tok = "", [], None
        t_llm0 = time.time()
        try:
            for raw in resp.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content") or ""
                if not piece:
                    continue
                if first_tok is None:
                    first_tok = (time.time() - t0) * 1000
                full.append(piece)
                yield sse("token", {"text": piece})
        except Exception as e:
            yield sse("error", {"message": f"流式读取失败：{e}"})
            return
        finally:
            try:
                resp.close()
            except Exception:
                pass

        answer = "".join(full).strip()
        refused = is_refuse_text(answer)
        t_llm = (time.time() - t_llm0) * 1000
        yield sse("done", {
            "llm_ms": int(t_llm),
            "refused": refused,
            "first_token_ms": int(first_tok) if first_tok else None,
            "total_ms": int((time.time() - t0) * 1000),
            "chars": len(answer),
        })

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 禁用 Nginx 缓冲
        },
    )


@app.get("/", include_in_schema=False)
def index():
    """返回流式前端页面"""
    from fastapi.responses import FileResponse, HTMLResponse
    for cand in (BASE / "docs" / "index_stream.html",
                 BASE / "index_stream.html",
                 BASE / "app" / "index_stream.html"):
        if cand.exists():
            return FileResponse(str(cand), media_type="text/html; charset=utf-8")
    return HTMLResponse("<h1>前端页面缺失</h1><p>请把 index_stream.html 放到 docs/ 目录</p>",
                        status_code=500)


@app.on_event("startup")
def _startup():
    print("=" * 60)
    print(f"[启动] 语料 {len(CHUNKS)} 块 | 集合 {COLLECTION} | max_tokens={MAX_TOKENS} | ctx={CTX_BLOCKS}")
    print(f"[同源] {'✅' if SAME_SOURCE_OK else '❌'} {SAME_SOURCE_MSG}")
    if not SAME_SOURCE_OK:
        print("[警告] 同源校验失败，/search 和 /chat 返回 503")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
