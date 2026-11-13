# -*- coding: utf-8 -*-
"""
tools_robust.py —— 带重试和降级的工具层（W8D5）

相比 tools.py 的增强：
    1. 自动重试（指数退避，最多 3 次）
    2. 超时区分（连接超时 / 读取超时）
    3. 错误分类（可重试 vs 不可重试）
    4. 降级链（ask_rag 失败 → search_docs）

错误分类：
    可重试：连接超时、5xx、429（限流）
    不可重试：4xx（400 参数错、404 路径错）

用法：
    python agent/tools_robust.py         # 自测（含故障注入）
"""
import os
import time
from typing import Optional

import requests

BASE = os.environ.get("RAG_BASE", "http://127.0.0.1:8000").rstrip("/")

# 超时：(连接, 读取)
TIMEOUT_SEARCH = (10, 25)
TIMEOUT_CHAT = (10, 100)
TIMEOUT_HEALTH = (10, 15)

# 重试配置
MAX_RETRIES = 3
BACKOFF_BASE = 1.0      # 1s, 2s, 4s

# 可重试的 HTTP 状态码
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# ============================================================
# 核心：带重试的请求
# ============================================================
def _request_with_retry(method: str, url: str, timeout: tuple, **kwargs) -> dict:
    """
    带指数退避重试的请求。

    返回：
        {"ok": True, "data": ..., "attempts": N, "elapsed_ms": X}
        {"ok": False, "error": "...", "error_type": "...", "attempts": N, "elapsed_ms": X}

    error_type:
        "timeout"     → 超时（可重试，但重试完仍失败）
        "connection"  → 连不上（可重试）
        "http_5xx"    → 服务端错误（可重试）
        "http_4xx"    → 客户端错误（不可重试）
        "unknown"     → 其他
    """
    t0 = time.time()
    last_error = None
    last_type = "unknown"
    attempts = 0

    for attempt in range(1, MAX_RETRIES + 1):
        attempts = attempt
        try:
            r = requests.request(method, url, timeout=timeout, **kwargs)

            # 5xx / 429 可重试
            if r.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {r.status_code}"
                last_type = "http_5xx"
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE * (2 ** (attempt - 1))
                    print(f"    [重试 {attempt}/{MAX_RETRIES}] {last_error}，{wait}s 后重试")
                    time.sleep(wait)
                    continue
                break

            # 4xx 不可重试
            if r.status_code >= 400:
                return {
                    "ok": False,
                    "error": f"HTTP {r.status_code}: {r.text[:100]}",
                    "error_type": "http_4xx",
                    "attempts": attempts,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                }

            return {
                "ok": True,
                "data": r.json(),
                "attempts": attempts,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }

        except requests.Timeout as e:
            last_error = f"超时（{timeout}）"
            last_type = "timeout"
        except requests.ConnectionError as e:
            last_error = f"连接失败：{type(e).__name__}"
            last_type = "connection"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            last_type = "unknown"

        if attempt < MAX_RETRIES and last_type in ("timeout", "connection"):
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"    [重试 {attempt}/{MAX_RETRIES}] {last_error}，{wait}s 后重试")
            time.sleep(wait)

    return {
        "ok": False,
        "error": last_error,
        "error_type": last_type,
        "attempts": attempts,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


# ============================================================
# 工具：带重试版
# ============================================================
def search_docs(query: str, top_k: int = 5, filters: Optional[dict] = None) -> dict:
    """检索（带重试）"""
    payload = {"query": query, "top_k": top_k}
    if filters:
        payload["filters"] = filters

    r = _request_with_retry("POST", f"{BASE}/search", TIMEOUT_SEARCH, json=payload)
    if r["ok"]:
        d = r["data"]
        return {
            "ok": True,
            "hits": d.get("hits", []),
            "total": d.get("total", 0),
            "filters_applied": d.get("filters_applied", []),
            "attempts": r["attempts"],
            "latency_ms": r["elapsed_ms"],
        }
    return {**r, "hits": [], "total": 0}


def ask_rag(question: str, mode: str = "fast") -> dict:
    """问答（带重试）"""
    r = _request_with_retry(
        "POST", f"{BASE}/chat", TIMEOUT_CHAT,
        json={"question": question, "mode": mode},
    )
    if r["ok"]:
        d = r["data"]
        return {
            "ok": True,
            "answer": d.get("answer", ""),
            "sources": d.get("sources", []),
            "refused": d.get("refused", False),
            "attempts": r["attempts"],
            "latency_ms": r["elapsed_ms"],
        }
    return {**r, "answer": "", "sources": [], "refused": False}


def health_check() -> dict:
    """健康检查（不重试，快速失败）"""
    try:
        r = requests.get(f"{BASE}/health", timeout=TIMEOUT_HEALTH)
        r.raise_for_status()
        d = r.json()
        return {
            "ok": d.get("status") == "healthy",
            "status": d.get("status"),
            "chunks_count": d.get("chunks_count", 0),
        }
    except Exception as e:
        return {"ok": False, "status": "unreachable", "error": f"{type(e).__name__}: {e}"}


# ============================================================
# 降级链
# ============================================================
def ask_with_fallback(question: str, mode: str = "fast") -> dict:
    """
    带降级的问答：
        1. 先试 ask_rag（完整答案）
        2. 失败 → 降级为 search_docs（至少给原文）
        3. 都失败 → 返回服务不可用

    返回额外字段：
        degraded: bool      是否降级了
        degrade_reason: str 降级原因
    """
    # 尝试 1：ask_rag
    print("    [降级链] 尝试 ask_rag ...")
    r1 = ask_rag(question, mode)

    if r1["ok"]:
        return {**r1, "degraded": False, "degrade_reason": ""}

    # 不可重试的错误（4xx）也不用降级，直接返回
    if r1.get("error_type") == "http_4xx":
        print(f"    [降级链] 4xx 错误，不降级")
        return {**r1, "degraded": False, "degrade_reason": ""}

    print(f"    [降级链] ask_rag 失败（{r1['error']}），降级为 search_docs")

    # 尝试 2：search_docs
    r2 = search_docs(question, top_k=5)

    if r2["ok"]:
        hits = r2.get("hits", [])
        if hits:
            # 用原文拼一个答案
            parts = [f"（降级模式：生成服务不可用，以下为检索到的原文）\n"]
            for i, h in enumerate(hits[:3], 1):
                parts.append(f"[{i}] {h.get('source', '-')}\n{h.get('text', '')[:300]}\n")
            answer = "\n".join(parts)
            return {
                "ok": True,
                "answer": answer,
                "sources": hits,
                "refused": False,
                "attempts": r1["attempts"] + r2["attempts"],
                "latency_ms": r1["elapsed_ms"] + r2["elapsed_ms"],
                "degraded": True,
                "degrade_reason": f"ask_rag 失败：{r1['error']}",
            }
        return {
            "ok": True,
            "answer": "未检索到相关文档。",
            "sources": [],
            "refused": True,
            "degraded": True,
            "degrade_reason": f"ask_rag 失败：{r1['error']}",
        }

    # 两个都失败
    print(f"    [降级链] search_docs 也失败")
    return {
        "ok": False,
        "answer": "服务暂时不可用，请稍后重试。",
        "sources": [],
        "refused": False,
        "error": f"ask: {r1['error']} | search: {r2['error']}",
        "error_type": "all_failed",
        "degraded": True,
        "degrade_reason": "两个工具都失败",
        "latency_ms": r1["elapsed_ms"] + r2["elapsed_ms"],
    }


# ============================================================
# 自测
# ============================================================
def _self_test_failures():
    """故障注入测试（独立函数，方便改 BASE）"""
    import tools_robust as me

    orig = me.BASE

    # ---- 测试 2：服务不可达 ----
    print("\n【测试 2】服务不可达（指向 9999 端口，测重试+降级）")
    me.BASE = "http://127.0.0.1:9999"
    try:
        t0 = time.time()
        r2 = me.ask_with_fallback("测试问题")
        elapsed = time.time() - t0
        print(f"  结果：ok={r2['ok']} degraded={r2.get('degraded')} error_type={r2.get('error_type')}")
        print(f"  答案：{r2['answer'][:70]}")
        print(f"  降级原因：{r2.get('degrade_reason')}")
        print(f"  总耗时：{elapsed:.1f}s（含指数退避重试 1s+2s）")
    finally:
        me.BASE = orig

    # ---- 测试 3：4xx ----
    print("\n【测试 3】4xx 错误（错误路径，应不重试、不降级）")
    me.BASE = "http://43.128.31.138/wrong"
    try:
        t0 = time.time()
        r3 = me._request_with_retry("GET", f"{me.BASE}/health", me.TIMEOUT_HEALTH)
        elapsed = time.time() - t0
        print(f"  ok={r3['ok']} error_type={r3.get('error_type')} attempts={r3.get('attempts')}")
        print(f"  error={str(r3.get('error', ''))[:60]}")
        print(f"  耗时：{elapsed:.1f}s（4xx 不重试，应该 <2s）")
    finally:
        me.BASE = orig


if __name__ == "__main__":
    import sys

    print(f"[后端] {BASE}")
    print("=" * 70)

    # 测试 1：正常情况
    print("\n【测试 1】正常情况（服务可用）")
    h = health_check()
    print(f"  健康检查：{h['ok']} ({h.get('status')})")

    if h["ok"]:
        r = ask_with_fallback("河南专升本考试考哪几科？")
        print(f"  结果：ok={r['ok']} degraded={r.get('degraded')} 尝试 {r.get('attempts')} 次")
        print(f"  答案：{r['answer'][:60]}...")

    # 测试 2、3：故障注入
    _self_test_failures()

    print("\n" + "=" * 70)
    print("自检：Agent 的失败处理和普通函数的 try-catch 有什么不同？")
    print("=" * 70)
    print("""
1. 分层处理，不是一个大 try-catch
   - 工具层：重试（网络抖动可自愈）
   - Agent 层：降级（换一个工具）
   - 用户层：友好提示（不暴露技术细节）

2. 错误要分类，不是都重试
   - 超时/5xx/429 → 可重试
   - 4xx → 不重试（参数错了，重试也没用）
   - 业务错误（refused）→ 不是异常，正常返回

3. 降级要有边界
   - ask_rag 失败 → search_docs（有意义的降级）
   - search_docs 失败 → 返回"不可用"（不做无意义重试）

4. 失败也要可观测
   - attempts（试了几次）
   - degraded（是否降级）
   - degrade_reason（为什么降级）

5. 用户体验优先
   - 普通 try-catch 只保证"不崩溃"
   - Agent 要保证"尽量给有用的东西"
""")
