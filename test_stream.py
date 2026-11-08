# -*- coding: utf-8 -*-
"""
test_stream.py —— 测试 /chat/stream 流式输出 + 首字延迟
用法：
    python test_stream.py                                    # 默认测公网
    python test_stream.py http://127.0.0.1:8000               # 测本地/服务器内部
    python test_stream.py http://43.128.31.138 "你的问题"      # 自定义问题
"""
import json
import sys
import time

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://43.128.31.138"
QUESTION = sys.argv[2] if len(sys.argv) > 2 else "河南专升本考试考哪几科？"
MODE = sys.argv[3] if len(sys.argv) > 3 else "fast"

print(f"[目标] {BASE}/chat/stream")
print(f"[问题] {QUESTION}")
print(f"[模式] {MODE}")
print("=" * 70)

t0 = time.time()
first_token_ms = None
meta_ms = None
tokens = []
events = []

try:
    resp = requests.post(
        f"{BASE}/chat/stream",
        json={"question": QUESTION, "mode": MODE},
        stream=True,
        timeout=(10, 180),
        headers={"Accept": "text/event-stream"},
    )
    print(f"[HTTP] {resp.status_code}")
    if resp.status_code >= 400:
        print(resp.text[:300])
        sys.exit(1)

    cur_event = None
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore")
        if line.startswith("event:"):
            cur_event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue

        elapsed = (time.time() - t0) * 1000
        events.append(cur_event)

        if cur_event == "meta":
            meta_ms = elapsed
            print(f"\n[META +{elapsed:6.0f}ms] 检索 {data.get('retrieve_ms')}ms | "
                  f"rerank {data.get('rerank_ms')}ms | 来源 {len(data.get('sources', []))} 条")
            for s in data.get("sources", [])[:2]:
                print(f"        {s['chunk_id']} [{s['doc_type']}] {s['snippet'][:50]}...")
            print("\n[答案流] ", end="", flush=True)

        elif cur_event == "token":
            text = data.get("text", "")
            if first_token_ms is None:
                first_token_ms = elapsed
                print(f"（首字 +{elapsed:.0f}ms）", flush=True)
            tokens.append(text)
            print(text, end="", flush=True)

        elif cur_event == "done":
            print(f"\n\n[DONE +{elapsed:6.0f}ms]")
            print(f"  LLM 耗时      : {data.get('llm_ms')}ms")
            print(f"  首字延迟      : {data.get('first_token_ms')}ms")
            print(f"  总耗时        : {data.get('total_ms')}ms")
            print(f"  输出字数      : {data.get('chars')}")
            print(f"  拒答          : {data.get('refused')}")

        elif cur_event == "error":
            print(f"\n[ERROR] {data}")

except Exception as e:
    print(f"\n[失败] {type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("=== 结论 ===")
print(f"首字延迟: {first_token_ms:.0f}ms" if first_token_ms else "首字延迟: 未收到")
print(f"meta 延迟: {meta_ms:.0f}ms（来源卡片可先展示）" if meta_ms else "meta: 未收到")
print(f"token 数: {len(tokens)}")
print(f"总耗时: {(time.time() - t0) * 1000:.0f}ms")

print("\n=== 自检题答案 ===")
if first_token_ms and meta_ms:
    print(f"对比非流式（约 6675ms 一次性返回）：")
    print(f"  用户感知延迟从 6675ms → {first_token_ms:.0f}ms（首字）")
    print(f"  提升 {6675 / max(first_token_ms, 1):.1f} 倍")
    print(f"\n体感差异：")
    print(f"  - 非流式：空白等待 6.7 秒，用户不知道系统是不是卡死")
    print(f"  - 流式：{meta_ms:.0f}ms 看到来源，{first_token_ms:.0f}ms 开始出字，全程有反馈")
