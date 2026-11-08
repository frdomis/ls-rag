#!/usr/bin/env python
# -*- coding: utf-8 -*-
# test_e2e.py —— W6D6 端到端测试 5 类 + 延迟测试
"""
用法（服务已启动后）：
    python test_e2e.py                    # 5 类功能测试
    python test_e2e.py --latency          # 额外跑延迟测试（两款模式各 10 次）
    python test_e2e.py --base http://localhost:8000
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent

# 5 类测试用例
CASES = [
    {
        "name": "1. 可答题-core（考试科目）",
        "question": "河南专升本考试考哪几科？",
        "expect": "answer",       # 应该有实质答案
        "mode": "fast",
    },
    {
        "name": "2. 可答题-core（分数线）",
        "question": "2026年河南专升本录取控制分数线是多少？",
        "expect": "answer",
        "mode": "fast",
    },
    {
        "name": "3. 真实提问题-user（退役士兵免试）",
        "question": "免试是不是不用考试啊，资格审核严不严，有过来人吗？",
        "expect": "answer",
        "mode": "high",
    },
    {
        "name": "4. 拒答题-语料缺失（跨专业材料）",
        "question": "跨专业报考审核要交什么材料？",
        "expect": "refuse",       # 应拒答
        "mode": "fast",
    },
    {
        "name": "5. 超范围（备考策略）",
        "question": "英语60出头冲100现实吗，阅读和单词先抓哪个？",
        "expect": "refuse",
        "mode": "fast",
    },
]


def call_chat(base, question, mode):
    t0 = time.time()
    r = requests.post(f"{base}/chat", json={"question": question, "mode": mode}, timeout=90)
    elapsed = (time.time() - t0) * 1000
    if r.status_code >= 400:
        return None, r.status_code, r.text[:200], elapsed
    return r.json(), r.status_code, None, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()

    # 健康检查
    try:
        h = requests.get(f"{args.base}/health", timeout=10).json()
        print(f"[健康] {h.get('status')} | {h.get('chunks_count')} 块 | 同源 {h.get('same_source')}")
        if h.get("status") != "healthy":
            print("[中止] 服务不健康")
            return
    except Exception as e:
        print(f"[错误] 服务不可达：{e}")
        return
    print()

    passed = failed = 0
    records = []
    for c in CASES:
        print(f"{'='*70}")
        print(f"{c['name']}  [mode={c['mode']}]")
        print(f"问：{c['question']}")
        data, code, err, elapsed = call_chat(args.base, c["question"], c["mode"])
        if err:
            print(f"❌ HTTP {code}: {err}")
            failed += 1
            continue

        ans = data.get("answer", "")
        refused = data.get("refused", False)
        srcs = data.get("sources", [])
        timings = data.get("timings", {})

        print(f"答：{ans[:150]}{'...' if len(ans) > 150 else ''}")
        print(f"拒答={refused} | 来源 {len(srcs)} 条 | 总 {data.get('latency_ms')}ms "
              f"（检索 {timings.get('retrieve_ms')} / rerank {timings.get('rerank_ms')} / LLM {timings.get('llm_ms')}）")
        if srcs:
            print(f"引用：{srcs[0]['chunk_id']} [{srcs[0]['doc_type']}] {srcs[0]['snippet'][:60]}...")

        ok = (refused if c["expect"] == "refuse" else (not refused and len(ans) > 20))
        print("✅ 通过" if ok else "❌ 未通过")
        if ok:
            passed += 1
        else:
            failed += 1
        records.append(dict(name=c["name"], question=c["question"], answer=ans,
                            refused=refused, mode=c["mode"], latency_ms=data.get("latency_ms"),
                            timings=timings, ok=ok, sources=[s["chunk_id"] for s in srcs]))
        print()

    print(f"{'='*70}")
    print(f"[结果] 通过 {passed}/{len(CASES)}")

    # 延迟测试
    if args.latency:
        print(f"\n{'='*70}")
        print("[延迟测试] 每种模式 10 次（同一问题，测 P50/P95）")
        q = "河南专升本考试考哪几科？"
        lat_data = {}
        for mode in ("fast", "high"):
            vals, retrieve, llm = [], [], []
            for i in range(10):
                d, code, err, el = call_chat(args.base, q, mode)
                if err:
                    print(f"  {mode} 第{i+1}次失败: {err[:60]}")
                    continue
                vals.append(d["latency_ms"])
                t = d.get("timings", {})
                retrieve.append(t.get("retrieve_ms", 0))
                llm.append(t.get("llm_ms", 0))
                time.sleep(0.5)
            if vals:
                vals.sort()
                p50 = vals[len(vals) // 2]
                p95 = vals[min(len(vals) - 1, int(len(vals) * 0.95))]
                lat_data[mode] = dict(p50=p50, p95=p95, mean=statistics.mean(vals),
                                      retrieve=statistics.mean(retrieve), llm=statistics.mean(llm))
                print(f"  {mode:5s}: P50={p50}ms  P95={p95}ms  均值={statistics.mean(vals):.0f}ms "
                      f"（检索 {statistics.mean(retrieve):.0f} + LLM {statistics.mean(llm):.0f}）")

    # 输出报告
    out = BASE / "docs" / "eval" / "e2e_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# W6D6 端到端测试报告\n\n")
        f.write(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- 服务：{args.base}\n")
        f.write(f"- 结果：**通过 {passed}/{len(CASES)}**\n\n")
        f.write("## 5 类测试\n\n| # | 用例 | 模式 | 拒答 | 来源数 | 延迟 | 结果 |\n|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(records, 1):
            f.write(f"| {i} | {r['name']} | {r['mode']} | {'是' if r['refused'] else '否'} | "
                    f"{len(r['sources'])} | {r['latency_ms']}ms | {'✅' if r['ok'] else '❌'} |\n")
        f.write("\n## 详细记录\n\n")
        for r in records:
            f.write(f"### {r['name']}\n\n")
            f.write(f"- **问**：{r['question']}\n")
            f.write(f"- **答**：{r['answer']}\n")
            f.write(f"- **模式**：{r['mode']} | **拒答**：{r['refused']}\n")
            f.write(f"- **延迟**：{r['latency_ms']}ms {r.get('timings', {})}\n")
            f.write(f"- **来源**：{r['sources']}\n\n")
        if args.latency and lat_data:
            f.write("## 延迟测试\n\n| 模式 | P50 | P95 | 均值 | 检索 | LLM |\n|---|---|---|---|---|---|\n")
            for m, d in lat_data.items():
                f.write(f"| {m} | {d['p50']}ms | {d['p95']}ms | {d['mean']:.0f}ms | "
                        f"{d['retrieve']:.0f}ms | {d['llm']:.0f}ms |\n")
    print(f"\n[输出] {out}")


if __name__ == "__main__":
    main()
