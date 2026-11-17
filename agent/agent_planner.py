# -*- coding: utf-8 -*-
"""
agent_planner.py —— 带任务规划的 Agent（W9D3，串行版）

流程图：
    [START]
       ↓
    [plan]      拆成 N 个子任务（W9D2 的 planner）
       ↓
    [execute]   依次跑每个子任务（串行，N × 8s）
       ↓
    [merge]     合并结果
       ↓
    [format]    格式化输出
       ↓
    [END]

对比 W8 的 main_agent：
    W8: route → ask/search/refuse → format
    W9: plan → execute(串行) → merge → format

用法：
    python agent/agent_planner.py                    # 跑 3 个复杂问题
    python agent/agent_planner.py "你的问题"          # 单条
    python agent/agent_planner.py --compare "问题"    # 对比拆解 vs 不拆
"""
import os
import sys
import time
from typing import Literal, TypedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from planner import plan_with_reason  # noqa: E402
from tools_robust import ask_with_fallback  # noqa: E402


# ============================================================
# State
# ============================================================
class PlanState(TypedDict, total=False):
    question: str
    plan_rule: str
    sub_tasks: list          # [{"dim", "question", "source"}]
    sub_results: list        # [{"ok", "answer", "sources", "latency_ms", "task"}]
    answer: str
    sources: list
    trace: list
    elapsed_ms: int
    partial: bool            # 是否有子任务失败
    failed_count: int


# ============================================================
# Node 1：plan —— 拆解
# ============================================================
def plan_node(state: PlanState) -> dict:
    q = state["question"]
    r = plan_with_reason(q)
    tasks = r["subtasks"]

    print(f"  [plan] {r['rule']} → {len(tasks)} 个子任务")
    for i, t in enumerate(tasks, 1):
        print(f"         {i}. [{t['dim']}] {t['question']}")

    return {
        "plan_rule": r["rule"],
        "sub_tasks": tasks,
        "trace": state.get("trace", []) + [f"plan:{len(tasks)}段"],
    }


# ============================================================
# Node 2：execute —— 串行执行（W9D3 先串行，W9D4 再并行）
# ============================================================
def execute_node(state: PlanState) -> dict:
    tasks = state["sub_tasks"]
    results = []

    for i, t in enumerate(tasks, 1):
        print(f"  [execute {i}/{len(tasks)}] {t['question']}")
        t0 = time.time()
        r = ask_with_fallback(t["question"])
        elapsed = int((time.time() - t0) * 1000)
        r["task"] = t
        r["sub_elapsed_ms"] = elapsed

        status = "✅" if r.get("ok") else "❌"
        print(f"           {status} {elapsed}ms | refused={r.get('refused')} | {len(r.get('sources', []))} 来源")

        results.append(r)

    failed = sum(1 for r in results if not r.get("ok"))
    return {
        "sub_results": results,
        "failed_count": failed,
        "partial": 0 < failed < len(results),
        "trace": state.get("trace", []) + [f"execute:{len(results)}段/{failed}失败"],
    }


# ============================================================
# Node 3：merge —— 合并
# ============================================================
def merge_node(state: PlanState) -> dict:
    results = state["sub_results"]
    total = len(results)
    failed = state.get("failed_count", 0)

    # 只有 1 个子任务 → 直接用它的答案，不合并
    if total == 1:
        r = results[0]
        return {
            "answer": r.get("answer", ""),
            "sources": r.get("sources", []),
            "trace": state.get("trace", []) + ["merge:单段直通"],
        }

    # 多段 → 按子任务编号拼接
    parts = []
    all_sources = []

    if failed > 0:
        parts.append(f"（注意：{total} 个子问题中有 {failed} 个未能回答，以下信息可能不完整）\n")

    for i, r in enumerate(results, 1):
        task = r.get("task", {})
        dim_label = {
            "professional": "专业维度",
            "score": "分数维度",
            "school": "院校维度",
            "policy": "政策维度",
        }.get(task.get("dim", ""), "问题")

        if not r.get("ok"):
            parts.append(f"### {i}. {dim_label}（{task.get('source', '')}）\n未能获取信息\n")
            continue

        if r.get("refused"):
            parts.append(f"### {i}. {dim_label}（{task.get('source', '')}）\n未找到相关信息\n")
            continue

        parts.append(f"### {i}. {dim_label}（{task.get('source', '')}）\n{r.get('answer', '')}\n")
        all_sources.extend(r.get("sources", []))

    answer = "\n".join(parts)
    # 去重来源（按 chunk_id）
    seen, uniq = set(), []
    for s in all_sources:
        cid = s.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            uniq.append(s)

    print(f"  [merge] {total} 段合并 → {len(answer)} 字 | {len(uniq)} 条去重来源")

    return {
        "answer": answer,
        "sources": uniq,
        "trace": state.get("trace", []) + [f"merge:{total}段"],
    }


# ============================================================
# Node 4：format
# ============================================================
def format_node(state: PlanState) -> dict:
    answer = state.get("answer", "")
    if state.get("partial"):
        print(f"  [format] 部分失败（{state.get('failed_count')} 段），已标注")
    else:
        print(f"  [format] 输出 {len(answer)} 字")
    return {"answer": answer, "trace": state.get("trace", []) + ["format"]}


# ============================================================
# 构建 Graph
# ============================================================
def build_planner_agent():
    g = StateGraph(PlanState)

    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_node("merge", merge_node)
    g.add_node("format", format_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "merge")
    g.add_edge("merge", "format")
    g.add_edge("format", END)

    return g.compile()


_AGENT = None


def run_planner_agent(question: str) -> dict:
    global _AGENT
    if _AGENT is None:
        _AGENT = build_planner_agent()

    t0 = time.time()
    r = _AGENT.invoke({"question": question, "trace": []})
    elapsed = int((time.time() - t0) * 1000)

    return {
        "answer": r.get("answer", ""),
        "sources": r.get("sources", []),
        "plan_rule": r.get("plan_rule", ""),
        "sub_tasks": r.get("sub_tasks", []),
        "sub_results": r.get("sub_results", []),
        "trace": r.get("trace", []),
        "elapsed_ms": elapsed,
        "partial": r.get("partial", False),
        "failed_count": r.get("failed_count", 0),
    }


# ============================================================
# 对比测试
# ============================================================
TEST_CASES = [
    ("我是计算机网络技术专业，230分，能报哪些公办学校？", "三维复杂问题"),
    ("专升本考哪几科？每科多少分？", "多问句"),
    ("2026年报名时间是什么时候？", "单一问题（不该拆）"),
]


def main():
    args = sys.argv[1:]

    if args and args[0] == "--compare" and len(args) > 1:
        q = " ".join(args[1:])
        print(f"[对比] {q}\n")

        print("=" * 78)
        print("【方案 A】直连 RAG（不拆）")
        print("=" * 78)
        t0 = time.time()
        r1 = ask_with_fallback(q)
        e1 = int((time.time() - t0) * 1000)
        print(f"  耗时 {e1}ms | ok={r1.get('ok')} | 来源 {len(r1.get('sources', []))}")
        print(f"  答案：{r1.get('answer', '')[:200]}...\n")

        print("=" * 78)
        print("【方案 B】拆解 + 串行执行")
        print("=" * 78)
        r2 = run_planner_agent(q)
        print(f"\n  耗时 {r2['elapsed_ms']}ms | {len(r2['sub_tasks'])} 段 | 失败 {r2['failed_count']}")
        print(f"\n  答案：\n{r2['answer'][:500]}")
        print(f"\n  轨迹：{' → '.join(r2['trace'])}")

        print("\n" + "=" * 78)
        print(f"结论：不拆 {e1}ms vs 拆解 {r2['elapsed_ms']}ms，差值 {r2['elapsed_ms'] - e1:+d}ms")
        return

    if args:
        q = " ".join(args)
        print(f"[问题] {q}\n")
        r = run_planner_agent(q)
        print(f"\n[答案]\n{r['answer']}")
        print(f"\n[规划] {r['plan_rule']} | {len(r['sub_tasks'])} 段")
        print(f"[耗时] {r['elapsed_ms']}ms | 失败 {r['failed_count']} 段")
        print(f"[轨迹] {' → '.join(r['trace'])}")
        return

    print("=" * 78)
    print("W9D3 串行执行测试（3 个用例）")
    print("=" * 78)

    for i, (q, desc) in enumerate(TEST_CASES, 1):
        print(f"\n{'═' * 78}")
        print(f"[用例 {i}] {desc}")
        print(f"  问题：{q}")
        print("─" * 78)

        r = run_planner_agent(q)

        print("─" * 78)
        print(f"  【结果】{len(r['sub_tasks'])} 段 | 失败 {r['failed_count']} | 《{r['elapsed_ms']}ms》")
        print(f"  【轨迹】{' → '.join(r['trace'])}")
        print(f"  【答案前 300 字】")
        print(f"    {r['answer'][:300].replace(chr(10), chr(10) + '    ')}")

    print("\n" + "=" * 78)
    print("自检：串行执行的延迟是多少？能不能接受？")
    print("=" * 78)
    print("""
【公式】串行延迟 ≈ N × 单次延迟
        单次 ask_with_fallback = 7-9 秒
        N=3 → 21-27 秒

【能不能接受】分场景：
  用户主动等待（打字聊天）→ ❌ 不可接受（超过 10 秒就流失）
  后台任务（离线生成、批量处理）→ ✅ 可以接受
  Agent 深度思考场景（用户预期慢一点）→ ⚠️ 勉强（15 秒内）

【所以 W9D4 要做并行】
  并行延迟 ≈ max(各子任务) ≈ 8-9 秒（不是 N × 8s）
  3 个并行 vs 3 个串行 → 24s → 9s（快 2.7 倍）

【但并行有代价】
  1. RAG 服务压力（2 核 2G，并发 3 可能打满）
  2. 部分失败处理更复杂（W9D5）
  3. 子任务必须无依赖
""")


if __name__ == "__main__":
    main()
