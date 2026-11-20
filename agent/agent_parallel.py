# -*- coding: utf-8 -*-
"""
agent_parallel.py —— 并行版规划 Agent（W9D4）

核心改动：execute 节点从"串行 for 循环"改成"线程池并行"

串行：3 个子任务 = 8 + 8 + 8 = 24s
并行：3 个子任务 = max(8, 8, 8) ≈ 8-9s

关键约束：
    max_workers ≤ 3（RAG 是 2 核 2G，并发太高会打满）
    子任务必须无依赖（有依赖的走串行）
    部分失败要能标注（W9D5 完善）

用法：
    python agent/agent_parallel.py                  # 串行 vs 并行对比
    python agent/agent_parallel.py "问题"            # 单条（并行版）
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from planner import plan_with_reason  # noqa: E402
from tools_robust import ask_with_fallback  # noqa: E402

# 并发上限（RAG 2核2G，不要超过 3）
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))


class PlanState(TypedDict, total=False):
    question: str
    plan_rule: str
    sub_tasks: list
    sub_results: list
    answer: str
    sources: list
    trace: list
    elapsed_ms: int
    partial: bool
    failed_count: int
    parallel: bool


# ============================================================
# Node：plan
# ============================================================
def plan_node(state: PlanState) -> dict:
    r = plan_with_reason(state["question"])
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
# Node：execute —— 并行版（W9D4 核心）
# ============================================================
def _run_one(task: dict) -> dict:
    """跑单个子任务（供线程池调用）"""
    t0 = time.time()
    r = ask_with_fallback(task["question"])
    r["task"] = task
    r["sub_elapsed_ms"] = int((time.time() - t0) * 1000)
    return r


def execute_node_parallel(state: PlanState) -> dict:
    """
    并行执行所有子任务。

    关键点：
        1. max_workers 限制并发（不打爆 RAG）
        2. as_completed 收集（不等固定顺序，谁先完成先记录）
        3. 最后按原顺序排序（保证输出顺序稳定）
        4. 单个失败不影响其他
    """
    tasks = state["sub_tasks"]
    n = len(tasks)

    # 只有 1 个任务 → 没必要开线程池
    if n == 1:
        print(f"  [execute] 单任务，直连")
        r = _run_one(tasks[0])
        status = "✅" if r.get("ok") else "❌"
        print(f"           {status} {r['sub_elapsed_ms']}ms | refused={r.get('refused')}")
        return {
            "sub_results": [r],
            "failed_count": 0 if r.get("ok") else 1,
            "partial": False,
            "parallel": False,
            "trace": state.get("trace", []) + ["execute:串行(1段)"],
        }

    workers = min(MAX_WORKERS, n)
    print(f"  [execute] 并行启动 {n} 个子任务（max_workers={workers}）")
    t_start = time.time()

    results = [None] * n          # 按原顺序占位
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_run_one, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "task": tasks[idx],
                    "sub_elapsed_ms": 0,
                    "answer": "",
                    "sources": [],
                }
            results[idx] = r
            status = "✅" if r.get("ok") else "❌"
            print(f"           {status} [{idx + 1}] {r['sub_elapsed_ms']}ms | refused={r.get('refused')}")

    wall = int((time.time() - t_start) * 1000)
    failed = sum(1 for r in results if not r.get("ok"))
    serial_est = sum(r["sub_elapsed_ms"] for r in results)

    print(f"  [execute] 并行完成：墙钟 {wall}ms | 各任务累计 {serial_est}ms | 加速 {serial_est / max(wall, 1):.1f}×")

    return {
        "sub_results": results,
        "failed_count": failed,
        "partial": 0 < failed < n,
        "parallel": True,
        "trace": state.get("trace", []) + [
            f"execute:并行{workers}并发/{wall}ms/加速{serial_est / max(wall, 1):.1f}×"
        ],
    }


# ============================================================
# Node：merge（与 W9D3 相同）
# ============================================================
def merge_node(state: PlanState) -> dict:
    results = state["sub_results"]
    total = len(results)
    failed = state.get("failed_count", 0)

    if total == 1:
        r = results[0]
        return {
            "answer": r.get("answer", ""),
            "sources": r.get("sources", []),
            "trace": state.get("trace", []) + ["merge:单段直通"],
        }

    parts = []
    if failed > 0:
        parts.append(f"（注意：{total} 个子问题中 {failed} 个未能回答，以下信息可能不完整）\n")

    all_sources = []
    for i, r in enumerate(results, 1):
        task = r.get("task", {})
        label = {
            "professional": "专业维度", "score": "分数维度",
            "school": "院校维度", "policy": "政策维度",
        }.get(task.get("dim", ""), "问题")

        if not r.get("ok"):
            parts.append(f"### {i}. {label}（{task.get('source', '')}）\n未能获取信息：{r.get('error', '')}\n")
        elif r.get("refused"):
            parts.append(f"### {i}. {label}（{task.get('source', '')}）\n未找到相关信息\n")
        else:
            parts.append(f"### {i}. {label}（{task.get('source', '')}）\n{r.get('answer', '')}\n")
            all_sources.extend(r.get("sources", []))

    seen, uniq = set(), []
    for s in all_sources:
        cid = s.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            uniq.append(s)

    answer = "\n".join(parts)
    print(f"  [merge] {total} 段 → {len(answer)} 字 | {len(uniq)} 条去重来源")

    return {
        "answer": answer,
        "sources": uniq,
        "trace": state.get("trace", []) + [f"merge:{total}段"],
    }


def format_node(state: PlanState) -> dict:
    print(f"  [format] 输出 {len(state.get('answer', ''))} 字"
          + ("（部分失败）" if state.get("partial") else ""))
    return {"trace": state.get("trace", []) + ["format"]}


# ============================================================
# 构建
# ============================================================
def build_parallel_agent():
    g = StateGraph(PlanState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node_parallel)
    g.add_node("merge", merge_node)
    g.add_node("format", format_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "merge")
    g.add_edge("merge", "format")
    g.add_edge("format", END)
    return g.compile()


_AGENT = None


def run_parallel_agent(question: str) -> dict:
    global _AGENT
    if _AGENT is None:
        _AGENT = build_parallel_agent()

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
        "parallel": r.get("parallel", False),
    }


# ============================================================
# 串行 vs 并行对比
# ============================================================
COMPARE_CASES = [
    "专升本考哪几科？每科多少分？",
    "我是计算机网络技术专业，230分，能报哪些公办学校？",
]


def run_serial(question: str) -> dict:
    """串行执行（对照）"""
    r = plan_with_reason(question)
    tasks = r["subtasks"]
    t0 = time.time()
    results = []
    print(f"  [串行] {len(tasks)} 个子任务，逐个执行")
    for i, t in enumerate(tasks, 1):
        print(f"    [{i}/{len(tasks)}] {t['question']}")
        r1 = _run_one(t)
        results.append(r1)
        print(f"      {'✅' if r1.get('ok') else '❌'} {r1['sub_elapsed_ms']}ms")
    return {
        "elapsed_ms": int((time.time() - t0) * 1000),
        "n": len(tasks),
        "results": results,
    }


def main():
    args = sys.argv[1:]

    if args and args[0] == "--compare":
        q = " ".join(args[1:]) if len(args) > 1 else COMPARE_CASES[0]
        print(f"[对比问题] {q}\n")

        print("=" * 78)
        print("【串行】")
        print("=" * 78)
        s = run_serial(q)
        print(f"\n  串行总耗时：{s['elapsed_ms']}ms（{s['n']} 段）\n")

        print("=" * 78)
        print("【并行】")
        print("=" * 78)
        p = run_parallel_agent(q)
        print(f"\n  并行总耗时：{p['elapsed_ms']}ms（{p['n'] if 'n' in p else len(p['sub_tasks'])} 段）\n")

        print("=" * 78)
        print("【结论】")
        print("=" * 78)
        print(f"  串行 {s['elapsed_ms']}ms → 并行 {p['elapsed_ms']}ms")
        print(f"  提升 {(s['elapsed_ms'] - p['elapsed_ms']) / s['elapsed_ms'] * 100:.0f}%")
        print(f"  加速 {s['elapsed_ms'] / max(p['elapsed_ms'], 1):.2f}×")
        return

    if args:
        q = " ".join(args)
        print(f"[问题] {q}\n")
        r = run_parallel_agent(q)
        print(f"\n[答案]\n{r['answer']}")
        print(f"\n[规划] {r['plan_rule']} | {len(r['sub_tasks'])} 段")
        print(f"[耗时] {r['elapsed_ms']}ms | 失败 {r['failed_count']} 段")
        print(f"[轨迹] {' → '.join(r['trace'])}")
        return

    # 默认：跑对比
    for q in COMPARE_CASES:
        print("\n" + "█" * 78)
        print(f"问题：{q}")
        print("█" * 78)

        print("\n" + "=" * 78)
        print("【串行】")
        print("=" * 78)
        s = run_serial(q)
        print(f"\n  串行：{s['elapsed_ms']}ms")

        print("\n" + "=" * 78)
        print("【并行】")
        print("=" * 78)
        p = run_parallel_agent(q)
        print(f"\n  并行：{p['elapsed_ms']}ms")

        print("\n" + "-" * 78)
        print(f"结果：{s['elapsed_ms']}ms → {p['elapsed_ms']}ms "
              f"（快 {(s['elapsed_ms'] - p['elapsed_ms']) / s['elapsed_ms'] * 100:.0f}%，"
              f"加速 {s['elapsed_ms'] / max(p['elapsed_ms'], 1):.2f}×）")
        print("-" * 78)

    print("\n" + "=" * 78)
    print("自检：并行有什么风险？")
    print("=" * 78)
    print("""
【风险 1：RAG 服务压力】
  RAG 跑在 2 核 2G 的腾讯云上，并发 3 个请求 = 3 个 LLM 调用同时进行
  可能症状：
    - 响应变慢（CPU 抢不到）
    - 内存不足（Python + Qdrant 本身占 800MB）
    - 极端情况 OOM 被杀
  缓解：
    - max_workers ≤ 3
    - 加超时（单个子任务超过 30s 就放弃）
    - W10 考虑加队列/限流

【风险 2：部分失败】
  3 个并行，1 个失败 → 不能整体崩
  处理：
    - 每个子任务独立 try（已实现）
    - merge 时标注"以下信息可能不完整"
    - 用户能看到哪几段成功了

【风险 3：资源竞争导致延迟反而变长】
  理论上并行 3 个 = 1 个的时间
  实际：RAG 服务处理能力有限，3 个并发可能每个都变慢
  实测数据：
    - 如果加速比 < 1.5×，说明服务被打满了
    - 这时应该降 max_workers 到 2

【风险 4：子任务有依赖时不能并行】
  "怎么报名？报名后干嘛？" → 第二句依赖第一句的结果
  规则式 planner 目前按句子拆，不会识别依赖
  缓解：planner 里排除"然后/之后/接着"这类词

【风险 5：调试变难】
  并行时日志会交错，难追踪
  缓解：
    - 每个子任务编号（已实现 [1] [2] [3]）
    - 记录每个的耗时
    - 按原顺序排列结果（不是完成顺序）
""")


if __name__ == "__main__":
    main()
