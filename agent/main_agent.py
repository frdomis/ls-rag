# -*- coding: utf-8 -*-
"""
main_agent.py —— 第一个完整 Agent（W8D4）

流程图：
    [START]
       ↓
    [route]  判断问题类型（规则式）
       ↓
    ┌──┴──────┬────────┐
    ↓         ↓        ↓
  [ask]   [search]  [refuse]
    ↓         ↓        ↓
    └──┬──────┴────────┘
       ↓
    [format] 整理输出
       ↓
     [END]

LangGraph 三件套：
    State  → AgentState（TypedDict）
    Node   → route / ask_node / search_node / refuse_node / format_node
    Edge   → route 之后是条件边，其余是固定边

用法：
    python agent/main_agent.py              # 跑 5 个测试用例
    python agent/main_agent.py "你的问题"    # 单条
"""
import os
import sys
import time
from typing import Literal, Optional, TypedDict

# 让 import 能找到同目录的 router / tools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from router import explain, route  # noqa: E402

# W8D5：使用带重试+降级的工具（性能比 tools.py 差一点，但容错强）
try:
    from tools_robust import ask_with_fallback, search_docs
    TOOLS_MODE = "robust"
except ImportError:
    from tools import ask_rag as _ask_rag  # 兜底
    from tools import search_docs
    TOOLS_MODE = "simple"

    def ask_with_fallback(question, mode="fast"):
        r = _ask_rag(question, mode)
        return {**r, "degraded": False, "degrade_reason": ""}


# ============================================================
# 1. State
# ============================================================
class AgentState(TypedDict, total=False):
    """贯穿整个图的状态。total=False 表示字段可选（节点只更新需要的字段）"""
    question: str          # 用户问题
    route: str             # 路由结果：ask / search / refuse
    route_reason: str      # 路由原因（可解释性）
    tool_result: dict      # 工具返回的原始结果
    answer: str            # 最终答案
    sources: list          # 引用来源
    trace: list            # 执行轨迹（可观测性）
    elapsed_ms: int        # 总耗时
    degraded: bool         # W8D5：是否降级
    degrade_reason: str    # W8D5：降级原因


REFUSE_TEXT = "未找到相关信息。本系统仅回答河南专升本政策类问题，不提供备考建议、生活信息或个人决策建议。"


# ============================================================
# 2. Node 定义
# ============================================================
def route_node(state: AgentState) -> dict:
    """节点 1：判断走哪条路"""
    q = state["question"]
    r = route(q)
    reason = explain(q)["reason"]
    print(f"  [route] {r}  ({reason[:40]})")
    return {
        "route": r,
        "route_reason": reason,
        "trace": state.get("trace", []) + [f"route:{r}"],
    }


def ask_node(state: AgentState) -> dict:
    """节点 2a：调 ask_with_fallback（重试 + 降级到 search）"""
    print(f"  [ask] 调用 ask_with_fallback（含重试+降级）...")
    result = ask_with_fallback(state["question"])

    if not result["ok"]:
        # 两个工具都失败
        print(f"    ❌ 完全失败：{result.get('error', '')[:60]}")
        return {
            "tool_result": result,
            "answer": "服务暂时不可用，请稍后重试。",
            "sources": [],       # 空 sources → format 不会加"以下是原文"前缀
            "degraded": True,
            "degrade_reason": result.get("degrade_reason", "所有工具失败"),
            "trace": state.get("trace", []) + ["ask:all_failed"],
        }

    degraded = result.get("degraded", False)
    attempts = result.get("attempts", 1)
    tag = "降级" if degraded else "正常"

    print(f"    {'⚠️' if degraded else '✅'} {tag} | {result.get('latency_ms')}ms | "
          f"{len(result.get('sources', []))} 来源 | 尝试 {attempts} 次")

    return {
        "tool_result": result,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "degraded": degraded,
        "degrade_reason": result.get("degrade_reason", ""),
        "trace": state.get("trace", []) + [
            f"ask:{'degraded' if degraded else 'ok'}({attempts}次)"
        ],
    }


def search_node(state: AgentState) -> dict:
    """节点 2b：调 search_docs（只给原文，不生成）"""
    print(f"  [search] 调用 search_docs ...")
    result = search_docs(state["question"], top_k=5)
    if not result["ok"]:
        print(f"    ❌ 失败：{result['error']}")
        return {
            "tool_result": result,
            "answer": f"检索失败（{result['error']}）。",
            "sources": [],
            "trace": state.get("trace", []) + ["search:failed"],
        }
    hits = result.get("hits", [])
    print(f"    ✅ {result.get('latency_ms')}ms | {len(hits)} 条")
    return {
        "tool_result": result,
        "answer": "",       # search 不生成答案，交给 format 整理
        "sources": hits,
        "trace": state.get("trace", []) + ["search:ok"],
    }


def refuse_node(state: AgentState) -> dict:
    """节点 2c：直接拒答（不调工具）"""
    print(f"  [refuse] 直接拒答，不调工具")
    return {
        "tool_result": {},
        "answer": REFUSE_TEXT,
        "sources": [],
        "trace": state.get("trace", []) + ["refuse"],
    }


def format_node(state: AgentState) -> dict:
    """节点 3：整理输出"""
    r = state["route"]
    sources = state.get("sources", [])
    degraded = state.get("degraded", False)

    if r == "refuse":
        out = state["answer"]

    elif r == "ask":
        out = state["answer"]
        # W8D5：降级且确实有原文时才加提示（避免和"服务不可用"重复）
        if degraded and sources:
            out = f"⚠️ 生成服务暂时不可用，以下是检索到的原文：\n\n{out}"

    elif r == "search":
        # search 只给了原文，这里整理成可读列表
        if not sources:
            out = "未检索到相关文档。"
        else:
            lines = [f"检索到 {len(sources)} 条相关原文：\n"]
            for i, h in enumerate(sources[:5], 1):
                src = h.get("source", "-")
                dtype = h.get("doc_type", "-")
                text = (h.get("text") or h.get("snippet") or "")[:200]
                lines.append(f"**[{i}]** {src} · {dtype}\n{text}\n")
            out = "\n".join(lines)
    else:
        out = state.get("answer", "")

    print(f"  [format] 输出 {len(out)} 字{('（降级）' if degraded else '')}")
    return {"answer": out, "trace": state.get("trace", []) + ["format"]}


# ============================================================
# 3. 条件边
# ============================================================
def route_after_route(state: AgentState) -> Literal["ask", "search", "refuse"]:
    """根据 route 字段决定下一个节点"""
    return state["route"]


# ============================================================
# 4. 构建 Graph
# ============================================================
def build_agent():
    g = StateGraph(AgentState)

    # 加节点
    g.add_node("route", route_node)
    g.add_node("ask", ask_node)
    g.add_node("search", search_node)
    g.add_node("refuse", refuse_node)
    g.add_node("format", format_node)

    # 入口
    g.add_edge(START, "route")

    # 条件边：route → ask / search / refuse
    g.add_conditional_edges(
        "route",
        route_after_route,
        {"ask": "ask", "search": "search", "refuse": "refuse"},
    )

    # 三条路都汇聚到 format
    g.add_edge("ask", "format")
    g.add_edge("search", "format")
    g.add_edge("refuse", "format")

    g.add_edge("format", END)

    return g.compile()


# ============================================================
# 5. 对外接口
# ============================================================
_AGENT = None


def run_agent(question: str) -> dict:
    """
    跑一次 Agent。

    返回：
        {
            "answer": str,
            "sources": list,
            "route": str,
            "trace": list,
            "elapsed_ms": int
        }
    """
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()

    t0 = time.time()
    result = _AGENT.invoke({"question": question, "trace": []})
    elapsed = int((time.time() - t0) * 1000)

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "route": result.get("route", ""),
        "route_reason": result.get("route_reason", ""),
        "trace": result.get("trace", []),
        "elapsed_ms": elapsed,
        "degraded": result.get("degraded", False),
        "degrade_reason": result.get("degrade_reason", ""),
    }


# ============================================================
# 6. 测试
# ============================================================
TEST_CASES = [
    ("河南专升本考试考哪几科？", "ask", "可答·考试科目"),
    ("退役大学生士兵免试专升本怎么报名？", "ask", "可答·免试"),
    ("2026年河南专升本录取控制分数线是多少？", "search", "列表·分数线"),
    ("英语怎么从60提到100？", "refuse", "超范围·备考策略"),
    ("我考了230分能上公办吗？", "refuse", "超范围·个人分数"),
]


def main():
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"[问题] {q}\n")
        r = run_agent(q)
        print(f"\n[答案]\n{r['answer']}")
        print(f"\n[路由] {r['route']}（{r['route_reason'][:50]}）")
        print(f"[来源] {len(r['sources'])} 条")
        print(f"[轨迹] {' → '.join(r['trace'])}")
        print(f"[耗时] {r['elapsed_ms']}ms")
        if r.get("degraded"):
            print(f"[⚠️ 降级] {r.get('degrade_reason', '')}")
        return

    print("=" * 78)
    print(f"Agent 端到端测试（5 个用例）| 工具模式：{TOOLS_MODE}")
    print("=" * 78)

    passed = 0
    for i, (q, expect, desc) in enumerate(TEST_CASES, 1):
        print(f"\n{'─' * 78}")
        print(f"[用例 {i}] {desc}")
        print(f"  问题：{q}")

        r = run_agent(q)

        ok = r["route"] == expect
        passed += ok

        print(f"  答案：{r['answer'][:120].replace(chr(10), ' ')}{'...' if len(r['answer']) > 120 else ''}")
        dag = f" | ⚠️降级({r.get('degrade_reason', '')[:20]})" if r.get("degraded") else ""
        print(f"  {'✅' if ok else '❌'} 路由={r['route']}（期望 {expect}）| 来源 {len(r['sources'])} 条 | {r['elapsed_ms']}ms{dag}")
        print(f"  轨迹：{' → '.join(r['trace'])}")

    print(f"\n{'=' * 78}")
    print(f"通过：{passed}/{len(TEST_CASES)}")
    print("=" * 78)

    print("""
自检：LangGraph 相比直接写 if-else 调用，优势在哪？

1. 可观测性
   - 每步 State 都能看到（trace 字段记录了完整路径）
   - 出问题能定位到具体节点
   - if-else 里只能靠 print 调试

2. 可扩展性
   - 加一个新节点只需 add_node + 连边，不改原有逻辑
   - if-else 加分支要动主函数，容易改坏

3. 可中断/可恢复
   - LangGraph 支持 checkpoint（W10 会用到）
   - 可以中途暂停、人工审核后继续
   - if-else 做不到

4. 条件边是声明式的
   - 路由逻辑（route_after_route）和流程（add_conditional_edges）分离
   - 改路由规则不影响图结构

5. 并行执行
   - LangGraph 支持多节点并行（W9 会用）
   - if-else 要自己写线程/协程

代价：
   - 多一层抽象，简单场景显得重
   - 需要理解 State/Node/Edge 概念
   - 调试时调用栈更深

结论：3 个以内分支，if-else 更快；一旦要加记忆、并行、人工审核，LangGraph 的价值就出来了。
""")


if __name__ == "__main__":
    main()
