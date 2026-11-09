# -*- coding: utf-8 -*-
"""
demo_graph.py —— LangGraph 最小可运行示例（W8D1）

核心三件套：
    State  贯穿流程的数据容器（TypedDict）
    Node   一个函数，读 State、返回要更新的字段
    Edge   连接节点，决定下一个跑谁

运行：
    python agent/demo_graph.py
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START


# ============================================================
# 1. 定义 State（贯穿整个图的数据）
# ============================================================
# 注意：State 只是"声明有哪些字段"，不是"数据本身"
# 每次 invoke 传入的初始 dict 就是这份 State 的初值
class State(TypedDict):
    question: str      # 用户问题
    route: str         # 路由结果（后面 D3 会用到）
    answer: str        # 最终答案
    steps: list        # 执行轨迹（用于观察发生了什么）


# ============================================================
# 2. 定义 Node（每个节点是一个函数）
# ============================================================
# 关键点：Node 返回的 dict 会被【合并】进 State，不是【替换】
# 只返回要更新的字段即可，其他字段保持不变
def node_a(state: State) -> dict:
    """第一个节点：预处理"""
    print(f"  [node_a] 收到问题：{state['question']}")
    return {
        "question": state["question"].strip(),
        "steps": state.get("steps", []) + ["node_a"],
    }


def node_b(state: State) -> dict:
    """第二个节点：模拟路由判断"""
    q = state["question"]
    if "考" in q or "科目" in q:
        route = "exam"
    elif "报名" in q or "时间" in q:
        route = "schedule"
    else:
        route = "unknown"
    print(f"  [node_b] 路由判断：{route}")
    return {
        "route": route,
        "steps": state.get("steps", []) + ["node_b"],
    }


def node_c(state: State) -> dict:
    """第三个节点：根据 route 生成答案（这是"条件边"之后的部分）"""
    print(f"  [node_c] 按 route={state['route']} 生成答案")
    answers = {
        "exam": "专升本考英语和专业综合两科。",
        "schedule": "报名时间以省教育考试院通知为准。",
        "unknown": "未找到相关信息。",
    }
    return {
        "answer": answers.get(state["route"], "未找到相关信息。"),
        "steps": state.get("steps", []) + ["node_c"],
    }


# ============================================================
# 3. 构建 Graph（连节点 + 连边）
# ============================================================
def build_simple_graph():
    """最简单的线性图：a → b → c → END"""
    g = StateGraph(State)

    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_node("c", node_c)

    g.add_edge(START, "a")     # 入口
    g.add_edge("a", "b")       # a 跑完跑 b
    g.add_edge("b", "c")       # b 跑完跑 c
    g.add_edge("c", END)       # c 跑完结束

    return g.compile()


# ============================================================
# 4. 条件边（Conditional Edge）—— Agent 的核心
# ============================================================
def route_after_b(state: State) -> Literal["c", "skip"]:
    """
    条件路由函数：返回下一个节点名。
    这是 Agent "做决策" 的地方。
    """
    if state["route"] == "unknown":
        return "skip"
    return "c"


def node_skip(state: State) -> dict:
    """跳过节点：直接拒答，不调工具"""
    print("  [node_skip] 未知问题 → 直接拒答")
    return {
        "answer": "未找到相关信息。本系统仅回答河南专升本政策类问题。",
        "steps": state.get("steps", []) + ["node_skip"],
    }


def build_conditional_graph():
    """带条件分支的图：a → b → (c 或 skip) → END"""
    g = StateGraph(State)

    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_node("c", node_c)
    g.add_node("skip", node_skip)

    g.add_edge(START, "a")
    g.add_edge("a", "b")

    # ⭐ 条件边：根据 b 的结果决定去 c 还是 skip
    g.add_conditional_edges("b", route_after_b, {
        "c": "c",
        "skip": "skip",
    })

    g.add_edge("c", END)
    g.add_edge("skip", END)

    return g.compile()


# ============================================================
# 5. 运行测试
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("测试 1：线性图（a → b → c）")
    print("=" * 70)
    app1 = build_simple_graph()
    result = app1.invoke({"question": "  专升本考试考哪几科？  ", "steps": []})
    print(f"\n最终结果：")
    print(f"  question: {result['question']!r}")
    print(f"  route:    {result['route']}")
    print(f"  answer:   {result['answer']}")
    print(f"  steps:    {result['steps']}")

    print("\n" + "=" * 70)
    print("测试 2：条件图（可答 → c，未知 → skip）")
    print("=" * 70)

    app2 = build_conditional_graph()

    test_cases = [
        "专升本考试考哪几科？",       # 可答 → exam → c
        "2026年报名时间是什么时候？",  # 可答 → schedule → c
        "英语怎么从60提到100？",       # 超范围 → unknown → skip
    ]

    for i, q in enumerate(test_cases, 1):
        print(f"\n--- 用例 {i}：{q}")
        r = app2.invoke({"question": q, "steps": []})
        print(f"  结果：route={r['route']} | answer={r['answer'][:30]}...")
        print(f"  轨迹：{r['steps']}")
