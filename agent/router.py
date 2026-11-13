# -*- coding: utf-8 -*-
"""
router.py —— 决策路由（W8D3）

Agent 的"大脑"：判断问题该走哪条路。

三条路：
    ask     → 调 ask_rag（检索+生成，给完整答案）        ~8s
    search  → 调 search_docs（只要原文列表，不生成）      ~1.6s
    refuse  → 直接拒答（不调工具，省时间省钱）            ~0ms

设计原则：
    1. 规则式，不用 LLM（快、便宜、可控、可解释）
    2. 拒答优先级最高（超范围问题不该浪费一次 LLM 调用）
    3. 列表类优先于问答类（用户要清单时，原文比生成有用）

用法：
    python agent/router.py         # 跑 10 道测试题，看准确率
"""
import re
import time
from typing import Literal

from tools import ask_rag, health_check, search_docs

Route = Literal["ask", "search", "refuse"]


# ============================================================
# 规则定义
# ============================================================
# ⚠️ 顺序很重要：先判断拒答，再判断列表，最后默认问答

# 规则 1：超范围 → 直接拒答（优先级最高）
REFUSE_PATTERNS = [
    # 备考策略（"怎么…提高/提分"允许中间有其他字符）
    r"怎么(才)?(能)?[^，。？]{0,8}(学|复习|准备|提高|提分|背|刷题|进步)",
    r"(提分|提高成绩|涨分|冲\d+分?)",
    r"押题|真题|模拟卷|刷题|练习册",
    r"先抓哪个|先刷哪个|怎么安排时间|时间怎么分配",
    r"(英语|数学|语文|专业课)[^，。？]{0,10}(怎么|如何)[^，。？]{0,8}(提|涨|学|练)",
    # 生活/事务
    r"酒店|住宿|吃饭|自习室|图书馆",
    r"准考证.*(打印|在哪)|档案|团关系",
    # 院校体验/个人决策
    r"值不值|该不该|要不要|后悔",
    r"课多不多|会不会跟不上|老师怎么样",
    # 预测
    r"会(涨|降|变)吗|会不会涨|预测",
    # 个人分数能否上
    r"我.{0,6}(\d{2,3})\s*分.{0,10}(能上|够|行|有戏)",
]

# 规则 2：要列表/原文 → search_docs（不生成）
SEARCH_PATTERNS = [
    r"(有哪些|哪些|列一下|列举|都有什么|都是啥)",
    r"(列表|清单|名单|一览)",
    r"(分数|分数线).*(多少|几)",
    r"多少个?(专业|学校|院校)",
    r"能报(考)?(哪些|什么)",
    r"招(收)?哪些",
    r"录取控制分数线",
]

# 规则 3：默认 → ask_rag（完整答案）
# 不写规则，是兜底


def route(question: str) -> Route:
    """
    判断问题该走哪条路。

    参数：
        question: 用户问题原文

    返回：
        "ask"    → 调 ask_rag
        "search" → 调 search_docs
        "refuse" → 直接拒答
    """
    q = (question or "").strip()
    if not q:
        return "refuse"

    # ① 先判拒答（超范围不该浪费 LLM 调用）
    for pat in REFUSE_PATTERNS:
        if re.search(pat, q):
            return "refuse"

    # ② 再判搜索（要列表/原文）
    for pat in SEARCH_PATTERNS:
        if re.search(pat, q):
            return "search"

    # ③ 默认问答
    return "ask"


def explain(question: str) -> dict:
    """返回路由结果 + 命中的规则（用于调试和解释）"""
    q = (question or "").strip()
    if not q:
        return {"route": "refuse", "reason": "空问题"}
    for i, pat in enumerate(REFUSE_PATTERNS):
        if re.search(pat, q):
            return {"route": "refuse", "reason": f"拒答规则#{i+1}: {pat}"}
    for i, pat in enumerate(SEARCH_PATTERNS):
        if re.search(pat, q):
            return {"route": "search", "reason": f"搜索规则#{i+1}: {pat}"}
    return {"route": "ask", "reason": "默认（未命中任何规则）"}


# ============================================================
# 测试用例（10 道：5 可答 / 2 列表 / 3 超范围）
# ============================================================
TEST_CASES = [
    # (问题, 期望路由, 说明)
    # ---- 5 道可答（→ ask）----
    ("河南专升本考试考哪几科？", "ask", "可答·考试科目"),
    ("2026年报名时间是什么时候？", "ask", "可答·报名时间"),
    ("退役大学生士兵免试专升本怎么报名？", "ask", "可答·免试"),
    ("专升本考上后学制几年？", "ask", "可答·学制"),
    ("招生章程是什么？", "ask", "可答·章程"),

    # ---- 2 道列表（→ search）----
    ("2026年河南专升本录取控制分数线是多少？", "search", "列表·分数线"),
    ("河南专升本能报哪些专业？", "search", "列表·专业清单"),

    # ---- 3 道超范围（→ refuse）----
    ("英语怎么从60提到100？", "refuse", "超范围·备考策略"),
    ("我考了230分能上公办吗？", "refuse", "超范围·个人分数"),
    ("为了公办再战一年值不值？", "refuse", "超范围·个人决策"),
]


def test_routing():
    """只测路由（不调工具），快"""
    print("=" * 78)
    print("路由准确率测试（不调用工具）")
    print("=" * 78)
    print(f"{'问题':<32} {'期望':<7} {'实际':<7} {'结果'}")
    print("-" * 78)

    correct = 0
    for q, expect, desc in TEST_CASES:
        got = route(q)
        ok = got == expect
        correct += ok
        mark = "✅" if ok else "❌"
        print(f"{q[:30]:<32} {expect:<7} {got:<7} {mark}  {desc}")
        if not ok:
            e = explain(q)
            print(f"    └─ 实际原因: {e['reason']}")

    acc = correct / len(TEST_CASES) * 100
    print("-" * 78)
    print(f"准确率：{correct}/{len(TEST_CASES)} = {acc:.0f}%")
    return correct, len(TEST_CASES)


def test_end_to_end():
    """测完整链路（真的调工具），慢，但能看到效果"""
    print("\n" + "=" * 78)
    print("端到端测试（真的调工具，会花时间）")
    print("=" * 78)

    for q, expect, desc in TEST_CASES:
        r = route(q)
        t0 = time.time()

        if r == "refuse":
            result = {"answer": "未找到相关信息。本系统仅回答河南专升本政策类问题。", "refused": True}
            summary = "直接拒答（0 次工具调用）"
        elif r == "search":
            res = search_docs(q, top_k=3)
            result = res
            summary = f"search_docs → {res.get('total', 0)} 条 / {res.get('latency_ms')}ms" if res["ok"] else f"失败: {res['error']}"
        else:
            res = ask_rag(q)
            result = res
            summary = f"ask_rag → {len(res.get('sources', []))} 来源 / {res.get('latency_ms')}ms" if res["ok"] else f"失败: {res['error']}"

        elapsed = int((time.time() - t0) * 1000)
        mark = "✅" if r == expect else "❌"
        print(f"\n{mark} [{desc}] {q}")
        print(f"   路由: {r}  |  {summary}  |  总 {elapsed}ms")

        if r == "ask" and result.get("ok"):
            print(f"   答案: {result.get('answer', '')[:80]}...")
        elif r == "search" and result.get("ok"):
            for i, h in enumerate(result.get("hits", [])[:2], 1):
                print(f"   [{i}] {h['chunk_id']} [{h['doc_type']}] {h['text'][:50]}...")


if __name__ == "__main__":
    import sys

    # 先检查服务
    h = health_check()
    if not h["ok"]:
        print(f"⚠️ RAG 服务不可达：{h.get('error')}")
        print("   本地启动：venv\\Scripts\\python.exe -m uvicorn api_service:app --port 8000")
        print("   或用公网：$env:RAG_BASE='http://43.128.31.138'")
        sys.exit(1)

    print(f"[服务] healthy | {h['chunks_count']} 块\n")

    # 1. 先测路由（快，纯逻辑）
    correct, total = test_routing()

    # 2. 如果路由准确率 >= 70%，跑端到端
    if correct / total >= 0.7:
        print("\n路由准确率达标，跑端到端测试（预计 30-60 秒）...")
        test_end_to_end()
    else:
        print("\n⚠️ 路由准确率低于 70%，先调规则，不跑端到端")

    print("\n" + "=" * 78)
    print("自检：规则式和 LLM 式路由各有什么优缺点？")
    print("=" * 78)
    print("""
【规则式】（当前方案）
优点：
  - 快：microsecond 级，不花 API 钱
  - 可控：每条规则可解释，出问题能定位
  - 稳定：同样输入永远同样输出
  - 可测试：能写单测
缺点：
  - 覆盖不全：没写到的说法就漏
  - 维护成本：问题类型变多时规则爆炸
  - 不懂语义："提分"和"提高成绩"要写两条

【LLM 式】（W9 再考虑）
优点：
  - 泛化强：没见过的问题也能judge
  - 不用写规则
  - 能处理多意图（"考几科以及报名时间"）
缺点：
  - 慢：每次 1-2 秒（等于多一次 LLM 调用）
  - 贵：每次路由都花钱
  - 不稳定：temperature 不为 0 时可能飘
  - 难测试：输出不确定

【结论】
  W8 用规则式（够用、快、可控）
  W9 如果规则维护不过来，再考虑"规则优先 + LLM 兜底"的混合方案
""")
