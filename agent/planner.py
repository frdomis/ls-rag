# -*- coding: utf-8 -*-
"""
planner.py —— 任务规划：把复杂问题拆成子任务（W9D2）

核心思想：
    一个问题如果同时问了好几件事，一次性回答会丢信息；
    拆成独立子问题，分别检索，再合并，质量更高。

拆解维度（4 个）：
    professional  专业     "我是XX专业" / "XX专业能报什么"
    score         分数     "230分" / "多少分能上"
    school        院校     "哪些学校" / "公办还是民办"
    policy        政策     单一问题，不拆

三条拆解规则：
    规则 1（专业+分数）：问"我能报什么" → 拆成 [专业维度, 分数维度]
    规则 2（院校+专业）：问"哪些学校招XX" → 拆成 [招生院校, 办学性质]
    规则 3（多问句）：含"？"多处或"和/以及/还有"连接 → 按句拆

用法：
    python agent/planner.py       # 跑 5 个测试用例
"""
import re
from typing import TypedDict


class SubTask(TypedDict):
    """一个子任务"""
    dim: str        # 维度：professional / score / school / policy
    question: str   # 改写后的子问题（可直接拿去检索）
    source: str     # 从原问题的哪个片段拆出来的（可解释性）


# ============================================================
# 提取器（从原问题里抽出关键信息）
# ============================================================
# 常见专科专业名（用于识别"我是XX专业"）
MAJOR_PATTERNS = [
    r"(?:我是|学的|读的|专科是|专业是)?([\u4e00-\u9fa5]{2,12}(?:技术|工程|管理|学|教育|设计|护理|会计|商务|贸易|营销|物流|英语|数学))\s*专业",
    r"([\u4e00-\u9fa5]{2,10})\s*专业",
]

# 分数
SCORE_RE = re.compile(r"(\d{2,3})\s*分")

# 院校意向
SCHOOL_HINTS = [
    ("公办", "公办"),
    ("民办", "民办"),
    ("哪些学校", "学校"),
    ("哪些院校", "院校"),
    ("哪个学校", "学校"),
]


def extract_major(q: str) -> str:
    """抽取专业名，抽不到返回空串"""
    for pat in MAJOR_PATTERNS:
        m = re.search(pat, q)
        if m:
            name = m.group(1).strip()
            # 排除明显不是专业的词
            if name not in ("什么", "哪些", "哪个", "这个", "那个"):
                return name
    return ""


def extract_score(q: str) -> str:
    """抽取分数"""
    m = SCORE_RE.search(q)
    return m.group(1) if m else ""


def extract_school_intent(q: str) -> str:
    """抽取院校意向（公办/民办/学校）"""
    for kw, label in SCHOOL_HINTS:
        if kw in q:
            return label
    return ""


# ============================================================
# 拆解规则
# ============================================================
def plan(question: str) -> list[SubTask]:
    """
    把复杂问题拆成 1-4 个子任务。

    返回的 list 长度 == 1 表示不拆（单一问题）。
    """
    q = (question or "").strip()
    if not q:
        return [{"dim": "policy", "question": "", "source": ""}]

    major = extract_major(q)
    score = extract_score(q)
    school = extract_school_intent(q)

    subtasks: list[SubTask] = []

    # ---- 规则 1：专业 + 分数（"我是XX专业，230分能报什么"）----
    if major and score:
        subtasks.append({
            "dim": "professional",
            "question": f"{major}专业能报考哪些本科专业",
            "source": major,
        })
        subtasks.append({
            "dim": "score",
            "question": f"2026年河南专升本录取控制分数线是多少",
            "source": f"{score}分",
        })
        if school:
            subtasks.append({
                "dim": "school",
                "question": f"{school}院校的招生专业有哪些",
                "source": school,
            })
        return subtasks

    # ---- 规则 2：专业 + 院校意向（"计算机网络技术能报哪些公办学校"）----
    if major and school:
        subtasks.append({
            "dim": "professional",
            "question": f"{major}专业能报考哪些本科专业",
            "source": major,
        })
        subtasks.append({
            "dim": "school",
            "question": f"{school}院校的招生专业和录取分数线",
            "source": school,
        })
        return subtasks

    # ---- 规则 3：多问句（"考哪几科？每科多少分？"）----
    # 用 ？/。/；拆分，且每句长度 ≥ 4
    parts = [p.strip() for p in re.split(r"[？?。；;]|，���且|以及|还有", q) if len(p.strip()) >= 4]
    if len(parts) >= 2:
        return [
            {"dim": "policy", "question": p if p.endswith("？") else p + "？", "source": p}
            for p in parts[:4]     # 最多 4 个
        ]

    # ---- 默认：不拆 ----
    return [{"dim": "policy", "question": q, "source": q}]


def plan_with_reason(question: str) -> dict:
    """返回拆解结果 + 用的哪条规则（调试用）"""
    tasks = plan(question)
    major = extract_major(question)
    score = extract_score(question)
    school = extract_school_intent(question)

    if len(tasks) == 1:
        rule = "不拆（单一问题）"
    elif major and score:
        rule = "规则1：专业+分数"
    elif major and school:
        rule = "规则2：专业+院校"
    else:
        rule = "规则3：多问句"

    return {
        "question": question,
        "rule": rule,
        "extracted": {"major": major, "score": score, "school": school},
        "subtasks": tasks,
        "count": len(tasks),
    }


# ============================================================
# 测试（5 个复杂问题）
# ============================================================
TEST_CASES = [
    # (问题, 期望拆几段, 说明)
    ("我是计算机网络技术专业，230分，能报哪些公办学校？", 3, "专业+分数+院校（完整三维）"),
    ("计算机网络技术专业能报哪些公办学校？", 2, "专业+院校"),
    ("专升本考哪几科？每科多少分？", 2, "多问句"),
    ("2026年报名时间是什么时候？", 1, "单一问题（不该拆）"),
    ("免试需要什么条件？怎么报名？", 2, "多问句"),
]


def main():
    print("=" * 78)
    print("W9D2 任务拆解测试（5 个用例）")
    print("=" * 78)

    ok_count = 0
    for i, (q, expect_n, desc) in enumerate(TEST_CASES, 1):
        r = plan_with_reason(q)
        ok = r["count"] == expect_n
        ok_count += ok

        print(f"\n{'─' * 78}")
        print(f"[用例 {i}] {desc}")
        print(f"  原问题：{q}")
        print(f"  提取：专业={r['extracted']['major'] or '-'} | 分数={r['extracted']['score'] or '-'} | 院校={r['extracted']['school'] or '-'}")
        print(f"  规则：{r['rule']}")
        print(f"  拆成 {r['count']} 段：")
        for j, t in enumerate(r["subtasks"], 1):
            print(f"    {j}. [{t['dim']}] {t['question']}")
        print(f"  {'✅' if ok else '❌'} 期望 {expect_n} 段，实际 {r['count']} 段")

    print(f"\n{'=' * 78}")
    print(f"通过：{ok_count}/{len(TEST_CASES)}")
if __name__ == "__main__":
    main()
