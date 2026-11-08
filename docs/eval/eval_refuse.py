#!/usr/bin/env python
# -*- coding: utf-8 -*-
# docs/eval/eval_refuse.py —— W6D2 拒答能力评估
"""
用法：
    python docs/eval/eval_refuse.py --smoke        # 1 条，验证链路
    python docs/eval/eval_refuse.py                # 全量（16 条 refuse_set + 3 条原题）
    python docs/eval/eval_refuse.py --show-answer  # 额外打印每条的回答全文（人工复核用）

指标定义（写进 metrics.md）：
    拒答准确率 = 正确拒答题数 / 拒答题总数

    "正确拒答" = 系统回答中不含实质性答案，且命中拒答关键词之一
    拒答关键词：未找到相关信息 / 没有相关信息 / 无法回答 / 资料中未找到 /
               资料中暂无 / 暂未找到 / 抱歉 / 参考资料中未 / 提供的参考资料中未 /
               无法确定 / 超出范围

    两组分开报（口径不同）：
      语料缺失组（R1-R5）：补文档后应变可答，当前应拒答
      超范围组（R6-R16 + Q12/Q35/Q40）：永远应拒答，这是产品边界

输出：
    docs/eval/refuse_report.md
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import jieba
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

load_dotenv()

TOP_K = 5
BASE = Path(__file__).resolve().parent
CHUNKS_PATH = Path(os.environ.get("CHUNKS_FILE", BASE.parent.parent / "docs" / "chunks" / "chunks_fixed_v2.jsonl"))
QUESTIONS_PATH = Path(os.environ.get("EVAL_FILE", BASE / "questions.all2.jsonl"))
ANNOTATED_PATH = Path(os.environ.get("ANNOTATED_FILE", BASE / "questions_annotated.jsonl"))
REFUSE_PATH = Path(os.environ.get("REFUSE_FILE", BASE / "refuse_set.jsonl"))
OUT_MD = BASE / "refuse_report.md"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION", "chunks")
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"
LLM_URL = "https://api.siliconflow.cn/v1/chat/completions"
GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek-ai/DeepSeek-V3")
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

# 拒答关键词（命中任一即视为"表达了不知道"）
REFUSE_WORDS = [
    "未找到相关信息", "没有相关信息", "无法回答", "资料中未找到", "资料中暂无",
    "暂未找到", "抱歉", "参考资料中未", "提供的参考资料中未", "无法确定",
    "超出范围", "无法提供", "不清楚", "无法判断", "没有找到", "暂无相关",
]

# 反问/引导词（即使命中拒答词，如果同时在编造具体信息，仍算失败）
HALLUCINATION_MARKERS = [
    r"\d{4}年\d{1,2}月\d{1,2}日",   # 编造的具体日期
    r"\d+分",                       # 编造的具体分数
    r"第[一二三四五六七八九十]+条",  # 编造的法条
]

PROMPT = """你是专升本政策问答助手。请严格根据以下材料回答问题。

**重要规则**：
- 如果材料中没有能直接回答问题所需的信息，必须只回答「未找到相关信息」，不要基于常识推测、不要编造
- 不要回答与专升本政策无关的问题（如备考方法、生活建议、个人决策）
- 不要给出材料中没有的具体日期、分数、金额

材料：
{ctx}

问题：{question}

回答："""


# ---------------- 工具 ----------------
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def norm(s):
    return "".join(str(s or "").lower().split())


def is_refuse_text(answer):
    """判断回答是否表达了'不知道'"""
    a = norm(answer)
    return any(norm(w) in a for w in REFUSE_WORDS)


def has_hallucination(answer):
    """检测是否编造了具体信息"""
    hits = []
    for pat in HALLUCINATION_MARKERS:
        if re.search(pat, answer or ""):
            # 排除"未找到 2026年3月1日 的信息"这种（拒答词在前的）
            if not is_refuse_text(answer):
                hits.append(pat)
    return hits


def embed_one(text):
    r = requests.post(EMBED_URL,
                      headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json={"model": EMBED_MODEL, "input": [text]}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def qdrant_ranked(client, vec, k):
    try:
        pts = client.query_points(collection_name=COLLECTION, query=vec, limit=k,
                                  with_payload=["chunk_id", "text", "source"]).points
    except AttributeError:
        pts = client.search(collection_name=COLLECTION, query_vector=vec, limit=k)
    out = []
    for h in pts:
        p = getattr(h, "payload", None) or {}
        out.append({"chunk_id": str(p.get("chunk_id") or h.id),
                    "text": p.get("text", ""), "source": p.get("source", "")})
    return out


def generate(question, docs):
    ctx = "\n\n".join(docs[:TOP_K]) if docs else "（无相关材料）"
    try:
        r = requests.post(LLM_URL,
                          headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                          json={"model": GEN_MODEL,
                                "messages": [{"role": "user", "content": PROMPT.format(ctx=ctx, question=question)}],
                                "temperature": 0}, timeout=60)
        if r.status_code >= 400:
            print(f"[LLM错误] {r.status_code} {r.text[:150]}")
            return f"[生成失败 {r.status_code}]"
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[生成失败: {e}]"


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show-answer", action="store_true", help="打印每条回答全文")
    args = ap.parse_args()
    if not API_KEY:
        sys.exit("[错误] 缺少 SILICONFLOW_API_KEY")

    chunks = [c for c in load_jsonl(CHUNKS_PATH) if c.get("text")]
    print(f"[数据] chunks={len(chunks)}")

    # 1) 新拒答集
    refuse_items = load_jsonl(REFUSE_PATH)
    for r in refuse_items:
        r["group"] = "语料缺失" if r.get("category") == "语料缺失" else "超范围"

    # 2) 原评估集里的拒答题（Q12/Q35/Q40）
    qs = load_jsonl(QUESTIONS_PATH)
    ann = {a["id"]: a for a in load_jsonl(ANNOTATED_PATH)}
    for q in qs:
        src = str(q.get("expected_source") or "")
        if ("超出" in src) or ("拒答" in src):
            refuse_items.append({
                "id": q.get("id"), "question": q["question"],
                "reason": "原评估集标注为超范围",
                "category": "超范围", "group": "超范围",
                "source_question": q["question"],
                "expected_behavior": "拒答：超出河南专升本政策范围",
            })

    if args.smoke:
        refuse_items = refuse_items[:1]
    elif args.limit:
        refuse_items = refuse_items[: args.limit]
    print(f"[拒答集] {len(refuse_items)} 条"
          f"（语料缺失 {sum(1 for r in refuse_items if r['group'] == '语料缺失')} / "
          f"超范围 {sum(1 for r in refuse_items if r['group'] == '超范围')}）\n")

    client = QdrantClient(url=QDRANT_URL)
    rows = []
    ok_by_group = {"语料缺失": [0, 0], "超范围": [0, 0]}

    for item in refuse_items:
        qid, qtext = item.get("id"), item.get("question")
        vec = embed_one(qtext)
        hits = qdrant_ranked(client, vec, TOP_K)
        ans = generate(qtext, [h["text"] for h in hits])
        refused = is_refuse_text(ans)
        hall = has_hallucination(ans)
        correct = refused and not hall

        g = item["group"]
        ok_by_group[g][1] += 1
        if correct:
            ok_by_group[g][0] += 1

        rows.append(dict(id=qid, question=qtext, group=g, category=item.get("category"),
                         answer=ans, refused=refused, hallucination=hall, correct=correct,
                         reason=item.get("reason"), source=item.get("source_question")))
        mark = "✅" if correct else "❌"
        print(f"{mark} {qid:4s} [{g}] 拒答词={'是' if refused else '否'} 编造={'是' if hall else '否'} | {qtext[:32]}")
        if args.show_answer or not correct:
            print(f"     回答：{ans[:150]}")

    total_n = len(rows)
    total_ok = sum(1 for r in rows if r["correct"])
    print(f"\n=== 拒答准确率 ===")
    for g, (ok, n) in ok_by_group.items():
        if n:
            print(f"  {g:6s}  {ok}/{n} = {ok / n * 100:5.1f}%")
    print(f"  {'合计':6s}  {total_ok}/{total_n} = {total_ok / max(total_n, 1) * 100:5.1f}%")

    failed = [r for r in rows if not r["correct"]]
    if failed:
        print(f"\n=== 失败 {len(failed)} 条（重点看这里） ===")
        for r in failed:
            why = "编造了具体信息" if r["hallucination"] else ("没表达不知道" if not r["refused"] else "未知")
            print(f"  {r['id']}: {why}")
            print(f"    问：{r['question'][:50]}")
            print(f"    答：{r['answer'][:120]}")

    # 报告
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(f"# 拒答能力评估报告\n\n- 日期：{__import__('time').strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- 拒答模型：`{GEN_MODEL}` / 检索：向量 top{TOP_K}\n\n")
        f.write("## 结果\n\n| 分组 | 正确拒答 | 准确率 |\n|---|---|---|\n")
        for g, (ok, n) in ok_by_group.items():
            f.write(f"| {g} | {ok}/{n} | {ok / max(n, 1) * 100:.1f}% |\n")
        f.write(f"| **合计** | **{total_ok}/{total_n}** | **{total_ok / max(total_n, 1) * 100:.1f}%** |\n")
        f.write("\n## 明细\n\n| 题号 | 分组 | 问题 | 拒答 | 编造 | 结论 |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['id']} | {r['group']} | {r['question'][:30]} | "
                    f"{'是' if r['refused'] else '否'} | {'是' if r['hallucination'] else '否'} | "
                    f"{'✅' if r['correct'] else '❌'} |\n")
        if failed:
            f.write("\n## 失败样例\n\n")
            for r in failed:
                f.write(f"### {r['id']}：{r['question']}\n\n")
                f.write(f"- 来源问题：{r['source']}\n- 应拒答原因：{r['reason']}\n")
                f.write(f"- 系统回答：\n\n> {r['answer']}\n\n")
    print(f"\n[输出] {OUT_MD}")


if __name__ == "__main__":
    main()
