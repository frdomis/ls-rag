#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
annotate_questions.py —— 生成 40 题的「块级标注」草稿
用法：
    python annotate_questions.py
输出：
    docs/eval/questions_annotated.jsonl          ← 草稿（你人工复核后改成正式版）
    docs/eval/questions_annotated.unmatched.md   ← 找不到候选的题（写明原因）

做法：对每题，按 expected_source 过滤候选 chunk，再用 expected_keypoints 的
      片段匹配打分，取前 3 个候选 chunk_id 供你确认。**必须人工复核**，别把草稿当答案。
"""
import json
import re
import os
import sys
import argparse

CHUNKS = os.environ.get("CHUNKS_FILE", "docs/chunks/chunks_fixed_v2.jsonl")
QUESTIONS = os.environ.get("EVAL_FILE", "docs/eval/questions.all2.jsonl")
OUT = os.environ.get("OUT_FILE", "docs/eval/questions_annotated.jsonl")
UNMATCHED = os.environ.get("UNMATCHED_FILE", "docs/eval/questions_annotated.unmatched.md")
TOP_CANDIDATES = 3


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm(s):
    return re.sub(r"\s+", "", (s or "")).lower()


def pieces(text, n=4):
    """把期望要点切成 4 字片段，用于宽松匹配。"""
    t = norm(text)
    return [t[i:i + n] for i in range(0, max(len(t) - n + 1, 1))]


def score_chunk(chunk, q):
    src = chunk.get("source", "")
    exp = q.get("expected_source", "")
    text = chunk.get("text", "")
    s = 0.0
    if exp and (exp in src or src in exp):
        s += 3.0
    kps = q.get("expected_keypoints") or []
    for kp in kps:
        for p in pieces(kp):
            if p and p in norm(text):
                s += 1.0
                break
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=CHUNKS)
    ap.add_argument("--eval", dest="eval_path", default=QUESTIONS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--unmatched", default=UNMATCHED)
    args = ap.parse_args()

    chunks = load_jsonl(args.chunks)
    qs = load_jsonl(args.eval_path)
    out, unmatched = [], []

    for q in qs:
        scored = sorted(
            ((score_chunk(c, q), c) for c in chunks),
            key=lambda x: -x[0],
        )
        cands = [(s, c) for s, c in scored[:TOP_CANDIDATES] if s > 0]
        if not cands:
            unmatched.append(q)
            continue
        out.append({
            "id": q.get("id"),
            "question": q.get("question") or q.get("query"),
            "expected_source": q.get("expected_source"),
            "chunk_ids": [c["id"] for _, c in cands],
            "confidence": ["high" if s >= 3 else "low" for s, _ in cands],
            "candidates": [
                {"chunk_id": c["id"], "score": round(s, 1),
                 "source": c.get("source", ""), "text": c.get("text", "")[:60].replace("\n", " ")}
                for s, c in cands
            ],
            "note": "",
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.unmatched, "w", encoding="utf-8") as f:
        f.write("# 找不到候选的题目（人工处理）\n\n")
        for q in unmatched:
            f.write(f"- {q.get('id')}: {q.get('question')} （expected_source={q.get('expected_source')}）\n")

    print(f"[输出] {args.out}：{len(out)}/{len(qs)} 题有候选")
    print(f"[输出] {args.unmatched}：{len(unmatched)} 题无候选")
    print("[下一步] 人工复核每题的 chunk_ids：删掉不相关的、补上正确的；high 的抽查即可，low 的逐条看")


if __name__ == "__main__":
    main()
