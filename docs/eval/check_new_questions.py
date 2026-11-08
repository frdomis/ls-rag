#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
check_new_questions.py —— 核查新题在语料里是否真有答案
用法：
    python check_new_questions.py            # 检查候选文件里全部新题
    python check_new_questions.py --json     # 只输出 JSON（便于追加到评估集）

判定三档：
    ✅ 有答案     期望要点能在语料里找到（4-gram 匹配 ≥ 60%）
    ⚠️ 部分有     找到 30%-60%，需要人工看是不是同一个意思
    ❌ 无答案     找到 < 30%，不该进评估集（或该进拒答集）

输出文件：
    _new_questions_check.md        人类可读报告
    _new_questions_verdict.json    机器可读结论
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

CHUNKS = Path(r"D:\vscode\ls-rag\docs\chunks\chunks_fixed_v2.jsonl")

# 待核查的新题（来自真实提问，见 user_questions.md）
CANDIDATES = [
    {
        "id": "N1",
        "question": "跨专业报考审核要交什么材料？",
        "expected_source": "河南省教育厅",
        "expected_keypoints": ["跨专业报考需符合专业对应目录", "审核由报考院校负责"],
    },
    {
        "id": "N2",
        "question": "建档立卡资格咋认定，对普通批分数线影响大吗？",
        "expected_source": "河南省教育厅",
        "expected_keypoints": ["建档立卡考生可享受专升本专项计划", "专项计划单独划定分数线"],
    },
    {
        "id": "N3",
        "question": "河南专升本平行志愿真能保底吗，咋有人说还能滑档？",
        "expected_source": "河南省教育厅",
        "expected_keypoints": ["平行志愿按分数优先原则投档", "不服从调剂或计划已满可能退档"],
    },
    {
        "id": "N4",
        "question": "报名信息填错了还能改吗，截止前咋操作？",
        "expected_source": "河南省教育厅",
        "expected_keypoints": ["报名信息需在报名截止前修改", "具体操作以省教育考试院公告为准"],
    },
    {
        "id": "N5",
        "question": "民办联合培养和普通民办咋选，毕业证一样不？",
        "expected_source": "各院校招生章程",
        "expected_keypoints": ["联合培养与普通招生的毕业证书相同", "培养地点和方式不同"],
    },
]


def norm(s):
    return re.sub(r"[\s\u3000]+", "", unicodedata.normalize("NFKC", s or "").lower())


def grams(text, n=4):
    t = norm(text)
    if len(t) <= n:
        return [t] if t else []
    return [t[i:i + n] for i in range(0, len(t) - n + 1, 2)]


def find_hits(kp, chunks):
    """返回包含该要点片段的 chunk 列表（按命中片段数降序）"""
    gs = grams(kp)
    if not gs:
        return []
    scored = []
    for c in chunks:
        t = norm(c["text"])
        matched = sum(1 for g in gs if g in t)
        if matched:
            scored.append((matched / len(gs), matched, len(gs), c))
    scored.sort(key=lambda x: -x[0])
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not CHUNKS.exists():
        sys.exit(f"[错误] 找不到 {CHUNKS}")
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8") if l.strip()]
    print(f"[数据] {len(chunks)} 块\n")

    verdicts, lines = [], ["# 新题语料核查报告\n"]
    for q in CANDIDATES:
        lines.append(f"## {q['id']}：{q['question']}\n")
        kp_results, total_ratio = [], 0.0
        for kp in q["expected_keypoints"]:
            hits = find_hits(kp, chunks)
            ratio = hits[0][0] if hits else 0.0
            total_ratio += ratio
            status = "✅" if ratio >= 0.6 else ("⚠️" if ratio >= 0.3 else "❌")
            lines.append(f"- {status} 要点：{kp}")
            lines.append(f"  - 最高匹配度 {ratio * 100:.0f}%" + (f"（{hits[0][1]}/{hits[0][2]} 片段）" if hits else ""))
            if hits and ratio > 0:
                for r, m, n, c in hits[:2]:
                    snippet = c["text"][:100].replace("\n", " ")
                    lines.append(f"    - `{c['id']}` [{c.get('doc_type', '?')}] {c.get('source', '?')}：{snippet}…")
            kp_results.append({"keypoint": kp, "ratio": round(ratio, 2),
                               "chunk_ids": [h[3]["id"] for h in hits[:3]]})
        avg = total_ratio / max(len(q["expected_keypoints"]), 1)
        verdict = "✅ 有答案" if avg >= 0.6 else ("⚠️ 部分有" if avg >= 0.3 else "❌ 无答案")
        print(f"{verdict}  {q['id']} 平均匹配 {avg * 100:.0f}% | {q['question'][:30]}")
        lines.append(f"\n**结论：{verdict}（平均 {avg * 100:.0f}%）**\n")
        verdicts.append({
            "id": q["id"], "question": q["question"], "verdict": verdict,
            "avg_ratio": round(avg, 2),
            "expected_source": q["expected_source"],
            "keypoints": kp_results,
            "chunk_ids": sorted({cid for k in kp_results for cid in k["chunk_ids"]}),
            "pass": avg >= 0.6,
        })

    out_md = Path("_new_questions_check.md")
    out_md.write_text("\n".join(lines) + "\n\n---\n\n## 汇总\n\n" +
                      "\n".join(f"- **{v['id']}** {v['verdict']}（{v['avg_ratio'] * 100:.0f}%）"
                                f" → {'进评估集' if v['pass'] else '进拒答集'}" for v in verdicts) + "\n",
                      encoding="utf-8")
    out_json = Path("_new_questions_verdict.json")
    out_json.write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps([v for v in verdicts if v["pass"]], ensure_ascii=False, indent=2))
    else:
        print(f"\n[输出] {out_md}")
        print(f"[输出] {out_json}")
        ok = [v["id"] for v in verdicts if v["pass"]]
        bad = [v["id"] for v in verdicts if not v["pass"]]
        print(f"\n✅ 可进评估集：{ok or '无'}")
        print(f"❌ 应进拒答集：{bad or '无'}")
        print("\n[下一步] 打开 md 报告人工确认；确认后把 pass=true 的题加进 questions.all2.jsonl 并做块级标注")


if __name__ == "__main__":
    main()
