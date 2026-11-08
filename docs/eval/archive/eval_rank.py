#!/usr/bin/env python  第一次导入rerank
# -*- coding: utf-8 -*-
# docs/eval/eval_rerank.py —— W5D3 rerank 接入 + 对比
"""
用法：
    python docs/eval/eval_rerank.py --smoke     # 先冒烟：1 道题跑通 rerank API，看返回结构 + 耗时
    python docs/eval/eval_rerank.py             # 全量 33 题，对比 4 种配置，写 docs/eval/rerank_compare.md

对比配置（同一分母 33）：
    A 向量不过滤 top5               ← 线上基线
    B 向量过滤 top5                 ← W5D2 最优（24/33）
    C 过滤后 BM25+向量 并集 top20 → rerank → top5
    D 不过滤 BM25+向量 并集 top20 → rerank → top5

重点盯两件事：
    1) rerank 到底救回了哪些题（候选里本来就有答案 vs 根本没有）
    2) 延迟（P50/P95）和每千次成本
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

TOP_K = 5
CAND_K = 20          # 每路召回 20，并集最多 40
BASE = Path(__file__).resolve().parent
CHUNKS_PATH = Path(os.environ.get("CHUNKS_FILE", BASE.parent.parent / "docs" / "chunks" / "chunks_fixed_v2.jsonl"))
QUESTIONS_PATH = Path(os.environ.get("EVAL_FILE", BASE / "questions.all2.jsonl"))
ANNOTATED_PATH = Path(os.environ.get("ANNOTATED_FILE", BASE / "questions_annotated.jsonl"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION", "chunks")
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

YEAR_RE = re.compile(r"(20\d{2})")
DT_RULES = [
    (re.compile(r"分数线|投档线|控制线|控分线"), "score"),
    (re.compile(r"章程|简章"), "charter"),
    (re.compile(r"报名|考试|志愿|录取|免试|退役|建档|专项|政策|通知|时间|几科|学制|学费"), "policy"),
]


# ---------------- 工具 ----------------
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def is_refuse(q):
    src = str(q.get("expected_source") or "")
    return ("超出" in src) or ("拒答" in src)


def infer_filters(question, corpus_years):
    must = []
    years = YEAR_RE.findall(question or "")
    if years and years[0] in corpus_years:
        must.append({"key": "year", "value": years[0]})
    for pat, dt in DT_RULES:
        if pat.search(question or ""):
            must.append({"key": "doc_type", "value": dt})
            break
    return must


def to_qdrant_filter(must):
    if not must:
        return None
    conds = [models.FieldCondition(key=m["key"], match=models.MatchValue(value=m["value"])) for m in must]
    return models.Filter(must=conds)


def embed_one(text):
    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def qdrant_ranked(client, vec, k, must=None):
    """返回 [(chunk_id, score)]，已按分数降序"""
    kw = dict(collection_name=COLLECTION, query=vec, limit=k, with_payload=["chunk_id"])
    f = to_qdrant_filter(must)
    if f:
        kw["query_filter"] = f
    try:
        pts = client.query_points(**kw).points
    except AttributeError:
        kw.pop("query", None)
        kw["query_vector"] = vec
        pts = client.search(**kw)
    out = []
    for h in pts:
        p = getattr(h, "payload", None) or {}
        out.append((str(p.get("chunk_id") or h.id), float(getattr(h, "score", 0.0))))
    return out


def rerank(query, documents, top_n=TOP_K, timeout=20):
    """调硅基流动 rerank；返回 (results, latency_ms, 估算成本)"""
    payload = {"model": RERANK_MODEL, "query": query, "documents": documents,
               "top_n": top_n, "return_documents": False}
    t0 = time.time()
    r = requests.post(
        RERANK_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    latency = (time.time() - t0) * 1000
    if r.status_code >= 400:
        print(f"[rerank错误] {r.status_code} {r.text[:300]}")
        r.raise_for_status()
    return r.json().get("results", []), latency


def merge_candidates(bm25_ranked, vec_ranked, k=CAND_K):
    """并集去重，用 RRF 名次融合决定顺序（不是简单拼接）"""
    score = {}
    for ranked in (bm25_ranked, vec_ranked):
        for rank, (cid, _) in enumerate(ranked[:k], 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (60 + rank)
    return [cid for cid, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def pct(vals, p):
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))
    return vals[idx]


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑 1 题，验证 rerank API")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not API_KEY:
        sys.exit("[错误] 缺少 SILICONFLOW_API_KEY")

    import jieba
    from rank_bm25 import BM25Okapi

    chunks = [c for c in load_jsonl(CHUNKS_PATH) if c.get("text")]
    by_id = {str(c["id"]): c for c in chunks}
    corpus_years = {str(c.get("year", "")) for c in chunks}
    qs = load_jsonl(QUESTIONS_PATH)
    ann = {a["id"]: a for a in load_jsonl(ANNOTATED_PATH)} if ANNOTATED_PATH.exists() else {}
    for q in qs:
        a = ann.get(str(q.get("id")), {})
        q["_cls"] = {
            "chunk_ids": {str(c) for c in (a.get("chunk_ids") or [])},
            "refuse": is_refuse(q),
            "unanswerable": (not is_refuse(q)) and not (a.get("chunk_ids") or []),
        }
    ans = [q for q in qs if not q["_cls"]["refuse"] and not q["_cls"]["unanswerable"]]
    if args.smoke:
        ans = ans[:1]
    elif args.limit:
        ans = ans[: args.limit]
    print(f"[数据] chunks={len(chunks)} | 可答={len(ans)} | rerank 模型={RERANK_MODEL}")

    bm_all = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in chunks])

    def bm25_ranked(question, cand_chunks):
        bm = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in cand_chunks])
        scores = bm.get_scores(list(jieba.cut_for_search(question)))
        idx = sorted(range(len(scores)), key=lambda j: -scores[j])[:CAND_K]
        return [(str(cand_chunks[j]["id"]), float(scores[j])) for j in idx]

    client = QdrantClient(url=QDRANT_URL)
    stats = {k: 0 for k in ("A", "B", "C", "D")}
    lat_rr, lat_all, n_docs = [], [], []
    rows, saved_cases = [], []
    n = 0

    for q in ans:
        qid, text, cids = q.get("id"), q["question"], q["_cls"]["chunk_ids"]
        must = infer_filters(text, corpus_years)
        sub = [c for c in chunks if all(str(c.get(m["key"], "")) == m["value"] for m in must)] if must else chunks
        if must and not sub:
            sub = chunks
        vec = embed_one(text)

        # A 向量不过滤 top5
        a_ids = [cid for cid, _ in qdrant_ranked(client, vec, TOP_K)]
        # B 向量过滤 top5（过滤结果为空则回退）
        b_ranked = qdrant_ranked(client, vec, TOP_K, must) if must else qdrant_ranked(client, vec, TOP_K)
        if must and not b_ranked:
            b_ranked = qdrant_ranked(client, vec, TOP_K)
        b_ids = [cid for cid, _ in b_ranked]

        # C/D 召回 top20 并集 → rerank
        t_in0 = time.time()
        for key, cand_chunks, use_filter in (("C", sub, True), ("D", chunks, False)):
            bm_r = bm25_ranked(text, cand_chunks)
            if use_filter and must:
                vec_r = qdrant_ranked(client, vec, CAND_K, must) or qdrant_ranked(client, vec, CAND_K)
            else:
                vec_r = qdrant_ranked(client, vec, CAND_K)
            cand_ids = merge_candidates(bm_r, vec_r)[:CAND_K]
            docs = [by_id[c]["text"] for c in cand_ids if c in by_id]
            cand_ids = [c for c in cand_ids if c in by_id]
            try:
                res, lat = rerank(text, docs, top_n=TOP_K)
            except Exception as e:
                print(f"  [{key}] rerank 失败，跳过: {e}")
                res, lat = [], 0.0
            lat_rr.append(lat)
            n_docs.append(len(docs))
            top_ids = [cand_ids[r["index"]] for r in res if 0 <= r.get("index", -1) < len(cand_ids)]
            if args.smoke and key == "C":
                print(f"\n[smoke] {qid} | 候选 {len(docs)} 条 | rerank {lat:.0f}ms")
                for r in res[:5]:
                    cid = cand_ids[r["index"]]
                    mark = "✅" if cid in cids else "  "
                    print(f"  {mark} idx={r['index']} score={r['relevance_score']:.4f} {cid} | {by_id[cid]['text'][:40]}")
                print(f"  标注答案块: {sorted(cids)}")
                print(f"  候选是否包含答案块: {bool(cids & set(cand_ids))}")
            stats[key] += 1 if (set(top_ids) & cids) else 0
            # 记录 rerank 救回情况
            if key == "C":
                prev_hit = bool(set(b_ids) & cids)
                now_hit = bool(set(top_ids) & cids)
                if now_hit and not prev_hit:
                    saved_cases.append(dict(qid=qid, q=text[:30],
                                            in_candidates=bool(cids & set(cand_ids))))
        stats["A"] += 1 if (set(a_ids) & cids) else 0
        stats["B"] += 1 if (set(b_ids) & cids) else 0
        n += 1
        if args.smoke:
            return

    print(f"\n=== rerank 对比（分母 {n}） ===")
    labels = {"A": "向量不过滤 top5（线上基线）", "B": "向量过滤 top5（W5D2 最优）",
              "C": "过滤 + BM25/向量 并集top20 → rerank", "D": "不过滤 + 并集top20 → rerank"}
    for k in "ABCD":
        print(f"{labels[k]:42s} {stats[k]}/{n} = {stats[k] / max(n, 1) * 100:5.1f}%")

    print(f"\n=== 延迟与成本 ===")
    if lat_rr:
        print(f"rerank 延迟: P50={pct(lat_rr, 50):.0f}ms  P95={pct(lat_rr, 95):.0f}ms  平均={statistics.mean(lat_rr):.0f}ms")
        print(f"平均候选文档数: {statistics.mean(n_docs):.1f} 条")
        print("每千次成本：按硅基流动 bge-reranker-v2-m3 计费（¥0.7/M tokens），实际以账单为准；")
        print("            本项目单次约 20 文档 × 100 token ≈ 2k tokens → 千次约 2M tokens ≈ ¥1.4（数量级估算）")
    print(f"\n=== rerank 救回（C 配置 vs B 配置） ===")
    if saved_cases:
        for s in saved_cases:
            print(f"  {s['qid']} 候选内含答案={s['in_candidates']} | {s['q']}")
        nocand = sum(1 for s in saved_cases if not s["in_candidates"])
        print(f"  共救回 {len(saved_cases)} 题；其中候选里本来有答案的 {len(saved_cases) - nocand} 题")
    else:
        print("  无（rerank 没有改变命中结果）")

    out = BASE / "rerank_compare.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# W5D3 rerank 对比\n\n")
        f.write(f"- rerank 模型：`{RERANK_MODEL}`\n- 分母：{n}\n\n| 配置 | recall@5 |\n|---|---|\n")
        for k in "ABCD":
            f.write(f"| {labels[k]} | {stats[k]}/{n} = {stats[k] / max(n, 1) * 100:.1f}% |\n")
        if lat_rr:
            f.write(f"\n- rerank 延迟 P50={pct(lat_rr, 50):.0f}ms / P95={pct(lat_rr, 95):.0f}ms\n")
            f.write(f"- 平均候选 {statistics.mean(n_docs):.1f} 条\n")
        f.write(f"\n## rerank 救回的题\n")
        for s in saved_cases:
            f.write(f"- {s['qid']}（候选内含答案={s['in_candidates']}）：{s['q']}\n")
    print(f"\n[输出] {out}")


if __name__ == "__main__":
    main()
