#!/usr/bin/env python
# -*- coding: utf-8 -*-
# docs/eval/eval_metadata_filter.py —— W5D2 元数据过滤对比
"""
用法：
    python docs/eval/eval_metadata_filter.py --diag        # 只跑 diag：三配置命中题号集合，验证 BM25 ⊆ 向量
    python docs/eval/eval_metadata_filter.py               # 跑过滤对比（不过滤 vs 过滤），写 docs/eval/filter_compare.md
    python docs/eval/eval_metadata_filter.py --limit 10    # 调试

口径：块级 recall@5，分母 33（可答题），拒答/语料缺失单列；一律 N/M = P%。

过滤规则（按问题里的线索推断 must 条件，命不中就不过滤）：
  - 年份：问题中抽出 20XX 且语料里有该年 → year=20XX
  - 分数线/投档/控制线 → doc_type=score
  - 章程/简章 → doc_type=charter
  - 报名/考试/志愿/录取/免试/退役/建档立卡/政策/通知 → doc_type in (policy, notice)
  兜底：过滤后 top-5 不足（BM25 候选 <5 或 Qdrant hits <5）→ 回退全量，并记录 fallback 次数
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

TOP_K = 5
CAND_K = 30
BASE = Path(__file__).resolve().parent
CHUNKS_PATH = Path(os.environ.get("CHUNKS_FILE", BASE.parent.parent / "docs" / "chunks" / "chunks_fixed_v2.jsonl"))
QUESTIONS_PATH = Path(os.environ.get("EVAL_FILE", BASE / "questions.all2.jsonl"))
ANNOTATED_PATH = Path(os.environ.get("ANNOTATED_FILE", BASE / "questions_annotated.jsonl"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION", "chunks")
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

YEAR_RE = re.compile(r"(20\d{2})")
DT_RULES = [
    (re.compile(r"分数线|投档线|控制线|控分线"), "score"),
    (re.compile(r"章程|简章"), "charter"),
    (re.compile(r"报名|考试|志愿|录取|免试|退役|建档|专项|政策|通知|时间|几科|学制|学费"), "policy"),
]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def norm(s):
    return "".join(str(s or "").lower().split())


def is_refuse(q):
    src = str(q.get("expected_source") or "")
    return ("超出" in src) or ("拒答" in src)


def infer_filters(question, corpus_years):
    must = []
    years = YEAR_RE.findall(question or "")
    if years:
        y = years[0]
        if y in corpus_years:
            must.append({"key": "year", "value": y})
    for pat, dt in DT_RULES:
        if pat.search(question or ""):
            must.append({"key": "doc_type", "value": dt})
            break
    return must


def match_chunk(chunk, must):
    for m in must:
        if m["key"] == "year" and str(chunk.get("year", "")) != m["value"]:
            return False
        if m["key"] == "doc_type" and str(chunk.get("doc_type", "")) != m["value"]:
            return False
    return True


def embed_one(text, retries=4):
    """带指数退避重试的 embedding（网络抖动时自动重试，避免整轮白跑）"""
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                EMBED_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": EMBED_MODEL, "input": [text]},
                timeout=(10, 90),          # (连接超时, 读取超时)
            )
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last = e
            wait = 2 ** attempt            # 1, 2, 4, 8 秒
            if attempt < retries - 1:
                print(f"  [重试 {attempt + 1}/{retries - 1}] {type(e).__name__}，{wait}s 后重试…", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"embedding 连续失败 {retries} 次：{last}")


def qdrant_hits(client, vec, k, must=None):
    try:
        from qdrant_client import models
    except ImportError:
        sys.exit("[依赖缺失] pip install qdrant-client")
    kwargs = dict(collection_name=COLLECTION, query=vec, limit=k, with_payload=["chunk_id"])
    if must:
        conds = []
        for m in must:
            if m["key"] == "year":
                conds.append(models.FieldCondition(key="year", match=models.MatchValue(value=m["value"])))
            else:
                conds.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=m["value"])))
        kwargs["query_filter"] = models.Filter(must=conds)
    try:
        return client.query_points(**kwargs).points
    except AttributeError:
        kwargs.pop("query", None)
        kwargs["query_vector"] = vec
        kwargs["query_filter"] = kwargs.get("query_filter")
        return client.search(**kwargs)


def fmt_hit(h, key="chunk_id"):
    p = getattr(h, "payload", None) or {}
    return str(p.get("chunk_id") or getattr(h, "id", ""))


def diag(qs, client, by_id, chunks):
    """三配置命中题号集合 + 子集验证（用真实名次做 RRF，不能用 set 排序）"""
    print("=== diag：三配置命中题号 ===")
    import jieba
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in chunks])
    bm_hits, vec_hits, rrf_hits = [], [], []
    for q in qs:
        if q["_cls"]["refuse"] or q["_cls"]["unanswerable"]:
            continue
        cids = q["_cls"]["chunk_ids"]
        # BM25 名次列表（top CAND_K）
        scores = bm.get_scores(list(jieba.cut_for_search(q["question"])))
        idx = sorted(range(len(scores)), key=lambda j: -scores[j])[:CAND_K]
        bm_ranked = [str(chunks[j]["id"]) for j in idx]
        # 向量名次列表（top CAND_K）
        vec = embed_one(q["question"])
        vec_ranked = [fmt_hit(h) for h in qdrant_hits(client, vec, CAND_K)]
        # RRF：按真实名次贡献 1/(k+rank)
        rrf_score = {}
        for rank, cid in enumerate(bm_ranked, 1):
            rrf_score[cid] = rrf_score.get(cid, 0) + 1.0 / (60 + rank)
        for rank, cid in enumerate(vec_ranked, 1):
            rrf_score[cid] = rrf_score.get(cid, 0) + 1.0 / (60 + rank)
        rrf_ids = {cid for cid, _ in sorted(rrf_score.items(), key=lambda kv: -kv[1])[:TOP_K]}
        hit = lambda ids: bool(set(ids[:TOP_K]) & cids)
        if hit(bm_ranked):
            bm_hits.append(q.get("id"))
        if hit(vec_ranked):
            vec_hits.append(q.get("id"))
        if rrf_ids & cids:
            rrf_hits.append(q.get("id"))
    print(f"BM25 命中: {bm_hits}")
    print(f"向量 命中: {vec_hits}")
    print(f"RRF  命中: {rrf_hits}")
    bm_only = sorted(set(bm_hits) - set(vec_hits), key=str)
    print(f"BM25 独有（向量没命中的）: {bm_only}  → BM25 ⊆ 向量? {not bm_only}")
    print(f"RRF 额外救回（向量没命中的）: {sorted(set(rrf_hits) - set(vec_hits), key=str)}")
    print("[结论] RRF 救回的数量 = 融合对单路（向量）的净增益")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not API_KEY:
        sys.exit("[错误] 缺少 SILICONFLOW_API_KEY")

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
    if args.limit:
        qs = qs[: args.limit]
    ans = [q for q in qs if not q["_cls"]["refuse"] and not q["_cls"]["unanswerable"]]
    print(f"[数据] chunks={len(chunks)} | 题={len(qs)} | 可答={len(ans)} | 语料年份={sorted(corpus_years)}")

    client = QdrantClient(url=QDRANT_URL)
    if args.diag:
        diag(qs, client, by_id, chunks)
        return

    import jieba
    from rank_bm25 import BM25Okapi

    bm = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in chunks])

    rows = []
    stats = {"bm25_nf": 0, "bm25_f": 0, "vec_nf": 0, "vec_f": 0, "fb": 0, "small": 0}
    for q in ans:
        cids = q["_cls"]["chunk_ids"]
        must = infer_filters(q["question"], corpus_years)
        # BM25 不过滤
        scores = bm.get_scores(list(jieba.cut_for_search(q["question"])))
        idx = sorted(range(len(scores)), key=lambda j: -scores[j])[:TOP_K]
        bm_ids = {str(chunks[j]["id"]) for j in idx}
        # BM25 过滤：仅当过滤结果为空才回退全量
        sub = [c for c in chunks if match_chunk(c, must)] if must else chunks
        fb = 1 if (must and len(sub) == 0) else 0
        if must and 0 < len(sub) < TOP_K:
            stats["small"] += 1
        stats["fb"] += fb
        cand = sub if sub else chunks
        sub_bm = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in cand])
        s2 = sub_bm.get_scores(list(jieba.cut_for_search(q["question"])))
        idx2 = sorted(range(len(s2)), key=lambda j: -s2[j])[:TOP_K]
        bm_f_ids = {str(cand[j]["id"]) for j in idx2}
        # 向量 不过滤 / 过滤：must 为空 = 不过滤（直接用不过滤结果）；过滤结果为空才回退
        vec = embed_one(q["question"])
        raw_nf = qdrant_hits(client, vec, TOP_K)
        v_nf = {fmt_hit(h) for h in raw_nf}
        if must:
            raw_f = qdrant_hits(client, vec, TOP_K, must)
            if len(raw_f) == 0:
                raw_f = raw_nf
                stats["fb"] += 1
        else:
            raw_f = raw_nf
        v_f = {fmt_hit(h) for h in raw_f}

        hit = lambda ids: bool(ids & cids)
        r = dict(qid=q.get("id"), q=q["question"], must=must,
                 bm_nf=hit(bm_ids), bm_f=hit(bm_f_ids), v_nf=hit(v_nf), v_f=hit(v_f), fb=fb)
        rows.append(r)
        stats["bm25_nf"] += r["bm_nf"]; stats["bm25_f"] += r["bm_f"]
        stats["vec_nf"] += r["v_nf"]; stats["vec_f"] += r["v_f"]

    n = len(rows)
    print(f"\n=== 过滤对比（分母 {n}） ===")
    print(f"BM25  不过滤: {stats['bm25_nf']}/{n} = {stats['bm25_nf']/n*100:.1f}%   过滤: {stats['bm25_f']}/{n} = {stats['bm25_f']/n*100:.1f}%")
    print(f"向量  不过滤: {stats['vec_nf']}/{n} = {stats['vec_nf']/n*100:.1f}%   过滤: {stats['vec_f']}/{n} = {stats['vec_f']/n*100:.1f}%")
    print(f"[兜底] 触发 {stats['fb']} 次（过滤后结果为空 → 回退全量）；过滤后候选<5 的题: {stats['small']} 道（不兜底，如实计算）")
    print("\n[变化题] 过滤改变命中结果的题：")
    for r in rows:
        if r["bm_nf"] != r["bm_f"] or r["v_nf"] != r["v_f"]:
            print(f"  {r['qid']} 过滤={r['must']} bm:{r['bm_nf']}→{r['bm_f']} vec:{r['v_nf']}→{r['v_f']} fb={r['fb']} | {r['q'][:32]}")

    out = BASE / "filter_compare.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# W5D2 元数据过滤对比\n\n")
        f.write(f"- BM25：不过滤 {stats['bm25_nf']}/{n}，过滤 {stats['bm25_f']}/{n}\n")
        f.write(f"- 向量：不过滤 {stats['vec_nf']}/{n}，过滤 {stats['vec_f']}/{n}\n")
        f.write(f"- 兜底触发 {stats['fb']} 次\n\n| 题号 | 问题 | must | BM25 | 向量 |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['qid']} | {r['q'][:28]} | {r['must'] or '-'} | {r['bm_nf']}→{r['bm_f']} | {r['v_nf']}→{r['v_f']} |\n")
    print(f"\n[输出] {out}")


if __name__ == "__main__":
    main()
