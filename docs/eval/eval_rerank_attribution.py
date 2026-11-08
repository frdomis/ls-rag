#!/usr/bin/env python
# -*- coding: utf-8 -*-
# docs/eval/eval_rerank_attribution.py —— W5D4 rerank 对比 + 归因
"""
用法：
    python docs/eval/eval_rerank_attribution.py --smoke   # 1 题，验证 API
    python docs/eval/eval_rerank_attribution.py           # 全量 33 题
    python docs/eval/eval_rerank_attribution.py --no-coverage   # 跳过要点覆盖率（省 LLM）

对比 5 种配置（同一分母 33）：
    A 向量不过滤 top5                      ← 线上基线
    B 向量过滤 top5                        ← W5D2 最优 24/33
    C 过滤 + BM25/向量 并集top20 → RRF top5  ← 只有多路，不 rerank
    D 过滤 + 并集top20 → rerank → top5       ← 多路 + rerank
    E 不过滤 + 并集top20 → rerank → top5     ← 不用过滤的 rerank

【归因框架】每道漏检题自动分成 4 类：
    ✅ 命中
    ❌ 候选缺失（candidate_gap）   答案块根本不在候选池 → rerank 救不了，要改召回/补语料
    ⚠️ 排序失误（ranking_error）   答案块在候选池但没进 top5 → rerank/融合该救
    ❓ 过滤误伤（filter_harm）     不过滤命中、过滤后不命中
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

TOP_K = 5
CAND_K = 20
BASE = Path(__file__).resolve().parent
CHUNKS_PATH = Path(os.environ.get("CHUNKS_FILE", BASE.parent.parent / "docs" / "chunks" / "chunks_fixed_v2.jsonl"))
QUESTIONS_PATH = Path(os.environ.get("EVAL_FILE", BASE / "questions.all2.jsonl"))
ANNOTATED_PATH = Path(os.environ.get("ANNOTATED_FILE", BASE / "questions_annotated.jsonl"))
OUT_MD = BASE / "rerank_compare.md"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION", "chunks")
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_URL = "https://api.siliconflow.cn/v1/chat/completions"
GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek-ai/DeepSeek-V3")
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

YEAR_RE = re.compile(r"(20\d{2})")
DT_RULES = [
    (re.compile(r"分数线|投档线|控制线|控分线"), "score"),
    (re.compile(r"章程|简章"), "charter"),
    (re.compile(r"报名|考试|志愿|录取|免试|退役|建档|专项|政策|通知|时间|几科|学制|学费"), "policy"),
]
STOP = set("的了和是在我你他她它啊吗呢吧不有这那什么怎么可以能也就又还很太才已将对关于等之与及或并而各其该本此")


# ---------------- 基础 ----------------
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def is_refuse(q):
    src = str(q.get("expected_source") or "")
    return ("超出" in src) or ("拒答" in src)


def norm(s):
    return "".join(str(s or "").lower().split())


def grams(text, n=4):
    t = norm(text)
    if len(t) <= n:
        return [t]
    return [t[i:i + n] for i in range(0, len(t) - n + 1, 2)]


def infer_filters(question, corpus_years):
    must = []
    ys = YEAR_RE.findall(question or "")
    if ys and ys[0] in corpus_years:
        must.append({"key": "year", "value": ys[0]})
    for pat, dt in DT_RULES:
        if pat.search(question or ""):
            must.append({"key": "doc_type", "value": dt})
            break
    return must


def qdrant_filter(must):
    if not must:
        return None
    return models.Filter(must=[models.FieldCondition(key=m["key"], match=models.MatchValue(value=m["value"]))
                              for m in must])


def pct(vals, p):
    if not vals:
        return 0.0
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


# ---------------- API ----------------
def embed_one(text):
    r = requests.post(EMBED_URL,
                      headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json={"model": EMBED_MODEL, "input": [text]}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def qdrant_ranked(client, vec, k, must=None):
    kw = dict(collection_name=COLLECTION, query=vec, limit=k, with_payload=["chunk_id"])
    f = qdrant_filter(must)
    if f is not None:
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


def rerank(query, docs, top_n=TOP_K, timeout=25):
    t0 = time.time()
    r = requests.post(RERANK_URL,
                      headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json={"model": RERANK_MODEL, "query": query, "documents": docs,
                            "top_n": top_n, "return_documents": False},
                      timeout=timeout)
    lat = (time.time() - t0) * 1000
    if r.status_code >= 400:
        print(f"[rerank错误] {r.status_code} {r.text[:200]}")
        r.raise_for_status()
    return r.json().get("results", []), lat


def llm_answer(question, docs):
    ctx = "\n\n".join(docs[:TOP_K])
    prompt = (f"你是专升本政策问答助手。严格根据材料回答；材料无相关内容时回答「未找到相关信息」。\n\n"
              f"材料：\n{ctx}\n\n问题：{question}\n\n回答：")
    try:
        r = requests.post(LLM_URL,
                          headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                          json={"model": GEN_MODEL, "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0}, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[生成失败: {e}]"


def judge_kp(question, kps, answer):
    """返回覆盖的要点数（本地 4-gram 匹配，零成本，和 metrics 口径一致）"""
    a = norm(answer)
    return sum(1 for kp in kps if any(g in a for g in grams(kp)))


# ---------------- 融合 ----------------
def rrf_merge(ranked_lists, k=CAND_K):
    score = {}
    for lst in ranked_lists:
        for rank, (cid, _) in enumerate(lst[:k], 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (60 + rank)
    return [cid for cid, _ in sorted(score.items(), key=lambda kv: -kv[1])]


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-coverage", action="store_true", help="跳过要点覆盖率（省 LLM 调用）")
    args = ap.parse_args()
    if not API_KEY:
        sys.exit("[错误] 缺少 SILICONFLOW_API_KEY")

    import jieba
    from rank_bm25 import BM25Okapi

    chunks = [c for c in load_jsonl(CHUNKS_PATH) if c.get("text")]
    by_id = {str(c["id"]): c for c in chunks}
    corpus_years = {str(c.get("year", "")) for c in chunks}
    corpus_norm = norm("".join(c["text"] for c in chunks))

    qs = load_jsonl(QUESTIONS_PATH)
    ann = {a["id"]: a for a in load_jsonl(ANNOTATED_PATH)} if ANNOTATED_PATH.exists() else {}
    for q in qs:
        a = ann.get(str(q.get("id")), {})
        q["_cls"] = {"chunk_ids": {str(c) for c in (a.get("chunk_ids") or [])},
                     "refuse": is_refuse(q),
                     "unanswerable": (not is_refuse(q)) and not (a.get("chunk_ids") or [])}
    ans = [q for q in qs if not q["_cls"]["refuse"] and not q["_cls"]["unanswerable"]]
    if args.smoke:
        ans = ans[:1]
    elif args.limit:
        ans = ans[:args.limit]
    print(f"[数据] chunks={len(chunks)} | 可答={len(ans)} | rerank={RERANK_MODEL}")

    def bm25_top(question, pool):
        if not pool:
            return []
        bm = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in pool])
        sc = bm.get_scores(list(jieba.cut_for_search(question)))
        idx = sorted(range(len(sc)), key=lambda j: -sc[j])[:CAND_K]
        return [(str(pool[j]["id"]), float(sc[j])) for j in idx]

    client = QdrantClient(url=QDRANT_URL)
    conf = {k: {"hit": 0, "cov": 0, "kp": 0} for k in "ABCDE"}
    lat_rr, n_cand, n = [], [], 0
    per_q = defaultdict(dict)
    cand_ids_map = {}

    for q in ans:
        qid, text, cids = q.get("id"), q["question"], q["_cls"]["chunk_ids"]
        must = infer_filters(text, corpus_years)
        sub = [c for c in chunks if all(str(c.get(m["key"], "")) == m["value"] for m in must)] if must else chunks
        if must and not sub:
            sub = chunks
        vec = embed_one(text)

        # A 向量不过滤
        a_ranked = qdrant_ranked(client, vec, TOP_K)
        a_ids = [c for c, _ in a_ranked]
        # B 向量过滤（空则回退）
        b_ranked = qdrant_ranked(client, vec, TOP_K, must) if must else a_ranked
        if must and not b_ranked:
            b_ranked = a_ranked
        b_ids = [c for c, _ in b_ranked]

        # C/D/E 候选池
        pools = {
            "C": (sub, must),
            "D": (sub, must),
            "E": (chunks, None),
        }
        got = {"A": a_ids, "B": b_ids}
        for key, (pool, m) in pools.items():
            vr = qdrant_ranked(client, vec, CAND_K, m) if m else qdrant_ranked(client, vec, CAND_K)
            br = bm25_top(text, pool)
            cand = rrf_merge([br, vr])[:CAND_K]
            got[f"{key}_cand"] = cand
            if key == "C":
                got["C"] = cand[:TOP_K]          # 纯多路 RRF，不 rerank
            else:
                docs = [by_id[c]["text"] for c in cand if c in by_id]
                cids_ok = [c for c in cand if c in by_id]
                try:
                    res, lat = rerank(text, docs, top_n=TOP_K)
                    lat_rr.append(lat); n_cand.append(len(docs))
                    got[key] = [cids_ok[r["index"]] for r in res if 0 <= r.get("index", -1) < len(cids_ok)]
                except Exception as e:
                    print(f"  [{key}] rerank 失败: {e}")
                    got[key] = cand[:TOP_K]
            cand_ids_map[qid] = cand

        # 统计 + 覆盖率
        for k in "ABCDE":
            ids = got.get(k, [])
            h = bool(set(ids) & cids)
            per_q[qid][k] = h
            if h:
                conf[k]["hit"] += 1
            if not args.no_coverage:
                kps = [kp for kp in (q.get("expected_keypoints") or []) if any(g in corpus_norm for g in grams(kp))]
                if kps:
                    docs = [by_id[c]["text"] for c in ids if c in by_id]
                    ans_text = llm_answer(text, docs)
                    cov = judge_kp(text, kps, ans_text)
                    conf[k]["cov"] += cov
                    conf[k]["kp"] += len(kps)
        n += 1
        if args.smoke:
            print(f"\n[smoke] {qid} | 候选池 {len(cand_ids_map[qid])} 条 | 答案块在候选池? {bool(cids & set(cand_ids_map[qid]))}")
            for k in "ABCDE":
                print(f"  {k}: {'✅' if per_q[qid][k] else '❌'} {got.get(k, [])[:3]}")
            print(f"  标注: {sorted(cids)}")
            return

    labels = {
        "A": "向量不过滤 top5（线上基线）",
        "B": "向量过滤 top5（W5D2 最优）",
        "C": "过滤 + 多路 RRF top5（不 rerank）",
        "D": "过滤 + 多路top20 → rerank",
        "E": "不过滤 + 多路top20 → rerank",
    }
    print(f"\n=== W5D4 对比（分母 {n}） ===")
    print(f"{'配置':38s} {'recall@5':>16s}   要点覆盖率")
    for k in "ABCDE":
        c = conf[k]
        cov_s = f"{c['cov']}/{c['kp']} = {c['cov'] / max(c['kp'], 1) * 100:.1f}%" if c["kp"] else "（未跑）"
        print(f"{labels[k]:38s} {c['hit']:>2d}/{n} = {c['hit'] / n * 100:5.1f}%   {cov_s}")

    # ---- 归因：以 D 为主 ----
    attr = Counter()
    detail = defaultdict(list)
    for qid, res in per_q.items():
        cand = set(cand_ids_map.get(qid, []))
        q = next(x for x in ans if x.get("id") == qid)
        cids = q["_cls"]["chunk_ids"]
        if res["D"]:
            attr["命中"] += 1
            continue
        if cids & cand:
            attr["排序失误 ranking_error"] += 1
            detail["排序失误"].append(qid)
        elif res["B"] and not res["D"]:
            attr["rerank 反伤"] += 1
            detail["rerank 反伤"].append(qid)
        else:
            attr["候选缺失 candidate_gap"] += 1
            detail["候选缺失"].append(qid)
        if res["A"] and not res["B"]:
            attr["过滤误伤 filter_harm"] += 1
            detail["过滤误伤"].append(qid)

    print(f"\n=== D 配置漏检归因（{n - conf['D']['hit']} 道） ===")
    for k, v in attr.most_common():
        print(f"  {k}: {v} {detail.get(k.split()[0], detail.get(k, [])) if k != '命中' else ''}")

    if lat_rr:
        print(f"\n=== 延迟与成本 ===")
        print(f"rerank 延迟 P50={pct(lat_rr, 50):.0f}ms  P95={pct(lat_rr, 95):.0f}ms  平均={statistics.mean(lat_rr):.0f}ms")
        print(f"平均候选 {statistics.mean(n_cand):.1f} 条 → 单次约 {statistics.mean(n_cand) * 150 / 1e6:.4f}M tokens")
        print(f"千次成本 ≈ ¥{statistics.mean(n_cand) * 150 / 1e6 * 1000 * 0.7:.2f}（按 ¥0.7/M tokens 估）")

    # ---- 写报告 ----
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# W5D4 rerank 对比实验\n\n")
        f.write(f"- rerank 模型：`{RERANK_MODEL}`\n- 分母：{n}\n- 日期：{time.strftime('%Y-%m-%d')}\n\n")
        f.write("| 配置 | recall@5 | 要点覆盖率 |\n|---|---|---|\n")
        for k in "ABCDE":
            c = conf[k]
            cov_s = f"{c['cov']}/{c['kp']} = {c['cov'] / max(c['kp'], 1) * 100:.1f}%" if c["kp"] else "—"
            f.write(f"| {labels[k]} | {c['hit']}/{n} = {c['hit'] / n * 100:.1f}% | {cov_s} |\n")
        if lat_rr:
            f.write(f"\n- P50 {pct(lat_rr, 50):.0f}ms / P95 {pct(lat_rr, 95):.0f}ms；平均候选 {statistics.mean(n_cand):.1f} 条\n")
            f.write(f"- 千次成本 ≈ ¥{statistics.mean(n_cand) * 150 / 1e6 * 1000 * 0.7:.2f}\n")
        f.write("\n## D 配置漏检归因\n\n")
        for k, v in attr.most_common():
            if k == "命中":
                continue
            f.write(f"- **{k}**: {v} 道 {detail.get(k, [])}\n")
    print(f"\n[输出] {OUT_MD}")


if __name__ == "__main__":
    main()
