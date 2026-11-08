#!/usr/bin/env python
# -*- coding: utf-8 -*-
# docs/eval/eval_query_rewrite.py —— W5D5 query 改写对比
"""
用法：
    python docs/eval/eval_query_rewrite.py --smoke          # 1 题，看改写效果
    python docs/eval/eval_query_rewrite.py --no-coverage    # 只算 recall（快）
    python docs/eval/eval_query_rewrite.py                  # 全量（含要点覆盖率）

策略（已选）：单次改写 + 单路检索；只对 oral=true 的题改写
对比 4 组（同一分母 33）：
    A 原始 query（基线，= W5D4 的配置 B：向量过滤 top5）
    B 改写后 query（只口语题替换，非口语题保持不变）
    C 改写后 + 多路 rerank（= W5D4 的 D 配置，看改写能否叠加）
    D 原始 query + 多路 rerank（= W5D4 的 D 配置原版，作对照）

重点：改写对口语题的效果，尤其 W5D4 归因出的 4 道候选缺失题（Q15/Q17/Q18/Q33）
成本护栏：每题最多 1 次改写 LLM 调用（单次改写 = 1 次，非"每层都改"）
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
OUT_MD = BASE / "query_rewrite_compare.md"
CACHE_PATH = BASE / "rewrite_cache.jsonl"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION", "chunks")
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_URL = "https://api.siliconflow.cn/v1/chat/completions"
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "deepseek-ai/DeepSeek-V3")
GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek-ai/DeepSeek-V3")
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

YEAR_RE = re.compile(r"(20\d{2})")
DT_RULES = [
    (re.compile(r"分数线|投档线|控制线|控分线|多少分"), "score"),
    (re.compile(r"章程|简章"), "charter"),
    (re.compile(r"报名|考试|志愿|录取|免试|退役|建档|专项|政策|通知|时间|几科|学制|学费"), "policy"),
]

# W5D4 归因出的候选缺失题（本次重点验证）
FOCUS_QIDS = {"Q15", "Q17", "Q18", "Q33"}

REWRITE_PROMPT = """你是检索查询优化助手。把用户的口语化提问改写成适合检索的书面问句。

要求：
1. 保留原意和关键限定（年份、专业、身份、分数等），不得改变问题意图
2. 补全专业术语（如"滑档"→"未被录取/征集志愿"，"一辈子就一次"→"报考次数限制"）
3. 去掉无信息量的口语成分（啊、呢、吧、我、是不是、晚不晚）
4. 只输出一行改写结果，不要任何解释、不要引号、不要编号

用户提问：{q}

改写后："""


# ---------------- 基础 ----------------
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def is_refuse(q):
    src = str(q.get("expected_source") or "")
    return ("超出" in src) or ("拒答" in src)


def norm(s):
    return "".join(str(s or "").lower().split())


def grams(t, n=4):
    t = norm(t)
    return [t] if len(t) <= n else [t[i:i + n] for i in range(0, len(t) - n + 1, 2)]


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


def pct(v, p):
    if not v:
        return 0.0
    v = sorted(v)
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
    return [(str((getattr(h, "payload", None) or {}).get("chunk_id") or h.id),
             float(getattr(h, "score", 0.0))) for h in pts]


def rewrite_query(q, cache):
    if q in cache:
        return cache[q], 0.0
    t0 = time.time()
    r = requests.post(LLM_URL,
                      headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json={"model": REWRITE_MODEL,
                            "messages": [{"role": "user", "content": REWRITE_PROMPT.format(q=q)}],
                            "temperature": 0, "max_tokens": 120}, timeout=60)
    lat = (time.time() - t0) * 1000
    if r.status_code >= 400:
        print(f"[改写失败] {r.status_code} {r.text[:150]}")
        return q, lat
    out = r.json()["choices"][0]["message"]["content"].strip()
    out = out.split("\n")[0].strip().strip("\"'“”‘’").strip()
    for pre in ("改写后：", "改写后:", "改写结果：", "改写结果:", "改写："):
        if out.startswith(pre):
            out = out[len(pre):].strip()
    if not out or len(out) > 200:
        return q, lat
    cache[q] = out
    return out, lat


def rerank(query, docs, top_n=TOP_K, timeout=25):
    t0 = time.time()
    r = requests.post(RERANK_URL,
                      headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                      json={"model": RERANK_MODEL, "query": query, "documents": docs,
                            "top_n": top_n, "return_documents": False}, timeout=timeout)
    lat = (time.time() - t0) * 1000
    if r.status_code >= 400:
        print(f"[rerank错误] {r.status_code} {r.text[:150]}")
        return [], lat
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
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[生成失败: {e}]"


def judge_kp(answer, kps):
    a = norm(answer)
    return sum(1 for kp in kps if any(g in a for g in grams(kp)))


def rrf_merge(lists, k=CAND_K):
    sc = {}
    for lst in lists:
        for rank, (cid, _) in enumerate(lst[:k], 1):
            sc[cid] = sc.get(cid, 0.0) + 1.0 / (60 + rank)
    return [cid for cid, _ in sorted(sc.items(), key=lambda kv: -kv[1])]


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-coverage", action="store_true")
    ap.add_argument("--oral-only", action="store_true", help="只跑口语题（调试用）")
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
    ann = {a["id"]: a for a in load_jsonl(ANNOTATED_PATH)}
    for q in qs:
        a = ann.get(str(q.get("id")), {})
        q["_cls"] = {"chunk_ids": {str(c) for c in (a.get("chunk_ids") or [])},
                     "refuse": is_refuse(q),
                     "unanswerable": (not is_refuse(q)) and not (a.get("chunk_ids") or []),
                     "oral": str(q.get("oral", "")).lower() == "true"}
    ans = [q for q in qs if not q["_cls"]["refuse"] and not q["_cls"]["unanswerable"]]
    if args.smoke:
        ans = [q for q in ans if q["_cls"]["oral"]][:3]
    elif args.oral_only:
        ans = [q for q in ans if q["_cls"]["oral"]]
    if args.limit:
        ans = ans[:args.limit]
    print(f"[数据] chunks={len(chunks)} | 可答={len(ans)} | 其中口语题={sum(1 for q in ans if q['_cls']['oral'])}")

    cache = {r["q"]: r["rewritten"] for r in load_jsonl(CACHE_PATH)}
    bm_all = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in chunks])
    client = QdrantClient(url=QDRANT_URL)

    def bm_top(text, pool):
        if not pool:
            return []
        bm = BM25Okapi([list(jieba.cut_for_search(c["text"])) for c in pool])
        sc = bm.get_scores(list(jieba.cut_for_search(text)))
        idx = sorted(range(len(sc)), key=lambda j: -sc[j])[:CAND_K]
        return [(str(pool[j]["id"]), float(sc[j])) for j in idx]

    conf = {k: {"hit": 0, "cov": 0, "kp": 0} for k in "ABCD"}
    oral_conf = {k: {"hit": 0, "n": 0} for k in "ABCD"}
    rw_lat, rows = [], []
    n = 0

    for q in ans:
        qid, orig, cids = q.get("id"), q["question"], q["_cls"]["chunk_ids"]
        is_oral = q["_cls"]["oral"]
        if is_oral:
            rw, lat = rewrite_query(orig, cache)
            rw_lat.append(lat)
        else:
            rw = orig
        rows.append(dict(qid=qid, orig=orig, rw=rw, oral=is_oral))
        if args.smoke:
            print(f"\n{qid} [oral={is_oral}]\n  原句: {orig}\n  改写: {rw}")

        got = {}
        for key, qtext in (("A", orig), ("B", rw)):
            must = infer_filters(qtext, corpus_years)
            vec = embed_one(qtext)
            rk = qdrant_ranked(client, vec, TOP_K, must) if must else qdrant_ranked(client, vec, TOP_K)
            if must and not rk:
                rk = qdrant_ranked(client, vec, TOP_K)
            got[key] = [c for c, _ in rk]

        # C/D：多路 top20 候选 → rerank
        for key, qtext in (("C", rw), ("D", orig)):
            must = infer_filters(qtext, corpus_years)
            sub = [c for c in chunks if all(str(c.get(m["key"], "")) == m["value"] for m in must)] if must else chunks
            if must and not sub:
                sub = chunks
            vec = embed_one(qtext)
            vr = qdrant_ranked(client, vec, CAND_K, must) if must else qdrant_ranked(client, vec, CAND_K)
            if must and not vr:
                vr = qdrant_ranked(client, vec, CAND_K)
            cand = rrf_merge([bm_top(qtext, sub), vr])[:CAND_K]
            ok_ids = [c for c in cand if c in by_id]
            try:
                res, _ = rerank(qtext, [by_id[c]["text"] for c in ok_ids], top_n=TOP_K)
                got[key] = [ok_ids[r["index"]] for r in res if 0 <= r.get("index", -1) < len(ok_ids)]
            except Exception as e:
                print(f"  [{key}] rerank 失败: {e}")
                got[key] = cand[:TOP_K]

        for k in "ABCD":
            ids = got.get(k, [])
            h = bool(set(ids) & cids)
            conf[k]["hit"] += 1 if h else 0
            if is_oral:
                oral_conf[k]["hit"] += 1 if h else 0
                oral_conf[k]["n"] += 1
            if not args.no_coverage:
                kps = [kp for kp in (q.get("expected_keypoints") or []) if any(g in corpus_norm for g in grams(kp))]
                if kps:
                    cov = judge_kp(llm_answer(orig, [by_id[c]["text"] for c in ids if c in by_id]), kps)
                    conf[k]["cov"] += cov
                    conf[k]["kp"] += len(kps)
        n += 1

    if args.smoke:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            for k, v in cache.items():
                f.write(json.dumps({"q": k, "rewritten": v}, ensure_ascii=False) + "\n")
        print(f"\n[缓存] {len(cache)} 条 → {CACHE_PATH}")
        return

    labels = {"A": "A 原始 query（基线）", "B": "B 改写后 query",
              "C": "C 改写 + 多路 rerank", "D": "D 原始 + 多路 rerank"}
    print(f"\n=== W5D5 query 改写对比（分母 {n}） ===")
    print(f"{'配置':30s} {'recall@5':>16s}  要点覆盖率")
    for k in "ABCD":
        c = conf[k]
        cov = f"{c['cov']}/{c['kp']} = {c['cov'] / max(c['kp'], 1) * 100:.1f}%" if c["kp"] else "（未跑）"
        print(f"{labels[k]:30s} {c['hit']:>2d}/{n} = {c['hit'] / n * 100:5.1f}%  {cov}")

    no = oral_conf["A"]["n"]
    print(f"\n=== 口语题细分（{no} 道） ===")
    for k in "ABCD":
        print(f"{labels[k]:30s} {oral_conf[k]['hit']}/{no} = {oral_conf[k]['hit'] / max(no, 1) * 100:.1f}%")

    if rw_lat:
        rw_lat = [x for x in rw_lat if x > 0]
        if rw_lat:
            print(f"\n改写延迟 P50={pct(rw_lat, 50):.0f}ms P95={pct(rw_lat, 95):.0f}ms")
            print(f"改写调用次数: {len(rw_lat)} 次（每题 1 次，缓存命中 {sum(1 for r in rows if r['oral']) - len(rw_lat)} 次）")

    # 改写示例
    examples = [r for r in rows if r["oral"] and r["orig"] != r["rw"]][:3]
    print(f"\n=== 改写示例 ===")
    for e in examples:
        print(f"  {e['qid']} 原句: {e['orig']}\n       改写: {e['rw']}")

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        for k, v in cache.items():
            f.write(json.dumps({"q": k, "rewritten": v}, ensure_ascii=False) + "\n")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# W5D5 query 改写对比\n\n")
        f.write(f"- 改写模型：`{REWRITE_MODEL}`（每题 1 次调用）\n- 分母：{n}（口语题 {no} 道）\n")
        f.write(f"- 日期：{time.strftime('%Y-%m-%d')}\n\n| 配置 | recall@5 | 要点覆盖率 |\n|---|---|---|\n")
        for k in "ABCD":
            c = conf[k]
            cov = f"{c['cov']}/{c['kp']} = {c['cov'] / max(c['kp'], 1) * 100:.1f}%" if c["kp"] else "—"
            f.write(f"| {labels[k]} | {c['hit']}/{n} = {c['hit'] / n * 100:.1f}% | {cov} |\n")
        f.write(f"\n## 口语题细分（{no} 道）\n\n| 配置 | recall@5 |\n|---|---|\n")
        for k in "ABCD":
            f.write(f"| {labels[k]} | {oral_conf[k]['hit']}/{no} = {oral_conf[k]['hit'] / max(no, 1) * 100:.1f}% |\n")
        f.write("\n## 改写示例\n\n")
        for e in [r for r in rows if r["oral"]][:5]:
            f.write(f"- **{e['qid']}**\n  - 原句：{e['orig']}\n  - 改写：{e['rw']}\n")
    print(f"\n[输出] {OUT_MD}")


if __name__ == "__main__":
    main()
