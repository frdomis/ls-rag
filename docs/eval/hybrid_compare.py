#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
混合检索三路对比脚本（v3：默认走线上 Qdrant + BGE-M3）
用法：
    python hybrid_compare.py                        # 默认：Qdrant+BGE-M3 向量 vs 本地 BM25，前 10 题
    python hybrid_compare.py --limit 40 --preview 60
    python hybrid_compare.py --vector local         # 本地 MiniLM（仅验证 id 空间用，会下载 ~470MB 模型）
    python hybrid_compare.py --dump-ids 5           # 查看 chunk id / 演示一次向量检索后退出

为什么默认 Qdrant：
  - 线上 /chat 用的是 Qdrant + BGE-M3(1024 维)；本地方案（MiniLM 384 维）分数不可与线上比较；
  - 首次跑本地模型会从 HuggingFace 下载 ~470MB，国内网络会长时间无响应（不是卡死，是在等网络）。
    如需本地模型：$env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_TELEMETRY="1"

依赖：pip install rank-bm25 jieba numpy requests qdrant-client
配置：.env 里需要 SILICONFLOW_API_KEY（硅基流动 BGE-M3）
"""
import argparse
import json
import os
import re
import sys
import unicodedata

import numpy as np
import requests

TOP_K = 5
BM25_K = 10
VECTOR_K = 10
RRF_K = 60
DEFAULT_CHUNKS = "docs/chunks/chunks_fixed_v2.jsonl"
DEFAULT_EVAL = "docs/eval/questions.all2.jsonl"
DEFAULT_COLLECTION = "chunks"
DEFAULT_QDRANT = "http://localhost:6333"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_BATCH = 16


# ---------------- 基础 ----------------
def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cid_of(chunk):
    for key in ("id", "chunk_id", "chunkId"):
        v = chunk.get(key)
        if v not in (None, ""):
            return str(v)
    raise ValueError(f"chunk 缺少 id 字段，keys={list(chunk.keys())}")


def norm(text):
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text):
    import jieba
    return [w for w in jieba.cut(norm(text)) if w.strip()]


def embed_texts(texts, batch=EMBED_BATCH):
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        sys.exit("[错误] 缺少 SILICONFLOW_API_KEY（写进项目 .env）")
    out = []
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        r = requests.post(
            EMBED_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": part},
            timeout=180,
        )
        r.raise_for_status()
        vecs = [d["embedding"] for d in r.json()["data"]]
        for v in vecs:
            assert len(v) == EMBED_DIM, f"维度异常 {len(v)}，BGE-M3 应为 {EMBED_DIM}"
        out.extend(vecs)
    return out


# ---------------- 索引 / 检索 ----------------
def build_bm25(chunks):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([tokenize(c["text"]) for c in chunks])


def bm25_search(query, top_k, chunks, bm25):
    scores = bm25.get_scores(tokenize(query))
    idx = np.argsort(-scores)[:top_k]
    return [
        {"rank": i + 1, "chunk_id": cid_of(chunks[j]), "score": round(float(scores[j]), 4)}
        for i, j in enumerate(idx)
    ]


def make_qdrant_backend(collection, url):
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        sys.exit("[依赖缺失] pip install qdrant-client")
    client = QdrantClient(url=url)
    info = client.get_collection(collection)
    dim = info.config.params.vectors.size
    print(f"[向量] Qdrant 集合 {collection}: {info.points_count} 点 / {dim} 维 / BGE-M3")
    if dim != EMBED_DIM:
        sys.exit(f"[错误] 集合维度 {dim} ≠ {EMBED_DIM}，不是 BGE-M3 建的集合")

    def search(query, top_k):
        qv = embed_texts([query])[0]
        try:
            hits = client.search(collection_name=collection, query_vector=qv, limit=top_k)
        except AttributeError:  # 新版 qdrant-client
            hits = client.query_points(collection_name=collection, query=qv, limit=top_k).points
        out = []
        for i, h in enumerate(hits):
            payload = h.payload or {}
            cid = payload.get("chunk_id") or str(h.id)
            out.append({"rank": i + 1, "chunk_id": str(cid), "score": round(float(h.score), 4)})
        return out

    return search, info.points_count


def make_local_backend(chunks, model_name):
    print("[模型] 本地加载中：首次会从 HuggingFace 下载 ~470MB；")
    print("       长时间无响应=在等网络。国内先设置：$env:HF_ENDPOINT='https://hf-mirror.com'")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("[依赖缺失] pip install sentence-transformers（或改用 --vector qdrant）")
    model = SentenceTransformer(model_name)
    vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

    def search(query, top_k):
        qv = model.encode([query], normalize_embeddings=True)[0]
        sims = vecs @ qv
        idx = np.argsort(-sims)[:top_k]
        return [
            {"rank": i + 1, "chunk_id": cid_of(chunks[j]), "score": round(float(sims[j]), 4)}
            for i, j in enumerate(idx)
        ]

    return search, None


def reciprocal_rank_fusion(bm25_list, vector_list, top_k, k=RRF_K):
    scores = {}
    for lst in (bm25_list, vector_list):
        seen = set()
        for rank, item in enumerate(lst, start=1):
            cid = item["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{"rank": i + 1, "chunk_id": cid, "rrf_score": round(s, 4)} for i, (cid, s) in enumerate(ranked)]


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=os.environ.get("CHUNKS_FILE", DEFAULT_CHUNKS))
    ap.add_argument("--eval", dest="eval_path", default=os.environ.get("EVAL_FILE", DEFAULT_EVAL))
    ap.add_argument("--limit", type=int, default=10, help="0 = 全部题目")
    ap.add_argument("--preview", type=int, default=0, help="每题附带 chunk 文本前 n 字")
    ap.add_argument("--vector", choices=["qdrant", "local"], default="qdrant")
    ap.add_argument("--collection", default=os.environ.get("COLLECTION", DEFAULT_COLLECTION))
    ap.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", DEFAULT_QDRANT))
    ap.add_argument("--dump-ids", type=int, default=0)
    args = ap.parse_args()
    load_env()

    chunks = load_jsonl(args.chunks)
    ids = [cid_of(c) for c in chunks]
    id_set = set(ids)
    dup = len(ids) - len(id_set)
    print(f"[数据] chunk 文件: {args.chunks} | {len(chunks)} 块 | 重复 id {dup}", "✅" if dup == 0 else "❌")
    if dup:
        sys.exit("[中止] 有重复 id，先跑 fix_chunks.py")

    eval_path = args.eval_path
    if not os.path.exists(eval_path) and os.path.exists("docs/eval/questions.all2.jsonl"):
        eval_path = "docs/eval/questions.all2.jsonl"
    eval_set = load_jsonl(eval_path)
    if args.limit:
        eval_set = eval_set[: args.limit]
    print(f"[数据] 评估题 {len(eval_set)} 条: {eval_path}")

    bm25 = build_bm25(chunks)
    if args.vector == "qdrant":
        vec_search, points_count = make_qdrant_backend(args.collection, args.qdrant_url)
        if points_count != len(chunks):
            print(f"[警告] Qdrant {points_count} 点 ≠ 文件 {len(chunks)} 块 —— 可能集合没重建！")
    else:
        model_name = os.environ.get("ST_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vec_search, _ = make_local_backend(chunks, model_name)

    if args.dump_ids:
        print(f"\n[ID 样例] 前 {args.dump_ids} 个 chunk id：")
        for c in chunks[: args.dump_ids]:
            print(f"  {cid_of(c)}  ← {c.get('source', '?')} / {c.get('section', '?')}")
        if eval_set:
            q = eval_set[0].get("question") or eval_set[0].get("query") or ""
            print(f"\n[演示] 「{q}」的向量检索结果：")
            for r in vec_search(q, 3):
                flag = "✅在文件内" if r["chunk_id"] in id_set else "❌不在文件内（payload 缺 chunk_id / 集合没重建）"
                print(f"  {r['chunk_id']}  score={r['score']}  {flag}")
        return

    texts = {cid_of(c): c["text"][: args.preview].replace("\n", " ") for c in chunks} if args.preview else {}
    overlaps, bm_hits, vec_hits, n = [], 0, 0, 0

    for idx, item in enumerate(eval_set, start=1):
        q = item.get("question") or item.get("query") or ""
        bm = bm25_search(q, BM25_K, chunks, bm25)[:TOP_K]
        vs = vec_search(q, VECTOR_K)[:TOP_K]
        hy = reciprocal_rank_fusion(bm25_search(q, BM25_K, chunks, bm25), vec_search(q, VECTOR_K), TOP_K)
        inter = set(x["chunk_id"] for x in bm) & set(x["chunk_id"] for x in vs)
        overlaps.append(len(inter)); n += 1
        bm_hits += sum(1 for x in bm if x["chunk_id"] in id_set)
        vec_hits += sum(1 for x in vs if x["chunk_id"] in id_set)

        def fmt(rows, key):
            parts = []
            for r in rows:
                s = f"{r['chunk_id']}" + (f"(rank{r['rank']})" if key == "rank" else f"(rrf{r['rrf_score']})")
                if texts.get(r["chunk_id"]):
                    s += f"「{texts[r['chunk_id']]}」"
                parts.append(s)
            return ", ".join(parts)

        print(f"Q{idx}: {q}")
        print("  BM25: " + fmt(bm, "rank"))
        print("  向量: " + fmt(vs, "rank"))
        print("  混合: " + fmt(hy, "rrf"))
        print(f"  [ID校验] BM25 {sum(1 for x in bm if x['chunk_id'] in id_set)}/{TOP_K} | 向量 {sum(1 for x in vs if x['chunk_id'] in id_set)}/{TOP_K} | 排序交集 {len(inter)}\n")

    if n:
        print(f"[汇总] 平均排序交集 {sum(overlaps) / n:.1f} / {TOP_K}（排序一致性，不是 id 空间指标）")
        print(f"[汇总] BM25 id 命中 {bm_hits}/{n * TOP_K} | 向量 id 命中 {vec_hits}/{n * TOP_K}")
        print("[验收] 两路 id 命中都应 = 100%；否则检查 payload.chunk_id 或集合是否重建")


if __name__ == "__main__":
    main()
