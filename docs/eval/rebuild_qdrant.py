#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
rebuild_qdrant.py —— 用 chunks_fixed_v2.jsonl 重建 Qdrant 集合（幂等）
用法：
    python rebuild_qdrant.py                      # 默认读 docs/chunks/chunks_fixed_v2.jsonl，集合名 chunks
    python rebuild_qdrant.py --chunks PATH --collection NAME

要点：
  - 向量维度断言 1024（BGE-M3），不对直接报错
  - point_id = uuid5(NAMESPACE_URL, chunk_id)  ← 确定性 UUID，重复 upsert 自动覆盖（幂等）
  - payload 带 chunk_id：检索结果才能与 BM25 / 评估标注对账
"""
import argparse
import json
import os
import sys
import uuid

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"
DIM = 1024
BATCH = 16


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def embed_batch(texts):
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        sys.exit("[错误] 缺少 SILICONFLOW_API_KEY（写进 .env）")
    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": MODEL, "input": texts},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()["data"]
    vecs = [d["embedding"] for d in data]
    for v in vecs:
        assert len(v) == DIM, f"维度异常 {len(v)}，BGE-M3 应为 {DIM}"
    return vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=os.environ.get("CHUNKS_FILE", "docs/chunks/chunks_fixed_v2.jsonl"))
    ap.add_argument("--collection", default=os.environ.get("COLLECTION", "chunks"))
    ap.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    args = ap.parse_args()
    load_env()

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8") if l.strip()]
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk 文件有重复 id，先跑 fix_chunks.py"
    print(f"[数据] {len(chunks)} 块 ← {args.chunks}")

    client = QdrantClient(url=args.qdrant_url)
    existing = [c.name for c in client.get_collections().collections]
    if args.collection in existing:
        client.delete_collection(args.collection)
        print(f"[集合] 已删除旧集合 {args.collection}")
    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    print(f"[集合] 新建 {args.collection}（dim={DIM}, cosine）")

    points = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        vecs = embed_batch([c["text"] for c in batch])
        for c, v in zip(batch, vecs):
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["id"])),
                vector=v,
                payload={
                    "chunk_id": c["id"],
                    "source": c.get("source", ""),
                    "year": c.get("year", ""),
                    "doc_type": c.get("doc_type", ""),
                    "section": c.get("section", ""),
                    "text": c["text"],
                },
            ))
        print(f"[embed] {min(i + BATCH, len(chunks))}/{len(chunks)}")

    client.upsert(collection_name=args.collection, points=points)
    info = client.get_collection(args.collection)
    print(f"[完成] upsert {len(points)} 个点 | 集合现有 {info.points_count} 个 | 维度 {info.config.params.vectors.size}")
    assert info.points_count == len(chunks), "点数与 chunk 数不一致！"
    print("[验收通过] 点数 = chunk 数；point_id 为 UUID；payload.chunk_id = 文件里的 id")


if __name__ == "__main__":
    main()
