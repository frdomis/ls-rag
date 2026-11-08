# check.py   ls_rag_docs_clean 与 bm25_chunks.json 完全一致
import json
import random
from qdrant_client import QdrantClient

BM25_PATH = "docs/eval/bm25_chunks.json"
COLLECTIONS = ["ls_rag_docs_clean", "ls_rag_docs"]
QDRANT_URL = "http://localhost:6333"
SAMPLE_N = 5

def load_local(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_qdrant(collection):
    client = QdrantClient(url=QDRANT_URL)
    pts, _ = client.scroll(collection, limit=10000, with_payload=True, with_vectors=False)
    return pts

def make_key(chunk):
    """统一用 (source, text) 作为唯一标识"""
    src = chunk.get("source", "") if isinstance(chunk, dict) else getattr(chunk, "source", "")
    txt = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
    return (src.strip(), txt.strip())

def main():
    print("=" * 60)
    print("W4D1 一致性校验: bm25_chunks.json VS Qdrant 集合")
    print("=" * 60)

    local_chunks = load_local(BM25_PATH)
    local_keys = {make_key(c) for c in local_chunks}
    print(f"\n[OK] 本地 bm25_chunks.json 共 {len(local_chunks)} 条\n")

    all_pass = True

    for col in COLLECTIONS:
        print(f"--- 正在读取 Qdrant 集合: {col} ---")
        pts = fetch_qdrant(col)
        qdrant_keys = {make_key(p.payload) for p in pts}
        print(f"[OK] Qdrant 集合 '{col}' 共 {len(pts)} 条\n")

        common = local_keys & qdrant_keys
        only_local = local_keys - qdrant_keys
        only_qdrant = qdrant_keys - local_keys

        print(f"  本地 chunks:     {len(local_chunks)}")
        print(f"  Qdrant 向量:     {len(pts)}")
        print(f"  共同 (source+text): {len(common)}")
        print(f"  仅本地有:         {len(only_local)}")
        print(f"  仅Qdrant有:       {len(only_qdrant)}")

        # 抽样比对
        if len(common) > 0:
            sample_local = random.sample(list(local_chunks), min(SAMPLE_N, len(local_chunks)))
            text_match = 0
            source_match = 0
            for c in sample_local:
                key = make_key(c)
                if key in qdrant_keys:
                    # 找到对应点后逐字段比对
                    for p in pts:
                        if make_key(p.payload) == key:
                            if p.payload.get("text", "").strip() == c.get("text", "").strip():
                                text_match += 1
                            if p.payload.get("source", "").strip() == c.get("source", "").strip():
                                source_match += 1
                            break
            print(f"  抽样 text 一致:   {text_match}/{len(sample_local)}")
            print(f"  抽样 source 一致: {source_match}/{len(sample_local)}")

        if len(common) == len(local_keys) == len(qdrant_keys):
            print(f"  >>> [PASS] 该集合与本地文件完全一致!\n")
        else:
            print(f"  >>> [FAIL] 该集合与本地文件不一致!\n")
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("结论: 一致性校验通过，W3 的 BM25 vs 向量结论有效。")
    else:
        print("结论: 存在不一致！需排查差异来源后重跑实验。")
    print("=" * 60)

    # 保存详细结果
    result = {
        "local_count": len(local_chunks),
        "collections": {}
    }
    # ... 可按需补充完整结果写入
    with open("docs/eval/consistency_check_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()