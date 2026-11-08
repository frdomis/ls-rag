#考试监考官（负责【评测/质检】）
import json
import os
from pathlib import Path

from app.search import search
from qdrant_client import QdrantClient
from collections import Counter

# ============ 诊断：看看库里实际存了哪些文件名 ============
def dump_sources():
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=10)
        points, _ = client.scroll("ls_rag_docs", limit=300, with_payload=["source"])
        src = [p.payload.get("source", "<无source>") for p in points]
        print("=== 库里 source 分布（Top20）===")
        for k, v in Counter(src).most_common(20):
            print(f"  {v:>4} | {repr(k)}")
        print(f"(共 {len(points)} 条)\n")
    except Exception as e:
        print(f"[诊断跳过] {e}\n")

# ============ 主流程 ============
EVAL_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
QS   = EVAL_DIR / "questions.jsonl"
OUT  = RESULTS_DIR / "exp0_bge.json"
K    = 5

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

qs = [json.loads(l) for l in QS.read_text(encoding="utf-8").splitlines() if l.strip()]
def is_retrieval_task(target: str) -> bool:
    return str(target or "").strip().endswith(".pdf")

valid_qs = [q for q in qs if is_retrieval_task(q.get("expected_source", ""))]
excluded_count = len(qs) - len(valid_qs)

print(f"\n⚠️ 已排除 {excluded_count} 道拒答/元数据题，参与评测题数: {len(valid_qs)}\n")
hits, details = 0, []
for q in valid_qs:
    rows = search(q["question"], top_k=K)

    # ===== 调试：只看第一条 =====
    if q["id"] == "Q1":
        print("=== Q1 原始返回 ===")
        print("条数:", len(rows))
        for i, r in enumerate(rows):
            print(i, list(r.keys()), "| source =", repr(r.get("source")), "| score =", r.get("score"))
        print("==================\n")
    # ============================

    hit_docs = [r["source"] for r in rows if r.get("source")]
    target = q.get("expected_source", "")

    hit = bool(target) and target in hit_docs
    if hit:
        hits += 1

    details.append({
        "id": q["id"],
        "target": target,
        "top5": hit_docs[:5],
        "hit": hit,
    })

rec = round(hits / len(valid_qs), 3)
print(f"\n>>> 净化后 Recall@{K} = {rec}  ({hits}/{len(valid_qs)})")
OUT.write_text(
    json.dumps({"k": K, "recall@5": rec, "hit": hits, "total": len(qs), "details": details},
               ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f">>> Recall@{K} = {rec}  ({hits}/{len(qs)})")
print("-" * 30)
for d in details:
    mark = "✅" if d["hit"] else "❌"
    print(f"  {mark} [{d['id']}]: target={d['target']}")
    if not d["hit"] and d["top5"]:
        print(f"       top5={[os.path.basename(str(s)) for s in d['top5']]}")

if __name__ == "__main__":
    dump_sources()