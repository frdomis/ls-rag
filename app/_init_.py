# app/_init_.py —— 图书馆管理员：PDF → 切块 → 向量 → Qdrant
import glob
import hashlib
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.embed import embed          # ★ 直接用，它已经是 list[str] -> list[list[float]]

# ============ 配置 ============
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "docs" / "raw"
COL = "ls_rag_docs"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
WRITE_BATCH = 64
EMBED_BATCH = 32


# ============ 解析 ============
def parse_pdf(path: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_pdfs(root: Path) -> list[dict]:
    out = []
    files = sorted(glob.glob(str(root / "*.pdf")))
    if not files:
        raise FileNotFoundError(f"目录 {root} 下没有 PDF")
    for p in files:
        text = parse_pdf(p)
        out.append({"source": Path(p).name, "text": text})
        print(f"  [解析] {Path(p).name} -> {len(text)} 字")
    return out


# ============ 切块 ============
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    out, i = [], 0
    while i < len(text):
        seg = text[i:i + size].strip()
        if seg:
            out.append(seg)
        i += size - overlap
    return out


def chunk_docs(docs: list[dict]) -> list[dict]:
    out = []
    for doc in docs:
        segs = chunk_text(doc["text"])
        for k, seg in enumerate(segs):
            out.append({"source": doc["source"], "start": k * (CHUNK_SIZE - CHUNK_OVERLAP), "text": seg})
        print(f"  [切块] {doc['source']} -> {len(segs)} chunks")
    return out


# ============ 重建索引 ============
def rebuild_index():
    client = QdrantClient(url="http://localhost:6333")

    # 1) 解析
    print("[1/4] 正在解析 PDF ...")
    docs = load_pdfs(RAW_DIR)

    # 2) 切块
    print("[2/4] 正在切块 ...")
    chunks = chunk_docs(docs)
    if not chunks:
        raise RuntimeError("没有切出任何 chunk，PDF 可能是扫描版图片")
    print(f"  合计 {len(chunks)} 个 chunk")

    # 3) 向量化（先探一条，把维度钉死）
    print("[3/4] 正在计算向量 ...")
    sample = embed(["__probe__"])
    dim = len(sample[0])
    assert isinstance(dim, int) and 64 <= dim <= 4096, f"维度异常: {dim}"
    print(f"  向量维度 {dim}")

    texts = [c["text"] for c in chunks]      # ★ 唯一一处定义 texts
    vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        vecs.extend(embed(texts[i:i + EMBED_BATCH]))
        print(f"  编码进度 {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")

    # 4) 删旧建新
    if client.collection_exists(COL):
        client.delete_collection(COL)
        print(f"[4/4] 已删除旧集合 {COL}")
    client.create_collection(COL, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    print(f"  已创建集合 {COL}，维度={dim}")

    # 5) 小批量写入
    for i in range(0, len(chunks), WRITE_BATCH):
        batch_chunks = chunks[i:i + WRITE_BATCH]
        pts = []
        for idx, c in enumerate(batch_chunks):
            vid = hashlib.md5(f"{c['source']}:{c['start']}".encode()).hexdigest()
            v = vecs[i + idx]
            pts.append(models.PointStruct(
                id=vid,
                vector=v if isinstance(v, list) else v.tolist(),
                payload={"text": c["text"][:1000], "source": c["source"], "start": c["start"]},
            ))
        client.upsert(COL, points=pts)
        print(f"  写入进度 {min(i + WRITE_BATCH, len(chunks))}/{len(chunks)} 条")

    # 6) 校验
    total = client.count(COL).count
    print(f"\n✅ 入库完成！共 {total} 个 chunk（维度 {dim}）")
    _print_distribution(client)
    return total


def _print_distribution(client: QdrantClient):
    sources, offset = [], None
    while True:
        points, offset = client.scroll(COL, limit=500, with_payload=["source"], offset=offset)
        sources.extend(p.payload.get("source", "<未知>") for p in points)
        if offset is None:
            break
    print("\n=== 库里 source 分布 ===")
    for src, n in sorted(Counter(sources).items(), key=lambda x: -x[1]):
        print(f"  {n:>6} | {'█' * max(1, n * 40 // len(sources))} {src}")


if __name__ == "__main__":
    rebuild_index()