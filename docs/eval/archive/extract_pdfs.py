# docs/eval/extract_pdfs.py     bm25_chunks.json来源
"""从原始 PDF 提取纯文本 chunk，供 BM25 评测使用（不动 Qdrant）"""
import json
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("请先安装: pip install pdfplumber")
    sys.exit(1)

PDF_DIR = Path(r"D:\vscode\ls-rag\docs\raw")   # ★★★ 改成你的 PDF 目录 ★★★
OUT = Path(__file__).parent / "bm25_chunks.json"
CHUNK_SIZE = 500
OVERLAP = 80


def slide(text, size, overlap):
    text = "".join(text.split())          # 去换行、去多余空白
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return [c for c in out if c]


def main():
    if not PDF_DIR.exists():
        print(f"[ERROR] 目录不存在: {PDF_DIR}"); sys.exit(1)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] {PDF_DIR} 下没有 .pdf"); sys.exit(1)
    print(f"找到 {len(pdfs)} 个 PDF，开始提取...")
    chunks = []
    for p in pdfs:
        try:
            with pdfplumber.open(p) as doc:
                for pi, page in enumerate(doc.pages, 1):
                    t = (page.extract_text() or "").strip()
                    if not t:
                        continue
                    for ci, c in enumerate(slide(t, CHUNK_SIZE, OVERLAP)):
                        chunks.append({"id": f"{p.name}#p{pi}#c{ci}", "text": c, "source": p.name, "page": pi})
        except Exception as e:
            print(f"  [FAIL] {p.name}: {e}")
    OUT.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    n = len(chunks); chars = sum(len(c["text"]) for c in chunks)
    bad = sum(1 for c in chunks if len(set(c["text"])) < 20 or "endobj" in c["text"])
    print(f"完成：{n} 个 chunk / {chars} 字 -> {OUT}")
    print(f"自检：疑似无效 chunk {bad} 条（>0 说明有扫描版/图片型 PDF，需 OCR）")


if __name__ == "__main__":
    main()