from pypdf import PdfReader
import os
import json

def extract_text(pdf_path):
    """提取 PDF 所有页面的文本"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text, size=800, overlap=50):
    """将文本按固定大小和重叠度切块"""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size]
        chunks.append(chunk)
        start += size - overlap
    return chunks

def main():
    raw_dir = "docs/raw"
    output_file = "docs/chunks2.jsonl"

    # 如果输出文件已存在，先清空它
    if os.path.exists(output_file):
        open(output_file, 'w', encoding='utf-8').close()

    # 检查 raw 文件夹是否存在
    if not os.path.exists(raw_dir):
        print(f"找不到文件夹: {raw_dir}")
        return

    doc_idx = 1
    for filename in os.listdir(raw_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(raw_dir, filename)
            print(f"正在处理: {filename}...")

            text = extract_text(filepath)
            if not text:
                print("  -> 未提取到文本，跳过")
                continue

            chunks = chunk_text(text, size=800, overlap=50)

            with open(output_file, 'a', encoding='utf-8') as f:
                # enumerate 加 1，使 chunk 序号从 1 开始（如 doc1_chunk1）
                for i, chunk in enumerate(chunks, 1):
                    chunk_data = {
                        "id": f"doc{doc_idx}_chunk{i}",
                        "source": filename,
                        "text": chunk.strip()
                    }
                    f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")

            doc_idx += 1
            print(f"  -> 成功写入 {len(chunks)} 个 chunk\n")

if __name__ == "__main__":
    main()