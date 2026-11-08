#!/usr/bin/env python
# -*- coding: utf-8 -*-
# docs/eval/ingest_incremental.py —— W6D3 增量补语料（不改历史块）
"""
用法：
    # 1) 先把新文档转成纯文本（或直接给 txt/md 文件）
    python docs/eval/ingest_incremental.py --file new_doc.txt --source "河南省教育考试院" \
        --doc-type score --year 2026 --dry-run

    # 2) 确认无误后正式写入（追加到 chunks_fixed_v2.jsonl）
    python docs/eval/ingest_incremental.py --file new_doc.txt --source "河南省教育考试院" \
        --doc-type score --year 2026

    # 3) 重建 Qdrant（复用 rebuild_qdrant.py）
    python rebuild_qdrant.py

设计原则（重要）：
    - **只追加，不改历史块**：老 139 块的 id 保持不变，历史数字不受影响
    - **id 规则一致**：sha1(source|text)[:12]
    - **去重**：新块的 id 若已存在则跳过（幂等，可重复运行）
    - **备份**：写入前自动备份 chunks_fixed_v2.jsonl
    - **字段顺序一致**：id → source → year → doc_type → section → char_count → text
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _find_chunks():
    """按脚本所在位置的几种可能，定位 chunks_fixed_v2.jsonl"""
    cands = [
        BASE / "docs" / "chunks" / "chunks_fixed_v2.jsonl",          # 脚本在项目根
        BASE.parent.parent / "docs" / "chunks" / "chunks_fixed_v2.jsonl",  # 脚本在 docs/eval/
        BASE / "chunks_fixed_v2.jsonl",                               # 脚本和文件同目录
    ]
    for p in cands:
        if p.exists():
            return p
    return cands[0]


CHUNKS_PATH = _find_chunks()
FIELD_ORDER = ["id", "source", "year", "doc_type", "section", "char_count", "text"]

# 切分参数（与历史一致：目标 800 字、重叠 50）
CHUNK_SIZE = 800
OVERLAP = 50
MIN_CHUNK = 80


def norm_ws(s):
    """归一化空白：保留换行，压缩多余空格"""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\u3000]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """按段落优先、句子兜底切分，块间重叠 overlap 字符"""
    text = norm_ws(text)
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = (buf + "\n" + p) if buf else p
            continue
        if buf:
            chunks.append(buf)
        # 单段超长 → 按句切
        if len(p) > size:
            sents = re.split(r"(?<=[。！？；])", p)
            cur = ""
            for s in sents:
                if len(cur) + len(s) <= size:
                    cur += s
                else:
                    if cur:
                        chunks.append(cur)
                    cur = s
            buf = cur
        else:
            buf = p
    if buf:
        chunks.append(buf)

    # 加重叠
    out = []
    for i, c in enumerate(chunks):
        if i > 0 and overlap:
            c = chunks[i - 1][-overlap:] + c
        if len(c.strip()) >= MIN_CHUNK:
            out.append(c.strip())
    return out


def make_id(source, text):
    return hashlib.sha1(f"{source}|{text}".encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="新语料文本文件（txt/md）")
    ap.add_argument("--source", required=True, help="来源名（如 河南省教育考试院）")
    ap.add_argument("--doc-type", default="policy", choices=["policy", "charter", "score", "notice", "faq"])
    ap.add_argument("--year", default="2026")
    ap.add_argument("--section", default="")
    ap.add_argument("--dry-run", action="store_true", help="只预览不写入")
    ap.add_argument("--chunks", default=str(CHUNKS_PATH))
    args = ap.parse_args()

    src = Path(args.file)
    if not src.exists():
        sys.exit(f"[错误] 找不到文件: {src}")
    raw = src.read_text(encoding="utf-8", errors="ignore")
    pieces = split_text(raw)
    print(f"[切分] {src.name} → {len(pieces)} 块（原文 {len(raw)} 字）")

    chunks_file = Path(args.chunks)
    if not chunks_file.exists():
        sys.exit(
            f"[错误] 找不到 chunk 文件：{chunks_file}\n"
            f"  当前工作目录：{Path.cwd()}\n"
            f"  脚本位置推算的默认路径：{CHUNKS_PATH}\n"
            f"  请用 --chunks 显式指定，例如：\n"
            f"    python ingest_incremental.py --file ... --source ... "
            f"--chunks docs/chunks/chunks_fixed_v2.jsonl"
        )
    old = [json.loads(l) for l in open(chunks_file, encoding="utf-8") if l.strip()] if chunks_file.exists() else []
    old_ids = {c["id"] for c in old}
    print(f"[现有] {len(old)} 块")

    new_recs, skipped = [], 0
    for p in pieces:
        cid = make_id(args.source, p)
        if cid in old_ids:
            skipped += 1
            continue
        rec = {
            "id": cid,
            "source": args.source,
            "year": args.year,
            "doc_type": args.doc_type,
            "section": args.section,
            "char_count": len(p),
            "text": p,
        }
        new_recs.append({k: rec[k] for k in FIELD_ORDER})
        old_ids.add(cid)

    print(f"[新增] {len(new_recs)} 块 | 跳过重复 {skipped} 块")
    if new_recs:
        print(f"[预览] 前 2 块：")
        for r in new_recs[:2]:
            print(f"  {r['id']} [{r['doc_type']}/{r['year']}] {r['text'][:70].replace(chr(10), ' ')}…")
    if not new_recs:
        print("[提示] 没有新块（可能已导入过）")
        return
    if args.dry_run:
        print("[dry-run] 未写入。去掉 --dry-run 正式写入。")
        return

    # 备份
    bak = chunks_file.with_suffix(f".bak.{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    shutil.copy2(chunks_file, bak)
    print(f"[备份] {bak.name}")

    with open(chunks_file, "a", encoding="utf-8") as f:
        for r in new_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(old) + len(new_recs)
    print(f"[完成] {len(old)} → {total} 块")
    print("[下一步] 重建 Qdrant：python rebuild_qdrant.py")
    print("[别忘了] 重跑评估：python docs/eval/eval_hybrid.py --no-quality")


if __name__ == "__main__":
    main()
