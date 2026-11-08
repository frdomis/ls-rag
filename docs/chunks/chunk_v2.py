
# -*- coding: utf-8 -*-
"""
chunk_v2.py  --  W4D2 结构切分重建

设计目标（对齐任务书）：
- 优先在 条款/编号/标题 边界切（第X条/章/节、一、、（一）、1.、markdown 标题），
  命中不到结构才按长度兜底。
- 块长 300-800 字可接受；关键是语义完整，不为凑长度切断一条政策。
- overlap 50-100 字；丢弃 <50 字碎片、页眉页脚、目录、页码、裸网址。
- 每块带元数据 source / year / doc_type / section（条款号，便于溯源）

"""

import os
import re
import sys
import json
import random
import hashlib
from pathlib import Path

# ------------------------- 路径 -------------------------
ROOT = Path(os.getcwd())
RAW_DIR = ROOT / "docs" / "raw"
OUT_DIR = ROOT / "docs" / "chunks"

# ------------------------- 切分参数 -------------------------
TARGET_MIN = 300      # 短于这个值尝试向前合并
TARGET_MAX = 600      # 兜底/二次切的目标上限（本轮下调，压中位数）
ABS_MAX = 800         # 硬上限：超过必须二次切（任务书 800）
ABS_MIN = 50          # 短于这个值直接丢弃（碎片/页眉页脚）
OVERLAP = 80          # 相邻块重叠字数（任务书 50-100）
EXPECTED_DIM = 1024   # 建 collection 前断言向量维度

# ------------------------- 结构边界正则 -------------------------
STRUCTURE_PATTERNS = [
    re.compile(r'^#{1,6}\s+\S.*$'),                              # markdown 标题
    re.compile(r'^第[0-9一二三四五六七八九十百千]+[章节篇部编].*$'),      # 第X章/节/篇
    re.compile(r'^第[0-9一二三四五六七八九十百千]+条.*$'),            # 第X条
    re.compile(r'^[一二三四五六七八九十百]+、.*$'),                  # 一、二、
    re.compile(r'^（[一二三四五六七八九十]+）.*$'),                   # （一）（二）
    re.compile(r'^\([0-9]+\)[^\d].*$'),                          # (1) (2)
    re.compile(r'^[0-9]+[.、]\s*\S.*$'),                           # 1. 2、
    re.compile(r'^（[0-9]+）[^\d].*$'),                             # （1）（2）
]

NOISE_PATTERNS = [
    re.compile(r'^\s*[-—=＝·—\s]*$'),                 # 空/分隔线
    re.compile(r'^\s*第?\s*[0-9]+\s*页?\s*$'),       # 页码 3 / 第3页
    re.compile(r'^\s*https?://\S+\s*$'),            # 裸网址
    re.compile(r'^[.\u2026\s]{6,}$'),                # 目录点线
    re.compile(r'^\s*[0-9]+\s*\.{3,}.*$'),           # 1 .......... 标题
]

SENT_SPLIT = re.compile(r'(?<=[。；;！？])')


# ------------------------- 清洗 -------------------------
def is_noise(line):
    s = line.strip()
    if not s:
        return True
    return any(p.match(s) for p in NOISE_PATTERNS)


def clean_body(text):
    lines = [ln for ln in text.splitlines() if not is_noise(ln)]
    body = "\n".join(lines)
    body = re.sub(r'\n{3,}', '\n\n', body)
    return body.strip()


def parse_front_matter(raw):
    meta = {"source": "unknown", "year": "", "doc_type": "unknown"}
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            fm = raw[3:end]
            for line in fm.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip().lower()
                    v = v.strip().strip('"').strip("'")
                    if k in meta:
                        meta[k] = v
            return meta, raw[end + 4:]
    return meta, raw


# ------------------------- 结构切分 -------------------------
def is_boundary(line):
    s = line.strip()
    if not s:
        return False
    return any(p.match(s) for p in STRUCTURE_PATTERNS)


def extract_section(line):
    s = line.strip()
    for pat in [r'^(第[0-9一二三四五六七八九十百千]+[条章节篇部编])',
                r'^([一二三四五六七八九十百]+、)',
                r'^(（[一二三四五六七八九十]+）)',
                r'^([0-9]+[.、])']:
        m = re.match(pat, s)
        if m:
            return m.group(1)
    m = re.match(r'^(#{1,6}\s+\S.*)', s)
    if m:
        return m.group(1).strip()[:20]
    return ""


def split_by_structure(text):
    lines = text.split("\n")
    segments = []
    cur_lines = []
    cur_section = ""
    hit_any = False
    for ln in lines:
        if is_boundary(ln):
            hit_any = True
            if cur_lines and "".join(cur_lines).strip():
                segments.append((cur_section, "\n".join(cur_lines).strip()))
            cur_section = extract_section(ln)
            cur_lines = [ln]
        else:
            cur_lines.append(ln)
    if cur_lines and "".join(cur_lines).strip():
        segments.append((cur_section, "\n".join(cur_lines).strip()))
    if not hit_any:
        return [("", text.strip())]
    return segments


# ------------------------- 长度兜底切 -------------------------
def length_split(text, target_max, overlap):
    sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return [text] if len(text) >= ABS_MIN else []
    pieces = []
    buf = ""
    for sent in sents:
        if len(buf) + len(sent) <= target_max:
            buf += sent
        else:
            if buf:
                pieces.append(buf)
            while len(sent) > target_max:
                pieces.append(sent[:target_max])
                sent = sent[target_max - overlap:] if overlap else sent[target_max:]
            buf = sent
    if buf.strip():
        pieces.append(buf)
    out = []
    for i, p in enumerate(pieces):
        if i == 0:
            out.append(p)
        else:
            tail = pieces[i - 1][-overlap:]
            out.append(tail + p)
    return out


def enforce_max(text):
    if len(text) <= ABS_MAX:
        return [text]
    return length_split(text, TARGET_MAX, OVERLAP)


def merge_short(segs):
    merged = []
    for section, text in segs:
        if merged and len(text) < TARGET_MIN:
            ps, pt = merged[-1][0], merged[-1][1]
            if len(pt) + len(text) <= ABS_MAX:
                merged[-1] = (ps or section, pt + "\n" + text, merged[-1][2])
                continue
        merged.append((section, text, False))
    for i, (sec, txt, frag) in enumerate(merged):
        if len(txt) < TARGET_MIN:
            merged[i] = (sec, txt, True)
    return merged


# ------------------------- 主切分 -------------------------
def build_chunks_from_body(body, source, year, doc_type):
    body = clean_body(body)
    if len(body) < ABS_MIN:
        return []
    segs = split_by_structure(body)
    expanded = []
    for section, text in segs:
        for sub in enforce_max(text):
            expanded.append((section, sub))
    finalized = merge_short(expanded)
    chunks = []
    for section, text, is_frag in finalized:
        if len(text) < ABS_MIN:
            continue
        chunks.append({
            "id": hashlib.md5((source + text[:60]).encode()).hexdigest()[:12],
            "source": source,
            "year": str(year),
            "doc_type": doc_type,
            "section": section or "（无编号）",
            "char_count": len(re.sub(r'\s', '', text)),
            "text": text,
        })
    return chunks


# ------------------------- 向量维度断言 -------------------------
def assert_vector_dim(dim, expected=EXPECTED_DIM):
    if dim != expected:
        raise AssertionError(
            "向量维度不匹配：实际 %d != 期望 %d。检查 embedding 模型输出维度后再建 collection。"
            % (dim, expected))
    return True


# ------------------------- 自测 -------------------------
def selftest():
    print("== chunk_v2 自测 ==")

    long_txt = "这是一段没有任何结构编号的说明文字用来测试超长二次切。" * 90
    cs = build_chunks_from_body(long_txt, "test_long", "2026", "policy")
    mx = max(len(c["text"]) for c in cs)
    print("用例1 无结构长文: 块数=%d max=%d" % (len(cs), mx))
    assert mx <= ABS_MAX, "超长二次切失败 max=%d" % mx

    mixed = ("第一条 报名时间为每年三月一日至十五日，逾期不予受理，由各省教育考试院统一安排现场确认。\n"
             "附。\n"
             "第二条 考试科目分为公共课与专业课，公共课含英语与计算机基础，"
             "专业课按报考专业确定，具体以当年公布的考试大纲为准，考试大纲由省教育考试院组织专家编写。\n"
             "尾巴残句残\n")
    cs2 = build_chunks_from_body(mixed, "test_mix", "2026", "charter")
    mn = min(len(c["text"]) for c in cs2)
    secs = [c["section"] for c in cs2]
    print("用例2 条款+碎片: 块数=%d min=%d sections=%s" % (len(cs2), mn, secs))
    assert mn >= ABS_MIN, "碎片未清干净 min=%d" % mn
    assert any("第一条" in s for s in secs), "条款号未抽到"

    short_a = "一、报名条件。具有河南户籍且拥有全日制专科学历的应届或往届毕业生均可报名。"
    short_b = "二、报名时间。统一安排在每年三月上旬。"
    cs3 = build_chunks_from_body(short_a + "\n" + short_b, "test_short", "2026", "notice")
    print("用例3 短块合并: 块数=%d lens=%s" % (len(cs3), [len(c["text"]) for c in cs3]))

    try:
        assert_vector_dim(1)
        raise RuntimeError("维度=1 应报错却没报")
    except AssertionError:
        print("用例4 维度断言: 维度=1 正确拦截")
    assert assert_vector_dim(1024) is True

    print("自测全部通过。")


# ------------------------- 运行 -------------------------
def run():
    if not RAW_DIR.exists():
        print("未找到 raw 目录：%s" % RAW_DIR)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    nfiles = 0
    for md in sorted(RAW_DIR.glob("*.md")):
        raw = md.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        cs = build_chunks_from_body(
            body, meta.get("source", md.stem),
            meta.get("year", ""), meta.get("doc_type", "unknown"))
        if cs:
            nfiles += 1
            all_chunks.extend(cs)
    out_jsonl = OUT_DIR / "chunks.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False))
            f.write("\n")
    sample = random.sample(all_chunks, min(10, len(all_chunks))) if all_chunks else []
    sample_txt = OUT_DIR / "sample_10.txt"
    with sample_txt.open("w", encoding="utf-8") as f:
        for i, c in enumerate(sample, 1):
            f.write("[块%d] source=%s year=%s type=%s section=%s len=%d\n%s\n\n---\n\n"
                    % (i, c["source"], c["year"], c["doc_type"], c["section"],
                       c["char_count"], c["text"]))
    lens = sorted(len(c["text"]) for c in all_chunks)
    n = len(lens)
    median = lens[n // 2] if n else 0
    mean = int(sum(lens) / n) if n else 0
    print("=" * 50)
    print("文件数: %d  块数: %d" % (nfiles, n))
    print("块长 min/median/mean/max = %d/%d/%d/%d"
          % (lens[0] if n else 0, median, mean, lens[-1] if n else 0))
    print("chunks: %s" % out_jsonl)
    print("抽查样本(10块): %s" % sample_txt)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
        