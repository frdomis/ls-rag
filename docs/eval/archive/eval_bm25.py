# docs/eval/eval_bm25_local.py
import json
import sys
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

TOP_K = 5
COVER_THRESHOLD = 0.5        # 无标注要点时此值不影响结果；有要点后建议 0.5
USE_AUTO_KEYPOINTS = True    # True=用题目/答案自动抽词当要点；False=只计来源命中

ROOT = Path(__file__).parent
CHUNKS_PATH = ROOT / "bm25_chunks.json"
QUESTIONS_PATH = ROOT / "questions_add.jsonl"

STOP = {"的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "我们", "你们",
        "他们", "啊", "吗", "呢", "吧", "不", "有", "这", "那", "什么", "怎么",
        "一个", "一下", "一下", "要", "会", "可以", "能", "也", "都", "就", "又",
        "还", "很", "太", "才", "已", "已经", "将", "被", "对", "关于", "等", "之",
        "与", "及", "或", "并", "而", "各", "其", "该", "本", "此", "以下", "以上"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(path):
    qs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict) and "question" in obj:
            qs.append(obj)
    return qs


def norm_src(s):
    s = str(s or "").strip().replace("\\", "/")
    while "//" in s:
        s = s.replace("//", "/")
    if "/" in s or ":" in s:
        s = Path(s).name
    return s.lower()


def _get_kp(q):
    """从题目记录里捞要点，支持多种字段名；返回 list[str]"""
    for k in ("expected_keypoints", "keypoints", "要点", "answer_points",
              "key_points", "reference_points", "points"):
        v = q.get(k)
        if v:
            if isinstance(v, str):
                return [x.strip() for x in v.replace("；", "\n").replace(";", "\n")
                                          .replace("\r", "").split("\n") if x.strip()]
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
    return []


def _auto_keywords(q):
    """没有标注要点时：从题目 + 参考答案里抽实词，作为宽松要点"""
    text = q.get("question", "")
    for k in ("answer", "参考答案", "expected_answer", "content"):
        v = q.get(k)
        if v:
            text += " " + str(v)
    toks = [t for t in jieba.lcut(text) if len(t) > 1 and t not in STOP and not t.replace(".", "").isdigit()]
    # 去重保序
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:12]


def build_bm25(chunks):
    toks = [list(jieba.cut_for_search(c["text"])) for c in chunks]
    return BM25Okapi(toks)


def search(bm25, chunks, query, k=5):
    scores = bm25.get_scores(list(jieba.cut_for_search(query)))
    idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(chunks[i], float(scores[i])) for i in idxs]


def parse_sources(q):
    v = q.get("expected_source")
    if not v:
        return None
    s = str(v).strip()
    if s.startswith(("—", "-")) or "拒答" in s or "超出覆盖范围" in s:
        return None
    if isinstance(v, list):
        s = ",".join(str(x) for x in v)
    return {norm_src(x) for x in s.split(",") if x.strip()}


def keypoint_coverage(hit_texts, keys):
    """词级覆盖率：长句分词算重合比例，短短语退化为子串匹配"""
    if not keys:
        return 0.0, 0
    joined = "".join(hit_texts)
    joined_toks = set(joined)          # 字符级即可覆盖短语场景；下面逐词 in joined
    hit = 0.0
    for k in keys:
        raw = str(k).strip()
        if not raw:
            continue
        if len(raw) <= 4:              # 短短语：严格子串
            hit += 1.0 if raw in joined else 0.0
        else:
            toks = [t for t in jieba.lcut(raw) if len(t) > 1 and t not in STOP]
            if not toks:
                hit += 1.0 if raw in joined else 0.0
            else:
                hit += sum(1 for t in toks if t in joined) / len(toks)
    return hit / len(keys), len(keys)


def main():
    if not CHUNKS_PATH.exists():
        print(f"[ERROR] 先运行 extract_pdfs.py 生成 {CHUNKS_PATH}"); sys.exit(1)

    chunks = [c for c in load_json(CHUNKS_PATH) if c.get("text")]
    qs = load_questions(QUESTIONS_PATH)
    print(f"[BM25] chunk={len(chunks)} 题目={len(qs)} TOP_K={TOP_K} 覆盖阈值={COVER_THRESHOLD}\n")

    bm25 = build_bm25(chunks)
    rows = []

    for qi, q in enumerate(qs, 1):
        hits_sc = search(bm25, chunks, q["question"], TOP_K)
        hits = [h for h, _ in hits_sc]

        exp = parse_sources(q)
        labeled = exp is not None
        got = {norm_src(h.get("source", "")) for h in hits}
        n_src = len((exp or set()) & got)

        keys = _get_kp(q)
        mode = "标注要点"
        if not keys and USE_AUTO_KEYPOINTS:
            keys = _auto_keywords(q)
            mode = "自动抽词"
        cov, total = keypoint_coverage([h["text"] for h in hits], keys)

        if q.get("type") == "no_answer":
            v = "拒答"
        elif not labeled:
            v = "未标注"
        elif total == 0:
            v = "对" if n_src > 0 else "错"
        elif n_src > 0 and cov >= COVER_THRESHOLD:
            v = "对"
        elif n_src > 0 or cov > 0:
            v = "偏"
        else:
            v = "错"

        rec = 1.0 if (n_src > 0 and cov >= COVER_THRESHOLD) else (0.5 if (n_src > 0 or cov > 0) else 0.0)
        rows.append(dict(qi=qi, q=q, v=v, labeled=labeled, n_src=n_src, cov=cov, total=total,
                         rec=rec, mode=mode, got=got))

        # 详情输出
        print(f"Q{qi:2d} [{v:4s}] src={n_src}/{len(exp) if exp else 0}  要点={int(cov*total)}/{total}({cov:.2f}) [{mode}] | {q['question'][:34]}")
        if exp and (exp - got):
            print(f"        来源未命中: {exp - got}")
        if keys and cov < 1.0:
            miss = [k for k in keys if str(k).strip() not in "".join(h["text"] for h in hits)]
            if miss:
                print(f"        要点未命中: {miss[:4]}")

    # 汇总
    labeled = [r for r in rows if r["labeled"]]
    noans = [r for r in rows if r["q"].get("type") == "no_answer"]
    denom = len(labeled) - len(noans)
    num = sum(r["rec"] for r in labeled if r["q"].get("type") != "no_answer")

    cnt = lambda vv: sum(1 for r in rows if r["v"] == vv)
    src_any = sum(1 for r in rows if r["n_src"] > 0)
    cov_any = sum(1 for r in rows if r["cov"] > 0)
    oral_lab = [r for r in rows if r["q"].get("oral") is True and r["labeled"] and r["q"].get("type") != "no_answer"]
    oral_r = sum(r["rec"] for r in oral_lab) / len(oral_lab) * 100 if oral_lab else 0
    fc = sum(1 for r in noans if r["n_src"] > 0)

    print("=" * 64)
    print(f"分母构成: 总{len(rows)} = 参与Recall{denom} + 拒答{cnt('拒答')} + 未标注{cnt('未标注')}")
    print(f"Recall@{TOP_K} = {num/denom*100:.1f}%  (分子={num:.2f})")
    print(f"判定分布: 对{cnt('对')} 偏{cnt('偏')} 错{cnt('错')} 未标注{cnt('未标注')} 拒答{cnt('拒答')}")
    print(f"--- 诊断 ---")
    print(f"来源至少命中1个: {src_any}/{len(rows)}   (文件级召回)")
    print(f"要点覆盖>0:      {cov_any}/{len(rows)}   (内容级召回)")
    print(f"口语化题 Recall: {oral_r:.1f}% ({len(oral_lab)}题)")
    if noans:
        print(f"拒答题误召率: {fc/len(noans)*100:.1f}% ({fc}/{len(noans)})")
    else:
        print("拒答题误召率: N/A（本题集无拒答题）")


if __name__ == "__main__":
    main()