

# docs/eval/eval_compare.py
# BM25 vs 向量检索：同 chunk、同题目、同判定口径
# W4D3：固定口径重跑 40 题 -> docs/eval/baseline_W4.md
# 运行： python docs/eval/eval_compare.py
# 判定口径（固定，勿改）：TOP_K=5, COVER_THRESHOLD=0.5, USE_AUTO_KEYPOINTS=True

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import jieba
from rank_bm25 import BM25Okapi
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# ================= 固定口径 =================
TOP_K = 5
COVER_THRESHOLD = 0.5
USE_AUTO_KEYPOINTS = True

# ================= 路径配置 =================
EVAL_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = EVAL_DIR.parent.parent / "docs" / "chunks" / "chunks.jsonl"
QUESTIONS_PATH = EVAL_DIR.parent.parent / "docs" / "eval" / "questions.all.jsonl"
BASELINE_OUT = EVAL_DIR / "baseline_W4.md"
BASELINE_W3 = EVAL_DIR / "baseline_W3.md"

# 【新增】修复后的切分与 id->真实source 映射表
FIXED_CHUNKS = EVAL_DIR.parent.parent / "docs" / "chunks" / "chunks_fixed.jsonl"
MAP_PATH = EVAL_DIR.parent.parent / "docs" / "chunks" / "chunks_fixed.jsonl.map.json"
SOURCE_MAP: Dict[str, str] = {}


def load_source_map(p: Path) -> Dict[str, str]:
    """加载 chunks_fixed.jsonl.map.json (id -> 修复后真实source)。缺失则返回空表。"""
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    except Exception as e:
        print("[WARN] 映射表解析失败({}): {}".format(p, e))
        return {}

# ================= 检索配置 =================
QDRANT_URL = "http://localhost:6333"
COLLECTION = "ls_rag_docs_clean_200"
API_KEY_ENV = "SILICONFLOW_API_KEY"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"

# ================= 停用词表 =================
STOP = set("的了和是我你他她它我们你们他们啊吗呢吧不有这那什么怎么一个一下要会可以能也都就又还很太才已经将对被关于等之与及或并而各其该本此以下以上".split())
STOP |= {"什么", "怎么", "一个", "一下", "可以", "已经", "以下", "以上", "关于"}
STOP |= {"的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "我们", "你们", "他们",
         "啊", "吗", "呢", "吧", "不", "有", "这", "那", "要", "会", "能", "也", "都",
         "就", "又", "还", "很", "太", "才", "已", "将", "被", "对", "等", "之", "与",
         "及", "或", "并", "而", "各", "其", "该", "本", "此"}


# ================= 数据加载 =================
def load_chunks_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    if not p.exists():
        raise FileNotFoundError(f"❌ 找不到 chunks 文件: {p}")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_questions(p: Path) -> List[Dict[str, Any]]:
    out = []
    if not p.exists():
        raise FileNotFoundError(f"❌ 找不到评测集文件: {p}")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict) and "question" in obj:
            out.append(obj)
    return out


# ================= 【核心修复】Source 标准化 =================
def norm_src(s: Any) -> str:
    """
    规范化 source 字符串
    【修复】移除 Path(s).name 操作，避免对纯机构名产生意外截断
    【修复】强制 lower() + 去除所有空白/标点，确保集合交集计算准确
    """
    if s is None:
        return ""
    s = str(s).strip()
    # 仅做安全清洗，不做路径解析
    s = re.sub(r'[\s\.\-_\(\)（）\[\]【】\\/]+', '', s)
    return s.lower()


def match_src(exp_set: Set[str], got_set: Set[str]) -> int:
    """
    【最终修复·核心】source 命中判定：双向子串匹配，替代原来的集合交集(==)。
    原因：脏截断名(年普通高等学校)与原始文件名(...2026年普通高等学校...办法.pdf)
    永远不可能逐字相等，交集恒为空 -> 文件级命中恒为0。
    规则：对每个期望值 e，只要 Top5 里有任一 g 与 e 互为子串(较短者长度>=3)即判命中。
    同时覆盖：脏值⊂原始名 / 脏值==脏值 / 原始名⊃脏值 三种真实情况。
    长度<3不参与，避免过短串乱命中；不同文档互不为子串不会误命中。
    """
    hits = 0
    for e in exp_set:
        if not e:
            continue
        for g in got_set:
            if not g:
                continue
            shorter, longer = (e, g) if len(e) <= len(g) else (g, e)
            if len(shorter) < 3:
                continue
            if shorter in longer:
                hits += 1
                break
    return hits


# ================= A/B 类标注修复 =================
# A 类：评测集脏截断名 -> fix_source 的 key 体系
EXPECT_KEY_MAP = {
    "年普通高等学校": "河南省教育厅",
    "年河南省退役大学": "河南省教育厅",
    # C 类：政策文件干净 stem，与 fix_source 重跑后的 key 对齐(identity)
    "2026河南省普通高校专升本招生录取控制分数": "2026河南省普通高校专升本招生录取控制分数",
    "河南省2026年普通高校专升本招生院校章程查询": "河南省2026年普通高校专升本招生院校章程查询",
}
# B 类：拒答/占位符标记，不参与文件级判定
REFUSE_TOKENS = ("—", "－", "--", "超出覆盖范围", "元数据缺失", "拒答")


def is_refuse(q):
    """题目 expected 里含拒答/超范围/元数据缺失等占位符 -> 判为拒答题。"""
    v = q.get("expected_source")
    if v is None:
        return False
    raws = v if isinstance(v, list) else [v]
    return any(any(tok in str(r) for tok in REFUSE_TOKENS) for r in raws)


# ================= 要点提取与覆盖度 =================
def _get_kp(q: Dict[str, Any]) -> List[str]:
    for k in ("expected_keypoints", "keypoints", "要点", "answer_points", "points"):
        v = q.get(k)
        if v:
            if isinstance(v, str):
                return [x.strip() for x in v.replace("；", "\n").replace(";", "\n").split("\n") if x.strip()]
            return [str(x).strip() for x in v if str(x).strip()]
    return []


def _auto_keywords(q: Dict[str, Any]) -> List[str]:
    text = q.get("question", "") + " " + " ".join(str(q.get(k, "")) for k in ("answer", "参考答案", "expected_answer"))
    toks = [t for t in jieba.lcut(text) if len(t) > 1 and t not in STOP and not t.replace(".", "").isdigit()]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:12]


def keypoint_coverage(hit_texts: List[str], keys: List[str]) -> Tuple[float, int]:
    if not keys:
        return 0.0, 0
    joined = "".join(hit_texts)
    hit = 0.0
    for k in keys:
        raw = str(k).strip()
        if not raw:
            continue
        if len(raw) <= 4:
            hit += 1.0 if raw in joined else 0.0
        else:
            toks = [t for t in jieba.lcut(raw) if len(t) > 1 and t not in STOP]
            hit += (sum(1 for t in toks if t in joined) / len(toks)) if toks else (1.0 if raw in joined else 0.0)
    return hit / len(keys), len(keys)


# ================= 【核心修复】期望 Source 解析 =================
def parse_sources(q: Dict[str, Any]) -> Optional[Set[str]]:
    """
    【修复】正确处理 list 类型的 expected_source，避免逗号截断问题
    """
    v = q.get("expected_source")
    if not v:
        return None
    
    # 拒答/超出范围标记
    if isinstance(v, str):
        s = v.strip()
        if s.startswith(("—", "-")) or "拒答" in s or "超出覆盖范围" in s:
            return None
        # 单个字符串，按逗号分割
        return {norm_src(EXPECT_KEY_MAP.get(x.strip(), x.strip())) for x in s.split(",") if x.strip()}
    
    # 列表类型：直接遍历，不再 join 后 split
    if isinstance(v, list):
        result = set()
        for item in v:
            item_str = str(item).strip()
            if item_str and not item_str.startswith(("—", "-")) and "拒答" not in item_str:
                result.add(norm_src(EXPECT_KEY_MAP.get(item_str, item_str)))
        return result if result else None
    
    return None


# ================= 【核心修复】单题评判 =================
def judge_one(q: Dict[str, Any], hits: List[Dict[str, Any]], exp: Optional[Set[str]]) -> Dict[str, Any]:
    labeled = exp is not None and len(exp) > 0
    
    # 【修复】防御性提取 + 正确的变量赋值顺序
    got: Set[str] = set()
    hit_texts: List[str] = []
    for h in hits:
        if isinstance(h, dict):
            src = h.get("source", "")
            if src:
                got.add(norm_src(src))
            hit_texts.append(h.get("text", "") or "")

    exp_norm = exp if exp else set()
    n_src = match_src(exp_norm, got)

    keys = _get_kp(q) or (_auto_keywords(q) if USE_AUTO_KEYPOINTS else [])
    cov, total = keypoint_coverage(hit_texts, keys)

    refuse = q.get("type") == "no_answer" or is_refuse(q)
    if refuse:
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
    return dict(v=v, labeled=labeled, n_src=n_src, cov=cov, total=total, rec=rec, got=got)


# ================= Embedding =================
def embed_one(text: str) -> List[float]:
    load_dotenv()
    key = os.environ.get(API_KEY_ENV, "")
    if not key:
        raise ValueError(f"❌ 未找到环境变量 {API_KEY_ENV}，请检查 .env 文件")
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    r = requests.post(EMBED_URL, headers=headers, json={"model": EMBED_MODEL, "input": [text]}, timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


# ================= BM25 检索 =================
def run_bm25(chunks: List[Dict[str, Any]], qs: List[Dict[str, Any]]) -> List[Tuple]:
    print("⏳ BM25 索引构建中...")
    toks = [list(jieba.cut_for_search(c["text"])) for c in chunks]
    bm = BM25Okapi(toks)
    out = []
    for i, q in enumerate(qs):
        scores = bm.get_scores(list(jieba.cut_for_search(q["question"])))
        idxs = sorted(range(len(scores)), key=lambda idx: -scores[idx])[:TOP_K]
        hits = [chunks[idx] for idx in idxs]
        out.append((q, judge_one(q, hits, parse_sources(q)), hits))
        if (i + 1) % 10 == 0:
            print(f"  ⏳ BM25 已处理 {i + 1}/{len(qs)} 题...")
    return out


# ================= 向量检索 =================
def run_vec(qs: List[Dict[str, Any]]) -> List[Tuple]:
    print("⏳ 向量检索启动中...")
    client = QdrantClient(url=QDRANT_URL)
    out = []
    for i, q in enumerate(qs):
        qv = embed_one(q["question"])
        res = client.query_points(
            collection_name=COLLECTION,
            query=qv,
            limit=TOP_K,
            with_payload=True,
        ).points
        hits = []
        for h in res:
            _src = h.payload.get("source", "")
            _mid = str(h.id) if getattr(h, "id", None) is not None else str(h.payload.get("id", ""))
            _src = SOURCE_MAP.get(_mid, _src)
            hits.append({"text": h.payload.get("text", ""),
                         "source": _src,
                         "section": h.payload.get("section", "")})
        out.append((q, judge_one(q, hits, parse_sources(q)), hits))
        if (i + 1) % 10 == 0:
            print(f"  ⏳ 向量检索已处理 {i + 1}/{len(qs)} 题...")
    return out


# ================= 聚合与报告 =================
def agg(rows: List[Tuple]) -> Dict[str, Any]:
    labeled = [r[1] for r in rows if r[1]["labeled"]]
    denom = len(labeled)
    num = sum(r[1]["rec"] for r in rows if r[1]["labeled"])
    recall = (num / denom * 100) if denom else 0.0
    cnt = lambda vv: sum(1 for _, rr, _ in rows if rr["v"] == vv)
    return dict(recall=recall, denom=denom, dui=cnt("对"), pian=cnt("偏"),
                cuo=cnt("错"), unlabel=cnt("未标注"),
                file_hit=sum(1 for _, rr, _ in rows if rr["n_src"] > 0),
                content_hit=sum(1 for _, rr, _ in rows if rr["cov"] > 0),
                n_rows=len(rows))


def report(name: str, rows: List[Tuple]) -> Dict[str, Any]:
    a = agg(rows)
    print("%s: Recall@%d = %.1f%%  | 对%d 偏%d 错%d 未标注%d  | 文件级命中 %d/%d  | 内容级命中 %d/%d"
          % (name, TOP_K, a["recall"], a["dui"], a["pian"], a["cuo"], a["unlabel"],
             a["file_hit"], a["n_rows"], a["content_hit"], a["n_rows"]))
    return a


def parse_w3(p: Path) -> Optional[Dict[str, float]]:
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8")
    def find(label):
        m = re.search(label + r"[^0-9]*Recall@[0-9]+\s*=\s*([0-9.]+)\s*%", txt)
        return float(m.group(1)) if m else None
    bm = find("BM25")
    ve = find("向量") or find("Vector")
    if bm is None and ve is None:
        return None
    return {"bm25": bm, "vec": ve}


# ================= 主流程 =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=None, help="切分文件；缺省时优先 chunks_fixed.jsonl")
    ap.add_argument("--map", default=None, help="id->真实source 映射表 json")
    args = ap.parse_args()

    global SOURCE_MAP
    map_path = Path(args.map) if args.map else MAP_PATH
    SOURCE_MAP = load_source_map(map_path)
    if SOURCE_MAP:
        print("已加载 source 映射表：{} 条  ({})".format(len(SOURCE_MAP), map_path.name))
    else:
        print("[INFO] 未加载到 source 映射表，将沿用各文件自带 source。")

    if args.chunks:
        chunks_path = Path(args.chunks)
    elif FIXED_CHUNKS.exists():
        chunks_path = FIXED_CHUNKS
        print("默认使用修复后切分：{}".format(FIXED_CHUNKS.name))
    else:
        chunks_path = CHUNKS_PATH
        print("未找到 chunks_fixed.jsonl，回落到原始切分：{}".format(CHUNKS_PATH.name))

    chunks = [c for c in load_chunks_jsonl(chunks_path) if c.get("text")]
    if SOURCE_MAP:
        n_over = 0
        for c in chunks:
            cid = str(c.get("id", ""))
            if cid and cid in SOURCE_MAP:
                c["source"] = SOURCE_MAP[cid]
                n_over += 1
        print("BM25 已按映射表覆盖 source：{}/{} 条".format(n_over, len(chunks)))
    qs = load_questions(QUESTIONS_PATH)
    header = "chunk=%d 题目=%d TOP_K=%d 覆盖阈值=%s" % (len(chunks), len(qs), TOP_K, COVER_THRESHOLD)
    print(header + "\n")

    print("=== BM25 ===")
    r1 = run_bm25(chunks, qs)
    a1 = report("BM25  ", r1)

    print("\n=== 向量检索 ===")
    r2 = run_vec(qs)
    a2 = report("Vector", r2)

    tbl = []
    squeezed = []
    for i, ((q1, j1, h1), (_, j2, h2)) in enumerate(zip(r1, r2), 1):
        mark = ""
        if j1["rec"] > j2["rec"]:
            mark = " <-向量被挤占"
            squeezed.append((i, q1["question"], h1, h2))
        elif j2["rec"] > j1["rec"]:
            mark = " <-向量胜"
        tbl.append((i, j1["v"], j2["v"], q1["question"], mark))

    print("\nBM25 Recall@%d = %.1f%%   向量 Recall@%d = %.1f%%   差值 %+.1fpp"
          % (TOP_K, a1["recall"], TOP_K, a2["recall"], a2["recall"] - a1["recall"]))

    w3 = parse_w3(BASELINE_W3) or {"bm25": 57.1, "vec": 52.9}
    md = []
    md.append("# baseline_W4 (W4D3 重跑基线 · 新切分 + 新口径)")
    md.append("")
    md.append("- 生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M"))
    md.append("- 生成命令：python docs/eval/eval_compare.py")
    md.append("- 固定口径：" + header)
    md.append("- 集合：" + COLLECTION + "（新切分 chunk，payload 字段已对齐 section）")
    md.append("- source 映射表：{} 条（BM25/向量命中均按 chunk id 覆盖为修复后真实来源）".format(len(SOURCE_MAP)))
    md.append("")
    md.append("## 1. 召回汇总 (Recall@%d)" % TOP_K)
    md.append("")
    md.append("| 检索 | Recall@%d | 对 | 偏 | 错 | 未标注 | 文件级命中 | 内容级命中 |" % TOP_K)
    md.append("|---|---|---|---|---|---|---|---|")
    md.append("| BM25 | %.1f%% | %d | %d | %d | %d | %d/%d | %d/%d |" % (a1["recall"], a1["dui"], a1["pian"], a1["cuo"], a1["unlabel"], a1["file_hit"], a1["n_rows"], a1["content_hit"], a1["n_rows"]))
    md.append("| 向量 | %.1f%% | %d | %d | %d | %d | %d/%d | %d/%d |" % (a2["recall"], a2["dui"], a2["pian"], a2["cuo"], a2["unlabel"], a2["file_hit"], a2["n_rows"], a2["content_hit"], a2["n_rows"]))
    md.append("")
    md.append("## 2. 与 W3 对比")
    md.append("")
    if w3:
        md.append("| 检索 | W3 Recall | W4 Recall | 变化 |")
        md.append("|---|---|---|---|")
        bm25w3 = ("%.1f" % w3["bm25"]) if w3.get("bm25") is not None else "—"
        vecw3 = ("%.1f" % w3["vec"]) if w3.get("vec") is not None else "—"
        db = ("%+.1fpp" % (a1["recall"] - w3["bm25"])) if w3.get("bm25") is not None else "—"
        dv_val = (a2["recall"] - w3["vec"]) if w3.get("vec") is not None else None
        dv = ("%+.1fpp" % dv_val) if dv_val is not None else "—"
        md.append("| BM25 | %s | %.1f%% | %s |" % (bm25w3, a1["recall"], db))
        md.append("| 向量 | %s | %.1f%% | %s |" % (vecw3, a2["recall"], dv))
        md.append("")
        if dv_val is not None:
            md.append("> 预期向量明显回升：本次向量 %+.1fpp。%s" % (dv_val, "达标。" if dv_val > 0 else "未回升——回到自检：假设错在哪一环？"))
    else:
        md.append("未找到 baseline_W3.md 或其中无可解析 Recall 数字。")
    md.append("")
    md.append("## 3. 逐题对照（对=命中且覆盖达标，偏=部分命中，错=未命中）")
    md.append("")
    md.append("| Q | BM25 | 向量 | 题目 | 备注 |")
    md.append("|---|---|---|---|---|")
    for i, v1, v2, question, mark in tbl:
        md.append("| %d | %s | %s | %s | %s |" % (i, v1, v2, question[:36], mark.strip()))
    md.append("")
    md.append("## 4. 被挤占的题（向量 rec 低于 BM25，逐题记录）")
    md.append("")
    if squeezed:
        for i, question, h1, h2 in squeezed:
            md.append("- **Q%d %s**" % (i, question))
            md.append("  - BM25 Top5 来源：%s" % str([x.get("source", "") for x in h1]))
            md.append("  - 向量 Top5 来源：%s" % str([x.get("source", "") for x in h2]))
    else:
        md.append("无——本次向量未在任何题上劣于 BM25。")
    md.append("")
    md.append("## 5. 自检")
    md.append("")
    md.append("- 如果向量没回升，假设错在哪一环？（切分粒度 / embedding 语义 / 排序 / id去重）：待填")
    md.append("")

    md.append("## 6. 真实 Source 对照（排查用·每题Top5实际source原样）")
    md.append("")
    md.append("### BM25 Top5 实际 source")
    md.append("")
    for (qj, jj, hh) in r1:
        md.append("- Q: %s | expected=%s | Top5=%s" % (
            qj.get("question", "")[:30], str(qj.get("expected_source", "")),
            [x.get("source", "") for x in hh]))
    md.append("")
    md.append("### Vector Top5 实际 source")
    md.append("")
    for (qj, jj, hh) in r2:
        md.append("- Q: %s | expected=%s | Top5=%s" % (
            qj.get("question", "")[:30], str(qj.get("expected_source", "")),
            [x.get("source", "") for x in hh]))
    md.append("")

    with BASELINE_OUT.open("w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n✅ 已写出基线文件：" + str(BASELINE_OUT))
    print("被挤占题数：%d" % len(squeezed))


if __name__ == "__main__":
    main()
    