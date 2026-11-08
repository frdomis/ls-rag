#!/usr/bin/env python
# -*- coding: utf-8 -*-
# docs/eval/eval_hybrid.py  —— 最终版
"""
混合检索评估（唯一官方脚本）
- 检索：BM25 / 向量(Qdrant+BGE-M3) / RRF 融合 / 加权融合（多组权重）
- 回答质量轴：生成答案 → LLM 逐要点判覆盖 → 归因 + 人工抽检导出
- 口径（2026-10-22 基线冻结）：
    * recall@5 = 块级命中题数 / 可答题数(33)，答案 chunk 来自 questions_annotated.jsonl
    * 拒答题(3) 单列拒答正确率；语料缺失题(4) 不进分母
    * 要点覆盖率 = 覆盖要点 / 语料可支撑要点(82)
    * 数字一律 N/M = P%，禁止裸百分比

用法：
    python docs/eval/eval_hybrid.py                # 跑全部（含回答质量轴，会调 LLM）
    python docs/eval/eval_hybrid.py --no-quality   # 只跑检索指标（快、省钱）
    python docs/eval/eval_hybrid.py --limit 10     # 只跑前 10 题（调试）

依赖：pip install jieba rank-bm25 requests qdrant-client python-dotenv
配置：.env 里 SILICONFLOW_API_KEY
"""
import argparse
import json
import os
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import jieba
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

load_dotenv()

# ---------------- 配置 ----------------
TOP_K = 5
CAND_K = 30
RRF_K = 60
COVER_THRESHOLD = 0.5
SAMPLE_N = 5

BASE = Path(__file__).resolve().parent          # docs/eval
CHUNKS_PATH = Path(os.environ.get("CHUNKS_FILE", BASE.parent.parent / "docs" / "chunks" / "chunks_fixed_v2.jsonl"))
QUESTIONS_PATH = Path(os.environ.get("EVAL_FILE", BASE / "questions_v2.jsonl"))
ANNOTATED_PATH = Path(os.environ.get("ANNOTATED_FILE", BASE / "questions_annotated.jsonl"))

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION", "chunks")
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
LLM_URL = "https://api.siliconflow.cn/v1/chat/completions"
GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek-ai/DeepSeek-V3")        # 生成答案
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-V3")    # 判要点覆盖
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

if not API_KEY:
    print("[错误] 缺少 SILICONFLOW_API_KEY，写进项目 .env")

STOP = {"的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "我们", "你们",
        "他们", "啊", "吗", "呢", "吧", "不", "有", "这", "那", "什么", "怎么",
        "一个", "一下", "要", "会", "可以", "能", "也", "都", "就", "又", "还",
        "很", "太", "才", "已", "已经", "将", "被", "对", "关于", "等", "之",
        "与", "及", "或", "并", "而", "各", "其", "该", "本", "此", "以下", "以上"}


# ---------------- 工具 ----------------
def load_json(p):
    p = Path(p)
    if not p.exists():
        sys.exit(f"[错误] 文件不存在: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_questions(p):
    out = []
    for i, line in enumerate(open(p, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[跳过] 第 {i} 行解析失败: {e}")
            continue
        if isinstance(obj, dict) and "question" in obj:
            out.append(obj)
    return out


def norm(s):
    return "".join(str(s or "").lower().split())


def grams(text, n=4):
    t = norm(text)
    if len(t) <= n:
        return [t]
    return [t[i:i + n] for i in range(0, len(t) - n + 1, 2)]


def _get_kp(q):
    for k in ("expected_keypoints", "keypoints", "要点", "answer_points", "points"):
        v = q.get(k)
        if v:
            if isinstance(v, str):
                return [x.strip() for x in v.replace("；", "\n").replace(";", "\n").split("\n") if x.strip()]
            return [str(x).strip() for x in v if str(x).strip()]
    return []


def is_refuse(q):
    src = str(q.get("expected_source") or "")
    return ("超出" in src) or ("拒答" in src)


# ---------------- 标注（块级） ----------------
def load_annotations():
    ann = {}
    if ANNOTATED_PATH.exists():
        for a in load_json(ANNOTATED_PATH):
            if a.get("id"):
                ann[str(a["id"])] = a
    return ann


def classify(q, ann):
    a = ann.get(str(q.get("id")), {})
    cids = [str(c) for c in (a.get("chunk_ids") or []) if str(c).strip()]
    return {
        "chunk_ids": set(cids),
        "refuse": is_refuse(q),
        "unanswerable": (not is_refuse(q)) and (not cids),
    }


# ---------------- 检索 ----------------
def run_bm25(chunks, bm, qs, k):
    """返回每题的 top-k，元素含 chunk_id。"""
    out = []
    for q in qs:
        scores = bm.get_scores(list(jieba.cut_for_search(q["question"])))
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out.append([{
            "chunk_id": str(chunks[i]["id"]),
            "text": chunks[i].get("text", ""),
            "source": chunks[i].get("source", ""),
            "year": chunks[i].get("year", ""),
        } for i in idxs])
    return out


def embed_one(text):
    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=60,
    )
    r.raise_for_status()
    v = r.json()["data"][0]["embedding"]
    assert len(v) == EMBED_DIM, f"维度异常 {len(v)}"
    return v


def run_vec(qs, client, k):
    out = []
    for q in qs:
        qv = embed_one(q["question"])
        try:
            res = client.query_points(
                collection_name=COLLECTION, query=qv, limit=k,
                with_payload=["text", "source", "year", "chunk_id"],
            ).points
        except AttributeError:  # 老版客户端
            res = client.search(collection_name=COLLECTION, query_vector=qv, limit=k)
        out.append([{
            "chunk_id": str((h.payload or {}).get("chunk_id") or h.id),
            "text": (h.payload or {}).get("text", ""),
            "source": (h.payload or {}).get("source", ""),
            "year": (h.payload or {}).get("year", ""),
        } for h in res])
    return out


# ---------------- 融合（按 chunk_id，不按 source+page） ----------------
def fuse_rrf(lists, k=TOP_K, rrf_k=RRF_K):
    score, order = {}, []
    for lst in lists:
        for rank, c in enumerate(lst):
            cid = c["chunk_id"]
            if cid not in score:
                score[cid] = 0.0
                order.append(cid)
            score[cid] += 1.0 / (rrf_k + rank + 1)
    order.sort(key=lambda x: -score[x])
    return [{"chunk_id": cid} for cid in order[:k]]


def fuse_weighted(lists, weights, k=TOP_K):
    score, order = {}, []
    for lst, w in zip(lists, weights):
        if not lst:
            continue
        for rank, c in enumerate(lst):
            cid = c["chunk_id"]
            if cid not in score:
                score[cid] = 0.0
                order.append(cid)
            score[cid] += w / (rank + 1)
    order.sort(key=lambda x: -score[x])
    return [{"chunk_id": cid} for cid in order[:k]]


def enrich(hits, by_id):
    """给融合结果补全 text/source（从 chunk 表按 id 取）。"""
    out = []
    for h in hits:
        c = by_id.get(h["chunk_id"], {})
        out.append({"chunk_id": h["chunk_id"], "text": c.get("text", ""), "source": c.get("source", "")})
    return out


# ---------------- 命中与指标 ----------------
def hit_of(hits, cids):
    """hits 必须是检索结果列表（元素为含 chunk_id 的 dict）。"""
    if not cids:
        return False
    for h in hits:
        if isinstance(h, dict) and str(h.get("chunk_id", "")) in cids:
            return True
    return False


def report(name, qs, hits_list):
    """块级 Recall@5：只在可答题上算；拒答题单列；按 batch 分组（core/user）。"""
    def _ans(q):
        return (not q["_cls"]["refuse"]) and (not q["_cls"]["unanswerable"])

    # 总体
    n_ans = sum(1 for q in qs if _ans(q))
    n_hit = sum(1 for q, hits in zip(qs, hits_list) if _ans(q) and hit_of(hits, q["_cls"]["chunk_ids"]))
    n_ref = sum(1 for q in qs if q["_cls"]["refuse"])

    # 按 batch 分组
    groups = {}
    for q, hits in zip(qs, hits_list):
        if not _ans(q):
            continue
        b = q.get("batch", "core")
        g = groups.setdefault(b, [0, 0])
        g[1] += 1
        if hit_of(hits, q["_cls"]["chunk_ids"]):
            g[0] += 1

    group_str = " | ".join(
        f"{b}: {ok}/{n}={ok / max(n, 1) * 100:.1f}%" for b, (ok, n) in sorted(groups.items())
    )
    print(f"{name:24s} 块级Recall@5 = {n_hit}/{n_ans} = {n_hit / max(n_ans, 1) * 100:5.1f}%"
          f" | 拒答组 {n_ref} 题单列")
    if len(groups) > 1:
        print(f"{'':24s} └─ 分组：{group_str}")
    return n_hit, n_ans



# ---------------- LLM 生成 + 判覆盖 ----------------
def _chat(model, messages, timeout=60):
    r = requests.post(
        LLM_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0},
        timeout=timeout,
    )
    if r.status_code >= 400:
        print(f"[LLM错误] {r.status_code} {r.text[:200]}")
        r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def answer_one(question, hits):
    ctx = "\n\n".join(h.get("text", "") for h in hits[:TOP_K])
    prompt = (f"你是专升本政策问答助手。请严格根据以下材料回答问题；材料没有相关内容时，直接回答"
              f"「未找到相关信息」，不要编造。\n\n材料：\n{ctx}\n\n问题：{question}\n\n回答：")
    try:
        return _chat(GEN_MODEL, [{"role": "user", "content": prompt}])
    except Exception as e:
        return f"[生成失败: {e}]"


def judge_coverage(question, keypoints, answer):
    kp_lines = "\n".join(f"- {kp}" for kp in keypoints)
    prompt = (f"你是评测员。逐条判断以下答案是否覆盖每个要点。\n\n问题：{question}\n期望要点：\n{kp_lines}\n"
              f"系统答案：\n{answer}\n\n只输出 JSON 数组（不要其他文字）：\n"
              f'[{{"keypoint": "要点原文", "covered": true, "reason": "简短说明"}}]')
    try:
        raw = _chat(JUDGE_MODEL, [{"role": "user", "content": prompt}])
        parsed = json.loads(raw[raw.find("["): raw.rfind("]") + 1]) if "[" in raw else []
        m = {str(j.get("keypoint", "")).strip(): bool(j.get("covered")) for j in parsed}
        return [(kp, m.get(kp.strip(), False)) for kp in keypoints]
    except Exception as e:
        print(f"[判定失败→本地兜底] {e}")
        return [(kp, False) for kp in keypoints]


def attribute(keypoint, hits, answer):
    joined = "".join(h.get("text", "") for h in hits)
    raw = str(keypoint).strip()
    in_recall = any(g in joined for g in grams(raw))
    in_answer = raw in answer
    if not in_recall:
        return "miss_recall"
    if in_recall and not in_answer:
        return "answer_drift"
    return "prompt_miss"


def eval_answer_quality(qs, hits_list, corpus_norm, config_name):
    """仅对可答题跑；要点分母 = 语料可支撑要点。"""
    print(f"\n=== {config_name} 回答质量轴 ===")
    results = []
    for i, (q, hits) in enumerate(zip(qs, hits_list)):
        if q["_cls"]["refuse"] or q["_cls"]["unanswerable"]:
            continue
        kps = [kp for kp in _get_kp(q) if any(g in corpus_norm for g in grams(kp))]
        if not kps:
            continue
        ans = answer_one(q["question"], hits)
        judged = judge_coverage(q["question"], kps, ans)
        covered = sum(1 for _, c in judged if c)
        rate = covered / len(kps)
        attrs = [attribute(kp, hits, ans) for kp, c in judged if not c]
        results.append(dict(qid=q.get("id", i + 1), question=q["question"], answer=ans,
                            total=len(kps), covered=covered, rate=rate,
                            judgments=[{"keypoint": kp, "covered": c} for kp, c in judged], attrs=attrs))
        print(f"  {q.get('id', i + 1)} | 覆盖 {covered}/{len(kps)} ({rate * 100:.0f}%) | 归因 {attrs}")
    return results


def export_samples(results, out_path):
    flat = []
    for r in results:
        for j in r["judgments"]:
            flat.append(dict(qid=r["qid"], question=r["question"], keypoint=j["keypoint"],
                             covered=j["covered"], answer=r["answer"], attrs=r["attrs"]))
    random.seed(42)
    sample = random.sample(flat, min(SAMPLE_N, len(flat)))
    out_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[抽检] 已导出 {len(sample)} 条样本 -> {out_path}")
    for s in sample:
        print(f"  {s['qid']} | 要点: {s['keypoint'][:36]} | 覆盖: {s['covered']} | 归因: {s['attrs']}")


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全部")
    ap.add_argument("--no-quality", action="store_true", help="跳过回答质量轴（省 LLM 调用）")
    args = ap.parse_args()

    chunks = [c for c in load_json(CHUNKS_PATH) if c.get("text")]
    by_id = {str(c["id"]): c for c in chunks}
    ids = [str(c["id"]) for c in chunks]
    assert len(ids) == len(set(ids)), "chunk 文件有重复 id"
    qs = load_questions(QUESTIONS_PATH)
    if args.limit:
        qs = qs[: args.limit]
    ann = load_annotations()
    missing_ann = [q.get("id") for q in qs if str(q.get("id")) not in ann]
    for q in qs:
        q["_cls"] = classify(q, ann)

    n_ans = sum(1 for q in qs if not q["_cls"]["refuse"] and not q["_cls"]["unanswerable"])
    n_ref = sum(1 for q in qs if q["_cls"]["refuse"])
    n_un = sum(1 for q in qs if q["_cls"]["unanswerable"])
    corpus_norm = norm("".join(c["text"] for c in chunks))

    print(f"[数据] chunks={len(chunks)} (id 唯一) | 题={len(qs)} | 标注缺失={len(missing_ann)}")
    print(f"[分组] 可答={n_ans} | 拒答={n_ref} | 语料缺失={n_un}")
    if missing_ann:
        print(f"[警告] 以下题没有块级标注，将按'不可答'处理: {missing_ann}")

    toks = [list(jieba.cut_for_search(c["text"])) for c in chunks]
    bm = BM25Okapi(toks)
    client = QdrantClient(url=QDRANT_URL)
    info = client.get_collection(COLLECTION)
    print(f"[集合] {COLLECTION}: {info.points_count} 点 / 维度 {info.config.params.vectors.size}")
    assert info.points_count == len(chunks), "Qdrant 点数与 chunk 数不一致，先重建集合"

    bc = run_bm25(chunks, bm, qs, CAND_K)
    vc = run_vec(qs, client, CAND_K)

    print("\n=== 基线 ===")
    report("BM25", qs, [lst[:TOP_K] for lst in bc])
    report("向量(BGE-M3)", qs, [lst[:TOP_K] for lst in vc])

    configs = {
        "RRF": lambda: [fuse_rrf([bc[i], vc[i]]) for i in range(len(qs))],
        "w(BM25=0.7,V=0.3)": lambda: [fuse_weighted([bc[i], vc[i]], [0.7, 0.3]) for i in range(len(qs))],
        "w(BM25=0.5,V=0.5)": lambda: [fuse_weighted([bc[i], vc[i]], [0.5, 0.5]) for i in range(len(qs))],
        "w(BM25=0.3,V=0.7)": lambda: [fuse_weighted([bc[i], vc[i]], [0.3, 0.7]) for i in range(len(qs))],
    }
    fused_raw = {name: fn() for name, fn in configs.items()}

    print("\n=== 混合 ===")
    # 注意：fused_raw[name] 是「每题一个 hit 列表」，enrich 要逐题套
    fused = {name: [enrich(h, by_id) for h in hits] for name, hits in fused_raw.items()}
    best_name, best_hits, best_score = None, None, -1
    for name, hits in fused.items():
        n_hit, n = report(name, qs, hits)
        if n_hit > best_score:
            best_name, best_hits, best_score = name, hits, n_hit

    denom33 = n_ans
    print(f"\n最优配置: {best_name}（块级 Recall@5 = {best_score}/{denom33} = {best_score / max(denom33,1)*100:.1f}%）")
    print("注：权重在测试集上选取，存在选择偏差，结论仅描述性。")

    if not args.no_quality:
        qa = eval_answer_quality(qs, best_hits, corpus_norm, best_name)
        if qa:
            tot = sum(r["total"] for r in qa)
            cov = sum(r["covered"] for r in qa)
            print(f"\n要点覆盖率: {cov}/{tot} = {cov / max(tot, 1) * 100:.1f}%（分母=语料可支撑要点）")
            print(f"归因分布: {dict(Counter(a for r in qa for a in r['attrs']))}")
            out = BASE / f"samples_{best_name.replace('(', '').replace(')', '').replace(',', '').replace('=', '_')}.json"
            export_samples(qa, out)

    print("\n[下一步] 数字写进 metrics.md 第 9/11 节：N/M = P%，附命令")


if __name__ == "__main__":
    main()
