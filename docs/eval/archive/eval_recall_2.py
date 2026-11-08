# docs/eval/eval_recall2.py —— 每题输出 recall@5 / 命中出处 / 要点覆盖 / 判定
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.embed import embed

# ============ 配置 ============
COL = "ls_rag_docs"
TOPK = 5
COVER_THRESHOLD = 0.8

FIELD_Q    = "question"
FIELD_KEYS = "expected_keypoints"
FIELD_SRC  = "expected_source"
FIELD_TYPE = "type"

ROOT     = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "docs" / "eval"
ALL      = EVAL_DIR / "questions.all.jsonl"


def load_questions(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        q = json.loads(line)
        q["_src"] = f"{path.name}:{i}"
        rows.append(q)
    return rows


def _parse_sources(q):
    v = q.get(FIELD_SRC)
    if not v:
        return None
    s = str(v).strip()
    # ★ 拒答/超范围标记：不计入 Recall 分母，改走误召率统计
    if s.startswith("—") or s.startswith("-") or "拒答" in s or "超出覆盖范围" in s:
        return None
    if isinstance(v, list):
        s = ",".join(str(x) for x in v)
    return {x.strip() for x in s.split(",") if x.strip()}


def normalize(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s)


def cover(hit_texts: list[str], keys: list[str]) -> tuple[int, int, list[str]]:
    if not keys:
        return 0, 0, []
    body = "\n".join(normalize(t) for t in hit_texts)
    hit, miss = [], []
    for k in keys:
        nk = normalize(k)
        (hit if nk and nk in body else miss).append(k)
    return len(hit), len(keys), miss


def judge(q, n_hit_src, labeled_src: bool, cov: int, total: int) -> str:
    if q.get(FIELD_TYPE) == "no_answer":
        return "拒答"
    if not labeled_src:
        return "未标注"
    if n_hit_src == 0:
        return "错"
    if total == 0:
        return "偏"
    return "对" if cov / total >= COVER_THRESHOLD else "偏"


def get_collection_stats(client: QdrantClient) -> dict:
    """v2 兼容：scroll 返回对象而非 (points, offset) 元组，需分页取完"""
    cnt = client.count(COL, exact=True).count
    dim, next_offset = None, None
    src_counter = Counter()
    while True:
        res = client.scroll(COL, limit=500, with_payload=["source"], offset=next_offset)
        pts = res.points if hasattr(res, "points") else (res[0] if isinstance(res, tuple) else res)
        for p in pts:
            src_counter[p.payload.get("source", "<未知>")] += 1
            if dim is None:
                vec = p.vector
                if isinstance(vec, dict):          # 多向量命名场景
                    vec = next(iter(vec.values()), None)
                if isinstance(vec, list):
                    dim = len(vec)
        next_offset = (res.next_page_offset
                       if hasattr(res, "next_page_offset")
                       else (res[1] if isinstance(res, tuple) else None))
        if not next_offset:
            break
    return {"count": cnt, "dim": dim, "sources": dict(src_counter.most_common())}


def retrieve(client, col, vec, k):
    """v2 兼容：优先 query_points，回退 search"""
    try:
        resp = client.query_points(col, query=vec, limit=k)
        return resp.points if hasattr(resp, "points") else list(resp)
    except Exception:
        return client.search(col, query_vector=vec, limit=k)


def run_eval():
    client = QdrantClient(url="http://localhost:6333")
    if not client.collection_exists(COL):
        raise SystemExit(f"❌ 集合 {COL} 不存在，先 python -m app._init_")
    if not ALL.exists():
        raise SystemExit(f"❌ {ALL} 不存在，先 python docs/data/merge_questions.py")

    qs = load_questions(ALL)
    stats = get_collection_stats(client)
    print(f"集合 {COL}: {stats['count']} chunks / {stats['dim']} 维")
    print(f"共 {len(qs)} 题\n")

    rows, timers = [], []
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        text = str(q.get(FIELD_Q, ""))
        hit_texts, hit_srcs, err = [], [], None
        try:
            vec = embed([text])[0]
            # 防御：embedding 返回标量/维度不对时立刻报错，避免静默入库污染基线
            if not isinstance(vec, list):
                vec = [float(vec)]
            if stats["dim"] and len(vec) != stats["dim"]:
                raise ValueError(f"向量维数不匹配: 期望{stats['dim']}, 得到{len(vec)}")
            t1 = time.time()
            hits = retrieve(client, COL, vec, TOPK)
            timers.append(time.time() - t1)
            hit_texts = [h.payload.get("text", "") for h in hits]
            hit_srcs = [h.payload.get("source", "<未知>") for h in hits]
        except Exception as e:
            err = str(e)[:60]
            timers.append(0.0)

        expected = _parse_sources(q)
        labeled = expected is not None
        n_hit = len(set(hit_srcs) & expected) if labeled else -1
        x, y, miss = cover(hit_texts, q.get(FIELD_KEYS, []))
        rec = n_hit / len(expected) if labeled else None
        j = "ERROR" if err else judge(q, n_hit, labeled, x, y)

        rows.append({
            "id": i, "q": text[:32], "src_line": q["_src"],
            "recall@5": f"{n_hit}/{len(expected)}" if labeled else "-",
            "rec_raw": rec,
            "cover": f"{x}/{y}" if y else "-",
            "miss": miss, "judgment": j,
            "top1": hit_srcs[0] if hit_srcs else "-",
            "latency_ms": round(timers[-1] * 1000), "error": err,
            "oral": bool(q.get("oral")),
        })
        mark = {"对": "✅", "偏": "⚠️", "错": "❌", "拒答": "🚫", "未标注": "❓", "ERROR": "💥"}.get(j, "?")
        err_note = f" ERROR:{rows[-1]['error']}" if rows[-1]['error'] else ""
        print(f"{i:>3} {mark} {text[:30]:<30} recall={rows[-1]['recall@5']:>6} cover={rows[-1]['cover']:<6} {j}{err_note}")

    write_report(rows, timers, time.time() - t0, len(qs), stats)


def write_report(rows, timers, total_sec, n, stats):
    cmd = "python " + " ".join(sys.argv)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "?"

    jud = Counter(r["judgment"] for r in rows)

    # ★ 口径修正：Recall 分母只含「有标注且非拒答」的题；拒答单独算误召率
    ann = [r for r in rows if r["rec_raw"] is not None and r["judgment"] != "拒答"]
    no_ans = [r for r in rows if r["judgment"] == "拒答"]
    avg_rec = sum(r["rec_raw"] for r in ann) / len(ann) * 100 if ann else 0.0
    full_rec = sum(1 for r in ann if r["rec_raw"] >= 1.0) / len(ann) * 100 if ann else 0.0
    false_recall = sum(1 for r in no_ans if (r["rec_raw"] or 0) > 0) / len(no_ans) * 100 if no_ans else 0.0

    lat_avg = sum(timers) / len(timers) * 1000 if timers else 0.0
    oral_n = sum(1 for r in rows if r["oral"])

    md = []
    md.append("# baseline_W3：新题 + 新语料重跑基线（真相日）")
    md.append("")
    md.append(f"**生成命令**：`{cmd}`")
    md.append(f"**git sha**：`{sha}`　**集合**：`{COL}`　**TOPK**：{TOPK}　**要点命中阈值**：{COVER_THRESHOLD}")
    md.append("")
    md.append("## 一、集合指纹")
    md.append("")
    md.append(f"- chunks：{stats['count']}　维度：{stats['dim']}")
    src_top = ", ".join(f"{k}={v}" for k, v in list(stats["sources"].items())[:6])
    md.append(f"- source 分布：{src_top}")
    md.append("")
    md.append("## 二、总览")
    md.append("")
    md.append("| 指标 | 值 |")
    md.append("|---|---|")
    md.append(f"| 题目数 | {n}（口语化 {oral_n}）|")
    md.append(f"| 有 sources 标注 | {len([r for r in rows if r['rec_raw'] is not None])} / {n} |")
    md.append(f"| **平均 Recall@5** | **{avg_rec:.1f}%** |")
    md.append(f"| 全中率 | {full_rec:.1f}% |")
    md.append(f"| 对 / 偏 / 错 | {jud.get('对',0)} / {jud.get('偏',0)} / {jud.get('错',0)} |")
    md.append(f"| 未标注 / 拒答 / ERROR | {jud.get('未标注',0)} / {jud.get('拒答',0)} / {jud.get('ERROR',0)} |")
    md.append(f"| 拒答题误召率 | {false_recall:.1f}%（越低越好）|")
    md.append(f"| 延迟 / 耗时 | {lat_avg:.0f} ms / {total_sec:.1f} s |")
    md.append("")
    md.append("> 📐 **指标口径**：Recall@5 = 每题 `(命中期望source数 / 期望source数)` 的平均值；")
    md.append("> 分母剔除无 `expected_source` 的题；`拒答`题不参与 Recall，单独计误召率。")
    md.append("")
    md.append("> ⚠️ W2 的 95% 因集合维度=1 已作废，本次是首个有效基线。本数字为**纯向量检索上限**，未加 BM25 / rerank，不与 W2 横向对比。")
    md.append("")
    md.append("## 三、逐题明细")
    md.append("")
    md.append("| # | 题目 | 出处行 | recall@5 | 要点覆盖 | 判定 | top1 | 延迟 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['id']} | {r['q']} | {r['src_line']} | {r['recall@5']} | {r['cover']} | {r['judgment']} | {r['top1']} | {r['latency_ms']}ms |")

    md.append("")
    md.append("## 四、错/偏题归因（人工补）")
    md.append("")
    md.append("| # | 题目 | 类型 | 原因 | 修复方向 |")
    md.append("|---|---|---|---|---|")
    md.append("| | | 切块断裂/表述不匹配/缺语料/标注有误 | | |")
    md.append("")
    md.append("## 五、局限")
    md.append("")
    md.append("1. 要点覆盖 = 关键词包含，同义改写被低估 → W4 换 LLM-as-judge。")
    md.append("2. 新题只有 answer_hint，无 keypoints，cover 列为 `-`，这类题只看 recall。")
    md.append("3. Recall 是文件级，3 份 PDF 时天然偏高。")
    md.append("4. 只评检索不评生成；跨文档推理题当前语料下无解。")
    md.append("")
    md.append("## 六、复现步骤")
    md.append("")
    md.append("```powershell")
    md.append("cd <项目根目录>")
    md.append("python docs/data/merge_questions.py")
    md.append("python -m app._init_")
    md.append("python docs/eval/eval_recall.py")
    md.append("```")
    md.append("换人执行，表格应一致（延迟 ±20%，集合指纹必须一致）。")

    out = EVAL_DIR / "baseline_W3.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")

    # ★ 调试输出：分母构成，便于手算验算
    print(f"\n分母构成: 总{n} = 有标注参与Recall{len(ann)} + 拒答{len(no_ans)} + 未标注{jud.get('未标注',0)} + ERROR{jud.get('ERROR',0)}")
    print(f"Recall 分子(sum rec_raw) = {sum(r['rec_raw'] for r in ann):.2f}")
    print(f"汇总: 对{jud.get('对',0)} 偏{jud.get('偏',0)} 错{jud.get('错',0)} "
          f"未标注{jud.get('未标注',0)} ERROR{jud.get('ERROR',0)} 平均Recall@5={avg_rec:.1f}%")
    print(f"✅ 已写入 {out}")


if __name__ == "__main__":
    run_eval()