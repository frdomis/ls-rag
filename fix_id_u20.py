# -*- coding: utf-8 -*-
"""
fix_id_u20.py —— 修复 U20/Q41 id 不一致（统一为 Q41）
用法（在项目根执行）：
    python fix_id_u20.py
"""
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
ANN = BASE / "docs" / "eval" / "questions_annotated.jsonl"
QF = BASE / "docs" / "eval" / "questions_v2.jsonl"

OLD, NEW = "U20", "Q41"

if not ANN.exists():
    raise SystemExit(f"[错误] 找不到 {ANN}")

anns = [json.loads(l) for l in open(ANN, encoding="utf-8") if l.strip()]
changed = 0
for a in anns:
    if a["id"] == OLD:
        a["id"] = NEW
        changed += 1

if changed:
    shutil.copy2(ANN, ANN.with_suffix(".jsonl.bak"))
    with open(ANN, "w", encoding="utf-8") as f:
        for a in anns:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"[标注] {OLD} → {NEW}，改动 {changed} 条（已备份 .jsonl.bak）")
else:
    print(f"[标注] 没有 {OLD}，可能已修过")

qs = [json.loads(l) for l in open(QF, encoding="utf-8") if l.strip()]
q_ids = {r["id"] for r in qs}
a_ids = {a["id"] for a in anns}

print(f"\n[题表] {len(qs)} 条 | [标注] {len(anns)} 条")
print(f"[差集] 题表有标注无: {sorted(q_ids - a_ids) or '无 ✅'}")
print(f"[差集] 标注有题表无: {sorted(a_ids - q_ids) or '无 ✅'}")

users = [r for r in qs if r.get("batch") == "user"]
print(f"\n[user 组] {len(users)} 条: {[r['id'] for r in users]}")
for u in users:
    ann = next((a for a in anns if a["id"] == u["id"]), None)
    print(f"  {u['id']}: 标注块 = {ann.get('chunk_ids') if ann else '❌ 缺失'}")

core = [r for r in qs if r.get("batch") == "core"]
print(f"[core 组] {len(core)} 条")
print("\n[下一步] python docs/eval/eval_hybrid_final.py --no-quality")
