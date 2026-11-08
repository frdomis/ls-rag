# -*- coding: utf-8 -*-
"""
patch_eval_by_batch.py —— 给 eval_hybrid_final.py 加「按 batch 分组统计」
用法（在项目根执行）：
    python patch_eval_by_batch.py

改动内容：
    1) report() 增加分组输出：core（原 40 题）/ user（真实提问）
    2) 保持原有总体输出不变（向后兼容）
    3) 自动备份原文件
"""
import re
import shutil
import sys
from pathlib import Path

TARGET = Path(r"D:\vscode\ls-rag\docs\eval\eval_hybrid_final.py")

NEW_REPORT = '''def report(name, qs, hits_list):
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

'''


def main():
    if not TARGET.exists():
        sys.exit(f"[错误] 找不到 {TARGET}")

    text = TARGET.read_text(encoding="utf-8")
    if "按 batch 分组" in text:
        print("[跳过] 已经是分组版，无需重复打补丁")
        return

    # 匹配旧的 report 函数（从 def report 到 return n_hit, n_ans）
    pat = re.compile(r"def report\(name, qs, hits_list\):.*?return n_hit, n_ans\n", re.S)
    if not pat.search(text):
        sys.exit("[错误] 未匹配到旧的 report 函数，请手动检查")

    bak = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, bak)
    print(f"[备份] {bak.name}")

    new_text = pat.sub(NEW_REPORT, text, count=1)

    # 确保 _cls 里带上 batch 字段（如果 classify 没带）
    if 'q["_cls"]["batch"]' not in new_text and 'q.get("batch"' in new_text:
        print("[提示] batch 字段从 q 顶层读取（q.get('batch')），无需改 _cls")

    TARGET.write_text(new_text, encoding="utf-8")
    print(f"[完成] 已更新 {TARGET.name}")
    print("[验证] 运行：python docs/eval/eval_hybrid_final.py --no-quality")


if __name__ == "__main__":
    main()
