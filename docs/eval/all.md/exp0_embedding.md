# exp0_embedding：本地轻量模型 vs BGE-M3 检索对比

> 日期：2026-09-22 ｜ 实验编号：exp0 ｜ 分支/commit：待补

## 一、结论先行

**BGE-M3 明显胜出**（Recall@5 约 0.78 vs 本地模型约 0.05）。
本地 `paraphrase-multilingual-MiniLM-L12-v2` 在本任务上基本不可用，
保留它仅作为基线对照。

同时明确：**检索质量主指标只用 Recall@5，不用“生成答案与期望答案的
embedding 相似度”当主指标**（理由见 metrics.md）。

## 二、实验设置（固定项，后续改动要注明）

| 项目 | 取值 |
|---|---|
| 语料 | `docs/raw/` 下 3 个 PDF：实施办法(258KB)、录取控制分数(550KB)、院校章程(72KB) |
| 切块 | 定长 600 字、滑窗步长 450（exp0 固定，未调优） |
| 入库 | 每次换模型先 `recreate_collection`，避免新旧向量混在同一集合 |
| 题数 | 20 题，其中 Q12/Q18 为拒答/元数据缺失题，不计入分母，有效 18 题 |
| 指标 | Recall@5（top5 是否包含正确出处文件） |
| 环境 | 本机 CPU，Qdrant `ls_rag_docs`，端口 6333 |

复现：
```bash
python -m app._init_
python -m docs.eval.eval_recall