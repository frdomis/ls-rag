# 河南专升本政策问答 RAG 系统

面向河南专升本考生的政策问答机器人。基于混合检索 + 元数据过滤 + 可选 rerank，回答报名、考试、志愿、录取、免试等政策类问题。

> **项目状态**：W6 末（2026-11-01）冻结 RAG 基础，进入生产化阶段
> **最新更新**：2026-10-30

---

## 一、核心指标

评估基于 **41 道题**的块级标注集，指标为**块级 recall@5**（检索结果 top5 是否包含标注的答案块）。

### core 组（按语料设计的题，33 道可答）

| 配置 | recall@5 | 说明 |
|---|---|---|
| **向量 + 元数据过滤（当前默认）** | **24/33 = 72.7%** | fast 模式 |
| **向量 + 过滤 + rerank** | **26/33 = 78.8%** | high 模式 |
| 向量单路（无过滤，原始基线） | 19/33 = 57.6% | 起点 |
| BM25 + 过滤 | 22/33 = 66.7% | — |
| BM25 单路 | 13/33 = 39.4% | — |
| RRF 混合 | 21/33 = 63.6% | 无过滤 |

**命令**：`python docs/eval/eval_hybrid_final.py --no-quality` / `python docs/eval/eval_metadata_filter.py`

### user 组（真实考生提问，1 道可答）

| 配置 | recall@5 |
|---|---|
| 向量 | 1/1 = 100% |
| BM25 | **0/1 = 0%** |

> ⚠️ **样本量仅 1 条，不具统计意义**。
> 真实提问 20 条中：能答 1 条（5%）／部分可答 4 条（20%）／不能答 15 条（75%）。
> 详见 [`docs/eval/user_questions_verdict.md`](docs/eval/user_questions_verdict.md)

### 拒答能力（19 条拒答测试集）

| 分组 | 结果 |
|---|---|
| 超范围（14 条） | **14/14 = 100%** |
| 语料缺失（5 条） | 4/5 = 80% |
| **合计** | **18/19 = 94.7%** |

**命令**：`python docs/eval/eval_refuse.py`

---

## 二、架构

```
用户问题
   ↓
[元数据过滤推断]  问题含"分数线"→ doc_type=score；含"2026"→ year=2026
   ↓
fast 模式 ──→ Qdrant 向量检索 top5 ──────────────┐
                                                │
high 模式 ──→ BM25 top20 + 向量 top20 ──┐        │
                                        ↓        │
                                  BGE-rerank top5│
                                                 ↓
                                          LLM 生成答案
                                                 ↓
                                    答案 + 引用来源 + 拒答判断
```

**技术栈**：
- 向量库：Qdrant（`chunks` 集合，142 点，BGE-M3 1024 维）
- Embedding：硅基流动 `BAAI/bge-m3`（免费）
- Rerank：硅基流动 `BAAI/bge-reranker-v2-m3`（¥0.7/M tokens）
- 生成：硅基流动 `deepseek-ai/DeepSeek-V3`
- 后端：FastAPI（W6D5-D6 实现）

---

## 三、两种模式

| | **fast**（默认） | **high** |
|---|---|---|
| 链路 | 过滤 + 向量 top5 + 生成 | 过滤 + 多路 top20 + rerank + 生成 |
| 检索延迟 | ~220ms | ~1.2s（rerank P95 870ms） |
| 千次成本 | **¥9.86** | **¥19.82** |
| core recall@5 | 24/33 = 72.7% | 26/33 = 78.8% |
| 适用场景 | 对话式问答、移动端、高频调用 | 政策查询（要准确引用）、批量离线生成、演示 |

**为什么默认 fast**：延迟是硬约束（220ms vs 1.2s 差 5.5 倍），而 +2 题的精度提升（6.1pp）不构成"能用/不能用"的分界。

---

## 四、已知缺陷

> 完整清单见 [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)

### 语料覆盖边界（最重要）

| 类别 | 真实提问占比 | 系统能答吗 |
|---|---|---|
| 政策/资格（报名、考试、投档、免试） | ~40% | ✅ 部分能答 |
| 备考策略（怎么学、押题有用吗） | ~25% | ❌ 不属于本产品 |
| 事务办理（酒店、准考证、档案） | ~20% | ❌ 不属于本产品 |
| 院校体验（课多不多、值不值） | ~15% | ❌ 不属于本产品 |

**20 条真实考生提问中，75% 超出知识库范围。**

### 语料缺失

| 缺什么 | 影响 |
|---|---|
| 院校投档线（历年） | "230 分能上公办吗"无法回答 |
| 总分一分一段表 | "考多少分能上本科"无法回答（现有的是专业分段） |
| 跨专业报考材料清单 | — |
| 退档/调剂细则 | — |
| 联合培养政策 | — |

### 技术局限

| 局限 | 数值/说明 |
|---|---|
| **块级命中 ≠ 答对了** | 检索到答案块不代表生成答案正确 |
| 覆盖率受 LLM 表达影响 | 出现过"recall 涨但覆盖率不涨"的反常，仅作辅助指标 |
| 元数据过滤误路由 | Q39（政策文件链接）因"政策"关键词误路由，影响 1/33 |
| 数据质量瑕疵 | 表格残余重复行、噪声率 4.3%、一分一段表压成单行 |
| rerank 延迟高 | P95 870ms，完整 high 链路约 1.2s |
| 单机 Qdrant | 无副本无备份 |

---

## 五、快速开始

```bash
# 依赖
pip install -r requirements.txt

# 环境变量（.env）
SILICONFLOW_API_KEY=sk-xxx
QDRANT_URL=http://localhost:6333
COLLECTION=chunks

# 启动 Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 重建索引（142 块）
python rebuild_qdrant.py

# 跑评估
python docs/eval/eval_hybrid_final.py --no-quality   # core/user 分组
python docs/eval/eval_refuse.py                       # 拒答准确率
python docs/eval/eval_metadata_filter.py              # 过滤对比
```

---

## 六、目录结构

```
ls-rag/
├── docs/
│   ├── api.md                    API 契约（3 个端点）
│   ├── KNOWN_ISSUES.md           已知问题清单（4 类）
│   ├── chunks/
│   │   └── chunks_fixed_v2.jsonl 语料（142 块）
│   ├── raw/                      原始语料（PDF/JSON 录入源）
│   └── eval/
│       ├── eval_hybrid_final.py      唯一评估入口（core/user 分组）
│       ├── eval_metadata_filter.py   元数据过滤对比
│       ├── eval_rerank_attribution.py rerank + 漏检归因
│       ├── eval_query_rewrite.py     query 改写对比
│       ├── eval_refuse.py            拒答评估
│       ├── hybrid_compare.py         三路对比 + ID 校验
│       ├── ingest_incremental.py     增量入库
│       ├── rebuild_qdrant.py         重建索引
│       ├── questions_v2.jsonl        题表（40 core + 1 user）
│       ├── questions_annotated.jsonl 块级标注（41 条）
│       ├── refuse_set.jsonl          拒答集（16 条）
│       └── user_questions_verdict.md 真实提问判定
├── 成本核算.md
└── README.md
```

---

## 七、方法论

这个项目最值得讲的三件事，都不是"指标涨了"：

### 1. 三个 Qdrant 集合导致数字漂移

早期排查发现同时存在 `chunks`（139点）、`ls_rag_docs_clean`（2403点）、`ls_rag_docs_clean_200`（200点）三个集合，三个脚本各指一个。**之前所有"对比实验"比的其实是三个不同的知识库。**

### 2. id 空间不一致导致 RRF 空转

```python
# 错误代码
chunks[idx].get("chunk_id", idx)   # 字段名是 id，取不到就回落成数组下标
# → BM25 返回整数 (17,16,89)，向量返回 hex (e7061bb895c2)
# → RRF 在融合两个不相交的 id 集合，同一个块被当成两个实体
```

**静默兜底比崩溃更危险。**

### 3. 评估集有系统性覆盖偏差

33 道 core 题全是按语料编的，天然在覆盖范围内 → 72.7% 是**高估**的。
采集 20 条真实提问后测出：**只有 1 条（5%）能被完整回答，75% 超出知识库范围。**

**所以产品定位收窄为"政策问答机器人"，并建立了拒答指标（94.7%）——"承认不知道"和"答对"同等重要。**

---

## 八、下一步（W7 生产化）

- HK 轻量服务器部署（免备案，40-60 元/月）
- 可用性测试第二轮（3-5 人真实使用，用真实提问替代 user 组统计）
- 公开产出：写一篇「我把 RAG 召回从 57.6% 做到 78.8%，以及 75% 真实问题答不了这件事」

---

## 九、许可

个人学习项目，数据来源为河南省教育考试院及公开招生章程。
