docs/eval/eval_io.md — eval.py 输入输出说明
本项目评估脚本的唯一输入输出文档。有疑问以本文为准。最后更新：2026-10-13

1. 输入
1.1 questions.all2.jsonl
位置：docs/eval/questions.all2.jsonl
每行一条 JSON，无数组包裹。每条记录结构：
字段	类型	必填	说明
question	string	是	评测题目文本
expected_keypoints	string[]	否	期望关键词列表，非空则该题计入分母
示例：
{"question": "2026年河南专升本报名截止时间是什么时候？", "expected_keypoints": ["11月3日", "11月17日"]}
{"question": "河南工业大学专升本学费多少？", "expected_keypoints": ["12000"]}
{"question": "随便问问不在资料范围内的事"}
最后一条没有 expected_keypoints，不计入分母，结果仍写入 report.json 供人工查看。
1.2 Qdrant 向量库
eval.py 不直接连 Qdrant，它调用 from app.rag import answer，由 app 层负责检索。你只需要确保：
•Qdrant 服务在运行（默认 http://localhost:6333）
•集合 ls_rag_docs 已入库文档
1.3 代码依赖
依赖	说明
app.rag.answer(question)	核心调用，返回 dict{answer, chunks_text} 或纯文本
Python 标准库：json, time, pathlib, argparse	无需 pip install

2. 输出
2.1 终端输出
静默模式（默认）：仅打印每题进度 + 末尾汇总块
===== 评估汇总 =====
命中 3 / 分母 5 = 60.0%
拒答数: 1
总耗时: 12.3s  平均: 2.5s/题
详细结果 -> docs/eval/report.json
详细模式（加 --verbose）：每题额外打印命中/拒答/异常详情
2.2 report.json
位置：docs/eval/report.json（每次运行覆盖）
顶层结构：
{
  "summary": {
    "total": 21,
    "total_hit": 3,
    "total_expect": 5,
    "hit_rate": "60.0%",
    "reject_count": 1,
    "elapsed": "12.3s",
    "avg_per_q": "2.5s"
  },
  "results": [
    {
      "question": "...",
      "expected_keypoints": ["11月3日", "11月17日"],
      "answer": "艺术类报名截止11月3日...",
      "chunks": [{"source": "...", "text": "...", "score": 0}],
      "hit_count": 2,
      "expect_count": 2,
      "keypoint_details": {"11月3日": true, "11月17日": true}
    }
  ]
}

3. 分母规则
场景	是否计入分母	说明
有 expected_keypoints 且非空	是	关键词逐条子串匹配，命中数累加到 total_hit
无 expected_keypoints 或为空列表	否	跳过命中统计，结果仍写入 report.json
回答触发拒答关键词	否	计入 reject_count，不计入 total_hit
分母 = 所有有 keypoints 的题目中，各题 keypoints 数量的总和
举例：3 道题，keypoints 分别为 [A,B] [C] [D,E] -> 分母 = 2+1+2 = 5

4. 拒答判定
回答中包含以下任一子串即判定为拒答：
关键词
未找到相关信息
没有相关信息
无法回答
资料中未找到
资料中暂无
暂未找到
抱歉
参考资料中未
提供的参考资料中未
拒答题：不统计命中，reject_count +1，answer 原文仍保存到 report.json。

5. 一键复跑
5.1 前置条件
1.Qdrant 服务运行中
2.docs/eval/questions.jsonl 已准备好
3.项目根目录包含 app/ 包（含 rag.py）
5.2 运行命令
# 静默模式（CI/日常跑分用）
python docs/eval/eval.py

# 详细模式（调试用，看每题详情）
python docs/eval/eval.py --verbose
5.3 复现保证
换电脑 / 换人跑，只要以下三者不变，report.json 内容完全一致：
1.questions.jsonl 内容不变
2.Qdrant 集合 ls_rag_docs 数据不变
3.embedding 模型不变（SiliconFlow 同一模型）

6. 对比实验
不同配置（如 BM25 vs 向量）分开跑，分别记录到不同的 report 文件：
# 配置A
python docs/eval/eval.py
cp docs/eval/report.json docs/eval/report_bm25.json

# 配置B（改配置后）
python docs/eval/eval.py
cp docs/eval/report.json docs/eval/report_vector.json
对比结果写入 metrics.md 末尾的「对比实验记录」表。

7. 相关文件索引
文件	用途
docs/eval/eval.py	评估脚本（唯一入口）
docs/eval/questions.all2.jsonl	评测题目
docs/eval/report.json	评估结果（每次覆盖）
docs/eval/metrics.md	评估口径定义
docs/eval/verify_chunks.py	BM25 vs Qdrant 一致性验证
docs/eval/eval_io.md	本文
