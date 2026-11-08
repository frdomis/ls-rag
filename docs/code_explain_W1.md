## 环节 1：文档收集
- 输入：原始 PDF/MD/TXT 文件
- 输出：docs/raw/ 目录下的文件
- 对应函数：无（手动操作）

## 环节 2：文本提取
- 输入：文件路径
- 输出：纯文本字符串
- 对应函数：extract_text(path)

## 环节 3：文本分块
- 输入：纯文本字符串
- 输出：list[str]（多个 chunk）
- 对应函数：chunk_text(text, size=500, overlap=50)

## 环节 4：向量化
- 输入：list[str]（多个 chunk 文本）
- 输出：list[list[float]]（多个 1024 维向量）
- 对应函数：embed_texts(texts)

## 环节 5：向量存储
- 输入：向量 + payload（text, source）
- 输出：Qdrant collection 中的数据点
- 对应函数：store.py 中的 upsert 逻辑

## 环节 6：检索
- 输入：用户问题（字符串）
- 输出：top_k 个最相关的 chunk（含 text, source, score）
- 对应函数：search(query, top_k=5)

## 环节 7：生成回答
- 输入：用户问题 + 检索到的 chunks
- 输出：LLM 生成的回答文本
- 对应函数：build_prompt() + answer()