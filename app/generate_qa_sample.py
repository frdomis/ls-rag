import os
from datetime import datetime
from app.rag import answer

# 确保 docs 目录存在
os.makedirs("docs", exist_ok=True)

# 测试问题
question = "河南省专升本报名时间是什么时候？"

# 运行问答
result = answer(question)

# 生成文档内容
doc_content = f"""# QA Sample 1

## 问题

{question}

## 检索到的 Chunks

{result['chunks_info'] if 'chunks_info' in result else '（未检索到相关内容）'}

## 最终回答

{result['answer'] if isinstance(result, dict) else result}

---
*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

# 写入文件
output_path = "docs/qa_sample_1.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(doc_content)

print(f"✅ 已生成：{output_path}")