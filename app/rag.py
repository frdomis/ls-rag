import os
from openai import OpenAI
from app.search import search
client=OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
def build_prompt(question:str,chunks2:list[dict]) -> list[dict]:
    system_message = {
    "role": "system",
    "content": (
        "你是一个专业的河南专升本政策智能助手。请严格根据我提供的参考资料回答用户问题。\n\n"
        "回答核心原则：\n"
        "1. **依据资料**：所有回答必须基于参考资料，严禁编造信息。\n"
        "2. **拒绝话术**：如果资料中确实完全没有相关信息，请回答“资料中未找到相关信息”。\n\n"
        "回答进阶要求（重要）：\n"
        "1. **信息整合**：如果答案分散在文档的不同段落（例如：考试科目分散在艺术、体育、文化课不同章节），请务必将它们**拼凑完整**后回答，不要只回答一部分。\n"
        "2. **语义推断**：如果用户提问的词汇（如“补录”）与文档词汇（如“征集志愿”）不同但含义一致，请**视为同一概念**进行回答。\n"
        "3. **部分匹配**：如果资料只包含部分答案（例如：只提到了艺术体育科目，未提文化课），请**如实回答已知部分**，并说明“资料未提及剩余部分”，**绝对不要**直接说“未找到相关信息”。\n\n"
        "格式要求：\n"
        "- 回答要清晰、分点陈述。\n"
        "- 关键信息请标注来源文件名。"
    )
}
    content_text=""
    for i,chunk in enumerate(chunks2,1):
        source=chunk.get("source","未知来源")
        text =chunk.get("text","")
        content_text += f"资料来源：{source}\n{text}\n"
    user_content=f"参考资料如下：\n\n{content_text}\n用户问题:{question}"
    user_message={
        "role":"user",
        "content":user_content
        }
    return [system_message,user_message]
def answer(question: str) -> dict:
    chunks2 = search(question, top_k=5)

    # 👇 添加这一行调试
    print(f"DEBUG: 检索到了 {len(chunks2)} 条资料")
    if chunks2:
        print(f"DEBUG: 第一条资料内容预览: {chunks2[0].get('text')[:50]}...")

    # 👇 构造 Chunks 的展示文本
    chunks_info = []
    for i, chunk in enumerate(chunks2):
        source = chunk.get("source", "未知来源")
        text = chunk.get("text", "")
        chunks_info.append(f"{i+1}. [来源: {source}] {text}")
    
    chunks_text = "\n".join(chunks_info) if chunks_info else "（未检索到相关资料）"

    messages = build_prompt(question, chunks2)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3
    )
    llm_answer = response.choices[0].message.content

    # 👇 重点：返回一个字典，包含回答和检索到的 Chunks
    return {
        "answer": llm_answer,
        "chunks_text": chunks_text
    }
if __name__ == "__main__":
    # W2D6 基线测试模式：逐条输入，按回车出结果
    print("=" * 50)
    print("W2D6 基线测试 | 输入问题后回车，输入 'quit' 退出")
    print("=" * 50)
    
    while True:
        q = input("\n❓ 问题: ").strip()
        if q.lower() in ('quit', 'exit', 'q'):
            break
        if not q:
            continue
            
        print(f"\n🔍 检索中...")
        result = answer(q)
        
        # 打印完整 chunks 信息（用于判断 Top1 是否正确）
        print(f"\n📚 检索到的资料:\n{result['chunks_text']}")
        print(f"\n💬 回答:\n{result['answer']}")
        print("-" * 50)