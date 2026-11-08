# appembed.py —— 嵌入：导入模块->调用API->写请求->排序，提取向量
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SILICONFLOW_API_KEY")
print('API_KEY:', repr(API_KEY))
BASE_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"


def embed(texts: list[str]) -> list[list[float]]:
    """批量嵌入。入参：字符串列表；返回：list[list[float]]，每个子列表长度为向量维度。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "input": texts
    }
    response = requests.post(BASE_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()["data"]
    # 兼容：部分 API 实现返回单个对象而非列表时，包成列表
    if not isinstance(data, list):
        data = [data]
    sorted_data = sorted(data, key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


def embed_text(text: str) -> list[float]:
    """单条文本的便捷入口，等价于 embed([text])[0]。"""
    return embed([text])[0]


if __name__ == "__main__":
    vectors = embed(["专升本报名条件是什么？"])
    print(f"向量维度: {len(vectors[0])}")
    print(f"前5个值: {vectors[0][:5]}")