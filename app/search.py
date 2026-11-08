from qdrant_client import QdrantClient
from app.embed import embed_text
client=QdrantClient(url="http://localhost:6333")
COLLECTION_NAME="chunks"
def search(query:str,top_k:int= 10) -> list[dict]:
    vector = embed_text(query)  
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True
    )
    return [{
        "text": hit.payload.get("text"),
        "source": hit.payload.get("source"),
        "chunk_id": hit.payload.get("chunk_id"),   
        "score": hit.score,
    } for hit in response.points]
    
    ]
if __name__=="__main__":
    query="专升本时间是什么"
    print(f"正在搜索:{query}")
    hits=search(query)
    if not hits:
        print("没有找到")
    else:
        for i, hit in enumerate(hits, 1):
            print(f"--- 结果 {i} (相似度得分: {hit['score']:.4f}) ---")
            print(f"来源: {hit['source']}")
            print(f"内容: {hit['text']}\n")