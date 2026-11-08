post层：
一（chat）：对话问答接口。接收用户问题，返回答案及引用来源
请求字段
{
  "question": "河南专升本计算机专业考什么科目？",   问题
  "mode": "high",    可选 回答模式high/fast
  "session_id": "sess_abc123"    对话ID
}
响应字段：
{
  "answer": "。",  模型回答
  "sources": [     答案来源
    {
      "doc_id": "doc_0042",
      "title": "2025年河南省专升本招生政策说明",
      "score": 0.92,
      "snippet": "计算机科学与技术专业考试科目为公共英语和专业综合..."
    }
  "latency_ms": , 端到端响应时间
  "mode_used": "string"   实际使用模式
}
状态码：
400  INVALID_QUESTION  问题为空或文本长度过长
400  INVALID_MODE      mode值不和法
429  RATE_LIMITED      请求频率超频
500  INTERNAL_ERROR    服务端异常
二(search)：纯检索接口。返回与查询相关的文档片段，不经过大模型生成。
请求字段
{
  "query": "string", 检索关键词或句子，最大长度500字符
  "top_k": integer    选填 返回结果数量
  "filters":object    选填 过滤条件
}
响应字段：
{
  "hits": [] 检索结果列表
  "total": integer 命中文档总数
  "latency_ms": number 检索耗时
}
状态码：
400   INVALID_QUERY 查询为空或超过长度时间限制
400   INVALID_TOP_K     top_k超出允许范围
500   INTERNAL_RROR    服务端异常

GET/healthy :健康检查接口。用于监控服务状态和知识库就绪情况。
无请参数
响应字段
{
  "status": "",服务状态："healthy" / "degraded" / "unhealthy"
  "chunks_count": ,知识库中的文本数量
  "collection": "",当前使用向量向量名称
  "rrf_enabled": ,rrf重排是否开启
  "uptime_seconds": 服务运行时长
}
状态码：
503  SERVICE_UNAVAILABLE    服务未就绪

### 流式输出（SSE）

| 端点 | `POST /chat/stream` |
|---|---|
| 事件 | `meta`（来源）→ `token`（逐字）→ `done`（统计） |
| **首字延迟** | **1577ms** |
| **meta 延迟** | **736ms** |
| 总耗时 | ~7000ms |

对比非流式（6675ms 一次性返回）：
- 用户感知延迟：6675ms → 1577ms（**4.2 倍**）
- 来源展示：6675ms → 736ms（**9.1 倍**）
