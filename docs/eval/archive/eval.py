# docs/eval/eval.py
import sys
import json
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent  # docs/eval -> docs -> ls-rag
sys.path.insert(0, str(ROOT))
from app.rag import answer
QUESTIONS_FILE = Path(__file__).parent / "questions.jsonl"
REPORT_FILE = Path(__file__).parent / "report.json"

REJECT_PHRASES = [
    "未找到相关信息",
    "没有相关信息",
    "无法回答",
    "资料中未找到",
    "资料中暂无",
    "暂未找到",
    "抱歉",
]

def is_rejected(ans: str) -> bool:
    return any(p in ans for p in REJECT_PHRASES)

def score(ans: str, keypoints: list[str]) -> tuple[int, list[bool]]:
    hits = [kp for kp in keypoints if kp in ans]
    return len(hits), [kp in ans for kp in keypoints]

def main():
    questions = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    results = []
    total_hit = 0
    total_expect = 0
    reject_count = 0
    t0 = time.time()

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question']}")
        result = {
            "question": q["question"],
            "expected_keypoints": q.get("expected_keypoints", []),
            "answer": "",
            "chunks": [],
            "hit_count": 0,
            "expect_count": len(q.get("expected_keypoints", [])),
        }

        try:
            resp = answer(q["question"])
            if isinstance(resp, dict):
                ans_text = resp.get("answer", "")
                chunks_text = resp.get("chunks_text", "")
                # 解析 chunks_text 成结构化列表
                chunks_list = []
                if chunks_text and chunks_text != "（未检索到相关资料）":
                    for line in chunks_text.split("\n"):
                        line = line.strip()
                        if line and line[0].isdigit():
                            # 格式: "1. [来源: xxx] 内容..."
                            parts = line.split("]", 1)
                            source = ""
                            text = ""
                            if len(parts) == 2:
                                source_part = parts[0]
                                if "[来源:" in source_part:
                                    source = source_part.split("[来源:", 1)[1].strip()
                                text = parts[1].strip()
                            chunks_list.append({"source": source, "text": text, "score": 0})
                result["chunks"] = chunks_list
            else:
                ans_text = str(resp)

            result["answer"] = ans_text

            if is_rejected(ans_text):
                reject_count += 1
                print(f"  -> 拒答")
            else:
                hit, details = score(ans_text, q.get("expected_keypoints", []))
                result["hit_count"] = hit
                result["keypoint_details"] = dict(zip(q.get("expected_keypoints", []), details))
                total_hit += hit
                total_expect += result["expect_count"]
                print(f"  -> 命中 {hit}/{result['expect_count']}")

        except Exception as e:
            result["answer"] = f"调用异常: {e}"
            print(f"  -> 异常: {e}")

        results.append(result)

    elapsed = time.time() - t0
    avg = elapsed / len(questions) if questions else 0

    report = {
        "summary": {
            "total": len(questions),
            "total_hit": total_hit,
            "total_expect": total_expect,
            "hit_rate": f"{total_hit/total_expect*100:.1f}%" if total_expect else "N/A",
            "reject_count": reject_count,
            "elapsed": f"{elapsed:.1f}s",
            "avg_per_q": f"{avg:.1f}s",
        },
        "results": results,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n===== 评估汇总 =====")
    print(f"命中 {total_hit} / 分母 {total_expect} = {report['summary']['hit_rate']}")
    print(f"拒答数: {reject_count}")
    print(f"总耗时: {elapsed:.1f}s  平均: {avg:.1f}s/题")
    print(f"\n详细结果 -> {REPORT_FILE}")

if __name__ == "__main__":
    main()