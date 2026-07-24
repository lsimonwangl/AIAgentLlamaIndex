"""
驗證 RouterQueryEngine 四條路由的選路結果。

執行方式：
    venv/Scripts/python.exe verify_routing.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from rag.router import build_router_query_engine

# 與 rag/router.py 中 query_engine_tools 的順序一致
TOOL_NAMES = ["SummaryIndex", "VectorStoreIndex", "DocumentSummaryIndex", "KeywordTableIndex"]

CASES = [
    ("我的旅遊紀錄整體風格是什麼", "SummaryIndex"),
    ("我想去日本大阪走那種沿溪谷、水很清的健行步道,幫我照過去走過的類似路線推薦", "VectorStoreIndex"),
    ("回顧我過去和這次最像的那幾趟完整旅行紀錄", "DocumentSummaryIndex"),
    ("我想在日本找老屋民宿，參考我住過的「木門厝」", "KeywordTableIndex"),
]


def main():
    router = build_router_query_engine()

    # ponytail: 直接呼叫 selector 驗證選路，避免跑 SummaryIndex 全量 tree_summarize
    print("\n=== 選路驗證 ===")
    failures = 0
    for query, expected in CASES:
        result = router._selector.select(router._metadatas, query)
        chosen = TOOL_NAMES[result.ind]
        status = "PASS" if chosen == expected else "FAIL"
        if chosen != expected:
            failures += 1
        print(f"[{status}] {query}")
        print(f"       預期 {expected}，實際 {chosen}")
        print(f"       理由：{result.selections[0].reason}")

    assert failures == 0, f"{failures} 個選路案例失敗"
    print("\n全部選路案例通過")


if __name__ == "__main__":
    main()
