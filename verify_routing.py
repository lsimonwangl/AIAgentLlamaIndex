"""
驗證 RouterQueryEngine 三條路由的選路結果與圖譜偏好聚合輸出。

執行方式：
    venv/Scripts/python.exe verify_routing.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from rag import build_router_query_engine

# 與 rag.py 中 query_engine_tools 的順序一致
TOOL_NAMES = ["SummaryIndex", "VectorStoreIndex", "PropertyGraphIndex"]

CASES = [
    ("這次我想自己一個人去大阪三天兩夜，照我過去獨旅的偏好規劃", "PropertyGraphIndex"),
    ("我的旅遊紀錄整體風格是什麼", "SummaryIndex"),
    ("幫我找像台南木門厝那種老屋民宿", "VectorStoreIndex"),
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

    print("\n=== PropertyGraphIndex 輸出驗證（獨旅偏好聚合，應涵蓋台南＋新竹） ===")
    response = router._query_engines[TOOL_NAMES.index("PropertyGraphIndex")].query(
        CASES[0][0]
    )
    print(str(response))

    assert failures == 0, f"{failures} 個選路案例失敗"
    print("\n全部選路案例通過")


if __name__ == "__main__":
    main()
