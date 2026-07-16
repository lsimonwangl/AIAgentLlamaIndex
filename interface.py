"""
Travel Agent - 終端機介面
=======================
interface.py 負責接收使用者輸入，顯示 RouterQueryEngine 的選路結果，
並輸出 Agent 的工具呼叫與最終回答。

相較於 Lab2 使用 astream_events 統一處理所有事件，
Lab3 分為兩階段執行：先呼叫 RouterQueryEngine 取得偏好，
再將偏好與問題交給 Agent 產生回答。

執行流程：
    0. 接收使用者每一輪輸入
    1. 呼叫 RouterQueryEngine 檢索偏好（顯示選路結果）
    2. 將偏好與使用者問題組合後交給 Agent
    3. 顯示 Agent 的工具呼叫記錄與最終回答
    4. 使用者輸入 exit、quit、bye 或掰掰時結束對話

此模組提供 run_cli() 函式供 main.py 呼叫。
"""

from datetime import datetime

from openai import APITimeoutError


DETAIL_QUERY_KEYWORDS = (
    "之前",
    "住過",
    "泡過",
    "去過",
    "吃過",
    "體驗過",
    "哪個",
    "哪些",
)


def build_preference_query(query: str) -> str:
    """把使用者問題轉成只檢索過往偏好的 RAG 查詢。"""
    if any(keyword in query for keyword in DETAIL_QUERY_KEYWORDS):
        return (
            "根據過往旅遊紀錄，找出與這次問題最相關的具體經驗與原文線索。"
            "只回答過往紀錄中的偏好或經驗，不要安排本次行程。"
            f"使用者問題：{query}"
        )

    return (
        "根據過往旅遊紀錄，整理使用者與這次問題相關的旅遊偏好。"
        "只整理偏好，不要安排本次行程。"
        f"使用者問題：{query}"
    )


async def run_query(router_engine, agent, query: str):
    """執行一次使用者問題：先用 Router 檢索偏好，再交給 Agent 回答。"""
    print("\nAgent 思考中...\n")

    # ── Stage 1: RouterQueryEngine 自動選路檢索 ──
    print("RouterQueryEngine 檢索中...")
    preference_query = build_preference_query(query)
    rag_response = router_engine.query(preference_query)

    # 顯示 Router 選了哪條路（SummaryIndex 或 VectorStoreIndex）
    if hasattr(rag_response, "metadata") and rag_response.metadata:
        selector_result = rag_response.metadata.get("selector_result")
        if selector_result:
            print(f"Router 選路結果：{selector_result}")

    # 顯示檢索到的偏好摘要
    rag_text = str(rag_response)
    print(f"偏好摘要：{rag_text[:200]}...")
    print()

    # ── Stage 2: 組合 Agent 輸入 ──
    today = datetime.now().strftime("%Y-%m-%d")
    agent_input = (
        f"今天日期：{today}\n\n"
        f"我過往的台灣旅遊紀錄顯示我的偏好：\n{rag_text}\n\n"
        f"使用者問題：{query}"
    )

    # ── Stage 3: Agent 回答 ──
    try:
        response = await agent.chat(agent_input)
    except APITimeoutError:
        print("LLM 請求逾時，這輪回答未完成。請稍後重試，或把問題縮小一點再問。")
        return

    # 顯示工具呼叫記錄
    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        print("\n--- 工具呼叫記錄 ---")
        for tool_call in tool_calls:
            tool_name = getattr(tool_call, "tool_name", "")
            tool_input = getattr(tool_call, "tool_kwargs", {})
            tool_output = getattr(tool_call, "tool_output", None)
            raw_output = getattr(tool_output, "raw_output", tool_output)
            print(f"{tool_name}: {str(tool_input)[:100]}...")
            print(f"回傳: {str(raw_output)[:150]}...")
        print()

    # 顯示最終回答
    print(str(response))


async def run_cli(router_engine, agent):
    """啟動多輪對話，直到使用者輸入 exit、quit、bye 或掰掰才結束。"""

    # 顯示啟動提示與範例問題
    print("""
==================================================
旅遊規劃助理已就緒（支援國內/國外規劃）
輸入問題開始對話，輸入 'exit' 或 'quit' 結束
範例：
   1. 幫我安排下周二三天兩夜的大阪的古蹟參訪行程
   2. 京都的氛圍跟我去過的哪個台灣地方比較像？
==================================================
""")

    # 持續接收使用者輸入，直到使用者主動結束
    turn = 1
    while True:
        try:
            query = input(f"[第 {turn} 輪] 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "bye", "掰掰"}:
            print("再見")
            break

        print()
        # 執行本輪查詢
        await run_query(router_engine, agent, query)
        print()
        turn += 1
