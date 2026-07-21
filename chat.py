"""
Travel Agent - 終端機輸出
=======================
chat.py 負責顯示啟動畫面、Router 檢索結果、工具呼叫與 Agent 最終回答，
並驅動多輪對話迴圈。

執行流程：
    0. 顯示啟動 banner 與範例問題
    1. 透過 prompt.read_query 接收使用者輸入
    2. 呼叫 RouterQueryEngine 檢索偏好（顯示選路結果）
    3. 將偏好與使用者問題交給 FunctionAgent，透過 stream_events 顯示工具呼叫
    4. 使用者結束輸入時印出告別訊息

此模組提供 run_chat() 函式供 main.py 呼叫。
"""

from datetime import datetime
from llama_index.core.agent.workflow import ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from openai import APIError
from prompt import read_query


async def run_query(router_engine, agent, ctx, query: str):
    """執行一次使用者問題：先用 Router 檢索偏好，再交給 Agent 回答。

    整輪包在 try/except 裡：NVIDIA 端點限流（429/503）或連線錯誤時
    只放棄這一輪，回到對話迴圈，不讓整個 session 掛掉。
    """
    print("\n⏳ Agent 思考中...\n")
    try:
        # ── Stage 1: RouterQueryEngine 自動選路檢索 ──
        print("🔍 RouterQueryEngine 檢索中...")
        rag_response = router_engine.query(query)

        # 顯示 Router 選了哪條路
        if hasattr(rag_response, "metadata") and rag_response.metadata:
            selector_result = rag_response.metadata.get("selector_result")
            if selector_result:
                # selections 是 0-based，但 LLM 的 reason 文字用 1-based（choice (1)）描述，
                # 這裡統一轉成 1-based 並附上工具名稱，避免「選 0 卻說選 1」的混淆
                tool_names = {
                    1: "SummaryIndex（整體偏好）",
                    2: "VectorStoreIndex（特定細節）",
                    3: "DocumentSummaryIndex（整趟紀錄回顧）",
                }
                for sel in selector_result.selections:
                    choice = sel.index + 1
                    print(f"📋 Router 選路結果：choice ({choice}) {tool_names.get(choice, '')}")
                    print(f"   理由：{sel.reason}")

        # 顯示檢索到的偏好摘要
        rag_text = str(rag_response)
        print(f"📋 偏好摘要：{rag_text[:200]}...")
        print()

        # ── Stage 2: 組合 Agent 輸入 ──
        today = datetime.now().strftime("%Y-%m-%d")
        agent_input = (
            f"今天日期：{today}\n\n"
            f"我過往的台灣旅遊紀錄顯示我的偏好：\n{rag_text}\n\n"
            f"使用者問題：{query}"
        )

        # ── Stage 3: Agent 回答，透過 stream_events 顯示工具呼叫 ──
        handler = agent.run(agent_input, ctx=ctx)

        async for event in handler.stream_events():
            if isinstance(event, ToolCall):
                print(f"🔧 呼叫工具: {event.tool_name}({event.tool_kwargs})")
            elif isinstance(event, ToolCallResult):
                print(f"✅ 工具回傳: {str(event.tool_output)[:150]}...")

        response = await handler

        # 顯示最終回答
        print(f"\n{response}")
    except APIError as error:
        print(f"\n⚠️ NVIDIA API 呼叫失敗，這輪回答未完成：{error}")
        print("   通常是端點限流（429/503），稍等一兩分鐘再重問一次即可。")


async def run_chat(router_engine, agent):
    """啟動多輪對話介面，直到使用者主動結束。"""

    # 建立 Context 物件，讓 Agent 在多輪對話間保留記憶
    ctx = Context(agent)

    print("""
==================================================
🧳 旅遊規劃助理已就緒（支援國內/國外規劃）
💡 輸入問題開始對話，輸入 'exit' 或 'quit' 結束
💡 範例：
   1. 統計我過去旅行平均一天排幾個景點、步調多快，照這個節奏安排日本大阪三天兩夜的行程
   2. 大阪住宿幫我找像台南那間民宿風格的住宿
==================================================
""")

    # 持續接收使用者輸入，直到 read_query 回傳 None
    turn = 1
    while True:
        query = read_query(turn)

        if query is None:
            print("\n👋 再見")
            break
        if not query:
            continue

        print()
        # 執行本輪查詢，ctx 在輪次間保留對話記憶
        await run_query(router_engine, agent, ctx, query)
        print()
        turn += 1
