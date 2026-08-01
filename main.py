"""
Travel Agent - 主程式入口與終端機互動
=====================================
main.py 負責把偏好檢索、Agent、外部工具串起來，並驅動終端機多輪對話迴圈。

相較於 Lab2 使用 LCEL 將各元件串成 chain，
Lab3 直接將 RouterQueryEngine 與 Agent 傳入 CLI 迴圈，
每輪由 Router 檢索偏好、交給 Agent 回答。

執行流程：
    0. 載入套件與環境變數
    1. 載入 Agent 可以使用的 MCP 外部工具
    2. 建立 RouterQueryEngine（讀回 index.py 離線建好的 Summary / Vector /
       DocumentSummary / KeywordTable 四索引，不在這裡重建）
    3. 建立負責回答問題的旅遊 Agent
    4. 顯示啟動 banner，透過 read_query 從終端機讀取使用者輸入
    5. 每輪呼叫 retrieve_preferences 用 RouterQueryEngine 檢索偏好（顯示選路結果），
       再由 run_agent_turn 交給 FunctionAgent 回答（透過 stream_events 顯示工具呼叫）
    6. 使用者結束輸入時印出告別訊息

執行方式（./data 語料有異動、或第一次執行時，需先建好索引）：
    python -m rag.index
    python main.py
"""

# 載入套件與環境變數
from dotenv import load_dotenv
load_dotenv()
import asyncio
from datetime import datetime

from llama_index.core.agent.workflow import ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from openai import APIError

from agent import build_agent
from rag.retrieval import build_router_query_engine, retrieve_preferences
from tools import load_mcp_tools


# ── 讀取使用者輸入 ───────────────────────────────────
def read_query(turn: int) -> str | None:
    """從終端機讀取一輪使用者輸入，並回傳給對話迴圈判斷下一步。"""
    try:
        # 顯示目前輪次並讀取使用者輸入，去掉前後空白
        query = input(f"[第 {turn} 輪] 你：").strip()
    except (EOFError, KeyboardInterrupt):
        # 使用者按 Ctrl+D 或 Ctrl+C 時，視為主動結束對話
        return None

    if query.lower() in {"exit", "quit"}:
        # 使用者輸入結束關鍵字時，通知對話迴圈結束
        return None

    # 回傳原始問題；若使用者只按 Enter，會回傳空字串讓呼叫端繼續等待
    return query


# ── Agent 執行一輪對話 ───────────────────────────────
async def run_agent_turn(agent, ctx, query: str, rag_text: str):
    """讓 Agent 用已檢索到的過往偏好回答一輪問題，邊呼叫工具邊產生回答。

    把過往經驗、今天日期與問題組成一段輸入交給 Agent 執行；中間把工具呼叫印
    出來，是為了讓終端機看得到 Agent 實際查了什麼，而不是只看到最後一段文字。
    """
    print("\n⏳ Agent 思考中...\n")

    # 組合 Agent 輸入
    today = datetime.now().strftime("%Y-%m-%d")
    agent_input = (
        f"今天日期：{today}\n\n"
        f"我過往的台灣旅遊紀錄顯示我的偏好：\n{rag_text}\n\n"
        f"使用者問題：{query}"
    )

    # agent.run() 立刻回傳 handler 而非等答案；ctx 帶入前幾輪對話記憶
    handler = agent.run(agent_input, ctx=ctx)

    # 邊跑邊收事件：ToolCall 是「決定要查什麼」，ToolCallResult 是「查回什麼」
    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            print(f"🔧 呼叫工具: {event.tool_name}({event.tool_kwargs})")
        elif isinstance(event, ToolCallResult):
            print(f"✅ 工具回傳: {str(event.tool_output)[:150]}...")

    response = await handler

    # 顯示最終回答
    print(f"\n{response}")


async def main():
    """依序備好工具、檢索器與 Agent，再啟動終端機多輪對話迴圈。"""

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
        # 每輪先做 RAG 檢索、再交給 Agent 執行；ctx 在輪次間保留對話記憶
        # 整輪包在 try/except 裡：NVIDIA 端點限流（429/503）或連線錯誤時
        # 只放棄這一輪，回到對話迴圈，不讓整個 session 掛掉
        try:
            # 建立 RouterQueryEngine：讀回 index.py 離線建好的四種索引，只做本機讀檔與
            router_engine = build_router_query_engine()
            rag_result = retrieve_preferences(router_engine, query)
            print()
            # 載入AI Agent外部工具，例如網路搜尋與天氣查詢
            tools = await load_mcp_tools()
            # 建立旅遊 Agent，負責整合工具、偏好資料並產生回答
            agent = build_agent(tools)
            # 建立 Context 物件，讓 Agent 在多輪對話間保留記憶
            # 整場對話共用同一個 ctx，使用者才能只補「三天兩夜」而不用重講目的地
            ctx = Context(agent)
            await run_agent_turn(agent, ctx, query, rag_result)
        except APIError as error:
            print(f"\n⚠️ NVIDIA API 呼叫失敗，這輪回答未完成：{error}")
            print("   通常是端點限流（429/503），稍等一兩分鐘再重問一次即可。")
        print()
        turn += 1


if __name__ == "__main__":
    asyncio.run(main())
