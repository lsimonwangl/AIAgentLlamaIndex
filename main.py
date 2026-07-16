"""
Travel Agent - 主程式入口
=======================
main.py 負責把偏好檢索、Agent、外部工具與終端機介面串成完整旅遊規劃流程。

相較於 Lab2 使用 LCEL 將各元件串成 chain，
Lab3 直接將 RouterQueryEngine 與 Agent 傳入 CLI 介面，
由 chat.py 依序呼叫 Router 檢索與 Agent 回答，
並由 prompt.py 處理使用者輸入。

執行流程：
    0. 載入套件與環境變數
    1. 載入 Agent 可以使用的 MCP 外部工具
    2. 建立 RouterQueryEngine（SummaryIndex + VectorStoreIndex）
    3. 建立負責回答問題的旅遊 Agent
    4. 啟動終端機互動介面，接收使用者多輪問題

執行方式：
    python main.py
"""

# 載入套件與環境變數
from dotenv import load_dotenv
load_dotenv()
import asyncio
from agent import build_agent
from chat import run_chat
from rag import build_router_query_engine
from tools import load_mcp_tools


async def main():
    # 載入外部工具，例如網路搜尋與天氣查詢
    mcp_clients, tools = await load_mcp_tools()

    # 保留 MCP client 連線，讓終端機對話期間可以持續呼叫工具
    _ = mcp_clients

    # 建立 RouterQueryEngine，自動在 SummaryIndex 與 VectorStoreIndex 之間選路
    router_engine = build_router_query_engine()

    # 建立旅遊 Agent，負責整合工具、偏好資料並產生回答
    agent = build_agent(tools)

    # 啟動終端機介面，讓使用者可以一輪一輪輸入問題
    await run_chat(router_engine, agent)


if __name__ == "__main__":
    asyncio.run(main())
