"""
Travel Agent - Tool 載入
=======================
tools.py 負責設定 Agent 可使用的 MCP tools，包含網路搜尋與天氣查詢。

相較於 Lab2 使用 langchain-mcp-adapters 的 MultiServerMCPClient，
Lab3 改用 llama-index-tools-mcp 的 McpToolSpec，
透過 MCP 標準協定連接外部工具，回傳 LlamaIndex Agent 可直接使用的工具清單。

執行流程：
    0. 載入套件
    1. 建立 tavily MCP 連線參數，並讀取 TAVILY_API_KEY
    2. 建立 open-meteo MCP 連線參數，供 Agent 查詢天氣資訊
    3. 使用 McpToolSpec 連接 MCP server 並取得工具清單
    4. 回傳所有 tools 給 main.py 使用

此模組提供 load_mcp_tools() 函式供 main.py 呼叫。
"""

import os

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


async def load_mcp_tools():
    """連接 MCP server 並回傳 LlamaIndex Agent 可以直接使用的工具清單。"""

    # 建立 Tavily MCP client（網路搜尋）
    tavily_client = BasicMCPClient(
        "npx",
        args=["-y", "tavily-mcp@latest"],
        env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
    )

    # 建立 Open-Meteo MCP client（天氣查詢）
    meteo_client = BasicMCPClient(
        "npx",
        args=["-y", "open-meteo-mcp-server"],
    )

    # 透過 McpToolSpec 連接 MCP server 並取得工具清單
    tavily_spec = McpToolSpec(client=tavily_client)
    meteo_spec = McpToolSpec(client=meteo_client)

    tavily_tools = await tavily_spec.to_tool_list_async()
    meteo_tools = await meteo_spec.to_tool_list_async()

    print(f"已載入 MCP 工具：Tavily {len(tavily_tools)} 個、Open-Meteo {len(meteo_tools)} 個")
    return tavily_tools + meteo_tools
