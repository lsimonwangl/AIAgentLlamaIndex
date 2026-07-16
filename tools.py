"""
Travel Agent - Tool 載入
=======================
tools.py 負責設定 Agent 可使用的 MCP tools，包含網路搜尋與天氣查詢。

相較於 Lab2 使用 langchain-mcp-adapters 的 MultiServerMCPClient，
Lab3 改用 llama-index-tools-mcp 的 BasicMCPClient + McpToolSpec，
透過 MCP 標準協定連接外部工具，回傳 LlamaIndex Agent 可直接使用的工具清單。

執行流程：
    0. 載入套件
    1. 使用 BasicMCPClient 以 stdio 方式連接 tavily MCP server
    2. 使用 BasicMCPClient 以 stdio 方式連接 open-meteo MCP server
    3. 透過 McpToolSpec 將 MCP 工具轉換為 LlamaIndex FunctionTool
    4. 回傳 MCP client 與 tools 給 main.py 使用

此模組提供 load_mcp_tools() 函式供 main.py 呼叫。
"""

# 載入套件
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


async def load_mcp_tools():
    """連接 MCP 工具服務，並回傳 LlamaIndex Agent 可以直接使用的工具清單。

    MCP 是一個讓 LLM 與外部工具溝通的標準協定。
    這裡用兩個元件把外部 MCP server 接成 Agent 可用的工具：
        - BasicMCPClient: 啟動並連接單一 MCP server，負責底層溝通
            - 第一個參數 command 為啟動服務的執行檔（這裡用 npx 直接執行 npm 套件）
            - args 為傳給 command 的參數；-y 表示自動同意安裝、@latest 抓最新版
            - 預設以 stdio（標準輸入輸出）作為 Agent 與 server 的溝通方式
        - McpToolSpec: 把該 server 暴露的 MCP 工具轉換成 LlamaIndex FunctionTool
            - to_tool_list_async() 向 server 詢問可用工具，產出 Agent 能直接呼叫的清單
    """

    # 建立 Tavily MCP client：提供網路搜尋功能，讓 Agent 能查詢即時資訊
    # TAVILY_API_KEY 由 load_dotenv() 載入 os.environ，npx 子程序會繼承，故此處不需另外傳 env
    tavily_client = BasicMCPClient("npx", args=["-y", "tavily-mcp@latest"])
    tavily_spec = McpToolSpec(client=tavily_client)              # 包成 ToolSpec
    tavily_tools = await tavily_spec.to_tool_list_async()        # 取得工具清單

    # 建立 Open-Meteo MCP client：提供免費天氣查詢服務，不需 API 金鑰
    meteo_client = BasicMCPClient("npx", args=["-y", "open-meteo-mcp-server"])
    meteo_spec = McpToolSpec(client=meteo_client)               # 包成 ToolSpec
    meteo_tools = await meteo_spec.to_tool_list_async()         # 取得工具清單

    # 兩個 server 的工具合併成單一清單回傳；client 一併回傳供 main.py 後續關閉連線
    print(f"🔧 已載入 MCP 工具：Tavily {len(tavily_tools)} 個、Open-Meteo {len(meteo_tools)} 個")
    return [tavily_client, meteo_client], tavily_tools + meteo_tools
