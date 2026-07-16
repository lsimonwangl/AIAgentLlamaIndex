"""
Travel Agent - 使用者輸入
=======================
prompt.py 負責從終端機讀取使用者每一輪輸入，並判斷是否要結束對話。

執行流程：
    0. 從終端機讀取使用者本輪輸入
    1. 處理 Ctrl+C、Ctrl+D 與 exit/quit 結束輸入
    2. 回傳使用者問題、空字串或 None 給 chat.py 的對話迴圈使用

此模組提供 read_query() 函式供 chat.py 呼叫。
"""


def read_query(turn: int) -> str | None:
    """從終端機讀取一輪使用者輸入，並回傳給對話迴圈判斷下一步。"""
    try:
        # 顯示目前輪次並讀取使用者輸入，去掉前後空白
        query = input(f"[第 {turn} 輪] 你：").strip()
    except (EOFError, KeyboardInterrupt):
        # 使用者按 Ctrl+D 或 Ctrl+C 時，視為主動結束對話
        return None

    if query.lower() in {"exit", "quit"}:
        # 使用者輸入結束關鍵字時，通知 chat.py 結束對話迴圈
        return None

    # 回傳原始問題；若使用者只按 Enter，會回傳空字串讓呼叫端繼續等待
    return query
