"""
Router RAG - 套件入口
=================================
rag 套件把離線建索引（index.py）與查詢時讀索引、組裝 RouterQueryEngine（retrieval.py）
分成兩個模組：index.py 只在語料異動時執行一次，retrieval.py 供 main.py 每次啟動呼叫。
"""
